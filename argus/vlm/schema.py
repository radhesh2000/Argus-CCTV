"""The contract with the model: one JSON object, always the same shape.

Ported from v1 (``detectors/vlm/engine.py`` on ``v1-archive``). v1 asked Ollama for a schema and
hoped; here the schema is enforced by the server's constrained decoder
(vLLM + xgrammar), so a malformed answer is not a failure mode that reaches this
module — but the parsing below still treats every field as untrusted, because a
model that must emit valid JSON can still emit *wrong* JSON, and a confidence of
"very high" or a box outside the frame must degrade rather than raise.

Two conventions are carried over from v1 deliberately:

* **``box_2d`` on a 0-1000 scale, top-left origin.** This is Qwen's grounding
  convention, and the served model is a Qwen3-VL derivative, so asking for the
  coordinate system it was trained to emit costs nothing and works better than
  imposing our own. ``normalize_box`` accepts 0-1 fractions and raw pixels too,
  because a model under schema constraint will occasionally answer in whichever
  it feels like and the snapshot should still get a box.
* **``triggered`` is separate from ``confidence``.** A model that says "yes, 0.3"
  is telling us something a single score cannot, and the policy engine's
  ``min_confidence`` floor is applied to the pair rather than baked into the
  prompt.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

# The JSON Schema handed to the server's constrained decoder. `strict` mode on
# the OpenAI wire format additionally requires `additionalProperties: false` and
# every property listed in `required`, so `box_2d` is nullable rather than
# optional — "no box" has to be representable *inside* the schema.
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "triggered": {
            "type": "boolean",
            "description": "true only if the described violation is clearly visible",
        },
        "confidence": {
            "type": "number",
            "description": "0.0-1.0 certainty in the answer above",
        },
        "reason": {
            "type": "string",
            "description": "one short sentence a safety officer can act on",
        },
        "box_2d": {
            "description": (
                "[x1, y1, x2, y2] on a 0-1000 scale, top-left origin, tightly "
                "around the violation. null when not triggered."
            ),
            "type": ["array", "null"],
            "items": {"type": "number"},
            "minItems": 4,
            "maxItems": 4,
        },
    },
    "required": ["triggered", "confidence", "reason", "box_2d"],
    "additionalProperties": False,
}

# Appended to every policy prompt. The schema constrains the *shape*; this
# constrains the *behaviour*, and the two are not the same job.
PROMPT_SUFFIX = (
    "\n\nAnswer with a single JSON object: "
    '{"triggered": bool, "confidence": 0-1 number, "reason": short string, '
    '"box_2d": [x1, y1, x2, y2] or null}. '
    "Judge only what is visible in this frame — do not speculate about what may "
    "be outside it or about to happen. Prefer triggered=false when the evidence "
    "is ambiguous. If triggered is true, box_2d MUST tightly locate the "
    "violation using coordinates on a 0-1000 scale with the origin at the "
    "top-left; otherwise box_2d must be null."
)


def verification_prompt(claim: str) -> str:
    """The second-opinion question the verification pass asks about a detector claim.

    A module-level function rather than a string built inside the scheduler
    because ``argus/eval/`` measures this exact wording: an eval harness that
    scores a *paraphrase* of the production prompt measures a model nobody
    deployed. One definition, two callers, and a change to it moves both.

    **The count clause is load-bearing and it is here because it was measured.**
    The wording before 2026-08-15 asked only *"is the detector's claim correct?"*,
    and Qwen3-VL-8B answered the question it was asked: it checked whether the
    scene looked over limit, upheld claims of 16, 20, 25, 30 and 40 people on a
    frame holding about sixteen, and gave the same sentence as its reason every
    time. The number was never in scope. Argus's one shipped detector makes a
    claim whose entire content *is* a number, so a verifier that does not audit
    the number vetoes nothing worth vetoing.

    Adding the clause below took over-claims at twice the plausible headcount
    from 15/24 to 24/24 vetoed on the same model and the same frames, at no cost
    to the truthful control (6/6 both ways) — the whole `moderate` column in
    docs/VLM_EVAL.md. It is scored verbatim by the counting slice, whose
    ``headcount_strict`` policy holds a frozen copy of this text: if the two ever
    disagree, that slice's two columns separate and say so.
    """
    return (
        "An automated safety detector has flagged this camera frame.\n\n"
        f"Detector claim: {claim}\n\n"
        "You are the second opinion before this alert is emailed to a safety "
        "officer. First count the people you can actually see in the image. "
        "The number in the claim is part of the claim: answer triggered=true "
        "only if the claim is supported by what you can see AND its count "
        "matches your own count. Answer triggered=false if the count is wrong, "
        "even when the situation it describes is broadly true. State your own "
        "count in your reason. box_2d must be null."
    )


@dataclass(slots=True)
class PolicyVerdict:
    """One model answer, already sanitised."""

    triggered: bool
    confidence: float
    reason: str
    box_2d: tuple[float, float, float, float] | None = None
    # Wall-clock seconds the inference took, for the health endpoint and the
    # latency numbers in docs/BENCHMARKS.md.
    latency_sec: float = 0.0
    # Server-reported token usage, when it offers it.
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "triggered": self.triggered,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "box_2d": list(self.box_2d) if self.box_2d else None,
            "latency_sec": round(self.latency_sec, 3),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }

    def fires(self, min_confidence: float) -> bool:
        """Whether this verdict should raise an event."""
        return self.triggered and self.confidence >= min_confidence


class VlmResponseError(ValueError):
    """The model's answer could not be read as a verdict."""


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    # A model that answers `nan` or `inf` for confidence would poison every
    # comparison downstream, silently and in the direction of firing.
    return number if number == number and abs(number) != float("inf") else default


def normalize_box(
    box: Any, *, width: int, height: int
) -> tuple[float, float, float, float] | None:
    """Turn whatever the model called a box into pixels on this frame.

    Three coordinate systems are accepted, distinguished by magnitude:
    0-1 fractions, the requested 0-1000 grid, and raw pixels. Guessing is
    unavoidable and cheap to get right — the alternative is discarding a
    perfectly good grounding box because the model used the units it preferred.

    Returns pixel ``(x1, y1, x2, y2)`` clamped to the frame, or None when the
    box is absent, malformed or degenerate.
    """
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    values = [_as_float(v, default=float("nan")) for v in box]
    if any(v != v for v in values):  # NaN from a non-numeric entry
        return None

    largest = max(abs(v) for v in values)
    if largest <= 1.0:
        scale_x, scale_y = width, height                 # 0-1 fractions
    elif largest <= 1000.0 and max(width, height) > 1000:
        scale_x, scale_y = width / 1000.0, height / 1000.0
    elif largest <= 1000.0:
        # Ambiguous below 1000 on a smaller frame: the 0-1000 grid is what we
        # asked for in the prompt, so believe the prompt over the coincidence.
        scale_x, scale_y = width / 1000.0, height / 1000.0
    else:
        scale_x = scale_y = 1.0                          # already pixels

    x1, y1, x2, y2 = (
        values[0] * scale_x, values[1] * scale_y, values[2] * scale_x, values[3] * scale_y,
    )
    # Models sometimes emit the corners the other way round.
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1, x2 = max(0.0, x1), min(float(width), x2)
    y1, y2 = max(0.0, y1), min(float(height), y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return (x1, y1, x2, y2)


def parse_verdict(content: str, *, width: int, height: int) -> PolicyVerdict:
    """Read a model reply into a :class:`PolicyVerdict`.

    ``width``/``height`` are the dimensions of the image *as sent*, which is not
    the decoder's frame size — the client downscales — so the box comes back in
    the sent image's pixel space and the caller rescales once, in one place.
    """
    text = (content or "").strip()
    if not text:
        raise VlmResponseError("empty response")
    # A constrained decoder returns bare JSON, but a server without constrained
    # decoding (or a fallback path) may wrap it in a ```json fence.
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text[:4].lower() == "json":
            text = text[4:].lstrip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise VlmResponseError(f"not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise VlmResponseError(f"expected an object, got {type(payload).__name__}")

    triggered = bool(payload.get("triggered"))
    confidence = _as_float(payload.get("confidence"), default=0.0)
    # Clamp rather than reject: a model answering 1.5 means "very sure", and
    # refusing the whole verdict over it would lose a real alert.
    confidence = min(1.0, max(0.0, confidence))
    reason = str(payload.get("reason") or "").strip()[:500]
    return PolicyVerdict(
        triggered=triggered,
        confidence=confidence,
        reason=reason,
        box_2d=normalize_box(payload.get("box_2d"), width=width, height=height) if triggered else None,
    )


__all__ = [
    "PROMPT_SUFFIX",
    "RESPONSE_SCHEMA",
    "PolicyVerdict",
    "VlmResponseError",
    "normalize_box",
    "parse_verdict",
    "verification_prompt",
]
