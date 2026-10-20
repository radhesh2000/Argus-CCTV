"""The unit of evaluation: one frame, one prompt, one expected answer.

Two decisions in here are worth more than the code around them.

**A case carries its own render mode.** ``FrameRef.crop_banner`` decides whether
the headline strip Argus burns onto an evidence snapshot is cropped off before
the image is sent. The triage slice keeps it, because keeping it is what the
appliance actually does — ``VlmRunner.dispatch`` hands the verifier the
*annotated* frame. The counting slice crops it, because a banner reading
"over sanctioned limit: 5 people (limit 3)" hands the model the answer in
pixels, and a counting benchmark the model can pass by reading our own caption
measures nothing. Phase 3 learned this the expensive way with SigLIP
(docs/PHASE3_NOTES.md); it is not a hypothetical.

**``expected=None`` is a first-class value, not a missing one.** An unlabelled
frame still gets sent and still gets timed — see the package docstring. Every
scoring function in ``metrics.py`` skips them explicitly rather than treating
None as False, because a harness that quietly scored unlabelled data would
report a precision number that moved when somebody added footage.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Slices, in the order a report should read them: what the deployment gave us,
# what we labelled, and the one failure mode this whole package exists for.
SLICES = ("triage", "policy", "counting")


@dataclass(slots=True)
class FrameRef:
    """A picture on disk, plus how to prepare it and where it came from."""

    path: Path
    # Free-text provenance: "event 4711 snapshot" or "packing_section_recording.mp4 @ 40.0s".
    origin: str = ""
    # Strip the burned-in headline before sending. See the module docstring.
    crop_banner: bool = False
    # Normalised ``(x1, y1, x2, y2)`` sub-region, or None for the whole frame.
    #
    # This is not a convenience. A headcount claim in this product is a claim
    # about a **zone** — `Zone 'Packing floor' over sanctioned limit: 5 people
    # (limit 3)` — never about a whole camera view, so a counting benchmark
    # asked over the whole frame is asking a question the appliance never asks.
    # It is also what makes the ground truth defensible: the demo footage holds
    # 14-20 people per frame at the edge of resolution, and nobody can label
    # that exactly, whereas a zone-sized region holding two is unarguable. A
    # count nobody would defend is not a label.
    region: tuple[float, float, float, float] | None = None

    def load(self) -> np.ndarray:
        """The frame as the BGR array ``VlmClient.evaluate`` accepts.

        BGR rather than RGB to match the decoder's output, so a frame read off
        disk here and a frame taken live in ``argus run`` reach the model in the
        same colour order. ``encode_frame`` re-orders once, in one place.

        Deliberately not ``argus.index.backfill.load_bgr``, which does the same
        job: importing it would drag ``argus.index.embedder`` — and with it
        torch and transformers — into an eval process whose only job is to POST
        JPEGs at an HTTP endpoint.

        ``crop_banner`` is applied first and ``region`` second, so a region is
        expressed against the picture a reviewer sees after the headline is
        gone. In the generated set the two never co-occur: banners exist only on
        evidence snapshots and regions only on frames cut from video.
        """
        from PIL import Image

        with Image.open(self.path) as image:
            rgb = image.convert("RGB")
            if self.crop_banner:
                from argus.media.annotate import banner_height

                top = banner_height(rgb.width)
                # Never crop into the scene: a pathologically small image would
                # otherwise come back a sliver.
                if top < rgb.height // 4:
                    rgb = rgb.crop((0, top, rgb.width, rgb.height))
            if self.region is not None:
                rgb = rgb.crop(_pixel_box(self.region, rgb.width, rgb.height))
            return np.ascontiguousarray(np.asarray(rgb)[:, :, ::-1])

    def digest(self) -> str:
        """Short content hash — what makes a label auditable across a re-extract."""
        return hashlib.sha256(self.path.read_bytes()).hexdigest()[:16]


def _pixel_box(region: tuple[float, float, float, float], width: int, height: int
               ) -> tuple[int, int, int, int]:
    """Normalised region -> pixel box, clamped and never degenerate."""
    x1, y1, x2, y2 = region
    x1, x2 = sorted((float(x1), float(x2)))
    y1, y2 = sorted((float(y1), float(y2)))
    left = max(0, min(width - 1, round(x1 * width)))
    top = max(0, min(height - 1, round(y1 * height)))
    right = max(left + 1, min(width, round(x2 * width)))
    bottom = max(top + 1, min(height, round(y2 * height)))
    return (left, top, right, bottom)


@dataclass(slots=True)
class EvalCase:
    """One question, asked of every model under test."""

    id: str
    slice: str
    # The policy this case belongs to. Scores are reported per policy, so this
    # is what separates "the model is bad at counting" from "the model is bad".
    policy: str
    prompt: str
    frame: FrameRef
    # True = the model should answer triggered; False = it should not;
    # None = unlabelled, so timed but never scored.
    expected: bool | None = None
    # Where the label came from, verbatim in the report. A number a reviewer
    # cannot trace to a human decision is a number they should not trust.
    label_source: str = ""
    note: str = ""

    @property
    def scored(self) -> bool:
        return self.expected is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "slice": self.slice,
            "policy": self.policy,
            "prompt": self.prompt,
            "frame": str(self.frame.path),
            "frame_origin": self.frame.origin,
            "crop_banner": self.frame.crop_banner,
            "region": list(self.frame.region) if self.frame.region else None,
            "expected": self.expected,
            "label_source": self.label_source,
            "note": self.note,
        }


@dataclass(slots=True)
class EvalSet:
    """The assembled cases, plus how they were assembled."""

    cases: list[EvalCase] = field(default_factory=list)
    # Human-readable notes about what was included and what was skipped, printed
    # by the CLI and reproduced in the report. "0 counting cases" must never be
    # something a reader has to infer from an empty table.
    notes: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.cases)

    @property
    def scored(self) -> list[EvalCase]:
        return [case for case in self.cases if case.scored]

    def by_slice(self, name: str) -> list[EvalCase]:
        return [case for case in self.cases if case.slice == name]

    def policies(self) -> list[str]:
        seen: dict[str, None] = {}
        for case in self.cases:
            seen.setdefault(case.policy, None)
        return list(seen)

    def summary(self) -> dict[str, dict[str, int]]:
        """Cases and labelled cases per slice — the header of every report."""
        out: dict[str, dict[str, int]] = {}
        for case in self.cases:
            bucket = out.setdefault(case.slice, {"cases": 0, "labelled": 0})
            bucket["cases"] += 1
            bucket["labelled"] += int(case.scored)
        return out

    def __str__(self) -> str:
        parts = [
            f"{name} {counts['labelled']}/{counts['cases']} labelled"
            for name, counts in sorted(self.summary().items())
        ]
        return f"eval set: {len(self.cases)} case(s) — " + (", ".join(parts) or "empty")


__all__ = ["SLICES", "EvalCase", "EvalSet", "FrameRef"]
