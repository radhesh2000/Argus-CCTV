"""The one module in this repository that can reach the internet.

Argus is sold on "raw video never leaves site". This module sends frames to
third-party APIs. Both of those are true at once only because of where the line
is drawn, so the line is drawn here, in code, and not in a README:

* **It is never imported by the runtime product.** Not by the engine, not by the
  API, not by the scheduler, not by ``argus/eval/harness.py`` at module scope.
  It is imported inside a function, after two gates pass.
  ``tests/test_eval.py::test_the_runtime_tree_never_imports_the_cloud_runner``
  imports the engine, the API and the VLM scheduler in a subprocess and asserts
  ``argus.eval`` is absent from ``sys.modules`` afterwards.
* **Two gates, not one.** ``OPENROUTER_API_KEY`` present in the environment
  **and** ``--cloud`` typed by a human. An environment variable alone cannot
  turn this on, because a developer who exported a key for something else has
  not consented to uploading a customer's footage; and a flag alone cannot,
  because there is nothing to authenticate with.
* **It says so, loudly, every time.** :func:`cloud_notice` is printed before the
  first request. An operator who runs this by accident finds out during the run,
  not in a bill.
* **The key is read from the environment and never written anywhere.** Not into
  the report, not into the JSON results, not into a log line. The results file
  records model ids and token counts.

**Why urllib.** Same reason ``argus/vlm/client.py`` uses it: this is a few
hundred JSON POSTs in a development session, the standard library does that
correctly, and the licence audit has one fewer package to clear.

**Why this module is lenient where the product is strict.** The runtime client
talks to one server we configure — vLLM with a constrained decoder — so a reply
that is not the schema is a bug worth failing on, and
:func:`argus.vlm.schema.parse_verdict` fails on it. This module talks to four
vendors it does not control, and on 2026-08-25 that difference cost a whole
424-case run: 387-420 errors per model out of 424. The post-mortem, from the
raw bodies rather than from the error strings:

* **Every candidate is a reasoning model, and OpenRouter budgets reasoning out
  of ``max_tokens``.** ``eval.max_tokens`` is 192 because that is what the
  appliance gives the local model, and all four vendors spent all 192 of it
  thinking. ``finish_reason`` was ``length`` on every failed call, with
  ``reasoning_tokens`` of 145-231. Qwen3.8-Max and Kimi K3 returned
  ``content: null`` — hence 396 and 363 "empty response". Gemini 3.7 Flash got
  as far as ``'Here is the'`` before the budget ran out — hence 383 "not JSON:
  Expecting value: line 1 column 1". Qwen3.8-27B sometimes reached the JSON and
  was cut off inside it — hence the "Unterminated string" tail. One cause, four
  symptoms. So the cloud path gets its own, much larger budget
  (``eval.cloud_max_tokens``) and asks for reasoning to be switched off.
* **"Switch reasoning off" is not universally accepted.** ``qwen/qwen3.8-max``
  and ``google/gemini-3.7-flash`` answer ``HTTP 400: Reasoning is mandatory for
  this endpoint and cannot be disabled``. So the ask degrades — see
  :meth:`OpenRouterRunner._degrade` — to bounded reasoning rather than failing,
  once per runner, not once per case.
* **``response_format`` is a hint out here, not a constraint.** It is passed
  through to providers that honour it and ignored by those that do not, and a
  provider that ignores it emits a ```` ```json ```` fence or a sentence of
  preamble. :func:`extract_verdict_json` strips both, and salvages a reply the
  budget cut in half.

The salvage has one hard limit, and it is the honest one: a fragment that never
reached ``triggered`` is an error, never a verdict. Handing ``{}`` to
``parse_verdict`` would score as "the model said no", which is a number the
model did not produce. See :func:`extract_verdict_json`.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from argus.eval.cases import EvalCase
from argus.eval.runners import RunnerResult
from argus.vlm.client import encode_frame
from argus.vlm.schema import PROMPT_SUFFIX, RESPONSE_SCHEMA, VlmResponseError, parse_verdict

# Sent so the traffic is identifiable at the other end. Neither header carries
# anything about the site, the footage or the operator.
ATTRIBUTION = {"X-Title": "Argus eval harness (development)"}

# What we ask for first: no reasoning at all. Reasoning tokens are billed and
# budgeted as completion tokens, and none of the reasoning is scored — the
# harness reads one JSON object. Paying for a chain of thought we throw away
# would inflate both the latency column and the cost column.
NO_REASONING: dict[str, Any] = {"effort": "none"}
# What we fall back to on an endpoint that refuses to switch it off. "low" is
# roughly 20% of `max_tokens` at OpenRouter, which at the cloud budget leaves
# far more than the ~60 tokens a verdict needs.
BOUNDED_REASONING: dict[str, Any] = {"effort": "low"}

# The structured-output ask, in descending order of how much it constrains the
# model. `json_schema` names the four fields; `json_object` only says "JSON".
# Both are advisory here; see the module docstring.
SCHEMA_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "argus_policy_verdict",
        "schema": RESPONSE_SCHEMA,
        "strict": True,
    },
}
OBJECT_FORMAT: dict[str, Any] = {"type": "json_object"}


class CloudDisabled(RuntimeError):
    """The cloud runner was asked for without both gates open."""


# --------------------------------------------------------------------------
# reading a third-party reply
# --------------------------------------------------------------------------


def message_text(message: dict[str, Any]) -> tuple[str, str]:
    """The most answer-shaped text in a reply, and which field it came from.

    ``content`` first, always. The fallbacks exist because a reasoning model
    whose budget ran out mid-thought returns ``content: null`` with the whole
    attempt sitting in ``reasoning`` — and a model that reasoned its way to
    ``{"triggered": true, ...}`` and was cut off before it could copy that into
    ``content`` has, in fact, answered. Reading it there is not generosity; the
    alternative is scoring a reply we can see as a non-reply.

    ``content`` is occasionally a list of parts rather than a string, which is
    the OpenAI content-parts shape leaking through a provider adapter.
    """
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(
            str(part.get("text") or "") for part in content if isinstance(part, dict)
        )
    if isinstance(content, str) and content.strip():
        return content, "content"
    for field_name in ("reasoning", "reasoning_content"):
        value = message.get(field_name)
        if isinstance(value, str) and value.strip():
            return value, field_name
    details = message.get("reasoning_details")
    if isinstance(details, list):
        joined = "\n".join(
            str(item.get("text") or "") for item in details if isinstance(item, dict)
        )
        if joined.strip():
            return joined, "reasoning_details"
    return "", "content"


def _strip_fences(text: str) -> str:
    """Remove a ```` ``` ```` code fence, opened or closed, with or without a tag."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    stripped = stripped[3:]
    # ```json, ```JSON, ```  — the tag runs to the end of the line.
    newline = stripped.find("\n")
    first = stripped[:newline] if newline >= 0 else stripped
    if first.strip().isalpha():
        stripped = stripped[newline + 1:] if newline >= 0 else ""
    closing = stripped.rfind("```")
    return (stripped[:closing] if closing >= 0 else stripped).strip()


def _close_open_structures(fragment: str) -> str:
    """Shut whatever a truncation left open, so the prefix can be parsed.

    A reply cut off at ``"reason": "a worker is stand`` still contains a
    verdict; the only thing wrong with it is the missing punctuation. This adds
    the punctuation back and nothing else — no values are invented, and a
    fragment that never reached ``triggered`` is rejected by the caller.
    """
    stack: list[str] = []
    in_string = escaped = False
    for char in fragment:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]" and stack:
            stack.pop()
    tail = fragment[:-1] if (in_string and escaped) else fragment
    if in_string:
        tail += '"'
    return tail + "".join(reversed(stack))


def _load_fragment(fragment: str) -> dict[str, Any] | None:
    """Parse the object at the head of ``fragment``, repairing a truncation.

    ``raw_decode`` rather than ``loads`` because ``fragment`` starts at a ``{``
    somewhere inside a reply and everything after the closing brace is the
    model's sign-off. ``loads`` calls that "Extra data" and fails, and the
    repair below would then answer the failure by deleting the object's last
    field — losing ``box_2d`` from a reply that was completely intact.

    Closing the open braces is enough for a cut inside a string. A cut inside a
    number or a key (``"confidence": 0.``) is not repairable, so the trailing
    member is dropped at the last comma and the close retried — bounded by the
    number of commas, which for this schema is three.
    """
    decoder = json.JSONDecoder()
    candidate = fragment
    while True:
        try:
            payload, _end = decoder.raw_decode(_close_open_structures(candidate))
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            return payload
        comma = _last_comma_outside_a_string(candidate)
        if comma < 0:
            return None
        candidate = candidate[:comma]


def _last_comma_outside_a_string(fragment: str) -> int:
    in_string = escaped = False
    last = -1
    for index, char in enumerate(fragment):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == ",":
            last = index
    return last


def extract_verdict_json(text: str) -> str:
    """The first JSON object in ``text`` that actually holds a verdict.

    Scans every ``{`` rather than only the first, because the text may be a
    model's reasoning with an example object in it before the real answer, and
    because prose either side of the JSON is normal from a provider that
    ignored ``response_format``.

    **An object without ``triggered`` is not a verdict.** This is the whole
    reason the function returns text for ``parse_verdict`` instead of just
    calling it: ``parse_verdict`` reads a missing ``triggered`` as ``false``,
    which is correct for a constrained decoder that cannot omit it and wrong
    for a cloud reply that was cut off before reaching it. Scoring that as "the
    model said no" would put an answer in the model's mouth, and a harness that
    does that is worse than useless — it is confidently wrong. So a truncation
    that stopped short of the verdict raises, and lands in the error column
    where it belongs.

    Returns re-serialised JSON, so the verdict itself is still read by the
    production :func:`argus.vlm.schema.parse_verdict` and the cloud numbers
    still describe the parser the appliance ships.
    """
    body = _strip_fences(text or "")
    if not body:
        raise VlmResponseError("empty response")
    seen_object = False
    for start in range(len(body)):
        if body[start] != "{":
            continue
        payload = _load_fragment(body[start:])
        if payload is None:
            continue
        seen_object = True
        if "triggered" in payload:
            return json.dumps(payload)
    if seen_object:
        raise VlmResponseError("JSON object without a `triggered` field")
    raise VlmResponseError(f"no JSON object in {len(body)} char(s) of reply")


def _worth_retrying(error: str) -> bool:
    """Whether a second attempt could plausibly land differently.

    An unreadable reply is worth one more roll — decoding is sampled even at
    temperature 0 across providers, and a run of several hundred cases should
    not be scored on a single bad draw. A 402 or a 404 is not: those are the
    same every time, and retrying them spends money to learn nothing.
    """
    text = error.lower()
    if "vlmresponseerror" in text or "unexpected response shape" in text:
        return True
    return any(
        marker in text
        for marker in ("http 408", "http 409", "http 429", "http 5", "unreachable", "timeout")
    )


def cloud_notice(models: list[str], case_count: int, endpoint: str) -> str:
    """The warning printed before any frame leaves the box."""
    return (
        "\n"
        "  ┌─ CLOUD EVALUATION ─────────────────────────────────────────────\n"
        f"  │ Sending {case_count} frame(s) to {endpoint}\n"
        f"  │ Models: {', '.join(models)}\n"
        "  │ These are THIRD-PARTY APIs. Frames leave this appliance.\n"
        "  │ Development evaluation only — no runtime path does this, and\n"
        "  │ the shipped product has no code that can.\n"
        "  └────────────────────────────────────────────────────────────────\n"
    )


def api_key(env_var: str = "OPENROUTER_API_KEY") -> str:
    """The key, or "". Never logged, never returned to a report."""
    return os.environ.get(env_var, "").strip()


@dataclass
class OpenRouterRunner:
    """One OpenRouter model, behind the same interface as the local one."""

    name: str
    base_url: str = "https://openrouter.ai/api/v1"
    key_env: str = "OPENROUTER_API_KEY"
    timeout_sec: float = 120.0
    # Deliberately not `eval.max_tokens`. The local runner's 192 is the budget
    # the appliance gives the served model and must not move, but out here the
    # same number buys a third of a chain of thought and no answer at all. See
    # the module docstring.
    max_tokens: int = 1536
    max_image_width: int = 1024
    jpeg_quality: int = 85
    # One retry per case on a reply we could not read; see `_worth_retrying`.
    attempts: int = 2
    backoff_sec: float = 2.0
    label: str = ""
    # Filled by `attach_pricing` from the catalog; USD per token.
    prompt_usd: float | None = field(default=None, init=False)
    completion_usd: float | None = field(default=None, init=False)
    vision_verified: bool | None = field(default=None, init=False)
    # Negotiated with the endpoint on the first request that objects, then left
    # alone. See `_degrade`.
    reasoning: dict[str, Any] | None = field(default=None, init=False)
    response_format: dict[str, Any] | None = field(default=None, init=False)
    # What the run should say about itself, in the report's notes.
    notes: list[str] = field(default_factory=list, init=False)
    retried: int = field(default=0, init=False)
    # Bumped whenever `_degrade` changes an option, so a request already in
    # flight under the old options can tell that it was overtaken.
    _generation: int = field(default=0, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.label = self.label or f"{self.name} (openrouter)"
        self.reasoning = dict(NO_REASONING)
        self.response_format = dict(SCHEMA_FORMAT)
        if not api_key(self.key_env):
            raise CloudDisabled(f"{self.key_env} is not set")

    # --- wire -------------------------------------------------------------

    def _request(self, path: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}{path}"
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(  # noqa: S310 - fixed https endpoint from config
            url,
            data=data,
            method="GET" if payload is None else "POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key(self.key_env)}",
                "Accept": "application/json",
                **ATTRIBUTION,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise RuntimeError(f"unreachable: {type(exc).__name__}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"non-JSON reply: {exc}") from exc

    def _payload(self, prompt: str, data_uri: str) -> dict[str, Any]:
        """The same question the local path asks, addressed to a stranger.

        The prompt, the image encoding and the schema are the local runner's,
        because changing them would measure a different model. What is added is
        everything OpenRouter needs to be told and vLLM does not: how much
        thinking to do, and how much room the answer gets.
        """
        payload: dict[str, Any] = {
            "model": self.name,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt + PROMPT_SUFFIX},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "usage": {"include": True},
        }
        if self.response_format is not None:
            payload["response_format"] = self.response_format
        if self.reasoning is not None:
            payload["reasoning"] = self.reasoning
        return payload

    def _degrade(self, detail: str) -> bool:
        """Drop one request option the endpoint just rejected. True if it moved.

        Providers differ in what they will accept, and the differences only
        surface as a 400 on a live call — the catalog does not say that Gemini
        3.7 Flash cannot have its reasoning switched off. Rather than encode a
        table of vendor quirks that will be stale by the time it is read, ask
        for the ideal request, listen to the refusal, and settle. Each option
        can only step down once, so this terminates after at most two extra
        requests per runner for the whole run.
        """
        text = detail.lower()
        if "http 400" not in text:
            return False
        with self._lock:
            if "reasoning" in text and self.reasoning != BOUNDED_REASONING:
                self.reasoning = dict(BOUNDED_REASONING)
                self._generation += 1
                self._note(
                    "reasoning cannot be disabled on this endpoint — bounded to "
                    f"effort={BOUNDED_REASONING['effort']} instead (its tokens are billed)"
                )
                return True
            if ("response_format" in text or "json_schema" in text or "schema" in text) \
                    and self.response_format != OBJECT_FORMAT:
                self.response_format = dict(OBJECT_FORMAT)
                self._generation += 1
                self._note("json_schema rejected — asking for json_object instead")
                return True
        return False

    def _note(self, text: str) -> None:
        line = f"cloud eval: {self.name}: {text}"
        if line not in self.notes:
            self.notes.append(line)

    def _post(self, prompt: str, data_uri: str) -> dict[str, Any]:
        """One completion, re-sent if the endpoint refuses an option we can drop.

        The generation check is the concurrency half of :meth:`_degrade`. With
        two cases in flight, both are sent with the options the endpoint is
        about to reject, and only the first gets to *be* the one that steps
        them down — the second finds the step already taken and would otherwise
        be recorded as a hard error for a request nobody would send again. It
        was sent under the old options, so it gets the same second chance.
        """
        while True:
            generation = self._generation
            try:
                return self._request(
                    "/chat/completions", self._payload(prompt, data_uri), self.timeout_sec,
                )
            except RuntimeError as exc:
                if not self._degrade(str(exc)) and self._generation == generation:
                    raise

    # --- runner interface -------------------------------------------------

    async def evaluate(self, case: EvalCase) -> RunnerResult:
        """Ask once; ask again if the answer was unreadable; then write it down.

        Latency is the kept attempt's, not the sum: the column answers "how long
        does one evaluation take", and a retry is a property of the harness
        rather than of the evaluation. How often it happened is a note instead.
        """
        frame = await asyncio.to_thread(case.frame.load)
        encoded = await asyncio.to_thread(
            encode_frame, frame,
            max_width=self.max_image_width, quality=self.jpeg_quality,
        )
        result = await self._attempt(case, encoded)
        for attempt in range(1, max(1, self.attempts)):
            if result.error is None or not _worth_retrying(result.error):
                break
            await asyncio.sleep(self.backoff_sec * attempt)
            self.retried += 1
            result = await self._attempt(case, encoded)
        return result

    async def _attempt(self, case: EvalCase, encoded: Any) -> RunnerResult:
        started = time.monotonic()
        try:
            body = await asyncio.to_thread(self._post, case.prompt, encoded.data_uri)
        except Exception as exc:  # noqa: BLE001 - a failed model is a measured model
            return RunnerResult(
                case_id=case.id, model=self.name, triggered=None,
                latency_sec=time.monotonic() - started,
                error=f"{type(exc).__name__}: {exc}"[:300],
            )
        elapsed = time.monotonic() - started

        usage = body.get("usage") or {}
        tokens = {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message")
        if not isinstance(message, dict):
            return RunnerResult(
                case_id=case.id, model=self.name, triggered=None, latency_sec=elapsed,
                error=f"unexpected response shape: {sorted(body)}"[:300], **tokens,
            )
        text, source = message_text(message)
        try:
            verdict = parse_verdict(
                extract_verdict_json(text), width=encoded.width, height=encoded.height,
            )
        except VlmResponseError as exc:
            return RunnerResult(
                case_id=case.id, model=self.name, triggered=None, latency_sec=elapsed,
                error=f"VlmResponseError: {exc} [{self._context(choice, usage, source)}]"[:300],
                **tokens,
            )
        return RunnerResult(
            case_id=case.id, model=self.name,
            triggered=verdict.triggered, confidence=verdict.confidence,
            reason=verdict.reason, latency_sec=elapsed, **tokens,
        )

    @staticmethod
    def _context(choice: dict[str, Any], usage: dict[str, Any], source: str) -> str:
        """Why the reply was unreadable, in the error string itself.

        The 2026-08-25 run recorded 396 rows of bare "empty response" and it
        took a packet capture to learn that all 192 completion tokens had gone
        into hidden reasoning. The three numbers that would have said so cost
        nothing to carry, so they are carried.
        """
        details = usage.get("completion_tokens_details") or {}
        parts = [f"finish={choice.get('finish_reason')}", f"read={source}"]
        if details.get("reasoning_tokens"):
            parts.append(f"reasoning_tokens={details['reasoning_tokens']}")
        if usage.get("completion_tokens") is not None:
            parts.append(f"completion_tokens={usage['completion_tokens']}")
        return ", ".join(parts)

    def pricing(self) -> tuple[float, float] | None:
        if self.prompt_usd is None or self.completion_usd is None:
            return None
        return (self.prompt_usd, self.completion_usd)

    # --- catalog ----------------------------------------------------------

    def catalog(self, timeout: float = 30.0) -> list[dict[str, Any]]:
        return list((self._request("/models", None, timeout) or {}).get("data") or [])


def attach_pricing(runners: list[OpenRouterRunner]) -> list[str]:
    """Fill in per-token prices and vision support from the catalog, once.

    Fetched rather than hard-coded because a price in a docstring is wrong the
    week after it is written, and the report's cost column is only worth
    printing if it came from the provider. One request serves every runner.

    Returns notes for the report — including any model the catalog says is not
    vision-capable, which is the check that would have caught ``z-ai/glm-5.2``
    before somebody spent an afternoon on it.
    """
    notes: list[str] = []
    if not runners:
        return notes
    try:
        entries = runners[0].catalog()
    except Exception as exc:  # noqa: BLE001 - pricing is a nice-to-have, not the run
        return [f"pricing: catalog unavailable ({type(exc).__name__}: {exc}) — cost column omitted"]

    by_id = {str(entry.get("id")): entry for entry in entries}
    for runner in runners:
        entry = by_id.get(runner.name)
        if entry is None:
            notes.append(f"pricing: {runner.name} not found in the catalog")
            continue
        pricing = entry.get("pricing") or {}
        try:
            runner.prompt_usd = float(pricing.get("prompt"))
            runner.completion_usd = float(pricing.get("completion"))
        except (TypeError, ValueError):
            notes.append(f"pricing: {runner.name} publishes no per-token price")
        modalities = ((entry.get("architecture") or {}).get("input_modalities")) or []
        runner.vision_verified = "image" in [str(m).lower() for m in modalities]
        if runner.vision_verified is False:
            notes.append(
                f"WARNING: {runner.name} advertises input modalities {modalities} — "
                "no image input. Its scores below are meaningless."
            )
    return notes


def build_runners(
    models: list[str], *, base_url: str, key_env: str, timeout_sec: float, max_tokens: int,
) -> list[OpenRouterRunner]:
    """Construct one runner per candidate. Raises CloudDisabled without a key."""
    return [
        OpenRouterRunner(
            name=model, base_url=base_url, key_env=key_env,
            timeout_sec=timeout_sec, max_tokens=max_tokens,
        )
        for model in models
    ]


__all__ = [
    "ATTRIBUTION",
    "BOUNDED_REASONING",
    "NO_REASONING",
    "OBJECT_FORMAT",
    "SCHEMA_FORMAT",
    "CloudDisabled",
    "OpenRouterRunner",
    "api_key",
    "attach_pricing",
    "build_runners",
    "cloud_notice",
    "extract_verdict_json",
    "message_text",
]
