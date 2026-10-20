"""Asking one model one question, and writing down what happened.

**The local runner is the production client.** ``LocalRunner`` builds a
:class:`argus.vlm.client.VlmClient` against a synthetic ``VlmConfig`` and calls
``evaluate`` — the same encoder, the same JSON-schema constraint, the same
downscale, the same parse. An eval harness with its own HTTP client would
measure a model this appliance does not run: a different image size is a
different token count, and an unconstrained decode is a different failure mode.
The only thing overridden is the model id and the endpoint, which is exactly the
axis under test.

**Failures are results.** A model that times out, refuses, or answers something
unparseable has told us something about itself, and dropping those rows would
flatter it. Every error becomes a :class:`RunnerResult` with ``triggered=None``,
which the scorer counts as an error and never as a verdict — a model that errors
on half the set cannot post a good F1 on the other half without the error count
sitting next to it.

**Concurrency is 1 by default.** The headline number this harness produces
alongside accuracy is latency, and latency at concurrency 4 is a load test.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from argus.config import ArgusConfig, VlmConfig
from argus.eval.cases import EvalCase
from argus.vlm.client import VlmClient, VlmUnavailable
from argus.vlm.schema import VlmResponseError


@dataclass(slots=True)
class RunnerResult:
    """One (model, case) outcome."""

    case_id: str
    model: str
    triggered: bool | None            # None = the model did not answer
    confidence: float = 0.0
    reason: str = ""
    latency_sec: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.triggered is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "model": self.model,
            "triggered": self.triggered,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "latency_sec": round(self.latency_sec, 3),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "error": self.error,
        }


def _emit(*parts: Any) -> None:
    """Flushed ``print`` — see :func:`argus.eval.harness.emit`."""
    print(*parts, flush=True)


class Runner(Protocol):
    """What the harness needs from a model, and nothing else."""

    name: str
    label: str

    async def evaluate(self, case: EvalCase) -> RunnerResult: ...

    def pricing(self) -> tuple[float, float] | None:
        """USD per prompt token and per completion token, when known."""


@dataclass
class LocalRunner:
    """An OpenAI-compatible endpoint on this box — the vLLM service by default."""

    name: str
    base_url: str
    api_key: str = "argus-local"
    timeout_sec: float = 120.0
    max_tokens: int = 192
    max_concurrency: int = 1
    max_image_width: int = 1024
    jpeg_quality: int = 85
    label: str = ""
    _client: VlmClient | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.label = self.label or f"{self.name} (local)"

    @classmethod
    def from_config(cls, config: ArgusConfig, *, model: str | None = None) -> LocalRunner:
        """Point at whatever ``argus vlm start`` is serving."""
        return cls(
            name=model or config.vlm.model,
            base_url=config.vlm.endpoint,
            api_key=config.vlm.api_key,
            timeout_sec=config.eval.timeout_sec,
            max_tokens=config.eval.max_tokens,
            max_concurrency=max(1, config.eval.concurrency),
            max_image_width=config.vlm.max_image_width,
            jpeg_quality=config.vlm.jpeg_quality,
        )

    @property
    def client(self) -> VlmClient:
        if self._client is None:
            self._client = VlmClient(VlmConfig(
                base_url=self.base_url,
                model=self.name,
                api_key=self.api_key,
                timeout_sec=self.timeout_sec,
                max_concurrency=self.max_concurrency,
                max_image_width=self.max_image_width,
                jpeg_quality=self.jpeg_quality,
                max_tokens=self.max_tokens,
                temperature=0.0,
            ))
        return self._client

    async def health(self) -> dict[str, Any]:
        return await self.client.health()

    async def evaluate(self, case: EvalCase) -> RunnerResult:
        image = await asyncio.to_thread(case.frame.load)
        started = time.monotonic()
        try:
            verdict = await self.client.evaluate(
                case.prompt, image, timeout=self.timeout_sec, max_tokens=self.max_tokens,
            )
        except (VlmUnavailable, VlmResponseError) as exc:
            return RunnerResult(
                case_id=case.id, model=self.name, triggered=None,
                latency_sec=time.monotonic() - started,
                error=f"{type(exc).__name__}: {exc}"[:300],
            )
        except Exception as exc:  # noqa: BLE001 - one bad case must not end the run
            return RunnerResult(
                case_id=case.id, model=self.name, triggered=None,
                latency_sec=time.monotonic() - started,
                error=f"{type(exc).__name__}: {exc}"[:300],
            )
        return RunnerResult(
            case_id=case.id, model=self.name,
            triggered=verdict.triggered, confidence=verdict.confidence,
            reason=verdict.reason, latency_sec=verdict.latency_sec,
            prompt_tokens=verdict.prompt_tokens,
            completion_tokens=verdict.completion_tokens,
        )

    def pricing(self) -> tuple[float, float] | None:
        """None — a local model's marginal cost is GPU seconds, not dollars.

        Reporting $0.00 per 1 000 evaluations would be a lie of a different
        kind: the hardware is paid for, but the GPU time is real and it comes
        out of the detector's budget. The report gives GPU-seconds instead.
        """
        return None


async def run_cases(
    runner: Runner,
    cases: list[EvalCase],
    *,
    concurrency: int = 1,
    log: Any = _emit,
    progress_every: int = 10,
) -> list[RunnerResult]:
    """Evaluate every case against one model.

    Order is preserved regardless of concurrency, because the disagreements dump
    joins results by case id and a reader comparing two models' reasons should
    see them in the same order they appear in the set.
    """
    if not cases:
        return []
    semaphore = asyncio.Semaphore(max(1, concurrency))
    done = 0
    started = time.monotonic()

    async def one(case: EvalCase) -> RunnerResult:
        nonlocal done
        async with semaphore:
            result = await runner.evaluate(case)
        done += 1
        if progress_every and done % progress_every == 0:
            rate = done / max(1e-6, time.monotonic() - started)
            log(f"  {runner.label}: {done}/{len(cases)} ({rate:.2f}/s)")
        return result

    return list(await asyncio.gather(*(one(case) for case in cases)))


__all__ = ["LocalRunner", "Runner", "RunnerResult", "run_cases"]
