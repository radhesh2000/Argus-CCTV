"""An async OpenAI-compatible client, on the standard library.

**Why no HTTP library.** The argus package's dependency rule is permissive
licences only and as few of them as possible, and this client's whole workload
is two or three JSON POSTs every six seconds. ``urllib.request`` does that
correctly, releases the GIL for the duration of the socket wait, and adds
nothing to the dependency tree; ``asyncio.to_thread`` over a bounded pool turns
it into the concurrent, awaitable surface the scheduler wants. The server on the
other end may be anything that speaks ``/v1/chat/completions`` — vLLM today,
something else tomorrow — which is the actual portability that matters here.

**Why concurrency is bounded.** The semaphore is not politeness. A single GB10
serves this model and the RF-DETR detector out of the same 128 GB and the same
SMs; letting a backlog of policy evaluations pile into vLLM's scheduler would
trade detection latency for VLM throughput, which is exactly the wrong trade on
a box whose job is to notice things quickly.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from argus.config import VlmConfig
from argus.media.annotate import to_image
from argus.vlm.schema import (
    PROMPT_SUFFIX,
    RESPONSE_SCHEMA,
    PolicyVerdict,
    VlmResponseError,
    parse_verdict,
)


class VlmUnavailable(RuntimeError):
    """The server could not be reached, or answered with an error status."""


@dataclass
class ClientStats:
    """What the health endpoint reports about inference."""

    requests: int = 0
    failures: int = 0
    last_error: str | None = None
    last_inference_at: float | None = None
    last_latency_sec: float = 0.0
    # Exponentially smoothed, 10-request memory — the same shape the decoder
    # uses for fps, so the two numbers describe comparable windows.
    avg_latency_sec: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    in_flight: int = 0
    queued: int = 0

    def as_dict(self) -> dict[str, Any]:
        age = None if self.last_inference_at is None else round(time.time() - self.last_inference_at, 1)
        return {
            "requests": self.requests,
            "failures": self.failures,
            "last_error": self.last_error,
            "last_inference_age_sec": age,
            "last_latency_sec": round(self.last_latency_sec, 3),
            "avg_latency_sec": round(self.avg_latency_sec, 3),
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "in_flight": self.in_flight,
            "queue_depth": self.queued,
        }


@dataclass
class _Encoded:
    data_uri: str
    width: int
    height: int
    bytes: int = 0


def encode_frame(
    frame: np.ndarray | Image.Image, *, max_width: int = 1024, quality: int = 85
) -> _Encoded:
    """BGR frame (or PIL image) -> a JPEG ``data:`` URI, downscaled.

    Downscaling is not a size optimisation, it is a cost one: a VLM tokenises an
    image by area, so halving the width quarters the prompt. 1024px is past the
    resolution where more pixels change the answer to "is that person holding a
    phone", and the decoder only hands us 960px anyway — so in practice this
    downscales nothing on the live path and protects the API path, where a
    caller can submit a 4K still.
    """
    image = frame if isinstance(frame, Image.Image) else to_image(frame)
    if image.width > max_width:
        height = round(image.height * max_width / image.width)
        image = image.resize((max_width, height), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=int(quality), optimize=True)
    payload = buffer.getvalue()
    return _Encoded(
        data_uri="data:image/jpeg;base64," + base64.b64encode(payload).decode("ascii"),
        width=image.width,
        height=image.height,
        bytes=len(payload),
    )


class VlmClient:
    """Talks to one OpenAI-compatible endpoint.

    Every public method is a coroutine. None of them raise for a *model* problem
    — an unparseable answer becomes a ``VlmResponseError`` the caller can log and
    skip — but all of them raise :class:`VlmUnavailable` when the server itself
    is unreachable, because that is a condition the scheduler must react to
    (degrade) rather than retry blindly.
    """

    def __init__(self, config: VlmConfig, *, session_id: str = "argus"):
        self.config = config
        self.session_id = session_id
        self.stats = ClientStats()
        self._semaphore: asyncio.Semaphore | None = None
        self._priority_semaphore: asyncio.Semaphore | None = None
        self._model_name: str | None = None

    # --- plumbing ---------------------------------------------------------

    @property
    def base_url(self) -> str:
        return self.config.endpoint

    def _limiter(self, priority: bool = False) -> asyncio.Semaphore:
        """Created lazily: a Semaphore binds to the running loop, and the client
        is constructed on the engine's thread before the scheduler's loop exists.

        Priority work gets its own single slot rather than competing for the
        shared budget. The only priority caller is the verification pass, which
        sits *in the path of an alert*; making it wait behind routine policy
        sampling would delay the one request whose latency a person actually
        feels. Total in-flight is therefore ``max_concurrency + 1``, which is
        what the GPU budget in ``vlm.server.gpu_memory_utilization`` is sized for.
        """
        if priority:
            if self._priority_semaphore is None:
                self._priority_semaphore = asyncio.Semaphore(1)
            return self._priority_semaphore
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.config.max_concurrency)
        return self._semaphore

    def _request(self, path: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
        """One blocking HTTP round trip. Always called through ``to_thread``."""
        url = f"{self.base_url}{path}"
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(  # noqa: S310 - fixed http(s) scheme from config
            url,
            data=data,
            method="GET" if payload is None else "POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            raise VlmUnavailable(f"{url} -> HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise VlmUnavailable(f"{url} unreachable: {type(exc).__name__}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise VlmUnavailable(f"{url} returned non-JSON: {exc}") from exc

    # --- health -----------------------------------------------------------

    async def models(self, timeout: float = 5.0) -> list[str]:
        payload = await asyncio.to_thread(self._request, "/models", None, timeout)
        return [str(entry.get("id")) for entry in payload.get("data", []) if entry.get("id")]

    async def health(self, timeout: float = 5.0) -> dict[str, Any]:
        """Is there a model behind this URL, and which one.

        Reports rather than raises: this backs ``/api/health``, and "the VLM is
        down" is a health answer, not a health failure.
        """
        started = time.monotonic()
        try:
            served = await self.models(timeout=timeout)
        except VlmUnavailable as exc:
            return {"available": False, "reason": str(exc), "model": None,
                    "endpoint": self.base_url}
        self._model_name = served[0] if served else None
        return {
            "available": True,
            "endpoint": self.base_url,
            "model": self._model_name,
            "models": served,
            "probe_sec": round(time.monotonic() - started, 3),
        }

    # --- inference --------------------------------------------------------

    def _payload(self, prompt: str, image: _Encoded, *, max_tokens: int | None) -> dict[str, Any]:
        return {
            "model": self.config.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt + PROMPT_SUFFIX},
                    {"type": "image_url", "image_url": {"url": image.data_uri}},
                ],
            }],
            "temperature": self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            # Constrained decoding. vLLM routes this through xgrammar, so the
            # reply is JSON matching RESPONSE_SCHEMA by construction rather than
            # by the model's goodwill — which is the single biggest reliability
            # difference from v1's Ollama `format=` hint.
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "argus_policy_verdict",
                    "schema": RESPONSE_SCHEMA,
                    "strict": True,
                },
            },
        }

    async def evaluate(
        self,
        prompt: str,
        frame: np.ndarray | Image.Image,
        *,
        timeout: float | None = None,
        max_tokens: int | None = None,
        priority: bool = False,
    ) -> PolicyVerdict:
        """Ask one policy question about one frame.

        ``priority`` takes the reserved slot instead of the shared budget — see
        :meth:`_limiter`. Raises :class:`VlmUnavailable` if the server is
        unreachable and :class:`VlmResponseError` if it answered with something
        unreadable.
        """
        image = await asyncio.to_thread(
            encode_frame, frame,
            max_width=self.config.max_image_width, quality=self.config.jpeg_quality,
        )
        payload = self._payload(prompt, image, max_tokens=max_tokens)
        limit = timeout or self.config.timeout_sec

        # Acquire and release explicitly rather than with `async with`, so that
        # `queued` and `in_flight` are decremented on the cancellation path too.
        # These two numbers are what the health endpoint calls queue depth, and
        # a counter that leaks on timeout would report a permanently busy server.
        limiter = self._limiter(priority)
        self.stats.queued += 1
        try:
            await limiter.acquire()
        finally:
            self.stats.queued -= 1

        self.stats.in_flight += 1
        started = time.monotonic()
        try:
            body = await asyncio.to_thread(self._request, "/chat/completions", payload, limit)
        except VlmUnavailable as exc:
            self.stats.failures += 1
            self.stats.last_error = str(exc)[:300]
            raise
        finally:
            self.stats.in_flight -= 1
            limiter.release()

        elapsed = time.monotonic() - started
        self._record(elapsed, body)

        try:
            choice = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise VlmResponseError(f"unexpected response shape: {exc}") from exc

        verdict = parse_verdict(choice, width=image.width, height=image.height)
        verdict.latency_sec = elapsed
        usage = body.get("usage") or {}
        verdict.prompt_tokens = usage.get("prompt_tokens")
        verdict.completion_tokens = usage.get("completion_tokens")
        # The box came back in the *sent* image's pixel space; hand it to the
        # caller as a 0-1 fraction so nobody downstream has to remember which
        # resolution the model actually saw.
        if verdict.box_2d is not None:
            x1, y1, x2, y2 = verdict.box_2d
            verdict.box_2d = (
                x1 / image.width, y1 / image.height, x2 / image.width, y2 / image.height,
            )
        return verdict

    def _record(self, elapsed: float, body: dict[str, Any]) -> None:
        stats = self.stats
        stats.requests += 1
        stats.last_inference_at = time.time()
        stats.last_latency_sec = elapsed
        stats.avg_latency_sec = (
            elapsed if stats.avg_latency_sec == 0.0
            else stats.avg_latency_sec + 0.1 * (elapsed - stats.avg_latency_sec)
        )
        usage = body.get("usage") or {}
        stats.total_prompt_tokens += int(usage.get("prompt_tokens") or 0)
        stats.total_completion_tokens += int(usage.get("completion_tokens") or 0)


__all__ = ["ClientStats", "VlmClient", "VlmUnavailable", "encode_frame"]
