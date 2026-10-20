"""The index writer: sample, batch, embed, store — without touching the engine.

``argus run`` owns one :class:`IndexRunner` and touches it in exactly three
places, the same shape as the VLM runner: it offers every decoded frame, it
offers the source frame of every event it raises, and it folds :meth:`stats`
into what it publishes. Everything else — the sampling clock, the queue, the
batching, the thumbnails, the vector writes — is behind this class and on its
own thread.

**Three rules, and they are the whole design.**

*Reuse the frames the engine already decoded.* The sampler is fed from
``Engine._on_frame``, the same callback the vision loop and the VLM read. A
second decode path for indexing would double the ffmpeg process count on a box
whose measured ceiling is the number of Python threads contending for the GIL
(docs/BENCHMARKS.md), and it would index a *different* frame from the one the
detector saw.

*Cost the decode path one comparison.* :meth:`offer` is a dict lookup, a float
subtraction and — for one frame in ``sample_interval_sec * fps`` — a bounded
queue put. At 48 cameras and 5 fps that is 240 comparisons a second and 4.8
enqueues. Everything expensive happens on the worker.

*Fall behind by indexing less, never by blocking.* The queue is bounded and
drops its **oldest** entry when full. A frame the sampler could not keep up with
is a gap in the index; a frame that blocked the decode path is a gap in the
video. Those are not the same failure and the queue is where that is decided.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from argus.config import ArgusConfig
from argus.index.embedder import Embedder, embedder_from_config
from argus.index.store import IndexedFrame, VectorStore, store_from_config


@dataclass(slots=True)
class PendingFrame:
    """One thing waiting to be embedded, already detached from the decoder."""

    camera_id: str
    ts: float
    image: np.ndarray
    kind: str = "sample"
    event_id: int | None = None
    # Set for event frames: the annotated snapshot already on disk is the
    # thumbnail, so no second image is written.
    snapshot_path: str | None = None
    queued_at: float = field(default_factory=time.monotonic)


def save_thumbnail(image: np.ndarray, path: Path, *, width: int, quality: int) -> tuple[int, int]:
    """Write the small JPEG a search result renders. Returns the frame's
    original ``(width, height)``, which is what the UI needs to place a box.

    Deliberately not ``argus.media.annotate.save_snapshot``: that writes
    evidence at the full decode resolution, and this writes a contact sheet.
    Measured on this box's own output — 162 KB per evidence snapshot against
    **15.1 KB** per 320 px index thumbnail — so at 48 cameras and a 10 s
    interval (414 720 frames a day) the difference between the two is
    **61 GB a day**.
    """
    from PIL import Image

    height, source_width = image.shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = Image.fromarray(np.ascontiguousarray(image[:, :, ::-1]))
    scale = min(1.0, width / max(1, source_width))
    if scale < 1.0:
        rgb = rgb.resize((max(1, int(source_width * scale)), max(1, int(height * scale))),
                         Image.BILINEAR)
    rgb.save(path, format="JPEG", quality=quality, optimize=False)
    return source_width, height


class IndexRunner:
    """Owns the embedding worker, its queue and the vector store."""

    def __init__(
        self,
        config: ArgusConfig,
        *,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        log: Callable[..., None] = print,
    ):
        self.config = config
        self.settings = config.index
        self.log = log
        # Both are built lazily by `start()` unless injected, so constructing a
        # runner never loads a 750 MB checkpoint — which matters because the
        # engine constructs one before it knows whether indexing is on.
        self._embedder = embedder
        self._store = store

        self._queue: list[PendingFrame] = []
        # A condition rather than a lock plus an event, because the worker has
        # two things to wait for — "anything at all" and "enough to be worth a
        # forward pass" — and both have to wake the instant `stop()` is called.
        self._queue_lock = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_sample: dict[str, float] = {}

        self.indexed = 0
        self.events_indexed = 0
        self.dropped = 0
        self.failures = 0
        self.batches = 0
        self.embed_seconds = 0.0
        self.last_error: str | None = None
        self.ready = False
        self.reason: str | None = None

    # --- lifecycle --------------------------------------------------------

    def start(self) -> bool:
        """Load the model, open the store, start the worker.

        Never raises. An index that will not come up is a degraded appliance —
        the detector, the VLM, the alerts and the console are untouched, and
        ``/api/health`` says which piece is missing. That is the same contract
        the VLM server has, and for the same reason.
        """
        if self._thread is not None:
            return self.ready
        try:
            if self._embedder is None:
                self._embedder = embedder_from_config(self.config)
            if self._store is None:
                self._store = store_from_config(
                    self.config, dim=self._embedder.dim, model=self._embedder.name
                )
        except Exception as exc:  # noqa: BLE001 - a missing checkpoint must not end the run
            self.reason = f"{type(exc).__name__}: {exc}"
            self.log(f"index: disabled — {self.reason}")
            return False
        self.ready = True
        self.reason = None
        self._thread = threading.Thread(target=self._loop, daemon=True, name="index")
        self._thread.start()
        return True

    def stop(self, timeout: float = 15.0) -> None:
        """Drain what is queued, then stop.

        Draining rather than dropping: the frames in the queue at shutdown are
        the most recent ones the appliance saw, which is exactly the window an
        operator searches first after an incident.
        """
        self._stop.set()
        with self._queue_lock:
            self._queue_lock.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    # --- intake (called from the decode path — keep it cheap) -------------

    def offer(self, frame: Any) -> bool:
        """Consider a decoded frame for the periodic sample.

        Returns whether it was taken, which is what makes the sampling clock
        testable without a clock.
        """
        if not self.ready:
            return False
        camera_id = frame.camera_id
        last = self._last_sample.get(camera_id)
        if last is not None and frame.ts - last < self.settings.sample_interval_sec:
            return False
        self._last_sample[camera_id] = frame.ts
        return self._enqueue(PendingFrame(camera_id=camera_id, ts=frame.ts, image=frame.image))

    def offer_event(
        self, camera_id: str, event_id: int, image: np.ndarray, ts: float,
        snapshot_path: str | None,
    ) -> bool:
        """Index the frame an event was raised on.

        The *source* frame, not the annotated snapshot: the banner, the boxes
        and the polygon are drawn by Argus, and embedding them teaches the index
        that "a red rectangle" is a feature of the scene. The annotated file is
        still what a result renders — ``snapshot_path`` — because that is what
        the operator recognises.

        Event frames bypass the sampling clock. An event is the most searchable
        thing this appliance produces and there are a few an hour, not five a
        second.
        """
        if not self.ready:
            return False
        return self._enqueue(PendingFrame(
            camera_id=camera_id, ts=ts, image=image, kind="event",
            event_id=event_id, snapshot_path=snapshot_path,
        ))

    def _enqueue(self, pending: PendingFrame) -> bool:
        with self._queue_lock:
            if len(self._queue) >= self.settings.queue_depth:
                # Drop the oldest *sample*; never an event. An event frame is
                # the one thing here that has a row in the appliance database
                # pointing at it, and a search that cannot find an alert it
                # raised is the failure this feature exists to prevent.
                victim = next(
                    (i for i, held in enumerate(self._queue) if held.kind == "sample"), None
                )
                if victim is None:
                    self.dropped += 1
                    return False
                self._queue.pop(victim)
                self.dropped += 1
            self._queue.append(pending)
            self._queue_lock.notify()
        return True

    # --- the worker -------------------------------------------------------

    def _take_batch(self) -> list[PendingFrame]:
        """Block until there is work, gather a batch, and hand it over.

        The gathering is the whole point, and it was measured. A worker that
        drains the queue the moment anything lands in it runs at **batch 1**
        forever on a lightly loaded appliance — the frames arrive slower than
        one forward pass, so there is never a second one waiting. Measured on
        this box at 10 indexed frames/s: 3 608 batches for 3 612 frames, and
        10.9 ms per frame against the 3.35 ms a full batch of 8 costs.

        So the worker waits up to ``batch_wait_sec`` **from the moment the
        oldest queued frame arrived** — ``PendingFrame.queued_at``, not from
        when the worker happened to look, so a batch that formed while the
        previous one was being embedded is not delayed a second time. It stops
        waiting the moment it has a full batch or the runner is stopping.
        Indexing is the one stage here with no latency requirement at all: a
        frame that becomes searchable half a second later is indistinguishable
        from one that does not.
        """
        with self._queue_lock:
            while not self._queue and not self._stop.is_set():
                self._queue_lock.wait(timeout=self.settings.batch_wait_sec)
            while (
                self._queue
                and len(self._queue) < self.settings.batch_size
                and not self._stop.is_set()
            ):
                remaining = (
                    self._queue[0].queued_at + self.settings.batch_wait_sec
                    - time.monotonic()
                )
                if remaining <= 0:
                    break
                self._queue_lock.wait(timeout=remaining)
            batch = self._queue[: self.settings.batch_size]
            del self._queue[: len(batch)]
        return batch

    def _loop(self) -> None:
        while True:
            batch = self._take_batch()
            if not batch:
                # Only ever empty because `stop()` woke the wait above; a
                # spurious wakeup leaves the queue empty too, hence the check
                # rather than an unconditional return.
                if self._stop.is_set():
                    return
                continue
            try:
                self._process(batch)
            except Exception as exc:  # noqa: BLE001 - one bad batch must not end indexing
                self.failures += len(batch)
                self.last_error = f"{type(exc).__name__}: {exc}"[:300]
                self.log(f"index: batch failed — {self.last_error}")

    def _process(self, batch: list[PendingFrame]) -> None:
        assert self._embedder is not None and self._store is not None
        started = time.monotonic()
        vectors = self._embedder.embed_images([pending.image for pending in batch])
        self.embed_seconds += time.monotonic() - started
        self.batches += 1

        rows: list[IndexedFrame] = []
        for pending, vector in zip(batch, vectors):
            if pending.kind == "event" and pending.snapshot_path:
                # Reuse the evidence snapshot: it is already on disk, already
                # served by /snapshots, and already the picture of this moment
                # the operator saw in the alert feed.
                thumb_path, thumb_kind = pending.snapshot_path, "snapshot"
                height, width = pending.image.shape[:2]
            else:
                # Millisecond epoch: unique per camera at any sane sample rate,
                # sorts chronologically in `ls`, and is the timestamp itself
                # rather than a counter that a restart would reset.
                relative = f"{pending.camera_id}/{int(pending.ts * 1000)}.jpg"
                width, height = save_thumbnail(
                    pending.image, self.config.index_dir / relative,
                    width=self.settings.thumb_width, quality=self.settings.thumb_quality,
                )
                thumb_path, thumb_kind = relative, "index"
            rows.append(IndexedFrame(
                camera_id=pending.camera_id, ts=pending.ts, kind=pending.kind,
                embedding=vector, thumb_path=thumb_path, thumb_kind=thumb_kind,
                event_id=pending.event_id, width=width, height=height,
            ))
        written = self._store.add(rows)
        self.indexed += written
        self.events_indexed += sum(1 for row in rows if row.kind == "event")

    # --- reporting --------------------------------------------------------

    @property
    def queued(self) -> int:
        with self._queue_lock:
            return len(self._queue)

    def stats(self) -> dict[str, Any]:
        """What the engine publishes and ``/api/health`` shows."""
        described = self._embedder.describe() if self._embedder is not None else {}
        return {
            "enabled": self.settings.enabled,
            "ready": self.ready,
            "reason": self.reason,
            "model": described.get("model", self.settings.model),
            "dim": described.get("dim"),
            "device": described.get("device"),
            "sample_interval_sec": self.settings.sample_interval_sec,
            "indexed": self.indexed,
            "events_indexed": self.events_indexed,
            "dropped": self.dropped,
            "failures": self.failures,
            "batches": self.batches,
            "queued": self.queued,
            "queue_depth": self.settings.queue_depth,
            # Seconds of GPU spent per frame indexed, averaged over the run.
            # This is the number that answers "what does search cost the vision
            # engine", and it is measured rather than assumed.
            "ms_per_frame": (
                round(self.embed_seconds * 1000.0 / self.indexed, 2) if self.indexed else None
            ),
            "embed_sec_total": round(self.embed_seconds, 1),
            "last_error": self.last_error,
        }


__all__ = ["IndexRunner", "PendingFrame", "save_thumbnail"]
