"""Post-detection analytics on their own threads, sharded by camera.

The Phase-1 capacity benchmark found the ceiling: not NVDEC (22% at 48 cameras),
not the GPU (52%), not the cores (3.8 of 20), but the single Python thread that
runs the batched forward pass *and* everything after it. This module is the
"everything after it" half, moved off that thread — and, measured, the wrong
answer on this box. It ships disabled. Read the last part of this docstring
before turning it on.

**Ordering is the contract.** A camera is bound to exactly one worker by a
stable hash of its id, and each worker drains its own queue first-in-first-out.
So frames from one camera are always analysed in the order the detector produced
them — which matters because ByteTrack carries Kalman state, and a tracker fed
frame 7 after frame 8 does not produce a slightly-worse answer, it produces a
wrong one. Different cameras run on different workers and proceed in parallel.

**Bounded, and it drops the oldest.** Each camera may have at most
``queue_depth`` frames waiting; submitting past that discards that camera's
*oldest* pending frame, never a newer one and never another camera's. This is
the rule the decoder already applies to its slot, for the same reason: falling
behind must cost sampling rate, not liveness, and never memory.

**This pool ships disabled, and the reason is a measurement that contradicts
the one it was built on.** Read this before turning it on.

Per-frame analytics on the GB10 is 0.545 ms, of which ByteTrack is 0.485 —
pure Python and numpy, holding the GIL throughout. On its own it does not
thread at all: 1 884 frames/s on one thread, 577 on two, 626 on twelve. The
case for a pool was never parallel analytics; it was *overlap with detection*,
because ``PersonDetector.detect`` releases the GIL for about 26% of its 1.55 ms.
In a two-thread microbenchmark that works exactly as designed::

    detection + analytics interleaved, 1 thread    467 f/s
    detection + 1 analytics worker                 593 f/s   (1.27x)
    detection + 2 analytics workers                545 f/s
    detection + 4 analytics workers                424 f/s

In the **running engine** it goes the other way. 64 cameras, 150 s, everything
else identical::

    analytics_workers 0    4.78 fps      analytics_workers 2    4.34 fps
    analytics_workers 1    4.37 fps

The microbenchmark had two threads. ``argus run`` at 64 cameras has about 130:
a decoder thread and a stderr drain per camera, ring recorders, the notifier,
retention, the report loop. Those threads already occupy every GIL gap the
forward pass opens, so the analytics worker does not fill idle time — it adds
one more contender, plus a condition-variable handoff per frame at 320 frames
a second. There is no idle GIL left to sell.

So the pool is kept, tested and **off**. It is where this work belongs the
moment the tracker stops holding the GIL — free-threaded CPython, or a native
tracker — and turning it on is one config key. Re-run the A/B before you do.
"""

from __future__ import annotations

import threading
import time
import zlib
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(slots=True)
class _Shard:
    """One worker's mailbox: a FIFO of work plus the per-camera depth counts."""

    queue: deque[tuple[str, tuple[Any, ...]]] = field(default_factory=deque)
    pending: dict[str, int] = field(default_factory=dict)


class AnalyticsPool:
    """Run ``handler(*args)`` off the caller's thread, in per-camera order.

    ``handler`` is invoked as ``handler(*args)`` with whatever :meth:`submit`
    was given after the camera id. Exceptions are handed to ``on_error`` and
    swallowed: one bad frame must not take a worker, and a dead worker would
    silently stop a slice of the fleet.
    """

    def __init__(
        self,
        workers: int,
        handler: Callable[..., None],
        *,
        queue_depth: int = 3,
        on_error: Callable[[str, BaseException], None] | None = None,
        name: str = "analytics",
    ):
        if workers < 1:
            raise ValueError("AnalyticsPool needs at least one worker")
        self.workers = workers
        self.handler = handler
        self.queue_depth = max(1, queue_depth)
        self.on_error = on_error
        self.name = name

        self.submitted = 0
        self.processed = 0
        self.dropped = 0

        self._shards = [_Shard() for _ in range(workers)]
        # One condition for everything: workers wait for their own shard to have
        # work, drainers wait for the whole pool to be quiet. Both predicates are
        # cheap and `notify_all` keeps the two kinds of waiter from stealing each
        # other's wakeups, which a plain `notify` would.
        self._condition = threading.Condition()
        self._busy = 0
        self._stop = False
        self._threads: list[threading.Thread] = []

    # --- routing ----------------------------------------------------------

    def shard_for(self, camera_id: str) -> int:
        """Which worker owns this camera. Stable, and deliberately not ``hash``:
        str hashing is salted per process, so the same fleet would shard
        differently on every restart and a capacity run would not be
        reproducible."""
        return zlib.crc32(camera_id.encode("utf-8")) % self.workers

    # --- lifecycle --------------------------------------------------------

    def start(self) -> "AnalyticsPool":
        if self._threads:
            return self
        for index in range(self.workers):
            thread = threading.Thread(
                target=self._run, args=(index,), daemon=True, name=f"{self.name}-{index}"
            )
            thread.start()
            self._threads.append(thread)
        return self

    def stop(self, timeout: float = 5.0) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        for thread in self._threads:
            thread.join(timeout=timeout)
        self._threads = []

    # --- submission -------------------------------------------------------

    def submit(self, camera_id: str, *args: Any) -> bool:
        """Queue work for ``camera_id``. False when it displaced an older frame.

        Never blocks: the vision loop's next batch is more valuable than this
        camera's previous frame, so the queue is trimmed rather than waited on.
        """
        shard = self._shards[self.shard_for(camera_id)]
        with self._condition:
            self.submitted += 1
            accepted = True
            if shard.pending.get(camera_id, 0) >= self.queue_depth:
                for position, (queued_id, _) in enumerate(shard.queue):
                    if queued_id == camera_id:
                        del shard.queue[position]
                        break
                shard.pending[camera_id] -= 1
                self.dropped += 1
                accepted = False
            shard.queue.append((camera_id, args))
            shard.pending[camera_id] = shard.pending.get(camera_id, 0) + 1
            self._condition.notify_all()
        return accepted

    # --- worker -----------------------------------------------------------

    def _run(self, index: int) -> None:
        shard = self._shards[index]
        while True:
            with self._condition:
                while not shard.queue and not self._stop:
                    self._condition.wait(0.5)
                if not shard.queue:
                    return                      # stopping, and nothing left to do
                camera_id, args = shard.queue.popleft()
                remaining = shard.pending.get(camera_id, 1) - 1
                if remaining > 0:
                    shard.pending[camera_id] = remaining
                else:
                    shard.pending.pop(camera_id, None)
                self._busy += 1
            try:
                self.handler(*args)
            except BaseException as exc:  # noqa: BLE001 - a worker must not die
                if self.on_error is not None:
                    try:
                        self.on_error(camera_id, exc)
                    except Exception:  # noqa: BLE001 - reporting must not kill it either
                        pass
            finally:
                with self._condition:
                    self.processed += 1
                    self._busy -= 1
                    self._condition.notify_all()

    def _queued(self) -> int:
        return sum(len(shard.queue) for shard in self._shards)

    # --- observation ------------------------------------------------------

    def depth(self) -> int:
        with self._condition:
            return self._queued()

    def drain(self, timeout: float = 5.0) -> bool:
        """Block until every submitted frame has been handled.

        For shutdown (an event whose row is half-written is worse than a slow
        stop) and for tests, which is the only place ordering can be asserted.
        """
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._busy or self._queued():
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._threads:
                    return False
                self._condition.wait(min(0.1, remaining))
            return True

    def stats(self) -> dict[str, Any]:
        with self._condition:
            return {
                "workers": self.workers,
                "queue_depth": self.queue_depth,
                "submitted": self.submitted,
                "processed": self.processed,
                "dropped": self.dropped,
                "queued": self._queued(),
            }


__all__ = ["AnalyticsPool"]
