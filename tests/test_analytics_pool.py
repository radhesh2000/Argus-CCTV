"""The one property the analytics pool must never break: per-camera order.

ByteTrack carries Kalman state, so a tracker fed frame 7 after frame 8 does not
produce a slightly-worse answer — it produces a wrong one, with people teleporting
and track ids churning, which is exactly the kind of bug that shows up as a bad
headcount three weeks after the change that caused it. Everything here runs
without a GPU and without a detector.
"""

from __future__ import annotations

import threading
import time

import pytest

from argus.vision.analytics import AnalyticsPool


class Recorder:
    """Records the order work arrived in, per camera, and can be made slow."""

    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.lock = threading.Lock()
        self.seen: dict[str, list[int]] = {}
        self.threads: dict[str, set[str]] = {}
        self.concurrent = 0
        self.peak_concurrent = 0

    def __call__(self, camera_id: str, index: int) -> None:
        with self.lock:
            self.concurrent += 1
            self.peak_concurrent = max(self.peak_concurrent, self.concurrent)
        if self.delay:
            time.sleep(self.delay)
        with self.lock:
            self.seen.setdefault(camera_id, []).append(index)
            self.threads.setdefault(camera_id, set()).add(threading.current_thread().name)
            self.concurrent -= 1


@pytest.fixture
def pool_factory():
    created: list[AnalyticsPool] = []

    def make(workers: int, handler, **kwargs) -> AnalyticsPool:
        pool = AnalyticsPool(workers, handler, **kwargs)
        created.append(pool)
        return pool.start()

    yield make
    for pool in created:
        pool.stop(timeout=5.0)


# --- ordering ---------------------------------------------------------------


def test_one_cameras_frames_are_handled_in_order(pool_factory):
    recorder = Recorder()
    # Depth above the submission count so nothing is dropped: this test is about
    # order, and back pressure has its own tests below.
    pool = pool_factory(4, recorder, queue_depth=500)
    for index in range(200):
        pool.submit("cam_a", "cam_a", index)

    assert pool.drain(timeout=10.0)
    assert recorder.seen["cam_a"] == list(range(200))


def test_every_camera_keeps_its_own_order_under_a_slow_handler(pool_factory):
    """The interesting case: several cameras interleaved across several
    workers, each handler slow enough that the queues really do build up."""
    recorder = Recorder(delay=0.002)
    pool = pool_factory(4, recorder, queue_depth=64)
    cameras = [f"cam_{i:02d}" for i in range(8)]
    for index in range(25):
        for camera in cameras:
            pool.submit(camera, camera, index)

    assert pool.drain(timeout=20.0)
    for camera in cameras:
        assert recorder.seen[camera] == list(range(25)), camera


def test_a_camera_is_always_handled_by_the_same_worker(pool_factory):
    """Ordering is not enforced by a lock, it is enforced by the routing: one
    camera, one worker, one FIFO. If that ever stops being true the guarantee
    is gone even though the frames still arrive."""
    recorder = Recorder(delay=0.001)
    pool = pool_factory(4, recorder, queue_depth=64)
    cameras = [f"cam_{i:02d}" for i in range(16)]
    for index in range(10):
        for camera in cameras:
            pool.submit(camera, camera, index)

    assert pool.drain(timeout=20.0)
    for camera in cameras:
        assert len(recorder.threads[camera]) == 1, f"{camera} moved between workers"


def test_the_shard_is_stable_across_pools_of_the_same_size():
    first = AnalyticsPool(4, lambda: None)
    second = AnalyticsPool(4, lambda: None)
    assert [first.shard_for(f"c{i}") for i in range(50)] == \
           [second.shard_for(f"c{i}") for i in range(50)]


def test_a_fleet_spreads_over_the_workers():
    pool = AnalyticsPool(4, lambda: None)
    counts = [0] * 4
    for index in range(1, 65):
        counts[pool.shard_for(f"bench_cam_{index:02d}")] += 1
    assert sum(counts) == 64
    assert min(counts) > 0, "a worker with no cameras is a wasted quarter of the pool"


# --- parallelism ------------------------------------------------------------


def test_different_cameras_run_at_the_same_time(pool_factory):
    """The point of the pool. Two cameras on two workers must overlap; if they
    serialised, the pool would be a queue with extra steps."""
    recorder = Recorder(delay=0.05)
    pool = pool_factory(4, recorder, queue_depth=8)
    # crc32 puts these four on four different workers; assert it rather than
    # assume it, so a routing change fails here rather than silently weakening
    # the test.
    cameras = [c for c in (f"cam_{i:02d}" for i in range(40))]
    by_shard: dict[int, str] = {}
    for camera in cameras:
        by_shard.setdefault(pool.shard_for(camera), camera)
    chosen = list(by_shard.values())
    assert len(chosen) == 4

    for camera in chosen:
        pool.submit(camera, camera, 0)
    assert pool.drain(timeout=10.0)
    assert recorder.peak_concurrent > 1


# --- back pressure ----------------------------------------------------------


def test_a_full_queue_drops_this_cameras_oldest_frame(pool_factory):
    """Falling behind must cost sampling rate, never liveness or memory — the
    same rule the decoder's slot applies one stage earlier. The *newest* frames
    are the ones kept, because they are the ones an operator is looking at."""
    gate = threading.Event()
    stuck = threading.Event()
    order: list[int] = []

    def handler(index: int) -> None:
        stuck.set()
        gate.wait(5.0)
        order.append(index)

    pool = pool_factory(1, handler, queue_depth=2)
    pool.submit("cam", 0)
    assert stuck.wait(5.0), "the worker never picked up the first frame"
    assert pool.submit("cam", 1) is True
    assert pool.submit("cam", 2) is True
    assert pool.submit("cam", 3) is False     # over depth: displaces frame 1
    assert pool.submit("cam", 4) is False     # displaces frame 2
    gate.set()

    assert pool.drain(timeout=10.0)
    assert order == [0, 3, 4]
    assert pool.stats()["dropped"] == 2


def test_one_backed_up_camera_does_not_drop_anothers_frames(pool_factory):
    gate = threading.Event()
    stuck = threading.Event()
    handled: list[str] = []

    def handler(camera_id: str) -> None:
        if camera_id == "slow":
            stuck.set()
            gate.wait(5.0)
        handled.append(camera_id)

    # Both cameras on one worker, so this is the worst case: a shared FIFO, a
    # shared thread, and only the per-camera accounting keeping them apart. The
    # depth is per camera, not per queue, which is the whole point.
    pool = AnalyticsPool(1, handler, queue_depth=3)
    pool.start()
    try:
        pool.submit("slow", "slow")
        assert stuck.wait(5.0), "the worker never picked up the first frame"
        for _ in range(7):                    # 3 fit, 4 evict this camera's oldest
            pool.submit("slow", "slow")
        for _ in range(3):
            pool.submit("quick", "quick")
        gate.set()
        assert pool.drain(timeout=10.0)
    finally:
        pool.stop(timeout=5.0)

    assert handled.count("quick") == 3, "a slow camera evicted a healthy one's frames"
    assert handled.count("slow") == 4, "the slow camera should have shed its own backlog"
    assert pool.stats()["dropped"] == 4


# --- robustness -------------------------------------------------------------


def test_a_failing_frame_is_reported_and_the_worker_survives(pool_factory):
    errors: list[tuple[str, str]] = []
    done: list[int] = []

    def handler(index: int) -> None:
        if index == 1:
            raise ValueError("bad frame")
        done.append(index)

    pool = pool_factory(1, handler, queue_depth=8,
                        on_error=lambda cam, exc: errors.append((cam, str(exc))))
    for index in range(4):
        pool.submit("cam", index)

    assert pool.drain(timeout=10.0)
    assert done == [0, 2, 3]
    assert errors == [("cam", "bad frame")]
    assert pool.stats()["processed"] == 4


def test_stop_finishes_what_was_already_queued(pool_factory):
    recorder = Recorder(delay=0.005)
    pool = pool_factory(2, recorder, queue_depth=64)
    for index in range(20):
        pool.submit("cam_a", "cam_a", index)

    assert pool.drain(timeout=10.0)
    pool.stop(timeout=5.0)
    assert recorder.seen["cam_a"] == list(range(20))


def test_stats_account_for_every_submission(pool_factory):
    pool = pool_factory(2, lambda index: None, queue_depth=64)
    for index in range(30):
        pool.submit("cam_a", index)
    assert pool.drain(timeout=10.0)

    stats = pool.stats()
    assert stats["submitted"] == 30
    assert stats["processed"] + stats["dropped"] == 30
    assert stats["queued"] == 0
    assert stats["workers"] == 2


def test_a_pool_needs_a_worker():
    with pytest.raises(ValueError):
        AnalyticsPool(0, lambda: None)
