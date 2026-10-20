"""The semantic index: embedder, vector store, writer, backfill, retention.

Everything here runs on :class:`FakeEmbedder` — no CUDA, no checkpoint, no
download — because what these tests are about is the plumbing: does the
sampling clock hold, does a full queue drop the right frame, does a kNN with a
camera filter return that camera, does retention take the index with it when it
deletes the evidence. Retrieval *quality* is not a unit test and is not
pretended to be one; it is the live verification in docs/PHASE3_NOTES.md.

The fake is not noise, though — see its docstring. An image painted for a
phrase embeds to exactly that phrase's vector, so the ranking assertions below
are real cosine comparisons.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from argus.api.app import create_app
from argus.config import ArgusConfig
from argus.db import Camera, Database, Event, Policy
from argus.index.backfill import backfill_events
from argus.index.embedder import QUERY_TEMPLATE, FakeEmbedder
from argus.index.runner import IndexRunner, save_thumbnail
from argus.index.store import IndexedFrame, IndexError_, VectorStore
from argus.media.annotate import save_snapshot


class FakeFrame:
    """What the decoder hands the engine, minus the decoder."""

    def __init__(self, camera_id: str, ts: float, image: np.ndarray | None = None):
        self.camera_id = camera_id
        self.ts = ts
        self.index = 0
        self.image = np.zeros((32, 32, 3), dtype=np.uint8) if image is None else image


@pytest.fixture
def index_config(config: ArgusConfig) -> ArgusConfig:
    config.index.model = "fake"
    config.index.sample_interval_sec = 10.0
    config.index.batch_size = 4
    config.index.batch_wait_sec = 0.05
    return config


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def store(index_config: ArgusConfig, embedder: FakeEmbedder) -> VectorStore:
    store = VectorStore(index_config.index_db_path, dim=embedder.dim, model=embedder.name)
    store.migrate()
    return store


def vector(embedder: FakeEmbedder, text: str) -> np.ndarray:
    return embedder.embed_texts([text])[0]


# --------------------------------------------------------------------------
# the embedder contract
# --------------------------------------------------------------------------


def test_both_towers_produce_unit_vectors_of_the_declared_width(embedder):
    images = embedder.embed_images([np.zeros((8, 8, 3), np.uint8), np.full((8, 8, 3), 200, np.uint8)])
    texts = embedder.embed_texts(["a forklift", "an empty aisle"])
    for block in (images, texts):
        assert block.shape[1] == embedder.dim
        assert np.allclose(np.linalg.norm(block, axis=1), 1.0, atol=1e-5)
    # Different inputs are different directions; the same input is the same one.
    assert not np.allclose(images[0], images[1])
    assert np.allclose(embedder.embed_texts(["a forklift"])[0], texts[0])


def test_an_empty_batch_is_an_empty_block_not_an_error(embedder):
    """The runner and the backfill both flush possibly-empty batches."""
    assert embedder.embed_images([]).shape == (0, embedder.dim)
    assert embedder.embed_texts([]).shape == (0, embedder.dim)


def test_a_painted_image_lands_on_its_phrase_through_a_jpeg_round_trip(embedder, tmp_path):
    """The property every ranking test below rests on, including after the
    JPEG write-and-read that backfill puts a frame through."""
    from argus.index.backfill import load_bgr

    painted = FakeEmbedder.paint("person near the conveyor", shape=(64, 64))
    path = tmp_path / "painted.jpg"
    save_snapshot(painted, path, quality=90)

    direct = embedder.embed_images([painted])[0]
    round_tripped = embedder.embed_images([load_bgr(path)])[0]
    phrase = vector(embedder, "person near the conveyor")
    assert float(direct @ phrase) == pytest.approx(1.0, abs=1e-5)
    assert float(round_tripped @ phrase) == pytest.approx(1.0, abs=1e-5)


# --------------------------------------------------------------------------
# the store
# --------------------------------------------------------------------------


def test_search_ranks_by_similarity_and_reports_it_as_a_score(store, embedder):
    for name, camera in (("conveyor", "line_1"), ("ocean", "yard"), ("forklift", "line_1")):
        store.add([IndexedFrame(
            camera_id=camera, ts=1000.0, kind="sample",
            embedding=embedder.embed_images([FakeEmbedder.paint(name)])[0],
            thumb_path=f"{camera}/{name}.jpg",
        )])

    rows = store.search(vector(embedder, "conveyor"), limit=3)
    assert [row["thumb_path"] for row in rows][0].endswith("conveyor.jpg")
    # vec0 reports cosine *distance*; the store hands the caller similarity,
    # because that is the number the confidence bar draws.
    assert rows[0]["score"] == pytest.approx(1.0, abs=1e-4)
    assert rows[-1]["score"] < 0.9


def test_camera_and_time_filters_narrow_the_scan_not_the_result_page(store, embedder):
    """A filtered search must return k *matching* rows, not k rows of which
    some match — the difference between pushing the filter into the kNN and
    applying it afterwards."""
    query = FakeEmbedder.paint("conveyor")
    for index in range(6):
        store.add([IndexedFrame(
            camera_id="line_1" if index % 2 else "yard",
            ts=1000.0 + index * 100,
            kind="sample",
            embedding=embedder.embed_images([query if index == 5 else FakeEmbedder.paint(str(index))])[0],
            thumb_path=f"f{index}.jpg",
        )])

    only_yard = store.search(vector(embedder, "conveyor"), limit=3, camera_id="yard")
    assert {row["camera_id"] for row in only_yard} == {"yard"}
    assert len(only_yard) == 3

    windowed = store.search(vector(embedder, "conveyor"), limit=10,
                            since_ts=1300.0, until_ts=1450.0)
    # A set, because the cut is by relevance and not by time — the caller
    # sorts chronologically after choosing *which* frames are worth showing.
    assert {row["ts"] for row in windowed} == {1300.0, 1400.0}

    # The best match is on line_1 at 1500 and is excluded by both filters above.
    best = store.search(vector(embedder, "conveyor"), limit=1)[0]
    assert best["camera_id"] == "line_1" and best["ts"] == 1500.0


def test_alerts_only_returns_k_alerts_and_not_the_alerts_among_the_best_k(store, embedder):
    """The console's "Alerts only" checkbox, and the reason `kind` is a vec0
    metadata column rather than a filter on the join.

    An index is mostly periodic samples — one per camera every 10 s against a
    few events an hour — so a `kind` applied *after* the kNN returns whatever
    handful of events happened to make the overall top k. The operator sees
    three tiles, concludes there were three matching alerts, and is wrong.
    """
    for index in range(12):
        # Every sample is a better match for "conveyor" than every event, so a
        # post-filter would return nothing at all here.
        store.add([IndexedFrame(
            camera_id="line_1", ts=1000.0 + index,
            kind="sample" if index % 3 else "event",
            event_id=None if index % 3 else index,
            embedding=embedder.embed_images([
                FakeEmbedder.paint("conveyor" if index % 3 else f"event {index}")
            ])[0],
            thumb_path=f"line_1/{index}.jpg",
        )])

    alerts = store.search(vector(embedder, "conveyor"), limit=4, kind="event")
    assert len(alerts) == 4
    assert {row["kind"] for row in alerts} == {"event"}
    # And the unfiltered search really is dominated by samples, which is what
    # makes the assertion above worth making.
    assert all(row["kind"] == "sample"
               for row in store.search(vector(embedder, "conveyor"), limit=4))


def test_reindexing_an_event_replaces_its_row_rather_than_duplicating_it(store, embedder):
    for suffix in ("first", "second"):
        store.add([IndexedFrame(
            camera_id="line_1", ts=1000.0, kind="event", event_id=7,
            embedding=embedder.embed_images([FakeEmbedder.paint(suffix)])[0],
            thumb_path=f"line_1/{suffix}.jpg", thumb_kind="snapshot",
        )])
    assert store.stats()["frames"] == 1
    assert store.search(vector(embedder, "second"), limit=1)[0]["thumb_path"].endswith("second.jpg")


def test_a_store_written_by_another_model_is_refused_with_the_command_that_fixes_it(index_config):
    VectorStore(index_config.index_db_path, dim=64, model="model-a").migrate()
    with pytest.raises(IndexError_) as raised:
        VectorStore(index_config.index_db_path, dim=64, model="model-b").migrate()
    assert "backfill --reset" in str(raised.value)
    # Same model, different width — also a refusal, and for the same reason.
    with pytest.raises(IndexError_):
        VectorStore(index_config.index_db_path, dim=128, model="model-a").migrate()


def test_reset_empties_both_tables(store, embedder):
    store.add([IndexedFrame(camera_id="c", ts=1.0, kind="sample",
                            embedding=vector(embedder, "x"), thumb_path="a.jpg")])
    store.reset()
    assert store.stats()["frames"] == 0
    assert store.search(vector(embedder, "x"), limit=5) == []


def test_a_mismatched_query_width_is_a_value_error_not_a_wrong_answer(store, embedder):
    with pytest.raises(ValueError):
        store.search(np.zeros(embedder.dim + 1, dtype=np.float32))


def test_deleting_a_row_removes_its_vector_too(store, embedder):
    """vec0 has no cascade. A frames row deleted without its vector would stay
    searchable forever and resolve to nothing."""
    store.add([IndexedFrame(camera_id="c", ts=100.0, kind="sample",
                            embedding=vector(embedder, "x"), thumb_path="a.jpg")])
    store.prune_older_than(200.0)
    assert store.stats()["frames"] == 0
    assert store.search(vector(embedder, "x"), limit=5) == []


def test_orphan_prunes_cover_events_cameras_and_paths(store, embedder):
    rows = [
        IndexedFrame(camera_id="live", ts=100.0, kind="event", event_id=1,
                     embedding=vector(embedder, "a"), thumb_path="live/1.jpg",
                     thumb_kind="snapshot"),
        IndexedFrame(camera_id="live", ts=200.0, kind="event", event_id=2,
                     embedding=vector(embedder, "b"), thumb_path="live/2.jpg",
                     thumb_kind="snapshot"),
        IndexedFrame(camera_id="gone", ts=300.0, kind="sample",
                     embedding=vector(embedder, "c"), thumb_path="gone/3.jpg"),
        IndexedFrame(camera_id="live", ts=400.0, kind="sample",
                     embedding=vector(embedder, "d"), thumb_path="live/4.jpg"),
    ]
    store.add(rows)

    # Event 2 was purged by the 180-day window; its index row goes with it.
    assert [r["event_id"] for r in store.prune_missing_events([1])] == [2]
    # The camera was deleted; so does everything it recorded.
    assert store.prune_cameras(["live"]) == 1
    # Retention deleted the thumbnail file; the row that renders it goes.
    assert store.prune_paths("index", ["live/4.jpg"]) == 1
    assert store.stats()["frames"] == 1


# --------------------------------------------------------------------------
# the writer
# --------------------------------------------------------------------------


def make_runner(config: ArgusConfig, embedder: FakeEmbedder, store: VectorStore) -> IndexRunner:
    runner = IndexRunner(config, embedder=embedder, store=store, log=lambda *_: None)
    runner.start()
    return runner


def test_sampling_honours_the_interval_per_camera(index_config, embedder, store):
    runner = make_runner(index_config, embedder, store)
    try:
        assert runner.offer(FakeFrame("a", ts=1000.0)) is True
        assert runner.offer(FakeFrame("a", ts=1005.0)) is False   # inside the window
        assert runner.offer(FakeFrame("b", ts=1005.0)) is True    # a different camera
        assert runner.offer(FakeFrame("a", ts=1010.0)) is True    # the window has passed
    finally:
        runner.stop()
    assert store.stats()["samples"] == 3


def test_a_full_queue_drops_the_oldest_sample_and_never_an_event(index_config, embedder, store):
    """Falling behind must cost index density, not the ability to find an alert
    — so the victim is always a periodic sample, even when an event is older."""
    index_config.index.queue_depth = 3
    runner = IndexRunner(index_config, embedder=embedder, store=store, log=lambda *_: None)
    runner.ready = True  # queue only; no worker to drain it

    runner.offer_event("a", 1, np.zeros((8, 8, 3), np.uint8), 1000.0, "a/1.jpg")
    for ts in (1010.0, 1020.0):
        runner.offer(FakeFrame("a", ts=ts))
    assert runner.queued == 3

    assert runner.offer(FakeFrame("a", ts=1030.0)) is True
    assert runner.dropped == 1
    held = runner._queue  # noqa: SLF001 - asserting the eviction policy itself
    assert [item.kind for item in held] == ["event", "sample", "sample"]
    assert [item.ts for item in held] == [1000.0, 1020.0, 1030.0]


def test_an_event_frame_is_indexed_against_its_snapshot_and_bypasses_the_clock(
    index_config, embedder, store
):
    runner = make_runner(index_config, embedder, store)
    try:
        runner.offer(FakeFrame("a", ts=1000.0))
        # Two events a second apart: neither is refused by the sample interval.
        runner.offer_event("a", 11, FakeEmbedder.paint("forklift"), 1000.5, "a/zone_1000.jpg")
        runner.offer_event("a", 12, FakeEmbedder.paint("spill"), 1001.5, "a/zone_1001.jpg")
    finally:
        runner.stop()

    rows = store.search(vector(embedder, "forklift"), limit=1)
    assert rows[0]["event_id"] == 11
    # The thumbnail is the evidence snapshot that already exists, served from
    # /snapshots — no second image is written for an event.
    assert rows[0]["thumb_kind"] == "snapshot"
    assert rows[0]["thumb_path"] == "a/zone_1000.jpg"
    assert not list(index_config.index_dir.rglob("*.jpg")) or all(
        "zone_" not in str(p) for p in index_config.index_dir.rglob("*.jpg")
    )
    assert store.stats()["events"] == 2


def test_a_sampled_frame_writes_its_own_small_thumbnail(index_config, embedder, store):
    runner = make_runner(index_config, embedder, store)
    try:
        runner.offer(FakeFrame("cam_a", ts=1234.5,
                               image=np.full((544, 960, 3), 90, dtype=np.uint8)))
    finally:
        runner.stop()

    written = list((index_config.index_dir / "cam_a").glob("*.jpg"))
    assert [path.name for path in written] == ["1234500.jpg"]
    row = store.search(vector(embedder, "anything"), limit=1)[0]
    assert row["thumb_kind"] == "index"
    # The row records the *source* frame's size, not the thumbnail's: the UI
    # places boxes in the original coordinate space.
    assert (row["width"], row["height"]) == (960, 544)


def test_a_runner_with_no_checkpoint_degrades_instead_of_raising(config):
    """The same contract the VLM server has: a missing model is a degraded
    appliance, never a dead one."""
    config.index.model = "not/a-real-checkpoint"
    runner = IndexRunner(config, log=lambda *_: None)
    assert runner.start() is False
    assert runner.ready is False
    assert "fetch_siglip" in (runner.reason or "")
    # And it stays inert rather than blowing up on the decode path.
    assert runner.offer(FakeFrame("a", ts=1.0)) is False
    assert runner.offer_event("a", 1, np.zeros((4, 4, 3), np.uint8), 1.0, None) is False


def test_stop_drains_what_is_queued_rather_than_dropping_it(index_config, embedder, store):
    """The frames in the queue at shutdown are the newest the box saw, which is
    exactly the window somebody searches after an incident."""
    index_config.index.batch_wait_sec = 5.0  # long enough that only the drain can finish it
    runner = IndexRunner(index_config, embedder=embedder, store=store, log=lambda *_: None)
    runner.ready = True
    for i in range(3):
        runner.offer(FakeFrame("a", ts=1000.0 + i * 20))
    runner._thread = None  # noqa: SLF001
    runner.start()
    runner.stop()
    assert store.stats()["frames"] == 3


def test_the_worker_waits_for_company_before_spending_a_forward_pass(
    index_config, embedder, store
):
    """Frames that arrive within ``batch_wait_sec`` of each other go in one pass.

    Measured, and it is the difference between 3.35 and 10.9 ms per frame on
    this box: a worker that drained the queue on arrival ran at batch 1 forever
    on a lightly loaded appliance, because the next frame is always further
    away than one forward pass. Indexing is the one stage with no latency
    requirement, so it is allowed to wait.
    """
    index_config.index.batch_wait_sec = 2.0
    runner = make_runner(index_config, embedder, store)
    try:
        for i in range(index_config.index.batch_size):
            runner.offer(FakeFrame("a", ts=1000.0 + i * 20))
    finally:
        runner.stop()
    assert runner.indexed == index_config.index.batch_size
    assert runner.batches == 1


def test_a_lone_frame_is_not_held_for_ever_waiting_for_a_batch(index_config, embedder, store):
    """The other half of the rule: the wait is bounded, so a two-camera site at
    a 10 s interval does not hold its first sample until the eighth arrives."""
    index_config.index.batch_wait_sec = 0.05
    runner = make_runner(index_config, embedder, store)
    try:
        runner.offer(FakeFrame("a", ts=1000.0))
        deadline = time.monotonic() + 5.0
        while runner.indexed == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert runner.indexed == 1, "a single frame was never embedded"
    finally:
        runner.stop()


def test_stats_report_the_cost_per_frame(index_config, embedder, store):
    runner = make_runner(index_config, embedder, store)
    try:
        for i in range(4):
            runner.offer(FakeFrame("a", ts=1000.0 + i * 20))
    finally:
        runner.stop()
    stats = runner.stats()
    assert stats["indexed"] == 4 and stats["ready"] is True
    assert stats["ms_per_frame"] is not None and stats["ms_per_frame"] >= 0
    assert stats["queued"] == 0


def test_a_thumbnail_is_downscaled_and_small(tmp_path):
    image = np.random.default_rng(0).integers(0, 255, (544, 960, 3), dtype=np.uint8)
    size = save_thumbnail(image, tmp_path / "t.jpg", width=320, quality=70)
    assert size == (960, 544)  # the *source* size is what is returned
    from PIL import Image

    with Image.open(tmp_path / "t.jpg") as written:
        assert written.size == (320, 181)


# --------------------------------------------------------------------------
# backfill
# --------------------------------------------------------------------------


def seed_event(config: ArgusConfig, database: Database, phrase: str, ts: float) -> int:
    database.upsert_camera(Camera(id="line_1", name="Line 1", source="/dev/null"))
    relative = f"line_1/{phrase}.jpg"
    save_snapshot(FakeEmbedder.paint(phrase, shape=(64, 64)),
                  config.snapshots_dir / relative, quality=90)
    return database.insert_event(Event(
        camera_id="line_1", detector_name="manpower_headcount", detector_type="detector",
        severity="high", description=phrase, ts=ts, snapshot_path=relative, stage="reviewed",
    ))


def test_backfill_indexes_existing_evidence_and_resumes(index_config, database, embedder, store):
    ids = [seed_event(index_config, database, phrase, 1000.0 + i)
           for i, phrase in enumerate(("conveyor", "forklift", "spill"))]

    first = backfill_events(index_config, database, embedder=embedder, store=store,
                            log=lambda *_: None)
    assert (first.scanned, first.indexed, first.failed) == (3, 3, 0)
    assert store.indexed_event_ids() == set(ids)
    assert store.search(vector(embedder, "forklift"), limit=1)[0]["event_id"] == ids[1]

    # Run it again: nothing is re-embedded, which is what makes an interrupted
    # backfill resumable rather than a restart.
    again = backfill_events(index_config, database, embedder=embedder, store=store,
                            log=lambda *_: None)
    assert (again.indexed, again.skipped_indexed) == (0, 3)


def test_backfill_counts_an_event_whose_snapshot_retention_deleted(
    index_config, database, embedder, store
):
    """Events live 180 days and snapshots 30, so this is the normal state of an
    old event — not an error, and not something to fail the run over."""
    event_id = seed_event(index_config, database, "conveyor", 1000.0)
    (index_config.snapshots_dir / "line_1" / "conveyor.jpg").unlink()
    database.insert_event(Event(camera_id="line_1", detector_name="d", detector_type="detector",
                                severity="low", description="no snapshot at all", ts=1001.0))

    result = backfill_events(index_config, database, embedder=embedder, store=store,
                             log=lambda *_: None)
    assert (result.scanned, result.indexed, result.missing_snapshot) == (2, 0, 2)
    assert result.failed == 0
    assert event_id not in store.indexed_event_ids()


def test_backfill_crops_the_headline_off_before_embedding(index_config, database, embedder):
    """SigLIP reads the banner, so backfill has to take it off.

    Measured on the demo footage, where both cameras play the same file and the
    only systematic difference between their snapshots is the zone name burned
    across the top: "person near the conveyor" scored +0.031 on the camera
    whose banner says "Conveyor side" — a quarter of the whole signal, bought
    with a caption Argus drew itself. See argus/index/backfill.py.

    The fake embedder makes that testable exactly: a painted frame embeds to
    its phrase's vector, and a banner drawn over it moves the mean pixel and so
    the vector. Cropping the banner back off restores it.
    """
    from argus.index.backfill import load_bgr
    from argus.media.annotate import banner_height, draw_banner, to_image

    painted = FakeEmbedder.paint("person near the conveyor", shape=(544, 960))
    image = to_image(painted)
    draw_banner(image, "Conveyor side: 3 people - limit 2")
    path = index_config.snapshots_dir / "line_1" / "annotated.jpg"
    save_snapshot(image, path, quality=90)

    phrase = vector(embedder, "person near the conveyor")
    with_banner = embedder.embed_images([load_bgr(path)])[0]
    cropped = embedder.embed_images([load_bgr(path, crop_banner=True)])[0]

    assert float(with_banner @ phrase) < 0.5, "the banner should have moved the vector"
    assert float(cropped @ phrase) == pytest.approx(1.0, abs=1e-5)
    assert load_bgr(path, crop_banner=True).shape[0] == 544 - banner_height(960)


def test_a_snapshot_too_small_to_crop_is_left_whole(index_config):
    """The guard that keeps the crop from eating a thumbnail-sized frame — and
    the reason every other backfill test below still sees the image it wrote."""
    from argus.index.backfill import load_bgr

    path = index_config.snapshots_dir / "line_1" / "tiny.jpg"
    save_snapshot(FakeEmbedder.paint("conveyor", shape=(64, 64)), path, quality=90)
    assert load_bgr(path, crop_banner=True).shape[:2] == (64, 64)


def test_backfill_reset_rebuilds_from_scratch(index_config, database, embedder, store):
    seed_event(index_config, database, "conveyor", 1000.0)
    store.add([IndexedFrame(camera_id="ghost", ts=1.0, kind="sample",
                            embedding=vector(embedder, "stale"), thumb_path="ghost.jpg")])

    result = backfill_events(index_config, database, embedder=embedder, store=store,
                             reset=True, log=lambda *_: None)
    assert result.indexed == 1
    assert store.stats()["frames"] == 1  # the stale sample went with the reset


# --------------------------------------------------------------------------
# retention
# --------------------------------------------------------------------------


def test_retention_takes_the_index_with_the_evidence(index_config, database, embedder, store):
    """The whole reason the index is pruned in the same sweep: a hit that
    resolves to a deleted file renders as a broken tile, which is worse than
    not being found."""
    from argus.retention import Retention

    event_id = seed_event(index_config, database, "conveyor", time.time())
    backfill_events(index_config, database, embedder=embedder, store=store, log=lambda *_: None)
    store.add([IndexedFrame(camera_id="line_1", ts=time.time(), kind="sample",
                            embedding=vector(embedder, "sample"), thumb_path="line_1/old.jpg")])
    thumb = index_config.index_dir / "line_1" / "old.jpg"
    save_thumbnail(np.zeros((64, 64, 3), np.uint8), thumb, width=32, quality=70)
    # Age the thumbnail past the window; its vector must go with it.
    old = time.time() - index_config.retention.index_days * 86400 - 3600
    import os

    os.utime(thumb, (old, old))

    result = Retention(index_config, database, log=lambda *_: None).sweep()
    assert result.index_rows_deleted >= 1
    assert not thumb.exists()
    assert store.stats()["frames"] == 1  # the event row survives; the aged sample does not

    # Now purge the event itself and sweep again: its index row is an orphan.
    database.purge_older_than("events", "created_at", "2999-01-01")
    assert database.get_event(event_id) is None
    Retention(index_config, database, log=lambda *_: None).sweep()
    assert store.stats()["frames"] == 0


def test_a_sweep_survives_an_index_it_cannot_open(index_config, database):
    """Clips filling the disk is an outage; a broken index is a missing
    feature. The sweep must not stop for the second."""
    from argus.retention import Retention

    index_config.index_db_path.parent.mkdir(parents=True, exist_ok=True)
    index_config.index_db_path.write_bytes(b"this is not a database")
    result = Retention(index_config, database, log=lambda *_: None).sweep()
    assert result.errors == []  # the index failure is logged, not surfaced as a sweep error
    assert result.index_rows_deleted == 0


# --------------------------------------------------------------------------
# the API
# --------------------------------------------------------------------------


@pytest.fixture
def client(index_config: ArgusConfig, database: Database, sign_in) -> TestClient:
    """Signed in as ADMIN — every route needs a session since Phase 5, and
    these tests are about search, not about who may run one."""
    client = TestClient(create_app(index_config, database))
    sign_in(client, "ADMIN")
    return client


@pytest.fixture
def indexed(index_config: ArgusConfig, database: Database, embedder: FakeEmbedder,
            store: VectorStore) -> dict[str, float]:
    """Three cameras' worth of painted frames, at known timestamps."""
    database.upsert_camera(Camera(id="line_1", name="Packing Line 1", source="/dev/null"))
    database.upsert_camera(Camera(id="yard", name="Yard Camera", source="/dev/null"))
    now = time.time()
    for phrase, camera, offset in (
        ("person near conveyor", "line_1", -60),
        ("stacked crates", "line_1", -3600),
        ("ocean waves", "yard", -120),
    ):
        store.add([IndexedFrame(
            camera_id=camera, ts=now + offset, kind="sample",
            embedding=embedder.embed_images([FakeEmbedder.paint(QUERY_TEMPLATE.format(phrase))])[0],
            thumb_path=f"{camera}/{abs(offset)}.jpg", width=960, height=544,
        )])
    return {"now": now}


def test_search_returns_ranked_results_with_everything_a_tile_needs(client, indexed):
    payload = client.get("/api/search", params={"q": "person near conveyor"}).json()
    assert payload["query"] == "person near conveyor"
    assert payload["count"] >= 1
    best = payload["results"][0]
    assert best["score"] == pytest.approx(1.0, abs=1e-4)
    assert best["camera_id"] == "line_1"
    assert best["camera_name"] == "Packing Line 1"   # resolved, not left as an id
    assert best["thumb_url"].startswith("/index/")
    assert best["event"] is None                      # a sample, not an alert
    assert payload["index"]["frames"] == 3
    assert payload["took_ms"] >= 0


def test_search_filters_by_camera_and_window(client, indexed):
    now = indexed["now"]
    # min_score=-1 so the *filter* is what is under test rather than the floor:
    # the yard frames are unrelated to this query and score near zero.
    only_yard = client.get("/api/search", params={"q": "person near conveyor",
                                                  "camera_id": "yard",
                                                  "min_score": -1}).json()
    assert {r["camera_id"] for r in only_yard["results"]} == {"yard"}

    recent = client.get("/api/search", params={"q": "stacked crates",
                                               "since_ts": now - 300,
                                               "min_score": -1}).json()
    # The crates frame is an hour old and outside the window, so the best
    # remaining match is something else entirely — and says so with a low score.
    assert recent["count"] == 2
    assert all(r["ts"] >= now - 300 for r in recent["results"])
    assert not any(r["score"] > 0.9 for r in recent["results"])


def test_an_event_result_carries_the_alert_it_belongs_to(
    client, index_config, database, embedder, store
):
    event_id = seed_event(index_config, database, "conveyor", time.time())
    backfill_events(index_config, database, embedder=embedder, store=store, log=lambda *_: None)

    results = client.get("/api/search", params={"q": "conveyor", "min_score": -1}).json()["results"]
    hit = next(r for r in results if r["event"])
    assert hit["event"]["id"] == event_id
    assert hit["event"]["description"] == "conveyor"
    assert hit["event"]["severity"] == "high"
    # An event tile shows the annotated evidence, from the evidence mount.
    assert hit["thumb_url"] == "/snapshots/line_1/conveyor.jpg"


def test_the_floor_drops_noise_rather_than_ranking_it(client, indexed):
    everything = client.get("/api/search", params={"q": "nothing like this at all",
                                                   "min_score": -1}).json()
    filtered = client.get("/api/search", params={"q": "nothing like this at all",
                                                 "min_score": 0.9}).json()
    assert everything["count"] == 3
    assert filtered["count"] == 0
    assert filtered["min_score"] == 0.9


def test_an_empty_query_is_rejected(client, indexed):
    assert client.get("/api/search", params={"q": "   "}).status_code == 422
    assert client.get("/api/search").status_code == 422


def test_search_status_answers_without_loading_a_model(client, index_config, indexed):
    payload = client.get("/api/search/status").json()
    assert payload["enabled"] is True
    assert payload["model_ready"] is True
    assert payload["query_model_loaded"] is False   # nothing has searched yet
    assert payload["index"]["frames"] == 3
    assert payload["error"] is None


def test_a_missing_checkpoint_is_a_503_naming_the_fix(index_config, database, sign_in):
    index_config.index.model = "not/a-real-checkpoint"
    client = TestClient(create_app(index_config, database))
    sign_in(client, "ADMIN")
    response = client.get("/api/search", params={"q": "anything"})
    assert response.status_code == 503
    assert "fetch_siglip" in response.json()["detail"]


# --------------------------------------------------------------------------
# saved searches, and the promotion the design is about
# --------------------------------------------------------------------------


def test_saved_search_crud(client, index_config, database):
    database.upsert_camera(Camera(id="line_1", name="Line 1", source="/dev/null"))
    created = client.post("/api/searches", json={
        "query": "person on the dock without a hi-vis vest",
        "label": "Dock hi-vis", "camera_id": "line_1", "window_hours": 12,
    })
    assert created.status_code == 201
    search_id = created.json()["search"]["id"]

    listed = client.get("/api/searches").json()["searches"]
    assert [s["label"] for s in listed] == ["Dock hi-vis"]

    renamed = client.patch(f"/api/searches/{search_id}", json={"label": "Dock PPE"})
    assert renamed.json()["search"]["label"] == "Dock PPE"
    assert client.patch(f"/api/searches/{search_id}", json={"nonsense": 1}).status_code == 422
    assert client.delete(f"/api/searches/{search_id}").status_code == 200
    assert client.get("/api/searches").json()["searches"] == []
    assert client.delete(f"/api/searches/{search_id}").status_code == 404


def test_saving_the_same_query_twice_renames_it_rather_than_duplicating(client):
    first = client.post("/api/searches", json={"query": "forklift near the walkway"}).json()
    second = client.post("/api/searches", json={"query": "forklift near the walkway",
                                                "label": "Forklift proximity"}).json()
    assert first["search"]["id"] == second["search"]["id"]
    assert second["search"]["label"] == "Forklift proximity"
    # Absent a label, the sentence is the label — nobody names a search twice.
    assert first["search"]["label"] == "forklift near the walkway"


def test_running_a_saved_search_counts_the_run(client, indexed):
    search_id = client.post("/api/searches",
                            json={"query": "person near conveyor"}).json()["search"]["id"]
    client.get("/api/search", params={"q": "person near conveyor", "saved_id": search_id})
    client.get("/api/search", params={"q": "person near conveyor", "saved_id": search_id})
    saved = client.get("/api/searches").json()["searches"][0]
    assert saved["run_count"] == 2 and saved["last_run_at"]


def test_promoting_a_search_creates_a_draft_shadow_vlm_policy(client, database):
    search_id = client.post("/api/searches", json={
        "query": "a worker on a ladder with nobody holding it",
        "label": "Unspotted ladder",
    }).json()["search"]["id"]

    promoted = client.post(f"/api/searches/{search_id}/promote", json={"severity": "high"})
    assert promoted.status_code == 201
    policy = promoted.json()["policy"]

    assert policy["kind"] == "vlm"
    assert policy["detector_key"] == "vlm_prompt"
    assert policy["label"] == "Unspotted ladder"
    assert policy["severity"] == "high"
    # The gate the whole product rests on: a promoted search still starts in
    # draft/shadow and walks the pilot before it can mail anyone.
    assert (policy["lifecycle"], policy["stage"]) == ("draft", "shadow")
    assert policy["routed_to"] is None
    # The operator's own sentence survives verbatim inside the prompt — the
    # policy that fires has to be the search that was validated.
    assert "a worker on a ladder with nobody holding it" in policy["prompt"]
    assert "triggered=true" in policy["prompt"]
    # And the provenance is recorded on the search.
    assert promoted.json()["search"]["policy_id"] == policy["id"]
    assert database.get_policy(policy["id"]) is not None


def test_a_search_cannot_be_promoted_twice(client):
    search_id = client.post("/api/searches", json={"query": "door propped open"}).json()["search"]["id"]
    assert client.post(f"/api/searches/{search_id}/promote", json={}).status_code == 201
    conflict = client.post(f"/api/searches/{search_id}/promote", json={})
    assert conflict.status_code == 409
    assert "already promoted" in conflict.json()["detail"]


def test_promoting_a_search_whose_policy_was_deleted_works_again(client, database):
    search_id = client.post("/api/searches", json={"query": "spill on the floor"}).json()["search"]["id"]
    policy_id = client.post(f"/api/searches/{search_id}/promote", json={}).json()["policy"]["id"]
    database.delete_policy(policy_id)
    # The FK is ON DELETE SET NULL, so the search is free again — and an
    # operator who deleted a policy by mistake is not locked out of remaking it.
    assert client.post(f"/api/searches/{search_id}/promote", json={}).status_code == 201


def test_promotion_rejects_a_policy_id_that_is_taken(client, database):
    database.upsert_policy(Policy(id="taken", kind="vlm", detector_key="vlm_prompt",
                                  label="Taken", prompt="?"))
    search_id = client.post("/api/searches", json={"query": "anything"}).json()["search"]["id"]
    response = client.post(f"/api/searches/{search_id}/promote", json={"policy_id": "taken"})
    assert response.status_code == 409


def test_promoting_a_search_that_does_not_exist_is_a_404(client):
    assert client.post("/api/searches/999/promote", json={}).status_code == 404


# --------------------------------------------------------------------------
# health and config
# --------------------------------------------------------------------------


def test_health_reports_index_coverage_without_the_engine(client, indexed):
    index = client.get("/api/health").json()["index"]
    assert index["source"] == "store"       # nothing is publishing writer counters
    assert index["writer"] is None          # and null must not be rendered as 0
    assert index["coverage"]["frames"] == 3
    assert index["reason"] is None


def test_health_names_the_missing_piece(index_config, database, sign_in):
    index_config.index.model = "not/a-real-checkpoint"
    client = TestClient(create_app(index_config, database))
    sign_in(client, "ADMIN")
    index = client.get("/api/health").json()["index"]
    assert index["model_ready"] is False
    assert "fetch_siglip.sh" in index["reason"]


def test_disk_reporting_separates_index_thumbnails_from_evidence(client, index_config):
    save_thumbnail(np.zeros((64, 64, 3), np.uint8),
                   index_config.index_dir / "cam" / "1.jpg", width=32, quality=70)
    disk = client.get("/api/health").json()["disk"]
    assert "index" in disk["by_directory_mb"]
