"""Event flow, end to end, without a GPU.

The detector is faked so these tests exercise the part that decides whether a
plant manager gets woken up: tracking, zone occupancy, the sanctioned limit,
the smoothed zone state, the annotated snapshot, and the event row. The real
detector has its own smoke test in ``test_detector.py``.

**A frame is no longer an incident**, which is why almost nothing here can raise
an event by calling ``_process`` once. Since ``detection.continuity`` a zone has
to be *seen* over its limit for ``sustain_sec`` before a row is written — see
``_breach`` below, ``argus/vision/zone_state.py`` for the arithmetic and
``tests/test_zone_state.py`` for the arithmetic's own tests.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from argus.db import Camera, Policy, Zone
from argus.engine import Engine
from argus.ingest.decoder import Frame

pytest.importorskip("supervision")

FULL_FRAME = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
WIDTH, HEIGHT = 320, 180


class FakeDetector:
    """Returns the boxes it was told to, so the test controls the headcount."""

    def __init__(self, boxes: np.ndarray, confidence: float = 0.9):
        self.boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        self.confidence = confidence
        self.calls = 0
        self.batch_sizes: list[int] = []

    def detect(self, images):
        import supervision as sv

        self.calls += 1
        self.batch_sizes.append(len(images))
        return [
            sv.Detections(
                xyxy=self.boxes.copy(),
                confidence=np.full(len(self.boxes), self.confidence, dtype=np.float32),
                class_id=np.zeros(len(self.boxes), dtype=int),
            )
            for _ in images
        ]

    def describe(self):
        return {"model": "fake", "backend": "fake"}


def _people(count: int) -> np.ndarray:
    """`count` boxes standing on the bottom half of the frame, well separated."""
    return np.array(
        [[10 + i * 30, 40, 35 + i * 30, 150] for i in range(count)], dtype=np.float32
    )


def _frame(camera_id: str, ts: float, index: int = 1) -> Frame:
    image = np.full((HEIGHT, WIDTH, 3), 40, dtype=np.uint8)
    image[:, :, 1] = 90                       # something other than flat grey
    return Frame(camera_id=camera_id, index=index, ts=ts, image=image)


@pytest.fixture
def seeded(config, database, sample_video):
    database.upsert_camera(Camera(id="cam", name="Packing line", source=str(sample_video)))
    database.upsert_zone(Zone(id="floor", camera_id="cam", name="Packing floor",
                              polygon=FULL_FRAME, sanctioned_limit=2))
    database.upsert_policy(Policy(
        id="manpower_headcount", kind="detector", detector_key="manpower_headcount",
        label="Manpower", enabled=True, stage="reviewed", lifecycle="production",
        severity="medium", routed_to="safety_manager", cooldown_sec=60,
    ))
    return database


@pytest.fixture(autouse=True)
def _a_detection_is_a_person_at_once(config):
    """Report a track on its first frame, as the tracker did before 2026-09-19.

    The appliance now waits three frames (`detection.tracker.min_consecutive_frames`)
    before a track counts as a person, and tests/test_tracker.py owns that
    behaviour. The tests *here* are about what the engine does with a count —
    zones, the smoothing window, conditions, the pool — and they state their
    expectations as exact timestamps from the first frame. Leaving the delay
    in would make every one of those numbers the sum of two mechanisms.
    `test_a_flicker_never_reaches_the_headcount` below is the one test here that
    wants the shipped default, and takes it back.
    """
    config.detection.tracker.min_consecutive_frames = 1


def _engine(config, database, detector, **kwargs) -> Engine:
    return Engine(config, database, use_gateway=False, detector=detector, record=False, **kwargs)


def _runtime(engine: Engine):
    engine._refresh_policies()
    camera = engine.db.get_camera("cam")
    return engine._build_runtime(camera)


#: The frame interval every test here runs at, matching `detection.fps`.
STEP = 0.2
START = 1_000_000.0

#: The timestamp at which a continuously over-limit run starting at `START`
#: opens its condition: the first frame credits nothing (there is no previous
#: one to measure from), so `sustain_sec` of observed breach lands here.
OPENS_AT = START + 5.0


def _breach(engine, runtime, detections, *, start=START, seconds=8.0, step=STEP, camera="cam"):
    """Feed one continuous stretch of frames and return the *next* timestamp.

    Eight seconds by default — comfortably past the shipped 5 s ``sustain_sec``
    without being so long that a test which meant "just short of opening"
    cannot say so by passing a smaller number.
    """
    ticks = int(round(seconds / step)) + 1
    for tick in range(ticks):
        engine._process(runtime, _frame(camera, start + tick * step), detections)
    return start + ticks * step


# --- the decision path -----------------------------------------------------


def test_a_sustained_breach_writes_one_event_with_evidence(config, seeded):
    detector = FakeDetector(_people(4))
    engine = _engine(config, seeded, detector)
    runtime = _runtime(engine)

    _breach(engine, runtime, detector.detect([None])[0])

    events = seeded.recent_events()
    assert len(events) == 1
    event = events[0]
    assert event["camera_id"] == "cam"
    assert event["zone_id"] == "floor"
    assert event["policy_id"] == "manpower_headcount"
    assert event["detector_type"] == "detector"
    assert event["severity"] == "medium"
    assert event["stage"] == "reviewed"           # inherited from the policy
    assert event["routed_to"] == "safety_manager"
    assert "4 people" in event["description"] and "limit 2" in event["description"]
    # The frame that *confirmed* the breach, five seconds in — not the first
    # frame that happened to be over the limit.
    assert event["ts"] == OPENS_AT

    snapshot = config.snapshots_dir / event["snapshot_path"]
    assert snapshot.exists() and snapshot.stat().st_size > 0


def test_a_single_over_limit_frame_writes_nothing(config, seeded):
    """The other half of the same sentence, and the reason this appliance had
    112 151 headcount rows: one frame is not an incident. ~13% of the detector's
    boxes are false positives (measured 2026-09-19), so a per-frame comparison
    turns detector noise into somebody's pager."""
    detector = FakeDetector(_people(9))
    engine = _engine(config, seeded, detector)
    engine._process(_runtime(engine), _frame("cam", START), detector.detect([None])[0])
    assert seeded.recent_events() == []


def test_snapshot_shows_the_violation(config, seeded):
    """The snapshot is the evidence: it must be the frame, annotated — same
    size, and visibly changed by the banner and boxes."""
    from PIL import Image

    detector = FakeDetector(_people(4))
    engine = _engine(config, seeded, detector)
    runtime = _runtime(engine)
    frame = _frame("cam", START)
    _breach(engine, runtime, detector.detect([None])[0])

    event = seeded.recent_events()[0]
    with Image.open(config.snapshots_dir / event["snapshot_path"]) as image:
        assert image.size == (WIDTH, HEIGHT)
        pixels = np.asarray(image)
    # The banner strip across the top must differ from the untouched frame.
    assert not np.allclose(pixels[:6].mean(), frame.image[:6, :, ::-1].mean(), atol=5)


def test_event_meta_carries_the_count_and_the_boxes(config, seeded):
    import json

    detector = FakeDetector(_people(4))
    engine = _engine(config, seeded, detector)
    _breach(engine, _runtime(engine), detector.detect([None])[0])

    meta = json.loads(seeded.recent_events()[0]["meta"])
    assert meta["count"] == 4
    assert meta["limit"] == 2
    assert meta["excess"] == 2
    assert meta["zone_name"] == "Packing floor"
    assert len(meta["boxes"]) == 4
    assert all(0.0 <= value <= 1.0 for box in meta["boxes"] for value in box[:4])
    assert len(meta["track_ids"]) == 4
    # What the decision was actually made on, so a reviewer can tell the number
    # in the banner from the number of boxes in the picture.
    assert meta["frame_count"] == 4
    assert meta["peak"] == 4
    assert meta["smoothing"]["sustain_sec"] == config.detection.continuity.sustain_sec


def test_at_the_limit_is_not_over_it(config, seeded):
    detector = FakeDetector(_people(2))
    engine = _engine(config, seeded, detector)
    _breach(engine, _runtime(engine), detector.detect([None])[0], seconds=60.0)
    assert seeded.recent_events() == []


def test_people_outside_the_zone_do_not_count(config, seeded):
    # A strip down the left edge: of four people at x-centres 0.07/0.16/0.26/0.35
    # of the frame, only the first two stand in it — so the zone is at its limit
    # of 2, not over it, even though four people are visible.
    seeded.update_zone("floor", polygon=[[0.0, 0.0], [0.2, 0.0], [0.2, 1.0], [0.0, 1.0]])
    detector = FakeDetector(_people(4))
    engine = _engine(config, seeded, detector)
    runtime = _runtime(engine)
    engine._process(runtime, _frame("cam", 1_000_000.0), detector.detect([None])[0])

    assert seeded.recent_events() == []
    assert engine._maybe_flush_occupancy(force=True) == 1
    assert seeded.occupancy_series(camera_id="cam")[0]["count"] == 2


def test_five_minutes_of_one_breach_is_one_event(config, seeded):
    """The change, in the smallest sentence that can carry it.

    Measured on this box on 2026-09-19, ``manpower_headcount`` had written
    10 958 events in 24 hours and 112 151 in total, because a zone that stayed
    busy re-fired a new row every ``cooldown_sec``. Five minutes of one
    continuous over-crowd is one incident; the minutes are ``observations``.
    """
    detector = FakeDetector(_people(5))
    engine = _engine(config, seeded, detector)
    runtime = _runtime(engine)

    _breach(engine, runtime, detector.detect([None])[0], seconds=300.0)

    events = seeded.recent_events(limit=500)
    assert len(events) == 1
    event = events[0]
    assert event["condition_key"] == "det:manpower_headcount:cam:floor"
    assert event["condition_state"] == "open"
    # Still open five minutes in, and last seen at the newest frame rather than
    # at the one it opened on.
    assert event["closed_at"] is None
    # Within one update cadence of the newest frame — the row is rewritten every
    # `update_sec`, not five times a second to restate a number nobody changed.
    cadence = config.detection.continuity.update_sec
    assert event["ts"] < event["last_seen_ts"] <= 1_000_300.0
    assert 1_000_300.0 - event["last_seen_ts"] < cadence
    # One write when it opened at t+5 s, then one per cadence to t+300 s.
    assert event["observations"] == 1 + int((300.0 - 5.0) // cadence)
    assert "peak 5" in event["description"] and "limit 2" in event["description"]


def test_the_same_five_minutes_before_this_change_was_five_events(config, seeded):
    """The flag is what makes the new behaviour a decision rather than something
    that happened to a site on upgrade — and this is the number it turns off.
    One row per 60 s cooldown, for as long as the zone stays busy, which across
    eight cameras is the 11.5k rows a day this appliance was writing."""
    config.detection.continuity.enabled = False
    detector = FakeDetector(_people(5))
    engine = _engine(config, seeded, detector)
    runtime = _runtime(engine)

    _breach(engine, runtime, detector.detect([None])[0], seconds=300.0)

    # The first frame, then one per 60 s cooldown for five minutes.
    events = seeded.recent_events(limit=500)
    assert len(events) == 6
    assert all(event["condition_key"] is None for event in events)


def test_an_ongoing_condition_is_extended_in_place_not_re_raised(config, seeded):
    """One row, one snapshot, a `last_seen_ts` that advances — the shape the
    console already draws for VLM conditions as "ongoing since · N
    observations"."""
    detector = FakeDetector(_people(5))
    engine = _engine(config, seeded, detector)
    runtime = _runtime(engine)
    detections = detector.detect([None])[0]

    next_ts = _breach(engine, runtime, detections, seconds=10.0)
    opened = seeded.recent_events()[0]
    assert opened["observations"] == 1
    assert seeded.max_event_id() == opened["id"]

    _breach(engine, runtime, detections, start=next_ts, seconds=120.0)
    extended = seeded.get_event(opened["id"])
    assert seeded.max_event_id() == opened["id"], "the condition raised a second row"
    assert extended["observations"] > opened["observations"]
    assert extended["last_seen_ts"] > opened["last_seen_ts"]
    assert extended["condition_updated_at"] >= opened["condition_updated_at"]
    # The same JPEG, refreshed in place: one evidence file per condition, not
    # one per sighting.
    assert extended["snapshot_path"] == opened["snapshot_path"]
    assert (config.snapshots_dir / extended["snapshot_path"]).exists()


def test_a_condition_closes_when_the_zone_clears_and_says_when_it_ended(config, seeded):
    detector = FakeDetector(_people(5))
    engine = _engine(config, seeded, detector)
    runtime = _runtime(engine)

    cleared_at = _breach(engine, runtime, detector.detect([None])[0], seconds=30.0)
    event_id = seeded.recent_events()[0]["id"]

    empty = FakeDetector(np.empty((0, 4), dtype=np.float32)).detect([None])[0]
    _breach(engine, runtime, empty, start=cleared_at, seconds=20.0)

    settings = config.detection.continuity
    closed = seeded.get_event(event_id)
    assert closed["condition_state"] == "closed"
    assert closed["close_reason"] == "cleared"
    # Stamped when the *smoothed* count came down, which trails the zone
    # emptying by about half a window while the median follows it — and not by
    # `clear_sec`, which is how long we then waited before believing it.
    assert cleared_at <= closed["closed_at"] <= cleared_at + settings.window_sec
    assert closed["closed_at"] < cleared_at + settings.clear_sec
    assert seeded.max_event_id() == event_id, "closing raised a row"


def test_a_zone_wobbling_at_its_limit_is_one_condition(config, seeded):
    """The flicker case, through the engine rather than the state machine: a
    count crossing the limit every few seconds used to be an event every
    cooldown, for as long as the shift lasted."""
    engine = _engine(config, seeded, FakeDetector(_people(1)))
    runtime = _runtime(engine)
    busy = FakeDetector(_people(4)).detect([None])[0]
    quiet = FakeDetector(_people(2)).detect([None])[0]

    ts = _breach(engine, runtime, busy, seconds=10.0)
    for _ in range(40):                           # ~10 minutes of 4-2-4-2...
        ts = _breach(engine, runtime, quiet, start=ts, seconds=6.0)
        ts = _breach(engine, runtime, busy, start=ts, seconds=6.0)

    events = seeded.recent_events(limit=500)
    assert len(events) == 1
    assert events[0]["condition_state"] == "open"


def test_a_new_breach_after_a_close_is_a_second_event(config, seeded):
    """Hysteresis is not suppression. A zone that clears and fills again has had
    two incidents, and the second one is somebody's afternoon going wrong."""
    engine = _engine(config, seeded, FakeDetector(_people(1)))
    runtime = _runtime(engine)
    busy = FakeDetector(_people(5)).detect([None])[0]
    empty = FakeDetector(np.empty((0, 4), dtype=np.float32)).detect([None])[0]

    ts = _breach(engine, runtime, busy, seconds=30.0)
    ts = _breach(engine, runtime, empty, start=ts, seconds=90.0)   # clears, then waits
    _breach(engine, runtime, busy, start=ts, seconds=30.0)

    events = seeded.recent_events(limit=500)
    assert len(events) == 2
    assert [event["condition_state"] for event in events] == ["open", "closed"]


def test_the_cooldown_still_rate_limits_a_brand_new_condition(config, seeded):
    """`cooldown_sec` keeps the one job it was always right about. A zone that
    clears and refills inside the policy's 60 s cooldown does not get a second
    row the moment it does — the condition is filed as soon as the cooldown
    allows, rather than being dropped as the old per-frame loop dropped it."""
    engine = _engine(config, seeded, FakeDetector(_people(1)))
    runtime = _runtime(engine)
    busy = FakeDetector(_people(5)).detect([None])[0]
    empty = FakeDetector(np.empty((0, 4), dtype=np.float32)).detect([None])[0]

    ts = _breach(engine, runtime, busy, seconds=20.0)
    ts = _breach(engine, runtime, empty, start=ts, seconds=15.0)
    assert seeded.recent_events()[0]["condition_state"] == "closed"

    # Refilled ~40 s after the first condition opened, inside its cooldown.
    ts = _breach(engine, runtime, busy, start=ts, seconds=10.0)
    assert len(seeded.recent_events(limit=500)) == 1, "the cooldown was ignored"

    # Still over the limit once the cooldown expires: the row is written then,
    # not lost.
    _breach(engine, runtime, busy, start=ts, seconds=30.0)
    assert len(seeded.recent_events(limit=500)) == 2


# --- a condition that outlives the process -----------------------------------
#
# The row is the condition, so it survives a restart whether or not anybody
# meant it to. Two things have to be true for that to be a feature rather than a
# leak: an engine that comes back to the same breach must continue the row it
# left, and one that never comes back to it must have the row closed by
# something. On this appliance the something cannot be the VLM scheduler — it is
# not running.


def test_a_restarted_engine_adopts_the_row_it_left_open(config, seeded):
    """No second card on the feed, no second email, and a span that covers the
    outage instead of claiming the problem began when the process did."""
    detector = FakeDetector(_people(5))
    detections = detector.detect([None])[0]

    first = _engine(config, seeded, detector)
    next_ts = _breach(first, _runtime(first), detections, seconds=30.0)
    opened = seeded.recent_events()[0]
    first.stop()
    # Deliberately still open: a clean stop does not close conditions, because
    # "the engine went down" is not evidence the zone emptied.
    assert seeded.get_event(opened["id"])["condition_state"] == "open"

    # A new engine and a new runtime — no state machine, no cache. That is what
    # a restart looks like from this code's point of view.
    second = _engine(config, seeded, detector)
    _breach(second, _runtime(second), detections, start=next_ts, seconds=60.0)

    assert seeded.max_event_id() == opened["id"], "the restart filed a duplicate"
    assert second.conditions_adopted == 1
    assert second.conditions_opened == 0
    resumed = seeded.get_event(opened["id"])
    assert resumed["condition_state"] == "open"
    assert resumed["last_seen_ts"] > opened["last_seen_ts"]
    assert resumed["observations"] > opened["observations"]


def test_a_condition_nobody_came_back_to_is_closed_as_stale(config, seeded):
    detector = FakeDetector(_people(5))
    engine = _engine(config, seeded, detector)
    _breach(engine, _runtime(engine), detector.detect([None])[0], seconds=30.0)
    event = seeded.recent_events()[0]
    stale_after = config.detection.continuity.stale_after_sec

    now = time.time()
    assert engine.close_stale_conditions(now) == [], "closed a condition it just wrote"
    assert engine.close_stale_conditions(now + stale_after + 1.0) == [event["id"]]

    closed = seeded.get_event(event["id"])
    assert closed["condition_state"] == "closed"
    assert closed["close_reason"] == "stale"
    # Closed at the last moment the zone was known over its limit — a fact about
    # the plant — not at the moment we noticed nobody had looked.
    assert closed["closed_at"] == closed["last_seen_ts"]


def test_the_sweep_runs_on_the_engines_own_refresh(config, seeded):
    """It cannot live in the VLM scheduler's tick: `vlm.enabled` gates that
    whole runner, and on this box the VLM service is not running. A detector
    condition whose camera died would sit "ongoing" on the feed forever."""
    detector = FakeDetector(_people(5))
    engine = _engine(config, seeded, detector)
    runtime = _runtime(engine)
    engine.runtimes[runtime.id] = runtime
    _breach(engine, runtime, detector.detect([None])[0], seconds=30.0)
    event_id = seeded.recent_events()[0]["id"]

    # Age the row past the window. Written in SQL because `condition_updated_at`
    # is deliberately not in `update_event`'s allowlist — it is the appliance's
    # own clock and nothing outside the condition path may set it. It is
    # wall-clock, not the frame clock; see Database.stale_conditions for why.
    with seeded.conn() as conn:
        conn.execute("UPDATE events SET condition_updated_at = ? WHERE id = ?",
                     (time.time() - 100_000, event_id))
    engine._refresh()

    assert seeded.get_event(event_id)["condition_state"] == "closed"
    assert runtime.conditions == {}, "the engine still believes it owns the row"


def test_the_engines_sweep_leaves_vlm_conditions_alone(config, seeded):
    """Two sweepers, two windows, and `detector_type` is what keeps them off
    each other's rows. The VLM waits fifteen minutes because its evidence
    arrives every few seconds at best; turning the engine's two-minute window on
    a prompt condition would retire a live card between two samples."""
    from argus.db import Event

    vlm_event = seeded.insert_event(Event(
        camera_id="cam", detector_name="waste", detector_type="vlm", severity="medium",
        description="loose laminate on the apron", ts=1.0,
        condition_key="waste:cam", condition_state="open",
    ))
    engine = _engine(config, seeded, FakeDetector(_people(1)))

    assert engine.close_stale_conditions(time.time() + 1_000_000.0) == []
    assert seeded.get_event(vlm_event)["condition_state"] == "open"


def test_turning_continuity_off_stops_the_sweep_too(config, seeded):
    """A site running the old behaviour has no condition rows to sweep, and a
    sweep that ran anyway would close whatever the VLM had open."""
    config.detection.continuity.enabled = False
    engine = _engine(config, seeded, FakeDetector(_people(1)))
    assert engine.close_stale_conditions(time.time() + 1_000_000.0) == []


def test_no_active_policy_means_no_events_but_still_occupancy(config, seeded):
    seeded.update_policy("manpower_headcount", lifecycle="draft")
    detector = FakeDetector(_people(6))
    engine = _engine(config, seeded, detector)
    runtime = _runtime(engine)
    engine._process(runtime, _frame("cam", 1_000_000.0), detector.detect([None])[0])

    assert engine.policies == []
    assert seeded.recent_events() == []
    assert engine._maybe_flush_occupancy(force=True) == 1
    assert seeded.occupancy_series(camera_id="cam")[0]["count"] == 6


def test_occupancy_stores_the_window_peak(config, seeded):
    """One row per zone per window, holding the peak — the number a safety
    officer asks about — rather than 5 rows a second forever."""
    engine = _engine(config, seeded, FakeDetector(_people(1)))
    for count in (1, 4, 2):
        engine._record_occupancy("cam", "floor", 1_000_000.0, count)

    assert engine._maybe_flush_occupancy(force=True) == 1
    series = seeded.occupancy_series(camera_id="cam")
    assert [row["count"] for row in series] == [4]
    assert series[0]["zone_id"] == "floor"


def test_a_flickering_detection_does_not_add_a_person(config, seeded):
    """ByteTrack only reports a track once it has been seen twice, so a
    one-frame ghost cannot push a zone over its limit."""
    engine = _engine(config, seeded, FakeDetector(_people(1)))
    runtime = _runtime(engine)
    steady = FakeDetector(_people(2)).detect([None])[0]
    ghosts = FakeDetector(_people(5)).detect([None])[0]

    engine._process(runtime, _frame("cam", 1_000_000.0), steady)      # 2 tracks confirmed
    engine._process(runtime, _frame("cam", 1_000_000.2), ghosts)      # 3 unconfirmed newcomers
    assert seeded.recent_events() == []


def test_zone_edits_are_picked_up_without_a_restart(config, seeded):
    detector = FakeDetector(_people(4))
    engine = _engine(config, seeded, detector)
    runtime = _runtime(engine)
    engine.runtimes[runtime.id] = runtime
    detections = detector.detect([None])[0]

    seeded.update_zone("floor", sanctioned_limit=10)
    engine._refresh()
    _breach(engine, engine.runtimes["cam"], detections, seconds=30.0)
    assert seeded.recent_events() == []

    seeded.update_zone("floor", sanctioned_limit=1)
    engine._refresh()
    _breach(engine, engine.runtimes["cam"], detections, start=1_000_100.0, seconds=30.0)
    assert len(seeded.recent_events()) == 1


def test_a_flicker_never_reaches_the_headcount(config, seeded):
    """Two people really there; a third box that lasts two frames is not a head.

    With the shipped `min_consecutive_frames` the blip is never reported by the
    tracker, so it never reaches zone occupancy — which is what stands between
    one spurious box and an over-limit zone (limit 2 here).
    """
    import supervision as sv

    config.detection.tracker.min_consecutive_frames = 3
    engine = _engine(config, seeded, FakeDetector(_people(2)))
    runtime = _runtime(engine)
    counts = []
    for index in range(12):
        present = 3 if index in (6, 7) else 2          # the blip: frames 6 and 7 only
        detections = sv.Detections(xyxy=_people(present),
                                   confidence=np.full(present, 0.9, dtype=np.float32),
                                   class_id=np.zeros(present, dtype=int))
        engine._process(runtime, _frame("cam", START + index * STEP, index), detections)
        counts.append(runtime.live_occupancy[0]["count"])
    assert max(counts) == 2, counts                    # the third box was never a person
    assert counts[-1] == 2                             # and the two real people are counted
    assert seeded.recent_events() == []


def test_tracker_ids_are_stable_across_frames(config, seeded):
    detector = FakeDetector(_people(3))
    engine = _engine(config, seeded, detector)
    runtime = _runtime(engine)
    detections = detector.detect([None])[0]

    seen = []
    for tick in range(6):
        tracked = engine.trackers.for_camera("cam", 5.0).update(detections)
        seen.append(sorted(tracked.tracker_id.tolist()))
    assert seen[-1] == seen[-2] and len(seen[-1]) == 3


def test_trackers_are_not_shared_between_cameras(config, seeded):
    engine = _engine(config, seeded, FakeDetector(_people(1)))
    assert engine.trackers.for_camera("cam_a") is not engine.trackers.for_camera("cam_b")
    assert engine.trackers.for_camera("cam_a") is engine.trackers.for_camera("cam_a")


# --- which weights are running ---------------------------------------------
#
# The detector used to build itself lazily, inside the vision thread, on the
# first batch. That is the wrong place to fail: a daemon thread that dies leaves
# an engine which decodes every camera, reports healthy, and finds nobody. It
# never mattered while the weights were a constant; `detection.model.checkpoint`
# makes them a config line, and a config line can be wrong.


class LoadableDetector(FakeDetector):
    """A fake that has the two optional methods the engine now looks for."""

    def __init__(self, boxes, *, fails: Exception | None = None, summary: str = "fake weights"):
        super().__init__(boxes)
        self.loads = 0
        self._fails = fails
        self._summary = summary

    def load(self):
        self.loads += 1
        if self._fails is not None:
            raise self._fails
        return self

    def weights_summary(self) -> str:
        return self._summary


def test_a_checkpoint_that_will_not_load_stops_the_engine_before_any_camera(
    config, seeded, capsys
):
    from argus.vision.checkpoint import CheckpointError

    detector = LoadableDetector(
        _people(2), fails=CheckpointError("no provenance.json beside those weights")
    )
    engine = _engine(config, seeded, detector)
    with pytest.raises(CheckpointError, match="provenance.json"):
        engine.start()

    assert detector.loads == 1
    # Nothing was spawned, so there is nothing to stop: the camera never left
    # the state the previous run put it in.
    assert not engine.runtimes
    assert "camera cam:" not in capsys.readouterr().out


@pytest.mark.slow
def test_the_startup_lines_do_not_print_a_camera_password(config, seeded, ffmpeg, capsys):
    """Without the gateway the decoder is handed the camera's own URL, and the
    `camera ...: decoding from ...` line used to print it whole — into the
    journal, and into every terminal somebody ran `argus run` in."""
    seeded.upsert_camera(Camera(
        id="nvr", name="NVR", source="rtsp://test:hunter2-pw@127.0.0.1:9/Streaming/Channels/101",
    ))
    engine = _engine(config, seeded, LoadableDetector(_people(2)))
    engine.start()
    try:
        deadline = time.time() + 10
        while time.time() < deadline and not engine.runtimes["nvr"].source.stats.last_error:
            time.sleep(0.05)
        # What the stats tick would store as the camera's status_detail.
        assert "hunter2-pw" not in (engine.runtimes["nvr"].source.stats.last_error or "")
        assert "hunter2-pw" not in engine.report()
    finally:
        engine.stop()
    output = capsys.readouterr()
    assert "hunter2-pw" not in output.out + output.err
    assert "decoding from rtsp://test:***@127.0.0.1:9/Streaming/Channels/101" in output.out


@pytest.mark.slow
def test_the_startup_banner_names_the_weights(config, seeded, ffmpeg, capsys):
    """Printed stock or fine-tuned, because the risk of an optional checkpoint
    is a site running one nobody knows about — or assuming one that is off."""
    detector = LoadableDetector(_people(2), summary="rf-detr-nano fine-tune: best.pth")
    engine = _engine(config, seeded, detector)
    engine.start()
    try:
        assert "detector: rf-detr-nano fine-tune: best.pth" in capsys.readouterr().out
        assert detector.loads == 1
    finally:
        engine.stop()


def test_a_detector_without_the_optional_methods_still_starts(config, seeded, capsys):
    """`Detector` is a two-method protocol and a fake may implement only it."""
    engine = _engine(config, seeded, FakeDetector(_people(2)))
    engine._refresh_policies()
    # Reaching start() would spawn ffmpeg; the guard being exercised is the
    # getattr, so call the same code the way start() does.
    assert getattr(engine.detector, "load", None) is None
    assert getattr(engine.detector, "weights_summary", None) is None


# --- the whole engine ------------------------------------------------------


@pytest.mark.slow
def test_engine_run_produces_events_from_the_demo_video(config, seeded, ffmpeg):
    """Decoders, vision loop, analytics and the event pipeline, wired together —
    the only fake left is the detector."""
    detector = FakeDetector(_people(4))
    engine = _engine(config, seeded, detector)
    engine.start()
    try:
        deadline = time.time() + 45
        while time.time() < deadline and not seeded.recent_events():
            time.sleep(0.5)
    finally:
        engine.stop()

    events = seeded.recent_events()
    assert events, "the engine produced no events from the demo recording"
    assert detector.calls > 0
    assert (config.snapshots_dir / events[0]["snapshot_path"]).exists()
    assert seeded.occupancy_series(camera_id="cam")
    assert seeded.get_camera("cam")["status"] == "offline"      # set on shutdown


@pytest.mark.slow
def test_camera_status_follows_the_decoder(config, database, ffmpeg, tmp_path):
    """A camera that never delivers a frame must not read 'live'."""
    database.upsert_camera(Camera(id="dead", name="Dead", source=str(tmp_path / "nope.mp4")))
    engine = _engine(config, database, FakeDetector(_people(1)))
    engine.config.detection.start_timeout_sec = 1.0
    engine.start()
    try:
        time.sleep(2.5)
        engine._refresh_camera_status()
        camera = database.get_camera("dead")
        assert camera["status"] == "error"
        assert camera["status_detail"]
    finally:
        engine.stop()


# --- what the live wall reads ----------------------------------------------


def test_live_state_is_published_for_the_wall(config, seeded):
    """Zone counts and boxes from the last analysed frame ride out in the stats
    file — the live wall's only source for "who is in this zone right now"."""
    detector = FakeDetector(_people(3))
    engine = _engine(config, seeded, detector)
    runtime = _runtime(engine)
    engine.runtimes[runtime.id] = runtime

    next_ts = _breach(engine, runtime, detector.detect([None])[0], seconds=10.0)

    assert runtime.live_ts == pytest.approx(next_ts - STEP)
    assert runtime.live_occupancy == [{
        "zone_id": "floor", "zone_name": "Packing floor",
        "count": 3, "smoothed_count": 3, "limit": 2, "over_limit": True,
    }]
    assert len(runtime.live_tracks) == 3
    # Normalized, and each row is [x1, y1, x2, y2, track_id, confidence].
    assert all(len(box) == 6 and all(0.0 <= v <= 1.0 for v in box[:4])
               for box in runtime.live_tracks)

    published = engine.stats()["cameras"][0]
    assert published["occupancy"] == runtime.live_occupancy
    assert published["tracks"] == runtime.live_tracks
    assert published["live_ts"] == pytest.approx(next_ts - STEP)


def test_the_wall_does_not_flash_amber_for_one_frame(config, seeded):
    """The tile border follows the *smoothed* count, because the raw one is
    wrong ~13% of the time per frame and a zone at its limit therefore strobed.
    The number printed on the tile stays instantaneous: a wall that lags the
    video is a wall nobody trusts."""
    engine = _engine(config, seeded, FakeDetector(_people(1)))
    runtime = _runtime(engine)
    steady = FakeDetector(_people(2)).detect([None])[0]
    spike = FakeDetector(_people(9)).detect([None])[0]

    ts = _breach(engine, runtime, steady, seconds=10.0)
    engine._process(runtime, _frame("cam", ts), spike)

    zone = runtime.live_occupancy[0]
    assert zone["over_limit"] is False, "one frame lit the wall"
    assert zone["smoothed_count"] == 2
    assert zone["count"] == 2, "ByteTrack has not confirmed the newcomers yet"


def test_the_wall_stays_lit_while_the_condition_is_open(config, seeded):
    """The other side of the same rule. A tile that went grey while the card
    beside it still read "ongoing" would be the worse of the two errors, so a
    zone with a condition open stays amber through a dip at the limit."""
    engine = _engine(config, seeded, FakeDetector(_people(1)))
    runtime = _runtime(engine)
    busy = FakeDetector(_people(5)).detect([None])[0]
    at_limit = FakeDetector(_people(2)).detect([None])[0]

    ts = _breach(engine, runtime, busy, seconds=20.0)
    assert seeded.recent_events()[0]["condition_state"] == "open"

    _breach(engine, runtime, at_limit, start=ts, seconds=5.0)   # inside clear_sec
    assert runtime.live_occupancy[0]["smoothed_count"] == 2
    assert runtime.live_occupancy[0]["over_limit"] is True


def test_live_state_reports_an_empty_zone_as_empty(config, seeded):
    detector = FakeDetector(np.empty((0, 4), dtype=np.float32))
    engine = _engine(config, seeded, detector)
    runtime = _runtime(engine)

    engine._process(runtime, _frame("cam", 1_000_000.0), detector.detect([None])[0])

    assert runtime.live_occupancy[0]["count"] == 0
    assert runtime.live_occupancy[0]["over_limit"] is False
    assert runtime.live_tracks == []


# --- the analytics pool, as the engine uses it ------------------------------
#
# The pool's own semantics are pinned in tests/test_analytics_pool.py. What is
# tested here is that the engine reaches the same verdicts through it as it did
# when everything ran inline on the vision thread — the change is meant to be
# invisible to every camera, zone and policy.


def test_a_pooled_engine_reaches_the_same_verdict_as_an_inline_one(config, seeded):
    detector = FakeDetector(_people(4))
    config.detection.analytics_workers = 2
    config.detection.analytics_queue_depth = 100      # deep enough that nothing is shed
    engine = _engine(config, seeded, detector)
    runtime = _runtime(engine)
    engine.runtimes[runtime.id] = runtime
    detections = detector.detect([None])[0]
    assert engine.analytics is not None
    engine.analytics.start()
    try:
        for tick in range(41):                        # 8 s at 5 fps, as inline
            engine._dispatch_analytics(runtime, _frame("cam", START + tick * STEP), detections)
        assert engine.analytics.drain(timeout=10.0)
    finally:
        engine.analytics.stop(timeout=5.0)

    events = seeded.recent_events()
    assert len(events) == 1
    assert "4 people" in events[0]["description"]
    assert events[0]["ts"] == OPENS_AT
    assert runtime.frames_processed == 41
    assert (config.snapshots_dir / events[0]["snapshot_path"]).exists()


def test_the_pool_preserves_frame_order_for_a_camera(config, seeded):
    """The engine's own ordering guarantee, from the outside: ByteTrack sees
    frames in capture order, so `_process` is called with ascending timestamps.
    A tracker fed frames out of order does not degrade, it lies."""
    detector = FakeDetector(_people(1))
    config.detection.analytics_workers = 3
    config.detection.analytics_queue_depth = 100      # deep enough that nothing is shed
    engine = _engine(config, seeded, detector)
    seen: list[float] = []
    inline = engine._process

    def record(runtime, frame, detections):
        seen.append(frame.ts)
        inline(runtime, frame, detections)

    engine.analytics.handler = record

    runtime = _runtime(engine)
    engine.runtimes[runtime.id] = runtime
    engine.analytics.start()
    try:
        detections = detector.detect([None])[0]
        for tick in range(60):
            engine._dispatch_analytics(runtime, _frame("cam", 1_000_000.0 + tick * 0.2), detections)
        assert engine.analytics.drain(timeout=15.0)
    finally:
        engine.analytics.stop(timeout=5.0)

    assert seen == sorted(seen)
    assert len(seen) == 60


def test_a_saturated_pool_drops_frames_instead_of_stalling_the_detector(config, seeded):
    """Back pressure must never reach the vision thread: it holds the GPU, and a
    detector that waits for analytics is the ceiling this pool exists to lift."""
    config.detection.analytics_workers = 1
    config.detection.analytics_queue_depth = 2
    detector = FakeDetector(_people(1))
    engine = _engine(config, seeded, detector)
    runtime = _runtime(engine)
    engine.runtimes[runtime.id] = runtime

    gate = threading.Event()
    stuck = threading.Event()

    def blocked(*_args) -> None:
        stuck.set()
        gate.wait(5.0)

    engine.analytics.handler = blocked
    engine.analytics.start()
    try:
        detections = detector.detect([None])[0]
        engine._dispatch_analytics(runtime, _frame("cam", 1_000_000.0), detections)
        assert stuck.wait(5.0)
        started = time.time()
        for tick in range(1, 40):
            engine._dispatch_analytics(runtime, _frame("cam", 1_000_000.0 + tick), detections)
        elapsed = time.time() - started
    finally:
        gate.set()
        engine.analytics.stop(timeout=5.0)

    assert elapsed < 1.0, "the vision thread waited on the analytics pool"
    assert runtime.frames_dropped > 0
    assert engine.stats()["analytics"]["dropped"] == runtime.frames_dropped


def test_analytics_can_be_turned_off_and_then_runs_inline(config, seeded):
    config.detection.analytics_workers = 0
    detector = FakeDetector(_people(4))
    engine = _engine(config, seeded, detector)
    runtime = _runtime(engine)
    detections = detector.detect([None])[0]

    assert engine.analytics is None
    for tick in range(41):
        engine._dispatch_analytics(runtime, _frame("cam", START + tick * STEP), detections)
    assert len(seeded.recent_events()) == 1
    assert engine.stats()["analytics"] is None


def test_the_pool_is_drained_before_shutdown(config, seeded):
    """A frame the detector already decided on may still owe an event row, a
    snapshot and a clip. Losing those is worse than a slower stop."""
    detector = FakeDetector(_people(4))
    config.detection.analytics_workers = 1
    engine = _engine(config, seeded, detector)
    runtime = _runtime(engine)
    engine.runtimes[runtime.id] = runtime
    detections = detector.detect([None])[0]
    # The breach is established on this side of the queue, so the one frame
    # handed to the pool is precisely the frame that owes the row.
    next_ts = _breach(engine, runtime, detections, seconds=4.8)
    assert seeded.recent_events() == []

    engine.analytics.start()
    engine._dispatch_analytics(runtime, _frame("cam", next_ts), detections)
    engine.stop()

    assert len(seeded.recent_events()) == 1


# --- the vision loop's batch grace ------------------------------------------


def test_the_loop_stops_waiting_once_the_batch_is_full(config, seeded):
    """The grace period buys coalescing on a small fleet. On a large one the
    batch is full before the loop finishes asking, and waiting the full 25 ms
    anyway cost roughly a third of the cycle at 64 cameras."""
    from argus.engine import _BATCH_GRACE_SEC

    detector = FakeDetector(_people(1))
    engine = _engine(config, seeded, detector)
    frame = _frame("cam", 1_000_000.0)
    for index in range(8):
        runtime = _runtime(engine)
        runtime.camera = {**runtime.camera, "id": f"cam{index}"}
        runtime.offer(frame)
        engine.runtimes[f"cam{index}"] = runtime

    started = time.perf_counter()
    engine._wait_for_batch(max_batch=8)
    full = time.perf_counter() - started

    for runtime in engine.runtimes.values():
        runtime.take()
    started = time.perf_counter()
    engine._wait_for_batch(max_batch=8)
    empty = time.perf_counter() - started

    assert full < _BATCH_GRACE_SEC / 2, "a full batch still waited out the grace period"
    assert empty >= _BATCH_GRACE_SEC * 0.8, "an empty batch gave up on stragglers too early"


# --- cold start -------------------------------------------------------------


@pytest.mark.slow
def test_cameras_are_started_one_at_a_time(config, seeded, ffmpeg, tmp_path):
    """A cold start is the worst moment of a fleet's life: three ffmpeg
    processes per camera, all spawning at once. Staggering costs one ramp and
    is why 96 cameras now reach hardware decode instead of being starved off it.
    """
    for index in range(3):
        seeded.upsert_camera(Camera(id=f"stagger{index}", name=f"Stagger {index}",
                                    source=str(tmp_path / "absent.mp4")))
    config.detection.start_stagger_sec = 0.4
    engine = _engine(config, seeded, FakeDetector(_people(1)))

    started = time.perf_counter()
    engine.start()
    elapsed = time.perf_counter() - started
    try:
        # 4 cameras (the seeded one plus three), 3 gaps of 0.4 s.
        assert elapsed >= 1.2
        assert len(engine.runtimes) == 4
        # Every camera got the fleet-scaled deadline, not the single-camera one.
        for runtime in engine.runtimes.values():
            assert runtime.source.start_timeout_sec == pytest.approx(21.0)
    finally:
        engine.stop()


def test_stop_waits_for_a_clip_that_is_already_being_cut(config, database, monkeypatch):
    """A transcoded clip is an NVENC pass, not a 10 ms remux.

    `timer.cancel()` stops a timer that has not fired and does nothing to one
    already inside `cut()`. Before this join, the interpreter exited underneath
    a running encode and left a partial mp4 with no moov atom in `data/clips/`.
    """
    import threading
    import time as time_module

    from argus.engine import Engine

    engine = Engine(config, database, use_gateway=False)
    started = threading.Event()
    finished = threading.Event()

    def slow_cut():
        started.set()
        time_module.sleep(0.5)
        finished.set()

    timer = threading.Timer(0.01, slow_cut)
    timer.daemon = True
    with engine._timers_lock:
        engine._timers.add(timer)
    timer.start()
    assert started.wait(2.0), "the cut never began"

    engine.stop()
    assert finished.is_set(), "stop() returned while a clip was still being written"


def test_stop_cancels_a_clip_whose_timer_has_not_fired(config, database):
    """The other half: a fleet that raised events in its last second must not
    turn shutdown into a minute of encoding footage nobody asked for."""
    import threading

    from argus.engine import Engine

    engine = Engine(config, database, use_gateway=False)
    ran = threading.Event()
    timer = threading.Timer(30.0, ran.set)
    timer.daemon = True
    with engine._timers_lock:
        engine._timers.add(timer)
    timer.start()

    started = __import__("time").monotonic()
    engine.stop()
    assert not ran.is_set()
    assert __import__("time").monotonic() - started < 5.0
