"""Ongoing conditions: one row per problem, however long it lasts.

The bug this file is about was measured, not imagined. On 2026-08-26 this
appliance's own database held **140** events for `anmol_packing_floor_waste`,
every one of them about the same laminate offcut on the same wrapper apron, 40
of them filed overnight at the policy's ~15-minute cooldown. The condition never
changed; only the clock did.

Three layers are tested here because the fix spans three:

* the identity rule itself (`argus/vlm/continuity.py`), as arithmetic;
* the scheduler, driven with synthetic timestamps so extending, clearing and
  going stale are asserted exactly and without a GPU;
* the seam — schema migration 7 applied to a *populated* database (a copy of
  the real one when this box has one), the events API's shape, and the SSE
  frame that redraws a card without moving the list under the reader.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from argus.config import ROOT, ArgusConfig
from argus.db import MIGRATIONS, SCHEMA_VERSION, Camera, Database, Event, Policy
from argus.db import Zone as DbZone
from argus.vlm import continuity
# Fixtures do not cross module boundaries in pytest unless they live in
# conftest, so the two harnesses this file needs are imported by name: the fake
# OpenAI-compatible server and its scheduler helpers from test_vlm, and the
# live FastAPI app (plus its signed-in client and its one-camera fleet) from
# test_api. Importing them beats copying them — a second copy of `FakeVlmServer`
# is a second wire contract to keep in step with the first.
from tests.test_api import (  # noqa: F401 - fixtures used by name
    anon,
    client,
    fleet,
    live_server,
    make_event,
)
from tests.test_vlm import (  # noqa: F401 - fixtures used by name
    FakeFrame,
    FakeVlmServer,
    RecordingNotifier,
    make_runner,
    seed_camera,
    seed_policy,
    server,
    tick,
    vlm_config,
)

KEY = "phone_usage_on_line:cam_a"
FIRES = {"triggered": True, "confidence": 0.9, "reason": "a phone, still", "box_2d": None}
QUIET = {"triggered": False, "confidence": 0.9, "reason": "nobody is holding one", "box_2d": None}

#: A zone covering a quarter of the frame, so `zone_crop` actually crops rather
#: than declining (see MIN_CROP_PX and the "most of the view" rule).
QUARTER = [[0.05, 0.05], [0.45, 0.05], [0.45, 0.45], [0.05, 0.45]]


def fires_at(box):
    return {"triggered": True, "confidence": 0.9, "reason": "a phone, still", "box_2d": box}


# --------------------------------------------------------------------------
# the identity rule, as arithmetic
# --------------------------------------------------------------------------


def test_iou_of_a_box_with_itself_is_one():
    assert continuity.iou([0.2, 0.2, 0.6, 0.6], [0.2, 0.2, 0.6, 0.6]) == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero_and_of_a_missing_box_is_none():
    """Zero and None are opposite facts and the caller must be able to tell
    them apart: `0.0` is "the model located this somewhere else", `None` is
    "the model declined to point". Collapsing them would read every
    box-less verdict as disagreement and split every condition it touched."""
    assert continuity.iou([0.0, 0.0, 0.1, 0.1], [0.8, 0.8, 0.9, 0.9]) == 0.0
    assert continuity.iou(None, [0.1, 0.1, 0.2, 0.2]) is None
    assert continuity.iou([0.1, 0.1, 0.2, 0.2], None) is None
    assert continuity.iou(None, None) is None


def test_iou_is_a_ratio_of_areas():
    # Two unit-quarter boxes overlapping in half of each: 0.5 / 1.5.
    assert continuity.iou([0.0, 0.0, 0.4, 0.2], [0.2, 0.0, 0.6, 0.2]) == pytest.approx(1 / 3)


def test_an_unscoped_policy_needs_its_boxes_to_agree():
    """With no zone to anchor it, the grounding box is the only evidence of
    *where*, so the threshold is what decides identity."""
    same_place = dict(last_seen_ts=0.0, ts=10.0, stale_after_sec=900.0, zone_scoped=False)
    assert continuity.continues(
        **same_place, open_box=[0.2, 0.2, 0.6, 0.6], new_box=[0.21, 0.21, 0.61, 0.61],
        min_iou=0.5,
    )
    moved = continuity.continues(
        **same_place, open_box=[0.0, 0.0, 0.2, 0.2], new_box=[0.7, 0.7, 0.9, 0.9], min_iou=0.1,
    )
    assert not moved
    assert moved.reason == continuity.MOVED
    assert moved.iou == 0.0


def test_the_threshold_is_the_knob_and_it_moves_the_answer():
    args = dict(last_seen_ts=0.0, ts=10.0, stale_after_sec=900.0, zone_scoped=False,
                open_box=[0.0, 0.0, 0.4, 0.2], new_box=[0.2, 0.0, 0.6, 0.2])   # IoU 1/3
    assert continuity.continues(**args, min_iou=0.3)
    assert not continuity.continues(**args, min_iou=0.4)


def test_a_zone_scoped_policy_does_not_compare_boxes():
    """Measured, not assumed. Over this appliance's own 139 consecutive
    laminate observations the *median* IoU between one sighting and the next is
    0.000 — the boxes are ~0.077 wide and their centres wander a median 0.11 of
    the frame — so a box test inside a zone would have split one ongoing mess
    into roughly a hundred. The zone is the anchor; the box is where in the
    zone, which is a different question."""
    match = continuity.continues(
        last_seen_ts=0.0, ts=10.0, stale_after_sec=900.0, zone_scoped=True,
        open_box=[0.0, 0.0, 0.1, 0.1], new_box=[0.9, 0.9, 1.0, 1.0], min_iou=0.9,
    )
    assert match and match.iou is None


def test_a_missing_box_is_not_disagreement():
    assert continuity.continues(
        last_seen_ts=0.0, ts=10.0, stale_after_sec=900.0, zone_scoped=False,
        open_box=[0.1, 0.1, 0.2, 0.2], new_box=None, min_iou=0.9,
    )


def test_a_condition_nobody_watched_for_too_long_is_over():
    stale = continuity.continues(
        last_seen_ts=0.0, ts=901.0, stale_after_sec=900.0, zone_scoped=True,
    )
    assert not stale
    assert stale.reason == continuity.STALE
    # Exactly at the window is still inside it.
    assert continuity.continues(last_seen_ts=0.0, ts=900.0, stale_after_sec=900.0,
                                zone_scoped=True)


def test_box_of_reads_a_row_and_shrugs_at_a_broken_one():
    assert continuity.box_of({"meta": {"box_2d": [0.1, 0.2, 0.3, 0.4]}}) == [0.1, 0.2, 0.3, 0.4]
    assert continuity.box_of({"meta": None}) is None
    assert continuity.box_of({"meta": {"box_2d": "elsewhere"}}) is None
    assert continuity.box_of({"meta": {"box_2d": [0.1, 0.2]}}) is None


# --------------------------------------------------------------------------
# the scheduler
# --------------------------------------------------------------------------


def zoned_policy(database: Database) -> None:
    """A policy scoped to a zone on one camera — the shape all four of the
    plant's own prompt policies have."""
    seed_camera(database, "cam_a")
    database.upsert_zone(DbZone(id="cam_a_apron", camera_id="cam_a",
                                name="Wrapper floor apron", polygon=QUARTER))
    seed_policy(database, interval_sec=6.0, cooldown_sec=45.0,
                scope=[{"camera_id": "cam_a", "zone_id": "cam_a_apron"}])


def test_a_condition_that_persists_accumulates_observations_and_stays_one_card(
    vlm_config, database, server
):
    """The whole feature, on the shape of policy that produced the bug."""
    server.replies = [FIRES]
    zoned_policy(database)
    notifier = RecordingNotifier()
    clips: list[tuple] = []
    runner = make_runner(vlm_config, database, server,
                         notifier=notifier, on_event=lambda *a: clips.append(a))
    runner.refresh(now=0.0)
    key = f"{KEY}:cam_a_apron"
    runner.schedules[key].next_due = 0.0

    for now in range(0, 60, 6):
        runner.offer(FakeFrame("cam_a", ts=float(now)))
        tick(runner, now=float(now))

    assert server.calls == 10
    assert database.count_events() == 1

    event = database.recent_events()[0]
    assert event["condition_key"] == key
    assert event["condition_state"] == "open"
    assert event["observations"] == 10
    assert event["ts"] == 0.0 and event["last_seen_ts"] == 54.0
    assert event["closed_at"] is None

    # Told once, clipped once. The nine re-sightings are not news: a second
    # email about a problem the recipient has already been told about is the
    # alert fatigue this replaces, and re-cutting the +-8 s clip every sample
    # would spend the ring buffer on the middle of a thing whose beginning is
    # the part worth keeping.
    assert notifier.dispatched == [event["id"]]
    assert len(clips) == 1


def test_an_untriggered_verdict_closes_the_condition(vlm_config, database, server):
    """The one piece of positive evidence this appliance ever gets that a
    condition ended: the model looked and said no."""
    server.replies = [FIRES]
    zoned_policy(database)
    runner = make_runner(vlm_config, database, server)
    runner.refresh(now=0.0)
    key = f"{KEY}:cam_a_apron"
    runner.schedules[key].next_due = 0.0

    for now in (0.0, 6.0):
        runner.offer(FakeFrame("cam_a", ts=now))
        tick(runner, now=now)

    server.replies = [QUIET]
    runner.offer(FakeFrame("cam_a", ts=12.0))
    tick(runner, now=12.0)

    event = database.recent_events()[0]
    assert event["condition_state"] == "closed"
    assert event["close_reason"] == "cleared"
    assert event["closed_at"] == 12.0
    assert event["observations"] == 2
    assert runner.schedules[key].closed == 1

    # And the next time it is true again, that is a new incident — with its own
    # start — rather than a resurrection of the old span.
    server.replies = [FIRES]
    runner.offer(FakeFrame("cam_a", ts=200.0))
    tick(runner, now=200.0)
    assert database.count_events() == 2
    assert database.recent_events()[0]["ts"] == 200.0


def test_an_unconfident_trigger_neither_extends_nor_closes(vlm_config, database, server):
    """`min_confidence` is the bar for raising, and hesitation is not a claim
    that the mess was cleared. Closing on it would end a real condition on the
    model's doubt and write a span that finished while the laminate was still
    on the floor."""
    server.replies = [FIRES]
    zoned_policy(database)
    vlm_config.vlm.min_confidence = 0.5
    runner = make_runner(vlm_config, database, server)
    runner.refresh(now=0.0)
    key = f"{KEY}:cam_a_apron"
    runner.schedules[key].next_due = 0.0
    runner.offer(FakeFrame("cam_a", ts=0.0))
    tick(runner, now=0.0)

    server.replies = [{"triggered": True, "confidence": 0.2, "reason": "maybe", "box_2d": None}]
    runner.offer(FakeFrame("cam_a", ts=6.0))
    tick(runner, now=6.0)

    event = database.recent_events()[0]
    assert event["condition_state"] == "open"
    assert event["observations"] == 1
    assert event["last_seen_ts"] == 0.0


def test_a_condition_nobody_observed_goes_stale_and_is_closed(vlm_config, database, server):
    """Silence is not evidence the problem ended — a camera dropped, or the
    model went degraded — so the span is closed at the *last observation*, not
    at the moment the sweep noticed.

    The sweep's own clock is wall-clock (`condition_updated_at`, when the
    verdict was written) while the frame carries 1000.0, and that split is the
    point of the test: staleness is a question about the appliance, the span is
    a question about the plant.
    """
    server.replies = [FIRES]
    zoned_policy(database)
    vlm_config.vlm.continuity.stale_after_sec = 100.0
    runner = make_runner(vlm_config, database, server)
    runner.refresh(now=0.0)
    runner.schedules[f"{KEY}:cam_a_apron"].next_due = 0.0
    runner.offer(FakeFrame("cam_a", ts=1000.0))
    tick(runner, now=1000.0)

    wrote_at = float(database.recent_events()[0]["condition_updated_at"])
    assert runner.close_stale_conditions(wrote_at + 50.0) == []   # inside the window
    closed = runner.close_stale_conditions(wrote_at + 101.0)

    event = database.recent_events()[0]
    assert closed == [event["id"]]
    assert event["condition_state"] == "closed"
    assert event["close_reason"] == "stale"
    assert event["closed_at"] == 1000.0                          # last seen, not now


def test_the_stale_sweep_is_not_confounded_by_the_model_s_latency(
    vlm_config, database, server
):
    """The defect the first live run found, pinned.

    A 20 s policy against a ~15 s model under a 30 s window closed a healthy
    condition between every pair of samples and opened a new one each time: the
    sweep was comparing `now` against the *frame* timestamp, so the window had
    to cover the interval AND the inference. Four rows in two minutes, which is
    the bug this whole feature exists to remove, reintroduced by its own
    housekeeping.
    """
    server.replies = [FIRES]
    zoned_policy(database)
    vlm_config.vlm.continuity.stale_after_sec = 30.0
    runner = make_runner(vlm_config, database, server)
    runner.refresh(now=0.0)
    runner.schedules[f"{KEY}:cam_a_apron"].next_due = 0.0

    # A frame captured 15 s before the verdict was written — the model's
    # latency. Under the old frame-clock rule `now - frame_ts` is already 15 s
    # of a 30 s budget before a single second of silence has passed.
    runner.offer(FakeFrame("cam_a", ts=1000.0))
    tick(runner, now=1000.0)
    wrote_at = float(database.recent_events()[0]["condition_updated_at"])

    # 20 s later the next sample is due and nothing has gone stale.
    assert runner.close_stale_conditions(wrote_at + 20.0) == []
    runner.offer(FakeFrame("cam_a", ts=1020.0))
    runner.schedules[f"{KEY}:cam_a_apron"].next_due = 0.0
    tick(runner, now=1020.0)

    events = database.recent_events()
    assert len(events) == 1
    assert events[0]["condition_state"] == "open"
    assert events[0]["observations"] == 2


def test_a_condition_left_open_by_a_stopped_appliance_is_closed_on_the_next_run(
    vlm_config, database, server
):
    """The case that matters most for the stale sweep. An appliance stopped
    with conditions open would otherwise come back claiming every one of them
    is still ongoing, and eventually write a span covering the hours it spent
    switched off."""
    zoned_policy(database)
    database.insert_event(Event(
        camera_id="cam_a", zone_id="cam_a_apron", policy_id="phone_usage_on_line",
        detector_name="phone_usage_on_line", detector_type="vlm", severity="medium",
        description="left open by a previous process", ts=1000.0, stage="reviewed",
        condition_key=f"{KEY}:cam_a_apron", condition_state="open",
    ))
    runner = make_runner(vlm_config, database, server)
    runner.refresh(now=9000.0)

    # An hour of wall-clock later, whatever the frames said.
    assert runner.close_stale_conditions(time.time() + 3600.0)
    assert database.recent_events()[0]["close_reason"] == "stale"


def test_a_condition_whose_policy_is_gone_is_still_closable(vlm_config, database, server):
    """Otherwise it sits 'ongoing' on the feed forever, because the only thing
    that could ever end it was deleted."""
    seed_camera(database, "cam_a")
    database.insert_event(Event(
        camera_id="cam_a", policy_id=None, detector_name="deleted_policy",
        detector_type="vlm", severity="low", description="orphan", ts=1000.0,
        stage="shadow", condition_key="deleted_policy:cam_a", condition_state="open",
    ))
    runner = make_runner(vlm_config, database, server)

    assert runner.close_stale_conditions(time.time() + 1_000_000.0)
    assert database.recent_events()[0]["condition_state"] == "closed"


def test_the_cooldown_still_rate_limits_a_flapping_policy(vlm_config, database, server):
    """The debouncer is not consulted on an extension and is not reset by a
    close, so its clock runs from the last time a condition *opened*.

    That is the conservative choice and it is what stops a policy that fires and
    clears on alternate samples from opening a new incident every interval —
    the alert storm `cooldown_sec` exists to prevent. The price is that a
    genuinely new incident arriving inside the previous one's cooldown waits for
    it, which is exactly what it did before any of this.
    """
    server.replies = [FIRES]
    zoned_policy(database)                                   # cooldown 45 s
    runner = make_runner(vlm_config, database, server)
    runner.refresh(now=0.0)
    key = f"{KEY}:cam_a_apron"
    runner.schedules[key].next_due = 0.0
    runner.offer(FakeFrame("cam_a", ts=0.0))
    tick(runner, now=0.0)

    server.replies = [QUIET]
    runner.offer(FakeFrame("cam_a", ts=6.0))
    tick(runner, now=6.0)
    assert database.recent_events()[0]["condition_state"] == "closed"

    server.replies = [FIRES]
    runner.offer(FakeFrame("cam_a", ts=12.0))                # inside the cooldown
    tick(runner, now=12.0)
    assert database.count_events() == 1                      # held, as it always was

    runner.offer(FakeFrame("cam_a", ts=60.0))                # past it
    tick(runner, now=60.0)
    assert database.count_events() == 2


def test_a_grounding_box_that_moved_starts_its_own_condition(vlm_config, database, server):
    """An *unscoped* policy has no polygon to anchor it, so two triggered
    verdicts in different parts of the view are two problems. The old one is
    closed rather than left dangling, because the invariant downstream — and
    the reason the open lookup is one indexed row — is one open condition per
    (policy, camera, zone)."""
    server.replies = [fires_at([100, 100, 300, 300])]
    seed_policy(database, interval_sec=6.0, cooldown_sec=10.0)     # no scope: whole camera
    seed_camera(database, "cam_a")
    runner = make_runner(vlm_config, database, server)
    runner.refresh(now=0.0)
    runner.schedules[KEY].next_due = 0.0
    runner.offer(FakeFrame("cam_a", ts=0.0))
    tick(runner, now=0.0)

    server.replies = [fires_at([700, 700, 900, 900])]              # the other corner
    runner.offer(FakeFrame("cam_a", ts=20.0))                      # past the cooldown
    tick(runner, now=20.0)

    events = database.recent_events()
    assert len(events) == 2
    assert events[0]["condition_state"] == "open" and events[0]["ts"] == 20.0
    assert events[1]["condition_state"] == "closed"
    assert events[1]["close_reason"] == "moved"


def test_a_moved_box_inside_the_cooldown_leaves_the_open_card_alone(
    vlm_config, database, server
):
    """Closing first and *then* being refused by the cooldown would retire the
    card on the feed and put nothing in its place — the reader would watch a
    live problem disappear."""
    server.replies = [fires_at([100, 100, 300, 300])]
    seed_policy(database, interval_sec=6.0, cooldown_sec=600.0)
    seed_camera(database, "cam_a")
    runner = make_runner(vlm_config, database, server)
    runner.refresh(now=0.0)
    runner.schedules[KEY].next_due = 0.0
    runner.offer(FakeFrame("cam_a", ts=0.0))
    tick(runner, now=0.0)

    server.replies = [fires_at([700, 700, 900, 900])]
    runner.offer(FakeFrame("cam_a", ts=6.0))
    tick(runner, now=6.0)

    events = database.recent_events()
    assert len(events) == 1
    assert events[0]["condition_state"] == "open"
    assert events[0]["observations"] == 1        # not extended either: it moved


def test_the_evidence_pointer_is_refreshed_in_place(vlm_config, database, server):
    """One evidence file per condition, not one per sighting — which for the
    laminate that started this would have been forty JPEGs of the same floor.
    The path is stable so the console can cache-bust it with `last_seen_ts`."""
    server.replies = [FIRES]
    zoned_policy(database)
    runner = make_runner(vlm_config, database, server)
    runner.refresh(now=0.0)
    runner.schedules[f"{KEY}:cam_a_apron"].next_due = 0.0
    runner.offer(FakeFrame("cam_a", ts=1000.0))
    tick(runner, now=1000.0)

    opened = database.recent_events()[0]
    snapshot = Path(vlm_config.snapshots_dir) / opened["snapshot_path"]
    assert snapshot.is_file()
    before = snapshot.stat().st_mtime_ns

    server.replies = [{"triggered": True, "confidence": 0.95,
                       "reason": "the same phone, ten minutes later", "box_2d": None}]
    runner.offer(FakeFrame("cam_a", ts=1600.0))
    tick(runner, now=1600.0)

    extended = database.recent_events()[0]
    assert extended["id"] == opened["id"]
    assert extended["snapshot_path"] == opened["snapshot_path"]      # same path
    assert snapshot.stat().st_mtime_ns != before                     # rewritten
    assert len(list(Path(vlm_config.snapshots_dir).rglob("*.jpg"))) == 1

    # The newest sentence is what the card shows; the opening one is kept,
    # because "what did it look like when this started" is the first question
    # asked of something that has been true for ten minutes.
    assert extended["description"] == "the same phone, ten minutes later"
    assert extended["confidence"] == pytest.approx(0.95)
    meta = json.loads(extended["meta"])
    assert meta["opened"]["reason"] == "a phone, still"
    assert meta["opened"]["ts"] == 1000.0
    assert meta["reason"] == extended["description"]
    # The things that describe the *policy* rather than the sighting survive
    # the extension untouched, so an evidence panel opened an hour in still
    # shows the prompt that was asked.
    assert meta["prompt"] == json.loads(opened["meta"])["prompt"]
    assert meta["zone_name"] == "Wrapper floor apron"


def test_continuity_can_be_turned_off_and_the_old_behaviour_returns(
    vlm_config, database, server
):
    """A site that wants one row per alert can have it, and the flag is what
    makes the new behaviour a decision rather than something that happened to
    them on upgrade."""
    server.replies = [FIRES]
    vlm_config.vlm.continuity.enabled = False
    zoned_policy(database)
    runner = make_runner(vlm_config, database, server)
    runner.refresh(now=0.0)
    runner.schedules[f"{KEY}:cam_a_apron"].next_due = 0.0

    for now in (0.0, 6.0, 60.0):                     # the third is past the 45 s cooldown
        runner.offer(FakeFrame("cam_a", ts=now))
        tick(runner, now=now)

    events = database.recent_events()
    assert len(events) == 2
    assert all(event["condition_key"] is None for event in events)
    assert all(event["condition_state"] is None for event in events)


def test_detector_events_are_untouched_by_any_of_this(database: Database):
    """Out of scope by design — `argus/vision/debounce.py` already holds a count
    in a polygon steady across frames, and a detector event never enters the VLM
    scheduler. It still has to *store* and *read back* cleanly beside the new
    columns, because one events table serves both shapes."""
    database.upsert_camera(Camera(id="cam_a", name="Cam A", source="/tmp/x.mp4"))
    event_id = database.insert_event(Event(
        camera_id="cam_a", detector_name="manpower_headcount", detector_type="detector",
        severity="high", description="5 people (limit 3)", ts=1000.0, stage="reviewed",
    ))
    row = database.get_event(event_id)
    assert row["condition_key"] is None
    assert row["condition_state"] is None
    assert row["last_seen_ts"] is None
    assert row["condition_updated_at"] is None
    # The default matters: it is what makes `SUM(observations)` right for a
    # detector event and for every row written before migration 7.
    assert row["observations"] == 1
    assert database.open_condition("anything") is None


# --------------------------------------------------------------------------
# the migration, against a database with rows in it
# --------------------------------------------------------------------------


def test_schema_version_constant_tracks_the_last_migration():
    """These are two hand-maintained numbers that must agree; when they drifted,
    `argus doctor` and the backup manifest both reported the wrong schema."""
    assert SCHEMA_VERSION == MIGRATIONS[-1].version


#: The columns and indexes migration 7 adds. Named once so the rewind below and
#: the assertions further down cannot drift apart.
_M7_COLUMNS = ("condition_key", "condition_state", "observations", "last_seen_ts",
               "closed_at", "close_reason", "condition_updated_at")
_M7_INDEXES = ("idx_events_condition_open", "idx_events_condition_updated")


def _rewind_to_six(database: Database) -> None:
    """Put a database back into the state a real appliance is in before upgrade.

    Needed because the fixture below prefers a *copy of this box's own
    database*, and this box's own database will be at v7 within the hour of
    this migration shipping. A test that quietly skipped from then on would
    stop proving the only thing it exists to prove: that migration 7 lands on
    twenty-five thousand real events without touching one of them.

    Indexes first — SQLite refuses to drop a column an index names.
    """
    if database.schema_version() < 7:
        return
    with database.conn() as conn:
        # Everything newer than 7 goes first. This helper undid exactly one
        # migration, which held until the box it copies moved on: migration 8
        # (`box_corrections`) reached the live database on 2026-09-20 and the
        # copy came back at v8 — "rewound" to a version that had never been six.
        # 8 is a table of its own with nothing in `events`, so dropping it is the
        # whole of undoing it; a later migration that alters an existing table
        # must add its own undo here, and the assertion below is what says so.
        conn.execute("DROP TABLE IF EXISTS box_corrections")
        conn.execute("DELETE FROM schema_migrations WHERE version = 8")
        newer = conn.execute("SELECT MAX(version) AS v FROM schema_migrations").fetchone()["v"]
        assert newer is None or newer <= 7, (
            f"_rewind_to_six does not know how to undo migration {newer}; teach it"
        )
        for index in _M7_INDEXES:
            conn.execute(f"DROP INDEX IF EXISTS {index}")          # noqa: S608 - literals
        for column in _M7_COLUMNS:
            conn.execute(f'ALTER TABLE events DROP COLUMN "{column}"')  # noqa: S608 - literals
        conn.execute("DELETE FROM schema_migrations WHERE version = 7")


@pytest.fixture
def populated_db(tmp_path: Path) -> Database:
    """A copy of this appliance's real database when there is one, else a
    hand-built stand-in with the same shape — rewound to schema 6 either way.

    Copied, never opened in place: the live engine and API are writing to the
    real file, and a test that touches it is a test that can corrupt a running
    plant's evidence. `sqlite3.backup` rather than `cp`, because the copy has to
    be consistent even though the engine is mid-write. The copy is what makes
    "does migration 7 apply to a database with twenty-five thousand events in
    it" answerable at all.
    """
    real = ROOT / "data" / "db" / "argus.sqlite3"
    target = tmp_path / "argus.sqlite3"
    if real.is_file() and real.stat().st_size > 0:
        source = sqlite3.connect(f"file:{real}?mode=ro", uri=True)
        try:
            destination = sqlite3.connect(target)
            with destination:
                source.backup(destination)
            destination.close()
        finally:
            source.close()
        database = Database(target)
        _rewind_to_six(database)
        return database

    database = Database(target)
    database.migrate()
    database.upsert_camera(Camera(id="cam_a", name="Cam A", source="/tmp/x.mp4"))
    database.upsert_policy(Policy(id="p", kind="vlm", detector_key="vlm_prompt",
                                  label="P", prompt="?"))
    for index in range(40):
        database.insert_event(Event(
            camera_id="cam_a", policy_id="p", detector_name="p", detector_type="vlm",
            severity="medium", description="laminate on the apron",
            ts=1000.0 + index * 900, stage="shadow",
        ))
    _rewind_to_six(database)
    return database


def test_migration_seven_applies_to_a_populated_database(populated_db: Database):
    before = populated_db.count_events()
    assert before > 0, "the fixture is meant to be populated"
    assert populated_db.schema_version() == 6

    applied = populated_db.migrate()

    assert 7 in applied
    assert populated_db.schema_version() == SCHEMA_VERSION
    # Nothing is backfilled and nothing is merged. Retroactively folding the 140
    # rows already on this box into conditions would rewrite history to agree
    # with a feature; the honest version is that continuity starts when it
    # shipped.
    assert populated_db.count_events() == before
    with populated_db.conn() as conn:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(events)")}
        assert set(_M7_COLUMNS) <= columns
        legacy = conn.execute(
            "SELECT COUNT(*) AS n FROM events WHERE condition_key IS NULL AND observations = 1"
        ).fetchone()["n"]
        assert legacy == before
        indexes = {row["name"] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'events'"
        )}
        assert set(_M7_INDEXES) <= indexes


def test_migration_seven_is_idempotent(populated_db: Database):
    populated_db.migrate()
    assert populated_db.migrate() == []


@pytest.mark.parametrize("query, params, index", [
    # Asked once per triggered verdict.
    ("SELECT * FROM events WHERE condition_key = ? AND condition_state = 'open' "
     "ORDER BY id DESC LIMIT 1", ("k",), "idx_events_condition_open"),
    # Asked once per SSE poll, per connected browser.
    ("SELECT * FROM events WHERE condition_updated_at > ? "
     "ORDER BY condition_updated_at LIMIT 200", (0.0,), "idx_events_condition_updated"),
    # Asked once per connection.
    ("SELECT MAX(condition_updated_at) FROM events WHERE condition_updated_at IS NOT NULL",
     (), "idx_events_condition_updated"),
])
def test_the_condition_queries_use_their_indexes(populated_db: Database, query, params, index):
    """A full scan of every event this appliance has ever raised — once per
    triggered verdict, or once per SSE poll per open browser — is the shape of
    thing that is invisible in a test and ruinous after a year of footage. Both
    indexes are *partial*, which is what makes them small and also what makes it
    easy to write a query the planner cannot match to them."""
    populated_db.migrate()
    with populated_db.conn() as conn:
        plan = " ".join(
            str(row["detail"]) for row in conn.execute(f"EXPLAIN QUERY PLAN {query}", params)
        )
    assert index in plan, plan


# --------------------------------------------------------------------------
# what the API says
# --------------------------------------------------------------------------


def open_condition_row(database: Database, **overrides) -> int:
    fields = dict(
        camera_id="line_a", zone_id="line_a_pack", policy_id="headcount_line_a",
        detector_name="headcount_line_a", detector_type="vlm", severity="high",
        description="laminate on the apron", ts=time.time() - 3600, stage="reviewed",
        condition_key="headcount_line_a:line_a:line_a_pack", condition_state="open",
    )
    fields.update(overrides)
    return database.insert_event(Event(**fields))


def test_the_stale_sweep_can_be_narrowed_to_one_origin(database: Database, fleet):
    """Two sweepers share this query and they do not share a window.

    The VLM waits fifteen minutes because its evidence arrives every few seconds
    at best; the engine waits two because its arrives five times a second
    (`detection.continuity.stale_after_sec`). Run unfiltered, the engine's
    window would retire a perfectly live prompt condition between two samples.
    """
    vlm_row = open_condition_row(database)
    detector_row = open_condition_row(
        database, detector_type="detector",
        condition_key="det:headcount_line_a:line_a:line_a_pack",
    )
    cutoff = time.time() + 1.0

    assert {row["id"] for row in database.stale_conditions(cutoff)} == {vlm_row, detector_row}
    assert [row["id"] for row in database.stale_conditions(cutoff, detector_type="detector")] \
        == [detector_row]
    assert [row["id"] for row in database.stale_conditions(cutoff, detector_type="vlm")] \
        == [vlm_row]


def test_the_events_api_serves_both_shapes(client, database: Database, fleet):
    """One table, two kinds of row. A detector event must come back exactly as
    it always did — the console renders both from the same list."""
    detector_id = make_event(database, fleet)
    condition_id = open_condition_row(database)
    database.observe_condition(condition_id, ts=time.time(), description="still there")

    events = {row["id"]: row for row in client.get("/api/events").json()["events"]}

    assert events[detector_id]["condition_key"] is None
    assert events[detector_id]["observations"] == 1
    condition = events[condition_id]
    assert condition["condition_state"] == "open"
    assert condition["observations"] == 2
    assert condition["last_seen_ts"] is not None
    assert condition["description"] == "still there"


def test_analytics_count_conditions_and_report_observations_separately(
    client, database: Database, fleet
):
    """The number the customer complained about. One laminate offcut observed
    forty times is one incident on this screen and forty observations beside
    it — before migration 7 it was forty of both, and the trend chart was
    drawing the policy's sampling interval."""
    condition_id = open_condition_row(database, ts=time.time() - 1800)
    for _ in range(39):
        database.observe_condition(condition_id, ts=time.time())

    summary = client.get("/api/analytics/summary?days=7").json()

    assert summary["totals"]["events"] == 1
    assert summary["totals"]["observations"] == 40
    assert summary["totals"]["ongoing"] == 1
    today = [day for day in summary["by_day"] if day["raw"] > 0]
    assert [(day["raw"], day["observations"]) for day in today] == [(1, 40)]
    assert [(row["raw"], row["observations"]) for row in summary["by_type"]] == [(1, 40)]
    assert [row["observations"] for row in summary["by_zone_shift"]][0]

    database.close_condition(condition_id, closed_at=time.time(), reason="cleared")
    assert client.get("/api/analytics/summary?days=7").json()["totals"]["ongoing"] == 0


def test_sse_redraws_an_extended_condition_without_a_new_row(
    live_server: str, database: Database, fleet
):
    """The calm-feed rule, over the wire. An observation arrives as an
    ``update`` frame carrying no ``id:`` — so a reconnecting browser's
    Last-Event-ID cursor cannot be dragged backwards, and every client already
    knows to merge an update over the row it holds rather than prepend it. The
    card changes; the list does not move."""
    import httpx  # noqa: F401 - imported for the same reason as in test_api

    from tests.test_api import authed, sse_frames, sse_frames_next

    with authed(live_server) as http:
        with http.stream("GET", f"{live_server}/api/events/stream") as response:
            frames = sse_frames(response)
            next(frames)                                        # hello

            condition_id = open_condition_row(database)
            opened = sse_frames_next(frames)
            assert opened["name"] == "event"
            assert opened["data"]["observations"] == 1

            database.observe_condition(condition_id, ts=time.time(),
                                       description="still there")
            update = sse_frames_next(frames)
            assert update["name"] == "update"
            assert update["data"]["id"] == condition_id
            assert update["data"]["observations"] == 2
            assert update["data"]["condition_state"] == "open"
            assert not update["raw"].startswith("id:")

            database.close_condition(condition_id, closed_at=time.time(), reason="cleared")
            closed = sse_frames_next(frames)
            assert closed["name"] == "update"
            assert closed["data"]["condition_state"] == "closed"
            assert closed["data"]["close_reason"] == "cleared"


def test_health_reports_the_continuity_settings_even_from_a_probe(config: ArgusConfig,
                                                                 database: Database):
    """`argus run` is the only process that can count conditions, so the probe
    path answers null for the counters — never 0 — while still reporting the
    settings, which are a fact about this box's config file."""
    from argus.health import health_report

    vlm = health_report(config, database)["vlm"]

    assert vlm["source"] == "probe"
    assert vlm["continuity"]["enabled"] is True
    assert vlm["continuity"]["min_iou"] == config.vlm.continuity.min_iou
    assert vlm["continuity"]["observations"] is None


def test_the_config_rejects_a_nonsense_threshold():
    from argus.config import ConfigError, VlmContinuityConfig

    with pytest.raises(ConfigError):
        VlmContinuityConfig(min_iou=1.5)
    with pytest.raises(ConfigError):
        VlmContinuityConfig(stale_after_sec=0)
