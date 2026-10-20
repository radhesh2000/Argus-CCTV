"""Retention tests.

Synthetic-but-real: actual files on disk with back-dated mtimes and actual rows
with back-dated ``created_at``, swept by the actual sweeper. The only thing
faked is the disk *percentage*, because filling a 3.7 TB volume to prove the
guard works is not a test anybody would run twice.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from argus.config import ArgusConfig
from argus.db import Camera, Database, Event, Zone
from argus.retention import Retention

DAY = 86400.0


def write_old(path: Path, age_days: float, *, size: int = 1024) -> Path:
    """A file of ``size`` bytes whose mtime is ``age_days`` in the past."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    when = time.time() - age_days * DAY
    os.utime(path, (when, when))
    return path


def insert_event(database: Database, *, age_days: float, **fields) -> int:
    created = (datetime.now(timezone.utc) - timedelta(days=age_days)).isoformat()
    return database.insert_event(Event(
        camera_id="line_a", detector_name="headcount", detector_type="detector",
        severity="high", description="over limit", ts=time.time() - age_days * DAY,
        created_at=created, **fields,
    ))


@pytest.fixture
def seeded(database: Database) -> Database:
    database.upsert_camera(Camera(id="line_a", name="Line A", source="rtsp://cam/a"))
    database.upsert_zone(Zone(id="z1", camera_id="line_a", name="Packing",
                              polygon=[[0, 0], [1, 0], [1, 1]]))
    return database


# --------------------------------------------------------------------------
# age sweep
# --------------------------------------------------------------------------


def test_old_evidence_is_deleted_and_recent_evidence_is_kept(
    config: ArgusConfig, seeded: Database
):
    old_clip = write_old(config.clips_dir / "line_a" / "event_1.mp4", 30, size=4096)
    new_clip = write_old(config.clips_dir / "line_a" / "event_2.mp4", 1)
    old_snapshot = write_old(config.snapshots_dir / "line_a" / "a.jpg", 60)
    new_snapshot = write_old(config.snapshots_dir / "line_a" / "b.jpg", 10)

    result = Retention(config, seeded, log=lambda _msg: None).sweep()

    assert not old_clip.exists()
    assert new_clip.exists()
    assert not old_snapshot.exists()
    assert new_snapshot.exists()
    assert result.files_deleted == 2
    assert result.bytes_deleted == 4096 + 1024
    assert result.by_kind["clips"] == 1
    assert result.by_kind["snapshots"] == 1
    assert not result.emergency


def test_deleting_a_clip_clears_the_row_that_pointed_at_it(
    config: ArgusConfig, seeded: Database
):
    """Events outlive their clips (180 days vs 14). A path left behind is an
    API that promises evidence and serves a 404."""
    write_old(config.clips_dir / "line_a" / "event_1.mp4", 30)
    event_id = insert_event(seeded, age_days=30, clip_path="line_a/event_1.mp4",
                            clip_start=time.time() - 30 * DAY)

    Retention(config, seeded, log=lambda _msg: None).sweep()

    event = seeded.get_event(event_id)
    assert event is not None                       # the row survives its clip
    assert event["clip_path"] is None
    assert event["clip_start"] is None


def test_snapshot_paths_are_cleared_but_clip_paths_of_other_events_are_not(
    config: ArgusConfig, seeded: Database
):
    write_old(config.snapshots_dir / "line_a" / "old.jpg", 60)
    write_old(config.snapshots_dir / "line_a" / "new.jpg", 1)
    stale = insert_event(seeded, age_days=60, snapshot_path="line_a/old.jpg")
    fresh = insert_event(seeded, age_days=1, snapshot_path="line_a/new.jpg")

    result = Retention(config, seeded, log=lambda _msg: None).sweep()

    assert result.paths_cleared == 1
    assert seeded.get_event(stale)["snapshot_path"] is None
    assert seeded.get_event(fresh)["snapshot_path"] == "line_a/new.jpg"


def test_orphan_files_with_no_row_are_still_reaped(config: ArgusConfig, seeded: Database):
    """A killed ffmpeg leaves segments nothing references. A row-driven sweep
    would never see them; the mtime walk does."""
    orphan = write_old(config.clips_dir / "line_a" / "orphan.mp4", 90)
    Retention(config, seeded, log=lambda _msg: None).sweep()
    assert not orphan.exists()


def test_ring_segments_left_by_a_crashed_run_are_swept(config: ArgusConfig, seeded: Database):
    config.evidence.ring_retention_sec = 300.0
    stale = write_old(config.ring_dir / "line_a" / "1700000000.mp4", 1)
    live = config.ring_dir / "line_a" / "1900000000.mp4"
    live.parent.mkdir(parents=True, exist_ok=True)
    live.write_bytes(b"fresh")

    Retention(config, seeded, log=lambda _msg: None).sweep()

    assert not stale.exists()
    assert live.exists()


def test_empty_camera_directories_are_removed_but_the_roots_survive(
    config: ArgusConfig, seeded: Database
):
    write_old(config.clips_dir / "line_a" / "event_1.mp4", 30)
    Retention(config, seeded, log=lambda _msg: None).sweep()
    assert not (config.clips_dir / "line_a").exists()
    # The static mount and the engine both expect the root to be there.
    assert config.clips_dir.exists()


# --------------------------------------------------------------------------
# rows
# --------------------------------------------------------------------------


def test_aged_rows_are_purged_with_their_dependents(config: ArgusConfig, seeded: Database):
    old_event = insert_event(seeded, age_days=200)
    recent_event = insert_event(seeded, age_days=5)
    seeded.record_review(old_event, "confirmed")
    seeded.log_notification(old_event, "smtp", "safety@plant.example.com", "sent")

    old_stamp = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    with seeded.conn() as conn:
        conn.execute(
            "INSERT INTO occupancy (camera_id, zone_id, ts, count, created_at) VALUES (?,?,?,?,?)",
            ("line_a", "z1", time.time() - 30 * DAY, 4, old_stamp),
        )
    seeded.insert_occupancy([{"camera_id": "line_a", "zone_id": "z1", "ts": time.time(), "count": 2}])

    result = Retention(config, seeded, log=lambda _msg: None).sweep()

    counts = seeded.counts()
    assert counts["events"] == 1
    assert seeded.get_event(recent_event) is not None
    assert seeded.get_event(old_event) is None
    # Both cascade off the deleted event.
    assert counts["reviews"] == 0
    assert counts["notifications"] == 0
    assert counts["occupancy"] == 1
    assert result.rows_deleted >= 2


def test_a_sweep_with_nothing_to_do_is_quiet_and_cheap(config: ArgusConfig, seeded: Database):
    insert_event(seeded, age_days=1)
    write_old(config.clips_dir / "line_a" / "event_1.mp4", 1)
    result = Retention(config, seeded, log=lambda _msg: None).sweep()
    assert (result.files_deleted, result.rows_deleted) == (0, 0)
    assert "nothing to reclaim" in str(result)


# --------------------------------------------------------------------------
# disk guard
# --------------------------------------------------------------------------


class FakeDisk:
    """Reports a full disk until enough files are gone, then reports relief.

    Mirrors reality: percentage does not move per file, it moves once a chunk
    of bytes has actually been freed.
    """

    def __init__(self, config: ArgusConfig, *, frees_after: int):
        self.config = config
        self.frees_after = frees_after
        self.calls = 0

    def __call__(self, _path) -> dict[str, float]:
        self.calls += 1
        remaining = sum(1 for _ in self.config.clips_dir.rglob("*.mp4"))
        percent = 95.0 if remaining > self.frees_after else 60.0
        return {"total_gb": 100.0, "used_gb": percent, "free_gb": 100.0 - percent,
                "percent": percent}


def test_disk_pressure_reclaims_oldest_first(
    config: ArgusConfig, seeded: Database, monkeypatch: pytest.MonkeyPatch
):
    config.retention.max_disk_percent = 80.0
    config.retention.min_keep_hours = 1.0
    # All well inside the 14-day clip window, so only the disk guard can touch
    # them — this proves the guard runs, not the age sweep.
    clips = [write_old(config.clips_dir / "line_a" / f"event_{i}.mp4", age_days=i / 24.0)
             for i in range(1, 9)]

    monkeypatch.setattr("argus.retention.disk_usage", FakeDisk(config, frees_after=3))
    monkeypatch.setattr("argus.retention._DISK_RECHECK_EVERY", 1)

    result = Retention(config, seeded, log=lambda _msg: None).sweep()

    assert result.emergency
    surviving = sorted(p.name for p in config.clips_dir.rglob("*.mp4"))
    # Oldest (highest index, since index == age in hours) went first.
    assert surviving == ["event_1.mp4", "event_2.mp4", "event_3.mp4"]
    assert all(not clip.exists() for clip in clips[3:])
    assert result.disk_before == 95.0
    assert result.disk_after == 60.0


def test_the_disk_guard_never_deletes_recent_evidence(
    config: ArgusConfig, seeded: Database, monkeypatch: pytest.MonkeyPatch
):
    """Freeing space by destroying the evidence for an alert nobody has looked
    at yet is the wrong trade. The guard stops and says so instead."""
    config.retention.max_disk_percent = 80.0
    config.retention.min_keep_hours = 6.0
    recent = [write_old(config.clips_dir / "line_a" / f"event_{i}.mp4", age_days=0.02)
              for i in range(3)]

    monkeypatch.setattr("argus.retention.disk_usage",
                        lambda _path: {"total_gb": 100.0, "used_gb": 95.0,
                                       "free_gb": 5.0, "percent": 95.0})

    result = Retention(config, seeded, log=lambda _msg: None).sweep()

    assert all(clip.exists() for clip in recent)
    assert result.emergency
    assert any("nothing older than" in error for error in result.errors)


def test_the_guard_stays_out_of_the_way_below_the_threshold(
    config: ArgusConfig, seeded: Database, monkeypatch: pytest.MonkeyPatch
):
    config.retention.max_disk_percent = 90.0
    clip = write_old(config.clips_dir / "line_a" / "event_1.mp4", age_days=1)
    monkeypatch.setattr("argus.retention.disk_usage",
                        lambda _path: {"total_gb": 100.0, "used_gb": 50.0,
                                       "free_gb": 50.0, "percent": 50.0})
    result = Retention(config, seeded, log=lambda _msg: None).sweep()
    assert not result.emergency
    assert clip.exists()


def test_the_disk_percent_the_guard_reads_excludes_the_reserved_blocks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """ext4 keeps 5% for root and this appliance runs unprivileged.

    `shutil.disk_usage` puts those blocks in `total` and in neither `used` nor
    `free`, so `used / total` can never exceed 94.9% on a default ext4 volume —
    a `max_disk_percent` above that would never fire, on a disk with no room
    left. Measured here 2026-08-25: 204.8 GB reserved of 4 031.9 GB.
    """
    import shutil as shutil_module

    from argus.health import disk_usage

    # 100 GB filesystem, 5 GB reserved: 90 used + 5 free available to us.
    monkeypatch.setattr(
        shutil_module, "disk_usage",
        lambda _p: shutil_module._ntuple_diskusage(100_000_000_000, 90_000_000_000,
                                                   5_000_000_000),
    )
    found = disk_usage(tmp_path)
    assert found["percent"] == pytest.approx(94.7, abs=0.1)   # not 90.0
    assert found["free_gb"] == pytest.approx(5.0, abs=0.1)


# --------------------------------------------------------------------------
# wiring
# --------------------------------------------------------------------------


def test_disabling_retention_stops_the_deleting_not_the_thread(
    config: ArgusConfig, seeded: Database
):
    """Since the housekeeping thread also takes backups and rotates logs, the
    retention switch turns off deleting alone. A site that manages storage some
    other way has not asked for its database to stop being backed up — that is
    `backup.enabled`, and it is a different line."""
    config.retention.enabled = False
    messages: list[str] = []
    retention = Retention(config, seeded, log=messages.append).start()
    assert retention._thread is not None
    assert "disabled" in messages[0]
    retention.stop()


def test_turning_every_job_off_starts_no_thread(config: ArgusConfig, seeded: Database):
    config.retention.enabled = False
    config.backup.enabled = False
    config.logs.rotate = False
    retention = Retention(config, seeded, log=lambda _m: None).start()
    assert retention._thread is None
    retention.stop()


def test_the_background_thread_sweeps_immediately_and_stops_cleanly(
    config: ArgusConfig, seeded: Database
):
    """An appliance that was off for a month must not wait another six hours
    before it notices, so the first sweep is at startup, not after a period."""
    stale = write_old(config.clips_dir / "line_a" / "event_1.mp4", 30)
    messages: list[str] = []
    retention = Retention(config, seeded, log=messages.append)
    retention.config.retention.interval_hours = 24.0
    retention.start()
    deadline = time.time() + 10
    while retention.last_result is None and time.time() < deadline:
        time.sleep(0.05)
    retention.stop(timeout=5)

    assert retention.last_result is not None
    assert not stale.exists()
    assert messages and "reclaimed" in messages[0]
    assert retention._thread is None


def test_a_failing_sweep_does_not_kill_the_thread(
    config: ArgusConfig, seeded: Database, monkeypatch: pytest.MonkeyPatch
):
    """Housekeeping must never take the engine down with it."""
    def boom(_path):
        raise OSError("disk went away")

    monkeypatch.setattr("argus.retention.disk_usage", boom)
    messages: list[str] = []
    retention = Retention(config, seeded, log=messages.append)
    retention.config.retention.interval_hours = 24.0
    retention.start()
    deadline = time.time() + 10
    while not messages and time.time() < deadline:
        time.sleep(0.05)
    retention.stop(timeout=5)
    assert messages and "sweep failed" in messages[0]


def test_clear_media_paths_refuses_a_column_that_is_not_a_path(seeded: Database):
    with pytest.raises(ValueError, match="not an evidence column"):
        seeded.clear_media_paths("description", ["anything"])


# --------------------------------------------------------------------------
# stage-tiered evidence
# --------------------------------------------------------------------------
#
# The failure these exist for, measured on the appliance 2026-08-29:
# `data/clips` at 578 GB after 2.5 days, 47 881 events, 19.4 MB a clip, and a
# flat `clips_days: 14` that gave a shadow-stage repeat of looped test footage
# exactly the same fortnight as an incident a human confirmed.


def tier_config(config: ArgusConfig, **days) -> ArgusConfig:
    """Set per-stage evidence windows on a config, leaving the rest flat."""
    for name, value in days.items():
        setattr(config.retention.evidence, f"{name}_days", float(value))
    return config


def evidence_event(database: Database, *, age_days: float, name: str,
                   stage: str = "shadow", verdict: str | None = None) -> int:
    """An event with a clip and a snapshot on disk, at a stage and a verdict."""
    event_id = insert_event(
        database, age_days=age_days, stage=stage,
        clip_path=f"line_a/{name}.mp4", snapshot_path=f"line_a/{name}.jpg",
        clip_start=time.time() - age_days * DAY,
    )
    if verdict is not None:
        database.record_review(event_id, verdict)
    return event_id


def test_a_shadow_clip_ages_out_while_a_confirmed_one_of_the_same_age_survives(
    config: ArgusConfig, seeded: Database
):
    """The whole point. Both clips are 5 days old and inside the flat 14-day
    window; only what the events turned out to be separates them."""
    tier_config(config, shadow=1, reviewed=2, confirmed=30, false_positive=30)
    noise = write_old(config.clips_dir / "line_a" / "noise.mp4", 5)
    incident = write_old(config.clips_dir / "line_a" / "incident.mp4", 5)
    evidence_event(seeded, age_days=5, name="noise", stage="shadow")
    kept = evidence_event(seeded, age_days=5, name="incident",
                          stage="reviewed", verdict="confirmed")

    result = Retention(config, seeded, log=lambda _msg: None).sweep()

    assert not noise.exists()
    assert incident.exists()
    assert result.by_kind["clips"] == 1
    # And the console renders evidence-aged-out rather than a 404, because the
    # row survived and its pointer did not.
    assert seeded.get_event(kept)["clip_path"] == "line_a/incident.mp4"


def test_a_human_verdict_beats_the_policy_stage(config: ArgusConfig, seeded: Database):
    """A verdict is a judgement about this event; a stage is a promise about the
    policy that raised it. Two reviewed-stage events, opposite outcomes, and the
    windows follow the verdict."""
    tier_config(config, shadow=1, reviewed=1, confirmed=30, false_positive=30)
    real = write_old(config.clips_dir / "line_a" / "real.mp4", 5)
    debunked = write_old(config.clips_dir / "line_a" / "debunked.mp4", 5)
    unjudged = write_old(config.clips_dir / "line_a" / "unjudged.mp4", 5)
    evidence_event(seeded, age_days=5, name="real", stage="reviewed", verdict="confirmed")
    evidence_event(seeded, age_days=5, name="debunked", stage="reviewed",
                   verdict="false_positive")
    evidence_event(seeded, age_days=5, name="unjudged", stage="reviewed")

    Retention(config, seeded, log=lambda _msg: None).sweep()

    assert real.exists()
    # A labelled negative is the one eval input a deployment produces for free —
    # argus/eval/dataset.py reads exactly these. It is not the shortest window.
    assert debunked.exists()
    assert not unjudged.exists()


def test_auto_routed_inherits_the_reviewed_window_unless_it_is_set(
    config: ArgusConfig, seeded: Database
):
    """On a mature site every alert is auto_routed. Folding it into
    `reviewed_days` silently would hand live alerting a window nobody chose, so
    it inherits visibly and overrides explicitly."""
    tier_config(config, shadow=1, reviewed=2)
    inherited = write_old(config.clips_dir / "line_a" / "inherited.mp4", 5)
    evidence_event(seeded, age_days=5, name="inherited", stage="auto_routed")
    Retention(config, seeded, log=lambda _msg: None).sweep()
    assert not inherited.exists()          # took reviewed_days = 2

    kept = write_old(config.clips_dir / "line_a" / "kept.mp4", 5)
    evidence_event(seeded, age_days=5, name="kept", stage="auto_routed")
    config.retention.evidence.auto_routed_days = 30.0
    Retention(config, seeded, log=lambda _msg: None).sweep()
    assert kept.exists()


def test_tiering_applies_to_snapshots_as_well_as_clips(
    config: ArgusConfig, seeded: Database
):
    tier_config(config, shadow=1, confirmed=30)
    noise = write_old(config.snapshots_dir / "line_a" / "noise.jpg", 5)
    incident = write_old(config.snapshots_dir / "line_a" / "incident.jpg", 5)
    evidence_event(seeded, age_days=5, name="noise", stage="shadow")
    evidence_event(seeded, age_days=5, name="incident", stage="shadow", verdict="confirmed")

    Retention(config, seeded, log=lambda _msg: None).sweep()

    assert not noise.exists()
    assert incident.exists()


def test_an_orphan_gets_the_flat_window_under_tiering(config: ArgusConfig, seeded: Database):
    """A file no row points at cannot have a tier. It keeps exactly the
    behaviour it had before tiers existed — the mtime walk still sees it, and
    `clips_days` still bounds it."""
    tier_config(config, shadow=1, confirmed=30)
    young_orphan = write_old(config.clips_dir / "line_a" / "orphan_new.mp4", 5)
    old_orphan = write_old(config.clips_dir / "line_a" / "orphan_old.mp4", 30)

    Retention(config, seeded, log=lambda _msg: None).sweep()

    assert young_orphan.exists()           # 5 d < clips_days 14, not shadow_days 1
    assert not old_orphan.exists()


def test_a_config_with_no_evidence_block_keeps_the_flat_windows(
    config: ArgusConfig, seeded: Database
):
    """The upgrade contract: a site that never writes the new block must not
    discover that its appliance started deleting evidence on a shorter clock."""
    assert not config.retention.evidence.configured
    clip = write_old(config.clips_dir / "line_a" / "shadow.mp4", 5)
    evidence_event(seeded, age_days=5, name="shadow", stage="shadow")

    Retention(config, seeded, log=lambda _msg: None).sweep()

    assert clip.exists()                   # 5 d is inside the flat 14


def test_the_tier_windows_need_no_schema_change(config: ArgusConfig, seeded: Database):
    """`stage` and `verdict` have been columns since migration 0, so tiering is
    a join and not a migration — which matters because an appliance that had to
    migrate before it could stop filling its disk would fill its disk first."""
    from argus.db import SCHEMA_VERSION, Database as Db

    for i in range(20):
        evidence_event(seeded, age_days=i / 4.0, name=f"e{i}",
                       stage="reviewed" if i % 2 else "shadow",
                       verdict="confirmed" if i % 5 == 0 else None)
    before = seeded.schema_version()
    reopened = Db(config.db_path)
    reopened.migrate()
    assert reopened.schema_version() == before == SCHEMA_VERSION

    tiers = reopened.media_tiers("clip_path")
    assert len(tiers) == 20
    assert tiers["line_a/e0.mp4"] == "confirmed"      # verdict wins over stage
    assert tiers["line_a/e1.mp4"] == "reviewed"
    assert tiers["line_a/e2.mp4"] == "shadow"


def test_media_tiers_refuses_a_column_that_is_not_evidence(seeded: Database):
    with pytest.raises(ValueError, match="not an evidence column"):
        seeded.media_tiers("description")


# --------------------------------------------------------------------------
# the hard budget
# --------------------------------------------------------------------------


def test_the_budget_evicts_lowest_stage_first_then_oldest(
    config: ArgusConfig, seeded: Database
):
    """Every clip here is one hour old and well inside every age window, so
    only the budget can touch them — and what it takes is decided by what the
    events turned out to be, oldest first within a tier."""
    tier_config(config, shadow=1, reviewed=2, confirmed=30, false_positive=30)
    config.retention.max_clips_gb = 30 / 1000.0        # 30 MB
    config.retention.min_keep_hours = 0.0
    # 10 MB each: five of them is 50 MB against a 30 MB budget, so two must go.
    for index, (name, stage, verdict) in enumerate([
        ("shadow_old", "shadow", None),
        ("shadow_new", "shadow", None),
        ("reviewed", "reviewed", None),
        ("confirmed", "reviewed", "confirmed"),
        ("false_positive", "reviewed", "false_positive"),
    ]):
        write_old(config.clips_dir / "line_a" / f"{name}.mp4",
                  age_days=(5 - index) / 24.0, size=10_000_000)
        evidence_event(seeded, age_days=(5 - index) / 24.0, name=name,
                       stage=stage, verdict=verdict)

    result = Retention(config, seeded, log=lambda _msg: None).sweep()

    surviving = sorted(p.stem for p in config.clips_dir.rglob("*.mp4"))
    assert surviving == ["confirmed", "false_positive", "reviewed"]
    assert result.by_kind["clips_over_budget"] == 2
    budget = result.budgets["clips"]
    assert budget["by_tier"] == {"shadow": 2}
    assert budget["under_budget"] is True
    assert budget["used_gb_before"] == 0.05


def test_the_budget_clears_the_rows_of_what_it_evicted(
    config: ArgusConfig, seeded: Database
):
    """Same contract as the age sweep: the row survives, the pointer does not,
    so the console renders evidence-aged-out instead of a broken player."""
    config.retention.max_clips_gb = 1 / 1000.0
    config.retention.min_keep_hours = 0.0
    write_old(config.clips_dir / "line_a" / "gone.mp4", age_days=0.1, size=4_000_000)
    event_id = evidence_event(seeded, age_days=0.1, name="gone", stage="shadow")

    result = Retention(config, seeded, log=lambda _msg: None).sweep()

    event = seeded.get_event(event_id)
    assert event is not None
    assert event["clip_path"] is None
    assert event["clip_start"] is None
    assert result.paths_cleared == 1


def test_the_budget_evicts_orphans_before_any_tier(config: ArgusConfig, seeded: Database):
    """Evidence for nothing is the cheapest thing in the directory, whatever its
    mtime says — so it goes before the newest shadow clip."""
    tier_config(config, shadow=1, confirmed=30)
    config.retention.max_clips_gb = 10 / 1000.0
    config.retention.min_keep_hours = 0.0
    write_old(config.clips_dir / "line_a" / "orphan.mp4", age_days=0.01, size=10_000_000)
    write_old(config.clips_dir / "line_a" / "shadow.mp4", age_days=0.5, size=10_000_000)
    evidence_event(seeded, age_days=0.5, name="shadow", stage="shadow")

    Retention(config, seeded, log=lambda _msg: None).sweep()

    assert not (config.clips_dir / "line_a" / "orphan.mp4").exists()
    assert (config.clips_dir / "line_a" / "shadow.mp4").exists()


def test_the_budget_never_deletes_evidence_younger_than_min_keep_hours(
    config: ArgusConfig, seeded: Database
):
    """A budget is not a licence to destroy the clip for an alert raised twenty
    minutes ago. It stops, says how much it could not reclaim and why."""
    config.retention.max_clips_gb = 1 / 1000.0
    config.retention.min_keep_hours = 6.0
    for index in range(3):
        write_old(config.clips_dir / "line_a" / f"fresh{index}.mp4",
                  age_days=0.02, size=4_000_000)

    messages: list[str] = []
    result = Retention(config, seeded, log=messages.append).sweep()

    assert len(list(config.clips_dir.rglob("*.mp4"))) == 3
    assert result.budgets["clips"]["under_budget"] is False
    assert result.budgets["clips"]["files_evicted"] == 0
    assert any("min_keep_hours" in error for error in result.errors)
    assert any("min_keep_hours" in message for message in messages)


def test_the_budget_logs_what_it_reclaimed_and_why(config: ArgusConfig, seeded: Database):
    config.retention.max_clips_gb = 5 / 1000.0
    config.retention.min_keep_hours = 0.0
    for index in range(4):
        write_old(config.clips_dir / "line_a" / f"c{index}.mp4",
                  age_days=(index + 1) / 24.0, size=4_000_000)

    messages: list[str] = []
    Retention(config, seeded, log=messages.append).sweep()

    line = next(m for m in messages if m.startswith("retention[budget]"))
    assert "clips was 0.02 GB against a 0.01 GB budget" in line
    assert "least-valuable-first" in line
    assert "orphan=3" in line


def test_no_budget_configured_deletes_nothing_extra(config: ArgusConfig, seeded: Database):
    """0 means no budget, and 0 is the dataclass default: an appliance upgraded
    into this code does not invent a ceiling for a disk it has never seen."""
    assert config.retention.max_clips_gb == 0.0
    for index in range(5):
        write_old(config.clips_dir / "line_a" / f"c{index}.mp4",
                  age_days=1, size=10_000_000)
    result = Retention(config, seeded, log=lambda _msg: None).sweep()
    assert len(list(config.clips_dir.rglob("*.mp4"))) == 5
    assert result.budgets == {}


def test_the_budget_only_counts_what_the_age_sweep_left(
    config: ArgusConfig, seeded: Database
):
    """Ages run first, so the budget must not claim credit for their work — and
    must not evict anything once they have already brought it under."""
    tier_config(config, shadow=1)
    config.retention.max_clips_gb = 15 / 1000.0
    config.retention.min_keep_hours = 0.0
    for index in range(3):                       # 30 MB of expired shadow clips
        write_old(config.clips_dir / "line_a" / f"old{index}.mp4",
                  age_days=3, size=10_000_000)
        evidence_event(seeded, age_days=3, name=f"old{index}", stage="shadow")
    write_old(config.clips_dir / "line_a" / "recent.mp4", age_days=0.5, size=10_000_000)
    evidence_event(seeded, age_days=0.5, name="recent", stage="shadow")

    result = Retention(config, seeded, log=lambda _msg: None).sweep()

    assert result.by_kind["clips"] == 3          # the ages took the three
    assert "clips_over_budget" not in result.by_kind
    assert result.budgets == {}                  # 10 MB left, under the 15 MB budget
    assert (config.clips_dir / "line_a" / "recent.mp4").exists()


def test_snapshots_have_their_own_budget_so_clips_cannot_crowd_them_out(
    config: ArgusConfig, seeded: Database
):
    """A clip is 126x a snapshot on this box. Under one pooled budget the stills
    that render in the triage list would be evicted to buy back the space of a
    single clip; sized separately, each kind is bounded on its own terms."""
    config.retention.max_clips_gb = 5 / 1000.0
    config.retention.max_snapshots_gb = 1.0
    config.retention.min_keep_hours = 0.0
    for index in range(3):
        write_old(config.clips_dir / "line_a" / f"c{index}.mp4",
                  age_days=(index + 1) / 24.0, size=10_000_000)
    stills = [write_old(config.snapshots_dir / "line_a" / f"s{index}.jpg",
                        age_days=(index + 1) / 24.0, size=150_000)
              for index in range(3)]

    result = Retention(config, seeded, log=lambda _msg: None).sweep()

    assert "clips" in result.budgets
    assert "snapshots" not in result.budgets
    assert all(still.exists() for still in stills)


# --------------------------------------------------------------------------
# the plan (`argus retention --dry-run`)
# --------------------------------------------------------------------------


def test_the_dry_run_plan_reports_per_tier_and_the_budget_eviction(
    config: ArgusConfig, seeded: Database
):
    tier_config(config, shadow=1, reviewed=2, confirmed=30, false_positive=30)
    config.retention.max_clips_gb = 10 / 1000.0
    config.retention.min_keep_hours = 0.0
    # Two expired shadow clips (the ages take these), plus 30 MB of live
    # evidence against a 10 MB budget (the budget takes 20 MB of that).
    for index in range(2):
        write_old(config.clips_dir / "line_a" / f"expired{index}.mp4",
                  age_days=3, size=1_000_000)
        evidence_event(seeded, age_days=3, name=f"expired{index}", stage="shadow")
    for index in range(3):
        write_old(config.clips_dir / "line_a" / f"live{index}.mp4",
                  age_days=(index + 1) / 24.0, size=10_000_000)
        evidence_event(seeded, age_days=(index + 1) / 24.0, name=f"live{index}",
                       stage="shadow")

    plan = Retention(config, seeded, log=lambda _msg: None).plan()
    clips = next(media for media in plan["media"] if media["kind"] == "clips")

    assert clips["tiered"] is True
    assert clips["budget_gb"] == 0.01
    assert clips["by_tier"]["shadow"] == {"days": 1.0, "files": 5, "gb": 0.03, "expiring": 2}
    assert clips["after_ages_gb"] == 0.03        # the 2 MB of expired is gone
    assert clips["over_budget"] is True
    assert clips["evict_files"] == 2             # 30 MB -> 10 MB
    assert clips["evict_gb"] == 0.02
    # And nothing was actually deleted.
    assert len(list(config.clips_dir.rglob("*.mp4"))) == 5


def test_the_plan_says_flat_when_no_tiers_are_configured(
    config: ArgusConfig, seeded: Database
):
    write_old(config.clips_dir / "line_a" / "c.mp4", age_days=1)
    plan = Retention(config, seeded, log=lambda _msg: None).plan()
    clips = next(media for media in plan["media"] if media["kind"] == "clips")
    assert clips["tiered"] is False
    assert clips["budget_gb"] == 0.0
    assert clips["by_tier"] == {"orphan": {"days": 14.0, "files": 1, "gb": 0.0,
                                           "expiring": 0}}


# --------------------------------------------------------------------------
# surfacing it: /api/health, `argus doctor`
# --------------------------------------------------------------------------


def test_health_reports_evidence_against_its_budget(config: ArgusConfig, seeded: Database):
    """The number the disk percentage cannot give you: this box sat at 36% full
    while `data/clips` grew 250 GB a day."""
    from argus.health import evidence_usage

    config.retention.max_clips_gb = 0.01
    write_old(config.clips_dir / "line_a" / "c.mp4", age_days=0.1, size=5_000_000)
    write_old(config.snapshots_dir / "line_a" / "s.jpg", age_days=0.1, size=150_000)

    evidence = evidence_usage(config)

    assert evidence["kinds"]["clips"] == {
        "used_gb": 0.01, "budget_gb": 0.01, "percent": 100.0,
        "over_budget": False, "at_budget": True,
    }
    # No budget configured is reported as unknown, not as 0% or 100%: a console
    # that guessed either would be telling an operator something the appliance
    # does not know.
    assert evidence["kinds"]["snapshots"]["budget_gb"] is None
    assert evidence["kinds"]["snapshots"]["percent"] is None
    assert evidence["tiered"] is False


def test_the_doctor_warns_when_evidence_is_over_budget(config: ArgusConfig, database: Database):
    from argus import doctor

    config.retention.max_clips_gb = 0.001
    config.retention.max_snapshots_gb = 10.0
    write_old(config.clips_dir / "line_a" / "c.mp4", age_days=0.1, size=5_000_000)

    check = next(c for c in doctor.check_disk(config) if c.name == "evidence budget")
    assert check.status == doctor.WARN
    assert "over budget" in check.detail
    assert check.fix == "argus retention"


def test_the_doctor_warns_when_nothing_bounds_evidence_at_all(
    config: ArgusConfig, database: Database
):
    """An appliance with no ceiling is the bug this work exists to fix, so the
    support tool says so before a site discovers it at 578 GB."""
    from argus import doctor

    check = next(c for c in doctor.check_disk(config) if c.name == "evidence budget")
    assert check.status == doctor.WARN
    assert "no hard ceiling" in check.detail
    assert "max_clips_gb" in (check.fix or "")


def test_the_doctor_is_quiet_when_the_budget_is_configured_and_held(
    config: ArgusConfig, database: Database
):
    from argus import doctor

    config.retention.max_clips_gb = 100.0
    config.retention.max_snapshots_gb = 20.0
    check = next(c for c in doctor.check_disk(config) if c.name == "evidence budget")
    assert check.status == doctor.OK
    assert "flat windows" in check.detail          # this config sets no tiers

    tier_config(config, shadow=1, confirmed=30)
    check = next(c for c in doctor.check_disk(config) if c.name == "evidence budget")
    assert "per-stage windows in force" in check.detail


def test_sitting_exactly_at_the_budget_is_not_a_warning(config: ArgusConfig,
                                                        database: Database):
    """The sweep reclaims to *exactly* the budget and then the cameras keep
    recording, so a busy site lives at 100% of its ceiling. A doctor that called
    that a fault would be amber forever on a working appliance — and an amber
    that is always on is one nobody reads."""
    from argus import doctor
    from argus.health import evidence_usage

    config.retention.max_clips_gb = 0.01
    config.retention.max_snapshots_gb = 10.0
    write_old(config.clips_dir / "line_a" / "c.mp4", age_days=0.1, size=10_200_000)

    evidence = evidence_usage(config)
    assert evidence["kinds"]["clips"]["at_budget"] is True
    assert evidence["kinds"]["clips"]["over_budget"] is False

    check = next(c for c in doctor.check_disk(config) if c.name == "evidence budget")
    assert check.status == doctor.OK
    assert "held at the ceiling" in check.detail


def test_the_sweep_leaves_the_directory_markers_alone(config: ArgusConfig, seeded: Database):
    """`data/clips/.gitkeep` is how the repository ships an empty evidence tree,
    and to an mtime walk it looks exactly like an orphan: old, and referenced by
    no event. The first sweep on this box that deleted anything at scale deleted
    it. Nothing Argus writes starts with a dot, so nothing here should."""
    marker = config.clips_dir / ".gitkeep"
    marker.write_bytes(b"")
    ancient = time.time() - 365 * DAY
    os.utime(marker, (ancient, ancient))
    config.retention.max_clips_gb = 1 / 1_000_000.0      # 1 KB: everything is over
    config.retention.min_keep_hours = 0.0
    write_old(config.clips_dir / "line_a" / "c.mp4", age_days=1, size=100_000)

    result = Retention(config, seeded, log=lambda _msg: None).sweep()

    assert marker.exists()
    assert not (config.clips_dir / "line_a" / "c.mp4").exists()   # the budget ran
    # And it is not counted as evidence the budget failed to reclaim, either.
    assert result.budgets["clips"]["by_tier"] == {"orphan": 1}
