from __future__ import annotations

import sqlite3
import time

import pytest

from argus.db import (
    MIGRATIONS,
    _UPDATABLE,
    Camera,
    Database,
    Event,
    Policy,
    Zone,
    source_kind,
)


# --------------------------------------------------------------------------
# migrations
# --------------------------------------------------------------------------


def test_migrate_creates_every_table(database: Database):
    with database.conn() as conn:
        names = {
            r["name"]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    assert {"cameras", "zones", "policies", "events", "reviews",
            "notifications", "occupancy", "schema_migrations"} <= names


def test_migrate_is_idempotent(config):
    db = Database(config.db_path)
    first = db.migrate()
    assert first == [m.version for m in MIGRATIONS]
    assert db.migrate() == []          # nothing left to apply
    assert db.schema_version() == MIGRATIONS[-1].version


def test_schema_version_is_zero_before_migrating(tmp_path):
    assert Database(tmp_path / "fresh.sqlite3").schema_version() == 0


def test_migration_versions_are_unique_and_ordered():
    versions = [m.version for m in MIGRATIONS]
    assert versions == sorted(versions)
    assert len(set(versions)) == len(versions)


def test_wal_and_pragmas_are_set(database: Database):
    with database.conn() as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] > 0
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_updatable_allowlists_match_the_real_schema(database: Database):
    """Guards the one place identifiers are interpolated into SQL: if a future
    migration renames a column, this fails instead of silently allowing a name
    that no longer exists."""
    with database.conn() as conn:
        for table, allowed in _UPDATABLE.items():
            actual = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
            assert allowed <= actual, f"{table}: stale allowlist entries {allowed - actual}"


# --------------------------------------------------------------------------
# cameras + zones
# --------------------------------------------------------------------------


def test_source_kind_classification():
    assert source_kind("rtsp://cam.local/stream") == "rtsp"
    assert source_kind("http://cam.local/stream.m3u8") == "http"
    assert source_kind("/data/clip.mp4") == "file"
    assert source_kind("./packing_section_recording.mp4") == "file"


def test_camera_upsert_is_idempotent(database: Database):
    camera = Camera(id="cam1", name="Line 1", source="/videos/a.mp4")
    database.upsert_camera(camera)
    database.upsert_camera(camera)
    assert len(database.list_cameras()) == 1

    database.upsert_camera(Camera(id="cam1", name="Line 1 renamed", source="rtsp://x/y"))
    row = database.get_camera("cam1")
    assert row["name"] == "Line 1 renamed"
    assert row["kind"] == "rtsp"          # re-derived from the new source
    assert len(database.list_cameras()) == 1


def test_camera_status_transitions(database: Database):
    database.upsert_camera(Camera(id="cam1", name="Line 1", source="/videos/a.mp4"))
    assert database.get_camera("cam1")["status"] == "offline"

    database.set_camera_status("cam1", "connecting")
    assert database.get_camera("cam1")["last_seen_at"] is None

    database.set_camera_status("cam1", "live")
    row = database.get_camera("cam1")
    assert row["status"] == "live" and row["last_seen_at"] is not None

    database.set_camera_status("cam1", "error", "connection refused")
    assert database.get_camera("cam1")["status_detail"] == "connection refused"

    with pytest.raises(ValueError):
        database.set_camera_status("cam1", "on fire")


def test_enabled_only_filter(database: Database):
    database.upsert_camera(Camera(id="on", name="On", source="/a.mp4", enabled=True))
    database.upsert_camera(Camera(id="off", name="Off", source="/b.mp4", enabled=False))
    assert [c["id"] for c in database.list_cameras(enabled_only=True)] == ["on"]
    assert len(database.list_cameras()) == 2


def test_create_camera_generates_unique_slug_ids(database: Database):
    first = database.create_camera("Packing line", "/a.mp4")
    second = database.create_camera("Packing line", "/b.mp4")
    assert first == "packing_line"
    assert second == "packing_line_2"


def test_zone_polygon_round_trips_as_json(database: Database):
    database.upsert_camera(Camera(id="cam1", name="c", source="/a.mp4"))
    polygon = [[0.1, 0.2], [0.9, 0.2], [0.9, 0.8]]
    database.upsert_zone(Zone(id="z1", camera_id="cam1", name="Floor", polygon=polygon,
                              sanctioned_limit=3))
    zone = database.get_zone("z1")
    assert zone["polygon"] == polygon
    assert zone["sanctioned_limit"] == 3
    assert database.list_zones("cam1")[0]["polygon"] == polygon


def test_zone_update_accepts_points_or_json(database: Database):
    database.upsert_camera(Camera(id="cam1", name="c", source="/a.mp4"))
    database.upsert_zone(Zone(id="z1", camera_id="cam1", name="Floor",
                              polygon=[[0, 0], [1, 0], [1, 1]]))
    database.update_zone("z1", polygon=[[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]], sanctioned_limit=9)
    zone = database.get_zone("z1")
    assert zone["polygon"] == [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]]
    assert zone["sanctioned_limit"] == 9


def test_deleting_a_camera_cascades_to_zones(database: Database):
    database.upsert_camera(Camera(id="cam1", name="c", source="/a.mp4"))
    database.upsert_zone(Zone(id="z1", camera_id="cam1", name="Floor",
                              polygon=[[0, 0], [1, 0], [1, 1]]))
    database.delete_camera("cam1")
    assert database.list_zones() == []


# --------------------------------------------------------------------------
# policies + lifecycle
# --------------------------------------------------------------------------


def _policy(**overrides) -> Policy:
    base = dict(id="p1", kind="vlm", detector_key="vlm_prompt", label="Exit blocked")
    base.update(overrides)
    return Policy(**base)


def test_policy_upsert_and_active_filter(database: Database):
    database.upsert_policy(_policy(id="draft_p", lifecycle="draft"))
    database.upsert_policy(_policy(id="pilot_p", lifecycle="pilot"))
    database.upsert_policy(_policy(id="prod_p", lifecycle="production"))
    database.upsert_policy(_policy(id="off_p", lifecycle="production", enabled=False))

    active = {p["id"] for p in database.list_policies(active_only=True)}
    assert active == {"pilot_p", "prod_p", "off_p"}       # draft never runs automatically
    enabled = {p["id"] for p in database.list_policies(active_only=True, enabled_only=True)}
    assert enabled == {"pilot_p", "prod_p"}


def test_policy_lifecycle_draft_to_pilot_to_production(database: Database):
    database.upsert_policy(_policy(lifecycle="draft"))

    assert database.start_pilot("p1") == 1
    row = database.get_policy("p1")
    assert row["lifecycle"] == "pilot" and row["pilot_started_at"] is not None

    # starting a pilot twice is a no-op: the guard is in the WHERE clause
    assert database.start_pilot("p1") == 0

    assert database.extend_pilot("p1", 3.0) == 1
    assert database.get_policy("p1")["pilot_duration_days"] == pytest.approx(5.0)

    assert database.promote_policy("p1", routed_to="safety_manager") == 1
    row = database.get_policy("p1")
    assert row["lifecycle"] == "production"
    assert row["stage"] == "auto_routed"
    assert row["routed_to"] == "safety_manager"
    assert row["promoted_at"] is not None


def test_promote_keeps_existing_route_when_none_given(database: Database):
    database.upsert_policy(_policy(lifecycle="pilot", routed_to="line_supervisor"))
    database.promote_policy("p1")
    assert database.get_policy("p1")["routed_to"] == "line_supervisor"


def test_invalid_enum_values_are_rejected_by_the_schema(database: Database):
    with pytest.raises(sqlite3.IntegrityError):
        database.upsert_policy(_policy(lifecycle="whenever"))
    with pytest.raises(sqlite3.IntegrityError):
        database.upsert_policy(_policy(stage="maybe"))


def test_update_rejects_unknown_columns(database: Database):
    """The v1 f-string SET clause would have happily written this."""
    database.upsert_policy(_policy())
    with pytest.raises(ValueError, match="not updatable"):
        database.update_policy("p1", lifecycle_typo="pilot")
    with pytest.raises(ValueError, match="not updatable"):
        database.update_policy("p1", **{"label = 'x' --": "y"})


# --------------------------------------------------------------------------
# events, reviews, accuracy
# --------------------------------------------------------------------------


@pytest.fixture
def seeded(database: Database) -> Database:
    database.upsert_camera(Camera(id="cam1", name="Line 1", source="/a.mp4"))
    database.upsert_zone(Zone(id="z1", camera_id="cam1", name="Floor",
                              polygon=[[0, 0], [1, 0], [1, 1]], sanctioned_limit=3))
    database.upsert_policy(_policy(id="manpower_headcount", kind="detector",
                                   detector_key="manpower_headcount", lifecycle="production",
                                   stage="reviewed"))
    return database


def _event(**overrides) -> Event:
    base = dict(
        camera_id="cam1", detector_name="manpower_headcount", detector_type="detector",
        severity="medium", description="5 people in Floor (limit 3)", ts=time.time(),
        stage="reviewed", confidence=0.8, zone_id="z1", policy_id="manpower_headcount",
    )
    base.update(overrides)
    return Event(**base)


def test_insert_and_read_event(seeded: Database):
    event_id = seeded.insert_event(_event(meta={"count": 5, "limit": 3}))
    row = seeded.get_event(event_id)
    assert row["detector_name"] == "manpower_headcount"
    assert row["zone_id"] == "z1"
    assert row["review_verdict"] is None
    assert '"count": 5' in row["meta"]


def test_event_rejects_bad_stage_and_severity(seeded: Database):
    with pytest.raises(ValueError, match="unknown stage"):
        seeded.insert_event(_event(stage="published"))
    with pytest.raises(ValueError, match="unknown severity"):
        seeded.insert_event(_event(severity="catastrophic"))


def test_events_since_and_max_id(seeded: Database):
    ids = [seeded.insert_event(_event()) for _ in range(3)]
    assert seeded.max_event_id() == ids[-1]
    assert [e["id"] for e in seeded.events_since(ids[0])] == ids[1:]


def test_recent_events_filters(seeded: Database):
    seeded.insert_event(_event(severity="high"))
    seeded.insert_event(_event(severity="low", stage="shadow"))
    assert len(seeded.recent_events(severity="high")) == 1
    assert len(seeded.recent_events(stage="shadow")) == 1
    assert len(seeded.recent_events(camera_id="cam1")) == 2
    assert len(seeded.recent_events(camera_id="nope")) == 0


def test_triage_queue_excludes_shadow_and_reviewed(seeded: Database):
    reviewed_id = seeded.insert_event(_event(stage="reviewed"))
    seeded.insert_event(_event(stage="shadow"))          # shadow never reaches a human
    pending_id = seeded.insert_event(_event(stage="auto_routed"))

    assert {e["id"] for e in seeded.unreviewed_events()} == {reviewed_id, pending_id}
    seeded.record_review(reviewed_id, "confirmed", reviewer="arjun")
    assert {e["id"] for e in seeded.unreviewed_events()} == {pending_id}


def test_review_is_one_per_event_and_latest_wins(seeded: Database):
    event_id = seeded.insert_event(_event())
    seeded.record_review(event_id, "confirmed", reviewer="a", note="real")
    seeded.record_review(event_id, "false_positive", reviewer="b", note="shadow on floor")

    review = seeded.get_review(event_id)
    assert review["verdict"] == "false_positive"
    assert review["reviewer"] == "b"
    with seeded.conn() as conn:
        assert conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0] == 1


def test_unknown_verdict_rejected(seeded: Database):
    event_id = seeded.insert_event(_event())
    with pytest.raises(ValueError, match="unknown verdict"):
        seeded.record_review(event_id, "probably")


def test_detector_accuracy_from_verdicts(seeded: Database):
    for _ in range(3):
        seeded.record_review(seeded.insert_event(_event()), "confirmed")
    seeded.record_review(seeded.insert_event(_event()), "false_positive")
    seeded.insert_event(_event())                       # pending
    seeded.insert_event(_event(stage="shadow"))         # shadow: not pending review

    stats = {row["detector_name"]: row for row in seeded.detector_accuracy()}["manpower_headcount"]
    assert stats["total"] == 6
    assert stats["confirmed"] == 3
    assert stats["false_positive"] == 1
    assert stats["reviewed"] == 4
    assert stats["pending"] == 1
    assert stats["precision"] == pytest.approx(0.75)
    assert stats["fp_rate"] == pytest.approx(0.25)


def test_accuracy_is_none_when_nothing_reviewed(seeded: Database):
    seeded.insert_event(_event())
    stats = seeded.detector_accuracy()[0]
    assert stats["precision"] is None and stats["fp_rate"] is None


def test_policy_analytics(seeded: Database):
    seeded.record_review(seeded.insert_event(_event(severity="high")), "false_positive")
    seeded.insert_event(_event(severity="medium"))
    analytics = seeded.policy_analytics("manpower_headcount")
    assert analytics["total"] == 2
    assert analytics["by_severity"] == {"high": 1, "medium": 1}
    assert analytics["reviewed_total"] == 1
    assert analytics["unreviewed"] == 1
    assert analytics["false_positive_rate"] == pytest.approx(1.0)
    assert len(analytics["by_day"]) == 1


def test_review_trend(seeded: Database):
    seeded.record_review(seeded.insert_event(_event()), "confirmed")
    seeded.record_review(seeded.insert_event(_event()), "false_positive")
    trend = seeded.review_trend()
    assert len(trend) == 1
    assert trend[0]["confirmed"] == 1 and trend[0]["false_positive"] == 1


def test_reviewed_events_for_export_carries_evidence(seeded: Database):
    event_id = seeded.insert_event(_event(snapshot_path="/snap.jpg", clip_path="/clip.mp4"))
    seeded.record_review(event_id, "false_positive", note="reflection")
    seeded.insert_event(_event())  # unreviewed: excluded

    exported = seeded.reviewed_events_for_export()
    assert len(exported) == 1
    assert exported[0]["snapshot_path"] == "/snap.jpg"
    assert exported[0]["verdict"] == "false_positive"
    assert exported[0]["note"] == "reflection"
    assert seeded.reviewed_events_for_export(detector="other") == []


# --------------------------------------------------------------------------
# notifications + occupancy + retention
# --------------------------------------------------------------------------


def test_notification_log(seeded: Database):
    event_id = seeded.insert_event(_event())
    seeded.log_notification(event_id, "smtp", "safety@plant.example.com", "sent")
    seeded.log_notification(event_id, "webhook", "https://erp/x", "failed", "HTTP 500")
    rows = seeded.notifications_for_event(event_id)
    assert [r["status"] for r in rows] == ["sent", "failed"]
    assert seeded.recent_notifications()[0]["detector_name"] == "manpower_headcount"
    with pytest.raises(ValueError):
        seeded.log_notification(event_id, "smtp", "x", "maybe")


def test_occupancy_series_and_latest_counts(seeded: Database):
    now = time.time()
    assert seeded.insert_occupancy([
        {"camera_id": "cam1", "zone_id": "z1", "ts": now - 10, "count": 2},
        {"camera_id": "cam1", "zone_id": "z1", "ts": now - 5, "count": 4},
        {"camera_id": "cam1", "zone_id": "z1", "ts": now, "count": 3},
    ]) == 3
    assert seeded.insert_occupancy([]) == 0

    series = seeded.occupancy_series(camera_id="cam1")
    assert [s["count"] for s in series] == [2, 4, 3]      # ascending by ts
    assert len(seeded.occupancy_series(since_ts=now - 6)) == 2
    assert seeded.latest_zone_counts("cam1")["z1"]["count"] == 3


def test_purge_older_than(seeded: Database):
    seeded.insert_event(_event())
    assert seeded.purge_older_than("events", "created_at", "2099-01-01") == 1
    assert seeded.recent_events() == []
    with pytest.raises(ValueError, match="purge not allowed"):
        seeded.purge_older_than("cameras", "created_at", "2099-01-01")


def test_event_ids_are_never_reused_after_a_purge(seeded: Database):
    """AUTOINCREMENT, not plain rowid: the live-feed cursor is `id > last_id`,
    so a recycled id after nightly retention would hide new events from every
    connected client."""
    old_id = seeded.insert_event(_event())
    seeded.purge_older_than("events", "created_at", "2099-01-01")
    assert seeded.insert_event(_event()) > old_id


def test_purge_cascades_to_reviews_and_notifications(seeded: Database):
    event_id = seeded.insert_event(_event())
    seeded.record_review(event_id, "confirmed")
    seeded.log_notification(event_id, "smtp", "x@y", "sent")
    seeded.purge_older_than("events", "created_at", "2099-01-01")
    with seeded.conn() as conn:
        assert conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM notifications").fetchone()[0] == 0


def test_counts(seeded: Database):
    counts = seeded.counts()
    assert counts["cameras"] == 1 and counts["zones"] == 1 and counts["policies"] == 1
    assert counts["events"] == 0


def test_rollback_on_error_leaves_no_partial_write(database: Database):
    database.upsert_camera(Camera(id="cam1", name="c", source="/a.mp4"))
    with pytest.raises(sqlite3.IntegrityError):
        with database.conn() as conn:
            conn.execute(
                "INSERT INTO cameras (id, name, source, kind, created_at, updated_at) "
                "VALUES ('cam2', 'x', '/b.mp4', 'file', '', '')"
            )
            conn.execute("INSERT INTO cameras (id, name, source, created_at, updated_at) "
                         "VALUES ('cam1', 'dupe', '/c.mp4', '', '')")
    assert {c["id"] for c in database.list_cameras()} == {"cam1"}
