"""Argus v2 appliance database (SQLite, WAL).

One SQLite file is the right shape for a single-box appliance: no server to
operate, atomic on-disk backup, and WAL gives us one writer plus any number of
concurrent readers (engine writes, API reads).

Differences from v1 (``pipeline/db.py`` on ``v1-archive``), all deliberate:

* **No ``jobs`` table.** v1 modelled "process a dropped video file" as a job and
  bolted live cameras on with pseudo-jobs. v2 is camera-first: a file source is
  just a camera whose source happens to be a path, so every event hangs off a
  real camera row.
* **``reviews`` is its own table** rather than two nullable columns on
  ``events``. A verdict is an act by a person at a time — it deserves a row with
  a reviewer and a note, and it keeps the accuracy queries honest.
* **Explicit, versioned migrations** instead of v1's "CREATE TABLE IF NOT
  EXISTS + a list of ALTERs replayed every boot".
* **No SQL built by string interpolation of values.** v1 assembled UPDATE SET
  clauses with f-strings; here every value is bound, and the only interpolated
  identifiers are column names checked against a per-table allowlist.

Semantics carried over from v1 unchanged, because they are the product:

* event ``stage``:      shadow -> reviewed -> auto_routed
* policy ``lifecycle``: draft -> pilot -> production (-> archived)
* review ``verdict``:   confirmed | false_positive, feeding the accuracy stats
  that gate promotion to auto_routed.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

Row = dict[str, Any]

SCHEMA_VERSION = 8

STAGES = ("shadow", "reviewed", "auto_routed")
LIFECYCLES = ("draft", "pilot", "production", "archived")
VERDICTS = ("confirmed", "false_positive")
BOX_VERDICTS = ("true_positive", "false_positive")
SEVERITIES = ("low", "medium", "high")
CAMERA_STATUSES = ("offline", "connecting", "live", "error")


class Migration(NamedTuple):
    version: int
    name: str
    sql: str


# --------------------------------------------------------------------------
# migrations
# --------------------------------------------------------------------------
#
# Append-only. Never edit a shipped migration — add the next one. `migrate()`
# applies anything the database has not recorded yet, inside one transaction per
# migration, and records it in schema_migrations.

_M001_INITIAL = """
CREATE TABLE cameras (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    source        TEXT NOT NULL,                 -- rtsp:// URL or a local file path
    kind          TEXT NOT NULL DEFAULT 'rtsp'
                  CHECK (kind IN ('rtsp', 'http', 'file')),
    enabled       INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
    status        TEXT NOT NULL DEFAULT 'offline'
                  CHECK (status IN ('offline', 'connecting', 'live', 'error')),
    status_detail TEXT,
    detect_fps    REAL,                          -- NULL = use config detection.fps
    last_seen_at  TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE zones (
    id               TEXT PRIMARY KEY,
    camera_id        TEXT NOT NULL REFERENCES cameras (id) ON DELETE CASCADE,
    name             TEXT NOT NULL,
    polygon          TEXT NOT NULL,              -- JSON [[x, y], ...] normalized 0-1
    sanctioned_limit INTEGER,                    -- NULL = count only, never alert
    color            TEXT NOT NULL DEFAULT '#22d3ee',
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
CREATE INDEX idx_zones_camera ON zones (camera_id);

CREATE TABLE policies (
    id                  TEXT PRIMARY KEY,
    kind                TEXT NOT NULL CHECK (kind IN ('detector', 'vlm')),
    detector_key        TEXT NOT NULL,           -- manpower_headcount | vlm_prompt | ...
    label               TEXT NOT NULL,
    enabled             INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
    stage               TEXT NOT NULL DEFAULT 'shadow'
                        CHECK (stage IN ('shadow', 'reviewed', 'auto_routed')),
    lifecycle           TEXT NOT NULL DEFAULT 'draft'
                        CHECK (lifecycle IN ('draft', 'pilot', 'production', 'archived')),
    severity            TEXT NOT NULL DEFAULT 'medium'
                        CHECK (severity IN ('low', 'medium', 'high')),
    routed_to           TEXT,                    -- role name; resolved via config smtp.recipients
    prompt              TEXT,                    -- vlm only
    interval_sec        REAL,                    -- vlm only: seconds between sampled frames
    cooldown_sec        REAL NOT NULL DEFAULT 30,
    threshold           REAL,                    -- detector only: limit / confidence floor
    pilot_started_at    TEXT,
    pilot_duration_days REAL NOT NULL DEFAULT 2,
    promoted_at         TEXT,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);

CREATE TABLE events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id     TEXT NOT NULL REFERENCES cameras (id) ON DELETE CASCADE,
    zone_id       TEXT REFERENCES zones (id) ON DELETE SET NULL,
    policy_id     TEXT REFERENCES policies (id) ON DELETE SET NULL,
    detector_name TEXT NOT NULL,                 -- usually the policy id; kept for orphaned events
    detector_type TEXT NOT NULL CHECK (detector_type IN ('detector', 'vlm')),
    severity      TEXT NOT NULL CHECK (severity IN ('low', 'medium', 'high')),
    confidence    REAL NOT NULL DEFAULT 0,
    stage         TEXT NOT NULL CHECK (stage IN ('shadow', 'reviewed', 'auto_routed')),
    routed_to     TEXT,
    description   TEXT NOT NULL,
    ts            REAL NOT NULL,                 -- unix epoch of the triggering frame
    snapshot_path TEXT,
    clip_path     TEXT,
    clip_start    REAL,                          -- epoch of the clip's first frame
    tracks_path   TEXT,
    meta          TEXT,                          -- JSON blob for detector-specific detail
    created_at    TEXT NOT NULL
);
CREATE INDEX idx_events_created_at ON events (created_at DESC);
CREATE INDEX idx_events_camera_ts  ON events (camera_id, ts DESC);
CREATE INDEX idx_events_detector   ON events (detector_name);

CREATE TABLE reviews (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id   INTEGER NOT NULL UNIQUE REFERENCES events (id) ON DELETE CASCADE,
    verdict    TEXT NOT NULL CHECK (verdict IN ('confirmed', 'false_positive')),
    reviewer   TEXT,
    note       TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX idx_reviews_verdict ON reviews (verdict);

CREATE TABLE notifications (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id   INTEGER NOT NULL REFERENCES events (id) ON DELETE CASCADE,
    channel    TEXT NOT NULL,                    -- smtp | slack | webhook | none
    target     TEXT NOT NULL,
    status     TEXT NOT NULL CHECK (status IN ('sent', 'failed')),
    detail     TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX idx_notifications_event ON notifications (event_id);

CREATE TABLE occupancy (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id  TEXT NOT NULL REFERENCES cameras (id) ON DELETE CASCADE,
    zone_id    TEXT REFERENCES zones (id) ON DELETE CASCADE,
    ts         REAL NOT NULL,                    -- unix epoch
    count      INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX idx_occupancy_zone_ts   ON occupancy (zone_id, ts);
CREATE INDEX idx_occupancy_camera_ts ON occupancy (camera_id, ts);
"""

# Phase 2 (VLM policy engine): the verification pass.
#
# A detector-origin event on its way to an auto_routed email can be shown to the
# VLM first, which may veto it. A vetoed event is **not deleted and not
# downgraded** — it keeps its row, its snapshot and its clip, and gains the
# reason it was held back. That is the whole point: "the model suppressed 40
# alerts last week, here they are" is a reviewable claim, whereas an alert that
# was silently never raised is indistinguishable from one that never happened.
#
# `suppressed` is separate from `stage` on purpose. Stage is the *policy's*
# promise about who gets told; suppression is a per-event judgement about one
# frame. Collapsing them would lose the ability to say "this policy is
# auto_routed and this particular alert was still held".
_M002_VLM_VERIFICATION = """
ALTER TABLE events ADD COLUMN suppressed        INTEGER NOT NULL DEFAULT 0;
ALTER TABLE events ADD COLUMN suppressed_reason TEXT;
ALTER TABLE events ADD COLUMN verified_by       TEXT;   -- e.g. 'vlm:Qwen3-VL-8B-Instruct'
ALTER TABLE events ADD COLUMN verified_at       TEXT;
ALTER TABLE events ADD COLUMN verify_confidence REAL;   -- the verifier's own confidence

-- Partial index: the triage queue and the analytics only ever ask for the
-- suppressed ones, and they are the minority.
CREATE INDEX idx_events_suppressed ON events (suppressed) WHERE suppressed = 1;
"""

# Phase 3 (semantic search): the searches an operator keeps.
#
# Saved searches live in the appliance database rather than in the vector
# store, and that split is the point: `data/db/index.sqlite3` holds *derived*
# data that `argus index backfill --reset` can rebuild from scratch, while a
# saved search is something a person wrote and nothing can regenerate. Deleting
# the index must never delete "person on the loading dock without a hi-vis" —
# and this is also the row a promotion reads to become a draft VLM policy.
#
# `policy_id` records that promotion, and is the reason this is a table rather
# than a JSON blob in config: "this policy started life as a search, on this
# date" is provenance, and it is what lets Policy Studio show where a prompt
# came from.
_M003_SAVED_SEARCHES = """
CREATE TABLE saved_searches (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    label      TEXT NOT NULL,
    query      TEXT NOT NULL,
    camera_id  TEXT REFERENCES cameras (id) ON DELETE SET NULL,
    -- Relative window in hours, not absolute timestamps: "the last shift" is
    -- what an operator means by a saved search, and a stored epoch pair would
    -- mean the same search returned less every day until it returned nothing.
    window_hours REAL,
    -- Set when this search has been promoted to a draft VLM policy.
    policy_id  TEXT REFERENCES policies (id) ON DELETE SET NULL,
    run_count  INTEGER NOT NULL DEFAULT 0,
    last_run_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX idx_saved_searches_query ON saved_searches (query, COALESCE(camera_id, ''));
"""

# Phase 5 (appliance hardening): accounts, sessions, and the login rate limit.
#
# Three tables because they have three different lifetimes. A *user* is a fact
# about the plant that outlives every reboot; a *session* is a fact about one
# browser that must be revocable and must expire; a *login attempt* is
# telemetry with a five-minute half-life. Folding them together would mean
# either pruning the roster or keeping a year of failed passwords.
#
# `sessions.id` holds sha256(cookie token), never the token — see
# argus/auth.py. A backup, a support bundle or an operator with `sqlite3`
# therefore cannot lift a live session out of this file.
#
# `users.username` is `COLLATE NOCASE` so `J.Whitfield` and `j.whitfield` are
# the same account rather than two accounts with different permissions; the
# application lower-cases on the way in as well, and the collation is what makes
# that a schema guarantee rather than a convention.
#
# `login_attempts` records failures *and* successes. Only the failures feed the
# rate limiter, but "who logged in, from where, when" is the first question
# asked after an incident, and a table that holds only the failures cannot
# answer it.
_M004_ACCOUNTS = """
CREATE TABLE users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL UNIQUE COLLATE NOCASE,
    display_name  TEXT NOT NULL,
    role          TEXT NOT NULL
                  CHECK (role IN ('ADMIN', 'RESPOND', 'REVIEW', 'SYSTEM')),
    password_hash TEXT NOT NULL,
    disabled      INTEGER NOT NULL DEFAULT 0 CHECK (disabled IN (0, 1)),
    last_login_at TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE sessions (
    id           TEXT PRIMARY KEY,           -- sha256 of the cookie token
    user_id      INTEGER NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    created_at   TEXT NOT NULL,
    expires_at   TEXT NOT NULL,              -- absolute deadline, ISO-8601 UTC
    last_seen_at TEXT NOT NULL,              -- rolls forward; feeds the idle timeout
    user_agent   TEXT,
    ip           TEXT
);
CREATE INDEX idx_sessions_user    ON sessions (user_id);
CREATE INDEX idx_sessions_expires ON sessions (expires_at);

CREATE TABLE login_attempts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    username   TEXT NOT NULL,                -- as typed (lower-cased), even if no such user
    ip         TEXT NOT NULL,
    ok         INTEGER NOT NULL CHECK (ok IN (0, 1)),
    created_at TEXT NOT NULL
);
CREATE INDEX idx_login_attempts_window ON login_attempts (created_at);
CREATE INDEX idx_login_attempts_who    ON login_attempts (username, ip, created_at);
"""

# Phase 5 (authoring): where a policy applies.
#
# Until now every policy ran on every camera. That is the right default for a
# two-camera demo and wrong for a plant: "no phones on the packing line" is a
# statement about the packing line, and asking a VLM the same question about the
# car park costs an inference every interval to answer "no" forever. At the
# measured 4.98 s an evaluation, an unscoped prompt policy on a 48-camera fleet
# is arithmetic that does not close.
#
# **An empty scope means every camera**, and that is what makes this migration
# safe to apply to a running appliance: no rows, no change in behaviour. The
# alternative — backfilling one row per (policy, camera) at migration time —
# would freeze today's fleet into every existing policy, so a camera enrolled
# tomorrow would silently not be watched by any of them.
#
# `zone_id` is nullable and means two different things to the two engines,
# because a zone means two different things to them:
#
#   detector policies  a scoped zone is the only zone that may raise this
#                      policy's events on that camera. The loop already
#                      iterates zones; this filters that iteration.
#   vlm policies       a scoped zone is what the model is *shown* — the frame
#                      cropped to the zone polygon (argus/vlm/evidence.py).
#                      Measured on the verification pass, which is the same
#                      question over the same footage: 12/12 answered correctly
#                      on the crop against 3/12 on the wide shot it came from.
#
# A row with `zone_id` NULL scopes to the whole camera. Both columns cascade:
# deleting a camera or a zone removes the scope rows that named it, and a policy
# that loses its last row goes back to meaning "everywhere" — which is the
# behaviour it had before anybody scoped it, and is visible in the console
# rather than silent, because the scope is on the policy card.
_M005_POLICY_SCOPES = """
CREATE TABLE policy_scopes (
    policy_id  TEXT NOT NULL REFERENCES policies (id) ON DELETE CASCADE,
    camera_id  TEXT NOT NULL REFERENCES cameras (id) ON DELETE CASCADE,
    zone_id    TEXT REFERENCES zones (id) ON DELETE CASCADE,
    created_at TEXT NOT NULL
);

-- One row per (policy, camera, zone). COALESCE because NULL is never equal to
-- NULL in SQL, so a plain UNIQUE would happily hold two "whole camera" rows.
CREATE UNIQUE INDEX idx_policy_scopes_unique
    ON policy_scopes (policy_id, camera_id, COALESCE(zone_id, ''));
CREATE INDEX idx_policy_scopes_camera ON policy_scopes (camera_id);
"""

# Phase 5 (authoring): why a policy is auto_routed on evidence that did not
# clear the bar.
#
# The bar (`rollout:` in argus.yaml) is enforced by the promote endpoint, and an
# override is allowed because a site whose validation happened elsewhere has to
# be able to say so — an appliance that refuses forever gets worked around with
# `sqlite3`, which is strictly worse because it leaves no trace. The price of
# the override is that it must name a reason, and the reason lands here, on the
# row, where the next person to ask "why is this live on twelve reviews?" looks.
#
# **This is its own migration rather than a line added to 5, and that is not
# tidiness.** Both were written on the same afternoon and 5 had not been
# committed — so editing it looked free. It is not: `argus vlm start` and a live
# run had already applied 5 to two databases on this box, which recorded version
# 5 as done and would never have replayed the added ALTER. The first request
# that touched the new column answered 500 with `no such column`. "Never edit a
# shipped migration" is really "never edit an *applied* one", and applied
# happens the first time anybody runs the branch.
_M006_PROMOTION_NOTE = """
ALTER TABLE policies ADD COLUMN promotion_note TEXT;
"""

# Phase 6: an ongoing condition is one event, observed many times.
#
# A prompt policy asks about a *state*. Loose laminate on the wrapper apron is
# true at 20:22 and still true at 20:37, and until now every expiry of the
# policy's cooldown wrote another row about it: 140 rows on this appliance for
# one apron, 40 of them in a single night. The feed became unreadable and the
# analytics counted one mess as forty incidents.
#
# So the row becomes the condition. A triggered verdict that continues the
# condition already open for its (policy, camera, zone) extends that row — a new
# `last_seen_ts`, one more `observations`, the newest sentence and the newest
# evidence — and inserts nothing. The condition closes when a sampled verdict
# comes back untriggered, or when nobody has observed it for
# `vlm.continuity.stale_after_sec`.
#
# **Columns on `events`, not a second table.** The console, the triage queue,
# the search index, the notifier, retention and the analytics all read `events`;
# a parallel `conditions` table would need every one of them taught to join it,
# and the first one forgotten would show forty cards again. A condition *is* an
# event that lasted.
#
# **Nullable everywhere, and nothing is backfilled.** The 140 rows already on
# this box stay 140 rows: merging them retroactively would rewrite history to
# agree with a feature, and the honest version is that continuity starts when it
# shipped. That bargain was struck again on 2026-09-19, when the same columns
# were turned on for detector-origin events (`detection.continuity`) — the claim
# here that "its debounce already owns continuity for counts in a polygon" was
# wrong, and 112 151 headcount rows are the evidence. Those rows stay too, as
# ordinary one-off events with every column below NULL, which is exactly what
# they were.
#
# `condition_updated_at` is separate from `last_seen_ts` because closing a
# condition changes it without observing it, and the SSE stream needs one
# monotonic marker it can poll to know a card must be redrawn. Without it the
# feed only ever hears about *new* ids, and an extended condition would sit
# stale on screen until the reader reloaded.
_M007_CONDITIONS = """
ALTER TABLE events ADD COLUMN condition_key       TEXT;
ALTER TABLE events ADD COLUMN condition_state     TEXT;   -- open | closed | NULL
ALTER TABLE events ADD COLUMN observations        INTEGER NOT NULL DEFAULT 1;
ALTER TABLE events ADD COLUMN last_seen_ts        REAL;   -- epoch of the newest observation
ALTER TABLE events ADD COLUMN closed_at           REAL;   -- epoch the condition ended
ALTER TABLE events ADD COLUMN close_reason        TEXT;   -- cleared | stale | moved
ALTER TABLE events ADD COLUMN condition_updated_at REAL;  -- epoch of the last write

-- Partial: the extend path asks "is there an open condition for this key?" on
-- every triggered verdict, and open conditions are a handful of rows against a
-- table that holds every event this appliance has ever raised.
CREATE INDEX idx_events_condition_open
    ON events (condition_key, id DESC) WHERE condition_state = 'open';

-- The SSE stream's cursor. Also partial: detector events never carry one.
CREATE INDEX idx_events_condition_updated
    ON events (condition_updated_at) WHERE condition_updated_at IS NOT NULL;
"""

# Evidence-viewer box review: per-detection ground truth, separate from the
# whole-event `reviews` verdict.
#
# `reviews.verdict` judges the *alert* — "was this zone genuinely over its
# limit". A reviewer can agree with that and still see one box in the crowd
# that is not a person at all (a mannequin, a reflection, a duplicate track),
# and "the alert was right but this one box was wrong" is information the
# whole-event verdict cannot express. Losing it is losing exactly the signal
# a retraining pass needs: which individual detections were correct.
#
# `track_id` keys off the same id ByteTrack assigned and the console already
# shows ("Worker 3") — a box that reached a saved event has a confirmed track
# (see `TrackerPool`/`minimum_consecutive_frames`), so it is a stable choice
# of key, not the ephemeral `-1` a flickering, unconfirmed detection would
# carry. One correction per (event, box): marking the same box again replaces
# the earlier verdict rather than accumulating duplicates.
_M008_BOX_CORRECTIONS = """
CREATE TABLE box_corrections (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id   INTEGER NOT NULL REFERENCES events (id) ON DELETE CASCADE,
    track_id   INTEGER NOT NULL,
    verdict    TEXT NOT NULL CHECK (verdict IN ('true_positive', 'false_positive')),
    reviewer   TEXT,
    note       TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX idx_box_corrections_event_track ON box_corrections (event_id, track_id);
CREATE INDEX idx_box_corrections_verdict ON box_corrections (verdict);
"""

MIGRATIONS: tuple[Migration, ...] = (
    Migration(1, "initial_schema_v2", _M001_INITIAL),
    Migration(2, "vlm_verification", _M002_VLM_VERIFICATION),
    Migration(3, "saved_searches", _M003_SAVED_SEARCHES),
    Migration(4, "accounts_and_sessions", _M004_ACCOUNTS),
    Migration(5, "policy_scopes", _M005_POLICY_SCOPES),
    Migration(6, "promotion_note", _M006_PROMOTION_NOTE),
    Migration(7, "ongoing_conditions", _M007_CONDITIONS),
    Migration(8, "box_corrections", _M008_BOX_CORRECTIONS),
)

# Column allowlists for the guarded partial-update helpers. Column *names*
# cannot be bound as SQL parameters, so they are validated against these sets
# and quoted; every value is always bound. test_db.py asserts these stay in sync
# with the real schema, so a future migration cannot silently drift.
_UPDATABLE: dict[str, frozenset[str]] = {
    "cameras": frozenset(
        {"name", "source", "kind", "enabled", "status", "status_detail",
         "detect_fps", "last_seen_at", "updated_at"}
    ),
    "zones": frozenset({"name", "polygon", "sanctioned_limit", "color", "updated_at"}),
    "policies": frozenset(
        {"kind", "detector_key", "label", "enabled", "stage", "lifecycle", "severity",
         "routed_to", "prompt", "interval_sec", "cooldown_sec", "threshold",
         "pilot_started_at", "pilot_duration_days", "promoted_at", "promotion_note",
         "updated_at"}
    ),
    "events": frozenset(
        {"severity", "stage", "routed_to", "description", "snapshot_path",
         "clip_path", "clip_start", "tracks_path", "meta",
         # migration 2 — the VLM verification pass
         "suppressed", "suppressed_reason", "verified_by", "verified_at",
         "verify_confidence"}
    ),
    # migration 3 — saved searches. `run_count` and `last_run_at` are absent on
    # purpose: they are bumped by their own guarded UPDATE, not by a PATCH body.
    "saved_searches": frozenset(
        {"label", "query", "camera_id", "window_hours", "policy_id", "updated_at"}
    ),
    # migration 4 — accounts. `username` is absent deliberately: renaming an
    # account would orphan every `reviews.reviewer` row that names it, so the
    # audit trail decides this, not a PATCH body. `password_hash` is absent too
    # — it is written only by `set_password`, which hashes.
    "users": frozenset({"display_name", "role", "disabled", "last_login_at", "updated_at"}),
}

# Columns holding a path to a file on disk. Retention nulls these when it
# deletes the file they point at; nothing else may be passed to
# `clear_media_paths`.
_MEDIA_COLUMNS = frozenset({"snapshot_path", "clip_path", "tracks_path"})

# Tie-break only, for the (unexpected) case of two event rows naming one file:
# whose claim on it is strongest. NOT the retention ordering — that is derived
# from the configured per-tier windows in argus/retention.py, so an operator who
# says a false positive is worth keeping longer than a shadow event gets exactly
# that, and there is no second value ranking here to contradict them.
_TIER_CLAIM = {"shadow": 0, "false_positive": 1, "reviewed": 2,
               "auto_routed": 3, "confirmed": 4}

# The columns of `users` that anything outside the login path may see. Written
# once, used by every user read, so `password_hash` cannot be added back to a
# response by a `SELECT *` in a hurry.
_USER_PUBLIC = ("id, username, display_name, role, disabled, last_login_at, "
                "created_at, updated_at")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return slug or "item"


def decoded_meta(event: Mapping[str, Any]) -> dict[str, Any] | None:
    """An event's ``meta`` as an object, whatever form the row arrived in.

    ``insert_event`` writes this column as JSON text, so the decoder belongs
    beside the encoder rather than being rewritten by each consumer. Undecodable
    text is None, not an exception: a corrupt detail blob must not stop an alert
    email or a triage screen from rendering the event it belongs to.
    """
    meta = event.get("meta")
    if meta is None or isinstance(meta, dict):
        return meta
    try:
        decoded = json.loads(meta)
    except (json.JSONDecodeError, TypeError):
        return None
    return decoded if isinstance(decoded, dict) else None


def source_kind(source: str) -> str:
    """Classify a camera source string. File paths are looped by the gateway."""
    lowered = source.lower()
    if lowered.startswith(("rtsp://", "rtsps://", "rtmp://")):
        return "rtsp"
    if lowered.startswith(("http://", "https://")):
        return "http"
    return "file"


# --------------------------------------------------------------------------
# write-path records
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Camera:
    id: str
    name: str
    source: str
    enabled: bool = True
    detect_fps: float | None = None
    kind: str | None = None  # derived from source when omitted

    def resolved_kind(self) -> str:
        return self.kind or source_kind(self.source)


@dataclass(slots=True)
class Zone:
    id: str
    camera_id: str
    name: str
    polygon: Sequence[Sequence[float]]
    sanctioned_limit: int | None = None
    color: str = "#22d3ee"


class PolicyScope(NamedTuple):
    """One place a policy applies. ``zone_id`` None means the whole camera."""

    camera_id: str
    zone_id: str | None = None


def as_scope(entry: Any) -> PolicyScope:
    """Accept the three spellings a scope arrives in and produce one.

    A bare camera id from ``seed.yaml``, a mapping from a JSON request body, or
    a :class:`PolicyScope` from code. Written once because the seed importer,
    the API and the tests would otherwise each grow their own coercion and
    disagree about whether ``{"camera_id": "x", "zone_id": ""}`` is scoped to a
    zone (it is not — empty string is not a zone).
    """
    if isinstance(entry, PolicyScope):
        return entry
    if isinstance(entry, str):
        return PolicyScope(entry, None)
    if isinstance(entry, Mapping):
        camera = entry.get("camera_id", entry.get("camera"))
        zone = entry.get("zone_id", entry.get("zone"))
        if not camera:
            raise ValueError(f"scope entry has no camera: {entry!r}")
        return PolicyScope(str(camera), str(zone) if zone else None)
    if isinstance(entry, Sequence) and len(entry) == 2:  # (camera_id, zone_id)
        camera, zone = entry
        return PolicyScope(str(camera), str(zone) if zone else None)
    raise ValueError(f"cannot read a policy scope from {entry!r}")


def scope_cameras(scope: Sequence[Mapping[str, Any]]) -> set[str] | None:
    """The camera ids a scope names, or None for "every camera"."""
    if not scope:
        return None
    return {str(row["camera_id"]) for row in scope}


def scope_zones(scope: Sequence[Mapping[str, Any]], camera_id: str) -> list[str | None]:
    """The zones this policy applies to on one camera.

    ``[None]`` means the whole camera — which is also the answer for an unscoped
    policy, so a caller can iterate this list without a special case for either.
    """
    if not scope:
        return [None]
    zones = [row.get("zone_id") for row in scope if str(row["camera_id"]) == camera_id]
    if not zones:
        return []
    # A whole-camera row wins over any zone rows beside it: "this camera, and
    # also this zone on it" is the camera.
    return [None] if any(zone is None for zone in zones) else zones


def scope_allows(
    scope: Sequence[Mapping[str, Any]], camera_id: str, zone_id: str | None = None
) -> bool:
    """Whether a policy may fire here. The detector loop's whole scope check."""
    zones = scope_zones(scope, camera_id)
    if not zones:
        return False
    return None in zones or zone_id in zones


@dataclass(slots=True)
class Policy:
    id: str
    kind: str  # detector | vlm
    detector_key: str
    label: str
    enabled: bool = True
    stage: str = "shadow"
    lifecycle: str = "draft"
    severity: str = "medium"
    routed_to: str | None = None
    prompt: str | None = None
    interval_sec: float | None = None
    cooldown_sec: float = 30.0
    threshold: float | None = None
    pilot_duration_days: float = 2.0
    # Where this policy applies. ``None`` leaves an existing scope alone (so an
    # upsert that does not care cannot silently unscope a policy somebody
    # scoped in the console); ``[]`` explicitly means every camera.
    scope: Sequence[Any] | None = None


@dataclass(slots=True)
class Event:
    camera_id: str
    detector_name: str
    detector_type: str  # detector | vlm
    severity: str
    description: str
    ts: float
    stage: str = "shadow"
    confidence: float = 0.0
    zone_id: str | None = None
    policy_id: str | None = None
    routed_to: str | None = None
    snapshot_path: str | None = None
    clip_path: str | None = None
    clip_start: float | None = None
    tracks_path: str | None = None
    meta: dict[str, Any] | None = None
    created_at: str = field(default_factory=utcnow)
    # --- migration 7: ongoing conditions (VLM-origin events only) ----------
    # Set by the VLM scheduler when it opens a condition. A detector event
    # leaves them None and is stored exactly as it was before migration 7.
    condition_key: str | None = None
    condition_state: str | None = None   # 'open' at insert; 'closed' later


# --------------------------------------------------------------------------
# database
# --------------------------------------------------------------------------


class Database:
    """Handle on the appliance SQLite file.

    A connection is opened per operation rather than cached: the engine, the
    gateway supervisor and (later) the API all touch this file from different
    threads and processes, and WAL plus a busy timeout makes short-lived
    connections both correct and cheap.
    """

    def __init__(self, path: str | Path, *, timeout: float = 10.0):
        self.path = Path(path)
        self.timeout = timeout

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Database({str(self.path)!r})"

    # --- connection -------------------------------------------------------

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=self.timeout)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute(f"PRAGMA busy_timeout = {int(self.timeout * 1000)}")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    @contextmanager
    def conn(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    # --- migrations -------------------------------------------------------

    def migrate(self) -> list[int]:
        """Apply every migration this database has not recorded. Returns the
        versions applied (empty when already current)."""
        applied: list[int] = []
        with self.conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version    INTEGER PRIMARY KEY,
                    name       TEXT NOT NULL,
                    applied_at TEXT NOT NULL
                )
                """
            )
            done = {r["version"] for r in conn.execute("SELECT version FROM schema_migrations")}
            for migration in MIGRATIONS:
                if migration.version in done:
                    continue
                conn.executescript(migration.sql)
                conn.execute(
                    "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
                    (migration.version, migration.name, utcnow()),
                )
                applied.append(migration.version)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        return applied

    def schema_version(self) -> int:
        with self.conn() as conn:
            table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_migrations'"
            ).fetchone()
            if table is None:
                return 0
            row = conn.execute("SELECT COALESCE(MAX(version), 0) AS v FROM schema_migrations").fetchone()
            return int(row["v"])

    # --- internals --------------------------------------------------------

    @staticmethod
    def _rows(cursor: sqlite3.Cursor) -> list[Row]:
        return [dict(r) for r in cursor.fetchall()]

    def _update(self, table: str, pk_column: str, pk: Any, fields: Mapping[str, Any]) -> int:
        """Partial UPDATE with column names validated against an allowlist.

        Values are always bound; only identifiers are interpolated, and only
        after membership in ``_UPDATABLE[table]`` is proven.
        """
        if not fields:
            return 0
        allowed = _UPDATABLE[table]
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"{table}: not updatable: {', '.join(sorted(unknown))}")
        assignments = ", ".join(f'"{column}" = ?' for column in fields)
        with self.conn() as conn:
            cur = conn.execute(
                f'UPDATE "{table}" SET {assignments} WHERE "{pk_column}" = ?',  # noqa: S608 - identifiers allowlisted above
                (*fields.values(), pk),
            )
            return cur.rowcount

    # --- cameras ----------------------------------------------------------

    def upsert_camera(self, camera: Camera) -> str:
        """Insert or update by id. Idempotent — this is what `argus seed` uses.
        Runtime status columns are intentionally not touched."""
        with self.conn() as conn:
            conn.execute(
                """
                INSERT INTO cameras (id, name, source, kind, enabled, detect_fps, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET
                    name       = excluded.name,
                    source     = excluded.source,
                    kind       = excluded.kind,
                    enabled    = excluded.enabled,
                    detect_fps = excluded.detect_fps,
                    updated_at = excluded.updated_at
                """,
                (
                    camera.id, camera.name, camera.source, camera.resolved_kind(),
                    int(camera.enabled), camera.detect_fps, utcnow(), utcnow(),
                ),
            )
        return camera.id

    def unique_id(self, table: str, base: str) -> str:
        """A free id derived from ``base`` — ``base``, else ``base_2``, ``base_3``.

        Public because the API mints ids the same way when a caller does not
        supply one, and two implementations of "make this slug unique" would
        eventually disagree about what the id for a second "Packing Line" is.
        """
        with self.conn() as conn:
            return _unique_id(conn, table, base)

    def create_camera(self, name: str, source: str, *, enabled: bool = True) -> str:
        camera_id = self.unique_id("cameras", slugify(name))
        self.upsert_camera(Camera(id=camera_id, name=name, source=source, enabled=enabled))
        return camera_id

    def get_camera(self, camera_id: str) -> Row | None:
        with self.conn() as conn:
            row = conn.execute("SELECT * FROM cameras WHERE id = ?", (camera_id,)).fetchone()
        return dict(row) if row else None

    def list_cameras(self, *, enabled_only: bool = False) -> list[Row]:
        with self.conn() as conn:
            if enabled_only:
                cur = conn.execute("SELECT * FROM cameras WHERE enabled = 1 ORDER BY name")
            else:
                cur = conn.execute("SELECT * FROM cameras ORDER BY name")
            return self._rows(cur)

    def update_camera(self, camera_id: str, **fields: Any) -> int:
        return self._update("cameras", "id", camera_id, {**fields, "updated_at": utcnow()})

    def set_camera_status(self, camera_id: str, status: str, detail: str | None = None) -> None:
        if status not in CAMERA_STATUSES:
            raise ValueError(f"unknown camera status: {status}")
        fields: dict[str, Any] = {"status": status, "status_detail": detail, "updated_at": utcnow()}
        if status == "live":
            fields["last_seen_at"] = utcnow()
        self._update("cameras", "id", camera_id, fields)

    def delete_camera(self, camera_id: str) -> int:
        with self.conn() as conn:
            return conn.execute("DELETE FROM cameras WHERE id = ?", (camera_id,)).rowcount

    # --- zones ------------------------------------------------------------

    def upsert_zone(self, zone: Zone) -> str:
        polygon_json = json.dumps([[float(x), float(y)] for x, y in zone.polygon])
        with self.conn() as conn:
            conn.execute(
                """
                INSERT INTO zones (id, camera_id, name, polygon, sanctioned_limit, color, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET
                    camera_id        = excluded.camera_id,
                    name             = excluded.name,
                    polygon          = excluded.polygon,
                    sanctioned_limit = excluded.sanctioned_limit,
                    color            = excluded.color,
                    updated_at       = excluded.updated_at
                """,
                (zone.id, zone.camera_id, zone.name, polygon_json,
                 zone.sanctioned_limit, zone.color, utcnow(), utcnow()),
            )
        return zone.id

    def list_zones(self, camera_id: str | None = None) -> list[Row]:
        with self.conn() as conn:
            if camera_id:
                cur = conn.execute("SELECT * FROM zones WHERE camera_id = ? ORDER BY name", (camera_id,))
            else:
                cur = conn.execute("SELECT * FROM zones ORDER BY camera_id, name")
            rows = self._rows(cur)
        for row in rows:
            row["polygon"] = json.loads(row["polygon"])
        return rows

    def get_zone(self, zone_id: str) -> Row | None:
        with self.conn() as conn:
            row = conn.execute("SELECT * FROM zones WHERE id = ?", (zone_id,)).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["polygon"] = json.loads(data["polygon"])
        return data

    def update_zone(self, zone_id: str, **fields: Any) -> int:
        if "polygon" in fields and not isinstance(fields["polygon"], str):
            fields["polygon"] = json.dumps([[float(x), float(y)] for x, y in fields["polygon"]])
        return self._update("zones", "id", zone_id, {**fields, "updated_at": utcnow()})

    def delete_zone(self, zone_id: str) -> int:
        with self.conn() as conn:
            return conn.execute("DELETE FROM zones WHERE id = ?", (zone_id,)).rowcount

    # --- policies ---------------------------------------------------------

    def upsert_policy(self, policy: Policy) -> str:
        with self.conn() as conn:
            conn.execute(
                """
                INSERT INTO policies
                    (id, kind, detector_key, label, enabled, stage, lifecycle, severity,
                     routed_to, prompt, interval_sec, cooldown_sec, threshold,
                     pilot_duration_days, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET
                    kind                = excluded.kind,
                    detector_key        = excluded.detector_key,
                    label               = excluded.label,
                    enabled             = excluded.enabled,
                    stage               = excluded.stage,
                    lifecycle           = excluded.lifecycle,
                    severity            = excluded.severity,
                    routed_to           = excluded.routed_to,
                    prompt              = excluded.prompt,
                    interval_sec        = excluded.interval_sec,
                    cooldown_sec        = excluded.cooldown_sec,
                    threshold           = excluded.threshold,
                    pilot_duration_days = excluded.pilot_duration_days,
                    updated_at          = excluded.updated_at
                """,
                (
                    policy.id, policy.kind, policy.detector_key, policy.label,
                    int(policy.enabled), policy.stage, policy.lifecycle, policy.severity,
                    policy.routed_to, policy.prompt, policy.interval_sec, policy.cooldown_sec,
                    policy.threshold, policy.pilot_duration_days, utcnow(), utcnow(),
                ),
            )
        if policy.scope is not None:
            self.set_policy_scope(policy.id, policy.scope)
        return policy.id

    def get_policy(self, policy_id: str) -> Row | None:
        with self.conn() as conn:
            row = conn.execute("SELECT * FROM policies WHERE id = ?", (policy_id,)).fetchone()
            if row is None:
                return None
            policy = dict(row)
            policy["scope"] = self._rows(conn.execute(
                "SELECT camera_id, zone_id FROM policy_scopes WHERE policy_id = ? "
                "ORDER BY camera_id, zone_id",
                (policy_id,),
            ))
        return policy

    # --- policy scope: where a policy applies ------------------------------

    def set_policy_scope(self, policy_id: str, scope: Sequence[Any]) -> int:
        """Replace a policy's scope wholesale. ``[]`` means every camera.

        Replace rather than merge: the console and the seed file both hand over
        the complete list they intend, and a merging setter would make "remove
        this camera" a second, different call that one of the two would forget.

        Unknown camera or zone ids raise, rather than being dropped: a scope
        that silently ignored a typo would produce a policy an operator believes
        is running on Line 3 and which is running nowhere.
        """
        entries = [as_scope(entry) for entry in scope]
        with self.conn() as conn:
            conn.execute("DELETE FROM policy_scopes WHERE policy_id = ?", (policy_id,))
            for entry in entries:
                if conn.execute(
                    "SELECT 1 FROM cameras WHERE id = ?", (entry.camera_id,)
                ).fetchone() is None:
                    raise ValueError(f"no such camera: {entry.camera_id}")
                if entry.zone_id is not None:
                    zone = conn.execute(
                        "SELECT camera_id FROM zones WHERE id = ?", (entry.zone_id,)
                    ).fetchone()
                    if zone is None:
                        raise ValueError(f"no such zone: {entry.zone_id}")
                    if str(zone["camera_id"]) != entry.camera_id:
                        raise ValueError(
                            f"zone {entry.zone_id} belongs to camera {zone['camera_id']}, "
                            f"not {entry.camera_id}"
                        )
                conn.execute(
                    "INSERT OR IGNORE INTO policy_scopes "
                    "(policy_id, camera_id, zone_id, created_at) VALUES (?, ?, ?, ?)",
                    (policy_id, entry.camera_id, entry.zone_id, utcnow()),
                )
        return len(entries)

    def policy_scope(self, policy_id: str) -> list[Row]:
        with self.conn() as conn:
            return self._rows(conn.execute(
                "SELECT camera_id, zone_id FROM policy_scopes WHERE policy_id = ? "
                "ORDER BY camera_id, zone_id",
                (policy_id,),
            ))

    def _scope_map(self, conn: sqlite3.Connection) -> dict[str, list[Row]]:
        """Every policy's scope in one query — the list endpoints and both
        engines read all of them at once, and N+1 here would be one query per
        policy on every 15 s refresh."""
        scope: dict[str, list[Row]] = {}
        for row in conn.execute(
            "SELECT policy_id, camera_id, zone_id FROM policy_scopes "
            "ORDER BY policy_id, camera_id, zone_id"
        ):
            scope.setdefault(str(row["policy_id"]), []).append(
                {"camera_id": row["camera_id"], "zone_id": row["zone_id"]}
            )
        return scope

    def list_policies(
        self,
        *,
        kind: str | None = None,
        enabled_only: bool = False,
        active_only: bool = False,
    ) -> list[Row]:
        """``active_only`` = the policies that actually run against live video:
        draft policies are testable on demand but never evaluated automatically."""
        clauses: list[str] = []
        params: list[Any] = []
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        if enabled_only:
            clauses.append("enabled = 1")
        if active_only:
            clauses.append("lifecycle IN ('pilot', 'production')")
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.conn() as conn:
            rows = self._rows(
                conn.execute(f"SELECT * FROM policies{where} ORDER BY kind, label", params)  # noqa: S608 - clauses are literals, values bound
            )
            scope = self._scope_map(conn)
        # Attached rather than left to the caller: both engines and the API read
        # policies through here, and a scope that has to be fetched separately
        # is a scope one of them will eventually forget to apply.
        for row in rows:
            row["scope"] = scope.get(str(row["id"]), [])
        return rows

    def update_policy(self, policy_id: str, **fields: Any) -> int:
        return self._update("policies", "id", policy_id, {**fields, "updated_at": utcnow()})

    def delete_policy(self, policy_id: str) -> int:
        with self.conn() as conn:
            return conn.execute("DELETE FROM policies WHERE id = ?", (policy_id,)).rowcount

    # --- policy lifecycle: draft -> pilot -> production --------------------

    def start_pilot(self, policy_id: str) -> int:
        """Draft -> pilot: the policy starts running and accumulating analytics."""
        with self.conn() as conn:
            cur = conn.execute(
                "UPDATE policies SET lifecycle = 'pilot', pilot_started_at = ?, updated_at = ? "
                "WHERE id = ? AND lifecycle = 'draft'",
                (utcnow(), utcnow(), policy_id),
            )
            return cur.rowcount

    def extend_pilot(self, policy_id: str, extra_days: float = 2.0) -> int:
        with self.conn() as conn:
            cur = conn.execute(
                "UPDATE policies SET pilot_duration_days = pilot_duration_days + ?, updated_at = ? "
                "WHERE id = ? AND lifecycle = 'pilot'",
                (extra_days, utcnow(), policy_id),
            )
            return cur.rowcount

    def promote_policy(self, policy_id: str, *, stage: str = "auto_routed",
                       routed_to: str | None = None) -> int:
        """Pilot proved out -> production. Defaults to auto_routed because a
        production policy that stays silent isn't doing its job."""
        if stage not in STAGES:
            raise ValueError(f"unknown stage: {stage}")
        with self.conn() as conn:
            cur = conn.execute(
                "UPDATE policies SET lifecycle = 'production', promoted_at = ?, stage = ?, "
                "routed_to = COALESCE(?, routed_to), updated_at = ? WHERE id = ?",
                (utcnow(), stage, routed_to, utcnow(), policy_id),
            )
            return cur.rowcount

    def note_promotion_override(
        self, policy_id: str, reason: str, evidence: Mapping[str, Any] | None = None
    ) -> int:
        """Record why a policy was auto-routed without clearing the bar.

        Written as one sentence rather than as columns: what makes this useful
        six months later is the operator's own words, and the numbers at the
        time are stamped beside them so "12 reviews" cannot be re-read as
        "12 reviews *now*".
        """
        precision = (evidence or {}).get("precision")
        measured = (
            "no verdicts" if precision is None
            else f"{float(precision):.1%} over {int((evidence or {}).get('reviewed') or 0)} reviews"
        )
        note = f"{utcnow()} — promoted past the gate ({measured}): {reason}"
        return self._update("policies", "id", policy_id,
                            {"promotion_note": note, "updated_at": utcnow()})

    # --- events -----------------------------------------------------------

    def insert_event(self, event: Event) -> int:
        if event.stage not in STAGES:
            raise ValueError(f"unknown stage: {event.stage}")
        if event.severity not in SEVERITIES:
            raise ValueError(f"unknown severity: {event.severity}")
        if event.condition_state not in (None, "open", "closed"):
            raise ValueError(f"unknown condition state: {event.condition_state}")
        with self.conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO events
                    (camera_id, zone_id, policy_id, detector_name, detector_type, severity,
                     confidence, stage, routed_to, description, ts, snapshot_path,
                     clip_path, clip_start, tracks_path, meta, created_at,
                     condition_key, condition_state, observations, last_seen_ts,
                     condition_updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.camera_id, event.zone_id, event.policy_id, event.detector_name,
                    event.detector_type, event.severity, event.confidence, event.stage,
                    event.routed_to, event.description, event.ts, event.snapshot_path,
                    event.clip_path, event.clip_start, event.tracks_path,
                    json.dumps(event.meta) if event.meta is not None else None,
                    event.created_at,
                    event.condition_key, event.condition_state, 1,
                    # A condition has been observed once at the moment it opens,
                    # so `last_seen_ts` starts equal to `ts` rather than NULL —
                    # "ongoing since 20:22, last seen 20:22" is the truth about a
                    # brand-new condition, and a NULL here would make every
                    # consumer write the same COALESCE.
                    event.ts if event.condition_key else None,
                    time.time() if event.condition_key else None,
                ),
            )
            return int(cur.lastrowid)

    # --- ongoing conditions (migration 7) ---------------------------------
    #
    # Only the VLM scheduler writes these. See _M007_CONDITIONS for why a
    # condition is a row on `events` rather than a table of its own.

    def open_condition(self, condition_key: str) -> Row | None:
        """The open condition for this (policy, camera, zone), if there is one.

        Newest first and limit 1: the invariant is one open condition per key,
        and this reads the newest so a database that somehow holds two (a crash
        between the close and the insert) converges on the live one rather than
        extending a corpse forever.
        """
        with self.conn() as conn:
            row = conn.execute(
                "SELECT * FROM events WHERE condition_key = ? AND condition_state = 'open' "
                "ORDER BY id DESC LIMIT 1",
                (condition_key,),
            ).fetchone()
        return dict(row) if row else None

    def observe_condition(
        self,
        event_id: int,
        *,
        ts: float,
        description: str | None = None,
        confidence: float | None = None,
        severity: str | None = None,
        meta: Mapping[str, Any] | None = None,
        snapshot_path: str | None = None,
    ) -> Row | None:
        """Record one more sighting of a condition already on file.

        ``observations`` is bumped in SQL rather than read-modify-written in
        Python: the engine's VLM loop and a retention sweep can touch the same
        row, and `observations = observations + 1` is the only form that cannot
        lose a count to a lost update.

        Returns the row as it now stands, so the caller can log or stream it
        without a second read.
        """
        sets = ["observations = observations + 1",
                "last_seen_ts = ?", "condition_updated_at = ?"]
        params: list[Any] = [float(ts), time.time()]
        for column, value in (
            ("description", description),
            ("confidence", confidence),
            ("severity", severity),
            ("snapshot_path", snapshot_path),
        ):
            if value is not None:
                sets.append(f"{column} = ?")
                params.append(value)
        if meta is not None:
            sets.append("meta = ?")
            params.append(json.dumps(dict(meta)))
        params.append(event_id)
        with self.conn() as conn:
            conn.execute(
                # noqa: S608 - every fragment above is a literal; values are bound
                f"UPDATE events SET {', '.join(sets)} WHERE id = ? AND condition_state = 'open'",
                params,
            )
        return self.get_event(event_id)

    def close_condition(self, event_id: int, *, closed_at: float, reason: str) -> int:
        """End a condition. Idempotent: closing a closed one changes nothing."""
        with self.conn() as conn:
            cur = conn.execute(
                "UPDATE events SET condition_state = 'closed', closed_at = ?, "
                "close_reason = ?, condition_updated_at = ? "
                "WHERE id = ? AND condition_state = 'open'",
                (float(closed_at), reason, time.time(), event_id),
            )
            return int(cur.rowcount)

    def stale_conditions(self, cutoff_ts: float, *, detector_type: str | None = None) -> list[Row]:
        """Open conditions the appliance has heard nothing about since ``cutoff_ts``.

        **Measured on ``condition_updated_at`` — when the verdict was written —
        and not on ``last_seen_ts``, which is when the frame was captured.** The
        two differ by the model's latency, and using the frame clock makes the
        stale window mean "interval + latency" rather than "interval". A live
        run on 2026-08-26 found exactly that: a 20 s policy against a 15 s model
        under a 30 s window closed a perfectly healthy condition between every
        pair of samples and opened a new one each time, because 20 + 15 > 30.
        Nothing was wrong with the condition; the question was being asked
        against the wrong clock.

        ``closed_at`` is still set from ``last_seen_ts`` by the caller: the
        moment the condition was last known *true* is a fact about the plant,
        while this is a fact about the appliance.

        Read-then-close rather than one UPDATE, because the caller has to know
        *which* rows it closed: the console is holding cards for them and the
        SSE stream is what tells it they ended.

        ``detector_type`` narrows the sweep to one origin, and since there are
        two sweepers with different windows it is what keeps them off each
        other's rows. The VLM scheduler waits 15 minutes because its evidence
        arrives every few seconds at best; the engine waits two minutes because
        its arrives five times a second, and turning that window on a VLM
        condition would retire a perfectly live card between two samples.
        """
        clause, params = "", [float(cutoff_ts)]
        if detector_type is not None:
            clause = " AND detector_type = ?"
            params.append(detector_type)
        with self.conn() as conn:
            return self._rows(conn.execute(
                "SELECT * FROM events WHERE condition_state = 'open' "  # noqa: S608 - literal clause
                f"AND COALESCE(condition_updated_at, last_seen_ts, ts) < ?{clause} ORDER BY id",
                params,
            ))

    def conditions_updated_since(self, marker: float, limit: int = 200) -> list[Row]:
        """Conditions written since ``marker`` — the SSE stream's second cursor.

        The first cursor is the event id, which only ever moves forward and so
        can only announce *new* rows. An extended condition is an old row with
        new content, and this is how the feed hears about it.
        """
        with self.conn() as conn:
            return self._rows(conn.execute(
                "SELECT e.*, r.verdict AS review_verdict, r.reviewer AS reviewer, "
                "r.created_at AS reviewed_at "
                "FROM events e LEFT JOIN reviews r ON r.event_id = e.id "
                "WHERE e.condition_updated_at > ? "
                "ORDER BY e.condition_updated_at LIMIT ?",
                (float(marker), limit),
            ))

    def condition_marker(self) -> float:
        """The newest ``condition_updated_at`` on file, or 0.

        The redundant-looking ``IS NOT NULL`` is what lets the *partial* index
        answer this. Without it SQLite has no predicate matching the index's own
        WHERE clause and reads every event this appliance has ever raised — once
        per SSE connection, which on a box with a year of footage is a scan a
        reader pays for by watching the feed load.
        """
        with self.conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(condition_updated_at), 0) AS v FROM events "
                "WHERE condition_updated_at IS NOT NULL"
            ).fetchone()
        return float(row["v"] or 0.0)

    def get_event(self, event_id: int) -> Row | None:
        with self.conn() as conn:
            row = conn.execute(
                "SELECT e.*, r.verdict AS review_verdict, r.created_at AS reviewed_at "
                "FROM events e LEFT JOIN reviews r ON r.event_id = e.id WHERE e.id = ?",
                (event_id,),
            ).fetchone()
        return dict(row) if row else None

    def update_event(self, event_id: int, **fields: Any) -> int:
        if "meta" in fields and not isinstance(fields["meta"], (str, type(None))):
            fields["meta"] = json.dumps(fields["meta"])
        return self._update("events", "id", event_id, fields)

    def record_verification(
        self,
        event_id: int,
        *,
        verified_by: str,
        confidence: float | None = None,
        suppressed: bool = False,
        reason: str | None = None,
    ) -> int:
        """Record the VLM verification pass's verdict on a detector event.

        Written for both outcomes, not just vetoes. An event that was checked
        and upheld is a different thing from one that was never checked — the
        first is evidence the verifier is working, and without a row for it the
        suppression rate has no denominator.
        """
        return self._update("events", "id", event_id, {
            "suppressed": int(bool(suppressed)),
            "suppressed_reason": reason if suppressed else None,
            "verified_by": verified_by,
            "verified_at": utcnow(),
            "verify_confidence": None if confidence is None else float(confidence),
        })

    @staticmethod
    def _event_filters(
        *,
        camera_id: str | None = None,
        severity: str | None = None,
        detector: str | None = None,
        detector_type: str | None = None,
        stage: str | None = None,
        zone_id: str | None = None,
        policy_id: str | None = None,
        verdict: str | None = None,
        suppressed: bool | None = None,
        since_id: int | None = None,
        since_ts: float | None = None,
        until_ts: float | None = None,
    ) -> tuple[str, list[Any]]:
        """Shared WHERE builder for the events list and its COUNT.

        One builder rather than two: a filter that is honoured by the list but
        silently ignored by the count produces a paginator that never ends.
        """
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("e.camera_id", camera_id),
            ("e.severity", severity),
            ("e.detector_name", detector),
            ("e.detector_type", detector_type),
            ("e.stage", stage),
            ("e.zone_id", zone_id),
            ("e.policy_id", policy_id),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        if verdict is not None:
            if verdict == "unreviewed":
                clauses.append("r.verdict IS NULL")
            elif verdict in VERDICTS:
                clauses.append("r.verdict = ?")
                params.append(verdict)
            else:
                raise ValueError(f"unknown verdict filter: {verdict!r}")
        if suppressed is not None:
            # VLM-vetoed alerts stay visible: this is how the review screen
            # offers "show me what the verifier held back".
            clauses.append("e.suppressed = ?")
            params.append(int(bool(suppressed)))
        for column, value in (("e.id > ?", since_id), ("e.ts >= ?", since_ts), ("e.ts <= ?", until_ts)):
            if value is not None:
                clauses.append(column)
                params.append(value)
        return (f" WHERE {' AND '.join(clauses)}" if clauses else ""), params

    def recent_events(self, *, limit: int = 100, offset: int = 0, **filters: Any) -> list[Row]:
        """Newest first, with every filter the API exposes. See ``_event_filters``."""
        where, params = self._event_filters(**filters)
        params += [limit, offset]
        with self.conn() as conn:
            return self._rows(
                conn.execute(
                    "SELECT e.*, r.verdict AS review_verdict, r.reviewer AS reviewer, "
                    "r.created_at AS reviewed_at "
                    f"FROM events e LEFT JOIN reviews r ON r.event_id = e.id{where} "
                    "ORDER BY e.id DESC LIMIT ? OFFSET ?",
                    params,
                )
            )

    def count_events(self, **filters: Any) -> int:
        where, params = self._event_filters(**filters)
        with self.conn() as conn:
            return int(
                conn.execute(
                    "SELECT COUNT(*) AS n FROM events e "
                    f"LEFT JOIN reviews r ON r.event_id = e.id{where}",
                    params,
                ).fetchone()["n"]
            )

    def distinct_detector_names(self) -> list[str]:
        """Detector names that actually produced events — the filter dropdown."""
        with self.conn() as conn:
            return [
                r["detector_name"]
                for r in conn.execute("SELECT DISTINCT detector_name FROM events ORDER BY detector_name")
            ]

    def events_since(self, last_id: int, limit: int = 50) -> list[Row]:
        """Ascending by id — backs the API's live feed cursor."""
        with self.conn() as conn:
            return self._rows(
                conn.execute("SELECT * FROM events WHERE id > ? ORDER BY id LIMIT ?", (last_id, limit))
            )

    def max_event_id(self) -> int:
        with self.conn() as conn:
            return int(conn.execute("SELECT COALESCE(MAX(id), 0) AS v FROM events").fetchone()["v"])

    def unreviewed_events(self, limit: int = 200, *, newest_first: bool = False) -> list[Row]:
        """Triage queue: non-shadow events with no verdict yet.

        Oldest first by default — a queue is meant to be drained from the front.
        ``newest_first`` exists because on an appliance that has been running
        unattended, the front of the queue is a fortnight old: measured here
        2026-08-25 with **7 682 unreviewed events**, of which the default page
        shows the oldest 200, so today's alert on the dispatch dock sat 7 482
        rows out of reach of the screen whose job is ruling on alerts.
        """
        order = "DESC" if newest_first else "ASC"
        with self.conn() as conn:
            return self._rows(
                conn.execute(
                    "SELECT e.* FROM events e LEFT JOIN reviews r ON r.event_id = e.id "
                    f"WHERE r.id IS NULL AND e.stage != 'shadow' ORDER BY e.id {order} LIMIT ?",
                    (limit,),
                )
            )

    def unreviewed_count(self) -> int:
        """How many events are actually waiting — not how many fit on a page.

        The console's nav badge showed the length of the returned list, so it
        read ``200`` on a box with 7 682 outstanding. One indexed count is
        cheaper than the 241 KB payload the badge was downloading every 15 s to
        get the same (wrong) number.
        """
        with self.conn() as conn:
            return int(
                conn.execute(
                    "SELECT COUNT(*) FROM events e LEFT JOIN reviews r ON r.event_id = e.id "
                    "WHERE r.id IS NULL AND e.stage != 'shadow'"
                ).fetchone()[0]
            )

    # --- reviews ----------------------------------------------------------

    def record_review(self, event_id: int, verdict: str, *, reviewer: str | None = None,
                      note: str | None = None) -> int:
        """One verdict per event; re-reviewing replaces it (latest wins)."""
        if verdict not in VERDICTS:
            raise ValueError(f"unknown verdict: {verdict!r} (expected one of {VERDICTS})")
        with self.conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO reviews (event_id, verdict, reviewer, note, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (event_id) DO UPDATE SET
                    verdict    = excluded.verdict,
                    reviewer   = excluded.reviewer,
                    note       = excluded.note,
                    updated_at = excluded.updated_at
                """,
                (event_id, verdict, reviewer, note, utcnow(), utcnow()),
            )
            if cur.lastrowid:
                return int(cur.lastrowid)
            row = conn.execute("SELECT id FROM reviews WHERE event_id = ?", (event_id,)).fetchone()
            return int(row["id"])

    def get_review(self, event_id: int) -> Row | None:
        with self.conn() as conn:
            row = conn.execute("SELECT * FROM reviews WHERE event_id = ?", (event_id,)).fetchone()
        return dict(row) if row else None

    def reviewed_events_for_export(self, detector: str | None = None) -> list[Row]:
        """Every human-reviewed event with its evidence paths — the labelled
        dataset the retraining loop consumes."""
        clause, params = ("", [])
        if detector:
            clause, params = (" AND e.detector_name = ?", [detector])
        with self.conn() as conn:
            return self._rows(
                conn.execute(
                    "SELECT e.*, r.verdict, r.reviewer, r.note, r.created_at AS reviewed_at "
                    f"FROM events e JOIN reviews r ON r.event_id = e.id WHERE 1 = 1{clause} "
                    "ORDER BY r.created_at",
                    params,
                )
            )

    # --- box corrections ----------------------------------------------------

    def record_box_correction(self, event_id: int, track_id: int, verdict: str, *,
                              reviewer: str | None = None, note: str | None = None) -> int:
        """One verdict per (event, box); re-marking the same box replaces it."""
        if verdict not in BOX_VERDICTS:
            raise ValueError(f"unknown box verdict: {verdict!r} (expected one of {BOX_VERDICTS})")
        with self.conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO box_corrections
                    (event_id, track_id, verdict, reviewer, note, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (event_id, track_id) DO UPDATE SET
                    verdict    = excluded.verdict,
                    reviewer   = excluded.reviewer,
                    note       = excluded.note,
                    updated_at = excluded.updated_at
                """,
                (event_id, track_id, verdict, reviewer, note, utcnow(), utcnow()),
            )
            if cur.lastrowid:
                return int(cur.lastrowid)
            row = conn.execute(
                "SELECT id FROM box_corrections WHERE event_id = ? AND track_id = ?",
                (event_id, track_id),
            ).fetchone()
            return int(row["id"])

    def list_box_corrections(self, event_id: int) -> list[Row]:
        with self.conn() as conn:
            return self._rows(
                conn.execute(
                    "SELECT * FROM box_corrections WHERE event_id = ? ORDER BY track_id",
                    (event_id,),
                )
            )

    # --- accuracy ---------------------------------------------------------

    def detector_accuracy(self, detector: str | None = None) -> list[Row]:
        """Per-detector precision straight from triage verdicts. This is the
        number that gates promotion to auto_routed and decides what to retrain.

        ``detector`` narrows it to one. Recording a verdict returns the updated
        figure for that detector alone, and without this filter every keystroke
        in the triage queue would group over the whole events table; with it,
        ``idx_events_detector`` turns the scan into a range.
        """
        where, params = ("", [])
        if detector is not None:
            where, params = (" WHERE e.detector_name = ?", [detector])
        with self.conn() as conn:
            rows = self._rows(
                conn.execute(
                    f"""
                    SELECT e.detector_name,
                           e.detector_type,
                           COUNT(*) AS total,
                           SUM(CASE WHEN r.verdict = 'confirmed'      THEN 1 ELSE 0 END) AS confirmed,
                           SUM(CASE WHEN r.verdict = 'false_positive' THEN 1 ELSE 0 END) AS false_positive,
                           SUM(CASE WHEN r.verdict IS NULL AND e.stage != 'shadow' THEN 1 ELSE 0 END) AS pending,
                           SUM(CASE WHEN e.suppressed = 1 THEN 1 ELSE 0 END) AS suppressed,
                           SUM(CASE WHEN e.verified_by IS NOT NULL THEN 1 ELSE 0 END) AS verified
                    FROM events e LEFT JOIN reviews r ON r.event_id = e.id{where}
                    GROUP BY e.detector_name, e.detector_type
                    ORDER BY total DESC
                    """,  # noqa: S608 - `where` is a literal, its value is bound
                    params,
                )
            )
        for row in rows:
            reviewed = row["confirmed"] + row["false_positive"]
            row["reviewed"] = reviewed
            row["precision"] = (row["confirmed"] / reviewed) if reviewed else None
            row["fp_rate"] = (row["false_positive"] / reviewed) if reviewed else None
            # Share of *checked* alerts the VLM held back. Denominator is the
            # verified count, not the total: events raised while the verifier
            # was off or unreachable were never eligible for a veto, and
            # counting them would make the verifier look asleep rather than absent.
            row["suppression_rate"] = (
                (row["suppressed"] / row["verified"]) if row["verified"] else None
            )
        return rows

    def policy_analytics(self, policy_id: str) -> Row:
        """Volume, severity mix, daily trend and review outcomes for one policy —
        the promotion decision hinges on the false-positive rate here."""
        with self.conn() as conn:
            total = int(
                conn.execute(
                    "SELECT COUNT(*) AS n FROM events WHERE detector_name = ?", (policy_id,)
                ).fetchone()["n"]
            )
            by_severity = {
                r["severity"]: r["n"]
                for r in conn.execute(
                    "SELECT severity, COUNT(*) AS n FROM events WHERE detector_name = ? GROUP BY severity",
                    (policy_id,),
                )
            }
            by_day = self._rows(
                conn.execute(
                    "SELECT substr(created_at, 1, 10) AS day, COUNT(*) AS n FROM events "
                    "WHERE detector_name = ? GROUP BY day ORDER BY day",
                    (policy_id,),
                )
            )
            by_review = {
                (r["verdict"] or "unreviewed"): r["n"]
                for r in conn.execute(
                    "SELECT r.verdict, COUNT(*) AS n FROM events e "
                    "LEFT JOIN reviews r ON r.event_id = e.id "
                    "WHERE e.detector_name = ? GROUP BY r.verdict",
                    (policy_id,),
                )
            }
            recent = self.recent_events(detector=policy_id, limit=12)

        confirmed = by_review.get("confirmed", 0)
        false_positive = by_review.get("false_positive", 0)
        reviewed_total = confirmed + false_positive
        return {
            "policy_id": policy_id,
            "total": total,
            "by_severity": by_severity,
            "by_day": by_day,
            "by_review": by_review,
            "reviewed_total": reviewed_total,
            "unreviewed": by_review.get("unreviewed", 0),
            "false_positive_rate": (false_positive / reviewed_total) if reviewed_total else None,
            "recent_events": recent,
        }

    def review_trend(self, days: int = 14) -> list[Row]:
        with self.conn() as conn:
            rows = self._rows(
                conn.execute(
                    """
                    SELECT substr(created_at, 1, 10) AS day,
                           SUM(CASE WHEN verdict = 'confirmed'      THEN 1 ELSE 0 END) AS confirmed,
                           SUM(CASE WHEN verdict = 'false_positive' THEN 1 ELSE 0 END) AS false_positive
                    FROM reviews GROUP BY day ORDER BY day DESC LIMIT ?
                    """,
                    (days,),
                )
            )
        return list(reversed(rows))

    # --- notifications ----------------------------------------------------

    def log_notification(self, event_id: int, channel: str, target: str, status: str,
                         detail: str | None = None) -> int:
        if status not in ("sent", "failed"):
            raise ValueError(f"unknown notification status: {status}")
        with self.conn() as conn:
            cur = conn.execute(
                "INSERT INTO notifications (event_id, channel, target, status, detail, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (event_id, channel, target, status, detail, utcnow()),
            )
            return int(cur.lastrowid)

    def notifications_for_event(self, event_id: int) -> list[Row]:
        with self.conn() as conn:
            return self._rows(
                conn.execute(
                    "SELECT * FROM notifications WHERE event_id = ? ORDER BY created_at", (event_id,)
                )
            )

    def recent_notifications(self, limit: int = 50) -> list[Row]:
        with self.conn() as conn:
            return self._rows(
                conn.execute(
                    "SELECT n.*, e.detector_name, e.severity, e.description FROM notifications n "
                    "JOIN events e ON e.id = n.event_id ORDER BY n.id DESC LIMIT ?",
                    (limit,),
                )
            )

    # --- occupancy --------------------------------------------------------

    def insert_occupancy(self, samples: Sequence[Mapping[str, Any]]) -> int:
        """Batched insert of ``{camera_id, zone_id, ts, count}`` samples."""
        if not samples:
            return 0
        created = utcnow()
        payload = [
            (s["camera_id"], s.get("zone_id"), float(s["ts"]), int(s["count"]), created)
            for s in samples
        ]
        with self.conn() as conn:
            conn.executemany(
                "INSERT INTO occupancy (camera_id, zone_id, ts, count, created_at) VALUES (?, ?, ?, ?, ?)",
                payload,
            )
        return len(payload)

    def occupancy_series(self, *, camera_id: str | None = None, zone_id: str | None = None,
                         since_ts: float | None = None, limit: int = 2000) -> list[Row]:
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (("camera_id", camera_id), ("zone_id", zone_id)):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        if since_ts is not None:
            clauses.append("ts >= ?")
            params.append(since_ts)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        with self.conn() as conn:
            rows = self._rows(
                conn.execute(
                    f"SELECT camera_id, zone_id, ts, count FROM occupancy{where} ORDER BY ts DESC LIMIT ?",
                    params,
                )
            )
        return list(reversed(rows))

    def latest_zone_counts(self, camera_id: str) -> dict[str, Row]:
        """Newest occupancy sample per zone for a camera (live-wall badges)."""
        with self.conn() as conn:
            rows = self._rows(
                conn.execute(
                    "SELECT zone_id, count, MAX(ts) AS ts FROM occupancy "
                    "WHERE camera_id = ? AND zone_id IS NOT NULL GROUP BY zone_id",
                    (camera_id,),
                )
            )
        return {row["zone_id"]: {"count": row["count"], "ts": row["ts"]} for row in rows}

    # --- saved searches ---------------------------------------------------
    #
    # The queries an operator keeps, and the record of which ones became
    # policies. See migration 3 for why they live here rather than in the
    # vector store.

    def create_saved_search(
        self, label: str, query: str, *, camera_id: str | None = None,
        window_hours: float | None = None,
    ) -> int:
        """Save a search. Re-saving the same query for the same camera renames
        it rather than making a second chip in the UI that does the same thing."""
        with self.conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO saved_searches
                    (label, query, camera_id, window_hours, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (query, COALESCE(camera_id, '')) DO UPDATE SET
                    label        = excluded.label,
                    window_hours = excluded.window_hours,
                    updated_at   = excluded.updated_at
                """,
                (label, query, camera_id, window_hours, utcnow(), utcnow()),
            )
            if cur.lastrowid:
                return int(cur.lastrowid)
            row = conn.execute(
                "SELECT id FROM saved_searches WHERE query = ? AND COALESCE(camera_id, '') = ?",
                (query, camera_id or ""),
            ).fetchone()
            return int(row["id"])

    def list_saved_searches(self) -> list[Row]:
        """Most recently run first, then newest — the chip order in the UI, so
        the search somebody uses every shift stays under their cursor."""
        with self.conn() as conn:
            return self._rows(conn.execute(
                "SELECT s.*, p.label AS policy_label, p.lifecycle AS policy_lifecycle "
                "FROM saved_searches s LEFT JOIN policies p ON p.id = s.policy_id "
                "ORDER BY COALESCE(s.last_run_at, s.created_at) DESC, s.id DESC"
            ))

    def get_saved_search(self, search_id: int) -> Row | None:
        with self.conn() as conn:
            row = conn.execute(
                "SELECT * FROM saved_searches WHERE id = ?", (search_id,)
            ).fetchone()
        return dict(row) if row else None

    def update_saved_search(self, search_id: int, **fields: Any) -> int:
        return self._update("saved_searches", "id", search_id, {**fields, "updated_at": utcnow()})

    def delete_saved_search(self, search_id: int) -> int:
        with self.conn() as conn:
            return conn.execute("DELETE FROM saved_searches WHERE id = ?", (search_id,)).rowcount

    def note_saved_search_run(self, search_id: int) -> int:
        """Count a use. Its own statement rather than a field on the PATCH
        allowlist, so a client cannot set the usage history it is measured by."""
        with self.conn() as conn:
            return conn.execute(
                "UPDATE saved_searches SET run_count = run_count + 1, last_run_at = ? WHERE id = ?",
                (utcnow(), search_id),
            ).rowcount

    # --- accounts ---------------------------------------------------------
    #
    # The password hash never leaves this section. Every read that a router or
    # the CLI can reach goes through `_USER_PUBLIC`, so "the roster endpoint
    # leaked the hashes" is not a mistake that can be made one field at a time.

    def create_user(self, username: str, display_name: str, role: str,
                    password_hash: str) -> int:
        """Insert an account. Raises ``sqlite3.IntegrityError`` on a duplicate
        username — the caller turns that into the 409 it means."""
        now = utcnow()
        with self.conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO users (username, display_name, role, password_hash,
                                   disabled, created_at, updated_at)
                VALUES (?, ?, ?, ?, 0, ?, ?)
                """,
                (username, display_name, role, password_hash, now, now),
            )
            return int(cur.lastrowid or 0)

    def get_user(self, user_id: int) -> Row | None:
        with self.conn() as conn:
            row = conn.execute(f"SELECT {_USER_PUBLIC} FROM users WHERE id = ?",  # noqa: S608
                               (user_id,)).fetchone()
            return dict(row) if row else None

    def get_user_by_username(self, username: str) -> Row | None:
        """Public columns only. The login path wants the hash and calls
        :meth:`credentials_for` instead, which is the only way to get it."""
        with self.conn() as conn:
            row = conn.execute(
                f"SELECT {_USER_PUBLIC} FROM users WHERE username = ?",  # noqa: S608
                (username,),
            ).fetchone()
            return dict(row) if row else None

    def credentials_for(self, username: str) -> Row | None:
        """The one query that returns ``password_hash``. Used by login and by
        ``argus user passwd``; named so it stands out in a grep."""
        with self.conn() as conn:
            row = conn.execute(
                "SELECT id, username, display_name, role, password_hash, disabled "
                "FROM users WHERE username = ?",
                (username,),
            ).fetchone()
            return dict(row) if row else None

    def list_users(self, *, include_disabled: bool = True) -> list[Row]:
        where = "" if include_disabled else " WHERE disabled = 0"
        with self.conn() as conn:
            return self._rows(conn.execute(
                f"SELECT {_USER_PUBLIC} FROM users{where} "  # noqa: S608
                f"ORDER BY disabled, username"
            ))

    def count_users(self, *, enabled_only: bool = False, role: str | None = None) -> int:
        clauses, params = [], []
        if enabled_only:
            clauses.append("disabled = 0")
        if role is not None:
            clauses.append("role = ?")
            params.append(role)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.conn() as conn:
            return int(conn.execute(
                f"SELECT COUNT(*) AS n FROM users{where}", params  # noqa: S608
            ).fetchone()["n"])

    def update_user(self, user_id: int, **fields: Any) -> int:
        return self._update("users", "id", user_id, {**fields, "updated_at": utcnow()})

    def set_password(self, user_id: int, password_hash: str) -> int:
        """Change a password and cut every session that used the old one.

        Both halves in one transaction: a password change that leaves the old
        browser logged in has not changed anything an attacker cares about.
        """
        with self.conn() as conn:
            changed = conn.execute(
                "UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?",
                (password_hash, utcnow(), user_id),
            ).rowcount
            if changed:
                conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            return changed

    def note_login(self, user_id: int) -> None:
        with self.conn() as conn:
            conn.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (utcnow(), user_id))

    def delete_user(self, user_id: int) -> int:
        """Present for completeness and used by tests; the product path is
        ``disabled = 1``, because deleting the account named in a year of
        ``reviews.reviewer`` rows makes the audit trail unreadable."""
        with self.conn() as conn:
            return conn.execute("DELETE FROM users WHERE id = ?", (user_id,)).rowcount

    # --- sessions ---------------------------------------------------------

    def create_session(self, session_id: str, user_id: int, expires_at: str, *,
                       user_agent: str | None = None, ip: str | None = None) -> str:
        now = utcnow()
        with self.conn() as conn:
            conn.execute(
                """
                INSERT INTO sessions (id, user_id, created_at, expires_at, last_seen_at,
                                      user_agent, ip)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, user_id, now, expires_at, now, user_agent, ip),
            )
        return session_id

    def get_session(self, session_id: str) -> Row | None:
        """A session joined to its user, in one query.

        One round trip because this runs on *every* request including the SSE
        poll and every evidence image on a triage screen — and because a session
        whose user was disabled a second ago must not survive on a cached copy
        of the roster.
        """
        with self.conn() as conn:
            row = conn.execute(
                """
                SELECT s.id AS session_id, s.user_id, s.created_at, s.expires_at,
                       s.last_seen_at, s.ip, s.user_agent,
                       u.username, u.display_name, u.role, u.disabled
                FROM sessions s JOIN users u ON u.id = s.user_id
                WHERE s.id = ?
                """,
                (session_id,),
            ).fetchone()
            return dict(row) if row else None

    def touch_session(self, session_id: str) -> int:
        with self.conn() as conn:
            return conn.execute(
                "UPDATE sessions SET last_seen_at = ? WHERE id = ?", (utcnow(), session_id)
            ).rowcount

    def delete_session(self, session_id: str) -> int:
        with self.conn() as conn:
            return conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,)).rowcount

    def delete_sessions_for_user(self, user_id: int) -> int:
        """Every browser this account is logged in from. What "disable" means."""
        with self.conn() as conn:
            return conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,)).rowcount

    def list_sessions_for_user(self, user_id: int) -> list[Row]:
        with self.conn() as conn:
            return self._rows(conn.execute(
                "SELECT id, created_at, expires_at, last_seen_at, ip, user_agent "
                "FROM sessions WHERE user_id = ? ORDER BY last_seen_at DESC",
                (user_id,),
            ))

    def purge_expired_sessions(self, now_iso: str | None = None) -> int:
        with self.conn() as conn:
            return conn.execute(
                "DELETE FROM sessions WHERE expires_at <= ?", (now_iso or utcnow(),)
            ).rowcount

    # --- login attempts (the rate limiter's memory) -----------------------

    def record_login_attempt(self, username: str, ip: str, ok: bool) -> None:
        with self.conn() as conn:
            conn.execute(
                "INSERT INTO login_attempts (username, ip, ok, created_at) VALUES (?, ?, ?, ?)",
                (username, ip, 1 if ok else 0, utcnow()),
            )

    def failed_login_count(self, since_iso: str, *, username: str | None = None,
                           ip: str | None = None) -> int:
        """Failures in a window, for a username, an ip, or either.

        Both dimensions are counted separately by the caller: locking only on
        username lets one attacker lock every operator out of the plant, and
        locking only on ip lets a botnet spray one password across the roster.
        """
        clauses = ["ok = 0", "created_at >= ?"]
        params: list[Any] = [since_iso]
        if username is not None:
            clauses.append("username = ?")
            params.append(username)
        if ip is not None:
            clauses.append("ip = ?")
            params.append(ip)
        with self.conn() as conn:
            return int(conn.execute(
                f"SELECT COUNT(*) AS n FROM login_attempts WHERE {' AND '.join(clauses)}",  # noqa: S608
                params,
            ).fetchone()["n"])

    def clear_login_failures(self, username: str, ip: str) -> int:
        """Called on a successful login: the operator who fat-fingered their
        password four times must not stay locked out after getting it right."""
        with self.conn() as conn:
            return conn.execute(
                "DELETE FROM login_attempts WHERE ok = 0 AND username = ? AND ip = ?",
                (username, ip),
            ).rowcount

    def purge_login_attempts(self, before_iso: str) -> int:
        with self.conn() as conn:
            return conn.execute(
                "DELETE FROM login_attempts WHERE created_at < ?", (before_iso,)
            ).rowcount

    def recent_login_attempts(self, limit: int = 50) -> list[Row]:
        with self.conn() as conn:
            return self._rows(conn.execute(
                "SELECT username, ip, ok, created_at FROM login_attempts "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ))

    # --- retention --------------------------------------------------------

    def all_event_ids(self) -> list[int]:
        """Every live event id — what the index prunes its orphans against.

        Returned whole rather than paged: at the 180-day window this is a few
        hundred thousand integers in the worst case, and the alternative (a
        cross-file join SQLite cannot do) is no alternative.
        """
        with self.conn() as conn:
            return [int(row["id"]) for row in conn.execute("SELECT id FROM events")]

    def media_tiers(self, column: str) -> dict[str, str]:
        """Every path in an evidence column, mapped to its **retention tier**.

        The tier is what the event turned out to be, resolved verdict-first: a
        human's ``confirmed`` / ``false_positive`` beats the policy's stage,
        because a verdict is a judgement about *this* event and a stage is a
        promise about the policy that raised it. With no verdict the stage
        stands — ``shadow``, ``reviewed``, ``auto_routed``.

        This is what makes stage-tiered retention possible without a schema
        change: the sweeper still walks the filesystem by mtime (so it still
        collects orphans no row points at) and consults this map to decide how
        long each file was entitled to live. Paths absent from it are orphans,
        and the caller gives them the flat window.

        Returned whole for the same reason as ``all_event_ids``: the alternative
        is a query per file, and there are 40 000 of them.

        Two events pointing at one file is not expected — clip names carry the
        event id — but a re-review or a hand-edited row could do it, so the
        strongest claim wins rather than whichever row SQLite yielded last.
        Biasing a tie toward keeping is the only safe direction: the cost of
        being wrong is a few MB, and the cost of the other direction is the
        evidence for a confirmed incident.
        """
        if column not in _MEDIA_COLUMNS:
            raise ValueError(f"not an evidence column: {column}")
        tiers: dict[str, str] = {}
        with self.conn() as conn:
            rows = conn.execute(
                f'SELECT e."{column}" AS path, e.stage AS stage, r.verdict AS verdict '  # noqa: S608 - column allowlisted above
                f'FROM events e LEFT JOIN reviews r ON r.event_id = e.id '
                f'WHERE e."{column}" IS NOT NULL'
            )
            for row in rows:
                path = str(row["path"])
                tier = str(row["verdict"] or row["stage"] or "shadow")
                current = tiers.get(path)
                if current is None or _TIER_CLAIM.get(tier, 0) > _TIER_CLAIM.get(current, 0):
                    tiers[path] = tier
        return tiers

    def clear_media_paths(self, column: str, paths: Sequence[str]) -> int:
        """Forget evidence files that retention has deleted.

        An event row outlives its clip (180 days vs 14), so once the file is
        gone the column has to go with it — otherwise the API keeps serving a
        path that 404s and a reviewer is told evidence exists when it does not.
        ``clip_start`` is cleared alongside ``clip_path`` because it describes a
        clip that no longer exists.
        """
        if column not in _MEDIA_COLUMNS:
            raise ValueError(f"not an evidence column: {column}")
        if not paths:
            return 0
        extra = ", clip_start = NULL" if column == "clip_path" else ""
        cleared = 0
        with self.conn() as conn:
            # Chunked so the IN list stays well inside SQLite's variable limit.
            for start in range(0, len(paths), 400):
                chunk = list(paths[start:start + 400])
                placeholders = ", ".join("?" * len(chunk))
                cleared += conn.execute(
                    f'UPDATE events SET "{column}" = NULL{extra} '  # noqa: S608 - column allowlisted above
                    f'WHERE "{column}" IN ({placeholders})',
                    chunk,
                ).rowcount
        return cleared

    def purge_older_than(self, table: str, column: str, cutoff_iso: str) -> int:
        """Delete rows older than a cutoff. Table/column are checked against a
        fixed map so this can never become an injection point."""
        allowed = {"events": "created_at", "occupancy": "created_at", "notifications": "created_at"}
        if allowed.get(table) != column:
            raise ValueError(f"purge not allowed for {table}.{column}")
        with self.conn() as conn:
            return conn.execute(
                f'DELETE FROM "{table}" WHERE "{column}" < ?', (cutoff_iso,)  # noqa: S608 - allowlisted above
            ).rowcount

    # --- aggregates for the analytics screen ------------------------------
    #
    # Three GROUP BYs and a latency list, all bounded by a time window. They
    # live here rather than in the API router for the same reason every other
    # query does: SQL belongs next to the schema it depends on, and the router
    # stays a shape that pytest can assert against without a database.
    #
    # Days and hours are bucketed by the appliance's *local* clock
    # ('localtime'), not UTC. A shift that starts at 06:00 starts at 06:00 on
    # the wall, and a supervisor asking about "yesterday" means the local day.

    @staticmethod
    def _window(since_ts: float | None, until_ts: float | None) -> tuple[str, list[Any]]:
        clauses, params = [], []
        for column, value in (("e.ts >= ?", since_ts), ("e.ts <= ?", until_ts)):
            if value is not None:
                clauses.append(column)
                params.append(value)
        return (f" WHERE {' AND '.join(clauses)}" if clauses else ""), params

    # Two numbers per bucket since migration 7, and the difference is the whole
    # point of that migration. `raw` counts **rows** — an incident, however long
    # it lasted. `observations` counts the sampled verdicts behind them, which
    # for a VLM condition is how many times the model saw the same mess. One
    # laminate offcut lying on an apron all night is 1 raw and 40 observations,
    # and a trend chart that plots 40 is describing the sampling interval rather
    # than the plant. Both are reported because neither alone is honest: the
    # first hides how long a thing went on, the second invents incidents.
    #
    # COALESCE because a detector event predates the column and every row
    # written before migration 7 has the default of 1 — which is right for both:
    # a detector event *is* one observation.

    def events_by_day(self, *, since_ts: float | None = None,
                      until_ts: float | None = None) -> list[Row]:
        """Raw detections and their review outcomes, one row per local day."""
        where, params = self._window(since_ts, until_ts)
        with self.conn() as conn:
            return self._rows(conn.execute(
                f"""
                SELECT strftime('%Y-%m-%d', e.ts, 'unixepoch', 'localtime') AS day,
                       COUNT(*) AS raw,
                       SUM(COALESCE(e.observations, 1)) AS observations,
                       SUM(CASE WHEN r.verdict = 'confirmed'      THEN 1 ELSE 0 END) AS confirmed,
                       SUM(CASE WHEN r.verdict = 'false_positive' THEN 1 ELSE 0 END) AS false_positive
                FROM events e LEFT JOIN reviews r ON r.event_id = e.id{where}
                GROUP BY day ORDER BY day
                """,  # noqa: S608 - `where` is built from literals, values are bound
                params,
            ))

    def events_by_type(self, *, since_ts: float | None = None,
                       until_ts: float | None = None) -> list[Row]:
        """Same split, grouped by what fired. The policy's label comes along so
        the UI does not have to join it back on the client."""
        where, params = self._window(since_ts, until_ts)
        with self.conn() as conn:
            return self._rows(conn.execute(
                f"""
                SELECT e.detector_name,
                       e.detector_type,
                       COALESCE(p.label, e.detector_name) AS label,
                       COUNT(*) AS raw,
                       SUM(COALESCE(e.observations, 1)) AS observations,
                       SUM(CASE WHEN r.verdict = 'confirmed'      THEN 1 ELSE 0 END) AS confirmed,
                       SUM(CASE WHEN r.verdict = 'false_positive' THEN 1 ELSE 0 END) AS false_positive
                FROM events e
                LEFT JOIN reviews r  ON r.event_id = e.id
                LEFT JOIN policies p ON p.id = e.policy_id{where}
                GROUP BY e.detector_name, e.detector_type, label
                ORDER BY raw DESC
                """,  # noqa: S608
                params,
            ))

    def events_by_zone_hour(self, *, since_ts: float | None = None,
                            until_ts: float | None = None) -> list[Row]:
        """Zone x local hour. The caller folds hours into shifts, because the
        shift rota is configuration and this is the database."""
        where, params = self._window(since_ts, until_ts)
        with self.conn() as conn:
            return self._rows(conn.execute(
                f"""
                SELECT e.zone_id,
                       e.camera_id,
                       z.name AS zone_name,
                       CAST(strftime('%H', e.ts, 'unixepoch', 'localtime') AS INTEGER) AS hour,
                       COUNT(*) AS raw,
                       SUM(COALESCE(e.observations, 1)) AS observations,
                       SUM(CASE WHEN r.verdict = 'confirmed' THEN 1 ELSE 0 END) AS confirmed
                FROM events e
                LEFT JOIN reviews r ON r.event_id = e.id
                LEFT JOIN zones z   ON z.id = e.zone_id{where}
                GROUP BY e.zone_id, e.camera_id, zone_name, hour
                ORDER BY zone_name, hour
                """,  # noqa: S608
                params,
            ))

    def events_by_type_hour(self, *, since_ts: float | None = None,
                            until_ts: float | None = None) -> list[Row]:
        """Policy x local hour — the per-policy analogue of
        ``events_by_zone_hour``. The caller folds hours into shifts, because
        the shift rota is configuration and this is the database."""
        where, params = self._window(since_ts, until_ts)
        with self.conn() as conn:
            return self._rows(conn.execute(
                f"""
                SELECT e.detector_name,
                       CAST(strftime('%H', e.ts, 'unixepoch', 'localtime') AS INTEGER) AS hour,
                       COUNT(*) AS raw,
                       SUM(COALESCE(e.observations, 1)) AS observations,
                       SUM(CASE WHEN r.verdict = 'confirmed' THEN 1 ELSE 0 END) AS confirmed
                FROM events e
                LEFT JOIN reviews r ON r.event_id = e.id{where}
                GROUP BY e.detector_name, hour
                ORDER BY e.detector_name, hour
                """,  # noqa: S608
                params,
            ))

    def events_by_type_severity(self, *, since_ts: float | None = None,
                                until_ts: float | None = None) -> list[Row]:
        """Policy x severity — real severity (low/medium/high), not a
        per-policy invention: every event already carries one."""
        where, params = self._window(since_ts, until_ts)
        with self.conn() as conn:
            return self._rows(conn.execute(
                f"""
                SELECT e.detector_name, e.severity, COUNT(*) AS n
                FROM events e{where}
                GROUP BY e.detector_name, e.severity
                """,  # noqa: S608
                params,
            ))

    def events_by_type_day(self, *, since_ts: float | None = None,
                           until_ts: float | None = None) -> list[Row]:
        """Policy x local day — the per-policy trend line. Same grouping as
        ``events_by_day``, with the policy added so one query covers every
        card instead of one round trip per policy."""
        where, params = self._window(since_ts, until_ts)
        with self.conn() as conn:
            return self._rows(conn.execute(
                f"""
                SELECT e.detector_name,
                       strftime('%Y-%m-%d', e.ts, 'unixepoch', 'localtime') AS day,
                       COUNT(*) AS raw,
                       SUM(CASE WHEN r.verdict = 'confirmed' THEN 1 ELSE 0 END) AS confirmed
                FROM events e LEFT JOIN reviews r ON r.event_id = e.id{where}
                GROUP BY e.detector_name, day
                ORDER BY e.detector_name, day
                """,  # noqa: S608
                params,
            ))

    def review_latencies(self, *, since_ts: float | None = None,
                         until_ts: float | None = None) -> list[Row]:
        """``(event ts, verdict recorded at)`` for reviews of events in the
        window. Returned as pairs rather than a median because the arithmetic is
        clearer in Python than in SQLite, which has no median function."""
        where, params = self._window(since_ts, until_ts)
        with self.conn() as conn:
            return self._rows(conn.execute(
                f"SELECT e.ts AS event_ts, r.created_at AS reviewed_at "
                f"FROM reviews r JOIN events e ON e.id = r.event_id{where}",  # noqa: S608
                params,
            ))

    def event_totals(self, *, since_ts: float | None = None,
                     until_ts: float | None = None) -> Row:
        """Headline counts for a window: volume, verdicts, and what is still
        waiting on a human."""
        where, params = self._window(since_ts, until_ts)
        with self.conn() as conn:
            row = conn.execute(
                f"""
                SELECT COUNT(*) AS events,
                       SUM(COALESCE(e.observations, 1)) AS observations,
                       SUM(CASE WHEN e.condition_state = 'open' THEN 1 ELSE 0 END) AS ongoing,
                       SUM(CASE WHEN r.verdict = 'confirmed'      THEN 1 ELSE 0 END) AS confirmed,
                       SUM(CASE WHEN r.verdict = 'false_positive' THEN 1 ELSE 0 END) AS false_positive,
                       SUM(CASE WHEN r.verdict IS NULL AND e.stage != 'shadow'
                                THEN 1 ELSE 0 END) AS pending,
                       SUM(CASE WHEN r.verdict IS NULL AND e.stage != 'shadow'
                                     AND e.severity = 'high' THEN 1 ELSE 0 END) AS open_high
                FROM events e LEFT JOIN reviews r ON r.event_id = e.id{where}
                """,  # noqa: S608
                params,
            ).fetchone()
        totals = {key: int(row[key] or 0) for key in
                  ("events", "observations", "ongoing", "confirmed", "false_positive",
                   "pending", "open_high")}
        reviewed = totals["confirmed"] + totals["false_positive"]
        totals["reviewed"] = reviewed
        totals["precision"] = (totals["confirmed"] / reviewed) if reviewed else None
        return totals

    def counts(self) -> dict[str, int]:
        """Row counts per table — cheap health/state summary for `argus run`."""
        tables = ("cameras", "zones", "policies", "events", "reviews", "notifications",
                  "occupancy", "saved_searches", "users", "sessions")
        with self.conn() as conn:
            return {t: int(conn.execute(f'SELECT COUNT(*) AS n FROM "{t}"').fetchone()["n"]) for t in tables}


def _unique_id(conn: sqlite3.Connection, table: str, base: str) -> str:
    """Suffix a slug until it is free. ``table`` is caller-controlled, never user
    input — the two call sites pass literals."""
    if table not in ("cameras", "zones", "policies"):
        raise ValueError(f"unexpected table: {table}")
    candidate, suffix = base, 2
    while conn.execute(f'SELECT 1 FROM "{table}" WHERE id = ?', (candidate,)).fetchone():
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def open_database(config: Any = None) -> Database:
    """Open (and migrate) the appliance database described by a config."""
    if config is None:
        from argus.config import load_config

        config = load_config()
    database = Database(config.db_path)
    database.migrate()
    return database
