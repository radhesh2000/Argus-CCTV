"""Bounded storage. The one thing v1 did not do.

v1 recorded continuously, kept everything, and produced 186 GB from a demo
loop — on an appliance that has to run unattended for years in a plant with
nobody watching the disk. Retention is therefore not a nice-to-have here; it is
the difference between a product and a science project.

Three mechanisms, because two were not enough:

* **The age sweep** applies the configured windows (snapshots 30 d, events
  180 d, occupancy 7 d), and for evidence it applies them **per tier** — how
  long a clip or a snapshot lives depends on what its event turned out to be.
  Evidence is deleted from disk *and* the event rows that referenced it are
  updated, so the API never serves a path to a file that is gone.
* **The evidence budget** (``retention.max_clips_gb`` /
  ``max_snapshots_gb``) is a hard ceiling, enforced by evicting the least
  valuable evidence first. This is the mechanism the other two do not have:
  ages bound how *long* a site keeps evidence, and nothing bounded how *much*.
* **The disk guard** reclaims oldest-first across every directory whenever the
  filesystem crosses ``retention.max_disk_percent`` — the backstop for a box
  where the other two were configured wrong, or where something that is not
  Argus filled the same disk. Nothing younger than ``min_keep_hours`` is ever
  touched by it, so an alert raised ten minutes ago always still has its clip.

## Why tiering and a budget, and not just a shorter window

Measured on this appliance, 2026-08-29: ``data/clips`` reached **578 GB in
2.5 days**. Six file cameras loop the same plant footage, so a zone violation
never clears; 47 881 events were cut, each with a 19.4 MB H.264 clip (3.1x the
pre-Phase-6 stream-copied size, a re-costing ``docs/PHASE6_NOTES.md`` flagged
and did not do). The only bound was ``clips_days: 14`` and a 94.9% disk guard
that could not have engaged before 3.5 TB. Growth: ~250 GB/day.

Shortening ``clips_days`` would have fixed that day and lost the evidence for
every confirmed incident along with the noise, because a flat window cannot
tell them apart. Tiering can: 47 875 of those events were unreviewed
shadow-stage repeats and 164 had a human verdict. The budget is the part that
does not depend on getting the ages right — a site whose event rate is 20x what
anyone predicted is a volume problem, and only a volume ceiling answers it.

All of it is ordinary Python — no cron, no systemd timer — running on a daemon
thread inside ``argus run``, because an appliance whose housekeeping depends on
an external scheduler is an appliance that ships with a way to fail silently.

The same thread carries the two jobs that are the *other* half of data safety,
for exactly that reason (see ``argus/backup.py``):

* **a nightly SQLite backup**, taken when one is owed rather than at a wall
  clock time, so a box that was powered down through its window still gets one;
* **log rotation** for the two files that are not on journald — go2rtc's and
  the vLLM server's.

They are one thread rather than three because they are one policy: everything
that keeps this appliance's disk and database healthy happens on the same tick,
in an order that makes sense (reclaim first, then back up what survived), and
fails into the same log line.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, NamedTuple

from argus import backup
from argus.config import ArgusConfig
from argus.db import Database
from argus.health import disk_usage, iter_files

# How much below max_disk_percent the emergency pass tries to get. Reclaiming
# to exactly the threshold means the next sweep fires again immediately.
_RECLAIM_MARGIN = 5.0

# Re-stat the filesystem every N deletions during the emergency pass rather
# than after each file: statvfs is a syscall, and deleting one 2 MB segment
# never moves a percentage point on a 3.7 TB disk.
_DISK_RECHECK_EVERY = 40

#: Every tier an event can resolve to, in the order the plan is printed. The
#: *retention* order is not this list — it is whatever windows the operator
#: configured (see `_tier_days`), so there is one place that says what evidence
#: is worth and it is the config file.
TIERS = ("shadow", "reviewed", "auto_routed", "confirmed", "false_positive")

#: Sort rank for a file no event row points at. Below every tier, always: an
#: orphan is a clip whose event was purged, or one a killed ffmpeg left behind,
#: and evidence for nothing is the first thing a budget should reclaim. It is
#: still protected by `min_keep_hours`, which is what covers the one orphan that
#: matters — a clip being written right now, whose row lands a second later.
_ORPHAN_RANK = -1.0


def _is_evidence(path: Path) -> bool:
    """Whether the sweeper owns this file.

    It owns everything the appliance writes and nothing anybody else put there,
    and the only such files in practice are the dot-files that mark a directory:
    ``data/clips/.gitkeep`` is how the repository ships an empty evidence tree,
    and the first sweep on this box that actually deleted anything at scale
    deleted it — the walk is by mtime and a marker file is old and belongs to no
    event, which is precisely the orphan signature. Nothing Argus writes starts
    with a dot (see argus/media/*, which names clips and frames after their
    event and camera), so this costs no evidence and stops the housekeeping job
    from reaching outside its own job.
    """
    return not path.name.startswith(".")


def _walk(directory: Path):
    """``iter_files`` narrowed to the files this job is allowed to delete. One
    function so the age sweep, the budget and the disk guard cannot disagree
    about what counts as evidence."""
    return (entry for entry in iter_files(directory) if _is_evidence(entry[2]))


class MediaTarget(NamedTuple):
    """One directory the sweeper owns: how long its files live, which events
    column points at them, whether its subdirectories may be removed, and the
    hard ceiling on its total size (0 = none)."""

    kind: str
    directory: Path
    days: float
    column: str | None
    prune_dirs: bool = True
    budget_gb: float = 0.0


@dataclass
class SweepResult:
    """What one sweep reclaimed. Logged verbatim, and returned by the API."""

    files_deleted: int = 0
    bytes_deleted: int = 0
    rows_deleted: int = 0
    paths_cleared: int = 0
    # Vector-index rows dropped because the thing they pointed at is gone.
    # Counted separately from `rows_deleted` because they live in a different
    # file, and a number that mixed the two would make it impossible to tell
    # "the index is being maintained" from "events are ageing out".
    index_rows_deleted: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)
    # One entry per media kind whose budget was exceeded this sweep: what it
    # held, what it was allowed, and what eviction actually got back. Empty on
    # a healthy appliance, which is the point — this dict being non-empty is
    # itself the signal that the age windows are not holding the site.
    budgets: dict[str, dict[str, Any]] = field(default_factory=dict)
    emergency: bool = False
    disk_before: float | None = None
    disk_after: float | None = None
    duration_sec: float = 0.0
    errors: list[str] = field(default_factory=list)

    def note(self, kind: str, count: int = 1) -> None:
        if count:
            self.by_kind[kind] = self.by_kind.get(kind, 0) + count

    @property
    def mb_deleted(self) -> float:
        return round(self.bytes_deleted / 1e6, 1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "files_deleted": self.files_deleted,
            "mb_deleted": self.mb_deleted,
            "rows_deleted": self.rows_deleted,
            "paths_cleared": self.paths_cleared,
            "index_rows_deleted": self.index_rows_deleted,
            "by_kind": self.by_kind,
            "budgets": self.budgets,
            "emergency": self.emergency,
            "disk_before_percent": self.disk_before,
            "disk_after_percent": self.disk_after,
            "duration_sec": round(self.duration_sec, 2),
            "errors": self.errors,
        }

    def __str__(self) -> str:
        if not self.files_deleted and not self.rows_deleted and not self.index_rows_deleted:
            return f"retention: nothing to reclaim (disk {self.disk_after}% full)"
        detail = ", ".join(f"{kind}={count}" for kind, count in sorted(self.by_kind.items()))
        prefix = "retention[disk-guard]" if self.emergency else "retention"
        if self.budgets and not self.emergency:
            prefix = "retention[budget]"
        index = f", {self.index_rows_deleted} index row(s)" if self.index_rows_deleted else ""
        return (
            f"{prefix}: reclaimed {self.mb_deleted} MB in {self.files_deleted} file(s), "
            f"{self.rows_deleted} row(s) purged, {self.paths_cleared} path(s) cleared"
            f"{index} ({detail}) — disk {self.disk_before}% -> {self.disk_after}%, "
            f"{self.duration_sec:.1f}s"
        )


def _cutoff_iso(days: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def _unlink(path: Path, result: SweepResult, kind: str) -> bool:
    """Delete one file, counting its size. Returns whether it went away."""
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    except OSError as exc:
        if len(result.errors) < 10:
            result.errors.append(f"{path.name}: {exc}")
        return False
    result.files_deleted += 1
    result.bytes_deleted += size
    result.note(kind)
    return True


def _prune_empty_dirs(directory: Path) -> None:
    """Remove per-camera subdirectories left empty by a sweep. The root is
    kept: the engine and the static mounts expect it to exist."""
    for root, dirs, files in os.walk(directory, topdown=False, onerror=lambda _exc: None):
        if Path(root) == directory or dirs or files:
            continue
        try:
            Path(root).rmdir()
        except OSError:
            pass


class Retention:
    """The housekeeping job. Usable as a one-shot (``sweep()``) or as a thread."""

    def __init__(self, config: ArgusConfig, database: Database, *, log=print):
        self.config = config
        self.db = database
        self.log = log
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_result: SweepResult | None = None
        self.last_backup: backup.Backup | None = None

    # --- media ------------------------------------------------------------

    def _media_targets(self) -> tuple[MediaTarget, ...]:
        retention = self.config.retention
        return (
            MediaTarget("clips", self.config.clips_dir, retention.clips_days, "clip_path",
                        budget_gb=retention.max_clips_gb),
            MediaTarget("snapshots", self.config.snapshots_dir, retention.snapshots_days,
                        "snapshot_path", budget_gb=retention.max_snapshots_gb),
            # Track overlays are written alongside snapshots and are useless
            # without them, so they share the snapshot window.
            MediaTarget("tracks", self.config.tracks_dir, retention.snapshots_days, "tracks_path"),
            # Ring segments are rolling footage; the engine prunes them while it
            # runs, this catches what a crashed run left behind. Its directories
            # are never pruned: a running ffmpeg writes the next segment into a
            # path it opened once, and removing that directory in the gap
            # between two segments would break recording on a live camera.
            MediaTarget("ring", self.config.ring_dir,
                        self.config.evidence.ring_retention_sec / 86400.0,
                        None, prune_dirs=False),
            # Index sample thumbnails. No events column — these frames are not
            # evidence for anything, they are the searchable record of an
            # ordinary minute. Their vector rows are dropped alongside them by
            # _sweep_index; see index_days in config.py for the arithmetic that
            # sets this window shorter than every other one.
            MediaTarget("index", self.config.index_dir,
                        self.config.retention.index_days, None),
        )

    # --- tiers ------------------------------------------------------------
    #
    # No schema change was needed for any of this, and that is worth stating
    # plainly rather than leaving to be inferred: the tier of an event is
    # already in the database — `events.stage` since migration 0, `reviews.
    # verdict` since migration 0 — and `Database.media_tiers` is one join over
    # columns that have always been there. A retention policy that required a
    # migration would also require every existing appliance to take one before
    # it could stop filling its disk.

    def _tier_days(self, target: MediaTarget) -> dict[str, float]:
        """``tier -> days`` for one media kind, or ``{}`` when this config has
        no per-stage block and the flat window applies to everything."""
        evidence = self.config.retention.evidence
        if target.column is None or not evidence.configured:
            return {}
        return {tier: evidence.days(tier, target.days) for tier in TIERS}

    def _tiers_for(self, target: MediaTarget) -> dict[str, str]:
        """``relative path -> tier`` for one media kind. One query, not one per
        file; empty when the kind is not evidence."""
        if target.column is None:
            return {}
        return self.db.media_tiers(target.column)

    def _sweep_media(self, result: SweepResult) -> None:
        """Age out evidence files, then repair the rows that pointed at them.

        The walk is by mtime rather than by database row on purpose: it also
        collects orphans — files whose event was purged earlier, or that a
        killed ffmpeg left behind — which a row-driven pass can never see. The
        tier map is consulted *from* that walk rather than driving it, so
        tiering is a refinement of the window and never a narrowing of what gets
        looked at.

        A file whose event has no tier entry (an orphan) gets the flat window,
        which is exactly what it got before tiers existed.

        One consequence, said out loud because it is a real trade: an event's
        tier can change after its evidence is gone. A shadow clip deleted at
        24 hours cannot be produced for a human who confirms that event on day
        three. Shadow events are not in the triage queue — `pending_reviews`
        filters `stage != 'shadow'` — so in practice nobody was going to be
        asked, but a site that wants a longer grace period sets
        `retention.evidence.shadow_days` and gets it.
        """
        for target in self._media_targets():
            if not target.directory.exists():
                continue
            now = time.time()
            tier_days = self._tier_days(target)
            tiers = self._tiers_for(target) if tier_days else {}
            flat_cutoff = now - target.days * 86400.0
            cutoffs = {tier: now - days * 86400.0 for tier, days in tier_days.items()}
            removed: list[str] = []
            for mtime, _size, path in _walk(target.directory):
                relative = str(path.relative_to(target.directory))
                cutoff = cutoffs.get(tiers.get(relative, ""), flat_cutoff)
                if mtime >= cutoff:
                    continue
                if _unlink(path, result, target.kind):
                    removed.append(relative)
            if target.column and removed:
                result.paths_cleared += self.db.clear_media_paths(target.column, removed)
            # A search result whose picture has been deleted renders as a broken
            # tile, which is worse than not being found — so the index forgets
            # a frame in the same pass that deletes its image, whether that
            # image was an index thumbnail or an evidence snapshot.
            if removed and target.kind in ("index", "snapshots"):
                self._prune_index_paths(
                    result, "index" if target.kind == "index" else "snapshot", removed
                )
            if target.prune_dirs:
                _prune_empty_dirs(target.directory)

    # --- the budget -------------------------------------------------------

    def _enforce_budgets(self, result: SweepResult) -> None:
        """Hold each evidence directory under its configured size, whatever the
        ages say.

        Runs **after** the age sweep, so it only ever has to deal with evidence
        that is still inside its window — a budget that fought the age sweep for
        the same files would report reclaiming space the ages were about to
        reclaim anyway.

        Eviction order is *least valuable first, oldest first within that*, and
        "value" is the window the operator configured for the tier. That is
        deliberate: there is no second ranking table in the code that could
        disagree with the config file. A site that says a false positive is
        worth keeping longer than a shadow event has thereby also said which of
        the two to evict first, and both statements are the same line of YAML.

        ``min_keep_hours`` applies here exactly as it does to the disk guard.
        Buying disk back by destroying the evidence for an alert raised twenty
        minutes ago is the wrong trade at any budget, so the pass stops and says
        so rather than crossing that line.
        """
        for target in self._media_targets():
            limit = int(target.budget_gb * 1e9)
            if limit <= 0 or not target.directory.exists():
                continue
            files = list(_walk(target.directory))
            used = sum(size for _mtime, size, _path in files)
            if used <= limit:
                continue

            tier_days = self._tier_days(target)
            tiers = self._tiers_for(target)
            floor = time.time() - self.config.retention.min_keep_hours * 3600.0

            def rank(path: Path, *, _t=target, _tiers=tiers, _days=tier_days) -> float:
                tier = _tiers.get(str(path.relative_to(_t.directory)))
                if tier is None:
                    return _ORPHAN_RANK
                return _days.get(tier, float(_t.days))

            candidates = sorted(
                ((rank(path), mtime, size, path) for mtime, size, path in files),
                key=lambda entry: (entry[0], entry[1]),
            )

            freed = 0
            removed: list[str] = []
            by_tier: dict[str, int] = {}
            protected = 0
            for _rank, mtime, size, path in candidates:
                if used - freed <= limit:
                    break
                if mtime >= floor:
                    protected += 1
                    continue
                if _unlink(path, result, f"{target.kind}_over_budget"):
                    freed += size
                    relative = str(path.relative_to(target.directory))
                    removed.append(relative)
                    label = tiers.get(relative) or "orphan"
                    by_tier[label] = by_tier.get(label, 0) + 1

            self._repair_after(target, result, removed)
            over = used - limit
            record = {
                "used_gb_before": round(used / 1e9, 2),
                "budget_gb": round(target.budget_gb, 2),
                "used_gb_after": round((used - freed) / 1e9, 2),
                "files_evicted": len(removed),
                "gb_reclaimed": round(freed / 1e9, 2),
                "by_tier": by_tier,
                "under_budget": (used - freed) <= limit,
            }
            result.budgets[target.kind] = record
            detail = ", ".join(f"{tier}={count}" for tier, count in sorted(by_tier.items()))
            self.log(
                f"retention[budget]: {target.kind} was {record['used_gb_before']} GB "
                f"against a {record['budget_gb']} GB budget ({over / 1e9:.2f} GB over) — "
                f"evicted {len(removed)} file(s), {record['gb_reclaimed']} GB, "
                f"least-valuable-first ({detail or 'nothing eligible'}); "
                f"now {record['used_gb_after']} GB"
            )
            if not record["under_budget"]:
                message = (
                    f"{target.kind} still {record['used_gb_after']} GB over a "
                    f"{record['budget_gb']} GB budget: {protected} file(s) are younger "
                    f"than min_keep_hours={self.config.retention.min_keep_hours}h"
                )
                result.errors.append(message)
                self.log(f"retention[budget]: {message}")

    def _repair_after(self, target: MediaTarget, result: SweepResult,
                      removed: list[str]) -> None:
        """The database work every deletion path owes: forget the pointers, drop
        the index rows, tidy the empty camera directories."""
        if not removed:
            return
        if target.column:
            result.paths_cleared += self.db.clear_media_paths(target.column, removed)
        if target.kind in ("index", "snapshots"):
            self._prune_index_paths(
                result, "index" if target.kind == "index" else "snapshot", removed
            )
        if target.prune_dirs:
            _prune_empty_dirs(target.directory)

    # --- rows -------------------------------------------------------------

    def _sweep_rows(self, result: SweepResult) -> None:
        """Purge aged rows. Events go last: deleting one cascades to its
        reviews and notifications, so purging those first would be wasted work.
        """
        retention = self.config.retention
        for table, days in (
            ("occupancy", retention.occupancy_days),
            ("notifications", retention.events_days),
            ("events", retention.events_days),
        ):
            deleted = self.db.purge_older_than(table, "created_at", _cutoff_iso(days))
            result.rows_deleted += deleted
            result.note(f"{table}_rows", deleted)

        # Phase 5. Two tables that nothing else prunes, and both grow forever
        # otherwise: a session row survives its own expiry until somebody
        # presents its cookie (which an expired session's owner never will), and
        # every mistyped password is a row.
        #
        # Neither is bounded by a `retention.*_days` window, because neither is
        # evidence — the deadlines are the auth settings themselves, and putting
        # them in the retention config would let an operator "keep sessions for
        # 90 days" without meaning to extend anybody's login.
        expired = self.db.purge_expired_sessions()
        result.rows_deleted += expired
        result.note("expired_sessions", expired)

        attempts = self.db.purge_login_attempts(
            _cutoff_iso(max(1.0, self.config.auth.attempt_retention_days))
        )
        result.rows_deleted += attempts
        result.note("login_attempts", attempts)

    # --- the semantic index -----------------------------------------------
    #
    # The index is derived data in a second SQLite file with an integer pointer
    # into the first (see argus/index/store.py). Nothing in SQLite enforces
    # that pointer across files, so retention does — in the same sweep that
    # deletes the evidence, because the window in which a search can return a
    # hit that resolves to nothing should be zero, not six hours.
    #
    # Every one of these is best-effort. The index is a feature; clips filling
    # the disk is an outage, and a store that cannot be opened (no extension,
    # locked, corrupt) must not stop the sweep that prevents the outage.

    def _open_index(self):
        """The vector store, or None. Never raises, never builds a model:
        the dimension and model id are read from the file's own metadata."""
        path = self.config.index_db_path
        if not path.exists():
            return None
        try:
            from argus.index.store import VectorStore

            meta = VectorStore(path, dim=0, model="").meta()
            if not meta:
                return None
            return VectorStore(path, dim=int(meta.get("dim", 0) or 0),
                               model=str(meta.get("model", "")))
        except Exception as exc:  # noqa: BLE001
            self._note_index_error(exc)
            return None

    def _note_index_error(self, exc: BaseException) -> None:
        self.log(f"retention: index prune skipped ({type(exc).__name__}: {exc})")

    def _prune_index_paths(self, result: SweepResult, thumb_kind: str,
                           paths: list[str]) -> None:
        store = self._open_index()
        if store is None:
            return
        try:
            result.index_rows_deleted += store.prune_paths(thumb_kind, paths)
        except Exception as exc:  # noqa: BLE001
            self._note_index_error(exc)

    def _sweep_index(self, result: SweepResult) -> None:
        """Drop index rows whose event, camera or age says they should go.

        Three prunes because there are three ways an index row outlives its
        subject: its event was purged by the 180-day window (or cascaded away
        with its camera), its camera was deleted, or the sample itself is older
        than ``index_days`` — the last of which also catches vectors whose
        thumbnail a *crashed* run never got round to writing.
        """
        store = self._open_index()
        if store is None:
            return
        try:
            cutoff = time.time() - self.config.retention.index_days * 86400.0
            for row in store.prune_older_than(cutoff):
                # The vector went; take its picture with it. Event rows point
                # at evidence snapshots on their own retention window, so they
                # are never deleted from here.
                if row.get("thumb_kind") == "index":
                    _unlink(self.config.index_dir / str(row["thumb_path"]), result,
                            "index_expired")
                result.index_rows_deleted += 1
            result.index_rows_deleted += len(
                store.prune_missing_events(self.db.all_event_ids())
            )
            result.index_rows_deleted += store.prune_cameras(
                [str(camera["id"]) for camera in self.db.list_cameras()]
            )
        except Exception as exc:  # noqa: BLE001
            self._note_index_error(exc)

    # --- disk pressure ----------------------------------------------------

    def _reclaim_to_threshold(self, result: SweepResult) -> None:
        """Delete oldest evidence first until the filesystem is back under the
        threshold. Stops at ``min_keep_hours``: past that point the only thing
        left to delete is evidence for alerts a human has not seen yet, and
        deleting *that* to save disk would be the wrong trade."""
        retention = self.config.retention
        target = max(1.0, retention.max_disk_percent - _RECLAIM_MARGIN)
        floor = time.time() - retention.min_keep_hours * 3600.0
        targets = {t.kind: t for t in self._media_targets()}

        candidates: list[tuple[float, Path, str]] = []
        for kind, media in targets.items():
            if media.directory.exists():
                candidates += [(mtime, path, kind) for mtime, _size, path in _walk(media.directory)]
        candidates.sort(key=lambda entry: entry[0])

        removed_by_kind: dict[str, list[str]] = {}
        deleted_since_check = 0
        percent = (result.disk_before if result.disk_before is not None
                   else disk_usage(self.config.data_dir)["percent"])

        for mtime, path, kind in candidates:
            if percent <= target:
                break
            if mtime >= floor:
                # Sorted oldest-first, so everything remaining is younger too.
                result.errors.append(
                    f"disk still {percent}% full with nothing older than "
                    f"{retention.min_keep_hours}h left to delete"
                )
                break
            if _unlink(path, result, f"{kind}_reclaimed"):
                relative = str(path.relative_to(targets[kind].directory))
                removed_by_kind.setdefault(kind, []).append(relative)
            deleted_since_check += 1
            if deleted_since_check >= _DISK_RECHECK_EVERY:
                deleted_since_check = 0
                percent = disk_usage(self.config.data_dir)["percent"]

        # The guard deletes oldest-first across every directory, index
        # thumbnails included, so the same database repair the age sweep and the
        # budget do is owed here too — not only in the age sweep above.
        for kind, paths in removed_by_kind.items():
            self._repair_after(targets[kind], result, paths)

    # --- the plan ---------------------------------------------------------

    def plan(self) -> dict[str, Any]:
        """What the next sweep would do, measured and not performed.

        Everything here is counted off the same walk and the same tier map the
        sweeper uses, so ``argus retention --dry-run`` cannot describe a policy
        the sweep does not implement. It walks the whole evidence tree, which on
        a 40 000-clip appliance is a few seconds of stat calls — the right price
        for the one command an operator runs *before* deleting 500 GB.
        """
        plans: list[dict[str, Any]] = []
        for target in self._media_targets():
            tier_days = self._tier_days(target)
            tiers = self._tiers_for(target) if target.column else {}
            now = time.time()
            limit = int(target.budget_gb * 1e9)
            floor = now - self.config.retention.min_keep_hours * 3600.0

            # (rank, mtime, size, days) per file — rank orders eviction, days
            # decides expiry, and they differ only for orphans.
            rows: list[tuple[float, float, int, float]] = []
            by_tier: dict[str, dict[str, Any]] = {}
            used = 0
            for mtime, size, path in _walk(target.directory):
                relative = str(path.relative_to(target.directory))
                tier = tiers.get(relative)
                label = tier or "orphan"
                days = (float(target.days) if tier is None
                        else tier_days.get(tier, float(target.days)))
                bucket = by_tier.setdefault(
                    label, {"days": days, "files": 0, "bytes": 0, "expiring": 0}
                )
                bucket["files"] += 1
                bucket["bytes"] += size
                if mtime < now - days * 86400.0:
                    bucket["expiring"] += 1
                used += size
                rows.append((_ORPHAN_RANK if tier is None else days, mtime, size, days))

            # The ages run first in a real sweep, so the projection subtracts
            # them before asking what eviction is left to do. A dry-run that
            # counted both against the same files would overstate the budget's
            # work by everything the windows were going to take anyway.
            surviving = sorted(
                (row for row in rows if row[1] >= now - row[3] * 86400.0),
                key=lambda row: (row[0], row[1]),
            )
            after_ages = sum(row[2] for row in surviving)
            evict_files = evict_bytes = 0
            remaining = after_ages
            if limit > 0:
                for _rank, mtime, size, _days in surviving:
                    if remaining <= limit:
                        break
                    if mtime >= floor:
                        continue
                    remaining -= size
                    evict_files += 1
                    evict_bytes += size

            plans.append({
                "kind": target.kind,
                "directory": str(target.directory),
                "flat_days": float(target.days),
                "tiered": bool(tier_days),
                "budget_gb": round(target.budget_gb, 2),
                "used_gb": round(used / 1e9, 2),
                "after_ages_gb": round(after_ages / 1e9, 2),
                "over_budget": limit > 0 and after_ages > limit,
                "evict_files": evict_files,
                "evict_gb": round(evict_bytes / 1e9, 2),
                "by_tier": {
                    label: {"days": bucket["days"], "files": bucket["files"],
                            "gb": round(bucket["bytes"] / 1e9, 2),
                            "expiring": bucket["expiring"]}
                    for label, bucket in sorted(by_tier.items(),
                                                key=lambda item: item[1]["days"])
                },
            })
        return {
            "media": plans,
            "rows": {
                "events_days": self.config.retention.events_days,
                "occupancy_days": self.config.retention.occupancy_days,
            },
            "disk": disk_usage(self.config.data_dir),
            "max_disk_percent": self.config.retention.max_disk_percent,
            "min_keep_hours": self.config.retention.min_keep_hours,
        }

    # --- entry points -----------------------------------------------------

    def sweep(self) -> SweepResult:
        """One full pass: age out media, purge rows, then relieve disk pressure
        if the age windows were not enough."""
        started = time.monotonic()
        result = SweepResult()
        result.disk_before = disk_usage(self.config.data_dir)["percent"]

        self._sweep_media(result)
        # Between the ages and the rows: the budget must see what the age sweep
        # already took (so it does not claim credit for it), and the index and
        # row prunes below must see what the budget took.
        self._enforce_budgets(result)
        self._sweep_rows(result)
        # After the rows, not before: `prune_missing_events` asks which events
        # still exist, and running it first would keep index entries for the
        # events this same sweep is about to purge until the next one.
        self._sweep_index(result)

        percent = disk_usage(self.config.data_dir)["percent"]
        if percent > self.config.retention.max_disk_percent:
            result.emergency = True
            self._reclaim_to_threshold(result)
            percent = disk_usage(self.config.data_dir)["percent"]

        result.disk_after = percent
        result.duration_sec = time.monotonic() - started
        self.last_result = result
        return result

    def housekeeping(self) -> SweepResult:
        """One full housekeeping tick: sweep, rotate logs, back up if owed.

        Each part is caught separately and on purpose. Backups and rotation are
        useful; reclaiming disk is what keeps the appliance *up*, so a backup
        that fails (a full destination, an unwritable share) must not be able to
        stop the sweep that prevents an outage — and the reverse is equally
        true, which is why the sweep runs first and its failure is reported
        rather than raised.
        """
        result = SweepResult()
        if self.config.retention.enabled:
            try:
                result = self.sweep()
                self.log(str(result))
            except Exception as exc:  # noqa: BLE001 - housekeeping must not end the run
                self.log(f"retention: sweep failed ({type(exc).__name__}: {exc})")

        try:
            for rotated in backup.rotate_logs(self.config):
                if "error" in rotated:
                    self.log(f"logs: {rotated['name']} rotation failed — {rotated['error']}")
                else:
                    self.log(f"logs: rotated {rotated['name']} ({rotated['mb']} MB) "
                             f"-> {rotated['rotated_to']}")
        except Exception as exc:  # noqa: BLE001
            self.log(f"logs: rotation failed ({type(exc).__name__}: {exc})")

        try:
            entry = backup.run_if_due(self.config)
        except backup.BackupError as exc:
            self.log(f"backup: skipped — {exc}")
        except Exception as exc:  # noqa: BLE001
            self.log(f"backup: failed ({type(exc).__name__}: {exc})")
        else:
            if entry is not None:
                self.last_backup = entry
                self.log(f"backup: {entry.stamp} — {entry.mb} MB in "
                         f"{entry.duration_sec:.1f}s ({len(entry.files)} file(s)), "
                         f"keeping {self.config.backup.keep}")
        return result

    def start(self) -> "Retention":
        """Run housekeeping on a daemon thread until ``stop()``.

        The first tick happens at startup rather than after a full interval: an
        appliance that was powered off for a month should not need six more
        hours before it notices, and it should not need 24 more before it has a
        backup either.

        ``retention.enabled: false`` no longer stops the thread, only the
        deleting: a site that manages storage some other way has not thereby
        asked for its database to stop being backed up. Each job has its own
        switch, and the thread runs while any of them is on.
        """
        if not self.config.retention.enabled:
            self.log("retention: disabled by config — nothing will be deleted")
        if not (self.config.retention.enabled or self.config.backup.enabled
                or self.config.logs.rotate):
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="retention")
        self._thread.start()
        return self

    def _loop(self) -> None:
        interval = self.config.retention.interval_hours * 3600.0
        while not self._stop.is_set():
            self.housekeeping()
            self._stop.wait(interval)

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None


__all__ = ["Retention", "SweepResult"]
