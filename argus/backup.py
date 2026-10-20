"""Nightly backups and log rotation — the two things that stop an appliance
from losing state quietly.

Retention (``argus/retention.py``) bounds what the box *keeps*. This module
covers the other half of data safety: what happens when the thing it kept goes
bad. Both jobs run on the same housekeeping thread inside ``argus run``, for the
same reason retention does: an appliance whose data safety depends on somebody
installing a cron entry is an appliance that ships without data safety.

**Backups use SQLite's online backup API**, not a file copy. ``argus.sqlite3``
is in WAL mode with a live writer (the vision engine) — ``cp`` of that file
while a transaction is open produces a database that is *sometimes* fine, which
is the worst kind. The backup API copies pages under the source's own lock and
restarts if a writer moves them, so the copy is a consistent snapshot of a
point in time, and it is a fully checkpointed one: the restored file needs no
WAL to be complete.

``index.sqlite3`` is copied the same way even though the ``vec0`` extension is
not loaded here. The backup API works at the pager level — it copies pages, not
rows — so the shadow tables behind a virtual table come across intact without
the module that understands them. Measured on this box: 52 MB in 81 ms.

**A backup is complete when its manifest exists.** The copy is assembled in
``.incoming-<stamp>`` and renamed into place only after every file is written
and hashed, so a power cut mid-copy leaves something ``list`` reports as
incomplete and ``prune`` removes first, never something ``restore`` will trust.

**The default destination is under ``data_dir``, and that is a real limitation
rather than an oversight.** A copy on the same disk defends against the failure
that actually happens to appliances — a bad migration, a mistaken delete, a
corrupted page — and defends against nothing at all if the NVMe fails. Point
``backup.directory`` at a mounted share and it defends against both; the code
does not care which it is.

**Log rotation is copy-and-truncate, not rename.** ``go2rtc`` and the vLLM
server are children of our own supervisors, which hold their log file open in
append mode for the life of the process. Renaming that file would leave the
writer happily appending to the rotated copy while the "current" log stayed
empty forever — the classic rotation bug. Copying and then truncating in place
keeps the inode, so an open file descriptor keeps working. The cost is a race:
lines written between the copy and the truncate are lost. That is the correct
trade for a debug log and would not be for evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from argus.config import ArgusConfig
from argus.health import disk_usage, read_engine_stats

#: Backup directory names are UTC timestamps, sortable as strings.
STAMP_FORMAT = "%Y%m%dT%H%M%SZ"

#: Written last; its presence is the definition of a complete backup.
MANIFEST_NAME = "manifest.json"

#: Where a backup is assembled before it is renamed into place.
_INCOMING_PREFIX = ".incoming-"

#: Read size for hashing. Large enough that a 20 GB index is not 20 million
#: syscalls, small enough that it never matters on a box with less RAM.
_HASH_CHUNK = 4 * 1024 * 1024

#: Multiple of the source size that must be free before a backup starts.
_HEADROOM = 1.15


class BackupError(RuntimeError):
    """A backup or restore that must not be allowed to proceed."""


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------


@dataclass(slots=True)
class BackupFile:
    name: str
    bytes: int
    sha256: str

    @property
    def mb(self) -> float:
        return round(self.bytes / 1e6, 1)


@dataclass(slots=True)
class Backup:
    """One directory under ``backups/``, as described by its manifest."""

    stamp: str
    path: Path
    created_at: str = ""
    reason: str = ""
    argus_version: str = ""
    schema_version: int = 0
    duration_sec: float = 0.0
    files: list[BackupFile] = field(default_factory=list)
    complete: bool = True
    error: str | None = None

    @property
    def bytes(self) -> int:
        return sum(f.bytes for f in self.files)

    @property
    def mb(self) -> float:
        return round(self.bytes / 1e6, 1)

    def age_sec(self, now: float | None = None) -> float | None:
        moment = parse_stamp(self.stamp)
        if moment is None:
            return None
        return (now if now is not None else time.time()) - moment.timestamp()

    def file(self, name: str) -> BackupFile | None:
        return next((f for f in self.files if f.name == name), None)

    def as_dict(self) -> dict[str, Any]:
        return {
            "stamp": self.stamp,
            "path": str(self.path),
            "created_at": self.created_at,
            "reason": self.reason,
            "argus_version": self.argus_version,
            "schema_version": self.schema_version,
            "duration_sec": round(self.duration_sec, 2),
            "mb": self.mb,
            "complete": self.complete,
            "error": self.error,
            "files": [asdict(f) for f in self.files],
        }


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def backup_dir(config: ArgusConfig) -> Path:
    return config.backups_dir


def stamp_for(when: datetime | None = None) -> str:
    return (when or datetime.now(timezone.utc)).strftime(STAMP_FORMAT)


def parse_stamp(stamp: str) -> datetime | None:
    """The timestamp in a directory name, ignoring any disambiguating suffix.

    Two backups inside one second get ``…Z-1``. Without this split that suffix
    would make :func:`due` think the newest backup has no age and take another
    one immediately, forever.
    """
    try:
        return datetime.strptime(stamp.split("-", 1)[0], STAMP_FORMAT).replace(
            tzinfo=timezone.utc)
    except ValueError:
        return None


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def sources(config: ArgusConfig) -> list[Path]:
    """The databases a backup copies, in manifest order.

    ``index.sqlite3`` is included by default and is the one that grows with the
    fleet — it holds a vector per periodic sample, so a 48-camera site at the
    default sample interval writes far more of it than of ``argus.sqlite3``. It
    is also *not* fully derived data: ``argus index backfill`` can rebuild
    vectors for event snapshots, but the periodic samples that make ordinary
    minutes searchable exist only here. ``backup.include_index: false`` is for
    a site that has decided searchable history is expendable and disk is not.
    """
    paths = [config.db_path]
    if config.backup.include_index:
        paths.append(config.index_db_path)
    return paths


# --------------------------------------------------------------------------
# listing
# --------------------------------------------------------------------------


def _read_manifest(directory: Path) -> Backup:
    stamp = directory.name
    manifest = directory / MANIFEST_NAME
    if not manifest.is_file():
        return Backup(stamp=stamp, path=directory, complete=False,
                      error="no manifest — interrupted or still being written")
    try:
        data = json.loads(manifest.read_text())
    except (OSError, ValueError) as exc:
        return Backup(stamp=stamp, path=directory, complete=False,
                      error=f"unreadable manifest: {exc}")
    files = [BackupFile(name=str(f.get("name")), bytes=int(f.get("bytes", 0)),
                        sha256=str(f.get("sha256", "")))
             for f in data.get("files", [])]
    return Backup(
        stamp=str(data.get("stamp", stamp)),
        path=directory,
        created_at=str(data.get("created_at", "")),
        reason=str(data.get("reason", "")),
        argus_version=str(data.get("argus_version", "")),
        schema_version=int(data.get("schema_version", 0)),
        duration_sec=float(data.get("duration_sec", 0.0)),
        files=files,
    )


def list_backups(config: ArgusConfig, *, include_incomplete: bool = True) -> list[Backup]:
    """Newest first. Incomplete directories are listed too, because "the last
    three backups all died half-way" is exactly the thing an operator must be
    able to see."""
    directory = backup_dir(config)
    if not directory.is_dir():
        return []
    found: list[Backup] = []
    for child in directory.iterdir():
        if not child.is_dir() or child.name.startswith(_INCOMING_PREFIX):
            continue
        entry = _read_manifest(child)
        if entry.complete or include_incomplete:
            found.append(entry)
    found.sort(key=lambda b: b.stamp, reverse=True)
    return found


def newest(config: ArgusConfig) -> Backup | None:
    return next((b for b in list_backups(config, include_incomplete=False)), None)


def find(config: ArgusConfig, stamp: str) -> Backup:
    """Resolve a stamp, ``latest``, or an unambiguous prefix of one."""
    entries = list_backups(config, include_incomplete=False)
    if not entries:
        raise BackupError(f"no complete backups in {backup_dir(config)}")
    if stamp in ("latest", "last", ""):
        return entries[0]
    exact = [b for b in entries if b.stamp == stamp]
    if exact:
        return exact[0]
    matches = [b for b in entries if b.stamp.startswith(stamp)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise BackupError(f"no backup {stamp!r} (have: {', '.join(b.stamp for b in entries[:5])})")
    raise BackupError(f"{stamp!r} matches {len(matches)} backups — give the full stamp")


def verify(entry: Backup) -> list[str]:
    """Re-hash a backup against its manifest. Empty list means intact."""
    problems: list[str] = []
    for record in entry.files:
        path = entry.path / record.name
        if not path.is_file():
            problems.append(f"{record.name}: missing")
            continue
        size = path.stat().st_size
        if size != record.bytes:
            problems.append(f"{record.name}: {size} bytes, manifest says {record.bytes}")
            continue
        if record.sha256 and sha256_of(path) != record.sha256:
            problems.append(f"{record.name}: sha256 mismatch")
    return problems


# --------------------------------------------------------------------------
# taking one
# --------------------------------------------------------------------------


def _copy_database(source: Path, destination: Path) -> None:
    """One SQLite online backup. Both connections are closed on every path."""
    src = sqlite3.connect(source, timeout=30.0)
    try:
        src.execute("PRAGMA busy_timeout = 30000")
        dst = sqlite3.connect(destination)
        try:
            # pages=-1 copies in one step while holding the read lock, which is
            # right here: these files are tens to hundreds of MB, the copy is
            # sub-second, and an incremental copy would only widen the window
            # in which a writer can force a restart.
            src.backup(dst, pages=-1)
        finally:
            dst.close()
    finally:
        src.close()


def _check_room(config: ArgusConfig, directory: Path, needed: int) -> None:
    usage = disk_usage(directory)
    limit = config.backup.max_disk_percent
    if usage["percent"] > limit:
        raise BackupError(
            f"disk {usage['percent']}% full, above backup.max_disk_percent ({limit}%) — "
            "refusing to back up onto a full appliance"
        )
    free = usage.get("free_gb") or 0.0
    if free * 1e9 < needed * _HEADROOM:
        raise BackupError(
            f"needs {needed * _HEADROOM / 1e9:.1f} GB free, filesystem has {free} GB"
        )


def run(config: ArgusConfig, *, reason: str = "manual",
        when: datetime | None = None) -> Backup:
    """Take one backup. Raises :class:`BackupError` rather than half-doing it."""
    if not config.backup.enabled and reason == "scheduled":
        raise BackupError("backup.enabled is false")

    directory = backup_dir(config)
    directory.mkdir(parents=True, exist_ok=True)

    present = [path for path in sources(config) if path.is_file()]
    if not present:
        raise BackupError(f"nothing to back up — no database at {config.db_path}")
    _check_room(config, directory, sum(p.stat().st_size for p in present))

    stamp = stamp_for(when)
    # Two backups inside one second is only reachable from a test or a script,
    # but a silent overwrite of the earlier one would be the wrong answer.
    suffix = 0
    while (directory / stamp).exists():
        suffix += 1
        stamp = f"{stamp_for(when)}-{suffix}"

    staging = directory / f"{_INCOMING_PREFIX}{stamp}"
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)

    started = time.monotonic()
    files: list[BackupFile] = []
    try:
        for source in present:
            target = staging / source.name
            _copy_database(source, target)
            files.append(BackupFile(name=source.name, bytes=target.stat().st_size,
                                    sha256=sha256_of(target)))
        entry = Backup(
            stamp=stamp,
            path=directory / stamp,
            created_at=(when or datetime.now(timezone.utc)).isoformat(),
            reason=reason,
            argus_version=_version(),
            schema_version=_schema_version(config),
            duration_sec=time.monotonic() - started,
            files=files,
        )
        manifest = {**entry.as_dict()}
        manifest.pop("path", None)
        manifest.pop("complete", None)
        manifest.pop("error", None)
        (staging / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
        # The rename is the commit. Nothing before it is a backup.
        os.replace(staging, directory / stamp)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return entry


def _version() -> str:
    from argus import __version__

    return __version__


def _schema_version(config: ArgusConfig) -> int:
    """Recorded so a restore can say which schema it is putting back."""
    from argus.db import Database

    try:
        return Database(config.db_path).schema_version()
    except sqlite3.Error:
        return 0


def prune(config: ArgusConfig) -> list[str]:
    """Keep the newest ``backup.keep`` complete backups; delete the rest.

    Incomplete directories and abandoned staging areas go first and do not
    count against the quota — an interrupted copy must never be the reason a
    good backup is deleted.
    """
    directory = backup_dir(config)
    if not directory.is_dir():
        return []
    removed: list[str] = []

    for child in directory.iterdir():
        if child.is_dir() and child.name.startswith(_INCOMING_PREFIX):
            shutil.rmtree(child, ignore_errors=True)
            removed.append(child.name)

    entries = list_backups(config)
    for entry in entries:
        if not entry.complete:
            shutil.rmtree(entry.path, ignore_errors=True)
            removed.append(entry.stamp)

    keep = max(1, int(config.backup.keep))
    for entry in [e for e in entries if e.complete][keep:]:
        shutil.rmtree(entry.path, ignore_errors=True)
        removed.append(entry.stamp)
    return removed


def due(config: ArgusConfig, *, now: float | None = None) -> bool:
    """Is a scheduled backup owed?

    Age-based rather than "at 03:00", deliberately. A wall-clock schedule on an
    appliance that is powered down overnight, or restarted during its window,
    silently skips a night and nothing says so. Age cannot skip: the first
    housekeeping tick after the interval elapses takes one. The cost is drift —
    backups walk later through the day by however long the tick was late —
    which nobody is harmed by.
    """
    if not config.backup.enabled:
        return False
    latest = newest(config)
    if latest is None:
        return True
    age = latest.age_sec(now)
    if age is None:
        return True
    return age >= config.backup.interval_hours * 3600.0


def run_if_due(config: ArgusConfig, *, now: float | None = None) -> Backup | None:
    """The housekeeping thread's entry point. Backs up, then prunes."""
    if not due(config, now=now):
        return None
    entry = run(config, reason="scheduled")
    prune(config)
    return entry


# --------------------------------------------------------------------------
# putting one back
# --------------------------------------------------------------------------


def _move_aside(path: Path, stamp: str) -> list[str]:
    """Move a database and its WAL sidecars out of the way, keeping them.

    The sidecars are the point. A restored file is fully checkpointed, so a
    leftover ``-wal`` from the database being replaced would be replayed on top
    of it at the next open — which is not "the restore did not work", it is
    "the restore produced a database that never existed".
    """
    moved: list[str] = []
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(path) + suffix)
        if candidate.exists():
            destination = Path(f"{candidate}.pre-restore-{stamp}")
            os.replace(candidate, destination)
            moved.append(destination.name)
    return moved


def restore(config: ArgusConfig, stamp: str, *, force: bool = False) -> dict[str, Any]:
    """Put a backup back. Never deletes the databases it replaces.

    Refused while the engine is running: two processes cannot agree about a
    file one of them is holding open in WAL mode, and a restore that half-lands
    under a live writer is a worse outcome than a restore that did not happen.
    ``--force`` exists for the case where the stats file is stale rather than
    the engine alive.
    """
    entry = find(config, stamp)

    problems = verify(entry)
    if problems:
        raise BackupError(f"backup {entry.stamp} is damaged: " + "; ".join(problems))

    engine = read_engine_stats(config)
    if engine.get("running") and not force:
        raise BackupError(
            "the vision engine is running (stop `argus run`, or pass --force if "
            "you know it is gone)"
        )

    moment = stamp_for()
    restored: list[dict[str, Any]] = []
    for record in entry.files:
        target = _target_for(config, record.name)
        if target is None:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        # Copy first, move aside second, rename third. A restore that fails
        # half-way through the copy must leave the appliance exactly as it was —
        # the order "move aside, then copy" would leave it with neither.
        staging = target.with_name(f".restoring-{moment}-{record.name}")
        try:
            shutil.copy2(entry.path / record.name, staging)
            aside = _move_aside(target, moment)
            os.replace(staging, target)
        except BaseException:
            staging.unlink(missing_ok=True)
            raise
        restored.append({
            "name": record.name,
            "path": str(target),
            "mb": record.mb,
            "kept_aside": aside,
        })

    from argus.db import Database

    applied = Database(config.db_path).migrate()
    return {
        "stamp": entry.stamp,
        "created_at": entry.created_at,
        "from_schema": entry.schema_version,
        "migrations_applied": applied,
        "schema_version": Database(config.db_path).schema_version(),
        "restored": restored,
        "pre_restore_suffix": f".pre-restore-{moment}",
    }


def _target_for(config: ArgusConfig, name: str) -> Path | None:
    """Where a manifest entry goes back to. Names, not stored paths: a backup
    taken on a box whose data_dir was elsewhere still restores here."""
    for candidate in (config.db_path, config.index_db_path):
        if candidate.name == name:
            return candidate
    return None


# --------------------------------------------------------------------------
# log rotation
# --------------------------------------------------------------------------


def rotate_logs(config: ArgusConfig) -> list[dict[str, Any]]:
    """Copy-and-truncate every oversized ``*.log`` under ``logs_dir``.

    Only the logs Argus writes itself are here — ``go2rtc.log`` and
    ``vlm-server.log``. Anything started by ``argus install-services`` logs to
    journald, which systemd already rotates by size and age, and duplicating
    that here would mean two rotation policies for one appliance.
    """
    settings = config.logs
    directory = config.logs_dir
    if not settings.rotate or not directory.is_dir():
        return []

    limit = int(settings.max_mb * 1e6)
    keep = max(1, int(settings.keep))
    rotated: list[dict[str, Any]] = []

    for path in sorted(directory.glob("*.log")):
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size < limit:
            continue

        # Shift the older generations up, dropping the oldest.
        oldest = Path(f"{path}.{keep}")
        if oldest.exists():
            oldest.unlink(missing_ok=True)
        for index in range(keep - 1, 0, -1):
            source = Path(f"{path}.{index}")
            if source.exists():
                os.replace(source, Path(f"{path}.{index + 1}"))

        try:
            shutil.copyfile(path, Path(f"{path}.1"))
            # Truncate rather than unlink: the writer holds this inode open in
            # append mode and would otherwise keep writing to a file with no
            # name. See the module docstring.
            os.truncate(path, 0)
        except OSError as exc:
            rotated.append({"name": path.name, "mb": round(size / 1e6, 1),
                            "error": str(exc)})
            continue
        rotated.append({"name": path.name, "mb": round(size / 1e6, 1),
                        "rotated_to": f"{path.name}.1"})
    return rotated


__all__ = [
    "Backup",
    "BackupError",
    "BackupFile",
    "backup_dir",
    "due",
    "find",
    "list_backups",
    "newest",
    "prune",
    "restore",
    "rotate_logs",
    "run",
    "run_if_due",
    "sources",
    "stamp_for",
    "verify",
]
