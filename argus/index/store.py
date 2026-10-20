"""The vector index: sqlite-vec, in its own SQLite file.

**Why sqlite-vec and not LanceDB.** The appliance already is a SQLite box —
WAL, one writer, atomic file backup, no server to operate. sqlite-vec keeps
that shape: it is a single loadable extension (MIT OR Apache-2.0, with an
aarch64 wheel on PyPI), so the vector store is *another SQLite file*, backed up
by copying it, inspected with the ``sqlite3`` CLI, and joined against the
appliance database in one process without a network hop. LanceDB would have
introduced a second storage engine, a second on-disk format and a second set of
failure modes for a working set that is measured in tens of thousands of rows,
not billions. No blocker was found; the choice is that the cheapest thing works.

**Why a separate file rather than a table in ``argus.sqlite3``.** A ``vec0``
virtual table is only readable by a connection that has loaded the extension.
Put it in the appliance database and ``sqlite3 argus.sqlite3 .dump``, ``VACUUM``,
and every support-bundle or backup tool that opens the file with a stock SQLite
stop working — an appliance whose database can only be opened by itself is a
worse appliance. So the vectors live in ``data/db/index.sqlite3``, and the link
back is an ordinary integer: ``frames.event_id`` is an ``events.id``. Nothing
enforces that reference across files, which is precisely why
:meth:`prune_missing_events` exists and why retention calls it.

**Brute force, on purpose.** ``vec0`` scans; it does not build an ANN graph.
At the appliance's scale that is the right answer and it is worth the
arithmetic: 48 cameras x 10 s x 14 days is 5.8 M vectors, and a 768-float scan
of that is ~17 GB of reads — too slow. But the search UI *always* filters by
time (the design's result strip is chronological), and one camera-day is 8 640
vectors, scanned in single-digit milliseconds. The store therefore pushes camera
and time filters into the same statement as the kNN via ``vec0``'s metadata
columns, so the scan is over the window and not over the corpus. Measured
numbers are in docs/PHASE3_NOTES.md.
"""

from __future__ import annotations

import sqlite3
import struct
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

Row = dict[str, Any]

INDEX_SCHEMA_VERSION = 1

# What produced a row. 'event' frames are the source frame of an event's
# annotated snapshot; 'sample' frames are the periodic per-camera sample.
KINDS = ("event", "sample")


class IndexError_(RuntimeError):
    """Raised for conditions an operator has to act on (wrong model, no
    extension), never for an empty result."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def serialize(vector: Sequence[float]) -> bytes:
    """Pack a float vector the way sqlite-vec wants it (little-endian f32)."""
    values = np.asarray(vector, dtype=np.float32).ravel()
    return struct.pack(f"<{len(values)}f", *values)


@dataclass(slots=True)
class IndexedFrame:
    """One thing worth searching for: a frame, where it came from, and when."""

    camera_id: str
    ts: float
    kind: str
    embedding: np.ndarray
    thumb_path: str          # relative to config.index_dir, or to snapshots_dir
    thumb_kind: str = "index"  # index | snapshot — which mount serves it
    event_id: int | None = None
    width: int | None = None
    height: int | None = None


class VectorStore:
    """The index file. Opened per operation, like :class:`argus.db.Database`."""

    def __init__(self, path: str | Path, *, dim: int, model: str, timeout: float = 10.0):
        self.path = Path(path)
        self.dim = int(dim)
        self.model = model
        self.timeout = timeout

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"VectorStore({str(self.path)!r}, dim={self.dim}, model={self.model!r})"

    # --- connection -------------------------------------------------------

    def connect(self) -> sqlite3.Connection:
        try:
            import sqlite_vec
        except ImportError as exc:  # pragma: no cover - declared dependency
            raise IndexError_(
                "sqlite-vec is not installed; the semantic index cannot open. "
                "Run `uv sync`."
            ) from exc
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=self.timeout)
        conn.row_factory = sqlite3.Row
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
        except (AttributeError, sqlite3.OperationalError) as exc:
            conn.close()
            raise IndexError_(
                "this Python's sqlite3 cannot load extensions, so sqlite-vec is "
                f"unusable ({exc}). Rebuild CPython with --enable-loadable-sqlite-extensions."
            ) from exc
        finally:
            # Narrow the window in which arbitrary extensions can be loaded to
            # the one call above.
            try:
                conn.enable_load_extension(False)
            except AttributeError:  # pragma: no cover
                pass
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute(f"PRAGMA busy_timeout = {int(self.timeout * 1000)}")
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

    # --- schema -----------------------------------------------------------

    def migrate(self) -> None:
        """Create the schema, or refuse a store written by another model.

        There is no migration ladder here as there is in ``argus/db.py``,
        because there is nothing to migrate: every row is derived data that
        ``argus index backfill`` can rebuild. A schema or model change therefore
        *rejects* the file and names the command that fixes it, rather than
        quietly ranking 768-float vectors against 1152-float ones — which
        sqlite-vec would refuse anyway, but only on the day somebody searched.
        """
        with self.conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS index_meta (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            existing = {
                row["key"]: row["value"] for row in conn.execute("SELECT key, value FROM index_meta")
            }
            if existing:
                if existing.get("model") != self.model or int(existing.get("dim", 0)) != self.dim:
                    raise IndexError_(
                        f"{self.path.name} was written by {existing.get('model')} "
                        f"({existing.get('dim')}-d) and this appliance is configured for "
                        f"{self.model} ({self.dim}-d). Vectors from two models are not "
                        "comparable — rebuild with `argus index backfill --reset`."
                    )
                return

            conn.executescript(
                f"""
                CREATE TABLE frames (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id  TEXT NOT NULL,
                    ts         REAL NOT NULL,
                    kind       TEXT NOT NULL CHECK (kind IN ('event', 'sample')),
                    -- events.id in argus.sqlite3. Not a foreign key: it is in
                    -- another file. prune_missing_events() is the enforcement.
                    event_id   INTEGER,
                    thumb_path TEXT NOT NULL,
                    thumb_kind TEXT NOT NULL DEFAULT 'index'
                               CHECK (thumb_kind IN ('index', 'snapshot')),
                    width      INTEGER,
                    height     INTEGER,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX idx_frames_camera_ts ON frames (camera_id, ts DESC);
                CREATE INDEX idx_frames_ts        ON frames (ts DESC);
                -- One index row per event, so re-indexing an event replaces it.
                CREATE UNIQUE INDEX idx_frames_event ON frames (event_id)
                    WHERE event_id IS NOT NULL;

                -- camera_id, ts and kind are declared as vec0 *metadata*
                -- columns, not kept only in `frames`: that is what lets a
                -- filtered search restrict the brute-force scan instead of
                -- ranking the whole corpus and discarding most of it
                -- afterwards. All three are duplicated in `frames` because a
                -- vec0 table cannot be joined on anything but its primary key.
                --
                -- `kind` is here despite being two-valued, and the reason is
                -- correctness rather than speed: the console's "Alerts only"
                -- filter asks for the best k *events*, and filtering after the
                -- kNN would hand it the events among the best k *frames* —
                -- two or three tiles out of a requested twenty-four on an
                -- index that is mostly periodic samples, with nothing on
                -- screen to say that is what happened.
                CREATE VIRTUAL TABLE frame_vectors USING vec0(
                    frame_id  INTEGER PRIMARY KEY,
                    embedding FLOAT[{self.dim}] distance_metric=cosine,
                    camera_id TEXT,
                    ts        FLOAT,
                    kind      TEXT
                );
                """
            )
            conn.executemany(
                "INSERT INTO index_meta (key, value) VALUES (?, ?)",
                [
                    ("model", self.model),
                    ("dim", str(self.dim)),
                    ("schema_version", str(INDEX_SCHEMA_VERSION)),
                    ("created_at", _now()),
                ],
            )

    def reset(self) -> None:
        """Drop everything. ``argus index backfill --reset`` and a model change."""
        with self.conn() as conn:
            conn.executescript(
                "DROP TABLE IF EXISTS frame_vectors;"
                "DROP TABLE IF EXISTS frames;"
                "DROP TABLE IF EXISTS index_meta;"
            )
        self.migrate()

    def meta(self) -> dict[str, str]:
        with self.conn() as conn:
            try:
                return {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM index_meta")}
            except sqlite3.OperationalError:
                return {}

    # --- writes -----------------------------------------------------------

    def add(self, frames: Sequence[IndexedFrame]) -> int:
        """Insert a batch. Re-indexing an event replaces its row.

        Written as one transaction per batch rather than per frame because the
        runner hands over up to ``index.batch_size`` frames at a time and a
        commit per row would make the disk, not the GPU, the cost of indexing.
        """
        if not frames:
            return 0
        written = 0
        with self.conn() as conn:
            for frame in frames:
                if frame.kind not in KINDS:
                    raise ValueError(f"unknown index kind: {frame.kind!r}")
                vector = np.asarray(frame.embedding, dtype=np.float32).ravel()
                if vector.size != self.dim:
                    raise ValueError(
                        f"embedding is {vector.size}-d, store is {self.dim}-d"
                    )
                if frame.event_id is not None:
                    old = conn.execute(
                        "SELECT id FROM frames WHERE event_id = ?", (frame.event_id,)
                    ).fetchone()
                    if old is not None:
                        conn.execute("DELETE FROM frame_vectors WHERE frame_id = ?", (old["id"],))
                        conn.execute("DELETE FROM frames WHERE id = ?", (old["id"],))
                cursor = conn.execute(
                    "INSERT INTO frames (camera_id, ts, kind, event_id, thumb_path, "
                    "thumb_kind, width, height, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (frame.camera_id, float(frame.ts), frame.kind, frame.event_id,
                     frame.thumb_path, frame.thumb_kind, frame.width, frame.height, _now()),
                )
                conn.execute(
                    "INSERT INTO frame_vectors (frame_id, embedding, camera_id, ts, kind) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (int(cursor.lastrowid), serialize(vector), frame.camera_id,
                     float(frame.ts), frame.kind),
                )
                written += 1
        return written

    # --- search -----------------------------------------------------------

    def search(
        self,
        embedding: Sequence[float],
        *,
        limit: int = 24,
        camera_id: str | None = None,
        since_ts: float | None = None,
        until_ts: float | None = None,
        kind: str | None = None,
    ) -> list[Row]:
        """kNN with the filters pushed into the same statement.

        Returns rows newest-of-the-best first is *not* what happens here — they
        come back by score, and the caller sorts. The design's result strip is
        chronological but the cut is by relevance, and doing that the other way
        round returns the most recent frames rather than the best ones.
        """
        vector = np.asarray(embedding, dtype=np.float32).ravel()
        if vector.size != self.dim:
            raise ValueError(f"query is {vector.size}-d, store is {self.dim}-d")

        clauses = ["v.embedding MATCH ?", "v.k = ?"]
        params: list[Any] = [serialize(vector), int(max(1, limit))]
        if camera_id:
            clauses.append("v.camera_id = ?")
            params.append(camera_id)
        if since_ts is not None:
            clauses.append("v.ts >= ?")
            params.append(float(since_ts))
        if until_ts is not None:
            clauses.append("v.ts <= ?")
            params.append(float(until_ts))
        if kind is not None:
            if kind not in KINDS:
                raise ValueError(f"unknown index kind: {kind!r}")
            clauses.append("v.kind = ?")
            params.append(kind)

        sql = (
            "SELECT f.id, f.camera_id, f.ts, f.kind, f.event_id, f.thumb_path, "
            "       f.thumb_kind, f.width, f.height, v.distance "
            "FROM frame_vectors v JOIN frames f ON f.id = v.frame_id "
            f"WHERE {' AND '.join(clauses)} "
            "ORDER BY v.distance"
        )
        with self.conn() as conn:
            rows = [dict(row) for row in conn.execute(sql, params)]
        for row in rows:
            # vec0's cosine distance is 1 - cosine similarity. The UI shows a
            # confidence bar, so hand it the similarity and let the score stay
            # the thing an operator can reason about.
            row["score"] = round(1.0 - float(row.pop("distance")), 4)
        return rows

    # --- retention --------------------------------------------------------
    #
    # The index is derived data with a pointer into another file, which makes
    # it exactly the kind of thing that rots silently. Three prunes, called by
    # argus/retention.py in the same sweep that deletes the evidence.

    def prune_older_than(self, cutoff_ts: float) -> list[Row]:
        """Delete rows older than an epoch. Returns what went, so the caller
        can delete the thumbnails they pointed at."""
        with self.conn() as conn:
            doomed = [dict(r) for r in conn.execute(
                "SELECT id, thumb_path, thumb_kind FROM frames WHERE ts < ?", (float(cutoff_ts),)
            )]
            self._delete_ids(conn, [int(r["id"]) for r in doomed])
        return doomed

    def prune_missing_events(self, live_event_ids: Sequence[int]) -> list[Row]:
        """Drop index rows whose event no longer exists.

        The caller passes every event id still in the appliance database, so a
        row referring to anything else is an orphan — the event was purged by
        the 180-day window, or its camera was deleted and cascaded. Without
        this, a search would keep returning a hit that resolves to nothing.
        """
        live = {int(event_id) for event_id in live_event_ids}
        with self.conn() as conn:
            rows = [dict(r) for r in conn.execute(
                "SELECT id, event_id, thumb_path, thumb_kind FROM frames WHERE event_id IS NOT NULL"
            )]
            doomed = [row for row in rows if int(row["event_id"]) not in live]
            self._delete_ids(conn, [int(r["id"]) for r in doomed])
        return doomed

    def prune_paths(self, thumb_kind: str, paths: Sequence[str]) -> int:
        """Forget rows whose thumbnail file has been deleted.

        Retention's media sweep walks directories by mtime, so it discovers
        deleted snapshots that no row-driven pass would ever see; this is the
        index's half of ``Database.clear_media_paths``. An index row without an
        image is worse than a missing search result: it is a result that renders
        as a broken tile.
        """
        if not paths:
            return 0
        removed = 0
        with self.conn() as conn:
            for start in range(0, len(paths), 400):
                chunk = list(paths[start:start + 400])
                placeholders = ", ".join("?" * len(chunk))
                ids = [int(r["id"]) for r in conn.execute(
                    f"SELECT id FROM frames WHERE thumb_kind = ? AND thumb_path IN ({placeholders})",  # noqa: S608 - placeholders only
                    [thumb_kind, *chunk],
                )]
                removed += self._delete_ids(conn, ids)
        return removed

    def prune_cameras(self, live_camera_ids: Sequence[str]) -> int:
        """Drop rows for cameras that no longer exist."""
        live = {str(camera_id) for camera_id in live_camera_ids}
        with self.conn() as conn:
            ids = [
                int(row["id"])
                for row in conn.execute("SELECT id, camera_id FROM frames")
                if str(row["camera_id"]) not in live
            ]
            return self._delete_ids(conn, ids)

    @staticmethod
    def _delete_ids(conn: sqlite3.Connection, ids: Sequence[int]) -> int:
        """Delete from both tables. vec0 has no ON DELETE CASCADE, so a row
        removed from ``frames`` alone would stay searchable forever."""
        if not ids:
            return 0
        for start in range(0, len(ids), 400):
            chunk = list(ids[start:start + 400])
            placeholders = ", ".join("?" * len(chunk))
            conn.execute(f"DELETE FROM frame_vectors WHERE frame_id IN ({placeholders})", chunk)  # noqa: S608
            conn.execute(f"DELETE FROM frames WHERE id IN ({placeholders})", chunk)  # noqa: S608
        return len(ids)

    # --- reporting --------------------------------------------------------

    def indexed_event_ids(self) -> set[int]:
        """Which events already have a vector — the backfill's resume point."""
        with self.conn() as conn:
            return {
                int(row["event_id"])
                for row in conn.execute("SELECT event_id FROM frames WHERE event_id IS NOT NULL")
            }

    def stats(self) -> dict[str, Any]:
        """Size, coverage and span. What ``argus index status`` prints and what
        the search screen shows above its results."""
        empty = {
            "frames": 0, "events": 0, "samples": 0, "cameras": 0,
            "oldest_ts": None, "newest_ts": None, "span_hours": 0.0,
            "model": self.model, "dim": self.dim,
            "path": str(self.path), "size_mb": 0.0, "exists": self.path.exists(),
        }
        if not self.path.exists():
            return empty
        with self.conn() as conn:
            try:
                row = conn.execute(
                    "SELECT COUNT(*) AS frames, "
                    "       SUM(CASE WHEN kind = 'event'  THEN 1 ELSE 0 END) AS events, "
                    "       SUM(CASE WHEN kind = 'sample' THEN 1 ELSE 0 END) AS samples, "
                    "       COUNT(DISTINCT camera_id) AS cameras, "
                    "       MIN(ts) AS oldest_ts, MAX(ts) AS newest_ts FROM frames"
                ).fetchone()
            except sqlite3.OperationalError:
                return empty
            per_camera = [dict(r) for r in conn.execute(
                "SELECT camera_id, COUNT(*) AS frames, MAX(ts) AS newest_ts "
                "FROM frames GROUP BY camera_id ORDER BY camera_id"
            )]
        oldest, newest = row["oldest_ts"], row["newest_ts"]
        return {
            **empty,
            "frames": int(row["frames"] or 0),
            "events": int(row["events"] or 0),
            "samples": int(row["samples"] or 0),
            "cameras": int(row["cameras"] or 0),
            "oldest_ts": oldest,
            "newest_ts": newest,
            "span_hours": round((newest - oldest) / 3600.0, 2) if oldest and newest else 0.0,
            "size_mb": round(self.path.stat().st_size / 1e6, 2),
            "by_camera": per_camera,
        }


def store_from_config(config: Any, *, dim: int, model: str) -> VectorStore:
    store = VectorStore(config.index_db_path, dim=dim, model=model)
    store.migrate()
    return store


__all__ = [
    "INDEX_SCHEMA_VERSION",
    "IndexError_",
    "IndexedFrame",
    "VectorStore",
    "serialize",
    "store_from_config",
]
