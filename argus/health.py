"""What the appliance can say about its own condition.

This backs ``GET /api/health``, and it is deliberately free of any HTTP or
FastAPI import so ``argus status`` and the tests can use it directly.

Three notes on the implementation, all of them consequences of the box:

* **GPU stats come from ``nvidia-smi``, not ``pynvml``.** The only thing that
  ever pulled ``nvidia-ml-py`` into this environment was ``ultralytics``, which
  is AGPL and has been deleted — a health endpoint that broke the day we dropped
  it would have been a trap, and this one did not. ``nvidia-smi`` ships with the
  driver, costs one subprocess, and is cached for a second.
* **GB10 reports no GPU memory.** NVML answers "Not Supported" for
  ``memory.used``/``memory.total`` on Grace Blackwell because the 128 GB is
  unified — it is not a separate pool to report. So memory is read from
  ``/proc/meminfo`` and labelled ``unified`` rather than left blank or, worse,
  invented.
* **Directory sizes are cached.** "How much disk is Argus using" is the number
  that made v1 fail, so it is reported per subdirectory; walking the tree on
  every poll of a dashboard is not, so the walk is cached for 30 s.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from argus.config import ArgusConfig
from argus.redact import redact_text

# Fields asked of nvidia-smi, in order. nounits keeps parsing to float().
_GPU_FIELDS = (
    "index",
    "name",
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "power.draw",
)

_GPU_CACHE_SEC = 1.0
_DIR_CACHE_SEC = 30.0

# Beyond this many seconds without a stats write, `argus run` is presumed gone.
# The engine writes every report tick (10 s by default).
ENGINE_STALE_SEC = 45.0

_MEMINFO = Path("/proc/meminfo")


@dataclass
class _Cached:
    """A value with a time-to-live. Not thread-safe on purpose: a duplicated
    refresh under a race costs one extra subprocess, and a lock here would
    serialise every health request behind the slowest probe."""

    ttl: float
    value: Any = None
    fetched_at: float = 0.0

    def get(self, produce) -> Any:
        now = time.monotonic()
        if self.value is None or now - self.fetched_at >= self.ttl:
            self.value = produce()
            self.fetched_at = now
        return self.value


_gpu_cache = _Cached(_GPU_CACHE_SEC)
_dir_cache: dict[str, _Cached] = {}


# --------------------------------------------------------------------------
# GPU
# --------------------------------------------------------------------------


def _query_nvidia_smi(timeout: float = 3.0) -> str | None:
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return None
    try:
        completed = subprocess.run(
            [binary, f"--query-gpu={','.join(_GPU_FIELDS)}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout, check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return completed.stdout


def _number(text: str) -> float | None:
    """A field nvidia-smi could not answer ('[N/A]', 'Not Supported') is None,
    never zero — a fabricated zero would read as an idle GPU."""
    try:
        return float(text)
    except ValueError:
        return None


def _parse_gpu(output: str) -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    for line in output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(_GPU_FIELDS):
            continue
        index, name, util, used_mb, total_mb, temp, power = parts
        used = _number(used_mb)
        total = _number(total_mb)
        gpus.append({
            "index": int(_number(index) or 0),
            "name": name,
            "utilization_percent": _number(util),
            "memory_used_mb": used,
            "memory_total_mb": total,
            "memory_percent": round(used / total * 100, 1) if used and total else None,
            "temperature_c": _number(temp),
            "power_w": _number(power),
        })
    return gpus


def gpu_info() -> dict[str, Any]:
    """GPU utilization, memory and temperature, or an honest explanation."""

    def probe() -> dict[str, Any]:
        output = _query_nvidia_smi()
        if output is None:
            return {"available": False, "reason": "nvidia-smi not available", "gpus": []}
        gpus = _parse_gpu(output)
        if not gpus:
            return {"available": False, "reason": "nvidia-smi returned no GPUs", "gpus": []}

        report: dict[str, Any] = {"available": True, "gpus": gpus, "memory_source": "device"}
        # Grace Blackwell has no discrete VRAM to report; the honest substitute
        # is the unified pool it actually allocates from.
        if all(gpu["memory_total_mb"] is None for gpu in gpus):
            memory = memory_info()
            report["memory_source"] = "unified"
            report["memory_note"] = (
                "GB10 shares one 128 GB pool with the CPU; NVML reports no "
                "device memory, so these figures are system memory."
            )
            if memory:
                for gpu in gpus:
                    gpu["memory_used_mb"] = memory["used_mb"]
                    gpu["memory_total_mb"] = memory["total_mb"]
                    gpu["memory_percent"] = memory["percent"]
        return report

    return _gpu_cache.get(probe)


# --------------------------------------------------------------------------
# memory & disk
# --------------------------------------------------------------------------


def memory_info() -> dict[str, Any]:
    """System memory from /proc/meminfo. On this box it is also GPU memory."""
    try:
        raw = _MEMINFO.read_text()
    except OSError:
        return {}
    values: dict[str, int] = {}
    for line in raw.splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            values[key] = int(parts[0]) * 1024
    total = values.get("MemTotal")
    # MemAvailable is the kernel's own estimate of what a workload can claim.
    # MemFree is not that number and using it would over-report pressure.
    available = values.get("MemAvailable")
    if not total or available is None:
        return {}
    used = total - available
    return {
        "total_mb": round(total / 1e6, 1),
        "used_mb": round(used / 1e6, 1),
        "available_mb": round(available / 1e6, 1),
        "percent": round(used / total * 100, 1),
    }


def iter_files(path: Path) -> Iterator[tuple[float, int, Path]]:
    """Every file below ``path`` as ``(mtime, size, path)``.

    A generator, not a list: summing sizes must not materialise a tuple per
    file on a directory holding weeks of segments, while retention — which has
    to sort by age — can still ``list()`` it. One walker either way, so
    "how big is this" and "what is oldest here" cannot drift apart.
    Unreadable entries and missing directories are skipped, not raised: this
    backs a health endpoint and a cleanup job, neither of which should fail
    because one file vanished mid-walk.
    """
    for root, _dirs, names in os.walk(path, onerror=lambda _exc: None):
        for name in names:
            entry = Path(root) / name
            try:
                info = entry.stat()
            except OSError:
                continue
            yield (info.st_mtime, info.st_size, entry)


def directory_size(path: Path) -> int:
    """Bytes used under ``path``. Metadata-only walk; missing dirs are 0."""
    return sum(size for _mtime, size, _path in iter_files(path))


def disk_usage(path: Path) -> dict[str, Any]:
    """Filesystem totals for the device holding ``path``.

    ``percent`` is ``used / (used + free)`` — ``df``'s Use%, not ``used /
    total``. The difference is the filesystem's **reserved blocks**: ext4 keeps
    5% for root by default, and ``shutil.disk_usage`` counts those in ``total``
    while leaving them out of both ``used`` and ``free``. Dividing by ``total``
    therefore reports a disk as five points emptier than it is *for this
    appliance*, which runs unprivileged and cannot touch the reserve.

    That matters because this number is the retention guard
    (``retention.max_disk_percent``, default 85) and the Health screen's
    headline. Measured on this appliance 2026-08-25: a 4 031.9 GB volume with
    739.6 GB used and 3 087.5 GB available to us — **204.8 GB, 5.08%, reserved
    and countable as neither**. ``used / total`` called that 18.34%; ``df`` and
    this function call it 19.33%.

    One point of difference is not the point. The ceiling is: under
    ``used / total`` this number **cannot exceed 94.9%** on a default ext4
    volume, so a ``max_disk_percent`` set anywhere above that never fires and
    the emergency reclaim never runs — on a disk with no room left. This
    spelling reaches 100 exactly when the last writable byte is gone, which is
    the only definition a guard can be set against.
    """
    target = path
    while not target.exists() and target != target.parent:
        target = target.parent
    usage = shutil.disk_usage(target)
    writable = usage.used + usage.free
    return {
        "total_gb": round(usage.total / 1e9, 1),
        "used_gb": round(usage.used / 1e9, 1),
        "free_gb": round(usage.free / 1e9, 1),
        "percent": round(usage.used / writable * 100, 1) if writable else 0.0,
    }


def _directory_breakdown(config: ArgusConfig) -> dict[str, float]:
    """Megabytes per evidence directory. Cached as one unit under one TTL:
    the six walks always happen together, and a dashboard polling every second
    should not re-walk weeks of segments to redraw the same numbers."""

    def measure() -> dict[str, float]:
        return {
            name: round(directory_size(directory) / 1e6, 1)
            for name, directory in (
                ("clips", config.clips_dir),
                ("snapshots", config.snapshots_dir),
                ("ring", config.ring_dir),
                ("tracks", config.tracks_dir),
                ("models", config.models_dir),
                # Its own line rather than folded into snapshots: index
                # thumbnails are the only thing on this box that grows on a
                # timer instead of on an event, so an operator watching the
                # disk needs to see them separately from evidence.
                ("index", config.index_dir),
                ("database", config.db_path.parent),
                # Neither is evidence and neither is swept by the age windows,
                # which is exactly why they are listed: a stalled log rotation
                # and a backup directory nobody pruned both show up here as a
                # number that grows, and nowhere else.
                ("backups", config.backups_dir),
                ("logs", config.logs_dir),
            )
        }

    cache = _dir_cache.setdefault(str(config.data_dir), _Cached(_DIR_CACHE_SEC))
    return cache.get(measure)


#: How far past its budget evidence has to be before anything calls it a
#: problem. The sweep reclaims to *exactly* the budget and then the cameras keep
#: recording, so a busy site sits at 100% of its ceiling permanently — that is
#: the mechanism working, not a fault, and a flag that is always true on a
#: healthy appliance is a flag operators learn to ignore. 5% of the budget is
#: roughly what this box writes between two sweeps at its measured 250 GB/day.
#: Past it, the sweep has genuinely not kept up: either it has not run, or
#: `min_keep_hours` is protecting evidence too young to touch.
_BUDGET_SLACK = 1.05


def evidence_usage(config: ArgusConfig,
                   breakdown: dict[str, float] | None = None) -> dict[str, Any]:
    """Evidence held against the budget it is allowed, per media kind.

    Derived from the cached directory breakdown rather than a fresh walk: this
    backs a dashboard vital that polls, and re-walking 40 000 clips to redraw
    the same two numbers is how a health endpoint becomes the load.

    ``budget_gb`` of 0 means the site configured no ceiling, and this reports
    that as ``percent: None`` rather than as 0% or 100%. A console that guessed
    either would be telling an operator something the appliance does not know.

    ``percent`` is the honest ratio; ``over_budget`` carries ``_BUDGET_SLACK``,
    so the console and ``argus doctor`` agree on when to go amber and there is
    one definition of "over" rather than one per surface.
    """
    breakdown = _directory_breakdown(config) if breakdown is None else breakdown
    kinds: dict[str, Any] = {}
    for kind in ("clips", "snapshots"):
        budget = config.retention.budget_gb(kind)
        used = round(breakdown.get(kind, 0.0) / 1000.0, 2)   # the breakdown is MB
        kinds[kind] = {
            "used_gb": used,
            "budget_gb": round(budget, 2) if budget > 0 else None,
            "percent": round(used / budget * 100, 1) if budget > 0 else None,
            "over_budget": budget > 0 and used > budget * _BUDGET_SLACK,
            # True from the moment the ceiling binds — which is what says
            # "this site is budget-bound, not age-bound", a fact worth knowing
            # before it becomes a problem.
            "at_budget": budget > 0 and used >= budget,
        }
    return {
        "kinds": kinds,
        "used_gb": round(sum(k["used_gb"] for k in kinds.values()), 2),
        "budget_gb": round(sum(config.retention.budget_gb(k) for k in kinds), 2) or None,
        "over_budget": any(k["over_budget"] for k in kinds.values()),
        "at_budget": any(k["at_budget"] for k in kinds.values()),
        # Whether per-stage windows are in force at all. The console says
        # "tiered" or names the flat window; both are true statements about a
        # real appliance and neither should be inferred from the numbers.
        "tiered": config.retention.evidence.configured,
    }


def disk_info(config: ArgusConfig) -> dict[str, Any]:
    """Filesystem pressure plus what Argus itself is responsible for."""
    breakdown = _directory_breakdown(config)
    usage = disk_usage(config.data_dir)
    return {
        "data_dir": str(config.data_dir),
        **usage,
        "max_percent": config.retention.max_disk_percent,
        "over_threshold": usage["percent"] > config.retention.max_disk_percent,
        "argus_mb": round(sum(breakdown.values()), 1),
        "by_directory_mb": breakdown,
        # The number the disk percentage cannot give you. A 3.7 TB appliance at
        # 36% full is "fine" right up until it is not, and on this box the
        # difference between those two states was two and a half days.
        "evidence": evidence_usage(config, breakdown),
    }


# --------------------------------------------------------------------------
# engine
# --------------------------------------------------------------------------


def read_engine_stats(config: ArgusConfig) -> dict[str, Any]:
    """The engine's last published statistics, with staleness attached.

    A missing or old file is reported as ``running: false`` rather than as an
    error: the API is perfectly usable with the engine stopped, and saying so
    plainly is more useful than a 500.
    """
    path = config.engine_stats_path
    try:
        payload = json.loads(path.read_text())
        age = max(0.0, time.time() - path.stat().st_mtime)
    except (OSError, json.JSONDecodeError):
        return {"running": False, "reason": "engine has not reported yet", "cameras": []}
    if not isinstance(payload, dict):
        return {"running": False, "reason": "engine stats file is malformed", "cameras": []}
    payload["age_sec"] = round(age, 1)
    if payload.get("stopped"):
        payload["running"] = False
        payload["reason"] = "engine shut down cleanly"
    else:
        payload["running"] = age <= ENGINE_STALE_SEC
        if not payload["running"]:
            payload["reason"] = f"no engine report for {age:.0f}s"
    return payload


def _redacted_ring(ring: Any, source: Any) -> Any:
    """The ring recorder's stats, with its ffmpeg error scrubbed like the decoder's."""
    if isinstance(ring, dict) and ring.get("last_error"):
        return {**ring, "last_error": redact_text(str(ring["last_error"]), source)}
    return ring


def camera_health(config: ArgusConfig, camera_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-camera achieved fps, joining what the engine measured onto what the
    database knows.

    ``target_fps`` is derived here rather than read from the engine: the answer
    is in the camera row and the config, and an achieved rate means nothing
    without the rate that was asked for. Cameras the engine never picked up
    still appear, so a disabled or crashed camera is visible rather than
    merely absent.
    """
    engine = read_engine_stats(config)
    measured = {entry.get("camera_id"): entry for entry in engine.get("cameras", [])}
    result = []
    for row in camera_rows:
        stats = measured.get(row["id"], {})
        result.append({
            "id": row["id"],
            "name": row["name"],
            "enabled": bool(row["enabled"]),
            "status": row["status"],
            # Both error fields are ffmpeg's own words about a URL, so both go
            # through the redactor: a row written before the decoder scrubbed
            # its stderr still has the password in it.
            "status_detail": redact_text(row["status_detail"], row["source"]),
            "last_seen_at": row["last_seen_at"],
            "target_fps": row["detect_fps"] or config.detection.fps,
            "fps": stats.get("fps"),
            "fps_avg": stats.get("fps_avg"),
            "frames": stats.get("frames"),
            "processed": stats.get("processed"),
            # Capture-to-analytics delay. Achieved fps alone cannot tell a box
            # that is keeping up from one that is sampling on time but running
            # further and further behind the live edge.
            "lag_ms": stats.get("lag_ms"),
            "events": stats.get("events"),
            "decode_mode": stats.get("mode"),
            "restarts": stats.get("restarts"),
            "last_error": redact_text(stats.get("last_error"), row["source"]),
            "ring": _redacted_ring(stats.get("ring"), row["source"]),
        })
    return result


def vlm_health(config: ArgusConfig, engine: dict[str, Any] | None = None) -> dict[str, Any]:
    """What the VLM policy engine is doing, from whichever source knows.

    Two sources, and the order matters. The **engine's** published stats are
    authoritative when it is running: only that process knows queue depth, how
    long since the last inference, and which policies are degraded. When it is
    not running, this falls back to probing the endpoint directly, so
    ``argus vlm health`` and ``/api/health`` still answer "is the model up?"
    on a box where the engine is stopped.

    **The two branches answer with the same keys**, because a console panel
    that has to guess which shape arrived ends up rendering a missing counter
    as a zero. What the probe cannot know is ``None`` — never 0, never ``[]``
    where a list would be read as "no policies are running". ``install`` is
    filled in from disk on both paths: it costs three stat calls and it is what
    turns "unreachable" into an actionable sentence.
    """
    engine = read_engine_stats(config) if engine is None else engine
    running = bool(engine.get("running"))
    published = engine.get("vlm") if running else None

    from argus.vlm.server import installed, probe

    base: dict[str, Any] = {
        # The id the appliance is configured to serve, next to whatever is
        # actually being served. They differ exactly when somebody pointed
        # vlm.base_url at another box, which is worth being able to see.
        "configured_model": config.vlm.model,
        "endpoint": config.vlm.endpoint,
        "install": installed(config),
    }
    if published:
        return {"source": "engine", **base, **published}

    server = probe(config)
    available = bool(server.get("available"))
    if not config.vlm.enabled:
        reason = (
            "vlm.enabled is false in the appliance config — prompt policies are "
            "stored but never evaluated"
        )
    elif not available:
        reason = str(server.get("reason") or "endpoint unreachable")
    elif not running:
        reason = "model is up, but `argus run` is not — nothing is sampling frames for it"
    else:
        reason = None

    return {
        "source": "probe",
        **base,
        "enabled": config.vlm.enabled,
        "available": available,
        "model": server.get("model") or config.vlm.model,
        "reason": reason,
        # Only the scheduler counts these. A probe cannot, and must not say 0.
        "degraded_for_sec": None,
        "policies": None,
        "schedules": None,
        "evaluations": None,
        "events": None,
        # The settings are a fact about this box's config file and true whether
        # or not anything is running; the counters are the scheduler's and a
        # probe must not answer 0 for them, for the same reason as above.
        "continuity": {
            "enabled": config.vlm.continuity.enabled,
            "min_iou": config.vlm.continuity.min_iou,
            "stale_after_sec": config.vlm.continuity.stale_after_sec,
            "observations": None,
            "closed": None,
        },
        "verification": {
            "enabled": config.vlm.verify.enabled,
            "stages": config.vlm.verify.stage_list,
            "checked": None,
            "suppressed": None,
            "failures": None,
        },
        "inference": None,
    }


def index_health(config: ArgusConfig, engine: dict[str, Any] | None = None) -> dict[str, Any]:
    """Is search working here, and over how much footage.

    Two sources, like the VLM's: the engine publishes the *writer*'s counters
    (frames indexed, dropped, ms/frame) because it is the only process that has
    them, and this reads the store directly for coverage because that is a fact
    about a file and true whether or not `argus run` is up. A box whose engine
    is stopped can still be searched, and this has to say so.

    Never raises. A missing checkpoint or an unreadable store is a condition
    with a fix, reported in ``reason`` beside the command that fixes it.
    """
    published = (engine or {}).get("index") if (engine or {}).get("running") else None
    model_ready = (
        config.index.model == "fake" or (config.index_model_path / "config.json").is_file()
    )
    coverage: dict[str, Any] | None = None
    reason: str | None = None
    if not model_ready:
        reason = (
            f"SigLIP weights missing at {config.index_model_path} — "
            "run scripts/fetch_siglip.sh"
        )
    elif not config.index_db_path.exists():
        reason = "nothing indexed yet — run `argus run`, or `argus index backfill`"
    else:
        try:
            from argus.index.store import VectorStore

            meta = VectorStore(config.index_db_path, dim=0, model="").meta()
            coverage = VectorStore(
                config.index_db_path, dim=int(meta.get("dim", 0) or 0),
                model=str(meta.get("model", "")),
            ).stats()
        except Exception as exc:  # noqa: BLE001 - health never 500s on one probe
            reason = f"{type(exc).__name__}: {exc}"
    return {
        "source": "engine" if published else "store",
        "enabled": config.index.enabled,
        "configured_model": config.index.model,
        "model_path": str(config.index_model_path),
        "model_ready": model_ready,
        "store_path": str(config.index_db_path),
        "sample_interval_sec": config.index.sample_interval_sec,
        "query_device": config.index.query_device,
        "reason": reason,
        # Writer counters: null when the engine is not running, and null must
        # render as "unknown" rather than as 0 — the same rule as vlm_health.
        "writer": published,
        "coverage": coverage,
    }


def health_report(config: ArgusConfig, database: Any) -> dict[str, Any]:
    """The whole picture: engine, cameras, GPU, disk, database."""
    engine = read_engine_stats(config)
    cameras = database.list_cameras()
    counts = database.counts()
    return {
        "status": "ok",
        "time": time.time(),
        "engine": {
            "running": engine.get("running", False),
            "reason": engine.get("reason"),
            "age_sec": engine.get("age_sec"),
            "started_at": engine.get("started_at"),
            "uptime_sec": engine.get("uptime_sec"),
            "detector": engine.get("detector"),
            "gateway": engine.get("gateway"),
            # What the last housekeeping sweep reclaimed. The number that
            # answers "why is the disk not filling up this time".
            "retention": engine.get("retention"),
        },
        "cameras": camera_health(config, cameras),
        "vlm": vlm_health(config, engine),
        "index": index_health(config, engine),
        "gpu": gpu_info(),
        "memory": memory_info(),
        "disk": disk_info(config),
        "database": {"path": str(config.db_path), "schema_version": database.schema_version(), **counts},
    }


__all__ = [
    "ENGINE_STALE_SEC",
    "camera_health",
    "directory_size",
    "disk_info",
    "disk_usage",
    "gpu_info",
    "health_report",
    "index_health",
    "iter_files",
    "memory_info",
    "read_engine_stats",
    "vlm_health",
]
