#!/usr/bin/env python
"""Argus capacity benchmark — the Phase-1 exit criterion.

The rebuild plan asserts an envelope of "24-40 cameras with sub-stream 5-fps
detection" and says, honestly, that no published NVDEC measurement for GB10
exists. This script replaces that estimate with numbers from this box.

Three modes, cheapest first:

``detector``
    Images/sec against batch size for the adopted precision path, plus
    single-image latency percentiles. Measured through
    ``argus.vision.detector.PersonDetector`` — preprocessing, forward, NMS and
    the copy back to numpy — because a bare ``model(x)`` is not what a camera
    pays for.

``decode``
    Decode ceiling on its own: K concurrent ffmpeg processes chewing the demo
    recording as fast as they can, NVDEC against software, with the GB10's
    decoder engine sampled while they run. This is the number the plan calls
    the appliance's true ceiling.

``capacity``
    The whole appliance. N synthetic cameras are registered through the normal
    camera path (same table, same seed-shaped rows), the real ``argus run`` is
    started as a subprocess, and for each N we measure achieved detection fps
    per camera against the 5 fps target, capture-to-analytics lag, host CPU,
    GPU sm/dec/enc, and resident memory. N steps up until achieved fps falls
    more than 20% below target; the last N that held is the answer.

Everything the capacity mode creates — cameras, zones, events, snapshots,
clips, ring segments, logs — is removed again by ``cleanup``, which the run
also calls for itself in a ``finally``.

    uv run python scripts/benchmark.py detector
    uv run python scripts/benchmark.py decode
    uv run python scripts/benchmark.py capacity --cameras 4,8,16,24,32
    uv run python scripts/benchmark.py cleanup
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from argus.config import ArgusConfig, load_config  # noqa: E402
from argus.db import Camera, Database, Zone  # noqa: E402
from argus.ingest.decoder import (  # noqa: E402
    FFMPEG,
    FFPROBE,
    hwaccel_available,
    is_hardware_mode,
)
from argus.vision.detector import PersonDetector  # noqa: E402

DEFAULT_VIDEO = ROOT / "packing_section_recording.mp4"

# Synthetic cameras are all named from this prefix, which is also how cleanup
# recognises them. Nothing else in the product may create an id starting with it.
BENCH_PREFIX = "bench_cam_"

# The demo zone, copied so the analytics and evidence paths run for real: with a
# limit of 3 the recording goes over it, so every synthetic camera raises events,
# writes annotated snapshots and cuts clips exactly as a deployed one would.
BENCH_POLYGON = [[0.05, 0.2], [0.95, 0.2], [0.95, 0.95], [0.05, 0.95]]
BENCH_LIMIT = 3

# Fraction of the target fps below which a camera counts as degraded.
DEGRADED_AT = 0.8


# ==========================================================================
# telemetry
# ==========================================================================


@dataclass
class GpuMonitor:
    """``nvidia-smi dmon`` in the background, parsed into timestamped samples.

    dmon is the only interface that reports the *decoder* and *encoder* engine
    utilisation separately from sm — and on a box with one NVDEC and one NVENC
    those are the columns that decide capacity. ``--query-gpu`` cannot see them.
    """

    interval_sec: int = 1
    samples: list[dict[str, float]] = field(default_factory=list)
    _proc: subprocess.Popen[str] | None = None
    _thread: threading.Thread | None = None

    def start(self) -> "GpuMonitor":
        if shutil.which("nvidia-smi") is None:
            return self
        # stdbuf when available: dmon writing into a pipe would otherwise be free
        # to block-buffer, and the samples lost in that buffer when we terminate
        # it are precisely the end of the measurement window.
        prefix = ["stdbuf", "-oL"] if shutil.which("stdbuf") else []
        self._proc = subprocess.Popen(
            [*prefix, "nvidia-smi", "dmon", "-s", "ucm", "-d", str(self.interval_sec)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        self._thread = threading.Thread(target=self._read, daemon=True, name="dmon")
        self._thread.start()
        return self

    def _read(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        columns: list[str] = []
        for line in self._proc.stdout:
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "#":
                # Two header lines: names, then units. Only the first is useful,
                # and its column order is not identical across driver versions.
                if fields[1] != "Idx":
                    columns = fields[1:]
                continue
            if not columns or len(fields) != len(columns):
                continue
            sample = {"t": time.time()}
            for name, value in zip(columns, fields):
                try:
                    sample[name] = float(value)
                except ValueError:
                    pass          # '-' for a metric this GPU does not report
            self.samples.append(sample)

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover - defensive
                self._proc.kill()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def between(self, start: float, end: float) -> list[dict[str, float]]:
        return [s for s in self.samples if start <= s["t"] <= end]


def cpu_times() -> tuple[float, float]:
    """(busy, total) jiffies across all cores, from /proc/stat."""
    fields = [float(v) for v in Path("/proc/stat").read_text().split("\n")[0].split()[1:]]
    total = sum(fields)
    idle = fields[3] + (fields[4] if len(fields) > 4 else 0.0)   # idle + iowait
    return total - idle, total


def descendants(root_pid: int) -> list[int]:
    """Every live process under ``root_pid``, itself included.

    Built from one sweep of /proc rather than repeated ``pgrep -P`` calls: at 32
    cameras the tree is ~100 processes and it is sampled once a second.
    """
    children: dict[int, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
        except OSError:
            continue                      # exited between listdir and read
        # comm is parenthesised and may contain spaces, so fields are counted
        # from after the closing paren: state, ppid, ...
        tail = stat[stat.rfind(")") + 2:].split()
        if len(tail) < 2:
            continue
        children.setdefault(int(tail[1]), []).append(int(entry.name))

    found, queue = [root_pid], [root_pid]
    while queue:
        parent = queue.pop()
        for child in children.get(parent, ()):
            found.append(child)
            queue.append(child)
    return found


def process_memory(pids: Sequence[int]) -> tuple[float, float]:
    """(rss_mb, pss_mb) summed over a process tree.

    PSS is reported alongside RSS because ~100 ffmpeg children share one copy of
    libavcodec: summing RSS counts those pages once per process and overstates
    the appliance's footprint by a wide margin. PSS divides shared pages by the
    number of sharers, which is the number that answers "will this fit in 128 GB".
    """
    rss_kb = pss_kb = 0.0
    for pid in pids:
        try:
            rollup = Path(f"/proc/{pid}/smaps_rollup").read_text()
        except OSError:
            continue
        for line in rollup.splitlines():
            if line.startswith("Rss:"):
                rss_kb += float(line.split()[1])
            elif line.startswith("Pss:"):
                pss_kb += float(line.split()[1])
    return rss_kb / 1024.0, pss_kb / 1024.0


def process_rss_mb(pid: int) -> float:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def count_named(pids: Sequence[int], name: str) -> int:
    total = 0
    for pid in pids:
        try:
            if Path(f"/proc/{pid}/comm").read_text().strip() == name:
                total += 1
        except OSError:
            continue
    return total


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


# ==========================================================================
# mode: detector throughput
# ==========================================================================


def demo_frame(width: int, height: int, video: Path = DEFAULT_VIDEO) -> np.ndarray:
    """A real frame from the demo recording.

    Noise would give the detector nothing to find, and an empty frame skips the
    NMS and the device-to-host copy that a busy one pays for.
    """
    if not video.exists():
        print(f"note: {video} missing, benchmarking on noise", file=sys.stderr)
        return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    raw = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-ss", "12", "-i", str(video),
         "-frames:v", "1", "-vf", f"scale={width}:{height}", "-pix_fmt", "bgr24",
         "-f", "rawvideo", "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy()


def _sweep_batches(
    detector: PersonDetector, frame: np.ndarray, batches: Sequence[int], iterations: int
) -> list[dict[str, Any]]:
    rows = []
    for batch in batches:
        images = [frame] * batch
        for _ in range(10):                       # warm the trace and the allocator
            detector.detect(images)
        durations = []
        for _ in range(iterations):
            started = time.perf_counter()
            detections = detector.detect(images)
            durations.append((time.perf_counter() - started) * 1000.0)
        durations.sort()
        rows.append({
            "batch": batch,
            "ms_per_batch": round(mean(durations), 2),
            "ms_per_frame": round(mean(durations) / batch, 3),
            "images_per_sec": round(batch * 1000.0 / mean(durations), 1),
            "p50_ms": round(durations[len(durations) // 2], 2),
            "p95_ms": round(durations[min(len(durations) - 1, int(len(durations) * 0.95))], 2),
            "max_ms": round(durations[-1], 2),
            "people_found": int(len(detections[0])),
        })
        print(f"  {detector.backend:<20} batch {batch:>2}: "
              f"{rows[-1]['images_per_sec']:>7.1f} img/s   "
              f"{rows[-1]['ms_per_frame']:>6.3f} ms/frame   "
              f"p50 {rows[-1]['p50_ms']:>6.2f}  p95 {rows[-1]['p95_ms']:>6.2f}  "
              f"max {rows[-1]['max_ms']:>6.2f} ms", flush=True)
    return rows


def bench_detector(
    config: ArgusConfig, batches: Sequence[int], iterations: int, *, compare: bool = False
) -> dict[str, Any]:
    """Throughput and latency of the shipped detector, batch by batch.

    ``compare`` also runs the paths that were *not* adopted (fp32, and eager
    instead of traced), which is how the precision decision in
    ``docs/BENCHMARKS.md`` was reached and how it can be re-checked when torch or
    the driver moves.
    """
    model = config.detection.model
    frame = demo_frame(config.detection.width, config.detection.height)
    paths = [(model.precision, model.trace)]
    if compare:
        paths += [(model.precision, not model.trace), ("fp32", True), ("fp32", False)]

    runs = []
    for precision, trace in paths:
        detector = PersonDetector(
            variant=model.variant, precision=precision, device=model.device,
            threshold=model.threshold, nms_iou=model.nms_iou, resolution=model.resolution,
            trace=trace,
            # One trace per batch size under test, otherwise the largest batches
            # fall back to eager and the table compares two different code paths.
            max_traced_shapes=len(batches) + 1,
            weights_dir=config.models_dir,
        )
        detector.load()
        rows = _sweep_batches(detector, frame, batches, iterations)
        runs.append({
            "adopted": (precision, trace) == (model.precision, model.trace),
            "detector": detector.describe(),
            "rows": rows,
        })
        del detector

    return {
        "frame": {"width": config.detection.width, "height": config.detection.height},
        "iterations": iterations,
        "runs": runs,
    }


# ==========================================================================
# mode: decode ceiling
# ==========================================================================


def _decode_command(video: Path, config: ArgusConfig, *, hwaccel: bool, loops: int) -> list[str]:
    """The engine's own decode chain, minus the pipe: same filters, same scale.

    ``-f null`` rather than ``-f rawvideo -`` on purpose — this mode measures how
    much video the box can *decode*, not how fast a Python process can drain
    1.5 MB frames off a pipe. The capacity mode measures that.
    """
    detection = config.detection
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin"]
    if hwaccel:
        # The engine's top rung: the frame stays on the device until `fps` has
        # thrown away the four in five the detector will never look at.
        cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        filters = (f"fps={detection.fps},hwdownload,format=nv12,"
                   f"scale={detection.width}:{detection.height}")
    else:
        filters = f"fps={detection.fps},scale={detection.width}:{detection.height}"
    cmd += ["-stream_loop", str(loops), "-i", str(video), "-an", "-sn",
            "-vf", filters, "-f", "null", "-"]
    return cmd


def bench_decode(
    config: ArgusConfig, concurrency: Sequence[int], video: Path, loops: int
) -> dict[str, Any]:
    probe = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames,r_frame_rate", "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    numerator, denominator = stream["r_frame_rate"].split("/")
    source_fps = float(numerator) / float(denominator)
    frames_per_pass = int(stream["nb_frames"]) * (loops + 1)

    modes = ["cpu"] + (["nvdec"] if hwaccel_available(FFMPEG) else [])
    rows = []
    for mode in modes:
        for streams in concurrency:
            gpu = GpuMonitor().start()
            busy_before, total_before = cpu_times()
            started = time.time()
            procs = [
                subprocess.Popen(_decode_command(video, config, hwaccel=mode == "nvdec",
                                                 loops=loops),
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                for _ in range(streams)
            ]
            for proc in procs:
                proc.wait()
            elapsed = time.time() - started
            busy_after, total_after = cpu_times()
            gpu.stop()

            decoded = frames_per_pass * streams
            samples = gpu.between(started, started + elapsed)
            cores = os.cpu_count() or 1
            rows.append({
                "mode": mode,
                "streams": streams,
                "seconds": round(elapsed, 2),
                "decoded_fps": round(decoded / elapsed, 1),
                "realtime_x": round(decoded / elapsed / source_fps, 1),
                "cpu_cores": round((busy_after - busy_before) / (total_after - total_before) * cores, 2),
                "gpu_dec_percent": round(mean([s.get("dec", 0.0) for s in samples]), 1),
                "gpu_sm_percent": round(mean([s.get("sm", 0.0) for s in samples]), 1),
            })
            print(f"  {mode:<5} x{streams:<3} {rows[-1]['decoded_fps']:>8.1f} fps decoded  "
                  f"= {rows[-1]['realtime_x']:>6.1f}x realtime  "
                  f"cpu {rows[-1]['cpu_cores']:>5.2f} cores  "
                  f"nvdec {rows[-1]['gpu_dec_percent']:>5.1f}%", flush=True)

    return {
        "source": {"path": str(video), "fps": round(source_fps, 2),
                   "frames_per_run": frames_per_pass},
        "ffmpeg": FFMPEG,
        "rows": rows,
    }


# ==========================================================================
# mode: end-to-end capacity
# ==========================================================================


def register_cameras(database: Database, count: int, video: Path) -> list[str]:
    """Create N synthetic cameras and one zone each, through the normal path.

    Deliberately the same ``upsert_camera``/``upsert_zone`` the seed importer and
    the API use — a benchmark that inserted rows by hand would be measuring a
    fleet the product cannot actually produce.
    """
    ids = []
    for index in range(1, count + 1):
        camera_id = f"{BENCH_PREFIX}{index:02d}"
        database.upsert_camera(Camera(
            id=camera_id,
            name=f"Benchmark camera {index:02d}",
            source=str(video.resolve()),
            enabled=True,
        ))
        database.upsert_zone(Zone(
            id=f"{camera_id}_zone",
            camera_id=camera_id,
            name="Bench floor",
            polygon=BENCH_POLYGON,
            sanctioned_limit=BENCH_LIMIT,
        ))
        ids.append(camera_id)
    return ids


def park_other_cameras(database: Database) -> list[str]:
    """Disable every real camera for the duration, and report which ones.

    N has to mean N: leaving the two demo cameras enabled would put 34 decoders
    on the box while the table says 32. They are re-enabled by :func:`cleanup`.
    """
    parked = []
    for row in database.list_cameras(enabled_only=True):
        if row["id"].startswith(BENCH_PREFIX):
            continue
        database.update_camera(row["id"], enabled=0)
        parked.append(row["id"])
    return parked


def cleanup(
    config: ArgusConfig,
    database: Database,
    *,
    restore: Sequence[str] = (),
    remove_logs: bool = False,
) -> dict[str, int]:
    """Remove every trace of a benchmark run.

    Rows cascade (zones, events, occupancy, reviews and notifications all
    reference cameras ``ON DELETE CASCADE``), but the files those events wrote
    do not, so the per-camera media directories are removed explicitly.

    Engine logs survive by default — the sweep calls this between points, and
    deleting the log of the run that just failed would be exactly wrong. The
    ``cleanup`` subcommand takes them.
    """
    removed_cameras = 0
    for row in database.list_cameras():
        camera_id = row["id"]
        if not camera_id.startswith(BENCH_PREFIX):
            continue
        database.delete_camera(camera_id)
        removed_cameras += 1

    removed_dirs = 0
    for base in (config.snapshots_dir, config.clips_dir, config.tracks_dir, config.ring_dir):
        if not base.exists():
            continue
        for entry in base.iterdir():
            if entry.is_dir() and entry.name.startswith(BENCH_PREFIX):
                shutil.rmtree(entry, ignore_errors=True)
                removed_dirs += 1

    log_dir = config.logs_dir / "benchmark"
    if remove_logs and log_dir.exists():
        shutil.rmtree(log_dir, ignore_errors=True)

    for camera_id in restore:
        database.update_camera(camera_id, enabled=1)

    return {"cameras": removed_cameras, "directories": removed_dirs, "restored": len(restore)}


@dataclass
class Sample:
    """One reading of the engine's published statistics, at the engine's clock."""

    t: float
    cameras: dict[str, dict[str, Any]]
    detector: dict[str, Any]
    analytics: dict[str, Any] = field(default_factory=dict)
    # True for the snapshot the engine writes at the end of `stop()`. Its
    # counters are final but its clock is not: stopping 32 decoders and 32 ring
    # recorders takes seconds during which nothing is processed, so including it
    # in a rate would charge the shutdown to the frame budget — and charge more
    # of it the more cameras there are, which is exactly the axis under test.
    stopped: bool = False


def read_engine_samples(path: Path, seen_mtime: float) -> tuple[Sample | None, float]:
    """Read ``engine_stats.json`` if it has been rewritten since ``seen_mtime``.

    The timestamp attached is the engine's own (``started_at + uptime_sec``, i.e.
    ``time.time()`` inside ``_write_stats``), not this process's read time: at 5 s
    report ticks, a one-second polling error would be a 1% error on every rate
    derived from the sample pair.
    """
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None, seen_mtime
    if mtime <= seen_mtime:
        return None, seen_mtime
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None, seen_mtime          # mid-rename; the next poll gets it
    started_at = payload.get("started_at")
    if not started_at:
        return None, mtime
    return Sample(
        t=float(started_at) + float(payload.get("uptime_sec") or 0.0),
        cameras={c["camera_id"]: c for c in payload.get("cameras", [])},
        detector=payload.get("detector") or {},
        analytics=payload.get("analytics") or {},
        stopped=bool(payload.get("stopped")),
    ), mtime


def _rate(first: Sample, last: Sample, camera_id: str, key: str) -> float:
    span = last.t - first.t
    if span <= 0:
        return 0.0
    before = first.cameras.get(camera_id, {}).get(key) or 0
    after = last.cameras.get(camera_id, {}).get(key) or 0
    return (after - before) / span


def run_capacity_point(
    config: ArgusConfig,
    database: Database,
    count: int,
    *,
    duration: float,
    warmup: float,
    video: Path,
    report_every: float,
) -> dict[str, Any]:
    """Run the real engine with ``count`` cameras and measure it."""
    camera_ids = register_cameras(database, count, video)
    target_fps = config.detection.fps
    stats_path = config.engine_stats_path
    stats_path.unlink(missing_ok=True)

    log_dir = config.logs_dir / "benchmark"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"capacity_{count:02d}.log"

    gpu = GpuMonitor().start()
    samples: list[Sample] = []
    memory: list[tuple[float, float, float, float, int]] = []   # t, engine rss, tree rss, tree pss, ffmpegs
    mtime = 0.0

    with log_path.open("w") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "argus.cli", "run",
             "--duration", str(duration), "--report-every", str(report_every)],
            cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        )
        started = time.time()
        # (wall clock, busy jiffies, total jiffies) once a second, so CPU can be
        # differenced over exactly the same window as the fps numbers instead of
        # over "whenever the subprocess happened to exit".
        cpu_timeline: list[tuple[float, float, float]] = []
        last_memory_sample = 0.0
        try:
            while proc.poll() is None:
                now = time.time()
                sample, mtime = read_engine_samples(stats_path, mtime)
                if sample is not None:
                    samples.append(sample)
                busy, total = cpu_times()
                cpu_timeline.append((now, busy, total))
                # Memory every 5 s, not every second: reading smaps_rollup for a
                # ~100-process tree is not free, and this benchmark must not be
                # the reason the box misses its frame budget.
                if now - last_memory_sample >= 5.0:
                    last_memory_sample = now
                    tree = descendants(proc.pid)
                    rss, pss = process_memory(tree)
                    memory.append((now, process_rss_mb(proc.pid), rss, pss,
                                   count_named(tree, "ffmpeg")))
                time.sleep(1.0)
            proc.wait(timeout=120)
        finally:
            if proc.poll() is None:      # pragma: no cover - only on interrupt
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
            gpu.stop()

    steady_start = started + warmup
    steady = [s for s in samples if s.t >= steady_start and not s.stopped]
    result: dict[str, Any] = {
        "cameras": count,
        "target_fps": target_fps,
        "duration_sec": duration,
        "warmup_sec": warmup,
        "engine_exit": proc.returncode,
        "samples": len(steady),
        "log": str(log_path),
    }
    if len(steady) < 2:
        result["error"] = ("engine published fewer than two steady-state samples — "
                           f"see {log_path}")
        return result

    first, last = steady[0], steady[-1]
    detect = [_rate(first, last, cid, "processed") for cid in camera_ids]
    decode = [_rate(first, last, cid, "frames") for cid in camera_ids]
    lag = [float(last.cameras.get(cid, {}).get("lag_ms") or 0.0) for cid in camera_ids]
    lag_peak = [float(last.cameras.get(cid, {}).get("lag_ms_peak") or 0.0) for cid in camera_ids]

    live = sum(1 for cid in camera_ids if last.cameras.get(cid, {}).get("status") == "live")
    # "nvdec" is now a rung rather than a flag: the top one keeps frames on the
    # device until after `fps`, the one below downloads everything. Both are
    # hardware decode, and a run where half the fleet is on each is exactly the
    # mixed state the decode-ladder change exists to expose rather than hide.
    modes: dict[str, int] = {}
    for cid in camera_ids:
        mode = str(last.cameras.get(cid, {}).get("mode") or "unknown")
        modes[mode] = modes.get(mode, 0) + 1
    nvdec = sum(count for mode, count in modes.items() if is_hardware_mode(mode))
    restarts = sum(int(last.cameras.get(cid, {}).get("restarts") or 0) for cid in camera_ids)
    events = sum(int(last.cameras.get(cid, {}).get("events") or 0) for cid in camera_ids)
    shed = sum(int(last.cameras.get(cid, {}).get("dropped") or 0) for cid in camera_ids)

    gpu_samples = gpu.between(steady_start, last.t)
    steady_memory = [m for m in memory if steady_start <= m[0] <= last.t]
    window = [entry for entry in cpu_timeline if steady_start <= entry[0] <= last.t]
    cores = os.cpu_count() or 1
    cpu_cores = 0.0
    if len(window) >= 2:
        (_, busy_before, total_before), (_, busy_after, total_after) = window[0], window[-1]
        if total_after > total_before:
            cpu_cores = (busy_after - busy_before) / (total_after - total_before) * cores

    result.update({
        "window_sec": round(last.t - first.t, 1),
        "live_cameras": live,
        "nvdec_cameras": nvdec,
        "decode_mode": "nvdec" if nvdec == count else ("mixed" if nvdec else "cpu"),
        "decode_modes": modes,
        "detect_fps_mean": round(mean(detect), 2),
        "detect_fps_min": round(min(detect), 2),
        "detect_fps_max": round(max(detect), 2),
        "decode_fps_mean": round(mean(decode), 2),
        "decode_fps_min": round(min(decode), 2),
        "lag_ms_mean": round(mean(lag), 1),
        "lag_ms_max": round(max(lag_peak), 1),
        "cpu_cores": round(cpu_cores, 2),
        "cpu_percent_of_box": round(cpu_cores / cores * 100.0, 1),
        "gpu_sm_percent": round(mean([s.get("sm", 0.0) for s in gpu_samples]), 1),
        "gpu_dec_percent": round(mean([s.get("dec", 0.0) for s in gpu_samples]), 1),
        "gpu_enc_percent": round(mean([s.get("enc", 0.0) for s in gpu_samples]), 1),
        "engine_rss_mb": round(max((m[1] for m in steady_memory), default=0.0), 1),
        "tree_rss_mb": round(max((m[2] for m in steady_memory), default=0.0), 1),
        "tree_pss_mb": round(max((m[3] for m in steady_memory), default=0.0), 1),
        "ffmpeg_processes": max((m[4] for m in steady_memory), default=0),
        "decoder_restarts": restarts,
        "events": events,
        "detector_ms_per_frame": last.detector.get("ms_per_frame"),
        "detector_last_batch": last.detector.get("last_batch_size"),
        "analytics_workers": last.analytics.get("workers"),
        "analytics_queued": last.analytics.get("queued"),
        # Frames the detector finished with and the analytics pool then shed.
        # Distinct from frames the vision loop never collected: those never
        # reached the GPU, these did.
        "analytics_dropped": shed,
        "sustained": mean(detect) >= target_fps * DEGRADED_AT and live == count,
    })
    return result


def bench_capacity(
    config: ArgusConfig,
    database: Database,
    counts: Sequence[int],
    *,
    duration: float,
    warmup: float,
    video: Path,
    report_every: float,
) -> dict[str, Any]:
    parked = park_other_cameras(database)
    if parked:
        print(f"parked {len(parked)} non-benchmark camera(s) for the duration: "
              f"{', '.join(parked)}")
    rows: list[dict[str, Any]] = []
    try:
        for count in counts:
            print(f"\n=== {count} cameras — {duration:.0f}s run "
                  f"({warmup:.0f}s warmup) ===", flush=True)
            row = run_capacity_point(
                config, database, count, duration=duration, warmup=warmup,
                video=video, report_every=report_every,
            )
            rows.append(row)
            # flush on every line: a sweep is 20+ minutes, it is normally watched
            # through a redirected log, and buffered progress is no progress.
            if "error" in row:
                print(f"  FAILED: {row['error']}", flush=True)
                break
            print(f"  detect {row['detect_fps_mean']:.2f} fps mean "
                  f"({row['detect_fps_min']:.2f} min) vs {row['target_fps']:.0f} target   "
                  f"lag {row['lag_ms_mean']:.0f} ms   "
                  f"cpu {row['cpu_cores']:.1f} cores   "
                  f"gpu sm/dec {row['gpu_sm_percent']:.0f}/{row['gpu_dec_percent']:.0f}%   "
                  f"rss {row['engine_rss_mb']:.0f} MB engine / {row['tree_pss_mb']:.0f} MB pss",
                  flush=True)
            modes = ", ".join(f"{mode}x{n}" for mode, n in sorted(row["decode_modes"].items()))
            print(f"  {row['live_cameras']}/{count} live, decode={modes}, "
                  f"{row['decoder_restarts']} restart(s), {row['events']} event(s), "
                  f"{row['ffmpeg_processes']} ffmpeg processes, "
                  f"{row['analytics_dropped']} frame(s) shed by analytics", flush=True)
            if not row["sustained"]:
                print(f"  -> below {DEGRADED_AT:.0%} of target; stopping the sweep", flush=True)
                break
            # Between points: give the box its file cache and cores back, and let
            # go2rtc's ports drain, so the next N starts from the same place.
            cleanup(config, database)
            time.sleep(10)
    finally:
        summary = cleanup(config, database, restore=parked)
        print(f"\ncleanup: removed {summary['cameras']} synthetic camera(s), "
              f"{summary['directories']} media directory(ies), "
              f"restored {summary['restored']} camera(s)")

    sustained = [row["cameras"] for row in rows if row.get("sustained")]
    return {
        "target_fps": config.detection.fps,
        "resolution": f"{config.detection.width}x{config.detection.height}",
        "max_sustainable_cameras": max(sustained) if sustained else 0,
        "rows": rows,
    }


# ==========================================================================
# entry point
# ==========================================================================


def _machine() -> dict[str, Any]:
    def first_line(cmd: list[str]) -> str | None:
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
        except (OSError, subprocess.SubprocessError):
            return None
        return (out.stdout or out.stderr).strip().splitlines()[0] if out.stdout or out.stderr else None

    return {
        "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cores": os.cpu_count(),
        "gpu": first_line(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]),
        "ffmpeg": f"{FFMPEG} — {first_line([FFMPEG, '-version']) or 'unknown'}",
    }


def _ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-c", "--config", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None, help="also write results here")
    sub = parser.add_subparsers(dest="mode", required=True)

    detector = sub.add_parser("detector", help="images/sec vs batch size, plus latency")
    detector.add_argument("--batches", default="1,2,4,8,16")
    detector.add_argument("--iterations", type=int, default=60)
    detector.add_argument("--compare", action="store_true",
                          help="also measure the precision/graph paths that were not adopted")

    decode = sub.add_parser("decode", help="NVDEC vs software decode ceiling")
    decode.add_argument("--concurrency", default="1,2,4,8,16")
    decode.add_argument("--loops", type=int, default=9, help="extra passes over the file per process")
    decode.add_argument("--video", type=Path, default=DEFAULT_VIDEO)

    capacity = sub.add_parser("capacity", help="N cameras through the real engine")
    capacity.add_argument("--cameras", default="4,8,16,24,32")
    capacity.add_argument("--duration", type=float, default=165.0, help="seconds per N")
    capacity.add_argument("--warmup", type=float, default=45.0,
                          help="seconds excluded from the steady-state window")
    capacity.add_argument("--report-every", type=float, default=5.0)
    capacity.add_argument("--video", type=Path, default=DEFAULT_VIDEO)

    sub.add_parser("cleanup", help="remove synthetic cameras and benchmark artifacts")

    args = parser.parse_args(argv)
    config = load_config(args.config)
    config.ensure_dirs()

    payload: dict[str, Any] = {"mode": args.mode, "machine": _machine()}
    print(json.dumps(payload["machine"], indent=2))

    if args.mode == "detector":
        payload["result"] = bench_detector(config, _ints(args.batches), args.iterations,
                                           compare=args.compare)
    elif args.mode == "decode":
        payload["result"] = bench_decode(config, _ints(args.concurrency), args.video, args.loops)
    elif args.mode == "capacity":
        database = Database(config.db_path)
        database.migrate()
        payload["result"] = bench_capacity(
            config, database, _ints(args.cameras),
            duration=args.duration, warmup=args.warmup, video=args.video,
            report_every=args.report_every,
        )
        print(f"\nmax sustainable cameras at {config.detection.fps:.0f} fps detection: "
              f"{payload['result']['max_sustainable_cameras']}")
    else:
        database = Database(config.db_path)
        database.migrate()
        payload["result"] = cleanup(config, database, remove_logs=True)
        print(json.dumps(payload["result"], indent=2))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
