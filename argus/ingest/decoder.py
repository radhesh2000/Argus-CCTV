"""Per-camera frame source: ffmpeg subprocess -> BGR numpy arrays.

One ffmpeg child per camera writes raw BGR24 frames to a pipe, already sampled
to the detection fps and scaled to the inference resolution, so Python never
touches a full-rate 1080p frame. Reading a fixed number of bytes per frame and
reshaping is the whole hot path — no OpenCV, no format negotiation.

Hardware decode is a **ladder**, best chain first, and every rung produces
byte-identical frames (verified: mean abs difference 0 over three 960x544
frames):

``nvdec``
    ``-hwaccel_output_format cuda`` with ``fps`` *before* ``hwdownload``. The
    decoded frame stays in device memory and only the 5 fps the detector wants
    is ever copied to the host — at 1080p that is 1 frame in 5, and it is worth
    **10.5x** (36.3 s -> 3.5 s for the same 4 passes over a 1080p25 file).
``nvdec-download``
    ``-hwaccel cuda`` alone: NVDEC decodes, but ffmpeg downloads *every* frame
    at full resolution and then throws 80% of them away on the CPU. This was the
    only hardware chain until 2026-08-12; it is kept as a rung because a codec
    NVDEC cannot handle silently falls back to software here, whereas the chain
    above fails outright.
``cpu``
    Software decode. Always reachable, ~2x the CPU per camera.

A rung is only tried when ffmpeg advertises the cuda hwaccel, which is known
from :func:`hwaccel_available` before any process is spawned. Demotion
distinguishes two failures that used to look the same:

* ffmpeg **refused** the chain (died without the watchdog firing) — the rung is
  disabled for this source permanently, because that is a real incompatibility;
* ffmpeg **started but was starved** (the watchdog killed it after
  ``start_timeout_sec`` with no frame) — the rung is demoted *temporarily* and
  re-armed after a backoff. Starting 96 cameras spawns ~290 ffmpeg processes at
  once and none of them gets its first frame quickly; the old code read that as
  "this build cannot do NVDEC" and latched software decode on the whole fleet
  for the life of the process.

Sources whose ``source`` is a local path are looped forever (``-stream_loop -1``
plus ``-re`` for real-time pacing), which is what makes the recorded demo file
behave like a live camera.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from argus.config import ROOT
from argus.db import source_kind
from argus.redact import redact_source, redact_text


def _resolve(tool: str) -> str:
    """Prefer the vendored build in ``bin/`` over whatever is on PATH.

    ``scripts/build_ffmpeg.sh`` puts an LGPL ffmpeg with NVDEC/NVENC there, the
    same way ``scripts/fetch_gateway.sh`` vendors go2rtc. Distributions ship
    ffmpeg builds that are both GPL (unshippable in a closed appliance) and
    CUDA-less (software decode only, which on this box is the difference between
    ~10 and ~100 cameras), so when the vendored one exists it wins.
    """
    vendored = ROOT / "bin" / tool
    if vendored.is_file() and os.access(vendored, os.X_OK):
        return str(vendored)
    return shutil.which(tool) or tool


FFMPEG = _resolve("ffmpeg")
FFPROBE = _resolve("ffprobe")

# Weight for the exponential moving average behind `DecoderStats.fps`. 0.1 keeps
# roughly a 10-frame memory: responsive enough to show a stalling camera, smooth
# enough to not flicker on jitter.
#
# **It averages the gap between frames, not the rate.** Those are not the same
# estimator and the difference is not academic. Averaging `1/delta` lets a
# single pair of frames that arrive 7 ms apart — which ffmpeg does routinely
# after a pipe hiccup or a `-stream_loop` seam — contribute 143 to the mean, and
# one such sample drags a 10-frame EWMA far above the truth. Measured on this
# appliance 2026-08-25: a camera whose frame counter advanced by a rock-steady
# 70-81 frames every 15 s (5.01 fps, over 45 minutes, zero drops) reported
# **133.39, 126.48 and 96.11 fps** to the console on eight separate samples.
# Averaging the gap and reporting its reciprocal is bounded by construction: one
# 7 ms gap among ten 200 ms ones moves the answer by about 10%.
_EWMA_ALPHA = 0.1

# The decode ladder, best first. See the module docstring for what each rung
# costs and why the middle one exists.
NVDEC = "nvdec"
NVDEC_DOWNLOAD = "nvdec-download"
CPU = "cpu"
DECODE_LADDER: tuple[str, ...] = (NVDEC, NVDEC_DOWNLOAD, CPU)


def is_hardware_mode(mode: str) -> bool:
    """True for any rung that puts the decode on NVDEC rather than the cores."""
    return mode.startswith("nvdec")


@dataclass(frozen=True, slots=True)
class Frame:
    """One decoded frame. ``image`` is BGR uint8, shape (height, width, 3)."""

    camera_id: str
    index: int          # monotonic within a FrameSource, across reconnects
    ts: float           # unix epoch at the moment the frame left the pipe
    image: np.ndarray


@dataclass(slots=True)
class DecoderStats:
    camera_id: str
    mode: str = "unknown"          # nvdec | cpu | unknown
    frames: int = 0
    restarts: int = 0
    # Recent achieved fps: the reciprocal of a smoothed inter-frame gap, not a
    # smoothed rate. See _EWMA_ALPHA for why that distinction is a bug fix.
    fps: float = 0.0
    fps_avg: float = 0.0           # frames / seconds since the first frame
    started_at: float | None = None
    last_frame_at: float | None = None
    last_error: str | None = None
    running: bool = False

    def as_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "mode": self.mode,
            "frames": self.frames,
            "restarts": self.restarts,
            "fps": round(self.fps, 2),
            "fps_avg": round(self.fps_avg, 2),
            "started_at": self.started_at,
            "last_frame_at": self.last_frame_at,
            "last_error": self.last_error,
            "running": self.running,
        }


def hwaccel_available(ffmpeg: str = FFMPEG) -> bool:
    """True when this ffmpeg advertises the cuda hwaccel.

    Cheap and cached by the caller: a build without CUDA fails loudly at
    ``Device creation failed``, and we would rather not pay a process spawn per
    reconnect to rediscover that.
    """
    try:
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-hwaccels"],
            capture_output=True, text=True, timeout=15, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return "cuda" in proc.stdout.split()


_hwaccel_cache: dict[str, bool] = {}


def _hwaccel_supported(ffmpeg: str) -> bool:
    if ffmpeg not in _hwaccel_cache:
        _hwaccel_cache[ffmpeg] = hwaccel_available(ffmpeg)
    return _hwaccel_cache[ffmpeg]


class FrameSource:
    """Iterable stream of :class:`Frame` for one camera.

    ``for frame in source:`` runs until :meth:`stop` is called (or
    ``max_restarts`` is exhausted), transparently restarting ffmpeg with
    exponential backoff when the source drops.
    """

    def __init__(
        self,
        camera_id: str,
        source: str,
        *,
        fps: float = 5.0,
        width: int = 960,
        height: int = 544,
        hwaccel: bool = True,
        realtime: bool = True,
        loop: bool | None = None,
        ffmpeg: str = FFMPEG,
        start_timeout_sec: float = 20.0,
        backoff_sec: float = 2.0,
        backoff_max_sec: float = 30.0,
        max_restarts: int | None = None,
        hwaccel_retry_sec: float = 120.0,
        hwaccel_retry_max_sec: float = 3600.0,
    ):
        self.camera_id = camera_id
        self.source = source
        self.fps = fps
        self.width = width
        self.height = height
        self.want_hwaccel = hwaccel
        self.realtime = realtime
        self.kind = source_kind(source)
        # Files loop by default so a recording behaves like a live camera.
        self.loop = self.kind == "file" if loop is None else loop
        self.ffmpeg = ffmpeg
        self.start_timeout_sec = start_timeout_sec
        self.backoff_sec = backoff_sec
        self.backoff_max_sec = backoff_max_sec
        self.max_restarts = max_restarts
        self.hwaccel_retry_sec = hwaccel_retry_sec
        self.hwaccel_retry_max_sec = hwaccel_retry_max_sec

        self.frame_bytes = width * height * 3
        self.stats = DecoderStats(camera_id=camera_id)

        self._stop = threading.Event()
        self._proc: subprocess.Popen[bytes] | None = None
        self._stderr: deque[str] = deque(maxlen=20)
        self._first_frame_at: float | None = None
        #: Smoothed seconds between frames; `DecoderStats.fps` is its reciprocal.
        self._interval: float = 0.0
        # Where on DECODE_LADDER this source currently sits, which rungs it has
        # proved it cannot use, and when a temporarily-demoted rung may be tried
        # again. ``_starved`` counts *consecutive* watchdog kills on a hardware
        # rung and drives the re-arm backoff, so a box that genuinely cannot
        # start NVDEC stops paying a start_timeout for it every couple of
        # minutes while a box that merely had a busy start recovers.
        self._ladder: tuple[str, ...] | None = None
        self._rung = 0
        self._disabled: set[str] = set()
        self._starved = 0
        self._retry_best_at = 0.0

    # --- command ----------------------------------------------------------

    def ladder(self) -> tuple[str, ...]:
        """The rungs this source may use, resolved once.

        ``hwaccel_available`` costs one process spawn, so it is done here rather
        than per reconnect — and *before* any decoder starts, which is the whole
        point: "this ffmpeg has no CUDA" is knowable without ever failing.
        """
        if self._ladder is None:
            usable = self.want_hwaccel and _hwaccel_supported(self.ffmpeg)
            self._ladder = DECODE_LADDER if usable else (CPU,)
        return self._ladder

    def current_mode(self, now: float | None = None) -> str:
        """Which rung the next spawn will use, re-arming a demoted one if due."""
        ladder = self.ladder()
        now = time.time() if now is None else now
        if self._rung > 0 and self._retry_best_at and now >= self._retry_best_at:
            # A starved attempt is not evidence about the hardware. Climb back.
            self._retry_best_at = 0.0
            self._rung = 0
        while self._rung < len(ladder) - 1 and ladder[self._rung] in self._disabled:
            self._rung += 1
        return ladder[self._rung]

    def _demote(self, mode: str, *, starved: bool, now: float | None = None) -> None:
        ladder = self.ladder()
        now = time.time() if now is None else now
        if starved:
            # The box was too busy to get this camera going, not incapable. Go
            # straight to software — the rung in between downloads *more* frames
            # per second than the one that just starved, so trying it would cost
            # another whole start_timeout to fail the same way — and arm a
            # climb back, backing off if it keeps happening.
            self._starved += 1
            delay = min(
                self.hwaccel_retry_sec * (2 ** (self._starved - 1)), self.hwaccel_retry_max_sec
            )
            self._retry_best_at = now + delay
            self._rung = len(ladder) - 1
            return
        # ffmpeg rejected this chain outright. That is a property of the build,
        # the driver or the codec, and it will not change while this runs.
        self._disabled.add(mode)
        self._rung = min(self._rung + 1, len(ladder) - 1)

    def promotion_due(self, now: float | None = None) -> bool:
        """Whether a temporarily-demoted source should climb back up now.

        Checked while *running*, not only on reconnect: a fleet that fell to
        software during a cold start decodes happily forever afterwards and
        would never reach a reconnect to be promoted at.
        """
        if self._rung == 0 or not self._retry_best_at:
            return False
        return (time.time() if now is None else now) >= self._retry_best_at

    def build_command(self, *, mode: str) -> list[str]:
        cmd = [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin"]

        if mode == NVDEC:
            cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        elif mode == NVDEC_DOWNLOAD:
            cmd += ["-hwaccel", "cuda"]

        if self.kind == "file":
            if self.loop:
                cmd += ["-stream_loop", "-1"]
            if self.realtime:
                # Pace the file at wall-clock speed so it behaves like a camera.
                cmd += ["-re"]
        elif self.kind == "rtsp":
            cmd += ["-rtsp_transport", "tcp"]

        if mode == NVDEC:
            # `fps` runs on the CUDA frames, so four frames in five are dropped
            # while still on the device and never cross PCIe. `hwdownload` needs
            # an explicit `format` — it will not guess which of the hardware
            # frame's supported layouts to produce.
            filters = f"fps={self.fps},hwdownload,format=nv12,scale={self.width}:{self.height}"
        else:
            filters = f"fps={self.fps},scale={self.width}:{self.height}"

        cmd += [
            "-i", self.source,
            "-an", "-sn",
            "-vf", filters,
            "-pix_fmt", "bgr24",
            "-f", "rawvideo",
            "-",
        ]
        return cmd

    # --- process lifecycle ------------------------------------------------

    def _spawn(self, *, mode: str) -> subprocess.Popen[bytes]:
        cmd = self.build_command(mode=mode)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            bufsize=0,
        )
        threading.Thread(
            target=self._drain_stderr, args=(proc,), daemon=True,
            name=f"decoder-stderr-{self.camera_id}",
        ).start()
        return proc

    def _drain_stderr(self, proc: subprocess.Popen[bytes]) -> None:
        """Keep the last few stderr lines. Unread stderr would eventually block
        ffmpeg on a full pipe, so this thread must exist regardless.

        Redacted as they arrive: these lines become ``last_error``, then the
        camera's ``status_detail``, then a cell on the Cameras screen, and
        ffmpeg's way of reporting a camera it cannot open is to quote the URL.
        """
        stream = proc.stderr
        if stream is None:
            return
        for raw in iter(stream.readline, b""):
            line = raw.decode("utf-8", "replace").strip()
            if line:
                self._stderr.append(redact_text(line, self.source))
        stream.close()

    def _kill(self, proc: subprocess.Popen[bytes] | None) -> None:
        """Stop an ffmpeg child, promptly and without ever raising.

        Closing stdout first is what makes this fast: once nobody is reading,
        ffmpeg blocks in ``write()`` on the full frame pipe, and a write-blocked
        process does not act on SIGTERM. Breaking the pipe makes that write fail
        so ffmpeg unwinds on its own; terminate/kill are the backstop.
        """
        if proc is None:
            return
        if proc.stdout is not None:
            try:
                proc.stdout.close()
            except (OSError, ValueError):
                pass
        if proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover - unkillable child
                self._stderr.append("ffmpeg did not exit after SIGKILL")

    def _watchdog(
        self, proc: subprocess.Popen[bytes], got_first: threading.Event, starved: threading.Event
    ) -> None:
        """Kill a source that connects but never delivers. An RTSP camera that
        accepts the TCP connection and then goes quiet would otherwise hang the
        read forever.

        ``starved`` is what separates "the box was busy" from "this chain does
        not work here": only a kill from *here* means ffmpeg was alive and
        producing nothing, which is what a cold-start stampede looks like.
        """
        if got_first.wait(self.start_timeout_sec):
            return
        if proc.poll() is None and not self._stop.is_set():
            starved.set()
            self._stderr.append(f"no frame within {self.start_timeout_sec:g}s — restarting")
            self._kill(proc)

    # --- reading ----------------------------------------------------------

    @staticmethod
    def _read_exactly(stdout, buffer: memoryview) -> bool:
        """Fill ``buffer`` completely from the pipe.

        A pipe hands over at most its buffer's worth per syscall, so a partial
        read is the normal case and must be resumed — treating one as
        end-of-stream is how a decoder ends up restarting forever without ever
        emitting a frame. Only a zero-length read (EOF) or a closed pipe ends
        the stream.
        """
        got, size = 0, len(buffer)
        while got < size:
            try:
                read = stdout.readinto(buffer[got:])
            except (OSError, ValueError):
                return False        # pipe closed under us by stop()
            if not read:
                return False        # EOF: ffmpeg exited
            got += read
        return True

    def _read_frames(self, proc: subprocess.Popen[bytes], got_first: threading.Event) -> Iterator[Frame]:
        stdout = proc.stdout
        assert stdout is not None
        while not self._stop.is_set():
            if self.promotion_due():
                return          # end this run so `frames()` respawns on a better rung
            # A fresh buffer per frame: consumers keep frames (evidence
            # snapshots, the tracker's history), so they cannot share storage.
            raw = bytearray(self.frame_bytes)
            if not self._read_exactly(stdout, memoryview(raw)):
                return
            now = time.time()
            got_first.set()
            if self._starved and self._rung == 0:
                # The best rung is delivering frames here, so whatever starved it
                # before was the moment, not the machine: the next bad start
                # deserves the first, short backoff rather than the last one.
                self._starved = 0
                self._retry_best_at = 0.0
            image = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 3)
            self._record_frame(now)
            yield Frame(camera_id=self.camera_id, index=self.stats.frames, ts=now, image=image)

    def _record_frame(self, now: float) -> None:
        stats = self.stats
        if self._first_frame_at is None:
            self._first_frame_at = now
        elif stats.last_frame_at is not None:
            delta = now - stats.last_frame_at
            if delta > 0:
                # Smooth the *interval*, then report its reciprocal. See
                # `_EWMA_ALPHA` for the eight samples that made this a fix.
                self._interval = delta if self._interval <= 0.0 else (
                    _EWMA_ALPHA * delta + (1 - _EWMA_ALPHA) * self._interval
                )
                stats.fps = 1.0 / self._interval
        stats.frames += 1
        stats.last_frame_at = now
        elapsed = now - self._first_frame_at
        stats.fps_avg = (stats.frames / elapsed) if elapsed > 0 else 0.0

    # --- public API -------------------------------------------------------

    def frames(self) -> Iterator[Frame]:
        """Yield frames forever, restarting the source on loss."""
        self.stats.started_at = time.time()
        self.stats.running = True
        backoff = self.backoff_sec
        attempt = 0
        try:
            while not self._stop.is_set():
                mode = self.current_mode()
                self.stats.mode = mode
                frames_before = self.stats.frames

                try:
                    proc = self._spawn(mode=mode)
                except OSError as exc:
                    self.stats.last_error = f"could not start ffmpeg: {exc}"
                    return
                self._proc = proc

                got_first = threading.Event()
                starved = threading.Event()
                threading.Thread(
                    target=self._watchdog, args=(proc, got_first, starved), daemon=True,
                    name=f"decoder-watchdog-{self.camera_id}",
                ).start()

                try:
                    yield from self._read_frames(proc, got_first)
                finally:
                    self._kill(proc)
                    self._proc = None

                if self._stop.is_set():
                    return

                produced = self.stats.frames > frames_before
                self.stats.last_error = "; ".join(self._stderr) or "source ended"

                if not produced and mode != CPU:
                    # Down the ladder and straight back round: an immediate retry
                    # on a chain that might work beats backing off into a loop of
                    # identical failures. Whether the demotion is permanent
                    # depends on *how* it failed.
                    self._demote(mode, starved=starved.is_set())
                    self._stderr.clear()
                    continue

                if produced and self.promotion_due():
                    # A deliberate restart to climb back to hardware decode, not
                    # a fault: no backoff, and not counted against the camera.
                    self._stderr.clear()
                    continue

                attempt += 1
                if self.max_restarts is not None and attempt > self.max_restarts:
                    return
                self.stats.restarts += 1
                self._stderr.clear()

                if produced:
                    backoff = self.backoff_sec  # healthy run: reset the backoff
                if self._stop.wait(backoff):
                    return
                backoff = min(backoff * 2, self.backoff_max_sec)
        finally:
            self.stats.running = False
            self._kill(self._proc)

    def __iter__(self) -> Iterator[Frame]:
        return self.frames()

    def stop(self) -> None:
        self._stop.set()
        self._kill(self._proc)


class FrameSourceThread:
    """Run a :class:`FrameSource` on a daemon thread, keeping the latest frame.

    Stage 2 replaces the callback with the batched detector; for now this is
    what lets ``argus run`` prove every camera is really decoding.
    """

    def __init__(self, source: FrameSource, on_frame=None):
        self.source = source
        self.on_frame = on_frame
        self._latest: Frame | None = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"decoder-{source.camera_id}"
        )

    def _run(self) -> None:
        for frame in self.source.frames():
            with self._lock:
                self._latest = frame
            if self.on_frame is not None:
                self.on_frame(frame)

    def start(self) -> "FrameSourceThread":
        self._thread.start()
        return self

    def latest(self) -> Frame | None:
        with self._lock:
            return self._latest

    @property
    def stats(self) -> DecoderStats:
        return self.source.stats

    def stop(self, timeout: float = 10.0) -> None:
        self.source.stop()
        self._thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        return self._thread.is_alive()


def source_for_camera(
    camera: dict,
    config,
    *,
    override_source: str | None = None,
    start_timeout_sec: float | None = None,
) -> FrameSource:
    """Build a FrameSource from a cameras row plus the appliance config.

    ``override_source`` lets the engine pull through the gateway's RTSP endpoint
    instead of hitting the original source a second time. ``start_timeout_sec``
    lets it widen the first-frame deadline for a large fleet, which the config
    default cannot know about.
    """
    detection = config.detection
    return FrameSource(
        camera_id=camera["id"],
        source=override_source or camera["source"],
        fps=camera.get("detect_fps") or detection.fps,
        width=detection.width,
        height=detection.height,
        hwaccel=detection.hwaccel,
        start_timeout_sec=(
            detection.start_timeout_sec if start_timeout_sec is None else start_timeout_sec
        ),
        backoff_sec=detection.reconnect_backoff_sec,
        backoff_max_sec=detection.reconnect_backoff_max_sec,
        hwaccel_retry_sec=detection.hwaccel_retry_sec,
        hwaccel_retry_max_sec=detection.hwaccel_retry_max_sec,
    )


def probe_source(source: str, *, timeout: float = 30.0) -> dict:
    """ffprobe summary of a source — used by setup/diagnostics, not the hot path.

    ``timeout`` is the wall-clock deadline *and* is handed to ffprobe as
    ``-rw_timeout``, so a camera at an address nothing answers gives up rather
    than being killed: a process that reports "connection timed out" tells an
    installer more than one that vanishes. It is a parameter because the caller
    decides how long is too long — 30 s is right for a setup script and far too
    long for a form somebody is standing in front of.
    """
    cmd = [
        FFPROBE, "-v", "error",
        # microseconds; ffmpeg's networking protocols all read this one.
        "-rw_timeout", str(int(max(1.0, timeout) * 1_000_000)),
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,codec_name,r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=0", source,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=max(1.0, timeout) + 2.0, check=False)
    if proc.returncode != 0:
        # Redacted before it is cut: this message reaches the enrolment form.
        detail = redact_text(proc.stderr.strip(), source) or ""
        raise RuntimeError(f"ffprobe failed for {redact_source(source)}: {detail[:200]}")
    out: dict = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            out[key] = value
    return out


__all__ = [
    "CPU",
    "DECODE_LADDER",
    "FFMPEG",
    "FFPROBE",
    "NVDEC",
    "NVDEC_DOWNLOAD",
    "DecoderStats",
    "Frame",
    "FrameSource",
    "FrameSourceThread",
    "hwaccel_available",
    "is_hardware_mode",
    "probe_source",
    "source_for_camera",
]
