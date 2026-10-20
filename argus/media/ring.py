"""Per-camera ring buffer: continuous recording, cut into evidence on demand.

An alert that arrives with only a still frame gets argued with. An alert that
arrives with the ten seconds before it does not — so every camera records
continuously into short segments, and an event cuts its ±8s out of them.

Two decisions carry this module:

* **Stream copy whenever the source allows it.** The GB10 has one NVENC and 20
  Grace cores; re-encoding every camera continuously (what v1 did — libx264
  ultrafast per camera, forever) spends the box on recording video it already
  had in the right format. The source is probed once, and H.264/H.265 is
  remuxed with ``-c copy`` at essentially zero CPU. Re-encoding is the fallback,
  not the default.
* **The clip is cut by concatenating whole segments, with ``-c copy`` again.**
  Frame-exact trimming would mean decoding and re-encoding, and it buys nothing:
  the reviewer wants context around the event, and segment boundaries land on
  keyframes about a second apart. The event's own timestamp is stored alongside
  the clip's start, so a player can jump straight to the moment.

Segments are named with the epoch second they started (``-strftime 1``), which
is what makes "give me the segments covering this event" a filename comparison
rather than a database.

**And one exception to the second decision.** Recording and *reviewing* do not
want the same codec. Stream-copying an H.265 camera is right for the ring — it
is free, and the ring is only ever read by ffmpeg. It is a gamble for the clip,
because the clip is read by a ``<video>`` element in somebody's browser, and
H.265 there is **conditional**: Chrome and Edge decode it only where the
operating system hands them a hardware decoder, so the same evidence file opens
on a control-room PC with a recent GPU and fails on a Linux thin client, a VM,
or a remote-desktop session. H.264 has no such condition anywhere. So
:meth:`RingRecorder.cut` checks what is actually in the segments and re-encodes
to H.264 on NVENC when the answer is not something every browser plays.

Two honest notes on that, because the evidence for it is narrower than the rule:

* the failure was *not* reproduced on this box. Playwright's Chromium is the
  open-source build and decodes neither H.265 nor H.264, and Firefox 153 with
  Ubuntu's ffmpeg played the stream-copied H.265 clip perfectly. What is
  measured is a portability hazard, not an outage — see
  ``docs/PRODUCTION_READINESS.md``, "Evidence clips from an HEVC camera";
* it costs about 2.2 s of NVENC and 3x the bytes per clip, so
  ``evidence.clip_transcode: never`` is there for a site that knows what its
  reviewers run. NVENC is otherwise idle, and the ring's copy path never uses it.
"""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from argus.db import source_kind
from argus.ingest.decoder import FFMPEG, probe_source
from argus.redact import redact_text

# Codecs an mp4 segment can hold untouched. Anything else is re-encoded.
_COPYABLE_CODECS = frozenset({"h264", "hevc", "h265", "av1", "mpeg4"})

# Codecs a browser will play out of a plain ``<video src=...>``. Deliberately
# SHORTER than the list above, and the gap is the whole point of the exception
# in this module's docstring: H.265 remuxes into mp4 perfectly and then does
# not play.
#
# H.264 is the only universally safe answer. AV1 is here because Chromium ships
# a software decoder for it (dav1d) on every platform, so an AV1 camera — rare
# today, less rare each year — costs nothing to review. MPEG-4 Part 2 is
# copyable and is not playable, which is why this cannot simply be "the codec
# survived the remux".
PLAYABLE_CODECS = frozenset({"h264", "av1"})


def probe_codec(source: str) -> str | None:
    """Video codec of a source, or None when it cannot be determined.

    A failed probe is not an error here: it just means the recorder cannot
    prove stream copy is safe and falls back to re-encoding.
    """
    try:
        return probe_source(source).get("codec_name")
    except (RuntimeError, OSError, subprocess.SubprocessError):
        return None


@dataclass(frozen=True, slots=True)
class Segment:
    path: Path
    start: float                 # epoch second the segment began
    end: float                   # epoch second it ended (next segment's start, or +nominal)


@dataclass(frozen=True, slots=True)
class Clip:
    path: Path
    start: float                 # epoch of the clip's first frame
    duration: float


@dataclass(slots=True)
class RingStats:
    mode: str = "unknown"        # copy | encode | unknown
    codec: str | None = None
    segments: int = 0
    restarts: int = 0
    running: bool = False
    last_error: str | None = None
    bytes_on_disk: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "codec": self.codec,
            "segments": self.segments,
            "restarts": self.restarts,
            "running": self.running,
            "last_error": self.last_error,
            "mb_on_disk": round(self.bytes_on_disk / 1e6, 1),
        }


class RingRecorder:
    """An ffmpeg child writing epoch-named segments, supervised and pruned."""

    def __init__(
        self,
        camera_id: str,
        source: str,
        directory: str | Path,
        *,
        segment_sec: float = 2.0,
        retention_sec: float = 180.0,
        mode: str = "auto",              # auto | copy | encode
        clip_transcode: str = "auto",    # auto | never
        ffmpeg: str = FFMPEG,
        restart_backoff_sec: float = 3.0,
    ):
        if mode not in ("auto", "copy", "encode"):
            raise ValueError(f"unknown ring mode {mode!r}")
        if clip_transcode not in ("auto", "never"):
            raise ValueError(f"unknown clip_transcode {clip_transcode!r}")
        self.camera_id = camera_id
        self.source = source
        self.directory = Path(directory)
        self.segment_sec = float(segment_sec)
        self.retention_sec = float(retention_sec)
        self.mode = mode
        self.clip_transcode = clip_transcode
        self.ffmpeg = ffmpeg
        self.restart_backoff_sec = restart_backoff_sec

        self.stats = RingStats(mode=mode)
        self._stop = threading.Event()
        self._proc: subprocess.Popen[bytes] | None = None
        self._thread: threading.Thread | None = None

    # --- command ----------------------------------------------------------

    def _decide_mode(self) -> str:
        if self.mode == "encode":
            # The segments will be h264_nvenc whatever the source is, so there
            # is nothing a probe could change and no reason to spawn ffprobe.
            return "encode"
        codec = probe_codec(self.source)
        self.stats.codec = codec
        if self.mode == "copy":
            return "copy"
        return "copy" if codec in _COPYABLE_CODECS else "encode"

    def segment_codec(self) -> str | None:
        """What is actually inside the segments on disk, not what the source is.

        These differ exactly when the recorder re-encoded, and the clip cutter
        needs the former: an H.265 camera recorded in ``encode`` mode already
        holds H.264 and must not be re-encoded a second time.
        """
        if self.stats.mode == "encode":
            return "h264"
        return self.stats.codec

    def clip_needs_transcode(self) -> bool:
        """Would a stream-copied clip from these segments fail to play?

        ``None`` — the probe failed and the mode was forced, so nothing here
        knows the codec — counts as yes. Being wrong in that direction costs one
        NVENC pass on a 16 s clip; being wrong the other way costs a reviewer an
        alert they cannot watch, and no log line to explain it.
        """
        if self.clip_transcode == "never":
            return False
        return self.segment_codec() not in PLAYABLE_CODECS

    def build_command(self, mode: str) -> list[str]:
        cmd = [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
        if source_kind(self.source) == "file":
            # A recording standing in for a camera: loop it, pace it at wall
            # clock, and regenerate timestamps so the loop seam does not reach
            # the segmenter as a multi-hour backwards jump.
            cmd += ["-fflags", "+genpts", "-stream_loop", "-1", "-re"]
        else:
            cmd += ["-rtsp_transport", "tcp"]
        cmd += ["-i", self.source, "-an", "-sn"]
        if mode == "copy":
            cmd += ["-c:v", "copy"]
        else:
            # NVENC, not libx264, for the fallback. Two reasons, and both are
            # requirements rather than preferences: libx264 is GPL and the
            # appliance's vendored ffmpeg is built LGPL without it, so the
            # command would simply fail; and re-encoding every unremuxable
            # camera on the Grace cores is the v1 mistake this module exists to
            # undo. The GB10's NVENC is otherwise idle — copy mode never uses it.
            cmd += ["-c:v", "h264_nvenc", "-preset", "p1", "-tune", "ll",
                    "-cq", "26", "-g", "25", "-pix_fmt", "yuv420p"]
        cmd += [
            "-f", "segment",
            "-segment_time", f"{self.segment_sec:g}",
            "-segment_format", "mp4",
            "-reset_timestamps", "1",
            "-strftime", "1",
            str(self.directory / "%s.mp4"),
        ]
        return cmd

    # --- lifecycle --------------------------------------------------------

    def start(self) -> "RingRecorder":
        self.directory.mkdir(parents=True, exist_ok=True)
        self._stop.clear()
        self._thread = threading.Thread(target=self._supervise, daemon=True, name=f"ring-{self.camera_id}")
        self._thread.start()
        return self

    def _supervise(self) -> None:
        mode = self._decide_mode()
        self.stats.mode = mode
        self.stats.running = True
        first = True
        try:
            while not self._stop.is_set():
                if not first:
                    self.stats.restarts += 1
                first = False
                try:
                    proc = subprocess.Popen(
                        self.build_command(mode),
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        stdin=subprocess.DEVNULL,
                    )
                except OSError as exc:
                    self.stats.last_error = f"could not start ffmpeg: {exc}"
                    return
                self._proc = proc
                threading.Thread(
                    target=self._drain_stderr, args=(proc,), daemon=True,
                    name=f"ring-stderr-{self.camera_id}",
                ).start()
                # The local reference matters: stop() clears self._proc from
                # another thread, and reading it here would race into None.
                while proc.poll() is None and not self._stop.is_set():
                    self._stop.wait(1.0)
                    self.prune()
                self._terminate(proc)
                if self._stop.is_set():
                    return
                self._stop.wait(self.restart_backoff_sec)
        finally:
            self.stats.running = False
            self._terminate()

    def _drain_stderr(self, proc: subprocess.Popen[bytes]) -> None:
        stream = proc.stderr
        if stream is None:
            return
        for raw in iter(stream.readline, b""):
            line = raw.decode("utf-8", "replace").strip()
            if line:
                self.stats.last_error = (redact_text(line, self.source) or "")[:200]
        stream.close()

    def _terminate(self, proc: subprocess.Popen[bytes] | None = None) -> None:
        proc = proc or self._proc
        self._proc = None
        if proc is None or proc.poll() is not None:
            return
        # 'q' on stdin is the graceful path, but stdin is closed here; SIGTERM
        # makes ffmpeg finalise the current segment's moov atom before exiting.
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover - stuck child
            proc.kill()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        self._terminate()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def is_running(self) -> bool:
        proc = self._proc
        return proc is not None and proc.poll() is None

    # --- segments ---------------------------------------------------------

    def segments(self, *, include_active: bool = False) -> list[Segment]:
        """Completed segments, oldest first.

        The newest file is the one ffmpeg is still writing: it has no moov atom
        yet and concatenating it produces a broken clip, so it is excluded
        unless explicitly asked for.
        """
        found: list[tuple[float, Path]] = []
        if not self.directory.exists():
            return []
        for path in self.directory.glob("*.mp4"):
            try:
                found.append((float(int(path.stem)), path))
            except ValueError:
                continue
        found.sort()
        if not include_active and found:
            found = found[:-1]
        result: list[Segment] = []
        for index, (start, path) in enumerate(found):
            nominal_end = start + self.segment_sec
            end = found[index + 1][0] if index + 1 < len(found) else nominal_end
            result.append(Segment(path=path, start=start, end=max(end, nominal_end)))
        self.stats.segments = len(result)
        return result

    def segments_covering(self, start: float, end: float) -> list[Segment]:
        """Every completed segment overlapping ``[start, end]``."""
        return [s for s in self.segments() if s.end >= start and s.start <= end]

    def prune(self) -> int:
        """Delete segments older than the retention window. Returns how many."""
        cutoff = time.time() - self.retention_sec
        removed = 0
        size = 0
        for segment in self.segments(include_active=True):
            if segment.start < cutoff:
                segment.path.unlink(missing_ok=True)
                removed += 1
            else:
                try:
                    size += segment.path.stat().st_size
                except OSError:
                    pass
        self.stats.bytes_on_disk = size
        return removed

    # --- cutting ----------------------------------------------------------

    def cut(self, start: float, end: float, destination: str | Path) -> Clip | None:
        """Concatenate the segments covering ``[start, end]`` into one clip.

        Stream copy when the segments already hold something a browser plays —
        a remux costing milliseconds and no NVENC. One NVENC pass otherwise,
        because the clip's only consumer is a ``<video>`` element.
        """
        segments = self.segments_covering(start, end)
        if not segments:
            self.stats.last_error = f"no ring segments cover {start:.0f}..{end:.0f}"
            return None
        return concat_segments(
            segments, destination, ffmpeg=self.ffmpeg, transcode=self.clip_needs_transcode()
        )

    def status(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "directory": str(self.directory),
            # Two facts `stats` does not carry: what is actually inside the
            # segments (which is not the source codec once the recorder has
            # re-encoded), and whether a clip cut from them needs an NVENC
            # pass before a browser will play it. The engine publishes this
            # per camera, so /api/system carries both.
            "segment_codec": self.segment_codec(),
            "clip_transcode": self.clip_needs_transcode(),
            **self.stats.as_dict(),
        }


# Constant-quality target for a transcoded evidence clip. Measured on this box
# against the H.265 it replaces: cq 24 is 30.8 MB for 18 s, cq 28 is 19.4 MB at
# 38.9 dB PSNR-Y, cq 32 is 11.7 MB at 36.2 dB. 28 is the knee — a reviewer
# looking for a phone in somebody's hand cannot see the difference from the
# source, and the file is 40% smaller than the setting above it.
_CLIP_CQ = "28"


def clip_commands(
    list_file: Path, target: Path, *, ffmpeg: str = FFMPEG, transcode: bool = False
) -> list[list[str]]:
    """The ffmpeg invocations to try, best first.

    Copying is one command. Transcoding is two, and the second is not defensive
    padding: ``mpeg4`` is a codec the ring will happily stream-copy and no
    browser will play, and it is also one NVDEC does not decode on this
    generation — so the fallback is exactly the case that reaches here most
    often after H.265.
    """
    head = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    source = ["-f", "concat", "-safe", "0", "-i", str(list_file)]
    tail = ["-movflags", "+faststart", "-an", str(target)]
    if not transcode:
        return [head + source + ["-c", "copy"] + tail]
    encode = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", _CLIP_CQ]
    return [
        # NVDEC in, NVENC out, frames never leaving device memory. Note the
        # absence of `-pix_fmt`: asking for yuv420p here inserts a CPU scaler
        # that cannot take a cuda frame, and ffmpeg fails with "Impossible to
        # convert between the formats supported by the filter". NVENC writes
        # 8-bit 4:2:0 regardless.
        head + ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"] + source + encode + tail,
        # Software decode, NVENC encode. Same wall clock on the clip measured
        # (2.2 s), 17x the CPU — which is the whole reason it is second.
        head + source + encode + ["-pix_fmt", "yuv420p"] + tail,
    ]


def concat_segments(
    segments: list[Segment], destination: str | Path, *, ffmpeg: str = FFMPEG,
    timeout: float = 120.0, transcode: bool = False,
) -> Clip | None:
    """Join ring segments into one mp4, stream-copied unless ``transcode``.

    The transcoding path is NVDEC in and NVENC out with the frames never leaving
    device memory: the Grace cores stay on the vision loop, which is the whole
    reason the ring copies in the first place. ``-preset p4`` rather than the
    recorder's ``p1`` because this runs once per event on eight seconds of
    video, not continuously on every camera — quality is worth the milliseconds
    when a human is going to look at the result and argue about it.
    """
    if not segments:
        return None
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    list_file = target.with_name(target.name + ".concat.txt")
    # Single quotes plus the concat demuxer's own escaping rule; paths are
    # generated by us (epoch names under the data dir), never user input.
    list_file.write_text("".join(f"file '{s.path}'\n" for s in segments))
    try:
        for command in clip_commands(list_file, target, ffmpeg=ffmpeg, transcode=transcode):
            try:
                subprocess.run(command, check=True, capture_output=True, timeout=timeout)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
                continue
            if target.exists() and target.stat().st_size > 0:
                return Clip(
                    path=target,
                    start=segments[0].start,
                    duration=segments[-1].end - segments[0].start,
                )
    finally:
        list_file.unlink(missing_ok=True)
    # Every chain failed. Whatever bytes the last one managed are a headerless
    # mp4 that no player opens, and leaving it behind means `data/clips/` fills
    # with files that look like evidence until somebody clicks one.
    target.unlink(missing_ok=True)
    return None


__all__ = [
    "PLAYABLE_CODECS", "Clip", "RingRecorder", "RingStats", "Segment",
    "clip_commands", "concat_segments", "probe_codec",
]
