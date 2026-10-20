"""Turning video into a labellable, reproducible set of stills.

The extraction itself is four lines of ffmpeg. The part worth reading is *what
gets written down*: every frame this module produces is described in
``evals/labels.yaml`` by its source video, its offset in seconds and a content
digest — never by its bytes.

That split is the same one Phase 3 made between the vector store and the saved
searches. A JPEG cut out of a video already in the repository is derived data:
regenerable exactly, worthless in a diff, and pure weight in a clone.
A human's answer to "is that person on their phone?" is not derived from
anything, cannot be regenerated, and is the only thing here worth versioning.
So ``evals/frames/`` is gitignored and ``evals/labels.yaml`` is committed, and
``ensure_frames`` re-cuts whatever a fresh checkout is missing.

The digest is what keeps that honest. If a source video is replaced by a
different file of the same name, the re-cut frames will not match the digests
recorded beside the labels, and the harness says so instead of scoring old
judgements against new pictures.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from argus.ingest.decoder import FFMPEG, FFPROBE

# Videos this box has to evaluate on, relative to the repo root. The stock clip
# in footage/ is gitignored and may be absent; more NVR footage is expected
# later, so the directory is scanned rather than listed.
VIDEO_SUFFIXES = (".mp4", ".mkv", ".mov", ".avi", ".m4v")

# How many directory levels below ``footage/`` :func:`find_videos` will walk.
# One covers "the customer sent a folder", which is what actually happened;
# two leaves room for the folder-of-folders an NVR export arrives as. It is
# bounded at all because ``footage/`` is also where somebody mounts a share,
# and an unbounded walk of one is a command that appears to hang.
MAX_SCAN_DEPTH = 2


class FrameError(RuntimeError):
    """A video could not be probed or a frame could not be cut."""


@dataclass(slots=True)
class SampledFrame:
    """One still, and the recipe that produced it."""

    id: str
    path: Path
    source: str        # repo-relative path of the video
    ts_sec: float      # offset into that video
    width: int = 0
    height: int = 0


def find_videos(root: Path, extra_dirs: tuple[Path, ...] = ()) -> list[Path]:
    """Test footage, in the order a reviewer would think of it.

    ``packing_section_recording.mp4`` first because it is the clip every other
    measurement in this repository was taken on, then whatever is in
    ``footage/``. Both may be absent — a checkout without the demo video is a
    valid checkout, and the harness has to say so rather than crash.

    ``footage/`` is walked **recursively**, and that is not tidiness. A
    customer's corpus does not arrive as loose files: Anmol's came as
    ``footage/anmol-samples/`` with six videos and a labelled image dataset
    beside them, and a non-recursive scan found none of it — so
    ``--build-labels`` silently produced a label file about the demo clip while
    the plant footage sat one directory down. Depth is capped at
    ``MAX_SCAN_DEPTH`` because ``footage/`` is also where an operator drops an
    NVR export, and an unbounded walk of a mounted share is a command that
    appears to hang.
    """
    videos: list[Path] = []
    primary = root / "packing_section_recording.mp4"
    if primary.is_file():
        videos.append(primary)
    for directory in (root / "footage", *extra_dirs):
        if not directory.is_dir():
            continue
        videos.extend(_scan(directory, depth=MAX_SCAN_DEPTH))
    return videos


def _scan(directory: Path, *, depth: int) -> list[Path]:
    """Videos under ``directory``, files before subdirectories, sorted at each
    level so the order a label file is built in is reproducible."""
    found: list[Path] = []
    try:
        entries = sorted(directory.iterdir())
    except OSError:
        return found
    for path in entries:
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
            found.append(path)
    if depth <= 0:
        return found
    for path in entries:
        # `is_dir()` follows symlinks; a link back up the tree would loop
        # forever, and a footage directory is exactly where somebody symlinks a
        # mount. The depth cap alone would end it, but not before it had walked
        # the same share twice.
        if path.is_dir() and not path.is_symlink():
            found.extend(_scan(path, depth=depth - 1))
    return found


def probe_duration(video: Path) -> float:
    """Seconds of content, via ffprobe. 0.0 when it cannot be determined."""
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [FFPROBE, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(video)],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise FrameError(f"ffprobe failed on {video.name}: {exc}") from exc
    try:
        return max(0.0, float(completed.stdout.strip()))
    except ValueError:
        return 0.0


def sample_offsets(duration: float, every_sec: float, limit: int) -> list[float]:
    """Where to cut, given a duration.

    Offsets start half an interval in rather than at 0.0 for a mundane reason
    that costs a whole frame otherwise: the first frame of a clip is often a
    fade-in or a black frame, and a black frame is a case nobody can label.
    """
    if duration <= 0 or every_sec <= 0 or limit <= 0:
        return []
    start = min(every_sec / 2.0, max(0.0, duration - 0.05))
    offsets: list[float] = []
    position = start
    while position < duration and len(offsets) < limit:
        offsets.append(round(position, 2))
        position += every_sec
    return offsets


def cut_frame(video: Path, ts_sec: float, out_path: Path, *, quality: int = 3) -> Path:
    """Extract one frame at ``ts_sec`` into ``out_path``.

    ``-ss`` before ``-i`` so ffmpeg seeks rather than decodes up to the offset:
    on a 12-minute source that is the difference between a second and a minute
    per frame, and the small loss of seek precision does not matter for a still
    somebody is about to look at and label.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin",
        "-ss", f"{ts_sec:.3f}", "-i", str(video),
        "-frames:v", "1", "-q:v", str(quality), "-y", str(out_path),
    ]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True,  # noqa: S603
                                   timeout=120, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise FrameError(f"ffmpeg failed on {video.name} @ {ts_sec}s: {exc}") from exc
    if completed.returncode != 0 or not out_path.is_file():
        detail = (completed.stderr or "").strip().splitlines()[-1:] or ["no output"]
        raise FrameError(f"ffmpeg failed on {video.name} @ {ts_sec}s: {detail[0]}")
    return out_path


def frame_id(video: Path, ts_sec: float) -> str:
    """A stable, filesystem-safe id: ``<video-stem>-<offset in centiseconds>``.

    Stable is the requirement. This id is the key a human's label is filed
    under, so it must survive re-extraction, a different working directory and
    a re-run of ``--build-labels`` with the same interval.
    """
    stem = "".join(char if char.isalnum() else "-" for char in video.stem).strip("-")
    return f"{stem[:40]}-{int(round(ts_sec * 100)):07d}"


def sample_video(
    video: Path, frames_dir: Path, *, every_sec: float, limit: int, root: Path,
) -> list[SampledFrame]:
    """Cut a video into stills at a fixed interval."""
    duration = probe_duration(video)
    out: list[SampledFrame] = []
    for offset in sample_offsets(duration, every_sec, limit):
        identifier = frame_id(video, offset)
        path = cut_frame(video, offset, frames_dir / f"{identifier}.jpg")
        out.append(SampledFrame(
            id=identifier, path=path,
            source=_relative(video, root), ts_sec=offset,
        ))
    return out


def ensure_frame(entry: dict, frames_dir: Path, root: Path) -> Path | None:
    """Re-cut a labelled frame that is not on disk. None when it cannot be.

    This is what makes the gitignored frames directory safe: a clone has
    labels and no pictures, and the first ``argus eval`` restores the pictures
    from the recipe. A source video that is genuinely gone (the stock clip in
    ``footage/``, say) returns None, and the caller drops those cases with a
    note rather than failing the run — losing a slice is not losing the harness.
    """
    path = frames_dir / f"{entry['id']}.jpg"
    if path.is_file():
        return path
    video = root / str(entry.get("source") or "")
    if not video.is_file():
        return None
    try:
        return cut_frame(video, float(entry.get("ts_sec") or 0.0), path)
    except (FrameError, TypeError, ValueError):
        return None


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


__all__ = [
    "FrameError",
    "SampledFrame",
    "cut_frame",
    "ensure_frame",
    "find_videos",
    "frame_id",
    "probe_duration",
    "sample_offsets",
    "sample_video",
]
