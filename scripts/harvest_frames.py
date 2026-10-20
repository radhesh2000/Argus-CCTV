#!/usr/bin/env python
"""Cut native-resolution stills out of the plant clips, one per second.

Phase 1 of `docs/DETECTION_V2.md` §5 is data, and §2.5 says why these stills
must be **native**: the engine decodes 2560x1440 down to 960x544 and the
detector squashes that to 512x512, so a far-end worker reaches the network at
about 4x19 px and is never seen. Anything we pre-label, train on, or judge a
teacher by has to keep the pixels the engine throws away.

One frame per second, not per frame: at 25 fps consecutive frames are the same
people in the same places, and a training set of near-duplicates buys nothing
but overfitting. A second apart, workers have moved.

**The split is a TIME split inside the same five clips, and that is weaker than
a different-day split.** `t < 40 s` is `train`, `40 <= t < 45 s` is a `guard`
band that nothing may ever use, and `t >= 45 s` is `heldout`. The guard exists
because a person standing still at t=39 is still standing there at t=41: without
a gap the "held-out" frames are the training frames with a different filename.
Even with it, held-out frames share this clip's lighting, camera pose, shift and
cast of workers, so a score measured on them is an *upper* bound on what the
same model does on a different day. §2.6 is blunt about the same problem in the
existing 63/16 split. This is the best available until more footage exists; the
real fix is the week of harvest in §5 Phase 1, not a cleverer split of one
minute of video.

`Broken_Biscuits.mov` is deliberately absent: §2.6 records that it is a screen
recording of a video call, and both the detector and the hand labels box a call
participant's webcam face as a worker.

    .venv/bin/python scripts/harvest_frames.py
    .venv/bin/python scripts/harvest_frames.py --clips Loading_Point.MOV --force
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FFMPEG = ROOT / "bin" / "ffmpeg"
FFPROBE = ROOT / "bin" / "ffprobe"

SOURCE_DIR = ROOT / "footage" / "anmol-samples"
OUTPUT_DIR = ROOT / "footage" / "anmol-harvest"

# The five real camera clips. Broken_Biscuits.mov is excluded on purpose (§2.6).
CLIPS: tuple[str, ...] = (
    "Line_No.3_Packing3_Bas.MP4",
    "Line_No.3_Packing4_Bas.MOV",
    "Line_No.3_Packing5_Bas.MP4",
    "Line_No.3_Packing7_Bas.MP4",
    "Loading_Point.MOV",
)

INTERVAL_SEC = 1.0
JPEG_QUALITY = 92

# Seconds. Everything below TRAIN_END may be trained on; the guard band between
# TRAIN_END and HELDOUT_START is never used for anything; from HELDOUT_START on
# is measurement only.
TRAIN_END = 40.0
HELDOUT_START = 45.0


def scene_of(clip: str) -> str:
    """``Line_No.3_Packing4_Bas.MOV`` -> ``Line_No.3_Packing4_Bas``.

    The same stem `argus.eval.detector.scene_of` derives from the hand-labelled
    frame names, so a harvested scene and a labelled scene sort together.
    """
    return Path(clip).stem


def split_of(seconds: float) -> str:
    if seconds < TRAIN_END:
        return "train"
    if seconds < HELDOUT_START:
        return "guard"
    return "heldout"


def probe(path: Path) -> dict[str, object]:
    """Width, height, fps and duration, straight from the container."""
    result = subprocess.run(
        [
            str(FFPROBE), "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,codec_name",
            "-show_entries", "format=duration",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    numerator, _, denominator = stream["r_frame_rate"].partition("/")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "codec": stream["codec_name"],
        "fps": round(float(numerator) / float(denominator or 1), 3),
        "duration_sec": round(float(payload["format"]["duration"]), 3),
    }


def extract(clip_path: Path, scene_dir: Path, *, width: int, height: int, force: bool) -> list[float]:
    """One native RGB frame per second, JPEG-encoded by Pillow.

    ffmpeg streams raw rgb24 rather than writing JPEGs itself so the quality
    setting is a real libjpeg quality number and not ffmpeg's inverted 2-31
    qscale, which is only loosely related to it. The cost is one uncompressed
    frame (11 MB at 2560x1440) crossing a pipe per second of video, which is
    nothing next to decoding the HEVC in the first place.

    ffmpeg's stderr goes to /dev/null: these HEVC clips begin with a run of
    harmless "Could not find ref with POC" warnings, and a scary log for a
    non-problem is worse than no log.
    """
    from PIL import Image

    scene_dir.mkdir(parents=True, exist_ok=True)
    frame_bytes = width * height * 3
    command = [
        str(FFMPEG), "-v", "error", "-i", str(clip_path),
        "-vf", f"fps={1.0 / INTERVAL_SEC}",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    timestamps: list[float] = []
    with open("/dev/null", "wb") as devnull:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=devnull)
        assert process.stdout is not None
        try:
            index = 0
            while True:
                raw = process.stdout.read(frame_bytes)
                if len(raw) < frame_bytes:
                    break
                # The fps filter emits output frame i at output pts i, which is
                # i * INTERVAL_SEC of input — the timestamp the name records.
                seconds = index * INTERVAL_SEC
                out_path = scene_dir / f"t{int(round(seconds)):03d}.jpg"
                if force or not out_path.exists():
                    image = Image.frombytes("RGB", (width, height), raw)
                    image.save(out_path, "JPEG", quality=JPEG_QUALITY, subsampling=0)
                timestamps.append(seconds)
                index += 1
        finally:
            process.stdout.close()
            process.wait()
    if process.returncode not in (0, None):
        raise RuntimeError(f"ffmpeg exited {process.returncode} on {clip_path.name}")
    return timestamps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips", nargs="*", default=list(CLIPS), help="clip filenames under footage/anmol-samples")
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--force", action="store_true", help="re-encode frames that already exist")
    args = parser.parse_args()

    if not FFMPEG.exists() or not FFPROBE.exists():
        print(f"missing {FFMPEG} / {FFPROBE} (run scripts/build_ffmpeg.sh)", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenes: list[dict[str, object]] = []
    frames: list[dict[str, object]] = []

    for clip in args.clips:
        clip_path = args.source_dir / clip
        if not clip_path.exists():
            print(f"missing clip {clip_path}", file=sys.stderr)
            return 1
        scene = scene_of(clip)
        info = probe(clip_path)
        print(f"{scene}: {info['width']}x{info['height']} {info['codec']} "
              f"{info['fps']} fps {info['duration_sec']}s", flush=True)
        timestamps = extract(
            clip_path, args.output_dir / scene,
            width=int(info["width"]), height=int(info["height"]), force=args.force,
        )
        counts = {"train": 0, "guard": 0, "heldout": 0}
        for seconds in timestamps:
            split = split_of(seconds)
            counts[split] += 1
            frames.append({
                "scene": scene,
                "file": f"{scene}/t{int(round(seconds)):03d}.jpg",
                "t_sec": round(seconds, 3),
                "split": split,
                "width": info["width"],
                "height": info["height"],
            })
        scenes.append({"scene": scene, "source": clip, **info, "frames": len(timestamps), "by_split": counts})
        print(f"  {len(timestamps)} frames  train={counts['train']} "
              f"guard={counts['guard']} heldout={counts['heldout']}", flush=True)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "scripts/harvest_frames.py",
        "interval_sec": INTERVAL_SEC,
        "jpeg_quality": JPEG_QUALITY,
        "resolution": "native (no resize, no crop)",
        "split_rule": {
            "train": f"t < {TRAIN_END:g} s",
            "guard": f"{TRAIN_END:g} <= t < {HELDOUT_START:g} s (never used, for anything)",
            "heldout": f"t >= {HELDOUT_START:g} s",
        },
        "split_warning": (
            "This is a TIME split within the same five ~60 s clips, not a different-day "
            "split. Held-out frames share the lighting, camera pose, shift and cast of "
            "workers of the training frames, so any score measured on them is an upper "
            "bound on a different day's performance. The 5 s guard band exists because "
            "people who stand still are otherwise the same sample on both sides of the "
            "boundary; it does not make the split independent. docs/DETECTION_V2.md "
            "§2.6 records the same limitation in the existing 63/16 hand-labelled split, "
            "and §5 Phase 1 says the real fix is more footage from more days. Per that "
            "section's rule -- no pseudo-labels in a test set, ever -- these frames may "
            "pre-label TRAINING data only; a headline accuracy claim needs a frozen, "
            "fully human-verified test set from days the training set never saw."
        ),
        "excluded": {
            "Broken_Biscuits.mov": "screen recording of a video call, not a camera (DETECTION_V2.md §2.6)",
        },
        "scenes": scenes,
        "frames": frames,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    totals = {"train": 0, "guard": 0, "heldout": 0}
    for frame in frames:
        totals[str(frame["split"])] += 1
    print(f"\n{len(frames)} frames from {len(scenes)} scenes -> {args.output_dir}")
    print(f"train={totals['train']}  guard={totals['guard']}  heldout={totals['heldout']}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
