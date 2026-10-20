#!/usr/bin/env python3
"""Count shipping cartons as they come down the Loading Point conveyor.

A lab prototype for Radhesh's use case: how many cartons went onto the trucks.
It follows the standing pattern for a new class (docs/REBUILD_PLAN.md):
auto-label with a slow zero-shot model, check by eye, train a fast closed-set
one. Then it adds what a count needs and a detector alone does not: a tracker
for stable ids and a line each id is counted at once.

Nothing in ``argus/`` imports this file. It works on the one Loading Point
recording we have (``footage/anmol-samples/Loading_Point.MOV``, 61 s, 2560x1440
HEVC) and writes to ``datasets/carton-loading-point/`` (gitignored).

    uv run python scripts/carton_counter.py frames              # masked crops, time split
    uv run python scripts/carton_counter.py autolabel           # OWLv2 drafts + preview
    uv run python scripts/carton_counter.py stage               # YOLO layout for train_detector
    uv run python scripts/train_detector.py --dataset datasets/carton-loading-point/yolo \\
        --class-name carton --output models/finetuned/carton-loading-point-nano
    uv run python scripts/carton_counter.py kymo                # time-strip for a hand count
    uv run python scripts/carton_counter.py count --checkpoint models/finetuned/carton-loading-point-nano/checkpoint_best_total.pth

WHY A MASKED CROP
=================
The camera sees thousands of cartons: the dispatch stacks fill half the frame.
Only the ones riding the belt are being loaded. So the model only ever sees the
belt: frames are cropped to the conveyor's bounding box and everything outside
the belt band is painted grey, in training and at inference alike. A stacked
carton is then never a training example and never a detection, and the count
line cannot be crossed by anything that is not on the belt.

THE SPLIT
=========
Same time split as scripts/harvest_frames.py: ``t < 40 s`` train, ``40-45 s`` a
guard band nothing uses, ``t >= 45 s`` held out. One 61 s clip is enough to prove
the pipeline, not to claim accuracy on another day or another light.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Footage, datasets, models and bin/ are gitignored, so a git worktree has none
# of them; ARGUS_DATA_ROOT points at the checkout that does.
DATA_ROOT = Path(os.environ.get("ARGUS_DATA_ROOT") or ROOT)
FFMPEG = DATA_ROOT / "bin" / "ffmpeg"
CLIP = DATA_ROOT / "footage" / "anmol-samples" / "Loading_Point.MOV"
DATA = DATA_ROOT / "datasets" / "carton-loading-point"
OWL = DATA_ROOT / "models" / "google" / "owlv2-base-patch16-ensemble"

# Geometry in native 2560x1440 pixels, read off a gridded frame of the clip.
# The top of the beam the cartons ride on runs along `beam(x)`; cartons stand up
# to ~170 px above it, and the dispatch stacks start well below it.
def beam(x: float) -> float:
    return 400 - 0.283 * (x - 1710)

CROP = (1380, 0, 2560, 524)                       # x0, y0, x1, y1 of the belt's bounding box; even sizes, or ffmpeg rounds the crop and every raw frame read after the first is shifted
CROP_W, CROP_H = CROP[2] - CROP[0], CROP[3] - CROP[1]
# The band starts at x=1520, short of the transfer table: workers' hands and the
# cartons parked at the end live left of that, and nothing there is counted.
ROI = [(1520, beam(1520) - 170), (2560, 0.0), (2560, beam(2560) + 30), (1520, beam(1520) + 30)]
LINE = ((1800, beam(1800) - 170), (1800, beam(1800) + 30))   # vertical, mid-belt, before the table
GREY = 114
SPLITS = (("train", 0.0, 40.0), ("guard", 40.0, 45.0), ("heldout", 45.0, 1e9))
QUERIES = ["a cardboard box", "a carton box"]


def to_crop(points):
    return [(x - CROP[0], y - CROP[1]) for x, y in points]


def roi_mask() -> np.ndarray:
    """Boolean H x W mask of the belt band, in crop coordinates."""
    from PIL import Image, ImageDraw
    canvas = Image.new("L", (CROP_W, CROP_H), 0)
    ImageDraw.Draw(canvas).polygon(to_crop(ROI), fill=255)
    return np.asarray(canvas) > 0


def apply_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = frame.copy()
    out[~mask] = GREY
    return out


def decode(clip: Path, fps: float | None):
    """Yield (t_seconds, H x W x 3 uint8 crop) from the clip, at `fps` or every frame."""
    vf = [f"crop={CROP_W}:{CROP_H}:{CROP[0]}:{CROP[1]}"]
    if fps:
        vf.insert(0, f"fps={fps}")
    cmd = [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-i", str(clip),
           "-vf", ",".join(vf), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    native_fps = fps or _clip_fps(clip)
    size = CROP_W * CROP_H * 3
    with subprocess.Popen(cmd, stdout=subprocess.PIPE) as proc:
        index = 0
        while True:
            raw = proc.stdout.read(size)
            if len(raw) < size:
                break
            yield index / native_fps, np.frombuffer(raw, np.uint8).reshape(CROP_H, CROP_W, 3)
            index += 1


def _clip_fps(clip: Path) -> float:
    probe = subprocess.run(
        [str(FFMPEG).replace("ffmpeg", "ffprobe"), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(clip)],
        capture_output=True, text=True, check=True).stdout.strip()
    num, _, den = probe.partition("/")
    return float(num) / float(den or 1)


def split_of(t: float) -> str:
    return next(name for name, lo, hi in SPLITS if lo <= t < hi)


# --- frames ------------------------------------------------------------------

def cmd_frames(args) -> int:
    from PIL import Image
    mask = roi_mask()
    counts: dict[str, int] = {}
    for t, frame in decode(args.clip, args.fps):
        split = split_of(t)
        if split == "guard":
            continue
        out = DATA / "raw" / split / f"t{round(t * 1000):06d}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(apply_mask(frame, mask)).save(out, quality=92)
        counts[split] = counts.get(split, 0) + 1
    print(f"frames at {args.fps} fps -> {DATA / 'raw'}: {counts}")
    return 0


# --- autolabel -----------------------------------------------------------------

def cmd_autolabel(args) -> int:
    import torch
    import torchvision
    from PIL import Image, ImageDraw
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Owlv2Processor.from_pretrained(str(OWL))
    model = Owlv2ForObjectDetection.from_pretrained(str(OWL)).to(device).eval()
    mask = roi_mask()
    images = sorted((DATA / "raw").glob("*/*.jpg"))
    stats = {"images": 0, "boxes": 0, "dropped_outside": 0, "dropped_size": 0}
    previews = []
    for path in images:
        image = Image.open(path).convert("RGB")
        inputs = processor(text=[QUERIES], images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # OWLv2 pads to a square; the target size is the padded square's side.
        side = max(image.size)
        post = processor.post_process_grounded_object_detection(
            outputs=outputs, threshold=args.threshold, target_sizes=torch.tensor([[side, side]]))[0]
        boxes, scores = post["boxes"].cpu(), post["scores"].cpu()
        keep = torchvision.ops.nms(boxes, scores, args.iou) if len(boxes) else []
        lines, drawn = [], []
        for i in keep:
            x0, y0, x1, y1 = [float(v) for v in boxes[i]]
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(CROP_W, x1), min(CROP_H, y1)
            w, h = x1 - x0, y1 - y0
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if not mask[int(min(cy, CROP_H - 1)), int(min(cx, CROP_W - 1))]:
                stats["dropped_outside"] += 1
                continue
            if not (args.min_px <= w <= args.max_px and args.min_px <= h <= args.max_px):
                stats["dropped_size"] += 1
                continue
            lines.append(f"0 {cx / CROP_W:.6f} {cy / CROP_H:.6f} {w / CROP_W:.6f} {h / CROP_H:.6f}")
            drawn.append((x0, y0, x1, y1, float(scores[i])))
        label = DATA / "labels-draft" / path.parent.name / (path.stem + ".txt")
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("\n".join(lines) + ("\n" if lines else ""))
        stats["images"] += 1
        stats["boxes"] += len(lines)
        if len(previews) < args.preview:
            canvas = image.copy()
            d = ImageDraw.Draw(canvas)
            for x0, y0, x1, y1, s in drawn:
                d.rectangle([x0, y0, x1, y1], outline=(255, 40, 40), width=4)
                d.text((x0 + 4, y0 + 4), f"{s:.2f}", fill=(255, 255, 0))
            d.text((6, 6), f"{path.parent.name}/{path.stem}  n={len(drawn)}", fill=(255, 255, 0))
            previews.append(canvas)
    if previews:
        cols = 3
        tw, th = CROP_W // 2, CROP_H // 2
        sheet = Image.new("RGB", (tw * cols, th * (-(-len(previews) // cols))), "black")
        for i, p in enumerate(previews):
            sheet.paste(p.resize((tw, th)), ((i % cols) * tw, (i // cols) * th))
        sheet.save(DATA / "autolabel_preview.jpg", quality=85)
    (DATA / "autolabel_stats.json").write_text(json.dumps({**stats, "threshold": args.threshold}, indent=2))
    print(json.dumps(stats))
    return 0


# --- bglabel -------------------------------------------------------------------

def cmd_bglabel(args) -> int:
    """Label every carton on the belt by background subtraction.

    OWLv2 found about half the cartons on this belt. This scene allows a better
    labeller: the camera is fixed and the belt never changes, so the per-pixel
    median of the training frames is the empty belt, and a carton is whatever
    differs from it. Computed from `train` frames only, so held-out frames are
    labelled against a background they did not help build.

    Writes the same YOLO drafts `autolabel` does, so `stage` takes either.
    """
    from PIL import Image, ImageDraw, ImageFilter
    from scipy import ndimage

    mask = roi_mask()
    train = sorted((DATA / "raw" / "train").glob("*.jpg"))
    background = np.median(np.stack([np.asarray(Image.open(p).convert("RGB"), dtype=np.int16)
                                     for p in train]), axis=0).astype(np.int16)
    Image.fromarray(background.astype(np.uint8)).save(DATA / "background.jpg", quality=92)
    stats = {"images": 0, "boxes": 0, "dropped_small": 0, "dropped_edge": 0}
    previews = []
    for path in sorted((DATA / "raw").glob("*/*.jpg")):
        image = Image.open(path).convert("RGB")
        blurred = np.asarray(image.filter(ImageFilter.GaussianBlur(3)), dtype=np.int16)
        diff = np.abs(blurred - background).max(axis=2)
        fg = (diff > args.diff) & mask
        fg = ndimage.binary_opening(fg, iterations=2)
        fg = ndimage.binary_closing(fg, iterations=6)
        fg = ndimage.binary_fill_holes(fg)
        labels, n = ndimage.label(fg)
        candidates = []
        for sl in ndimage.find_objects(labels):
            y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
            w, h = x1 - x0, y1 - y0
            if w < args.min_px or h < args.min_px:
                stats["dropped_small"] += 1
                continue
            # A carton still entering at the top-right edge is only partly in frame;
            # a box that runs to the crop's right edge is not a whole-carton label.
            if x1 >= CROP_W - 15 and w < args.edge_min_w:
                stats["dropped_edge"] += 1
                continue
            # A rail crossing a carton's top can split off a flat sliver of it.
            if h < 0.3 * w:
                stats["dropped_flat"] = stats.get("dropped_flat", 0) + 1
                continue
            candidates.append((x0, y0, x1, y1))
        # ...and that sliver, or a shadow piece, can sit inside the carton's own box.
        kept = []
        for box in sorted(candidates, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True):
            x0, y0, x1, y1 = box
            area = (x1 - x0) * (y1 - y0)
            inside = any(
                max(0, min(x1, k[2]) - max(x0, k[0])) * max(0, min(y1, k[3]) - max(y0, k[1])) > 0.5 * area
                for k in kept)
            if inside:
                stats["dropped_inside"] = stats.get("dropped_inside", 0) + 1
                continue
            kept.append(box)
        lines, drawn = [], []
        for x0, y0, x1, y1 in kept:
            w, h = x1 - x0, y1 - y0
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            lines.append(f"0 {cx / CROP_W:.6f} {cy / CROP_H:.6f} {w / CROP_W:.6f} {h / CROP_H:.6f}")
            drawn.append((x0, y0, x1, y1))
        label = DATA / "labels-draft" / path.parent.name / (path.stem + ".txt")
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("\n".join(lines) + ("\n" if lines else ""))
        stats["images"] += 1
        stats["boxes"] += len(lines)
        if path.parent.name == "train" and len(previews) < args.preview and int(path.stem[1:]) % 3000 == 0:
            canvas = image.copy()
            d = ImageDraw.Draw(canvas)
            for x0, y0, x1, y1 in drawn:
                d.rectangle([x0, y0, x1, y1], outline=(0, 255, 120), width=4)
            d.text((6, 6), f"{path.parent.name}/{path.stem}  n={len(drawn)}", fill=(255, 255, 0))
            previews.append(canvas)
    if previews:
        cols, tw, th = 3, CROP_W // 2, CROP_H // 2
        sheet = Image.new("RGB", (tw * cols, th * (-(-len(previews) // cols))), "black")
        for i, p in enumerate(previews):
            sheet.paste(p.resize((tw, th)), ((i % cols) * tw, (i // cols) * th))
        sheet.save(DATA / "bglabel_preview.jpg", quality=85)
    (DATA / "bglabel_stats.json").write_text(json.dumps({**stats, "diff": args.diff}, indent=2))
    print(json.dumps(stats))
    return 0


# --- stage ---------------------------------------------------------------------

def cmd_stage(args) -> int:
    """`raw/{train,heldout}` + `labels-draft` -> `yolo/{images,labels}/{train,val}`."""
    import shutil
    out = DATA / "yolo"
    counts = {}
    for src_split, dst_split in (("train", "train"), ("heldout", "val")):
        for image in sorted((DATA / "raw" / src_split).glob("*.jpg")):
            label = DATA / "labels-draft" / src_split / (image.stem + ".txt")
            (out / "images" / dst_split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / dst_split).mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, out / "images" / dst_split / image.name)
            shutil.copy2(label, out / "labels" / dst_split / label.name)
            counts[dst_split] = counts.get(dst_split, 0) + 1
    print(f"staged {counts} -> {out}")
    return 0


# --- kymograph -------------------------------------------------------------------

def cmd_kymo(args) -> int:
    """One row per frame, sampled along the belt: each carton becomes a streak.

    The path runs parallel to the beam, 60 px above it, through the middle of a
    riding carton. Time runs down the image. The count line is the red column;
    cartons are counted by eye as streaks that cross it.
    """
    from PIL import Image, ImageDraw
    xs = np.linspace(CROP[0], CROP[2] - 1, args.samples)
    ys = np.array([beam(x) - 60 for x in xs])
    px, py = (xs - CROP[0]).astype(int), np.clip((ys - CROP[1]).astype(int), 0, CROP_H - 1)
    rows = [frame[py, px] for _, frame in decode(args.clip, None)]
    image = Image.fromarray(np.stack(rows))
    d = ImageDraw.Draw(image)
    line_col = int(np.argmin(np.abs(xs - LINE[0][0])))
    d.line([(line_col, 0), (line_col, image.height)], fill=(255, 0, 0), width=2)
    fps = _clip_fps(args.clip)
    for second in range(0, int(len(rows) / fps) + 1, 5):
        y = int(second * fps)
        d.line([(0, y), (12, y)], fill=(255, 255, 0), width=2)
        d.text((14, y - 6), f"{second}s", fill=(255, 255, 0))
    image.save(DATA / "kymograph.jpg", quality=90)
    print(f"kymograph {image.size} (x along belt, y = time at {fps:g} fps) -> {DATA / 'kymograph.jpg'}; "
          f"count line at column {line_col}")
    return 0


# --- count -----------------------------------------------------------------------

def cmd_count(args) -> int:
    import supervision as sv
    from PIL import Image, ImageDraw
    from rfdetr import RFDETRNano

    sys.path.insert(0, str(ROOT))
    from argus.vision.tracker import CameraTracker

    model = RFDETRNano(pretrain_weights=str(args.checkpoint))
    mask = roi_mask()
    fps = args.fps or _clip_fps(args.clip)
    tracker = CameraTracker("loading_point", fps=fps)
    (a, b) = to_crop(LINE)
    line = sv.LineZone(start=sv.Point(*a), end=sv.Point(*b), triggering_anchors=[sv.Position.CENTER])
    crossings = []
    writer = None
    if args.video:
        out_video = DATA / f"count_{fps:g}fps.mp4"
        writer = subprocess.Popen(
            [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{CROP_W}x{CROP_H}", "-r", f"{fps}", "-i", "-", "-c:v", "h264_nvenc", "-pix_fmt", "yuv420p",
             str(out_video)], stdin=subprocess.PIPE)
    frames = 0
    for t, frame in decode(args.clip, args.fps):
        frames += 1
        masked = apply_mask(frame, mask)
        dets = model.predict(Image.fromarray(masked), threshold=args.threshold)
        dets = tracker.update(dets)
        before = line.in_count + line.out_count
        line.trigger(dets)
        if line.in_count + line.out_count > before:
            crossings.append(round(t, 2))
        if writer:
            canvas = Image.fromarray(masked)
            d = ImageDraw.Draw(canvas)
            for (x0, y0, x1, y1), tid in zip(dets.xyxy, dets.tracker_id if dets.tracker_id is not None else []):
                d.rectangle([x0, y0, x1, y1], outline=(0, 255, 120), width=4)
                d.text((x0 + 4, y0 + 4), f"#{tid}", fill=(0, 255, 120))
            d.line([a, b], fill=(255, 40, 40), width=6)
            d.text((10, 10), f"t={t:5.1f}s  count={line.in_count + line.out_count}", fill=(255, 255, 0))
            writer.stdin.write(np.asarray(canvas).tobytes())
    if writer:
        writer.stdin.close()
        writer.wait()
    result = {"fps": fps, "frames": frames, "threshold": args.threshold,
              "count": line.in_count + line.out_count, "in": line.in_count, "out": line.out_count,
              "crossing_times_s": crossings}
    (DATA / f"count_{fps:g}fps.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("frames"); p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--clip", type=Path, default=CLIP); p.set_defaults(func=cmd_frames)
    p = sub.add_parser("autolabel"); p.add_argument("--threshold", type=float, default=0.15)
    p.add_argument("--iou", type=float, default=0.5); p.add_argument("--min-px", type=float, default=40)
    p.add_argument("--max-px", type=float, default=420); p.add_argument("--preview", type=int, default=12)
    p.set_defaults(func=cmd_autolabel)
    p = sub.add_parser("bglabel"); p.add_argument("--diff", type=int, default=35)
    p.add_argument("--min-px", type=int, default=50); p.add_argument("--edge-min-w", type=int, default=120)
    p.add_argument("--preview", type=int, default=12); p.set_defaults(func=cmd_bglabel)
    p = sub.add_parser("stage"); p.set_defaults(func=cmd_stage)
    p = sub.add_parser("kymo"); p.add_argument("--samples", type=int, default=600)
    p.add_argument("--clip", type=Path, default=CLIP); p.set_defaults(func=cmd_kymo)
    p = sub.add_parser("count"); p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--fps", type=float, default=5.0); p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--video", action="store_true"); p.add_argument("--clip", type=Path, default=CLIP)
    p.set_defaults(func=cmd_count)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
