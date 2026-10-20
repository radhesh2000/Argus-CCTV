#!/usr/bin/env python
"""Train the student on the views it will be asked to answer on.

`docs/DETECTION_V2.md` §2.5: *"It is not a config switch; the model must be
trained on what it will be shown."* The production Nano was fine-tuned on full
frames squashed to 384 and then asked to work on native tiles, where it doubled
its false positives. The SAHI result quoted there is the same story — slicing at
inference alone is worth ~+5 AP50, slicing in training as well ~+14.

So every frame enters this corpus **twice**: once whole, and once as the same
3x2 overlapping native tiles `scripts/teacher_label.py` uses, planned by the
same `plan_tiles` so a tile here is pixel-identical to a tile at inference.

Three sources, and the split rule is the load-bearing part:

| source | frames | labels | split |
|---|---|---|---|
| `footage/anmol-harmonized` train | 63 | human presence, teacher extent | train |
| `footage/anmol-harvest` **train** | 200 | teacher @ score >= 0.5 | train |
| `footage/anmol-harmonized` val | 16 | human presence, teacher extent | valid |

`guard` and `heldout` are never read. The guard band exists to be unused; the
heldout frames are the only thing resembling a measurement of this student, and
a model trained on its own test set measures nothing. `--splits` cannot name
them: the check is in code rather than in a habit.

**Teacher labels are taken at 0.5, not the 0.3 the sidecars were written at.**
Filtering the written boxes is exactly equivalent to re-running the merge at
0.5 — NMS processes boxes in descending score, so whether a box survives depends
only on higher-scoring boxes, and every box the 0.3 pass dropped was dropped by
one scoring above it. 0.3 is right for a human reviewer, who can delete a wrong
box in one click; nothing reviews these, so the operating point moves to where
the teacher's centre-in-box precision is 0.95 rather than 0.80
(docs/TEACHER_LABELS.md §3).

**A box clipped to a tile is kept only if ~40% of it survives.** A sliver of
shoulder at a tile edge is not a person, and a detector taught that it is will
put boxes on shoulders. Boxes that keep more than that are taught clipped, which
is what the model is asked to emit on a tile — and the seam rule at inference
drops those boxes anyway, so the only thing this teaches is *where the person
is*, which is correct in the tile's own frame.

    .venv/bin/python scripts/build_student_dataset.py
    .venv/bin/python scripts/build_student_dataset.py --limit 4 --out-dir /tmp/smoke

**What v1 got wrong, and the three flags that fix it.** That corpus was 14%
full frames and 86% tiles, and the model optimised for the 86%: it became the
best detector measured on tiles and went blind on the full-frame engine path
the appliance actually runs (docs/STUDENT_DETECTOR.md §8). Three things follow.

* `--engine-view` adds a **third** view of every frame: the frame resized to
  960x544 the way the decoder does it — filtered, aspect ignored — which the
  detector then squashes to its square input. The full frame and the engine
  view are not the same picture. One is 2560x1440 downsampled once; the other
  is downsampled twice, the second time through a different aspect ratio. v1
  was shown only the first and asked to answer on the second.
* `--full-repeat N` shows each full-frame view N times per epoch, with the
  trainer's own augmentation making each copy different. This is the knob that
  buys parity, and it is oversampling rather than tile-subsampling because
  throwing tiles away would change two things at once — the mix and the amount
  of data — and only one of them is the question.
* `--tile-keep K` then trims tiles from K=6 so the *total* corpus stays the
  size v1's was. Which K tiles are kept rotates with the frame index, so every
  tile position is still seen on K/6 of the frames rather than some region of
  every frame being dropped for good.
* `--exclude` drops harvested frames a `scripts/leakage_check.py
  --write-exclusions` report names, which is how the val split stops being
  partly inside the training set.

    # parity: 2 native + 2 engine + 4 tiles = 8 views, half of them full-frame
    .venv/bin/python scripts/build_student_dataset.py \
        --out-dir footage/anmol-student-parity --exclude footage/leak-exclusions.json \
        --engine-view --full-repeat 2 --tile-keep 4
    # engine-only: 4 native + 4 engine = 8 views, no tiles at all
    .venv/bin/python scripts/build_student_dataset.py \
        --out-dir footage/anmol-student-engineonly --exclude footage/leak-exclusions.json \
        --engine-view --full-repeat 4 --no-tiles

Output is plain single-class YOLO (`0 cx cy w h`) under
`footage/anmol-student/`, which is the layout `scripts/train_detector.py`
stages from — it does the shift to person-on-class-1 itself, and that decision
stays in one place.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.teacher_label import (  # noqa: E402
    TILE_COLS, TILE_OVERLAP, TILE_ROWS, Tile, _assert_not_ground_truth, plan_tiles,
)

HARMONIZED = ROOT / "footage" / "anmol-harmonized"
HARVEST = ROOT / "footage" / "anmol-harvest"
TEACHER_LABELS = HARVEST / "labels_teacher"
OUT_DIR = ROOT / "footage" / "anmol-student"

# Rendered views (tile crops, engine-view resizes) live here rather than beside
# each dataset: two datasets cut from the same frames want the same pixels, and
# 492 MB of identical JPEGs per experiment is a way to run out of disk deciding
# nothing.
CACHE_DIR = ROOT / "footage" / "anmol-student"

# The engine's decode size — config/argus.yaml detection.width/height. A view at
# this size, squashed to the model's square input, is the pair of resamples the
# appliance performs on every frame it looks at.
ENGINE_WIDTH, ENGINE_HEIGHT = 960, 544

# See the module docstring: equivalent to re-merging at this threshold.
TEACHER_TRAIN_THRESHOLD = 0.50

# A clipped box must keep this much of its original area to stay a person.
MIN_KEPT_AREA = 0.40

# A box thinner than this in either direction after clipping is a line, not a
# person, whatever its area ratio says.
MIN_SIDE_PX = 4.0

TILE_JPEG_QUALITY = 92

# The splits this script will read from the harvest, ever.
ALLOWED_HARVEST_SPLITS = frozenset({"train"})


@dataclass(slots=True)
class SourceFrame:
    """One native frame and its boxes, before any view is cut from it."""

    key: str                     # the flat stem every view is named after
    scene: str
    image: Path
    width: int
    height: int
    boxes: np.ndarray            # (N, 4) xyxy in native pixels
    origin: str                  # "harmonized" | "teacher"
    split: str                   # "train" | "val"


# --------------------------------------------------------------------------
# reading the two corpora
# --------------------------------------------------------------------------


def _read_yolo(path: Path, width: int, height: int) -> np.ndarray:
    boxes: list[list[float]] = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 5:
            raise SystemExit(f"{path}:{number}: expected 'class cx cy w h'")
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        boxes.append([(cx - bw / 2) * width, (cy - bh / 2) * height,
                      (cx + bw / 2) * width, (cy + bh / 2) * height])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def read_harmonized(root: Path, split: str) -> list[SourceFrame]:
    from PIL import Image
    from argus.eval.detector import scene_of

    images_dir, labels_dir = root / "images" / split, root / "labels" / split
    if not images_dir.is_dir():
        raise SystemExit(f"{images_dir} is missing — run scripts/harmonize_labels.py first")
    frames: list[SourceFrame] = []
    for image in sorted(images_dir.iterdir()):
        if image.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        label = labels_dir / f"{image.stem}.txt"
        if not label.exists():
            raise SystemExit(f"no harmonised label for {image.name}")
        with Image.open(image) as handle:
            width, height = handle.size
        frames.append(SourceFrame(
            key=image.stem, scene=scene_of(image.stem), image=image.resolve(),
            width=width, height=height, boxes=_read_yolo(label, width, height),
            origin="harmonized", split="train" if split == "train" else "val",
        ))
    return frames


def read_harvest(harvest: Path, labels: Path, splits: Iterable[str], threshold: float) -> list[SourceFrame]:
    """Harvested frames with the teacher's boxes, filtered to ``threshold``.

    The score comes from the per-frame JSON sidecar rather than the `.txt`,
    which carries no scores: a label file that cannot say how confident the
    teacher was cannot be filtered honestly, and filtering is the whole point
    of reading the sidecar.
    """
    splits = set(splits)
    forbidden = splits - ALLOWED_HARVEST_SPLITS
    if forbidden:
        raise SystemExit(
            f"refusing to build training data from harvest split(s) {sorted(forbidden)}. "
            f"`guard` exists to be unused and `heldout` is the measurement — "
            f"docs/TEACHER_LABELS.md §2."
        )
    manifest = json.loads((harvest / "manifest.json").read_text())
    frames: list[SourceFrame] = []
    for entry in manifest["frames"]:
        if entry["split"] not in splits:
            continue
        scene, name = str(entry["file"]).split("/", 1)
        stem = Path(name).stem
        sidecar = labels / scene / f"{stem}.json"
        if not sidecar.exists():
            raise SystemExit(f"no teacher sidecar at {sidecar} — run scripts/teacher_label.py")
        payload = json.loads(sidecar.read_text())
        kept = [b["xyxy"] for b in payload["boxes"] if float(b["score"]) >= threshold]
        frames.append(SourceFrame(
            # `_frame_` so `argus.eval.detector.scene_of` still reads the scene
            # off the flat filename; `t` marks it as a harvested second rather
            # than a hand-labelled frame index.
            key=f"{scene}_frame_{stem}",
            scene=scene, image=(harvest / str(entry["file"])).resolve(),
            width=int(payload["width"]), height=int(payload["height"]),
            boxes=np.asarray(kept, dtype=np.float32).reshape(-1, 4),
            origin="teacher", split="train",
        ))
    return frames


# --------------------------------------------------------------------------
# views
# --------------------------------------------------------------------------


def clip_to_tile(boxes: np.ndarray, tile: Tile) -> tuple[np.ndarray, int]:
    """Boxes in the tile's own pixels, slivers dropped.

    Returns the surviving boxes and how many were dropped for being truncated
    past the point where what is left is still recognisably a person.
    """
    if len(boxes) == 0:
        return np.zeros((0, 4), np.float32), 0
    clipped = boxes.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], tile.x0, tile.x1)
    clipped[:, 1] = np.clip(clipped[:, 1], tile.y0, tile.y1)
    clipped[:, 2] = np.clip(clipped[:, 2], tile.x0, tile.x1)
    clipped[:, 3] = np.clip(clipped[:, 3], tile.y0, tile.y1)

    widths, heights = clipped[:, 2] - clipped[:, 0], clipped[:, 3] - clipped[:, 1]
    original = np.maximum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1e-6)
    keep = (
        (widths * heights / original >= MIN_KEPT_AREA)
        & (widths >= MIN_SIDE_PX) & (heights >= MIN_SIDE_PX)
    )
    survivors = clipped[keep]
    survivors[:, 0::2] -= tile.x0
    survivors[:, 1::2] -= tile.y0
    # A box that never intersected the tile at all is not a "dropped sliver" —
    # it is simply somewhere else, and counting it would make the drop rate a
    # statement about tile geometry rather than about truncation.
    touched = (
        (boxes[:, 2] > tile.x0) & (boxes[:, 0] < tile.x1)
        & (boxes[:, 3] > tile.y0) & (boxes[:, 1] < tile.y1)
    )
    return survivors, int((touched & ~keep).sum())


def write_label(path: Path, boxes: np.ndarray, width: int, height: int) -> None:
    _assert_not_ground_truth(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for x0, y0, x1, y1 in boxes:
        lines.append(
            f"0 {(x0 + x1) / 2 / width:.6f} {(y0 + y1) / 2 / height:.6f} "
            f"{(x1 - x0) / width:.6f} {(y1 - y0) / height:.6f}"
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def link_image(source: Path, destination: Path) -> None:
    _assert_not_ground_truth(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        destination.unlink()
    destination.symlink_to(source)


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------


def tiles_kept(tiles: list[Tile], keep: int, index: int) -> list[Tile]:
    """``keep`` of the planned tiles, rotating with the frame index.

    Taking the first ``keep`` every time would delete a fixed strip of every
    frame from the corpus — precisely the "a band of the image only ever reaches
    the network at full-frame scale" failure `plan_tiles` was written to avoid.
    Rotating instead means every tile position appears on ``keep``/``len`` of
    the frames, and the whole image is still covered at tile scale.
    """
    total = len(tiles)
    if keep >= total:
        return tiles
    start = (index * keep) % total
    return [tiles[(start + offset) % total] for offset in range(keep)]


def engine_view(image: Path, cache: Path, quality: int) -> Path:
    """The frame at 960x544, cached — the decoder's output, reproduced.

    Pillow's BILINEAR on a reduction is a properly filtered resample, which is
    what ffmpeg's scaler does in the live pipeline and what
    `argus/eval/detector.py:_resize` uses to stand in for it. The aspect ratio
    is *not* preserved, deliberately: the appliance does not preserve it either,
    and a student trained on the undistorted frame is a student that has never
    seen the shapes it will be asked about.
    """
    from PIL import Image

    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(image) as handle:
            handle.convert("RGB").resize(
                (ENGINE_WIDTH, ENGINE_HEIGHT), Image.BILINEAR
            ).save(cache, "JPEG", quality=quality, subsampling=0)
    return cache


def build(frames: list[SourceFrame], out_dir: Path, *, cols: int, rows: int, overlap: float,
          tiles_on: bool, quality: int, cache_dir: Path, engine_on: bool,
          full_repeat: int, tile_keep: int) -> dict[str, Any]:
    from PIL import Image

    import shutil

    stats: Counter = Counter()
    per_split: dict[str, Counter] = {"train": Counter(), "val": Counter()}
    # A rebuild with different flags must not inherit the previous one's views.
    # Only the symlink trees go: the rendered caches below are content-addressed
    # by name and are the expensive part.
    for stale in (out_dir / "images", out_dir / "labels"):
        if stale.exists():
            shutil.rmtree(_assert_not_ground_truth(stale))
    tile_dir = _assert_not_ground_truth(cache_dir / "tiles")
    tile_dir.mkdir(parents=True, exist_ok=True)
    engine_dir = _assert_not_ground_truth(cache_dir / "engine")

    for index, frame in enumerate(frames):
        images_dir = out_dir / "images" / frame.split
        labels_dir = out_dir / "labels" / frame.split

        def emit(stem: str, image: Path, boxes: np.ndarray, width: int, height: int,
                 kind: str) -> None:
            link_image(image, images_dir / f"{stem}.jpg")
            write_label(labels_dir / f"{stem}.txt", boxes, width, height)
            stats["views"] += 1
            stats[f"{kind}_views"] += 1
            stats["boxes"] += len(boxes)
            per_split[frame.split]["views"] += 1
            per_split[frame.split]["boxes"] += len(boxes)

        # --- the whole frame, and the 960x544 the engine would hand over -----
        # Both are full-frame views and both are repeated `full_repeat` times.
        # The repeats are separate files pointing at the same pixels: the
        # trainer's resize/crop/flip augmentation is sampled per __getitem__,
        # so N copies are N different pictures of the same moment, which is
        # what oversampling is supposed to mean.
        full_forms: list[tuple[str, Path, np.ndarray, int, int]] = [
            ("full", frame.image, frame.boxes, frame.width, frame.height),
        ]
        if engine_on:
            scaled = frame.boxes.copy()
            scaled[:, 0::2] *= ENGINE_WIDTH / frame.width
            scaled[:, 1::2] *= ENGINE_HEIGHT / frame.height
            full_forms.append((
                "engine",
                engine_view(frame.image, engine_dir / f"{frame.key}.jpg", quality).resolve(),
                scaled, ENGINE_WIDTH, ENGINE_HEIGHT,
            ))

        for form, image, boxes, width, height in full_forms:
            for repeat in range(full_repeat):
                suffix = "" if repeat == 0 else f"-rep{repeat + 1}"
                emit(f"{frame.key}__{form}{suffix}", image, boxes, width, height, "full")

        if not tiles_on:
            continue

        # --- the same 3x2 native tiles the teacher and the two-pass use ------
        source: Any = None
        planned = plan_tiles(frame.width, frame.height, cols=cols, rows=rows, overlap=overlap)
        for tile in tiles_kept(planned, tile_keep, index):
            boxes, dropped = clip_to_tile(frame.boxes, tile)
            stats["slivers_dropped"] += dropped
            name = f"{frame.key}__r{tile.row}c{tile.col}"
            crop_path = tile_dir / f"{name}.jpg"
            if not crop_path.exists():
                if source is None:
                    with Image.open(frame.image) as handle:
                        source = handle.convert("RGB")
                source.crop((tile.x0, tile.y0, tile.x1, tile.y1)).save(
                    crop_path, "JPEG", quality=quality, subsampling=0
                )
            emit(name, crop_path.resolve(), boxes, tile.x1 - tile.x0, tile.y1 - tile.y0, "tile")
            stats["empty_tiles"] += int(len(boxes) == 0)

    return {"totals": dict(stats), "by_split": {k: dict(v) for k, v in per_split.items()}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--harmonized", type=Path, default=HARMONIZED)
    parser.add_argument("--harvest", type=Path, default=HARVEST)
    parser.add_argument("--teacher-labels", type=Path, default=TEACHER_LABELS)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--harvest-splits", nargs="*", default=["train"])
    parser.add_argument("--teacher-threshold", type=float, default=TEACHER_TRAIN_THRESHOLD)
    parser.add_argument("--cols", type=int, default=TILE_COLS)
    parser.add_argument("--rows", type=int, default=TILE_ROWS)
    parser.add_argument("--overlap", type=float, default=TILE_OVERLAP)
    parser.add_argument("--no-tiles", action="store_true", help="full frames only (an ablation)")
    parser.add_argument("--engine-view", action="store_true",
                        help=f"also emit each frame at {ENGINE_WIDTH}x{ENGINE_HEIGHT}, the way "
                             f"the decoder hands it to the detector today")
    parser.add_argument("--full-repeat", type=int, default=1,
                        help="how many augmented copies of each full-frame view an epoch sees "
                             "(the parity knob)")
    parser.add_argument("--tile-keep", type=int, default=0,
                        help="keep only this many of the cols x rows tiles per frame, rotating "
                             "which ones (0 = all of them)")
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR,
                        help="where rendered tiles and engine views live, shared between datasets")
    parser.add_argument("--exclude", type=Path, default=None,
                        help="a scripts/leakage_check.py --write-exclusions report: those "
                             "harvested frames are left out of training")
    parser.add_argument("--quality", type=int, default=TILE_JPEG_QUALITY)
    parser.add_argument("--limit", type=int, default=0, help="stop after N source frames per corpus")
    args = parser.parse_args()

    if args.full_repeat < 1:
        raise SystemExit("--full-repeat must be at least 1")

    harmonized_train = read_harmonized(args.harmonized, "train")
    harmonized_val = read_harmonized(args.harmonized, "val")
    harvest_train = read_harvest(args.harvest, args.teacher_labels, args.harvest_splits,
                                 args.teacher_threshold)

    excluded: list[str] = []
    if args.exclude is not None:
        report = json.loads(args.exclude.read_text())
        excluded = sorted(report["excluded"])
        wanted = {args.harvest.resolve() / name for name in excluded}
        before = len(harvest_train)
        harvest_train = [f for f in harvest_train if f.image not in wanted]
        removed = before - len(harvest_train)
        if removed != len(excluded):
            raise SystemExit(
                f"{args.exclude} names {len(excluded)} frames to exclude but only {removed} "
                f"of them are in the harvest split(s) being read — the report and the corpus "
                f"disagree, and building anyway would ship a leak."
            )
        print(f"leak fix: {removed} harvested frames dropped "
              f"(±{report.get('window_sec')} s of a val frame) — {args.exclude}")

    if args.limit:
        harmonized_train = harmonized_train[:args.limit]
        harmonized_val = harmonized_val[:args.limit]
        harvest_train = harvest_train[:args.limit]

    frames = [*harmonized_train, *harvest_train, *harmonized_val]
    keys = Counter(f.key for f in frames)
    clashes = [k for k, n in keys.items() if n > 1]
    if clashes:
        raise SystemExit(f"view names would collide: {clashes[:5]}")

    print(f"harmonised train {len(harmonized_train):>3} frames, "
          f"{sum(len(f.boxes) for f in harmonized_train):>5} boxes")
    print(f"harvest    train {len(harvest_train):>3} frames, "
          f"{sum(len(f.boxes) for f in harvest_train):>5} boxes  "
          f"(teacher @ >= {args.teacher_threshold:g})")
    print(f"harmonised val   {len(harmonized_val):>3} frames, "
          f"{sum(len(f.boxes) for f in harmonized_val):>5} boxes")

    tile_keep = args.tile_keep or args.cols * args.rows
    result = build(frames, args.out_dir, cols=args.cols, rows=args.rows, overlap=args.overlap,
                   tiles_on=not args.no_tiles, quality=args.quality, cache_dir=args.cache_dir,
                   engine_on=args.engine_view, full_repeat=args.full_repeat, tile_keep=tile_keep)

    totals = result["totals"]
    full_views, tile_views = totals.get("full_views", 0), totals.get("tile_views", 0)
    print(f"\n{totals['views']} views "
          f"({full_views} full + {tile_views} tiles), "
          f"{totals['boxes']} boxes")
    print(f"  full-frame share {full_views / max(totals['views'], 1):.0%} "
          f"({'native only' if not args.engine_view else 'native + engine 960x544'}, "
          f"x{args.full_repeat}; "
          f"{'no tiles' if args.no_tiles else f'{tile_keep} of {args.cols * args.rows} tiles per frame'})")
    print(f"  train {result['by_split']['train']['views']:>4} views / "
          f"{result['by_split']['train']['boxes']:>5} boxes")
    print(f"  val   {result['by_split']['val']['views']:>4} views / "
          f"{result['by_split']['val']['boxes']:>5} boxes")
    print(f"  {totals.get('slivers_dropped', 0)} truncated boxes dropped "
          f"(< {MIN_KEPT_AREA:.0%} of the box left in the tile)")
    print(f"  {totals.get('empty_tiles', 0)} tiles with no people — negative examples")

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "scripts/build_student_dataset.py",
        "views": {
            "full_frame_native": True,
            "full_frame_engine": f"{ENGINE_WIDTH}x{ENGINE_HEIGHT}" if args.engine_view else None,
            "full_repeat": args.full_repeat,
            "tiles": (f"{tile_keep} of {args.cols * args.rows} "
                      f"({args.cols}x{args.rows} @ {args.overlap:g} overlap, rotating)"
                      if not args.no_tiles else None),
            "full_frame_share": round(full_views / max(totals["views"], 1), 4),
        },
        "leak_fix": {
            "report": str(args.exclude) if args.exclude else None,
            "harvest_frames_excluded": len(excluded),
            "excluded": excluded,
        },
        "sources": {
            "harmonized_train": {"frames": len(harmonized_train),
                                 "boxes": int(sum(len(f.boxes) for f in harmonized_train)),
                                 "root": str(args.harmonized.relative_to(ROOT))},
            "harvest_train": {"frames": len(harvest_train),
                              "boxes": int(sum(len(f.boxes) for f in harvest_train)),
                              "root": str(args.harvest.relative_to(ROOT)),
                              "splits": sorted(set(args.harvest_splits)),
                              "teacher_threshold": args.teacher_threshold},
            "harmonized_val": {"frames": len(harmonized_val),
                               "boxes": int(sum(len(f.boxes) for f in harmonized_val)),
                               "root": str(args.harmonized.relative_to(ROOT))},
        },
        "min_kept_area": MIN_KEPT_AREA,
        "min_side_px": MIN_SIDE_PX,
        **result,
        "never_used": ["harvest guard", "harvest heldout"],
        "warning": (
            "The harvest half of `train` is PSEUDO-labelled by a stock-COCO RF-DETR Large "
            "teacher, and the harmonised half takes its box EXTENTS from the same teacher. "
            "Presence in the harmonised half is human. Nothing here is a test set: the val "
            "split is the 16 hand-labelled frames, which early stopping reads, and the "
            "harvest heldout frames share a 60-second clip with the training frames "
            "(docs/TEACHER_LABELS.md §2)."
        ),
    }
    summary_path = _assert_not_ground_truth(args.out_dir / "summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
