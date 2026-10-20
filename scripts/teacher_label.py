#!/usr/bin/env python
"""Pseudo-label the harvest with RF-DETR Large on native tiles.

`docs/DETECTION_V2.md` §5 Phase 1: pre-label with a teacher that sees
*privileged pixels* — pixels the shipping engine never gets. The engine squashes
2560x1440 to 960x544 and then to 512x512 (§2.5); this teacher is the biggest
Apache-2.0 model in the family (RF-DETR Large, 704 px input, stock COCO weights)
and it is shown the frame twice: once whole, and once as six native-resolution
tiles. A far-end worker who reaches the engine at 4x19 px reaches a tile at
about 13x62 px.

**Only Nano / Small / Medium / Large.** RF-DETR XL and 2XL are under Roboflow's
PML, not Apache-2.0 (§6), so they are not downloadable by anything here.

The merge is the part worth reading. Tiling a detector normally buys recall and
pays for it in seam duplicates: one person straddling two tiles becomes two half
boxes, and §2.2's hard-won "zero duplicates" property dies. The rule here is
spatial, not score-based, and is the same one §5 Phase 2 specifies for the
student's ROI pass:

1. A tile owns its interior. Any tile box with an edge within ``border_margin``
   px of a tile border that is **not** an image border is dropped outright — it
   is a person cut in half, and either the neighbouring tile (25% overlap: a
   256 px horizontal, 206 px vertical margin) or the full-frame pass sees that
   same person whole.
2. What survives, plus the full-frame boxes, goes through one class-agnostic NMS
   at IoU 0.55.

Rule 1 is what makes rule 2 safe: without it NMS has to choose between a half
box and a whole box that barely overlap, and it keeps both.

Detections are cached *before* the merge, with each box's score and which pass
produced it, so a threshold sweep is one inference run and every operating point
is the real pipeline rather than a post-hoc filter (see `scripts/teacher_eval.py`).

    .venv/bin/python scripts/teacher_label.py --threshold 0.4
    .venv/bin/python scripts/teacher_label.py --splits train --limit 5 --dry-run

This writes to ``footage/anmol-harvest/labels_teacher/`` and nowhere else. It
never touches the hand-labelled dataset under
``footage/anmol-samples/cctv_person_tracking_dataset/`` — that is the only
ground truth this project has, and a pseudo-label leaking into it would be
undetectable and unrecoverable.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HARVEST_DIR = ROOT / "footage" / "anmol-harvest"
LABELS_DIR = HARVEST_DIR / "labels_teacher"

# The hand-labelled corpus. Named here only so `_assert_not_ground_truth` can
# refuse to write anywhere inside it.
GROUND_TRUTH_DIR = ROOT / "footage" / "anmol-samples" / "cctv_person_tracking_dataset"

# Teacher defaults.
VARIANT = "large"
TILE_COLS, TILE_ROWS = 3, 2
TILE_OVERLAP = 0.25
BORDER_MARGIN_PX = 6.0
MERGE_NMS_IOU = 0.55

# Every box above this is cached, so one inference run feeds the whole sweep.
# Well under any operating point anyone would ship.
SCORE_FLOOR = 0.05

# Refuse to touch the GPU below this (GiB of MemAvailable). The GB10's memory is
# unified and shared with whatever else is on the box: §2.1 is the story of a
# tenant holding 95 of 121 GiB and every model on the appliance going dark.
MIN_AVAILABLE_GIB = 10.0


# --------------------------------------------------------------------------
# house rules
# --------------------------------------------------------------------------


def mem_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1024 * 1024)
    raise RuntimeError("MemAvailable missing from /proc/meminfo")


def require_memory(minimum_gib: float = MIN_AVAILABLE_GIB) -> float:
    available = mem_available_gib()
    if available < minimum_gib:
        raise SystemExit(
            f"MemAvailable is {available:.1f} GiB, below the {minimum_gib:.0f} GiB floor. "
            f"Not starting GPU work on a shared box (docs/DETECTION_V2.md §2.1)."
        )
    return available


def _assert_not_ground_truth(path: Path) -> Path:
    """The one write this script must never perform."""
    resolved = path.resolve()
    if resolved == GROUND_TRUTH_DIR.resolve() or GROUND_TRUTH_DIR.resolve() in resolved.parents:
        raise SystemExit(f"refusing to write inside the hand-labelled dataset: {resolved}")
    return path


# --------------------------------------------------------------------------
# tiling
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Tile:
    """One native-resolution crop, and which of its borders are seams."""

    row: int
    col: int
    x0: int
    y0: int
    x1: int
    y1: int
    interior_left: bool
    interior_top: bool
    interior_right: bool
    interior_bottom: bool

    @property
    def name(self) -> str:
        return f"tile:{self.row},{self.col}"


def plan_tiles(
    width: int, height: int, *, cols: int = TILE_COLS, rows: int = TILE_ROWS, overlap: float = TILE_OVERLAP
) -> list[Tile]:
    """A cols x rows grid of equal tiles with ``overlap`` fractional overlap.

    Every tile is exactly the same size and the outermost ones are flush with
    the image edge, so the grid covers the frame with no remainder strip — a
    remainder strip is a band of the image that only ever reaches the network at
    full-frame scale, which is the blindness this whole exercise is about.
    """
    if cols < 1 or rows < 1:
        raise ValueError("need at least one tile in each direction")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")

    def spans(extent: int, count: int) -> list[tuple[int, int]]:
        if count == 1:
            return [(0, extent)]
        size = min(extent, int(np.ceil(extent / (1 + (count - 1) * (1 - overlap)))))
        return [(round(i * (extent - size) / (count - 1)), round(i * (extent - size) / (count - 1)) + size)
                for i in range(count)]

    xs, ys = spans(width, cols), spans(height, rows)
    tiles: list[Tile] = []
    for row, (y0, y1) in enumerate(ys):
        for col, (x0, x1) in enumerate(xs):
            tiles.append(Tile(
                row=row, col=col, x0=x0, y0=y0, x1=x1, y1=y1,
                interior_left=x0 > 0, interior_top=y0 > 0,
                interior_right=x1 < width, interior_bottom=y1 < height,
            ))
    return tiles


# --------------------------------------------------------------------------
# the teacher
# --------------------------------------------------------------------------


@dataclass(slots=True)
class RawDetections:
    """Everything both passes saw, before any merging.

    Caching at this stage rather than after the merge is what lets the threshold
    sweep in `scripts/teacher_eval.py` re-run the *real* merge at every operating
    point instead of filtering an already-merged set, which would quietly report
    a different pipeline from the one that writes the labels.
    """

    width: int
    height: int
    xyxy: np.ndarray                 # (N, 4) float32, image pixels
    scores: np.ndarray               # (N,) float32
    source: list[str]                # per box: "full" or "tile:r,c"
    dropped_on_seam: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "width": self.width, "height": self.height,
            "dropped_on_seam": self.dropped_on_seam,
            "boxes": [
                {"xyxy": [round(float(v), 2) for v in box], "score": round(float(score), 4), "source": src}
                for box, score, src in zip(self.xyxy, self.scores, self.source)
            ],
        }


class Teacher:
    """RF-DETR over one full frame plus a grid of native tiles.

    Thin on purpose: `argus.vision.detector.PersonDetector` already owns the
    RF-DETR plumbing (FP16, the person-logit shortcut, boxes back in source
    pixels), and duplicating it here would be a second thing to keep honest. It
    is configured with ``nms_iou=1.0`` so no NMS happens inside a pass — the
    merge below is the only place boxes are suppressed, and it needs to see the
    raw output of each pass to do that.

    ``detector`` injects an already-built ``PersonDetector`` instead of the
    stock-COCO Large this class defaults to. That is how
    `scripts/student_eval.py` runs the *student* and the *production Nano* down
    this exact two-pass path: the merge, the seam rule and the tile geometry are
    the ones that wrote the labels, not a second implementation that has drifted
    from them. A caller passing its own detector owns its thresholds and should
    pass ``nms_iou=1.0`` for the same reason this class does.
    """

    def __init__(
        self,
        *,
        variant: str = VARIANT,
        cols: int = TILE_COLS,
        rows: int = TILE_ROWS,
        overlap: float = TILE_OVERLAP,
        border_margin: float = BORDER_MARGIN_PX,
        nms_iou: float = MERGE_NMS_IOU,
        score_floor: float = SCORE_FLOOR,
        batch: int = 2,
        full_frame: bool = True,
        precision: str = "fp16",
        detector: Any = None,
    ):
        from argus.vision.detector import PersonDetector

        self.variant = variant
        self.cols, self.rows, self.overlap = cols, rows, overlap
        self.border_margin = float(border_margin)
        self.nms_iou = float(nms_iou)
        self.score_floor = float(score_floor)
        self.batch = max(1, int(batch))
        self.full_frame = bool(full_frame)
        # threshold == score_floor and track_floor == 0 makes `emit_floor` the
        # floor, exactly as the eval harness does it.
        self._detector = detector if detector is not None else PersonDetector(
            variant=variant, precision=precision, threshold=score_floor,
            nms_iou=1.0, trace=False,
        )
        if detector is not None:
            self.variant = getattr(detector, "variant", variant)
            self.score_floor = float(getattr(detector, "emit_floor", score_floor))
        self._tiles: dict[tuple[int, int], list[Tile]] = {}

    def describe(self) -> dict[str, Any]:
        return {
            "model": f"rf-detr-{self.variant}",
            "weights": "stock COCO (Apache-2.0)",
            "resolution": self._detector.resolution,
            "precision": self._detector.precision,
            "passes": (["full-frame"] if self.full_frame else []) + [f"{self.cols}x{self.rows} native tiles"],
            "tile_overlap": self.overlap,
            "border_margin_px": self.border_margin,
            "merge_nms_iou": self.nms_iou,
            "score_floor": self.score_floor,
            "person_class_index": self._detector.person_class_index,
        }

    def load(self) -> "Teacher":
        self._detector.load()
        return self

    def tiles_for(self, width: int, height: int) -> list[Tile]:
        key = (width, height)
        if key not in self._tiles:
            self._tiles[key] = plan_tiles(width, height, cols=self.cols, rows=self.rows, overlap=self.overlap)
        return self._tiles[key]

    # --- the two passes ---------------------------------------------------

    def raw(self, image: np.ndarray) -> RawDetections:
        """Both passes over one BGR frame, seam boxes already dropped."""
        height, width = image.shape[:2]
        boxes: list[np.ndarray] = []
        scores: list[np.ndarray] = []
        source: list[str] = []
        dropped = 0

        if self.full_frame:
            detections = self._detector.detect([image])[0]
            boxes.append(np.asarray(detections.xyxy, dtype=np.float32).reshape(-1, 4))
            scores.append(np.asarray(detections.confidence, dtype=np.float32).reshape(-1))
            source.extend(["full"] * len(detections.xyxy))

        tiles = self.tiles_for(width, height)
        for chunk in _chunks(tiles, self.batch):
            crops = [np.ascontiguousarray(image[t.y0:t.y1, t.x0:t.x1]) for t in chunk]
            for tile, detections in zip(chunk, self._detector.detect(crops)):
                tile_boxes = np.asarray(detections.xyxy, dtype=np.float32).reshape(-1, 4)
                tile_scores = np.asarray(detections.confidence, dtype=np.float32).reshape(-1)
                keep = self._interior_only(tile_boxes, tile)
                dropped += int((~keep).sum())
                tile_boxes = tile_boxes[keep]
                tile_boxes[:, 0::2] += tile.x0
                tile_boxes[:, 1::2] += tile.y0
                boxes.append(tile_boxes)
                scores.append(tile_scores[keep])
                source.extend([tile.name] * len(tile_boxes))

        return RawDetections(
            width=width, height=height,
            xyxy=np.concatenate(boxes).reshape(-1, 4) if boxes else np.zeros((0, 4), np.float32),
            scores=np.concatenate(scores).reshape(-1) if scores else np.zeros((0,), np.float32),
            source=source, dropped_on_seam=dropped,
        )

    def _interior_only(self, boxes: np.ndarray, tile: Tile) -> np.ndarray:
        """True for boxes that touch no seam — the tile's own people.

        A box whose edge sits on a seam is a person the tile can only see part
        of, so its extent is wrong by an unknown amount. Dropping it is not a
        loss of recall: with 25% overlap that person is at least 206 px inside
        the neighbouring tile, and the full-frame pass is looking at them too.
        Keeping it *is* a loss of precision, and worse, of the property §2.2
        measured and §5 Phase 2 wants preserved — no two boxes on one person.
        """
        if len(boxes) == 0:
            return np.zeros(0, dtype=bool)
        margin = self.border_margin
        width, height = tile.x1 - tile.x0, tile.y1 - tile.y0
        touching = np.zeros(len(boxes), dtype=bool)
        if tile.interior_left:
            touching |= boxes[:, 0] <= margin
        if tile.interior_top:
            touching |= boxes[:, 1] <= margin
        if tile.interior_right:
            touching |= boxes[:, 2] >= width - margin
        if tile.interior_bottom:
            touching |= boxes[:, 3] >= height - margin
        return ~touching


def _chunks(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


# --------------------------------------------------------------------------
# merge
# --------------------------------------------------------------------------


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Class-agnostic NMS, returning kept indices in descending score order.

    Numpy rather than torchvision so the merge can be replayed on CPU, off the
    shared GPU, at as many thresholds as a sweep wants.
    """
    if len(boxes) == 0:
        return np.zeros(0, dtype=int)
    from argus.eval.detector import iou_matrix

    order = np.argsort(-scores)
    overlaps = iou_matrix(boxes[order], boxes[order])
    keep: list[int] = []
    suppressed = np.zeros(len(order), dtype=bool)
    for position in range(len(order)):
        if suppressed[position]:
            continue
        keep.append(int(order[position]))
        suppressed |= overlaps[position] >= iou_threshold
        suppressed[:position + 1] = True
    return np.asarray(keep, dtype=int)


def merge(raw: RawDetections, *, threshold: float, nms_iou: float = MERGE_NMS_IOU) -> tuple[np.ndarray, np.ndarray]:
    """Cached raw detections -> the teacher's boxes at one operating point.

    Threshold first, then NMS, in that order: it is the order the live pipeline
    would run in, and it is not interchangeable — suppressing with boxes that
    will then be thrown away for being too weak is a different answer.
    """
    keep = raw.scores >= threshold
    boxes, scores = raw.xyxy[keep], raw.scores[keep]
    order = nms(boxes, scores, nms_iou)
    return boxes[order], scores[order]


# --------------------------------------------------------------------------
# writing labels
# --------------------------------------------------------------------------


def write_yolo(path: Path, boxes: np.ndarray, width: int, height: int) -> None:
    _assert_not_ground_truth(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for x0, y0, x1, y1 in boxes:
        cx, cy = (x0 + x1) / 2 / width, (y0 + y1) / 2 / height
        bw, bh = (x1 - x0) / width, (y1 - y0) / height
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def write_sidecar(
    path: Path, *, frame: str, raw: RawDetections, boxes: np.ndarray, scores: np.ndarray,
    threshold: float, teacher: dict[str, Any],
) -> None:
    _assert_not_ground_truth(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "frame": frame,
        "width": raw.width, "height": raw.height,
        "threshold": threshold,
        "teacher": teacher,
        "dropped_on_seam": raw.dropped_on_seam,
        "n_raw": int(len(raw.xyxy)),
        "boxes": [
            {"xyxy": [round(float(v), 2) for v in box], "score": round(float(score), 4)}
            for box, score in zip(boxes, scores)
        ],
    }
    path.write_text(json.dumps(payload, indent=1) + "\n")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--harvest-dir", type=Path, default=HARVEST_DIR)
    parser.add_argument("--out-dir", type=Path, default=LABELS_DIR)
    parser.add_argument("--splits", nargs="*", default=["train", "heldout"],
                        help="which manifest splits to label (the guard band is never labelled)")
    parser.add_argument("--threshold", type=float, required=True,
                        help="score threshold; choose it from scripts/teacher_eval.py, not by guessing")
    parser.add_argument("--variant", default=VARIANT, choices=["nano", "small", "medium", "large"],
                        help="XL/2XL are Roboflow PML, not Apache-2.0, and are deliberately absent")
    parser.add_argument("--cols", type=int, default=TILE_COLS)
    parser.add_argument("--rows", type=int, default=TILE_ROWS)
    parser.add_argument("--overlap", type=float, default=TILE_OVERLAP)
    parser.add_argument("--border-margin", type=float, default=BORDER_MARGIN_PX)
    parser.add_argument("--nms-iou", type=float, default=MERGE_NMS_IOU)
    parser.add_argument("--batch", type=int, default=2, help="tiles per forward pass; keep small, the box is shared")
    parser.add_argument("--limit", type=int, default=0, help="stop after N frames (a smoke test)")
    parser.add_argument("--dry-run", action="store_true", help="run the teacher, write nothing")
    args = parser.parse_args()

    if "guard" in args.splits:
        raise SystemExit("the guard band exists to be unused; refusing to label it")

    manifest_path = args.harvest_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"no manifest at {manifest_path} — run scripts/harvest_frames.py first")
    manifest = json.loads(manifest_path.read_text())
    frames = [f for f in manifest["frames"] if f["split"] in args.splits]
    if args.limit:
        frames = frames[:args.limit]
    if not frames:
        raise SystemExit(f"no frames in splits {args.splits}")

    available = require_memory()
    print(f"MemAvailable {available:.1f} GiB — starting", flush=True)

    from argus.eval.detector import _load_bgr

    teacher = Teacher(
        variant=args.variant, cols=args.cols, rows=args.rows, overlap=args.overlap,
        border_margin=args.border_margin, nms_iou=args.nms_iou, batch=args.batch,
    ).load()
    description = teacher.describe()
    print(json.dumps(description), flush=True)

    import torch

    per_scene: dict[str, dict[str, int]] = {}
    total_boxes = total_seam = 0
    for index, frame in enumerate(frames, start=1):
        image_path = args.harvest_dir / str(frame["file"])
        raw = teacher.raw(_load_bgr(image_path))
        boxes, scores = merge(raw, threshold=args.threshold, nms_iou=args.nms_iou)
        scene, stem = str(frame["file"]).split("/", 1)
        stem = Path(stem).stem
        if not args.dry_run:
            write_yolo(args.out_dir / scene / f"{stem}.txt", boxes, raw.width, raw.height)
            write_sidecar(
                args.out_dir / scene / f"{stem}.json", frame=str(frame["file"]), raw=raw,
                boxes=boxes, scores=scores, threshold=args.threshold, teacher=description,
            )
        bucket = per_scene.setdefault(scene, {"frames": 0, "boxes": 0, "seam_dropped": 0,
                                              "train": 0, "heldout": 0})
        bucket["frames"] += 1
        bucket["boxes"] += len(boxes)
        bucket["seam_dropped"] += raw.dropped_on_seam
        bucket[str(frame["split"])] += 1
        total_boxes += len(boxes)
        total_seam += raw.dropped_on_seam
        if index % 25 == 0 or index == len(frames):
            peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
            print(f"  {index}/{len(frames)} frames · {total_boxes} boxes · peak GPU {peak:.2f} GiB", flush=True)

    peak_gib = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "scripts/teacher_label.py",
        "teacher": description,
        "threshold": args.threshold,
        "splits": args.splits,
        "frames": len(frames),
        "boxes": total_boxes,
        "seam_boxes_dropped": total_seam,
        "peak_gpu_gib": round(peak_gib, 2),
        "by_scene": per_scene,
        "provenance_warning": (
            "These are PSEUDO-labels from a stock-COCO teacher, not ground truth. "
            "docs/TEACHER_LABELS.md records how far they can be trusted and which "
            "height band needs human verification. Per docs/DETECTION_V2.md §5 they "
            "may pre-label training data only -- never a test set."
        ),
    }
    if not args.dry_run:
        summary_path = _assert_not_ground_truth(args.out_dir / "summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"\nwrote {total_boxes} boxes over {len(frames)} frames -> {args.out_dir}")
    print(f"peak GPU {peak_gib:.2f} GiB · {total_seam} seam boxes dropped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
