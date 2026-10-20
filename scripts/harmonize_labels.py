#!/usr/bin/env python
"""One box convention for the hand labels: the whole visible person.

`docs/TEACHER_LABELS.md` §7 calls this "the blocking item, and it is worth more
than any model change on this list", and `docs/DETECTION_V2.md` §2.5's
2026-09-20 correction records the decision: **one convention everywhere — the
whole VISIBLE person (COCO)**. Right now the 79 hand-labelled frames mix two.
Boxes under 60 px tall have a median height/width of 1.00 and 88% are squarish;
44% of the 60-150 px band is squarish too. A square box is a head. Near workers
were boxed whole, distant workers were boxed at the head, in the same frame, on
people of the same size, with no rule.

That is not only a scoring problem. `argus/vision/zones.py: anchors` places a
person in a zone by the **bottom centre** of their box — their feet — so a
head-only box puts a distant worker's "feet" at their chin, a body length from
where they stand. Far-end zone counts are wrong for exactly the people the
fine-tune was credited with finding.

**Humans supply presence, the teacher supplies extent.** Nobody re-draws 1015
boxes by hand. A human box is proof that a person is there — the one thing the
teacher cannot be trusted to supply (it misses ~4% outright and an unknown share
of its extra boxes are real people nobody labelled). The teacher's box, when it
is demonstrably on the same person, is proof of how far down that person goes —
the one thing the hand labels are demonstrably inconsistent about. Each side
contributes what it is good for and neither is trusted for the other.

Five categories come out, and the provenance sidecar names every box's:

* ``extent_from_teacher`` — a head-like human box (squarish, and much shorter
  than its teacher box) inside a teacher box that plausibly is the same person.
  The teacher's extent is adopted. This is the repair.
* ``human_kept_agreed`` — human and teacher already agree (IoU >= 0.5). The
  human box is kept: where the two conventions coincide the human is the better
  localiser, and adopting the teacher there would import its errors for nothing.
* ``human_kept_disputed`` — matched, but neither head-like nor in agreement.
  Kept as drawn and flagged. These need eyes, not an automatic rule.
* ``extent_unverified`` — a human box no teacher box plausibly covers. Kept.
  The person is real (a human drew them); the extent is unconfirmed.
* ``teacher_only`` — a teacher box at score >= 0.6 that claims no human box.
  Added. These are either people nobody labelled or false positives, and §3 of
  TEACHER_LABELS.md says nobody has ever counted which; this script reports the
  number and renders them so somebody can.

**Many-to-one is refused.** One teacher box must not absorb two humans' labels.
In the tight clusters of partly occluded workers behind the packing machines the
teacher stacks overlapping boxes, and a rule that let one of them claim two
human boxes would collapse two workers into one and silently delete a person
from the corpus. Matching is strictly one-to-one, resolved best-pair-first; a
human who loses a contested teacher box keeps their own box as
``extent_unverified`` rather than borrowing somebody else's body.

    .venv/bin/python scripts/harmonize_labels.py
    .venv/bin/python scripts/harmonize_labels.py --overlays 8

Reads the teacher detections `scripts/teacher_eval.py` already cached, so this
touches no GPU. Writes `footage/anmol-harmonized/` and
`docs/screens-v3/harmonized/` and refuses to write anywhere inside the
hand-labelled dataset — that is the only ground truth this project has.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.teacher_label import (  # noqa: E402
    MERGE_NMS_IOU, _assert_not_ground_truth, merge,
)
from scripts.teacher_eval import cache_path, load_cache  # noqa: E402

DATASET = ROOT / "footage" / "anmol-samples" / "cctv_person_tracking_dataset"
OUT_DIR = ROOT / "footage" / "anmol-harmonized"
OVERLAY_DIR = ROOT / "docs" / "screens-v3" / "harmonized"

# The teacher's shipped operating point (docs/TEACHER_LABELS.md §5). Extent may
# be taken from any box at or above it.
TEACHER_THRESHOLD = 0.30
# A teacher box nobody labelled is only *added* well above that. At 0.30 the
# teacher's centre-in-box precision is 0.80; at 0.60 it is 0.98. A wrong box
# added to the training set is a person taught to exist, so the bar is the
# high-precision end, not the high-recall end.
TEACHER_ONLY_THRESHOLD = 0.60

# --- what "plausibly the same person" means -------------------------------
#
# The human box's centre inside the teacher box is the necessary condition; a
# head drawn on a body always satisfies it. On its own it is too weak in a
# crowd, so one of two sufficient conditions must hold as well.
#
# `UPPER_PART_FRACTION`: the human box lies entirely within the top 60% of the
# teacher box. A head is in the top third of a standing person and the top half
# of a seated or bench-height one; 60% admits head-and-shoulders on a partly
# occluded worker without admitting a box on somebody's knees.
UPPER_PART_FRACTION = 0.60
# `CONTAINED_FRACTION`: this much of the human box's area lies inside the
# teacher box. "Overlaps it substantially" for a box that is not in the upper
# part — a whole-body label the teacher drew slightly differently.
CONTAINED_FRACTION = 0.60

# --- what "already agree" and "head-like" mean ----------------------------
AGREE_IOU = 0.50
# docs/TEACHER_LABELS.md §3.2 measures the hand labels' aspect: a standing
# person is ~2.5:1 and 88% of the far band is under 1.3:1. 1.3 is that paper's
# own cut, so the same boxes are called squarish here as there.
SQUARISH_ASPECT = 1.30
# "Much shorter": the human box is at most this fraction of the teacher box's
# height. §3.2 measures the median teacher/label height ratio at 1.62 far and
# 2.08 on the val frames, so 0.65 (a ratio of 1.54) sits just inside the
# population this is meant to catch and well clear of the 1.003 near band.
SHORT_FRACTION = 0.65
# A head-AND-SHOULDERS label is not square (a 64x41 box is 1.56:1) but is still
# plainly not a body. Admitted only with both a stricter height ratio and the
# upper-part test, which together are hard to satisfy by accident.
SHOULDERS_ASPECT = 2.00
SHOULDERS_SHORT_FRACTION = 0.50

# A teacher-only box is dropped if it lands on a person the harmonised set
# already has. NMS at 0.55 inside the teacher's own merge does not prevent this:
# a stacked box on a partly occluded worker can sit at IoU 0.4 with the box that
# claimed the human label and still be the same person.
TEACHER_ONLY_MAX_IOU = 0.40

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

CATEGORIES = (
    "extent_from_teacher",
    "human_kept_agreed",
    "human_kept_disputed",
    "extent_unverified",
    "teacher_only",
)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def _areas(boxes: np.ndarray) -> np.ndarray:
    return np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)


def intersection_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(A, B) pairwise intersection area."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    inter_w = np.clip(
        np.minimum(a[:, 2][:, None], b[:, 2][None, :]) - np.maximum(a[:, 0][:, None], b[:, 0][None, :]), 0, None
    )
    inter_h = np.clip(
        np.minimum(a[:, 3][:, None], b[:, 3][None, :]) - np.maximum(a[:, 1][:, None], b[:, 1][None, :]), 0, None
    )
    return (inter_w * inter_h).astype(np.float32)


@dataclass(slots=True)
class Pairing:
    """Why one human box and one teacher box are believed to be one person."""

    human: int
    teacher: int
    iou: float
    contained: float          # share of the human box inside the teacher box
    upper: bool               # the human box sits in the teacher box's upper part
    height_ratio: float       # human height / teacher height
    aspect: float             # human height / width

    @property
    def quality(self) -> float:
        """Rank for one-to-one resolution: containment first, IoU as tie-break.

        Containment rather than IoU because the whole point is pairs whose IoU
        is *low* — a head inside a body. Two humans contesting one teacher box
        are separated by which is more completely inside it, and IoU only
        breaks ties among equally-contained candidates.
        """
        return self.contained + 0.25 * self.iou

    @property
    def head_like(self) -> bool:
        squarish = self.aspect < SQUARISH_ASPECT and self.height_ratio <= SHORT_FRACTION
        shoulders = (
            self.aspect < SHOULDERS_ASPECT
            and self.height_ratio <= SHOULDERS_SHORT_FRACTION
            and self.upper
        )
        return squarish or shoulders


def plausible_pairs(human: np.ndarray, teacher: np.ndarray) -> list[Pairing]:
    """Every (human, teacher) pair that could be the same person.

    The necessary condition is the human box's centre inside the teacher box.
    The sufficient condition is either position (the human box sits in the
    teacher box's upper part — a head on a body) or overlap (most of the human
    box's area is inside it — the same person, drawn differently).
    """
    from argus.eval.detector import iou_matrix

    if len(human) == 0 or len(teacher) == 0:
        return []
    inter = intersection_matrix(human, teacher)
    ious = iou_matrix(human, teacher)
    human_area = np.maximum(_areas(human), 1e-6)
    contained = inter / human_area[:, None]

    cx = ((human[:, 0] + human[:, 2]) / 2)[:, None]
    cy = ((human[:, 1] + human[:, 3]) / 2)[:, None]
    centre_in = (
        (cx >= teacher[:, 0][None, :]) & (cx <= teacher[:, 2][None, :])
        & (cy >= teacher[:, 1][None, :]) & (cy <= teacher[:, 3][None, :])
    )

    teacher_height = np.maximum(teacher[:, 3] - teacher[:, 1], 1e-6)
    human_height = np.maximum(human[:, 3] - human[:, 1], 1e-6)
    human_width = np.maximum(human[:, 2] - human[:, 0], 1e-6)
    # The human box's lower edge, as a fraction of the way down the teacher box.
    depth = (human[:, 3][:, None] - teacher[:, 1][None, :]) / teacher_height[None, :]
    upper = depth <= UPPER_PART_FRACTION

    pairs: list[Pairing] = []
    for h in range(len(human)):
        for t in range(len(teacher)):
            if not centre_in[h, t]:
                continue
            if not (upper[h, t] or contained[h, t] >= CONTAINED_FRACTION):
                continue
            pairs.append(Pairing(
                human=h, teacher=t,
                iou=float(ious[h, t]), contained=float(contained[h, t]),
                upper=bool(upper[h, t]),
                height_ratio=float(human_height[h] / teacher_height[t]),
                aspect=float(human_height[h] / human_width[h]),
            ))
    return pairs


def assign_one_to_one(pairs: Sequence[Pairing]) -> tuple[dict[int, Pairing], list[Pairing]]:
    """Strictly one human to one teacher box, agreement first, then quality.

    Two phases because the two kinds of pair are not comparable on one scale.
    An IoU >= 0.5 pair is an identification: those boxes are the same person and
    nothing should be allowed to take that teacher box away. Everything else is
    an inference from position, and among inferences the most-contained wins.

    Returns the accepted pairing per human index, and the pairs that lost a
    contest — the many-to-one conflicts, which are counted and reported because
    they are where the packing-line clusters live.
    """
    accepted: dict[int, Pairing] = {}
    taken_teacher: set[int] = set()
    displaced: list[Pairing] = []

    agreeing = sorted((p for p in pairs if p.iou >= AGREE_IOU), key=lambda p: -p.iou)
    inferred = sorted((p for p in pairs if p.iou < AGREE_IOU), key=lambda p: -p.quality)
    for pair in [*agreeing, *inferred]:
        if pair.human in accepted or pair.teacher in taken_teacher:
            # Lost: either this human already has a better teacher box, or this
            # teacher box already belongs to a better-fitting human.
            if pair.human not in accepted:
                displaced.append(pair)
            continue
        accepted[pair.human] = pair
        taken_teacher.add(pair.teacher)
    return accepted, displaced


# --------------------------------------------------------------------------
# harmonising one frame
# --------------------------------------------------------------------------


@dataclass(slots=True)
class HarmonisedBox:
    xyxy: tuple[float, float, float, float]
    origin: str                          # one of CATEGORIES
    source_human: int | None = None
    source_teacher: int | None = None
    teacher_score: float | None = None
    original_xyxy: tuple[float, float, float, float] | None = None
    note: str = ""


@dataclass(slots=True)
class FrameResult:
    stem: str
    scene: str
    split: str
    width: int
    height: int
    boxes: list[HarmonisedBox]
    counts: Counter = field(default_factory=Counter)
    conflicts: int = 0
    teacher_only_suppressed: int = 0


def harmonise_frame(
    stem: str, scene: str, split: str, width: int, height: int,
    human: np.ndarray, teacher: np.ndarray, teacher_scores: np.ndarray,
) -> FrameResult:
    from argus.eval.detector import iou_matrix

    usable = teacher_scores >= TEACHER_THRESHOLD
    teacher, teacher_scores = teacher[usable], teacher_scores[usable]

    pairs = plausible_pairs(human, teacher)
    accepted, displaced = assign_one_to_one(pairs)

    out: list[HarmonisedBox] = []
    for index in range(len(human)):
        original = tuple(float(v) for v in human[index])
        pair = accepted.get(index)
        if pair is None:
            contested = any(p.human == index for p in displaced)
            out.append(HarmonisedBox(
                xyxy=original, origin="extent_unverified", source_human=index,
                note="lost a contested teacher box to a better-fitting human label"
                     if contested else "no teacher box plausibly covers this label",
            ))
            continue
        if pair.head_like:
            out.append(HarmonisedBox(
                xyxy=tuple(float(v) for v in teacher[pair.teacher]),
                origin="extent_from_teacher", source_human=index, source_teacher=pair.teacher,
                teacher_score=float(teacher_scores[pair.teacher]), original_xyxy=original,
                note=f"head-like label (aspect {pair.aspect:.2f}, "
                     f"{pair.height_ratio:.2f}x the teacher box's height)",
            ))
        elif pair.iou >= AGREE_IOU:
            out.append(HarmonisedBox(
                xyxy=original, origin="human_kept_agreed", source_human=index,
                source_teacher=pair.teacher, teacher_score=float(teacher_scores[pair.teacher]),
                note=f"IoU {pair.iou:.2f} with the teacher — the conventions already coincide",
            ))
        else:
            out.append(HarmonisedBox(
                xyxy=original, origin="human_kept_disputed", source_human=index,
                source_teacher=pair.teacher, teacher_score=float(teacher_scores[pair.teacher]),
                note=f"matched at IoU {pair.iou:.2f} but not head-like "
                     f"(aspect {pair.aspect:.2f}, {pair.height_ratio:.2f}x height) — kept as drawn",
            ))

    # Teacher boxes nobody labelled. Added only well above the operating point,
    # and only where they do not land on somebody the harmonised set already has.
    claimed = {pair.teacher for pair in accepted.values()}
    suppressed = 0
    if len(out):
        existing = np.asarray([b.xyxy for b in out], dtype=np.float32).reshape(-1, 4)
    else:
        existing = np.zeros((0, 4), dtype=np.float32)
    for index in np.argsort(-teacher_scores):
        index = int(index)
        if index in claimed or teacher_scores[index] < TEACHER_ONLY_THRESHOLD:
            continue
        box = teacher[index].reshape(1, 4)
        if len(existing) and float(iou_matrix(box, existing).max()) > TEACHER_ONLY_MAX_IOU:
            suppressed += 1
            continue
        out.append(HarmonisedBox(
            xyxy=tuple(float(v) for v in teacher[index]), origin="teacher_only",
            source_teacher=index, teacher_score=float(teacher_scores[index]),
            note="no human label claims this box",
        ))
        existing = np.concatenate([existing, box])

    counts = Counter(b.origin for b in out)
    return FrameResult(
        stem=stem, scene=scene, split=split, width=width, height=height, boxes=out,
        counts=counts, conflicts=len(displaced), teacher_only_suppressed=suppressed,
    )


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------


def write_frame(result: FrameResult, out_dir: Path, image_path: Path) -> None:
    """YOLO labels, a symlinked image, and the provenance sidecar.

    The image is symlinked rather than copied for the reason
    `train_detector.py:stage_split` gives: these are 2560x1440 frames of
    somebody's factory, and one copy on this disk is already one more than ideal.
    """
    images_dir = _assert_not_ground_truth(out_dir / "images" / result.split)
    labels_dir = _assert_not_ground_truth(out_dir / "labels" / result.split)
    prov_dir = _assert_not_ground_truth(out_dir / "provenance" / result.split)
    for directory in (images_dir, labels_dir, prov_dir):
        directory.mkdir(parents=True, exist_ok=True)

    link = images_dir / image_path.name
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(image_path.resolve())

    lines = []
    for box in result.boxes:
        x0, y0, x1, y1 = box.xyxy
        cx, cy = (x0 + x1) / 2 / result.width, (y0 + y1) / 2 / result.height
        bw, bh = (x1 - x0) / result.width, (y1 - y0) / result.height
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    (labels_dir / f"{result.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

    (prov_dir / f"{result.stem}.json").write_text(json.dumps({
        "frame": result.stem,
        "scene": result.scene,
        "split": result.split,
        "width": result.width,
        "height": result.height,
        "source_image": str(image_path.relative_to(ROOT)),
        "generator": "scripts/harmonize_labels.py",
        "convention": "whole visible person (COCO)",
        "teacher": {
            "labels_from": "footage/anmol-harvest/eval_cache/teacher_raw_large_3x2_ov25.json",
            "model": "rf-detr-large stock COCO, full frame + 3x2 native tiles",
            "extent_threshold": TEACHER_THRESHOLD,
            "add_threshold": TEACHER_ONLY_THRESHOLD,
        },
        "counts": dict(result.counts),
        "conflicts_resolved": result.conflicts,
        "teacher_only_suppressed": result.teacher_only_suppressed,
        "boxes": [
            {
                "xyxy": [round(v, 2) for v in box.xyxy],
                "origin": box.origin,
                "source_human_index": box.source_human,
                "source_teacher_index": box.source_teacher,
                "teacher_score": round(box.teacher_score, 4) if box.teacher_score is not None else None,
                "original_xyxy": [round(v, 2) for v in box.original_xyxy] if box.original_xyxy else None,
                "note": box.note,
            }
            for box in result.boxes
        ],
    }, indent=1) + "\n")


# --------------------------------------------------------------------------
# overlays
# --------------------------------------------------------------------------

BEFORE_RGB = (255, 128, 64)      # the hand label as drawn
AFTER_RGB = (60, 220, 130)       # the harmonised box
ADDED_RGB = (255, 96, 200)       # teacher_only
UNVERIFIED_RGB = (120, 170, 255)  # kept, no teacher counterpart
CROP_RGB = (255, 200, 40)


def _draw(draw: Any, boxes: Sequence[Sequence[float]], colour: tuple[int, int, int], *,
          scale: float = 1.0, offset: tuple[float, float] = (0.0, 0.0), width: int = 2) -> None:
    for x0, y0, x1, y1 in boxes:
        draw.rectangle(
            [(x0 - offset[0]) * scale, (y0 - offset[1]) * scale,
             (x1 - offset[0]) * scale, (y1 - offset[1]) * scale],
            outline=colour, width=width,
        )


def render_overlay(result: FrameResult, image_path: Path, out_path: Path, *,
                   crop_size: int = 520, crops: int = 3) -> None:
    """Before and after on one canvas, with native-resolution crops underneath.

    The crops are centred on the boxes that *changed*, because a whole-frame
    view at 1600 px wide makes a 40 px repair invisible — which is precisely the
    repair this script exists to make.
    """
    from PIL import Image, ImageDraw

    with Image.open(image_path) as handle:
        source = handle.convert("RGB")

    changed = [b for b in result.boxes if b.origin == "extent_from_teacher"]
    added = [b for b in result.boxes if b.origin == "teacher_only"]
    unverified = [b for b in result.boxes if b.origin == "extent_unverified"]
    kept = [b for b in result.boxes if b.origin in ("human_kept_agreed", "human_kept_disputed")]

    view_width = 1600
    scale = view_width / source.width
    view_height = int(round(source.height * scale))
    canvas = Image.new("RGB", (view_width, view_height + crop_size + 34), (18, 20, 24))
    canvas.paste(source.resize((view_width, view_height), Image.LANCZOS), (0, 0))
    draw = ImageDraw.Draw(canvas)

    _draw(draw, [b.original_xyxy for b in changed if b.original_xyxy], BEFORE_RGB, scale=scale)
    _draw(draw, [b.xyxy for b in changed], AFTER_RGB, scale=scale)
    _draw(draw, [b.xyxy for b in kept], AFTER_RGB, scale=scale)
    _draw(draw, [b.xyxy for b in unverified], UNVERIFIED_RGB, scale=scale)
    _draw(draw, [b.xyxy for b in added], ADDED_RGB, scale=scale)

    # Crop windows over the repairs, then the additions, spread out.
    chosen: list[tuple[int, int]] = []
    for box in sorted(changed, key=lambda b: b.xyxy[3] - b.xyxy[1]) + added + unverified:
        cx = int((box.xyxy[0] + box.xyxy[2]) / 2)
        cy = int((box.xyxy[1] + box.xyxy[3]) / 2)
        if any(abs(cx - px) < crop_size * 0.6 and abs(cy - py) < crop_size * 0.6 for px, py in chosen):
            continue
        chosen.append((cx, cy))
        if len(chosen) == crops:
            break

    strip_top = view_height + 4
    for slot, (cx, cy) in enumerate(chosen):
        x0 = int(np.clip(cx - crop_size // 2, 0, max(source.width - crop_size, 0)))
        y0 = int(np.clip(cy - crop_size // 2, 0, max(source.height - crop_size, 0)))
        crop = source.crop((x0, y0, x0 + crop_size, y0 + crop_size))
        crop_draw = ImageDraw.Draw(crop)
        _draw(crop_draw, [b.original_xyxy for b in changed if b.original_xyxy], BEFORE_RGB, offset=(x0, y0))
        _draw(crop_draw, [b.xyxy for b in changed], AFTER_RGB, offset=(x0, y0))
        _draw(crop_draw, [b.xyxy for b in kept], AFTER_RGB, offset=(x0, y0))
        _draw(crop_draw, [b.xyxy for b in unverified], UNVERIFIED_RGB, offset=(x0, y0))
        _draw(crop_draw, [b.xyxy for b in added], ADDED_RGB, offset=(x0, y0))
        left = slot * (crop_size + 8)
        canvas.paste(crop, (left, strip_top))
        draw.rectangle([left, strip_top, left + crop_size - 1, strip_top + crop_size - 1],
                       outline=CROP_RGB, width=2)
        draw.rectangle([x0 * scale, y0 * scale, (x0 + crop_size) * scale, (y0 + crop_size) * scale],
                       outline=CROP_RGB, width=2)

    draw.text((8, view_height + crop_size + 12), (
        f"{result.stem} ({result.split})   "
        f"orange = hand label as drawn ({len(changed)} replaced)   "
        f"green = harmonised ({len(changed) + len(kept)})   "
        f"blue = kept, no teacher counterpart ({len(unverified)})   "
        f"pink = teacher-only, added at >= {TEACHER_ONLY_THRESHOLD:g} ({len(added)})   "
        f"yellow = native crop windows ({crop_size} px)"
    ), fill=(215, 220, 228))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, "PNG", optimize=True)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _band(height: float) -> str:
    return "far <60" if height < 60 else ("mid 60-150" if height < 150 else "near >150")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--overlay-dir", type=Path, default=OVERLAY_DIR)
    parser.add_argument("--overlays", type=int, default=8, help="how many before/after PNGs to render")
    parser.add_argument("--no-overlays", action="store_true")
    parser.add_argument("--nms-iou", type=float, default=MERGE_NMS_IOU)
    args = parser.parse_args()

    from argus.eval.detector import load_split

    path = cache_path("large", 3, 2, 0.25)
    if not path.exists():
        raise SystemExit(
            f"no teacher cache at {path} — run scripts/teacher_eval.py first "
            f"(it caches the raw detections this reads)."
        )
    cached = load_cache(path)

    results: list[FrameResult] = []
    for split in ("train", "val"):
        for frame in load_split(args.dataset, split):
            raw = cached.get(frame.stem)
            if raw is None:
                raise SystemExit(f"teacher cache has no entry for {frame.stem}; re-run scripts/teacher_eval.py --refresh")
            boxes, scores = merge(raw, threshold=TEACHER_THRESHOLD, nms_iou=args.nms_iou)
            result = harmonise_frame(
                frame.stem, frame.scene, split, frame.width, frame.height,
                frame.boxes, boxes, scores,
            )
            write_frame(result, args.out_dir, frame.path)
            results.append(result)

    # --- report -----------------------------------------------------------
    totals: Counter = Counter()
    by_scene: dict[str, Counter] = {}
    by_split: dict[str, Counter] = {"train": Counter(), "val": Counter()}
    band_before: Counter = Counter()
    band_after: Counter = Counter()
    conflicts = suppressed = 0
    grew: list[float] = []
    for result in results:
        totals.update(result.counts)
        by_split[result.split].update(result.counts)
        by_scene.setdefault(result.scene, Counter()).update(result.counts)
        conflicts += result.conflicts
        suppressed += result.teacher_only_suppressed
        for box in result.boxes:
            band_after[_band(box.xyxy[3] - box.xyxy[1])] += 1
            if box.original_xyxy:
                band_before[_band(box.original_xyxy[3] - box.original_xyxy[1])] += 1
                grew.append((box.xyxy[3] - box.xyxy[1]) / max(box.original_xyxy[3] - box.original_xyxy[1], 1e-6))
            elif box.origin != "teacher_only":
                band_before[_band(box.xyxy[3] - box.xyxy[1])] += 1

    human_in = sum(1 for r in results for b in r.boxes if b.origin != "teacher_only")
    print(f"\n{len(results)} frames · {human_in} human boxes in · "
          f"{sum(totals.values())} harmonised boxes out")
    for category in CATEGORIES:
        print(f"  {category:<22} {totals[category]:>5}   "
              f"(train {by_split['train'][category]}, val {by_split['val'][category]})")
    print(f"  {'many-to-one refused':<22} {conflicts:>5}   "
          f"human labels that lost a contested teacher box")
    print(f"  {'teacher-only dropped':<22} {suppressed:>5}   "
          f"landed on a person already in the set")
    if grew:
        print(f"\n  median height growth of a repaired box: {float(np.median(grew)):.2f}x "
              f"(n={len(grew)})")
    print("\n  height band, human boxes only:")
    for band in ("far <60", "mid 60-150", "near >150"):
        print(f"    {band:<12} before {band_before[band]:>4}   after {band_after[band]:>4}")

    print("\n  by scene:")
    for scene in sorted(by_scene):
        counts = by_scene[scene]
        print(f"    {scene:<24} " + "  ".join(f"{c.split('_')[0][:4]}={counts[c]}" for c in CATEGORIES))

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "scripts/harmonize_labels.py",
        "convention": "whole visible person (COCO)",
        "source_labels": str(args.dataset.relative_to(ROOT)),
        "teacher_cache": str(path.relative_to(ROOT)),
        "thresholds": {
            "teacher_extent": TEACHER_THRESHOLD,
            "teacher_only_add": TEACHER_ONLY_THRESHOLD,
            "agree_iou": AGREE_IOU,
            "squarish_aspect": SQUARISH_ASPECT,
            "short_fraction": SHORT_FRACTION,
            "shoulders_aspect": SHOULDERS_ASPECT,
            "shoulders_short_fraction": SHOULDERS_SHORT_FRACTION,
            "upper_part_fraction": UPPER_PART_FRACTION,
            "contained_fraction": CONTAINED_FRACTION,
            "teacher_only_max_iou": TEACHER_ONLY_MAX_IOU,
        },
        "frames": len(results),
        "human_boxes_in": human_in,
        "boxes_out": sum(totals.values()),
        "counts": dict(totals),
        "counts_by_split": {k: dict(v) for k, v in by_split.items()},
        "counts_by_scene": {k: dict(v) for k, v in sorted(by_scene.items())},
        "conflicts_refused": conflicts,
        "teacher_only_suppressed": suppressed,
        "median_height_growth_of_repaired": round(float(np.median(grew)), 3) if grew else None,
        "band_before": dict(band_before),
        "band_after": dict(band_after),
        "warning": (
            "PRESENCE is human. EXTENT is, for the repaired boxes, a stock-COCO teacher's. "
            "teacher_only boxes are neither verified nor rejected -- docs/TEACHER_LABELS.md "
            "§7 condition 2 still stands: a human must review the far field and 100% of "
            "anything called a test set."
        ),
    }
    summary_path = _assert_not_ground_truth(args.out_dir / "summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {args.out_dir}")

    if not args.no_overlays:
        # The frames with the most repairs: an overlay of a frame nothing changed
        # in shows a harmoniser that looks perfect and tells the reader nothing.
        ranked = sorted(results, key=lambda r: -(r.counts["extent_from_teacher"] * 10
                                                 + r.counts["teacher_only"]))
        picked: list[FrameResult] = []
        seen: set[str] = set()
        for result in ranked:                      # one per scene first
            if result.scene not in seen:
                picked.append(result)
                seen.add(result.scene)
        picked += [r for r in ranked if r not in picked]
        for result in picked[:args.overlays]:
            image = args.dataset / "images" / result.split / f"{result.stem}.jpg"
            out_path = args.overlay_dir / f"{result.stem}.png"
            render_overlay(result, image, out_path)
            print(f"  {out_path.relative_to(ROOT)}  "
                  f"repaired={result.counts['extent_from_teacher']} "
                  f"added={result.counts['teacher_only']} "
                  f"unverified={result.counts['extent_unverified']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
