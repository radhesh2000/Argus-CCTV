#!/usr/bin/env python
"""Is the teacher credible? Measure it against the 79 hand-labelled frames.

Nobody should train on a pseudo-label they have not audited. This runs the
*exact* pipeline `scripts/teacher_label.py` writes labels with — RF-DETR Large,
stock COCO, full frame plus 3x2 native tiles, seam boxes dropped, one NMS — over
every frame a human has drawn boxes on, and reports what it gets right and
wrong at several score thresholds.

What is reported, and why each column is here:

* **P / R / F1 at IoU 0.5.** The number everything else in the project quotes.
* **Recall by person height in native pixels** (far < 60, mid 60-150,
  near > 150). §2.5 is the reason: the production detector gets 3 of 12 far-field
  people. A teacher that does not move that band is not worth running.
* **Centre-in-box match rate.** §2.6: on a 30 px person the hand label and the
  detector disagree about where the box *ends*, so one correct detection is
  scored as a false positive **and** a miss. The centre criterion asks only
  "is there a box on this person", which is the question a pre-labelling teacher
  is actually being asked — a human is going to adjust the extents anyway. The
  gap between the two columns is the size of the labelling disagreement, not
  the size of the teacher's error.
* **Duplicates.** A prediction that lands on a truth box another prediction
  already claimed. §2.2 measured 0 for the production detector and §5 Phase 2
  wants that property kept; tiling is the thing most likely to break it.

The reference to beat (production fine-tuned Nano, 16 val frames, from §2.2/§2.5):
P 0.870 / R 0.806, far 3/12, mid 57/70, near 81/93 — and the same weights on
3x2 tiles: P 0.749 / R 0.886.

    .venv/bin/python scripts/teacher_eval.py
    .venv/bin/python scripts/teacher_eval.py --no-overlays --refresh

Inference is cached to ``footage/anmol-harvest/eval_cache/`` before the merge,
so re-reading the table or re-rendering overlays never touches the GPU again.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.teacher_label import (  # noqa: E402
    MERGE_NMS_IOU, RawDetections, Teacher, merge, require_memory,
)

DATASET = ROOT / "footage" / "anmol-samples" / "cctv_person_tracking_dataset"
CACHE_DIR = ROOT / "footage" / "anmol-harvest" / "eval_cache"
OVERLAY_DIR = ROOT / "docs" / "screens-v3" / "teacher"

THRESHOLDS: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6)

# Person height in NATIVE pixels — the frames are 2560x1440, so these are the
# bands §2.5 reports the production detector in.
BANDS: tuple[tuple[str, float, float], ...] = (
    ("far", 0.0, 60.0),
    ("mid", 60.0, 150.0),
    ("near", 150.0, 1e9),
)

MATCH_IOU = 0.5
DUPLICATE_IOU = 0.5
CROWD_IOU = 0.3      # the "pairs of predicted boxes overlapping" metric from §2.2

# Screen recording of a video call, not a camera (§2.6). Kept in the headline
# number because it is in the corpus, but broken out because both the detector
# and the hand labels box a call participant's webcam face as a worker.
NOT_A_CAMERA = "Broken_Biscuits"


# --------------------------------------------------------------------------
# matching
# --------------------------------------------------------------------------


@dataclass(slots=True)
class FrameMatch:
    """One frame scored against its hand labels, both ways."""

    scene: str
    stem: str
    n_pred: int
    n_truth: int
    tp_iou: int
    tp_centre: int
    duplicates: int
    overlapping_pairs: int
    truth_heights: np.ndarray        # (T,) native px
    claimed_iou: np.ndarray          # (T,) score of the claimer, -1 if unclaimed
    claimed_centre: np.ndarray
    # Per centre-matched pair, so "the teacher missed them" can be told apart
    # from "the teacher found them and drew a differently-shaped box".
    pair_band: list[str]
    pair_iou: list[float]
    pair_height_ratio: list[float]   # teacher box height / hand label height


def greedy_match(
    boxes: np.ndarray, scores: np.ndarray, truth: np.ndarray, *, criterion: str, iou: float = MATCH_IOU
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy assignment by descending score — VOC/COCO's rule, and `argus.eval`'s.

    Returns ``(assigned_truth_per_prediction, claim_score_per_truth)``, the first
    being ``-1`` where a prediction matched nothing. Deliberately the same
    algorithm as ``argus.eval.detector._match`` so these numbers and the ones in
    §2.2/§2.5 mean the same thing; it is re-implemented rather than imported
    because reaching into another module's private for a number that will be
    quoted in a document is how two tables quietly stop being comparable.
    """
    from argus.eval.detector import centre_matrix, iou_matrix

    if criterion == "centre":
        overlaps_full, floor = centre_matrix(boxes, truth), 1e-6
    else:
        overlaps_full, floor = iou_matrix(boxes, truth), iou

    order = np.argsort(-scores)
    claimed = np.full(len(truth), -1.0, dtype=np.float64)
    assigned = np.full(len(boxes), -1, dtype=int)
    for row in order:
        best, best_value = -1, floor
        for column in range(len(truth)):
            if claimed[column] >= 0:
                continue
            if overlaps_full[row, column] >= best_value:
                best, best_value = column, float(overlaps_full[row, column])
        if best >= 0:
            claimed[best] = float(scores[row])
            assigned[row] = best
    return assigned, claimed


def score_frame(
    scene: str, stem: str, boxes: np.ndarray, scores: np.ndarray, truth: np.ndarray
) -> FrameMatch:
    from argus.eval.detector import iou_matrix

    assigned_iou, claimed_iou = greedy_match(boxes, scores, truth, criterion="iou")
    assigned_centre, claimed_centre = greedy_match(boxes, scores, truth, criterion="centre")

    # A duplicate is a box that found a real person and was scored a false
    # positive only because a higher-scoring box got there first. That is the
    # failure tiling causes, and it is not the same thing as a hallucination.
    duplicates = 0
    if len(boxes) and len(truth):
        overlaps = iou_matrix(boxes, truth)
        taken = claimed_iou >= 0
        for index in range(len(boxes)):
            if assigned_iou[index] >= 0:
                continue
            if bool(np.any((overlaps[index] >= DUPLICATE_IOU) & taken)):
                duplicates += 1

    # How wrong is the box on a person the teacher *did* find? §2.6 says the two
    # box conventions disagree on small people; this measures by how much,
    # rather than leaving "IoU 0.5 failed" to mean either thing.
    pair_band: list[str] = []
    pair_iou: list[float] = []
    pair_ratio: list[float] = []
    if len(boxes) and len(truth):
        overlaps = iou_matrix(boxes, truth)
        for index, column in enumerate(assigned_centre):
            if column < 0:
                continue
            truth_height = float(truth[column, 3] - truth[column, 1])
            pair_band.append(_band_of(truth_height))
            pair_iou.append(float(overlaps[index, column]))
            pair_ratio.append(float(boxes[index, 3] - boxes[index, 1]) / max(truth_height, 1e-6))

    pairs = 0
    if len(boxes) > 1:
        self_iou = iou_matrix(boxes, boxes)
        np.fill_diagonal(self_iou, 0.0)
        pairs = int(np.triu(self_iou > CROWD_IOU, k=1).sum())

    heights = (truth[:, 3] - truth[:, 1]) if len(truth) else np.zeros(0, np.float32)
    return FrameMatch(
        scene=scene, stem=stem, n_pred=len(boxes), n_truth=len(truth),
        tp_iou=int((assigned_iou >= 0).sum()), tp_centre=int((assigned_centre >= 0).sum()),
        duplicates=duplicates, overlapping_pairs=pairs,
        truth_heights=heights, claimed_iou=claimed_iou, claimed_centre=claimed_centre,
        pair_band=pair_band, pair_iou=pair_iou, pair_height_ratio=pair_ratio,
    )


def _band_of(height: float) -> str:
    for name, low, high in BANDS:
        if low <= height < high:
            return name
    return BANDS[-1][0]


def aggregate(matches: Sequence[FrameMatch]) -> dict[str, Any]:
    n_pred = sum(m.n_pred for m in matches)
    n_truth = sum(m.n_truth for m in matches)
    tp_iou = sum(m.tp_iou for m in matches)
    tp_centre = sum(m.tp_centre for m in matches)

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    precision, recall = ratio(tp_iou, n_pred), ratio(tp_iou, n_truth)
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else None
    p_centre, r_centre = ratio(tp_centre, n_pred), ratio(tp_centre, n_truth)
    f1_centre = (2 * p_centre * r_centre / (p_centre + r_centre)) if p_centre and r_centre else None

    bands: dict[str, dict[str, int]] = {}
    for name, low, high in BANDS:
        found = total = found_centre = 0
        for match in matches:
            in_band = (match.truth_heights >= low) & (match.truth_heights < high)
            total += int(in_band.sum())
            found += int((match.claimed_iou[in_band] >= 0).sum())
            found_centre += int((match.claimed_centre[in_band] >= 0).sum())
        pair_iou = [i for m in matches for i, b in zip(m.pair_iou, m.pair_band) if b == name]
        pair_ratio = [r for m in matches for r, b in zip(m.pair_height_ratio, m.pair_band) if b == name]
        bands[name] = {
            "found": found, "found_centre": found_centre, "total": total,
            # Of the people the teacher put a box on, how well does that box fit?
            "median_iou_of_hits": round(float(np.median(pair_iou)), 3) if pair_iou else None,
            "median_height_ratio": round(float(np.median(pair_ratio)), 3) if pair_ratio else None,
        }

    return {
        "frames": len(matches),
        "predictions": n_pred,
        "truth": n_truth,
        "tp": tp_iou,
        "fp": n_pred - tp_iou,
        "fn": n_truth - tp_iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_centre": p_centre,
        "recall_centre": r_centre,
        "f1_centre": f1_centre,
        "duplicates": sum(m.duplicates for m in matches),
        "overlapping_pairs": sum(m.overlapping_pairs for m in matches),
        "bands": bands,
    }


# --------------------------------------------------------------------------
# corpus + cache
# --------------------------------------------------------------------------


def cache_path(variant: str, cols: int, rows: int, overlap: float) -> Path:
    return CACHE_DIR / f"teacher_raw_{variant}_{cols}x{rows}_ov{int(overlap * 100)}.json"


def load_cache(path: Path) -> dict[str, RawDetections]:
    payload = json.loads(path.read_text())
    cached: dict[str, RawDetections] = {}
    for key, entry in payload["frames"].items():
        boxes = np.asarray([b["xyxy"] for b in entry["boxes"]], dtype=np.float32).reshape(-1, 4)
        scores = np.asarray([b["score"] for b in entry["boxes"]], dtype=np.float32).reshape(-1)
        cached[key] = RawDetections(
            width=entry["width"], height=entry["height"], xyxy=boxes, scores=scores,
            source=[b["source"] for b in entry["boxes"]], dropped_on_seam=entry["dropped_on_seam"],
        )
    return cached


def run_teacher(frames: Sequence[Any], teacher: Teacher, path: Path) -> tuple[dict[str, RawDetections], float]:
    from argus.eval.detector import _load_bgr
    import torch

    cached: dict[str, RawDetections] = {}
    for index, frame in enumerate(frames, start=1):
        cached[frame.stem] = teacher.raw(_load_bgr(frame.path))
        if index % 20 == 0 or index == len(frames):
            peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
            print(f"  {index}/{len(frames)} frames · peak GPU {peak:.2f} GiB", flush=True)
    peak_gib = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    reserved_gib = torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else 0.0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "teacher": teacher.describe(),
        "peak_gpu_allocated_gib": round(peak_gib, 3),
        "peak_gpu_reserved_gib": round(reserved_gib, 3),
        "frames": {stem: raw.to_json() for stem, raw in cached.items()},
    }, indent=1) + "\n")
    return cached, reserved_gib


# --------------------------------------------------------------------------
# conditions
# --------------------------------------------------------------------------


def restrict(raw: RawDetections, sources: str) -> RawDetections:
    """The same cached detections, seen through one pass only.

    ``full`` and ``tiles`` are not separate inference runs — they are the merge
    replayed over a subset of what was already cached, which is what makes the
    contribution of each pass readable off one table.
    """
    if sources == "both":
        return raw
    wanted = [i for i, src in enumerate(raw.source)
              if (src == "full") == (sources == "full")]
    index = np.asarray(wanted, dtype=int)
    return RawDetections(
        width=raw.width, height=raw.height,
        xyxy=raw.xyxy[index].reshape(-1, 4), scores=raw.scores[index].reshape(-1),
        source=[raw.source[i] for i in wanted], dropped_on_seam=raw.dropped_on_seam,
    )


def sweep(
    frames: Sequence[Any], cached: dict[str, RawDetections], splits: dict[str, str], *, sources: str,
    thresholds: Sequence[float], nms_iou: float, subset: str,
) -> dict[str, dict[str, Any]]:
    keep = [f for f in frames if _in_subset(f, splits, subset)]
    results: dict[str, dict[str, Any]] = {}
    for threshold in thresholds:
        matches = []
        for frame in keep:
            boxes, scores = merge(restrict(cached[frame.stem], sources), threshold=threshold, nms_iou=nms_iou)
            matches.append(score_frame(frame.scene, frame.stem, boxes, scores, frame.boxes))
        results[f"{threshold:.2f}"] = aggregate(matches)
    return results


def _in_subset(frame: Any, splits: dict[str, str], subset: str) -> bool:
    """``LabelledFrame`` has ``slots=True``, so which split a frame came from is
    carried beside it rather than stapled onto it."""
    if subset == "all":
        return True
    if subset == "cameras":
        return frame.scene != NOT_A_CAMERA
    if subset == "val":
        return splits[frame.stem] == "val"
    if subset == "val-cameras":
        return splits[frame.stem] == "val" and frame.scene != NOT_A_CAMERA
    raise ValueError(subset)


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------


def _fmt(value: float | None, places: int = 3) -> str:
    return "-" if value is None else f"{value:.{places}f}"


def table(title: str, rows: dict[str, dict[str, Any]], *, note: str = "") -> str:
    """The band columns carry both criteria: ``a/b (c)`` is *a* of *b* people found
    at IoU 0.5, and *c* of them found by the weaker centre-in-box test. On a 30 px
    person those two numbers come apart, and the gap is §2.6's label-extent
    disagreement rather than anything the teacher did — which is exactly the
    distinction someone deciding whether to pre-label with it needs to see."""
    head = (
        f"\n### {title}\n{note}\n\n"
        "| score | P | R | F1 | P·ctr | R·ctr | far <60 | mid 60-150 | near >150 | dup | pairs | boxes |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for threshold, result in rows.items():
        bands = result["bands"]
        cells = " ".join(
            f"| {bands[name]['found']}/{bands[name]['total']} ({bands[name]['found_centre']}) "
            f"{_fmt(bands[name]['median_iou_of_hits'], 2)}"
            for name, _, _ in BANDS
        )
        lines.append(
            f"| {threshold} | {_fmt(result['precision'])} | {_fmt(result['recall'])} | {_fmt(result['f1'])} "
            f"| {_fmt(result['precision_centre'])} | {_fmt(result['recall_centre'])} "
            f"{cells} | {result['duplicates']} | {result['overlapping_pairs']} | {result['predictions']} |"
        )
    return head + "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# overlays
# --------------------------------------------------------------------------

HUMAN_RGB = (64, 156, 255)
TEACHER_RGB = (60, 220, 130)
MISS_RGB = (255, 96, 96)
CROP_RGB = (255, 200, 40)


def _draw_boxes(draw: Any, boxes: np.ndarray, colour: tuple[int, int, int], *, scale: float = 1.0,
                offset: tuple[float, float] = (0.0, 0.0), width: int = 2) -> None:
    for x0, y0, x1, y1 in boxes:
        draw.rectangle(
            [(x0 - offset[0]) * scale, (y0 - offset[1]) * scale,
             (x1 - offset[0]) * scale, (y1 - offset[1]) * scale],
            outline=colour, width=width,
        )


def render_overlay(
    frame: Any, boxes: np.ndarray, scores: np.ndarray, out_path: Path, *, threshold: float,
    crop_size: int = 520, crops: int = 3,
) -> None:
    """One PNG a human can judge the far field from.

    Top: the whole frame, scaled to fit a screen, teacher boxes against hand
    labels. Bottom: native-resolution crops centred on the *smallest* hand
    labels, because that is the only place the interesting disagreement lives
    and at 1600 px wide a 40 px person is four pixels of nothing.
    """
    from PIL import Image, ImageDraw

    with Image.open(frame.path) as handle:
        source = handle.convert("RGB")

    truth = frame.boxes
    matched, claimed = greedy_match(boxes, scores, truth, criterion="centre")
    missed = truth[claimed < 0] if len(truth) else truth

    view_width = 1600
    scale = view_width / source.width
    view_height = int(round(source.height * scale))
    strip_height = crop_size if len(truth) else 0
    canvas = Image.new("RGB", (view_width, view_height + strip_height + 34), (18, 20, 24))
    canvas.paste(source.resize((view_width, view_height), Image.LANCZOS), (0, 0))
    draw = ImageDraw.Draw(canvas)

    _draw_boxes(draw, truth, HUMAN_RGB, scale=scale, width=2)
    _draw_boxes(draw, boxes, TEACHER_RGB, scale=scale, width=2)
    _draw_boxes(draw, missed, MISS_RGB, scale=scale, width=3)

    # Crop windows: the smallest hand labels, spread out so three windows do not
    # all land on the same three people.
    chosen: list[tuple[int, int]] = []
    if len(truth):
        heights = truth[:, 3] - truth[:, 1]
        for index in np.argsort(heights):
            cx = int((truth[index, 0] + truth[index, 2]) / 2)
            cy = int((truth[index, 1] + truth[index, 3]) / 2)
            if any(abs(cx - px) < crop_size * 0.6 and abs(cy - py) < crop_size * 0.6 for px, py in chosen):
                continue
            chosen.append((cx, cy))
            if len(chosen) == crops:
                break

    strip_top = view_height + 4
    for slot, (cx, cy) in enumerate(chosen):
        x0 = int(np.clip(cx - crop_size // 2, 0, source.width - crop_size))
        y0 = int(np.clip(cy - crop_size // 2, 0, source.height - crop_size))
        crop = source.crop((x0, y0, x0 + crop_size, y0 + crop_size))
        crop_draw = ImageDraw.Draw(crop)
        _draw_boxes(crop_draw, truth, HUMAN_RGB, offset=(x0, y0), width=2)
        _draw_boxes(crop_draw, boxes, TEACHER_RGB, offset=(x0, y0), width=2)
        _draw_boxes(crop_draw, missed, MISS_RGB, offset=(x0, y0), width=3)
        left = slot * (crop_size + 8)
        canvas.paste(crop, (left, strip_top))
        draw.rectangle([left, strip_top, left + crop_size - 1, strip_top + crop_size - 1],
                       outline=CROP_RGB, width=2)
        # Mark where this crop came from, on the overview.
        draw.rectangle([x0 * scale, y0 * scale, (x0 + crop_size) * scale, (y0 + crop_size) * scale],
                       outline=CROP_RGB, width=2)

    tp = int(greedy_match(boxes, scores, truth, criterion="iou")[0].sum())
    caption = (
        f"{frame.stem}   blue = human label ({len(truth)})   green = teacher @ {threshold:g} ({len(boxes)}), "
        f"{tp} over IoU 0.5   red = human label the teacher missed ({len(missed)})   "
        f"yellow = native-resolution crop windows ({crop_size} px)"
    )
    draw.text((8, view_height + strip_height + 12), caption, fill=(215, 220, 228))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, "PNG", optimize=True)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--variant", default="large", choices=["nano", "small", "medium", "large"])
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=MERGE_NMS_IOU)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--thresholds", type=float, nargs="*", default=list(THRESHOLDS))
    parser.add_argument("--overlay-threshold", type=float, default=0.4)
    parser.add_argument("--refresh", action="store_true", help="re-run inference even if the cache is warm")
    parser.add_argument("--no-overlays", action="store_true")
    parser.add_argument("--overlay-dir", type=Path, default=OVERLAY_DIR)
    parser.add_argument("--report", type=Path, default=None, help="write the JSON result here")
    args = parser.parse_args()

    from argus.eval.detector import load_split

    frames = []
    splits: dict[str, str] = {}
    for split in ("train", "val"):
        for frame in load_split(args.dataset, split):
            splits[frame.stem] = split
            frames.append(frame)
    print(f"{len(frames)} hand-labelled frames, {sum(len(f.boxes) for f in frames)} boxes", flush=True)

    path = cache_path(args.variant, args.cols, args.rows, args.overlap)
    peak_gib = 0.0
    if path.exists() and not args.refresh:
        cached = load_cache(path)
        missing = [f.stem for f in frames if f.stem not in cached]
        if missing:
            raise SystemExit(f"cache {path} is missing {len(missing)} frames; re-run with --refresh")
        print(f"using cached detections from {path}", flush=True)
    else:
        available = require_memory()
        print(f"MemAvailable {available:.1f} GiB — running the teacher", flush=True)
        teacher = Teacher(variant=args.variant, cols=args.cols, rows=args.rows,
                          overlap=args.overlap, nms_iou=args.nms_iou, batch=args.batch).load()
        print(json.dumps(teacher.describe()), flush=True)
        cached, peak_gib = run_teacher(frames, teacher, path)

    header = json.loads(path.read_text())
    report: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "scripts/teacher_eval.py",
        "dataset": str(args.dataset.relative_to(ROOT)),
        "teacher": header["teacher"],
        "peak_gpu_reserved_gib": round(peak_gib, 3) or header.get("peak_gpu_reserved_gib"),
        "reference_production_nano_val16": {
            "precision": 0.870, "recall": 0.806,
            "bands": {"far": "3/12", "mid": "57/70", "near": "81/93"},
            "with_3x2_tiles": {"precision": 0.749, "recall": 0.886},
            "source": "docs/DETECTION_V2.md §2.2, §2.5",
        },
        "conditions": {},
    }

    printed: list[str] = []
    for subset, subset_label in (
        ("all", "all 79 hand-labelled frames"),
        ("cameras", f"69 frames, excluding {NOT_A_CAMERA} (a screen recording, §2.6)"),
        ("val", "the 16 val frames — directly comparable to the §2.2/§2.5 numbers"),
    ):
        grid = f"{args.cols}x{args.rows}"
        for sources, sources_label in (("both", f"full frame + {grid} tiles"), ("full", "full frame only"),
                                       ("tiles", f"{grid} tiles only")):
            rows = sweep(frames, cached, splits, sources=sources, thresholds=args.thresholds,
                         nms_iou=args.nms_iou, subset=subset)
            report["conditions"][f"{subset}/{sources}"] = rows
            if sources == "both" or subset == "val":
                printed.append(table(f"{subset_label} — {sources_label}", rows))

    print("".join(printed))

    if not args.no_overlays:
        rendered = render_overlays(frames, cached, args, report)
        report["overlays"] = rendered
        print(f"\n{len(rendered)} overlays -> {args.overlay_dir}")

    report_path = args.report or (CACHE_DIR / "teacher_eval.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"report: {report_path}")
    return 0


def render_overlays(frames: Sequence[Any], cached: dict[str, RawDetections], args: Any,
                    report: dict[str, Any]) -> list[str]:
    """One frame per camera scene, each the frame with the most far-field people.

    Picked by far-field count rather than at random: an overlay of a frame whose
    people are all in the foreground would show a teacher that looks perfect and
    tell a reader nothing.
    """
    by_scene: dict[str, tuple[int, Any]] = {}
    for frame in frames:
        if frame.scene == NOT_A_CAMERA or len(frame.boxes) == 0:
            continue
        heights = frame.boxes[:, 3] - frame.boxes[:, 1]
        far = int(((heights > 0) & (heights < 60)).sum())
        mid = int(((heights >= 60) & (heights < 150)).sum())
        rank = far * 100 + mid
        if rank > by_scene.get(frame.scene, (-1, None))[0]:
            by_scene[frame.scene] = (rank, frame)

    written: list[str] = []
    for scene in sorted(by_scene):
        _, frame = by_scene[scene]
        boxes, scores = merge(cached[frame.stem], threshold=args.overlay_threshold, nms_iou=args.nms_iou)
        out_path = args.overlay_dir / f"{scene}.png"
        render_overlay(frame, boxes, scores, out_path, threshold=args.overlay_threshold)
        written.append(str(out_path.relative_to(ROOT)))
        print(f"  {out_path.relative_to(ROOT)}  truth={len(frame.boxes)} teacher={len(boxes)}", flush=True)
    return written


if __name__ == "__main__":
    raise SystemExit(main())
