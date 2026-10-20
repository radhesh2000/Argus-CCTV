"""Measure the *shipped* detector against hand-drawn boxes on real plant footage.

`argus eval` asks whether a VLM can be trusted to verify a claim. This asks the
question underneath it: does the COCO RF-DETR Nano the appliance ships find the
people in *this* factory at all — before anybody spends a week fine-tuning it.

The corpus is Radhesh's export from Anmol Industries' biscuit plant: YOLO-format
person boxes over frames from eight scenes, under `footage/anmol-samples/`,
gitignored because it is a customer's plant. Absent, this command says so and
exits 2 rather than pretending.

Four decisions shape every number below.

**The engine's path is measured, not a fair one.** The appliance decodes to
960x544 (`detection.width/height`) and the detector then squashes that to the
model's 384x384 input. A 2560x1440 frame goes through two aspect distortions and
a 7x downscale before RF-DETR sees it, and a person 60 px tall in the source is
23 px by then. That is not a flattering pipeline and it is the one that ships,
so it is the headline. The other conditions exist to split "the detector cannot
see these people" from "the pipeline threw them away" — different repairs.

**Matching is greedy, per frame, by descending confidence, at IoU 0.5** — the
VOC/COCO convention, and what the headline AP assumes.

**IoU 0.5 is also scored loosely, and that is not a softer grade.** A visual
audit of these labels (see `docs/DETECTOR_EVAL.md`) found the annotator's box
convention differs from the detector's: truth boxes frequently cover a person's
visible upper body where the detector returns the whole figure, and a handful
sit off their subject entirely. Against labels like that, IoU 0.5 conflates
*did not find the person* with *found the person and drew a different box* —
which are the difference between "fine-tune the detector" and "fix the labels".
So every table also carries a `center` criterion: a prediction claims a truth
box when it contains that box's centre. It is a weaker localisation claim and a
much better presence claim, and headcount is a presence question.

**Recall is bucketed by how tall the person is in the decoded frame**, because
that is the number an operator can act on. `detection.height` is a config line;
"the detector is bad at this plant" is not.

Scores are read at a 0.01 floor so one inference pass carries the whole sweep;
every operating point is a slice of it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from argus.config import ROOT

# Where the corpus lands. Gitignored: it is a customer's plant, and the seed
# file that enrols its videos (`config/seed.anmol.yaml`) points at the same tree.
DEFAULT_DATASET = ROOT / "footage" / "anmol-samples" / "cctv_person_tracking_dataset"

# The engine's threshold (`detection.model.threshold`), so it gets a row of its
# own in every table whatever else is swept.
ENGINE_THRESHOLD = 0.5
OPERATING_POINTS: tuple[float, ...] = (0.25, 0.4, ENGINE_THRESHOLD, 0.6, 0.75)

# Every detection above this is kept, so one pass feeds the whole sweep.
SCORE_FLOOR = 0.01

MATCH_IOU = 0.5
LOOSE_IOU = 0.25

# Person height in the *decoded* frame (960x544), which is the size the detector
# actually works at. 24 px is roughly where a 384x384 model's stride stops
# resolving a torso; 96 px is a person in the foreground.
HEIGHT_BUCKETS: tuple[tuple[float, float], ...] = ((0, 24), (24, 48), (48, 96), (96, 1e9))

# Grey used to pad a letterboxed frame — what YOLO/DETR training pipelines pad
# with, so it is the least surprising thing to show a detector.
_PAD_VALUE = 114

CONDITIONS: tuple[str, ...] = ("engine", "native", "native-prefiltered", "native-letterbox")
SHIPPED_CONDITION = "engine"


class DetectorEvalError(RuntimeError):
    """The corpus is missing or unreadable."""


# --------------------------------------------------------------------------
# corpus
# --------------------------------------------------------------------------


@dataclass(slots=True)
class LabelledFrame:
    """One image and the person boxes somebody drew on it, in image pixels."""

    path: Path
    scene: str
    width: int
    height: int
    boxes: np.ndarray            # (N, 4) float32 xyxy

    @property
    def stem(self) -> str:
        return self.path.stem


def scene_of(stem: str) -> str:
    """``Line_No.3_Packing3_Bas_frame_07`` -> ``Line_No.3_Packing3_Bas``.

    The export names every frame after the video it was cut from, which is the
    only per-scene signal in the dataset — there is no manifest. Scene is the
    axis that matters: a mean over eight rooms hides the one the detector is
    blind in.
    """
    head, sep, _ = stem.rpartition("_frame_")
    return head if sep else stem


def _read_yolo_labels(path: Path, width: int, height: int) -> np.ndarray:
    """YOLO ``class cx cy w h`` (normalized) -> pixel xyxy.

    Class is ignored: the export is single-class (verified — 1015 boxes, all
    class 0). A multi-class export would need this to filter, and would also
    need a different question asked of it.
    """
    boxes: list[list[float]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 5:
            raise DetectorEvalError(f"{path}:{line_number}: expected 'class cx cy w h'")
        try:
            cx, cy, bw, bh = (float(value) for value in parts[1:5])
        except ValueError as exc:
            raise DetectorEvalError(f"{path}:{line_number}: {exc}") from exc
        boxes.append([
            (cx - bw / 2) * width, (cy - bh / 2) * height,
            (cx + bw / 2) * width, (cy + bh / 2) * height,
        ])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def load_split(dataset: Path, split: str) -> list[LabelledFrame]:
    """Every image in ``images/<split>`` with the label file beside it."""
    images_dir = dataset / "images" / split
    labels_dir = dataset / "labels" / split
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise DetectorEvalError(
            f"dataset split {split!r} not found under {dataset} "
            f"(expected images/{split} and labels/{split})"
        )
    from PIL import Image

    frames: list[LabelledFrame] = []
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            # A label-less image is not a negative example here: the export
            # pairs them, so a missing file means a broken copy, not an empty
            # scene. Skipping it silently would quietly shrink the denominator.
            raise DetectorEvalError(f"no label file for {image_path.name} (looked at {label_path})")
        with Image.open(image_path) as handle:
            width, height = handle.size
        frames.append(LabelledFrame(
            path=image_path,
            scene=scene_of(image_path.stem),
            width=width,
            height=height,
            boxes=_read_yolo_labels(label_path, width, height),
        ))
    if not frames:
        raise DetectorEvalError(f"no images in {images_dir}")
    return frames


# --------------------------------------------------------------------------
# conditions — how a frame is presented to the detector
# --------------------------------------------------------------------------


def _load_bgr(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as handle:
        rgb = np.asarray(handle.convert("RGB"), dtype=np.uint8)
    return np.ascontiguousarray(rgb[:, :, ::-1])


def _resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Pillow's BILINEAR, which for a reduction is a properly filtered resample.

    That filtering is not incidental — it is the whole difference between the
    `engine` and `native` conditions, and it is also what ffmpeg's scaler does
    in the real pipeline, so this is the faithful stand-in for it.
    """
    from PIL import Image

    resized = Image.fromarray(image[:, :, ::-1]).resize((width, height), Image.BILINEAR)
    return np.ascontiguousarray(np.asarray(resized, dtype=np.uint8)[:, :, ::-1])


def prepare(
    frame: LabelledFrame, condition: str, *, width: int, height: int, model_resolution: int = 384
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """The image the detector sees, and how to get its boxes back to source pixels.

    Ground truth never moves: predictions are mapped back into the frame the
    labels were drawn in, so every condition is scored against one fixed truth.
    """
    image = _load_bgr(frame.path)

    def rescale(from_width: int, from_height: int) -> Callable[[np.ndarray], np.ndarray]:
        sx, sy = frame.width / from_width, frame.height / from_height

        def back(boxes: np.ndarray) -> np.ndarray:
            out = boxes.copy()
            out[:, 0::2] *= sx
            out[:, 1::2] *= sy
            return out

        return back

    if condition == "engine":
        # Exactly `-vf scale=960:544` out of the decoder: filtered, aspect
        # ignored. The detector then squashes 960x544 to 384x384 itself.
        return _resize(image, width, height), rescale(width, height)

    if condition == "native":
        # Full frame in. The detector's own resize to 384x384 is
        # `F.interpolate(antialias=False)`, so this is a 6.7x *point-sampled*
        # reduction — the condition that shows what that costs.
        return image, lambda boxes: boxes

    if condition == "native-prefiltered":
        # Full frame, but resampled to the model's exact input with a real
        # filter, so the detector's internal resize is a no-op. Same pixels
        # reaching the network as `native` intends, minus the aliasing.
        return (
            _resize(image, model_resolution, model_resolution),
            rescale(model_resolution, model_resolution),
        )

    if condition == "native-letterbox":
        side = max(frame.width, frame.height)
        pad_x, pad_y = (side - frame.width) // 2, (side - frame.height) // 2
        canvas = np.full((side, side, 3), _PAD_VALUE, dtype=np.uint8)
        canvas[pad_y:pad_y + frame.height, pad_x:pad_x + frame.width] = image

        def unpad(boxes: np.ndarray) -> np.ndarray:
            out = boxes.copy()
            out[:, 0::2] -= pad_x
            out[:, 1::2] -= pad_y
            return out

        return canvas, unpad

    raise DetectorEvalError(f"unknown condition {condition!r} (expected one of {list(CONDITIONS)})")


# --------------------------------------------------------------------------
# matching
# --------------------------------------------------------------------------


def iou_matrix(predictions: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """(P, T) pairwise IoU. Empty either side gives an empty matrix, not a crash."""
    if len(predictions) == 0 or len(truth) == 0:
        return np.zeros((len(predictions), len(truth)), dtype=np.float32)
    px1, py1, px2, py2 = (predictions[:, i][:, None] for i in range(4))
    tx1, ty1, tx2, ty2 = (truth[:, i][None, :] for i in range(4))
    inter_w = np.clip(np.minimum(px2, tx2) - np.maximum(px1, tx1), 0, None)
    inter_h = np.clip(np.minimum(py2, ty2) - np.maximum(py1, ty1), 0, None)
    inter = inter_w * inter_h
    area_p = np.clip(px2 - px1, 0, None) * np.clip(py2 - py1, 0, None)
    area_t = np.clip(tx2 - tx1, 0, None) * np.clip(ty2 - ty1, 0, None)
    union = area_p + area_t - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def centre_matrix(predictions: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """(P, T): IoU where the truth box's centre falls inside the prediction, else 0.

    The presence criterion. Ranking eligible pairs by IoU keeps the greedy
    matching sane when two people overlap — the better-fitting box wins the
    truth box it is actually on.
    """
    if len(predictions) == 0 or len(truth) == 0:
        return np.zeros((len(predictions), len(truth)), dtype=np.float32)
    cx = ((truth[:, 0] + truth[:, 2]) / 2)[None, :]
    cy = ((truth[:, 1] + truth[:, 3]) / 2)[None, :]
    inside = (
        (cx >= predictions[:, 0][:, None]) & (cx <= predictions[:, 2][:, None])
        & (cy >= predictions[:, 1][:, None]) & (cy <= predictions[:, 3][:, None])
    )
    return np.where(inside, iou_matrix(predictions, truth), 0.0).astype(np.float32)


def _match(
    boxes: np.ndarray, scores: np.ndarray, truth: np.ndarray, *, criterion: str, iou: float
) -> tuple[list[tuple[float, bool]], np.ndarray]:
    """Greedy assignment by descending confidence.

    Returns ``(score, is_true_positive)`` per prediction and, per truth box, the
    score of whatever claimed it (``-1.0`` for unclaimed). The second half is
    what makes recall-by-size possible without matching twice.
    """
    if criterion == "centre":
        overlaps_full = centre_matrix(boxes, truth)
        floor = 1e-6
    else:
        overlaps_full = iou_matrix(boxes, truth)
        floor = iou
    order = np.argsort(-scores)
    overlaps = overlaps_full[order]
    claimed = np.full(len(truth), -1.0, dtype=np.float64)
    scored: list[tuple[float, bool]] = []
    for row, index in enumerate(order):
        best, best_value = -1, floor
        for column in range(len(truth)):
            if claimed[column] >= 0:
                continue
            if overlaps[row, column] >= best_value:
                best, best_value = column, overlaps[row, column]
        score = float(scores[index])
        if best >= 0:
            claimed[best] = score
        scored.append((score, best >= 0))
    return scored, claimed


def match_frame(
    boxes: np.ndarray, scores: np.ndarray, truth: np.ndarray, *, iou: float = MATCH_IOU
) -> list[tuple[float, bool]]:
    """``(score, is_true_positive)`` per prediction, greedy by confidence.

    Greedy rather than optimal assignment: it is what VOC, COCO and every AP
    number anyone will compare this to do.
    """
    return _match(boxes, scores, truth, criterion="iou", iou=iou)[0]


def average_precision(scored: Sequence[tuple[float, bool]], n_truth: int) -> float | None:
    """All-point interpolated AP over one pooled set of predictions.

    ``None`` when there is nothing to score — an AP of 0.0 for a scene with no
    ground truth reads as failure rather than as absence.
    """
    if n_truth == 0:
        return None
    if not scored:
        return 0.0
    ordered = sorted(scored, key=lambda item: -item[0])
    hits = np.cumsum([1.0 if is_tp else 0.0 for _, is_tp in ordered])
    ranks = np.arange(1, len(ordered) + 1, dtype=np.float64)
    precision = hits / ranks
    recall = hits / n_truth
    # Monotone envelope, right to left: precision at recall r is the best
    # precision achievable at recall >= r.
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.sum(np.diff(np.concatenate(([0.0], recall))) * envelope))


@dataclass(slots=True)
class OperatingPoint:
    threshold: float
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float | None:
        denominator = self.tp + self.fp
        return None if denominator == 0 else self.tp / denominator

    @property
    def recall(self) -> float | None:
        denominator = self.tp + self.fn
        return None if denominator == 0 else self.tp / denominator

    @property
    def f1(self) -> float | None:
        precision, recall = self.precision, self.recall
        if not precision or not recall:
            return None
        return 2 * precision * recall / (precision + recall)

    def as_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold, "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "precision": self.precision, "recall": self.recall, "f1": self.f1,
        }


def operating_point(
    scored: Sequence[tuple[float, bool]], n_truth: int, threshold: float
) -> OperatingPoint:
    """Confusion counts if the detector had run at ``threshold``.

    Sliced out of the 0.01-floor pass rather than re-inferred. Exact for
    everything except NMS, which ran once over the low-threshold set — a box
    suppressed there stays suppressed here, the same order ``_postprocess``
    applies them in (threshold first, then NMS).
    """
    point = OperatingPoint(threshold=threshold)
    for score, is_tp in scored:
        if score < threshold:
            continue
        if is_tp:
            point.tp += 1
        else:
            point.fp += 1
    point.fn = n_truth - point.tp
    return point


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------


@dataclass(slots=True)
class FrameResult:
    """Raw predictions and truth for one frame, matched lazily.

    Predictions are kept rather than reduced to a verdict because every question
    below — a different IoU, the presence criterion, recall by person height —
    is a different matching over the same inference, and re-running the detector
    to ask them would be four passes to learn one thing.
    """

    stem: str
    scene: str
    truth: np.ndarray            # (T, 4) source pixels
    boxes: np.ndarray            # (P, 4) source pixels
    scores: np.ndarray           # (P,)
    truth_heights: np.ndarray    # (T,) person height in the decoded frame
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def n_truth(self) -> int:
        return len(self.truth)

    def matched(self, *, criterion: str = "iou", iou: float = MATCH_IOU):
        key = (criterion, iou)
        if key not in self._cache:
            self._cache[key] = _match(
                self.boxes, self.scores, self.truth, criterion=criterion, iou=iou
            )
        return self._cache[key]


@dataclass(slots=True)
class ConditionResult:
    condition: str
    split: str
    input_size: str
    frames: list[FrameResult] = field(default_factory=list)
    ms_per_frame: float = 0.0

    def _select(self, scene: str | None) -> list[FrameResult]:
        return [f for f in self.frames if scene is None or f.scene == scene]

    def _pool(
        self, scene: str | None, criterion: str, iou: float
    ) -> tuple[list[tuple[float, bool]], int]:
        scored: list[tuple[float, bool]] = []
        n_truth = 0
        for frame in self._select(scene):
            scored.extend(frame.matched(criterion=criterion, iou=iou)[0])
            n_truth += frame.n_truth
        return scored, n_truth

    def ap(
        self, scene: str | None = None, *, criterion: str = "iou", iou: float = MATCH_IOU
    ) -> float | None:
        scored, n_truth = self._pool(scene, criterion, iou)
        return average_precision(scored, n_truth)

    def point(
        self, threshold: float, scene: str | None = None,
        *, criterion: str = "iou", iou: float = MATCH_IOU,
    ) -> OperatingPoint:
        scored, n_truth = self._pool(scene, criterion, iou)
        return operating_point(scored, n_truth, threshold)

    def recall_by_height(
        self, threshold: float = ENGINE_THRESHOLD, *,
        buckets: Sequence[tuple[float, float]] = HEIGHT_BUCKETS,
        criterion: str = "iou", iou: float = MATCH_IOU, scene: str | None = None,
    ) -> list[tuple[tuple[float, float], int, int]]:
        """``(bucket, found, total)`` per person-height band, in decoded pixels."""
        found = [0] * len(buckets)
        total = [0] * len(buckets)
        for frame in self._select(scene):
            claimed = frame.matched(criterion=criterion, iou=iou)[1]
            for index, height in enumerate(frame.truth_heights):
                for position, (low, high) in enumerate(buckets):
                    if low <= height < high:
                        total[position] += 1
                        if claimed[index] >= threshold:
                            found[position] += 1
                        break
        return [(bucket, found[i], total[i]) for i, bucket in enumerate(buckets)]

    def scenes(self) -> list[str]:
        return sorted({frame.scene for frame in self.frames})

    def truth_count(self, scene: str | None = None) -> int:
        return sum(frame.n_truth for frame in self._select(scene))

    def frame_count(self, scene: str | None = None) -> int:
        return len(self._select(scene))


@dataclass(slots=True)
class DetectorEvalRun:
    dataset: Path
    detector: dict[str, Any]
    results: list[ConditionResult] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    # A second run of the *same* corpus, conditions and matching, by a detector
    # carrying fine-tuned weights. Nested rather than kept side by side because
    # the only thing a fine-tune's numbers mean is the difference from these
    # ones: quoted alone they are a model scored on the split that stopped it.
    finetuned: "DetectorEvalRun | None" = None

    def splits(self) -> list[str]:
        seen: list[str] = []
        for result in self.results:
            if result.split not in seen:
                seen.append(result.split)
        return seen

    def conditions(self) -> list[str]:
        seen: list[str] = []
        for result in self.results:
            if result.condition not in seen:
                seen.append(result.condition)
        return seen

    def get(self, split: str, condition: str) -> ConditionResult | None:
        for result in self.results:
            if result.split == split and result.condition == condition:
                return result
        return None

    def as_dict(self) -> dict[str, Any]:
        def scene_block(result: ConditionResult, scene: str | None) -> dict[str, Any]:
            return {
                "frames": result.frame_count(scene),
                "truth": result.truth_count(scene),
                "ap50": result.ap(scene),
                "ap25": result.ap(scene, iou=LOOSE_IOU),
                "ap_centre": result.ap(scene, criterion="centre"),
                "operating_points": [
                    result.point(threshold, scene).as_dict() for threshold in OPERATING_POINTS
                ],
                "centre_at_engine_threshold": result.point(
                    ENGINE_THRESHOLD, scene, criterion="centre"
                ).as_dict(),
            }

        return {
            "dataset": str(self.dataset),
            "detector": self.detector,
            "match_iou": MATCH_IOU,
            "loose_iou": LOOSE_IOU,
            "score_floor": SCORE_FLOOR,
            "started_at": self.started_at,
            **({"finetuned": self.finetuned.as_dict()} if self.finetuned is not None else {}),
            "conditions": [
                {
                    "condition": result.condition,
                    "split": result.split,
                    "input_size": result.input_size,
                    "ms_per_frame": round(result.ms_per_frame, 2),
                    **scene_block(result, None),
                    "recall_by_height": [
                        {"min_px": low, "max_px": None if high > 1e8 else high,
                         "found": found, "total": total}
                        for (low, high), found, total in result.recall_by_height()
                    ],
                    "scenes": {scene: scene_block(result, scene) for scene in result.scenes()},
                }
                for result in self.results
            ],
        }


def run_condition(
    detector: Any, frames: Iterable[LabelledFrame], condition: str, split: str,
    *, width: int, height: int, model_resolution: int = 384,
    progress: Callable[[str], None] | None = None,
) -> ConditionResult:
    """One pass of the detector over one split under one condition."""
    frames = list(frames)
    input_size = {
        "engine": f"{width}x{height}",
        "native": "native",
        "native-prefiltered": f"native -> {model_resolution}x{model_resolution}, filtered",
        "native-letterbox": "native, padded square",
    }.get(condition, condition)
    result = ConditionResult(condition=condition, split=split, input_size=input_size)

    elapsed = 0.0
    warmed = False
    for index, frame in enumerate(frames, start=1):
        image, back = prepare(
            frame, condition, width=width, height=height, model_resolution=model_resolution
        )
        if not warmed:
            # The first call on a new input shape pays torch.jit.trace, which is
            # seconds. Timing it would put that in every ms/frame figure.
            detector.detect([image])
            warmed = True
        started = time.perf_counter()
        detections = detector.detect([image])[0]
        elapsed += (time.perf_counter() - started) * 1000.0
        boxes = np.asarray(detections.xyxy, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(
            detections.confidence if detections.confidence is not None else [], dtype=np.float32
        )
        if len(boxes):
            boxes = back(boxes)
        # Person height as the *detector* sees it, whatever this condition did:
        # the decoded frame is what `detection.height` controls, so this is the
        # axis the fix is on.
        scale = height / frame.height
        truth_heights = (frame.boxes[:, 3] - frame.boxes[:, 1]) * scale
        result.frames.append(FrameResult(
            stem=frame.stem, scene=frame.scene, truth=frame.boxes,
            boxes=boxes, scores=scores, truth_heights=truth_heights,
        ))
        if progress is not None and (index % 20 == 0 or index == len(frames)):
            progress(f"  {condition}/{split}: {index}/{len(frames)} frames")
    result.ms_per_frame = elapsed / max(1, len(frames))
    return result


def run_detector_eval(
    config: Any,
    *,
    dataset: Path | None = None,
    splits: Sequence[str] = ("val", "train"),
    conditions: Sequence[str] = CONDITIONS,
    detector: Any = None,
    checkpoint: Path | str | None = None,
    progress: Callable[[str], None] | None = None,
) -> DetectorEvalRun:
    """Score the configured detector over every (split, condition) pair.

    The detector is built with a 0.01 threshold rather than the config's 0.5, so
    one pass carries the whole sweep. Nothing else about it is changed: same
    variant, same precision, same NMS IoU, same weights the engine loads.

    ``checkpoint`` adds a second pass by a detector carrying fine-tuned weights,
    over the same frames, the same conditions and the same matching, attached to
    the run as ``finetuned``. Two passes rather than two commands, because the
    thing being asked is a *difference*, and a difference measured across two
    invocations is a difference plus whatever else moved between them.
    """
    dataset = Path(dataset) if dataset is not None else DEFAULT_DATASET
    if not dataset.is_dir():
        raise DetectorEvalError(
            f"corpus not found at {dataset}. It is gitignored (customer footage); "
            f"place the export there or pass --dataset."
        )
    splits_frames = {split: load_split(dataset, split) for split in splits}

    if detector is None:
        detector = _eval_detector(config)

    def score(engine: Any, label: str) -> DetectorEvalRun:
        describe = getattr(engine, "describe", None)
        run = DetectorEvalRun(
            dataset=dataset,
            detector=describe() if callable(describe) else {"model": type(engine).__name__},
        )
        width, height = config.detection.width, config.detection.height
        resolution = int(getattr(engine, "resolution", 0) or 384)
        for split, frames in splits_frames.items():
            for condition in conditions:
                if progress is not None:
                    progress(f"{label}{condition} / {split}: {len(frames)} frames")
                run.results.append(run_condition(
                    engine, frames, condition, split,
                    width=width, height=height, model_resolution=resolution,
                    progress=progress,
                ))
        return run

    run = score(detector, "")
    if checkpoint is not None:
        # A second detector rather than a reloaded one: the baseline's traces and
        # weights stay exactly as they were scored, so nothing about the
        # comparison depends on the order the two ran in. Nano is 14 M
        # parameters; both fit on this box with room for the VLM.
        run.finetuned = score(_eval_detector(config, checkpoint=checkpoint), "finetuned ")
    return run


def _eval_detector(config: Any, *, checkpoint: Path | str | None = None) -> Any:
    """The engine's detector, at the 0.01 floor that makes one pass a sweep.

    ``verify_checkpoint=False``: the eval is precisely the step that runs
    *before* anything is promoted, so a run directory whose sidecar has not been
    matched against its weights yet is the normal case here. The engine does
    verify — see argus/vision/checkpoint.py.
    """
    from argus.vision.detector import detector_from_config

    detector = detector_from_config(config)
    if checkpoint is not None:
        detector.checkpoint = Path(checkpoint)
        detector.verify_checkpoint = False
    detector.threshold = SCORE_FLOOR
    detector.load()
    return detector


__all__ = [
    "CONDITIONS",
    "DEFAULT_DATASET",
    "ENGINE_THRESHOLD",
    "HEIGHT_BUCKETS",
    "LOOSE_IOU",
    "MATCH_IOU",
    "OPERATING_POINTS",
    "SCORE_FLOOR",
    "SHIPPED_CONDITION",
    "ConditionResult",
    "DetectorEvalError",
    "DetectorEvalRun",
    "FrameResult",
    "LabelledFrame",
    "OperatingPoint",
    "average_precision",
    "centre_matrix",
    "iou_matrix",
    "load_split",
    "match_frame",
    "operating_point",
    "prepare",
    "run_condition",
    "run_detector_eval",
    "scene_of",
]
