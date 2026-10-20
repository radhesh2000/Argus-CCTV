#!/usr/bin/env python
"""The student against the detector that is actually running, on four questions.

A checkpoint that scores better on a rewritten val split has proved almost
nothing: the split was rewritten to the convention the student was trained on,
which is a thumb on the scale worth naming out loud rather than burying in a
caveat. So this measures four different things, and they are allowed to
disagree:

**a. Harmonised val, 16 frames.** P/R/F1 at IoU 0.5 *and* at centre-in-box, by
box-height band, for both detectors down both inference paths. The production
Nano was fine-tuned on the OLD mixed labels, so at IoU 0.5 the harmonised
targets favour the student by construction — it agrees with the convention the
boxes were rewritten into. **Centre-in-box is the fairer cross-check**: it asks
only "is there a box on this person", which no convention change alters.

**b. Agreement with the teacher on the 85 heldout frames.** This is
*agreement with a stock-COCO teacher*, not accuracy. The teacher is wrong about
~4% of people outright and its far-field extents are untrustworthy
(docs/TEACHER_LABELS.md §7); a student that agrees with it perfectly would have
reproduced its errors perfectly. The number is here because disagreement is
informative and the heldout frames are the only unseen pixels available.

**c. Headcount through the real tracker.** The metric the product is for. Each
detector's boxes, at its `emit_floor`, through `argus.vision.tracker.CameraTracker`
with the shipped `tracker_settings_from_config`, over two clips at 5 fps, with
the same 3 s median (`statistics.median_low`, the function
`argus/vision/zone_state.py` uses) against the same hand count. **One hand count
per clip held constant**: read the difference between rows, never the absolute.

**d. Latency.** ms per frame, FP16, batch 1 and batch 8, per detector and path,
because the two-pass path is seven forward passes per frame and the appliance
runs eight cameras. Reported as a ratio against production on the engine path,
which is the only number that survives being measured beside a running
appliance.

**e. Foot-anchor error.** For every box matched to a harmonised truth box, how
far the box's BOTTOM edge sits from the truth's, in native pixels and as a
fraction of that person's height, per band. This is not a tie-breaker metric:
`argus/vision/zones.py:anchors` decides zone membership from the bottom-centre of
the box, so a detector whose boxes stop at a distant worker's shoulders puts
that worker in the wrong zone even when the headcount is right. The production
Nano was fine-tuned on labels that were head-only for distant workers
(docs/TEACHER_LABELS.md §2.5), so this is where a harmonised student should win
whether or not it wins anywhere else. Neither of the criteria the other tables
use can measure it — IoU 0.5 pre-selects the boxes whose extent already agrees,
and centre-in-box drops a head-only box entirely — so this one matches on the
**head** and leaves the bottom edge free. See `head_matrix`.

Two inference paths, everywhere:

* ``engine`` — the frame resized to 960x544 and then squashed to the model's
  square input, which is what production does today.
* ``two-pass`` — the native frame whole, plus the same 3x2 native tiles, merged
  with `scripts/teacher_label.py`'s seam-drop rule and one NMS at 0.55. Not a
  config switch: §2.5.

**Several students at once, each down the paths that make sense for it.**
`--students name=dir@path,path` puts as many checkpoints in one table as there
are questions to answer, and scores each only where it could ship: a student
trained without tiles has no two-pass row, and inventing one would be a
measurement of nothing. Every subject shares one decode of every frame and one
tracker per clip, so the only thing differing between two rows is the detector.

    .venv/bin/python scripts/student_eval.py --student models/finetuned/<run>
    .venv/bin/python scripts/student_eval.py --student <run> --parts latency
    .venv/bin/python scripts/student_eval.py \
        --students v1=models/finetuned/<a>@engine,two-pass \
                   parity=models/finetuned/<b>@engine,two-pass \
                   engineonly=models/finetuned/<c>@engine \
        --production-paths engine --val-exclude footage/leakage-val.json

Every process this starts caps itself with
``torch.cuda.set_per_process_memory_fraction`` — the box is shared and
`docs/DETECTION_V2.md` §2.1 is what happens when something forgets.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.teacher_label import MERGE_NMS_IOU, Teacher, merge  # noqa: E402

HARMONIZED = ROOT / "footage" / "anmol-harmonized"
HARVEST = ROOT / "footage" / "anmol-harvest"
TEACHER_LABELS = HARVEST / "labels_teacher"
FFMPEG = ROOT / "bin" / "ffmpeg"

# The engine's decode size (config/argus.yaml detection.width/height).
ENGINE_WIDTH, ENGINE_HEIGHT = 960, 544

# Detector operating point for the frame tables. The shipped
# `detection.model.threshold`; the tracker separately sees down to `track_floor`.
OPERATING_POINT = 0.50
# Everything above this is cached once, so a threshold sweep costs no inference.
# It is the detectors' own `emit_floor` (`detection.model.track_floor`): asking
# for boxes below what the detector emits would produce a sweep whose low rows
# were quietly truncated, and a table that looks like a measurement.
SCORE_FLOOR = 0.15
SWEEP: tuple[float, ...] = (0.30, 0.40, 0.50, 0.60)

BANDS: tuple[tuple[str, float, float], ...] = (
    ("far <60", 0.0, 60.0),
    ("mid 60-150", 60.0, 150.0),
    ("near >150", 150.0, 1e9),
)

MATCH_IOU = 0.5
# How far a prediction's top edge may sit from a truth box's, as a fraction of
# that person's height, and still be called a box on the same head. See
# `head_matrix` — this is the tolerance that lets a head-only box and a
# whole-person box on one worker be compared at the feet.
HEAD_TOP_TOLERANCE = 0.30
GPU_MEMORY_FRACTION = 0.16
MIN_AVAILABLE_GIB = 14.0

# The clips, the key each is reported under, the scene its hand labels live
# in, and the hand count — ONE number per clip, counted once by a human, so a
# difference between rows is evidence and an absolute is not.
#
# `None` means no human ever counted this clip and the reference is derived
# from the harmonised hand labels of its scene instead (`label_reference`).
# Packing 5 is here as an out-of-sample check: the tracker settings and the
# two hand counts above were chosen while looking at Packing 4 and Packing 7,
# and a detector that wins on those two and nowhere else has been fitted to
# them by the people choosing, not by gradient descent.
CLIPS: tuple[tuple[str, str, str, int | None], ...] = (
    ("Line_No.3_Packing4_Bas.MOV", "packing4", "Line_No.3_Packing4_Bas", 21),
    ("Line_No.3_Packing7_Bas.MP4", "packing7", "Line_No.3_Packing7_Bas", 25),
    ("Line_No.3_Packing5_Bas.MP4", "packing5", "Line_No.3_Packing5_Bas", None),
)
CLIP_FPS = 5.0
MEDIAN_WINDOW_SEC = 3.0


# --------------------------------------------------------------------------
# the shared box
# --------------------------------------------------------------------------


def claim_gpu_budget(fraction: float = GPU_MEMORY_FRACTION,
                     minimum_gib: float = MIN_AVAILABLE_GIB) -> dict[str, Any]:
    import torch

    available = 0.0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            available = int(line.split()[1]) / (1024 * 1024)
    if available < minimum_gib:
        raise SystemExit(
            f"MemAvailable is {available:.1f} GiB, below the {minimum_gib:.0f} GiB floor "
            f"(docs/DETECTION_V2.md §2.1)."
        )
    info = {"mem_available_gib": round(available, 1), "memory_fraction": fraction}
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction)
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        info["cap_gib"] = round(total * fraction, 1)
        print(f"MemAvailable {available:.1f} GiB · capped at {info['cap_gib']:.1f} GiB", flush=True)
    return info


# --------------------------------------------------------------------------
# the two detectors, and the two paths
# --------------------------------------------------------------------------


@dataclass
class Subject:
    """One detector seen through one inference path."""

    name: str
    path: str                     # "engine" | "two-pass"
    detector: Any
    teacher: Any = None           # the two-pass wrapper, or None
    label: str = ""

    def boxes(self, native: np.ndarray, engine_image: np.ndarray, threshold: float
              ) -> tuple[np.ndarray, np.ndarray]:
        """Boxes in NATIVE pixels, whichever path this is.

        Both paths return the same coordinate system on purpose: ground truth
        never moves (`argus/eval/detector.py: prepare`), so the only thing that
        differs between the rows of a table is what the detector was shown.
        """
        if self.path == "engine":
            detections = self.detector.detect([engine_image])[0]
            boxes = np.asarray(detections.xyxy, dtype=np.float32).reshape(-1, 4)
            scores = np.asarray(detections.confidence, dtype=np.float32).reshape(-1)
            keep = scores >= threshold
            boxes, scores = boxes[keep], scores[keep]
            sx = native.shape[1] / engine_image.shape[1]
            sy = native.shape[0] / engine_image.shape[0]
            boxes = boxes.copy()
            boxes[:, 0::2] *= sx
            boxes[:, 1::2] *= sy
            return boxes, scores
        raw = self.teacher.raw(native)
        return merge(raw, threshold=threshold, nms_iou=MERGE_NMS_IOU)


def parse_students(entries: Sequence[str], default_paths: Sequence[str]
                   ) -> list[tuple[str, Path, list[str]]]:
    """``name=dir`` or ``name=dir@engine,two-pass`` -> (name, dir, paths).

    The per-student path list is part of the spec rather than a global flag
    because it is a property of the checkpoint: a student trained on no tiles
    has no two-pass row to report, and a table that showed one anyway would be
    quoting a number for a pipeline nobody would build.
    """
    out: list[tuple[str, Path, list[str]]] = []
    for entry in entries:
        name, sep, rest = entry.partition("=")
        if not sep:
            raise SystemExit(f"--students wants name=dir[@paths], got {entry!r}")
        location, _, paths = rest.partition("@")
        chosen = [p.strip() for p in paths.split(",") if p.strip()] or list(default_paths)
        unknown = set(chosen) - {"engine", "two-pass"}
        if unknown:
            raise SystemExit(f"{entry!r}: unknown path(s) {sorted(unknown)}")
        out.append((name, Path(location), chosen))
    names = [name for name, _, _ in out]
    if len(set(names)) != len(names):
        raise SystemExit(f"two students cannot share a name: {names}")
    return out


def build_subjects(config: Any, students: Sequence[tuple[str, Path, list[str]]], *,
                   production_paths: Sequence[str], threshold: float, track_floor: float
                   ) -> list[Subject]:
    """Production and every student, each down its own paths, all FP16 here.

    The production detector is built the way `detector_from_config` builds it —
    same variant, resolution, precision, nms_iou, checkpoint and the digest
    check that goes with it — because the question is what the appliance is
    doing today, not what a similarly-configured model would do.

    The two-pass instances are separate objects with ``nms_iou=1.0``: the merge
    is the only place boxes may be suppressed on that path (see
    `scripts/teacher_label.py`), and a per-pass NMS inside it would be a
    different pipeline from the one that wrote the teacher labels.
    """
    from argus.vision.detector import PersonDetector

    model = config.detection.model
    subjects: list[Subject] = []
    specs: list[tuple[str, dict[str, Any], list[str]]] = [(
        "production", {
            "variant": model.variant, "precision": model.precision, "device": model.device,
            "resolution": model.resolution, "weights_dir": config.models_dir,
            "checkpoint": config.detector_checkpoint_path,
        }, list(production_paths),
    )]
    for name, student_dir, paths in students:
        provenance = json.loads((student_dir / "provenance.json").read_text())
        specs.append((
            name, {
                "variant": str(provenance["variant"]).removeprefix("rf-detr-"),
                "precision": model.precision, "device": model.device,
                "resolution": int(provenance["resolution"]),
                "weights_dir": config.models_dir,
                "checkpoint": student_dir / "checkpoint_best_total.pth",
            }, paths,
        ))

    for name, spec, paths in specs:
        for path in paths:
            detector = PersonDetector(
                **spec, threshold=threshold, track_floor=track_floor,
                nms_iou=1.0 if path == "two-pass" else model.nms_iou,
                trace=model.trace and path == "engine",
            ).load()
            teacher = Teacher(detector=detector) if path == "two-pass" else None
            subjects.append(Subject(
                name=name, path=path, detector=detector, teacher=teacher,
                label=f"{name}/{path}",
            ))
            print(f"  {name:<12} {path:<9} rf-detr-{detector.variant} @ {detector.resolution} "
                  f"{detector.precision}  emit_floor {detector.emit_floor:g}", flush=True)
    return subjects


# --------------------------------------------------------------------------
# matching and scoring (same algorithm as argus.eval.detector, on purpose)
# --------------------------------------------------------------------------


def head_matrix(predictions: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """(P, T): IoU where the prediction is drawn on the same *head*, else 0.

    Neither of the harness's two criteria can be used to measure how wrong a
    box's bottom edge is, and for opposite reasons. IoU 0.5 pre-selects the
    boxes whose extent already agrees, so it measures the error only where
    there isn't one. `centre_matrix` asks whether the **truth** box's centre —
    mid-torso — falls inside the prediction, so a head-only box on a standing
    worker does not match at all and never reaches the table.

    That second failure is fatal here, because a head-only box is precisely the
    thing being measured: docs/TEACHER_LABELS.md §2.5 found 88% of the sub-60 px
    hand labels were squarish heads, and the production Nano was fine-tuned on
    them. A diagnostic that silently drops every box exhibiting the defect
    would report that production has no foot-anchor error.

    So eligibility is the one thing a head-only box and a whole-person box on
    the same person still share: the prediction's horizontal centre inside the
    truth's x-range, and its **top** edge within 30% of the person's height of
    the truth's top edge. Both boxes start at the same head; only the bottom
    edge is in dispute, which is the measurement. Ranking stays IoU, as in
    `centre_matrix`, so the better-fitting box still wins a contested person.
    """
    from argus.eval.detector import iou_matrix

    if len(predictions) == 0 or len(truth) == 0:
        return np.zeros((len(predictions), len(truth)), dtype=np.float32)
    heights = np.maximum(truth[:, 3] - truth[:, 1], 1e-6)[None, :]
    centre_x = ((predictions[:, 0] + predictions[:, 2]) / 2)[:, None]
    eligible = (
        (centre_x >= truth[:, 0][None, :]) & (centre_x <= truth[:, 2][None, :])
        & (np.abs(predictions[:, 1][:, None] - truth[:, 1][None, :]) <= HEAD_TOP_TOLERANCE * heights)
    )
    return np.where(eligible, iou_matrix(predictions, truth), 0.0).astype(np.float32)


def match(boxes: np.ndarray, scores: np.ndarray, truth: np.ndarray, *, criterion: str
          ) -> tuple[np.ndarray, np.ndarray]:
    from argus.eval.detector import centre_matrix, iou_matrix

    if criterion == "centre":
        overlaps, floor = centre_matrix(boxes, truth), 1e-6
    elif criterion == "head":
        overlaps, floor = head_matrix(boxes, truth), 1e-6
    else:
        overlaps, floor = iou_matrix(boxes, truth), MATCH_IOU
    order = np.argsort(-scores)
    claimed = np.full(len(truth), -1.0)
    assigned = np.full(len(boxes), -1, dtype=int)
    for row in order:
        best, best_value = -1, floor
        for column in range(len(truth)):
            if claimed[column] >= 0:
                continue
            if overlaps[row, column] >= best_value:
                best, best_value = column, float(overlaps[row, column])
        if best >= 0:
            claimed[best] = float(scores[row])
            assigned[row] = best
    return assigned, claimed


@dataclass
class Tally:
    """One (subject, criterion) accumulated over a set of frames."""

    predictions: int = 0
    truth: int = 0
    tp: int = 0
    duplicates: int = 0
    bands: dict[str, list[int]] = field(default_factory=lambda: {n: [0, 0] for n, _, _ in BANDS})

    def add(self, boxes: np.ndarray, scores: np.ndarray, truth: np.ndarray, criterion: str) -> None:
        from argus.eval.detector import iou_matrix

        assigned, claimed = match(boxes, scores, truth, criterion=criterion)
        self.predictions += len(boxes)
        self.truth += len(truth)
        self.tp += int((assigned >= 0).sum())
        if len(boxes) and len(truth):
            overlaps = iou_matrix(boxes, truth)
            taken = claimed >= 0
            for index in range(len(boxes)):
                if assigned[index] < 0 and bool(np.any((overlaps[index] >= MATCH_IOU) & taken)):
                    self.duplicates += 1
        heights = (truth[:, 3] - truth[:, 1]) if len(truth) else np.zeros(0)
        for name, low, high in BANDS:
            inside = (heights >= low) & (heights < high)
            self.bands[name][1] += int(inside.sum())
            self.bands[name][0] += int((claimed[inside] >= 0).sum())

    def as_dict(self) -> dict[str, Any]:
        precision = self.tp / self.predictions if self.predictions else None
        recall = self.tp / self.truth if self.truth else None
        f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
        return {
            "predictions": self.predictions, "truth": self.truth, "tp": self.tp,
            "precision": precision, "recall": recall, "f1": f1,
            "duplicates": self.duplicates,
            "bands": {name: {"found": found, "total": total} for name, (found, total) in self.bands.items()},
        }


# --------------------------------------------------------------------------
# a. the harmonised val frames   /   b. teacher agreement on heldout
# --------------------------------------------------------------------------


def _engine_view(native: np.ndarray) -> np.ndarray:
    from argus.eval.detector import _resize

    return _resize(native, ENGINE_WIDTH, ENGINE_HEIGHT)


Cache = dict[str, list[tuple[np.ndarray, np.ndarray]]]


def detect_all(subjects: Sequence[Subject], frames: Sequence[Any]) -> Cache:
    """Every subject's boxes on every frame, once, down to ``SCORE_FLOOR``.

    One pass, then every table below is arithmetic on it: the threshold sweep,
    the all-frames and leak-free val subsets, and the foot-anchor diagnostic
    are all the same detections counted differently, which is the only way they
    can be guaranteed to describe the same run.
    """
    from argus.eval.detector import _load_bgr

    cache: Cache = {subject.label: [] for subject in subjects}
    for index, frame in enumerate(frames, start=1):
        native = _load_bgr(frame.path)
        engine = _engine_view(native)
        for subject in subjects:
            cache[subject.label].append(subject.boxes(native, engine, SCORE_FLOOR))
        if index % 20 == 0 or index == len(frames):
            print(f"    {index}/{len(frames)} frames", flush=True)
    return cache


def tally_frames(subjects: Sequence[Subject], frames: Sequence[Any], cache: Cache, *,
                 thresholds: Sequence[float], keep: set[str] | None = None) -> dict[str, Any]:
    """The P/R/F1 tables over ``frames``, or the subset of them in ``keep``."""
    results: dict[str, Any] = {}
    for subject in subjects:
        per_threshold: dict[str, Any] = {}
        for threshold in thresholds:
            tallies = {"iou": Tally(), "centre": Tally()}
            for frame, (boxes, scores) in zip(frames, cache[subject.label]):
                if keep is not None and frame.stem not in keep:
                    continue
                selected = scores >= threshold
                for criterion, tally in tallies.items():
                    tally.add(boxes[selected], scores[selected], frame.boxes, criterion)
            per_threshold[f"{threshold:.2f}"] = {k: v.as_dict() for k, v in tallies.items()}
        results[subject.label] = per_threshold
    return results


def score_frames(subjects: Sequence[Subject], frames: Sequence[Any], *,
                 thresholds: Sequence[float]) -> dict[str, Any]:
    """Detect once, tally once — the whole of what part b needs."""
    return tally_frames(subjects, frames, detect_all(subjects, frames), thresholds=thresholds)


# --------------------------------------------------------------------------
# e. where the feet are
# --------------------------------------------------------------------------


def foot_anchor(subjects: Sequence[Subject], frames: Sequence[Any], cache: Cache, *,
                threshold: float, keep: set[str] | None = None) -> dict[str, Any]:
    """Vertical error of the box BOTTOM edge, per band, for matched boxes.

    Signed, and the sign is the interesting half: a *negative* median means the
    detector's boxes stop short of the person's feet, which is the head-only
    label convention showing through, and it moves the zone anchor up the image
    — towards the camera-far side of whatever line a zone is drawn on.

    Reported as a median rather than a mean because a single box that latched
    onto a person's reflection contributes an error of several hundred pixels
    and would decide the number on its own.

    Matched by `head_matrix`, which is the only one of the three criteria that
    can see the defect — see its docstring.
    """
    results: dict[str, Any] = {}
    for subject in subjects:
        per_band: dict[str, list[tuple[float, float]]] = {name: [] for name, _, _ in BANDS}
        for frame, (boxes, scores) in zip(frames, cache[subject.label]):
            if keep is not None and frame.stem not in keep:
                continue
            selected = scores >= threshold
            chosen, chosen_scores = boxes[selected], scores[selected]
            assigned, _ = match(chosen, chosen_scores, frame.boxes, criterion="head")
            for index, truth_index in enumerate(assigned):
                if truth_index < 0:
                    continue
                truth = frame.boxes[truth_index]
                height = float(truth[3] - truth[1])
                if height <= 0:
                    continue
                error = float(chosen[index][3] - truth[3])
                for name, low, high in BANDS:
                    if low <= height < high:
                        per_band[name].append((error, error / height))
                        break
        results[subject.label] = {
            name: {
                "n": len(pairs),
                "median_px": round(float(np.median([e for e, _ in pairs])), 1) if pairs else None,
                "median_abs_px": round(float(np.median([abs(e) for e, _ in pairs])), 1) if pairs else None,
                "median_frac": round(float(np.median([f for _, f in pairs])), 3) if pairs else None,
                "median_abs_frac": round(float(np.median([abs(f) for _, f in pairs])), 3) if pairs else None,
            }
            for name, pairs in per_band.items()
        }
    return results


@dataclass(slots=True)
class PseudoFrame:
    """A harvested frame and the teacher's boxes on it, shaped like a LabelledFrame."""

    path: Path
    scene: str
    width: int
    height: int
    boxes: np.ndarray

    @property
    def stem(self) -> str:
        return self.path.stem


def heldout_frames(harvest: Path, labels: Path, threshold: float) -> list[PseudoFrame]:
    manifest = json.loads((harvest / "manifest.json").read_text())
    frames: list[PseudoFrame] = []
    for entry in manifest["frames"]:
        if entry["split"] != "heldout":
            continue
        scene, name = str(entry["file"]).split("/", 1)
        payload = json.loads((labels / scene / f"{Path(name).stem}.json").read_text())
        kept = [b["xyxy"] for b in payload["boxes"] if float(b["score"]) >= threshold]
        frames.append(PseudoFrame(
            path=harvest / str(entry["file"]), scene=scene,
            width=int(payload["width"]), height=int(payload["height"]),
            boxes=np.asarray(kept, dtype=np.float32).reshape(-1, 4),
        ))
    return frames


# --------------------------------------------------------------------------
# c. headcount through the real tracker
# --------------------------------------------------------------------------


def label_reference(harmonized: Path, scene: str) -> dict[str, Any]:
    """A reference count for a clip nobody counted, off its own hand labels.

    **This is a weak reference and it is weak in a knowable direction.** It is
    the median of how many people were labelled on a handful of still frames
    from the clip, so it inherits everything the labels inherit: the
    harmonisation added 38 teacher-only boxes the humans never drew, and a
    frame is not a shift. Applied to the two clips a human *did* count, the
    same derivation lands one person high on both — which is the only
    calibration available and is reported beside the number.
    """
    from argus.eval.detector import scene_of

    counts: list[int] = []
    frames: list[str] = []
    for split in ("train", "val"):
        for label in sorted((harmonized / "labels" / split).glob("*.txt")):
            if scene_of(label.stem) != scene:
                continue
            counts.append(len([line for line in label.read_text().splitlines() if line.strip()]))
            frames.append(label.stem)
    if not counts:
        raise SystemExit(f"no harmonised labels for scene {scene!r} under {harmonized}")
    return {
        "scene": scene,
        "labelled_frames": len(counts),
        "counts": counts,
        # median_low, because that is the statistic the live headcount smooths
        # with (`argus/vision/zone_state.py`) and the one the ±1/±2 columns are
        # measured against.
        "median_low": int(statistics.median_low(counts)),
        "mean": round(float(np.mean(counts)), 1),
        "min": min(counts),
        "max": max(counts),
    }


def stream_native(clip: Path, fps: float, width: int, height: int) -> Iterator[np.ndarray]:
    """Native BGR frames at ``fps``, straight out of ffmpeg.

    The same decode `scripts/harvest_frames.py` uses, for the same reason: the
    two-pass path needs the pixels the engine throws away, so the clip is
    decoded once at full size and the 960x544 engine view is derived from it in
    process. One decode, both paths, identical frames — which is the only way
    the two rows of the table differ by the detector alone.
    """
    command = [
        str(FFMPEG), "-v", "error", "-i", str(clip),
        "-vf", f"fps={fps}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    frame_bytes = width * height * 3
    with open("/dev/null", "wb") as devnull:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=devnull)
        assert process.stdout is not None
        try:
            while True:
                raw = process.stdout.read(frame_bytes)
                if len(raw) < frame_bytes:
                    break
                yield np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
        finally:
            process.stdout.close()
            process.wait()


def probe(clip: Path) -> tuple[int, int]:
    result = subprocess.run(
        [str(ROOT / "bin" / "ffprobe"), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", str(clip)],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    return int(stream["width"]), int(stream["height"])


def headcount(subjects: Sequence[Subject], clip: Path, key: str, hand_count: int,
              settings: Any, *, fps: float = CLIP_FPS) -> dict[str, Any]:
    """Every subject's boxes through its own tracker, over one clip.

    One tracker per subject per clip, never shared: `argus/vision/tracker.py`
    says why per camera, and the same argument applies to two detectors whose
    boxes would otherwise associate with each other's tracks.

    The boxes handed to the tracker are always in the engine's 960x544 space,
    including on the two-pass path. That is not a detail: ByteTrack's Kalman
    filter and its lost-track buffer work in pixels, and feeding one path
    2560x1440 boxes and the other 960x544 ones would change the tracker's
    behaviour as well as the detector's — two variables, one row.
    """
    import supervision as sv
    from argus.vision.tracker import CameraTracker

    width, height = probe(clip)
    trackers = {s.label: CameraTracker(f"{key}:{s.label}", fps, settings) for s in subjects}
    counts: dict[str, list[int]] = {s.label: [] for s in subjects}
    identities: dict[str, set[int]] = {s.label: set() for s in subjects}
    scale_x, scale_y = ENGINE_WIDTH / width, ENGINE_HEIGHT / height

    frames = 0
    for native in stream_native(clip, fps, width, height):
        native = np.ascontiguousarray(native)
        engine = _engine_view(native)
        frames += 1
        for subject in subjects:
            floor = subject.detector.emit_floor
            boxes, scores = subject.boxes(native, engine, floor)
            if len(boxes):
                # `Subject.boxes` answers in native pixels whichever path it is;
                # the tracker is given the engine's space for both.
                boxes = boxes.copy()
                boxes[:, 0::2] *= scale_x
                boxes[:, 1::2] *= scale_y
            tracked = trackers[subject.label].update(sv.Detections(
                xyxy=boxes.astype(np.float32).reshape(-1, 4),
                confidence=scores.astype(np.float32).reshape(-1),
                class_id=np.zeros(len(boxes), dtype=int),
            ))
            counts[subject.label].append(len(tracked))
            if tracked.tracker_id is not None:
                identities[subject.label].update(int(i) for i in tracked.tracker_id)
        if frames % 50 == 0:
            print(f"    {key} {frames} frames", flush=True)

    window = max(1, int(round(MEDIAN_WINDOW_SEC * fps)))
    out: dict[str, Any] = {"clip": clip.name, "hand_count": hand_count, "frames": frames,
                           "fps": fps, "median_window_frames": window, "by_subject": {}}
    for subject in subjects:
        series = counts[subject.label]
        # statistics.median_low, over a trailing window_sec of samples — the
        # function and the window argus/vision/zone_state.py uses, so this is the
        # number the live headcount would have worked from.
        smoothed = [statistics.median_low(series[max(0, i - window + 1):i + 1])
                    for i in range(len(series))]
        out["by_subject"][subject.label] = {
            "mean_count": round(float(np.mean(series)), 2) if series else None,
            "mean_smoothed": round(float(np.mean(smoothed)), 2) if smoothed else None,
            "within_1": round(sum(abs(v - hand_count) <= 1 for v in smoothed) / max(len(smoothed), 1), 3),
            "within_2": round(sum(abs(v - hand_count) <= 2 for v in smoothed) / max(len(smoothed), 1), 3),
            "identities": len(identities[subject.label]),
            "median_smoothed": int(statistics.median_low(smoothed)) if smoothed else None,
        }
    return out


# --------------------------------------------------------------------------
# d. latency
# --------------------------------------------------------------------------


def latency(subjects: Sequence[Subject], sample: Path, *, warmup: int = 5, runs: int = 20
            ) -> dict[str, Any]:
    """ms per FRAME at batch 1 and batch 8, per subject.

    "Per frame" means per camera-frame, not per forward pass, which is the whole
    point of measuring the two-pass path: one frame there is seven views.

    * engine, batch 1 — one 960x544 frame per forward.
    * engine, batch 8 — eight frames coalesced, as `detection.model.max_batch`
      does across cameras; divided by 8.
    * two-pass, batch 1 — the seven views forwarded one at a time, summed.
    * two-pass, batch 8 — the same seven views in a single forward, which is
      what the appliance's batch budget would allow for ONE camera.
    """
    import torch
    from argus.eval.detector import _load_bgr
    from scripts.teacher_label import plan_tiles

    native = _load_bgr(sample)
    engine = _engine_view(native)
    tiles = plan_tiles(native.shape[1], native.shape[0])
    views = [native] + [np.ascontiguousarray(native[t.y0:t.y1, t.x0:t.x1]) for t in tiles]

    def timed(detector: Any, batches: Sequence[Sequence[np.ndarray]]) -> float:
        for _ in range(warmup):
            for batch in batches:
                detector.detect(list(batch))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(runs):
            for batch in batches:
                detector.detect(list(batch))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return (time.perf_counter() - started) * 1000.0 / runs

    out: dict[str, Any] = {}
    for subject in subjects:
        if subject.path == "engine":
            single = timed(subject.detector, [[engine]])
            eight = timed(subject.detector, [[engine] * 8]) / 8
        else:
            single = timed(subject.detector, [[view] for view in views])
            eight = timed(subject.detector, [views[:8]])
        out[subject.label] = {"ms_per_frame_batch1": round(single, 2),
                              "ms_per_frame_batch8": round(eight, 2),
                              "views_per_frame": 1 if subject.path == "engine" else len(views)}
        print(f"  {subject.label:<22} batch1 {single:6.2f} ms   batch8 {eight:6.2f} ms/frame", flush=True)
    return out


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------


def _f(value: Any, places: int = 3) -> str:
    return "–" if value is None else f"{float(value):.{places}f}"


def markdown(report: dict[str, Any]) -> str:
    """The report as the tables that go in docs/STUDENT_DETECTOR.md.

    Rendered from the JSON rather than transcribed into the document by hand:
    a number that is typed twice is a number that will eventually disagree with
    itself, and the one in the document is the one people quote.
    """
    out: list[str] = []
    for key, caption in (("val", "every val frame"),
                         ("val_clean", "leak-free val frames only")):
        if key not in report:
            continue
        point = f"{report['operating_point']:.2f}"
        # Read the denominator off the report rather than writing it in: a
        # header that says "16 frames" beside a table built from three is the
        # kind of error nobody re-checks.
        first = next(iter(report[key].values()))[point]["iou"]
        frames = report.get("val_frames", {}).get(key)
        scope = (f"{frames} frames, " if frames else "") + f"{first['truth']} harmonised boxes"
        for criterion, title in (("iou", "IoU 0.5"), ("centre", "centre-in-box")):
            out.append(f"\n**Harmonised val — {caption} ({scope}), score >= {point}, {title}**\n")
            out.append("| detector · path | P | R | F1 | dup | boxes | far <60 | mid 60-150 | near >150 |")
            out.append("|---|---|---|---|---|---|---|---|---|")
            for label in sorted(report[key]):
                block = report[key][label][point][criterion]
                bands = " | ".join(
                    f"{block['bands'][name]['found']}/{block['bands'][name]['total']}"
                    for name, _, _ in BANDS
                )
                out.append(
                    f"| {label} | {_f(block['precision'])} | {_f(block['recall'])} "
                    f"| {_f(block['f1'])} | {block['duplicates']} | {block['predictions']} | {bands} |"
                )

        out.append(f"\n**Threshold sweep — {caption}, IoU 0.5 (P / R / F1)**\n")
        out.append("| detector · path | " + " | ".join(f"{t:.2f}" for t in SWEEP) + " |")
        out.append("|---|" + "---|" * len(SWEEP))
        for label in sorted(report[key]):
            cells = []
            for threshold in SWEEP:
                block = report[key][label][f"{threshold:.2f}"]["iou"]
                cells.append(f"{_f(block['precision'], 2)} / {_f(block['recall'], 2)} / {_f(block['f1'], 2)}")
            out.append(f"| {label} | " + " | ".join(cells) + " |")

    if "foot_anchor" in report:
        out.append(f"\n**Foot-anchor error — box bottom edge minus truth bottom edge, over "
                   f"boxes drawn on the same head as a truth box, at score >= "
                   f"{report['operating_point']:.2f}. Negative = the box stops short of the "
                   f"feet, which moves the zone anchor up the image.**\n")
        out.append("| detector · path | " + " | ".join(
            f"{name} px (frac), n" for name, _, _ in BANDS) + " |")
        out.append("|---|" + "---|" * len(BANDS))
        for label in sorted(report["foot_anchor"]):
            cells = []
            for name, _, _ in BANDS:
                band = report["foot_anchor"][label][name]
                cells.append(
                    "–" if band["n"] == 0 else
                    f"{band['median_px']:+.0f} ({band['median_frac']:+.2f}), n={band['n']}"
                )
            out.append(f"| {label} | " + " | ".join(cells) + " |")

    if "heldout" in report:
        point = f"{report['operating_point']:.2f}"
        reference = next(iter(report["heldout"].values()))[point]["iou"]["truth"]
        out.append(f"\n**Agreement with the teacher, {reference} teacher boxes on the "
                   f"heldout frames — NOT accuracy**\n")
        out.append("| detector · path | agree P (IoU.5) | agree R | P (centre) | R (centre) | boxes |")
        out.append("|---|---|---|---|---|---|")
        for label in sorted(report["heldout"]):
            iou = report["heldout"][label][point]["iou"]
            ctr = report["heldout"][label][point]["centre"]
            out.append(f"| {label} | {_f(iou['precision'])} | {_f(iou['recall'])} "
                       f"| {_f(ctr['precision'])} | {_f(ctr['recall'])} | {iou['predictions']} |")

    if "headcount" in report:
        for key, block in report["headcount"].items():
            source = block.get("reference_source", "hand count")
            out.append(f"\n**Headcount — {block['clip']}, {source} ~{block['hand_count']}, "
                       f"{block['frames']} frames at {block['fps']:g} fps**\n")
            if block.get("reference_detail"):
                detail = block["reference_detail"]
                out.append(f"*Reference derived from {detail['labelled_frames']} harmonised "
                           f"hand-labelled frames of this scene: median {detail['median_low']}, "
                           f"mean {detail['mean']}, range {detail['min']}–{detail['max']}. "
                           f"A weak reference — see the caveat.*\n")
            out.append("| detector · path | mean | 3 s-median mean | ±1 | ±2 | identities |")
            out.append("|---|---|---|---|---|---|")
            for label in sorted(block["by_subject"]):
                row = block["by_subject"][label]
                out.append(
                    f"| {label} | {_f(row['mean_count'], 1)} | {_f(row['mean_smoothed'], 1)} "
                    f"| {row['within_1'] * 100:.1f}% | {row['within_2'] * 100:.1f}% | {row['identities']} |"
                )

    if "latency" in report:
        # Against production on the engine path, because the absolutes are
        # inflated by whatever else the box is doing and the ratio is not.
        base = report["latency"].get("production/engine")
        out.append("\n**Latency, FP16, this GB10 — ms per camera-frame, and x production "
                   "on the engine path**\n")
        out.append("| detector · path | views/frame | batch 1 | x prod | batch 8 | x prod |")
        out.append("|---|---|---|---|---|---|")
        for label in sorted(report["latency"]):
            row = report["latency"][label]
            ratios = [
                "–" if not base else f"{row[key] / base[key]:.2f}x"
                for key in ("ms_per_frame_batch1", "ms_per_frame_batch8")
            ]
            out.append(f"| {label} | {row['views_per_frame']} | {row['ms_per_frame_batch1']:.2f} ms "
                       f"| {ratios[0]} | {row['ms_per_frame_batch8']:.2f} ms | {ratios[1]} |")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--student", type=Path, default=None,
                        help="a models/finetuned/<run> directory with provenance.json "
                             "(shorthand for --students student=<dir>)")
    parser.add_argument("--students", nargs="*", default=[],
                        help="name=dir, or name=dir@engine,two-pass to restrict the paths a "
                             "checkpoint is scored on")
    parser.add_argument("--production-paths", nargs="*", default=None,
                        help="paths for the shipped detector (default: --paths)")
    parser.add_argument("--harmonized", type=Path, default=HARMONIZED)
    parser.add_argument("--harvest", type=Path, default=HARVEST)
    parser.add_argument("--teacher-labels", type=Path, default=TEACHER_LABELS)
    parser.add_argument("--teacher-threshold", type=float, default=0.30,
                        help="the threshold the teacher's shipped labels were written at")
    parser.add_argument("--paths", nargs="*", default=["engine", "two-pass"])
    parser.add_argument("--val-exclude", type=Path, default=None,
                        help="a scripts/leakage_check.py --json report; val is then reported "
                             "twice, all frames and only the ones it did not flag")
    parser.add_argument("--clips", nargs="*", default=None,
                        help="which headcount clips to run, by key (default: all)")
    parser.add_argument("--parts", nargs="*",
                        default=["val", "heldout", "headcount", "latency"],
                        help="val, heldout, headcount, latency (foot-anchor comes with val)")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after N frames per part (a smoke test, never a result)")
    parser.add_argument("--report", type=Path, default=ROOT / "footage" / "student-eval.json")
    parser.add_argument("--markdown", type=Path, default=None,
                        help="also write the tables, so the document quotes the JSON rather "
                             "than a transcription of it")
    parser.add_argument("--from-report", type=Path, default=None,
                        help="re-render the tables from an existing report and exit (no GPU)")
    args = parser.parse_args()

    if args.from_report is not None:
        text = markdown(json.loads(args.from_report.read_text()))
        if args.markdown:
            args.markdown.write_text(text)
        print(text)
        return 0

    from argus.config import load_config
    from argus.vision.tracker import tracker_settings_from_config

    config = load_config()
    budget = claim_gpu_budget()

    entries = list(args.students)
    if args.student is not None:
        entries.insert(0, f"student={args.student}")
    if not entries:
        raise SystemExit("nothing to compare — pass --student or --students name=dir")
    students = parse_students(entries, args.paths)

    print("\ndetectors:")
    subjects = build_subjects(
        config, students, production_paths=args.production_paths or args.paths,
        threshold=config.detection.model.threshold,
        track_floor=config.detection.model.track_floor,
    )

    def describe(directory: Path) -> dict[str, Any]:
        provenance = json.loads((directory / "provenance.json").read_text())
        return {
            "dir": str(directory.relative_to(ROOT)) if directory.is_relative_to(ROOT) else str(directory),
            "variant": provenance["variant"],
            "resolution": provenance["resolution"],
            "dataset_digest": provenance["dataset"]["digest"],
            "epochs_run": provenance["training"]["epochs_run"],
            "wall_clock_sec": provenance["training"]["wall_clock_sec"],
            "peak_gpu_reserved_gib": provenance["training"].get("gpu_budget", {}).get(
                "peak_reserved_gib"),
            "best_epoch": provenance["training"].get("best", {}).get("epoch"),
        }

    report: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "scripts/student_eval.py",
        "gpu_budget": budget,
        "students": {
            name: {**describe(directory), "paths": paths} for name, directory, paths in students
        },
        "production": {
            "variant": config.detection.model.variant,
            "resolution": config.detection.model.resolution,
            "checkpoint": str(config.detector_checkpoint_path),
            "threshold": config.detection.model.threshold,
            "track_floor": config.detection.model.track_floor,
        },
        "operating_point": OPERATING_POINT,
        "caveats": {
            "val": (
                "The harmonised val labels were rewritten to the whole-visible-person "
                "convention the student was trained on. At IoU 0.5 this favours the student "
                "on box EXTENT by construction. Centre-in-box is the fairer cross-check: it "
                "asks only whether a box is on the person, which the convention change does "
                "not touch. 16 frames / ~11 correlated people per frame -- DETECTION_V2 §2.6 "
                "puts the effective n at 30-45."
            ),
            "heldout": (
                "AGREEMENT WITH A TEACHER, NOT ACCURACY. The reference boxes are RF-DETR "
                "Large's pseudo-labels; the teacher misses ~4% of people outright and its "
                "far-field extents are untrustworthy. The heldout frames are also a TIME "
                "split inside the same 60 s clips as the training frames, so they share the "
                "lighting, pose, shift and cast -- an upper bound on a different day."
            ),
            "headcount": (
                "ONE hand count per clip, held constant, on two 60 s clips; the third clip's "
                "reference is derived from that scene's harmonised hand labels and is weaker "
                "still. Report differences between rows as directional. This cannot support a "
                "percentage claim about the product."
            ),
            "foot_anchor": (
                "Matched by centre-in-box, so the sample is every box that landed on a person "
                "rather than every box that already agreed about extent. The truth extents are "
                "themselves part teacher: 100 of the 1053 harmonised boxes took their bottom "
                "edge from RF-DETR Large, so a student trained on that teacher is being scored "
                "partly against its own source."
            ),
        },
    }

    if "val" in args.parts:
        from argus.eval.detector import load_split

        frames = load_split(args.harmonized, "val")
        if args.limit:
            frames = frames[:args.limit]
        print(f"\na. harmonised val — {len(frames)} frames, "
              f"{sum(len(f.boxes) for f in frames)} boxes", flush=True)
        cache = detect_all(subjects, frames)
        report["val"] = tally_frames(subjects, frames, cache, thresholds=SWEEP)
        report["val_frames"] = {"val": len(frames)}

        clean: set[str] | None = None
        if args.val_exclude is not None:
            flagged = {
                row["frame"] for row in json.loads(args.val_exclude.read_text())["rows"]
                if row["split"] == "val" and row["verdict"] in ("duplicate", "near-duplicate")
            }
            clean = {frame.stem for frame in frames if frame.stem not in flagged}
            print(f"    leak-free subset: {len(clean)} of {len(frames)} frames "
                  f"({len(flagged)} flagged by {args.val_exclude})", flush=True)
            report["val_clean"] = tally_frames(subjects, frames, cache, thresholds=SWEEP,
                                               keep=clean)
            report["val_frames"]["val_clean"] = len(clean)
            report["val_flagged_frames"] = sorted(flagged)

        print("\ne. foot-anchor error (box bottom vs truth bottom)", flush=True)
        report["foot_anchor"] = foot_anchor(subjects, frames, cache, threshold=OPERATING_POINT)
        del cache

    if "heldout" in args.parts:
        frames = heldout_frames(args.harvest, args.teacher_labels, args.teacher_threshold)
        if args.limit:
            frames = frames[:args.limit]
        print(f"\nb. teacher agreement on heldout — {len(frames)} frames, "
              f"{sum(len(f.boxes) for f in frames)} teacher boxes", flush=True)
        report["heldout"] = score_frames(subjects, frames, thresholds=(OPERATING_POINT,))

    if "headcount" in args.parts:
        settings = tracker_settings_from_config(config)
        report["tracker_settings"] = {
            "activation_threshold": settings.activation_threshold,
            "match_threshold": settings.match_threshold,
            "lost_track_sec": settings.lost_track_sec,
            "min_consecutive_frames": settings.min_consecutive_frames,
            "median_window_sec": MEDIAN_WINDOW_SEC,
        }
        report["headcount"] = {}
        for filename, key, scene, hand in CLIPS:
            if args.clips is not None and key not in args.clips:
                continue
            clip = ROOT / "footage" / "anmol-samples" / filename
            # Every clip's labels are summarised, not only the one that needs
            # it: the two with hand counts are what calibrates how far the
            # derived reference can be trusted on the one without.
            detail = label_reference(args.harmonized, scene)
            reference = hand if hand is not None else detail["median_low"]
            source = "hand count" if hand is not None else "label-derived reference"
            print(f"\nc. headcount — {filename} ({source} ~{reference}; "
                  f"labels on {detail['labelled_frames']} frames say median "
                  f"{detail['median_low']} / mean {detail['mean']})", flush=True)
            block = headcount(subjects, clip, key, reference, settings)
            block["reference_source"] = source
            block["reference_detail"] = detail if hand is None else None
            block["label_reference"] = detail
            report["headcount"][key] = block

    if "latency" in args.parts:
        sample = ROOT / "footage" / "anmol-harvest" / "Line_No.3_Packing4_Bas" / "t000.jpg"
        print("\nd. latency (FP16, this GPU):", flush=True)
        report["latency"] = latency(subjects, sample)

    import torch
    if torch.cuda.is_available():
        report["peak_gpu_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 1024**3, 2)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=float) + "\n")
    tables = markdown(report)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(tables)
    print(tables)
    print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
