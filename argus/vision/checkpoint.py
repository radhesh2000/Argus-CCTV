"""Reading a fine-tuned detector checkpoint's provenance, and refusing it without one.

`scripts/train_detector.py` can produce weights trained on a site's own labelled
frames. `detection.model.checkpoint` is the config line that loads them. This is
the gate between the two, and it is deliberately strict: **a checkpoint the
appliance cannot describe is a checkpoint the appliance will not load.**

That is not ceremony. Three facts decide whether a `.pth` on this disk is safe
to put in front of a headcount policy, and none of them is visible in the file's
name:

* **which logit is person.** RF-DETR's COCO head is indexed by raw COCO
  category id, so `PersonDetector._postprocess` reads slot 1. A fine-tune
  trained single-class puts person on slot 0 — the same weights, the same
  shape, silently counting nobody. There is no way to tell from the tensor.
* **what it was trained on.** Weights fitted to one plant's cameras are a
  liability on another's, and "which plant" is not a property of the file.
* **whether it is the file that was measured.** The number an operator decided
  on came from a specific set of weights; a checkpoint that no longer hashes to
  that is a different model wearing its evaluation.

So a checkpoint is loaded only alongside the `provenance.json` its training run
wrote next to it. Absent, malformed, or hashing to something else, the engine
refuses to start rather than starting wrong — the same choice the decoder makes
when a camera's codec is not what was enrolled.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The sidecar `scripts/train_detector.py` writes into every run directory.
PROVENANCE_NAME = "provenance.json"

# Hashing 120 MB costs ~0.2 s once at startup. Read in chunks rather than whole
# so the appliance never holds a second copy of the weights in RAM to check the
# first one.
_CHUNK = 1 << 20


class CheckpointError(RuntimeError):
    """The configured checkpoint cannot be trusted, and so will not be loaded."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(slots=True, frozen=True)
class CheckpointProvenance:
    """What the appliance knows about a set of fine-tuned weights.

    Frozen because it is read once at load and then quoted — in the engine's
    startup banner, in `argus status`, and in `describe()`. A value that can be
    edited after the fact is a value that can disagree with the weights.
    """

    path: Path
    person_class_index: int
    num_classes: int
    sha256: str | None = None
    created_at: str | None = None
    dataset_digest: str | None = None
    dataset_images: int = 0
    dataset_boxes: int = 0
    epochs_run: int = 0
    best_epoch: int | None = None
    val_map_50: float | None = None
    val_map_50_95: float | None = None
    variant: str = ""
    resolution: int = 0
    # What the run cost. Quoted in docs/DETECTOR_EVAL.md so the next site reads
    # a measured number rather than "a few minutes".
    wall_clock_sec: float | None = None

    def summary(self) -> str:
        """One line, for a log an operator reads at 3 a.m.

        Says where the weights came from and what they scored, because the two
        questions asked of a non-default model are always "whose is this" and
        "was it any better".
        """
        bits = [f"{self.path.name}"]
        if self.variant:
            bits.append(self.variant)
        if self.created_at:
            bits.append(f"trained {self.created_at}")
        if self.dataset_images:
            bits.append(f"{self.dataset_images} labelled frames/{self.dataset_boxes} boxes")
        if self.dataset_digest:
            bits.append(f"corpus {self.dataset_digest[:12]}")
        if self.epochs_run:
            bits.append(f"{self.epochs_run} epochs")
        if self.val_map_50 is not None:
            bits.append(f"val mAP@50 {self.val_map_50:.3f}")
        bits.append(f"person=logit {self.person_class_index}")
        return " — ".join([bits[0], ", ".join(bits[1:])]) if len(bits) > 1 else bits[0]

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "person_class_index": self.person_class_index,
            "num_classes": self.num_classes,
            "created_at": self.created_at,
            "dataset_digest": self.dataset_digest,
            "dataset_images": self.dataset_images,
            "dataset_boxes": self.dataset_boxes,
            "epochs_run": self.epochs_run,
            "best_epoch": self.best_epoch,
            "val_map_50": self.val_map_50,
            "val_map_50_95": self.val_map_50_95,
            "variant": self.variant,
            "resolution": self.resolution,
            "wall_clock_sec": self.wall_clock_sec,
        }


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def load_provenance(checkpoint: Path, *, verify_digest: bool = True) -> CheckpointProvenance:
    """Read the sidecar beside ``checkpoint``, or raise saying exactly what is wrong.

    ``verify_digest=False`` exists for the eval harness, which may be pointed at
    a checkpoint mid-experiment before anything has been promoted; the engine
    never passes it.
    """
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise CheckpointError(f"detector checkpoint not found: {checkpoint}")

    sidecar = checkpoint.parent / PROVENANCE_NAME
    if not sidecar.is_file():
        raise CheckpointError(
            f"{checkpoint} has no {PROVENANCE_NAME} beside it. Fine-tuned weights are "
            f"loaded only with the record of what they were trained on and which logit "
            f"is person — see argus/vision/checkpoint.py. Re-run "
            f"scripts/train_detector.py, or point detection.model.checkpoint at a run "
            f"directory that has one."
        )
    try:
        document = json.loads(sidecar.read_text())
    except (OSError, ValueError) as exc:
        raise CheckpointError(f"{sidecar}: {exc}") from exc
    if not isinstance(document, dict):
        raise CheckpointError(f"{sidecar}: expected a JSON object")

    person_index = document.get("person_class_index")
    if not isinstance(person_index, int) or isinstance(person_index, bool) or person_index < 0:
        raise CheckpointError(
            f"{sidecar}: person_class_index must be a non-negative integer "
            f"(got {person_index!r}). Without it nothing knows which output is a person."
        )

    entry = (document.get("checkpoints") or {}).get(checkpoint.name)
    if not isinstance(entry, dict):
        raise CheckpointError(
            f"{sidecar} does not describe {checkpoint.name} "
            f"(it describes {sorted((document.get('checkpoints') or {}))}). "
            f"The sidecar belongs to a different run."
        )

    recorded = entry.get("sha256")
    if verify_digest and isinstance(recorded, str) and recorded:
        actual = file_sha256(checkpoint)
        if actual != recorded:
            raise CheckpointError(
                f"{checkpoint} does not match the weights {sidecar} was written for "
                f"(sha256 {actual[:12]} vs {recorded[:12]}). Whatever was measured, "
                f"this is not it."
            )

    num_classes = entry.get("num_classes")
    head_width = entry.get("head_width")
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise CheckpointError(f"{sidecar}: checkpoints.{checkpoint.name}.num_classes is missing")
    if isinstance(head_width, int) and person_index >= head_width:
        raise CheckpointError(
            f"{sidecar}: person_class_index {person_index} is outside a head "
            f"{head_width} wide — the sidecar contradicts itself."
        )

    training = document.get("training") or {}
    dataset = document.get("dataset") or {}
    splits = dataset.get("splits") or {}
    best_epoch, map_50, map_50_95 = _scores_for(entry, training)

    return CheckpointProvenance(
        path=checkpoint,
        person_class_index=person_index,
        num_classes=num_classes,
        sha256=recorded if isinstance(recorded, str) else None,
        created_at=document.get("created_at") if isinstance(document.get("created_at"), str) else None,
        dataset_digest=dataset.get("digest") if isinstance(dataset.get("digest"), str) else None,
        dataset_images=sum(
            int(block.get("images", 0)) for block in splits.values() if isinstance(block, dict)
        ),
        dataset_boxes=sum(
            int(block.get("boxes", 0)) for block in splits.values() if isinstance(block, dict)
        ),
        epochs_run=int(training.get("epochs_run") or 0),
        best_epoch=best_epoch,
        val_map_50=map_50,
        val_map_50_95=map_50_95,
        variant=str(document.get("variant") or ""),
        resolution=int(document.get("resolution") or 0),
        wall_clock_sec=_number(training.get("wall_clock_sec")),
    )


def _scores_for(
    entry: dict[str, Any], training: dict[str, Any]
) -> tuple[int | None, float | None, float | None]:
    """The epoch and val mAP of *this* checkpoint, not of the run's best epoch.

    A training run keeps two tracks — the live weights and their EMA — and
    writes the winner out as ``checkpoint_best_total.pth``. They peak at
    different epochs and score differently, so reading the regular track's
    numbers off a file that came from the EMA track is quoting one model's
    result on another model's weights. ``best_total_source`` is the field the
    trainer records precisely so this can be got right.
    """
    ema = entry.get("best_total_source") == "ema"
    best = training.get("best_ema" if ema else "best") or {}
    prefix = "val/ema_mAP" if ema else "val/mAP"
    epoch = entry.get("epoch")
    if _number(epoch) is None:
        # `checkpoint_best_total.pth` is a copy and carries no epoch of its own;
        # the track it was copied from does.
        epoch = best.get("epoch")
    return (
        int(epoch) if _number(epoch) is not None else None,
        _number(best.get(f"{prefix}_50")),
        _number(best.get(f"{prefix}_50_95")),
    )


__all__ = [
    "PROVENANCE_NAME",
    "CheckpointError",
    "CheckpointProvenance",
    "file_sha256",
    "load_provenance",
]
