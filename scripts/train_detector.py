#!/usr/bin/env python3
"""Fine-tune an RF-DETR on a site's own labelled frames.

    ./scripts/setup_detector_training.sh          # once
    uv run python scripts/train_detector.py       # ~N minutes on a GB10

`argus detector-eval` established what the shipped COCO detector does in Anmol
Industries' biscuit plant: 44.6% presence recall on `val`, against the 75% a
headcount policy needs (`docs/DETECTOR_EVAL.md`). This is the other half of that
question — what 63 labelled frames from the plant itself can buy.

**It produces a candidate, not a default.** The output is weights plus a
`provenance.json`, in a gitignored directory. Whether they are worth loading is
decided by `argus detector-eval --checkpoint <path>`, on the val split, against
the stock checkpoint, per scene. Nothing here writes config.

Seven decisions are baked in, and each one is a thing that would otherwise be
silently wrong. **Decision 7 is the one that was learned the hard way**: a run
here once handed back the weights it started from, and a document was written
on the strength of it.

**1. Person keeps COCO's class slot.** The corpus is single-class YOLO (`0` =
person). Trained as-is, RF-DETR would build a 2-wide head and put person on
logit 0 — while `PersonDetector._postprocess` reads logit 1, because RF-DETR's
COCO head is indexed by raw COCO category id and person is id 1. So the staged
copy shifts the class column to `1` and declares two classes,
`["__background__", "person"]`. Two things follow, both wanted: the shipped
inference path needs no change at all, and the head trim from COCO's 91 rows to
3 (`_resize_linear` keeps the first N) hands the fine-tune *COCO's own person
row* as its starting point instead of the unused id-0 row. On 63 images that
initialisation is not a detail.

**2. The resolution is the engine's, not the trainer's default.** Training at a
resolution the appliance will not run produces weights that score better in a
notebook and worse in the appliance, which decodes to 960x544 and squashes to
whatever `detection.model.resolution` says. So `--resolution` defaults to the
variant's own native input and, whenever it is set, the config line that would
have to move with it is printed beside it. *If the resolution budget is raised,
it is raised in `config/argus.yaml` first and here second.*

**2a. `--variant` exists, and it stops at Large.** RF-DETR XL and 2XL are under
Roboflow's PML, not Apache-2.0 (`docs/DETECTION_V2.md` §6), so they are not
choices here — a licence is not a flag anybody should be able to pass by
accident. Nano/Small/Medium/Large are Apache-2.0. Every variant keeps decision 1:
person on class slot 1, whatever the backbone.

**3. The val split is never trained on and never tuned against by hand.**
Early stopping reads `val/mAP_50_95`, so val is *used* — that is the honest
caveat on every number this produces and it is written into the provenance.
16 frames cannot be an evaluation; they are a stopping rule.

**4. Everything that could make the run non-reproducible is recorded.** Dataset
digest (sha256 over every image and label, order-independent per split), base
checkpoint digest, package versions, seed, the full resolved `TrainConfig`, and
the metric history. `provenance.json` is the artifact; the weights are a
by-product of it.

**5. It runs in `.venv-train`, and re-execs itself to get there.** The training
stack is ~15 packages the appliance never imports; keeping them out of `.venv`
is what makes `scripts/license_audit.py --check` a statement about the product.
`uv run python scripts/train_detector.py` still works — the first thing this
file does is notice it is in the wrong interpreter and hand over.

**6. It will not take more of a shared box than it asked for.** The GB10's
121.7 GiB is unified and shared: `docs/DETECTION_V2.md` §2.1 is the story of one
tenant holding 95 of it and every model on the appliance going dark for twenty
days. So the run refuses to start below a MemAvailable floor and then caps
itself with `torch.cuda.set_per_process_memory_fraction`, which turns "this
training run starved the box" into "this training run got a CUDA OOM" — a
failure that costs one person an experiment instead of costing everybody the
appliance. The cap is a wall to fail against, not a budget to fill.

**7. The checkpoint it hands back is checked against the one it started from.**
rfdetr can select the pre-training EMA snapshot as `checkpoint_best_total.pth`
— it did, on 2026-09-20, for the run `docs/STUDENT_DETECTOR.md`'s verdict was
measured on. The failure is silent in every way that matters: right size, right
head width, loads, passes the digest check, detects people. `audit_best_total`
compares its backbone against the pretrained file and repairs it from
`checkpoint_best_regular.pth`; `--audit-only` applies the same check to a run
that has already finished. A metric history full of improving epochs is not
evidence that the weights on disk came from one of them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
TRAIN_VENV = Path(os.environ.get("ARGUS_TRAIN_VENV") or ROOT / ".venv-train")

# The corpus `argus detector-eval` scores. Gitignored: it is a customer's plant.
DEFAULT_DATASET = ROOT / "footage" / "anmol-samples" / "cctv_person_tracking_dataset"
DEFAULT_OUTPUT_ROOT = ROOT / "models" / "finetuned"

# RF-DETR's weight cache. Pointed at the appliance's own `data/models` — the
# same directory `PersonDetector.load` sets RF_HOME to — so the fine-tune starts
# from the exact file the engine loads today, and `provenance.json` can name it
# by digest rather than by "whatever pip fetched into a home directory".
os.environ.setdefault("RF_HOME", str(ROOT / "data" / "models"))

# The class space the appliance already speaks. Index 1 is person because
# RF-DETR's COCO head is indexed by raw COCO category id; see the module
# docstring and `argus/vision/detector.py:COCO_PERSON_CLASS_ID`.
CLASS_NAMES = ["__background__", "person"]
PERSON_CLASS_INDEX = 1

# Apache-2.0 only. RF-DETR XL/2XL are under Roboflow's PML and are deliberately
# absent — see decision 2a and docs/DETECTION_V2.md §6.
VARIANTS: dict[str, str] = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}

# Each variant's own native input, used when `--resolution` is not given. The
# engine runs the Nano at 512 rather than its native 384 (config/argus.yaml,
# measured), so a default is a starting point and never an assumption.
NATIVE_RESOLUTION: dict[str, int] = {"nano": 384, "small": 512, "medium": 576, "large": 704}

# Fraction of the GB10's unified pool this process may allocate before CUDA
# refuses it — about 19.5 GiB of 121.7. See decision 6.
GPU_MEMORY_FRACTION = 0.16
# MemAvailable below this and no GPU work starts at all. The same floor
# scripts/teacher_label.py uses, raised because training holds its memory for
# minutes rather than milliseconds.
MIN_AVAILABLE_GIB = 14.0

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# Checkpoints the run leaves behind. `checkpoint_best_total.pth` is the winner
# of (regular, EMA) and is the one a config line would point at; the two it was
# chosen between are kept because "which one won" is a fact about the run.
# `last*.pth` is the final epoch, which is exactly the epoch early stopping
# decided was worse — 120 MB to say so.
KEPT_CHECKPOINTS = ("checkpoint_best_total.pth", "checkpoint_best_ema.pth",
                    "checkpoint_best_regular.pth")


# --------------------------------------------------------------------------
# interpreter handover
# --------------------------------------------------------------------------


def reexec_into_training_venv() -> None:
    """Hand over to `.venv-train` if this interpreter cannot train.

    Guarded by an environment marker rather than a path comparison: a venv can
    be reached by several paths (symlink, relative invocation), and an exec loop
    is a much worse failure than a clear error.
    """
    try:
        import pytorch_lightning  # noqa: F401
    except ImportError:
        pass
    else:
        return

    if os.environ.get("ARGUS_TRAIN_REEXEC") == "1":
        sys.exit(
            f"{TRAIN_VENV} exists but still cannot import pytorch_lightning.\n"
            "Rebuild it: ./scripts/setup_detector_training.sh"
        )
    python = TRAIN_VENV / "bin" / "python"
    if not python.exists():
        sys.exit(
            "the training venv is missing — the appliance venv deliberately does "
            "not carry the training stack.\n"
            "  ./scripts/setup_detector_training.sh"
        )
    os.environ["ARGUS_TRAIN_REEXEC"] = "1"
    os.execv(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]])


# --------------------------------------------------------------------------
# the shared box
# --------------------------------------------------------------------------


def mem_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1024 * 1024)
    raise RuntimeError("MemAvailable missing from /proc/meminfo")


def claim_gpu_budget(fraction: float, minimum_gib: float) -> dict[str, Any]:
    """Refuse to start on a full box, then cap what this process may take.

    Both halves matter and they fail differently. The floor is about *other*
    people: starting a training run on a box with 6 GiB left is how §2.1
    happened. The cap is about *this* process: without it an over-large batch
    quietly consumes the pool that the live appliance's decoder and detector are
    also drawing from, and the symptom appears somewhere else, on somebody
    else's camera. With it, the same mistake is a CUDA OOM in this terminal.
    """
    import torch

    available = mem_available_gib()
    if available < minimum_gib:
        raise SystemExit(
            f"MemAvailable is {available:.1f} GiB, below the {minimum_gib:.0f} GiB floor. "
            f"Not starting training on a shared box (docs/DETECTION_V2.md §2.1)."
        )
    budget: dict[str, Any] = {
        "mem_available_gib_at_start": round(available, 1),
        "memory_fraction": fraction,
    }
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        torch.cuda.set_per_process_memory_fraction(fraction)
        budget["device_total_gib"] = round(total, 1)
        budget["cap_gib"] = round(total * fraction, 1)
        print(f"MemAvailable {available:.1f} GiB · this process capped at "
              f"{budget['cap_gib']:.1f} GiB of {total:.1f} GiB unified", flush=True)
    return budget


# --------------------------------------------------------------------------
# corpus
# --------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_dirs(dataset: Path, split: str) -> tuple[Path, Path]:
    return dataset / "images" / split, dataset / "labels" / split


def read_split(dataset: Path, split: str) -> list[tuple[Path, Path]]:
    """Every (image, label) pair in one split, in a stable order.

    A label file that is missing is an error rather than an empty frame: the
    export pairs them, so absence means a broken copy — and silently treating it
    as "no people here" would teach the detector that this scene is empty.
    """
    images_dir, labels_dir = _split_dirs(dataset, split)
    if not images_dir.is_dir() or not labels_dir.is_dir():
        sys.exit(
            f"split {split!r} not found under {dataset}\n"
            f"  expected {images_dir} and {labels_dir}"
        )
    pairs: list[tuple[Path, Path]] = []
    for image in sorted(images_dir.iterdir()):
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = labels_dir / f"{image.stem}.txt"
        if not label.exists():
            sys.exit(f"no label file for {image.name} (looked at {label})")
        pairs.append((image, label))
    if not pairs:
        sys.exit(f"no images in {images_dir}")
    return pairs


def read_boxes(label: Path) -> list[tuple[int, float, float, float, float]]:
    """YOLO `class cx cy w h` lines, validated.

    Normalisation is checked rather than assumed: a box whose centre or extent
    falls outside [0, 1] is a coordinate-system bug in the export, and training
    on it teaches the detector a systematic offset that no metric here would
    name.
    """
    boxes = []
    for number, line in enumerate(label.read_text().splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 5:
            sys.exit(f"{label}:{number}: expected 'class cx cy w h', got {line!r}")
        try:
            class_id = int(float(parts[0]))
            cx, cy, w, h = (float(value) for value in parts[1:5])
        except ValueError as exc:
            sys.exit(f"{label}:{number}: {exc}")
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            sys.exit(f"{label}:{number}: box is not normalized: {line!r}")
        boxes.append((class_id, cx, cy, w, h))
    return boxes


def stage_split(
    pairs: list[tuple[Path, Path]], destination: Path, *, shift: int
) -> dict[str, Any]:
    """Write one split into RF-DETR's YOLO layout, shifting the class column.

    Images are symlinked, not copied: they are 2560x1440 frames of somebody's
    factory, and one copy of a customer's footage on this disk is already one
    more than ideal. Labels are rewritten because the class column changes.

    Returns the split's manifest — counts, class histogram, and a digest over
    the *source* files, so the digest identifies the corpus rather than this
    staging run.
    """
    images_out = destination / "images"
    labels_out = destination / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    per_file: list[str] = []
    classes: dict[int, int] = {}
    boxes_total = 0
    for image, label in pairs:
        boxes = read_boxes(label)
        lines = []
        for class_id, cx, cy, w, h in boxes:
            classes[class_id] = classes.get(class_id, 0) + 1
            lines.append(f"{class_id + shift} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        boxes_total += len(boxes)

        link = images_out / image.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(image.resolve())
        # An empty file for a frame with no boxes — a negative example, which the
        # YOLO loader reads as "look here and find nothing". Not the same as a
        # missing file, which `read_split` refuses.
        body = "\n".join(lines) + "\n" if lines else ""
        (labels_out / f"{image.stem}.txt").write_text(body)
        per_file.append(f"{image.name} {_sha256(image)} {_sha256(label)}")

    return {
        "images": len(pairs),
        "boxes": boxes_total,
        "source_classes": {str(k): v for k, v in sorted(classes.items())},
        # Order-independent in the sense that matters: the lines are sorted, so
        # a filesystem that enumerates differently still digests the same.
        "digest": hashlib.sha256("\n".join(sorted(per_file)).encode()).hexdigest(),
        "scenes": sorted({_scene_of(image.stem) for image, _ in pairs}),
    }


def _scene_of(stem: str) -> str:
    """`Line_No.3_Packing3_Bas_frame_07` -> `Line_No.3_Packing3_Bas`.

    Same rule as `argus/eval/detector.py:scene_of`, duplicated rather than
    imported because this script never imports `argus` — it runs in a venv the
    appliance is not installed into.
    """
    head, sep, _ = stem.rpartition("_frame_")
    return head if sep else stem


def stage_dataset(dataset: Path, destination: Path) -> dict[str, Any]:
    """The corpus, in the layout RF-DETR's YOLO loader expects.

    `images/<split>` becomes `<split>/images`, `val` becomes `valid`, and a
    `data.yaml` names the two classes. RF-DETR's own validity check
    (`is_valid_yolo_dataset`) wants exactly this and nothing else.
    """
    if destination.exists():
        shutil.rmtree(destination)
    manifest: dict[str, Any] = {"root": str(dataset), "splits": {}}
    for split, staged in (("train", "train"), ("val", "valid")):
        pairs = read_split(dataset, split)
        manifest["splits"][split] = stage_split(
            pairs, destination / staged, shift=PERSON_CLASS_INDEX
        )

    (destination / "data.yaml").write_text(
        "# Generated by scripts/train_detector.py — do not edit.\n"
        "#\n"
        "# Two classes, and the first one has no instances. That is deliberate:\n"
        "# it puts `person` on class index 1, which is the raw COCO category id\n"
        "# RF-DETR's pretrained head already uses and the index\n"
        "# `PersonDetector._postprocess` reads. See the module docstring.\n"
        "path: .\n"
        "train: train/images\n"
        "val: valid/images\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {json.dumps(CLASS_NAMES)}\n"
    )
    manifest["class_names"] = CLASS_NAMES
    manifest["label_class_shift"] = PERSON_CLASS_INDEX
    manifest["digest"] = hashlib.sha256(
        "\n".join(
            f"{split}:{block['digest']}" for split, block in sorted(manifest["splits"].items())
        ).encode()
    ).hexdigest()
    return manifest


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------


def base_checkpoint_info(model: Any, resolution: int) -> dict[str, Any]:
    """Where the pretrained weights came from, and what they hash to.

    RF-DETR resolves `pretrain_weights` against its own cache (`RF_HOME`), so
    the path is not knowable before the model is constructed — this runs after.
    """
    from rfdetr.assets.model_weights import get_model_cache_dir

    name = str(getattr(model.model_config, "pretrain_weights", "") or "")
    info: dict[str, Any] = {"name": name, "resolution": resolution}
    for candidate in (Path(name), Path(get_model_cache_dir()) / name):
        if candidate.is_file():
            info["path"] = str(candidate)
            info["sha256"] = _sha256(candidate)
            info["bytes"] = candidate.stat().st_size
            break
    return info


def read_csv_metrics(output_dir: Path) -> list[dict[str, Any]]:
    """Per-epoch validation metrics, read back off the CSV logger.

    PTL's CSVLogger is the only logger this venv has (tensorboard is not
    installed and is switched off), and it writes exactly what the provenance
    needs: one row per logged step, with the val metrics on the epochs that ran
    validation. RF-DETR points the logger at ``output_dir`` itself; the
    ``lightning_logs/version_*`` layout is PTL's own default and is checked
    second so a future rfdetr that stops overriding it still works.
    """
    import csv

    candidates = [output_dir / "metrics.csv"]
    candidates += [
        version / "metrics.csv"
        for version in sorted((output_dir / "lightning_logs").glob("version_*"))
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            wanted = {
                key: float(value)
                for key, value in row.items()
                if value not in (None, "") and (key.startswith("val/") or key == "epoch")
            }
            # Every step logs `epoch`; only the validation steps log a metric.
            # A train-only row would otherwise land in the history as an epoch
            # with no score and quietly beat `max()` on a missing key.
            if "val/mAP_50_95" in wanted:
                rows.append(wanted)
    return rows


def train(
    args: argparse.Namespace, dataset_dir: Path, output_dir: Path, *, dataset_digest: str
) -> dict[str, Any]:
    """Run the fine-tune and return everything worth recording about it."""
    import rfdetr
    import torch
    from pytorch_lightning import seed_everything

    budget = claim_gpu_budget(args.memory_fraction, args.min_available_gib)

    seed_everything(args.seed, workers=True)
    resolution = args.resolution
    model_cls = getattr(rfdetr, VARIANTS[args.variant])
    # `resolution` is a model-config field, so it is set at construction: the
    # backbone's positional encoding is sized from it and cannot be changed
    # afterwards. `gradient_checkpointing` likewise — it is how a 576 px Medium
    # fits under the cap, at roughly 30% more time per epoch.
    model = model_cls(resolution=resolution, gradient_checkpointing=args.gradient_checkpointing)
    base = base_checkpoint_info(model, resolution)

    settings: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "dataset_file": "yolo",          # not auto-detected; the default is COCO json
        "output_dir": str(output_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum,
        "lr": args.lr,
        "lr_encoder": args.lr_encoder,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "resolution": resolution,
        "device": args.device,
        "class_names": CLASS_NAMES,
        "early_stopping": True,
        "early_stopping_patience": args.patience,
        "early_stopping_min_delta": args.min_delta,
        # The EMA weights are what usually win on a split this small, so the
        # stopping rule should watch the thing being kept.
        "early_stopping_use_ema": True,
        # Off: not installed in `.venv-train`, and a training run that needs a
        # web UI to be legible has recorded the wrong things. CSV is the
        # provenance source.
        "tensorboard": False,
        "wandb": False,
        "progress_bar": "tqdm" if not args.quiet else None,
        "checkpoint_interval": max(args.epochs, 1),   # only the best-model files
        "run_test": False,
        # Carried inside every .pth this run writes, under
        # `checkpoint["args"]["notes"]`. The sidecar `provenance.json` is the
        # readable copy; this is the one that survives the weights being moved.
        "notes": {
            "produced_by": "argus scripts/train_detector.py",
            "dataset_digest": dataset_digest,
            "class_names": CLASS_NAMES,
            "person_class_index": PERSON_CLASS_INDEX,
            "variant": args.variant,
            "resolution": resolution,
            "seed": args.seed,
        },
    }
    if args.no_ema:
        settings["use_ema"] = False
        settings["early_stopping_use_ema"] = False

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model.train(**settings)
    elapsed = time.perf_counter() - started
    if torch.cuda.is_available():
        budget["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        budget["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 1024**3, 2)

    history = read_csv_metrics(output_dir)

    def _best(key: str) -> dict[str, Any]:
        row = max((r for r in history if key in r), key=lambda r: r[key], default={})
        return {name: round(value, 6) for name, value in row.items()}

    return {
        "base_checkpoint": base,
        "settings": settings,
        "gpu_budget": budget,
        "wall_clock_sec": round(elapsed, 1),
        "epochs_run": int(max((row.get("epoch", -1) for row in history), default=-1)) + 1,
        # Both tracks, because `checkpoint_best_total.pth` is the winner of the
        # two and which one won is a fact about this run, not a constant.
        "best": _best("val/mAP_50_95"),
        "best_ema": _best("val/ema_mAP_50_95"),
        "history": [
            {key: round(value, 6) for key, value in row.items()} for row in history
        ],
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "versions": _versions(),
    }


def _versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for name in ("torch", "torchvision", "rfdetr", "pytorch-lightning", "torchmetrics", "numpy"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            continue
    out["python"] = sys.version.split()[0]
    return out


def _backbone_is_identical(left: Path, right: Path, *, samples: int = 5) -> bool:
    """Are these two checkpoints the same weights, bit for bit?

    Compares the first ``samples`` multi-dimensional backbone tensors the two
    files share. The backbone rather than the head because the head is trimmed
    from COCO's 91 rows to 3 on the way in and would differ between two files
    that are otherwise the same network; the backbone is copied verbatim and a
    single step of training moves every tensor in it.
    """
    import torch

    def state(path: Path) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        return checkpoint.get("model") or checkpoint.get("state_dict") or {}

    a, b = state(left), state(right)
    keys = [
        key for key in a
        if "backbone" in key and key.endswith("weight") and key in b and a[key].ndim > 1
    ][:samples]
    return bool(keys) and all(torch.equal(a[key], b[key]) for key in keys)


def audit_best_total(output: Path, base_path: Path | None, history: list[dict[str, Any]]
                     ) -> dict[str, Any]:
    """Refuse to ship the pre-training snapshot as the best checkpoint.

    **This is guarding against a real defect, found 2026-09-20.** rfdetr's
    `BestModelCheckpoint` delegates the *regular* track to PyTorch Lightning's
    `ModelCheckpoint`, which knows not to record a best score during the
    pre-training sanity-check validation. Its own EMA high-water mark
    (`_best_ema`, `training/callbacks/best_model.py`) has no such guard. So the
    sanity check — two batches, on whichever views sort first — sets
    `_best_ema`, writes `checkpoint_best_ema.pth` from the *untrained* EMA
    copy, and if those two batches happened to be easy, no real epoch ever
    beats it. `on_fit_end` then prefers EMA on a strict ``>`` and copies that
    file over `checkpoint_best_total.pth`.

    The result is a run that trains perfectly well for an hour and a half and
    hands back the weights it started from, with a provenance.json full of
    honest metrics from epochs whose weights were thrown away. Nothing
    downstream can tell: the file is the right size, loads, has the right head
    width, and detects people — it is stock COCO, which detects people.

    The test is the one thing the defect cannot fake: if the backbone is bit
    identical to the pretrained file, no gradient ever reached it. The repair
    is `checkpoint_best_regular.pth`, whose selection went through the guarded
    parent and is the epoch the metric history says won.
    """
    total = output / "checkpoint_best_total.pth"
    regular = output / "checkpoint_best_regular.pth"
    ema = output / "checkpoint_best_ema.pth"
    verdict: dict[str, Any] = {"checked": True, "repaired": False}

    if not total.exists() or base_path is None or not Path(base_path).is_file():
        verdict["checked"] = False
        verdict["reason"] = "no best_total or no base checkpoint to compare against"
        return verdict

    if not _backbone_is_identical(total, Path(base_path)):
        verdict["reason"] = "best_total's backbone differs from the pretrained weights"
        return verdict

    # Two conditions, not one. "Identical to the base" alone would also be true
    # of a run that legitimately froze its backbone — and then the regular
    # checkpoint would be identical to the base as well. Requiring that the
    # regular track *did* move is what makes this a statement about which file
    # was selected rather than about how the model was configured.
    if not regular.exists() or _backbone_is_identical(regular, Path(base_path)):
        verdict["reason"] = (
            "best_total matches the pretrained weights, but so does "
            "checkpoint_best_regular.pth — the backbone did not train in this run at all, "
            "which is a different problem from the wrong file being selected"
        )
        return verdict

    # It is the pre-training snapshot. Say so loudly, then repair it.
    best_ema_epoch = max(
        (row for row in history if "val/ema_mAP_50_95" in row),
        key=lambda row: row["val/ema_mAP_50_95"], default={},
    ).get("epoch")
    verdict.update({
        "reason": (
            "checkpoint_best_total.pth was bit-identical to the pretrained base — it is the "
            "EMA snapshot rfdetr writes during the pre-training sanity check, not a trained "
            "epoch. See audit_best_total()."
        ),
        "best_ema_epoch_in_history": best_ema_epoch,
        "discarded": describe_checkpoint(total),
    })
    shutil.copy2(regular, total)
    verdict["repaired"] = True
    verdict["replaced_with"] = "checkpoint_best_regular.pth"
    verdict["now"] = describe_checkpoint(total)
    if ema.exists():
        # Renamed rather than deleted: it is evidence, and it must stop looking
        # like a checkpoint anybody would point a config line at.
        ema.rename(output / "checkpoint_untrained_sanity_ema.pth.rejected")
        verdict["ema_file_renamed_to"] = "checkpoint_untrained_sanity_ema.pth.rejected"
    return verdict


def describe_checkpoint(path: Path) -> dict[str, Any]:
    """What a loader needs to know about the weights, read back off disk.

    Read back rather than assumed: the head width and the person index are the
    two facts that make these weights loadable by `PersonDetector`, and
    asserting them from the training config would prove nothing about the file.
    """
    import torch

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 - a config value torch's allowlist does not know
        # This file was written by this process a second ago, so the pickle is
        # as trusted as anything gets. The safe path is still tried first, so a
        # checkpoint that does not need the fallback never takes it.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model") or checkpoint.get("state_dict") or {}
    head = next(
        (tensor for name, tensor in state.items() if name.endswith("class_embed.bias")), None
    )
    head_width = int(head.shape[0]) if head is not None else 0
    info = {
        "file": path.name,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "epoch": checkpoint.get("epoch"),
        # num_classes excludes the trailing no-object slot: the head is built
        # `num_classes + 1` wide (`rfdetr/models/lwdetr.py`).
        "num_classes": max(head_width - 1, 0),
        "head_width": head_width,
        "class_names": CLASS_NAMES,
        "person_class_index": PERSON_CLASS_INDEX,
        "model_name": checkpoint.get("model_name"),
    }
    # Only `checkpoint_best_total.pth` carries this: it is a copy of whichever
    # of the regular and EMA tracks scored higher, and which one that was
    # changes what the number means.
    if "best_total_source" in checkpoint:
        info["best_total_source"] = checkpoint["best_total_source"]
    return info


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help="YOLO corpus root holding images/<split> and labels/<split>")
    parser.add_argument("--output", type=Path, default=None,
                        help="run directory (default: models/finetuned/<today>)")
    parser.add_argument("--class-name", default="person",
                        help="name of the corpus's one class. It always trains into slot 1 (see "
                             "decision 1); only the person detector reads that slot as a person, so "
                             "anything else (e.g. carton) is a separate model, never a production "
                             "detection.model.checkpoint")
    parser.add_argument("--variant", default="nano", choices=sorted(VARIANTS),
                        help="XL/2XL are Roboflow PML, not Apache-2.0, and are deliberately absent")
    parser.add_argument("--resolution", type=int, default=0,
                        help="model input, square (default: the variant's native). Whatever is "
                             "set here must also be set as detection.model.resolution, or the "
                             "appliance runs these weights at a size they were not trained for")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="recompute activations in the backward pass: ~30%% slower per epoch, "
                             "and what makes a 576 px Medium fit the memory cap")
    parser.add_argument("--memory-fraction", type=float, default=GPU_MEMORY_FRACTION,
                        help="share of the unified pool this process may allocate before CUDA "
                             "refuses it (the box is shared — docs/DETECTION_V2.md §2.1)")
    parser.add_argument("--min-available-gib", type=float, default=MIN_AVAILABLE_GIB,
                        help="refuse to start below this much MemAvailable")
    parser.add_argument("--epochs", type=int, default=60,
                        help="the time box. 63 images is a handful of seconds an epoch; "
                             "early stopping normally ends the run first (default: 60)")
    parser.add_argument("--patience", type=int, default=12,
                        help="epochs without a val/mAP_50_95 improvement before stopping")
    parser.add_argument("--min-delta", type=float, default=0.001,
                        help="improvement below this does not reset the patience counter")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="effective batch is batch-size x this (default 4x4 = 16)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-encoder", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-ema", action="store_true",
                        help="train without the EMA copy (it usually wins on a split this small)")
    parser.add_argument("--keep-all", action="store_true",
                        help="keep last.pth / last_ema.pth as well as the best-model files")
    parser.add_argument("-q", "--quiet", action="store_true", help="no progress bar")
    parser.add_argument("--audit-only", action="store_true",
                        help="train nothing: re-run audit_best_total() over an existing "
                             "--output directory and update its provenance.json")
    args = parser.parse_args(argv)
    if not args.resolution:
        args.resolution = NATIVE_RESOLUTION[args.variant]
    # RF-DETR's windowed DINOv2 backbone needs the input to be a whole number of
    # patches per window. Caught here rather than in a shape error 200 layers in.
    if args.resolution % 56 and args.resolution % 32:
        print(f"warning: resolution {args.resolution} is not a multiple of 32; "
              f"RF-DETR may refuse it", file=sys.stderr)
    return args


def audit_only(output: Path) -> int:
    """Apply `audit_best_total` to a run that has already finished.

    Runs that finished before the defect was found are sitting on disk with a
    `checkpoint_best_total.pth` that may be the weights they started from, and
    `docs/STUDENT_DETECTOR.md`'s verdict was measured on one of them. This is
    how they get checked without a retrain.
    """
    provenance_path = output / "provenance.json"
    if not provenance_path.is_file():
        print(f"no provenance.json under {output}", file=sys.stderr)
        return 2
    provenance = json.loads(provenance_path.read_text())
    training = provenance.get("training", {})
    base = (training.get("base_checkpoint") or {}).get("path")

    audit = audit_best_total(output, base, training.get("history", []))
    provenance["best_total_audit"] = audit
    provenance["checkpoints"] = {
        name: describe_checkpoint(output / name)
        for name in KEPT_CHECKPOINTS if (output / name).exists()
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")

    if not audit.get("checked"):
        print(f"{output.name}: NOT CHECKED — {audit.get('reason')}")
    elif audit.get("repaired"):
        print(f"{output.name}: REPAIRED — {audit['reason']}")
        print(f"  best_total was epoch {audit['discarded'].get('epoch')} "
              f"sha256 {audit['discarded']['sha256'][:12]}… (untrained)")
        print(f"  best_total is  epoch {audit['now'].get('epoch')} "
              f"sha256 {audit['now']['sha256'][:12]}… (from checkpoint_best_regular.pth)")
    else:
        print(f"{output.name}: ok — {audit['reason']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    global CLASS_NAMES
    args = parse_args(argv)
    CLASS_NAMES = ["__background__", args.class_name]
    if args.audit_only:
        if args.output is None:
            print("--audit-only needs --output <run directory>", file=sys.stderr)
            return 2
        return audit_only(args.output.expanduser().resolve())
    dataset = args.dataset.expanduser().resolve()
    if not dataset.is_dir():
        print(
            f"corpus not found at {dataset}. It is gitignored (customer footage); "
            f"place the export there or pass --dataset.",
            file=sys.stderr,
        )
        # 2, not 1: "the corpus is not on this box" is a different fact from
        # "training failed", and the same convention `argus detector-eval` uses.
        return 2

    default_name = f"{date.today().isoformat()}-{args.variant}{args.resolution}"
    output = (args.output or DEFAULT_OUTPUT_ROOT / default_name).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    staged = output / "dataset"

    print(f"corpus:  {dataset}")
    print(f"staging: {staged}")
    manifest = stage_dataset(dataset, staged)
    for split, block in manifest["splits"].items():
        print(f"  {split:<5} {block['images']:>3} images, {block['boxes']:>4} boxes, "
              f"{len(block['scenes'])} scenes, classes {block['source_classes']}")
    print(f"  digest {manifest['digest'][:16]}…  ->  classes {CLASS_NAMES} "
          f"(person on index {PERSON_CLASS_INDEX})")
    print(f"model:   rf-detr-{args.variant} @ {args.resolution} px"
          + ("  (gradient checkpointing)" if args.gradient_checkpointing else ""))
    print(f"output:  {output}\n")

    result = train(args, staged, output, dataset_digest=manifest["digest"])

    if not args.keep_all:
        # `last*.pth` is the final epoch — which is the epoch early stopping just
        # decided was worse. `*.ckpt` is PTL's own periodic checkpoint, ~480 MB
        # each because it carries optimizer and scheduler state for a resume
        # nobody wants: this run is 5 minutes, and it is cheaper to repeat it
        # from `provenance.json` than to keep a gigabyte to avoid repeating it.
        for stale in [output / "last.pth", output / "last_ema.pth", *output.glob("*.ckpt")]:
            if stale.exists():
                stale.unlink()

    # Before anything describes or digests the checkpoints: is the one a config
    # line would point at actually a trained epoch? See audit_best_total().
    audit = audit_best_total(
        output, (result.get("base_checkpoint") or {}).get("path"), result.get("history", [])
    )
    if audit.get("repaired"):
        print(f"\n!! {audit['reason']}\n   repaired from checkpoint_best_regular.pth\n")

    checkpoints = {}
    for name in KEPT_CHECKPOINTS:
        path = output / name
        if path.exists():
            checkpoints[name] = describe_checkpoint(path)

    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "produced_by": "scripts/train_detector.py",
        "variant": f"rf-detr-{args.variant}",
        "person_class_index": PERSON_CLASS_INDEX,
        "class_names": CLASS_NAMES,
        "resolution": args.resolution,
        "dataset": manifest,
        "training": result,
        "checkpoints": checkpoints,
        "best_total_audit": audit,
        # Said in the artifact, not only in a doc: early stopping reads the val
        # split, so val is a stopping rule these weights have seen and not a
        # held-out test set. Any number quoted from it inherits that.
        "caveat": (
            "val was used for early stopping and best-checkpoint selection, so it is "
            "not held out. 16 frames / 175 boxes is a signal, not an evaluation."
        ),
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    print()
    print(f"epochs run:   {result['epochs_run']} of {args.epochs} "
          f"(early stop patience {args.patience})")
    print(f"wall clock:   {result['wall_clock_sec']:.0f} s "
          f"({result['wall_clock_sec'] / max(result['epochs_run'], 1):.1f} s/epoch)")
    budget = result.get("gpu_budget", {})
    if "peak_reserved_gib" in budget:
        print(f"peak GPU:     {budget['peak_reserved_gib']:.2f} GiB reserved "
              f"({budget['peak_allocated_gib']:.2f} allocated) of a "
              f"{budget.get('cap_gib', 0):.1f} GiB cap")
    for label, key, row in (
        ("regular", "val/mAP_50_95", result["best"]),
        ("ema    ", "val/ema_mAP_50_95", result["best_ema"]),
    ):
        if row:
            print(f"best {label}: epoch {int(row.get('epoch', -1)):>3}  "
                  f"{key} {row.get(key, float('nan')):.4f}  "
                  f"val/mAP_50 {row.get('val/mAP_50', float('nan')):.4f}")
    for name, info in checkpoints.items():
        source = f"  <- {info['best_total_source']}" if "best_total_source" in info else ""
        print(f"  {name:<30} {info['bytes'] / 1e6:6.1f} MB  head={info['head_width']} "
              f"num_classes={info['num_classes']}{source}")
    print(f"provenance:   {output / 'provenance.json'}")
    print()
    print("Next — the number that decides whether these ship:")
    if args.variant == "nano":
        print(f"  uv run argus detector-eval --checkpoint {output / 'checkpoint_best_total.pth'}")
    else:
        # `argus detector-eval` builds its detector from `detection.model` and
        # swaps only the checkpoint path (argus/eval/detector.py:_eval_detector),
        # so pointing it at a Medium would load Medium weights into a Nano and
        # fail somewhere unhelpful. Say so here rather than let somebody find out.
        print(f"  .venv/bin/python scripts/student_eval.py --student {output}")
        print("  (`argus detector-eval` takes the variant from config, not from the")
        print("   checkpoint, so it cannot score a non-nano run.)")
        print()
        print("These weights are not the shipped variant/resolution. Loading them "
              "means changing BOTH config lines, together:")
        print(f"  detection.model.variant: {args.variant}")
        print(f"  detection.model.resolution: {args.resolution}")
    return 0


if __name__ == "__main__":
    reexec_into_training_venv()
    raise SystemExit(main())
