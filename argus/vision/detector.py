"""Person detection with RF-DETR (Apache-2.0), batched across cameras.

Why RF-DETR and not YOLO: v1 ran Ultralytics YOLOv8n, which is AGPL-3.0 and is
the single reason v1 cannot ship. RF-DETR Nano is Apache-2.0, transformer-based
(no NMS by design) and, measured here, faster than the appliance needs.

**Batching is the point.** v1 held one model instance *per camera* and ran them
one frame at a time. Here every camera's newest frame goes into one tensor and
one forward pass: on the GB10 that takes a 4.9 ms single-frame pass down to
1.3 ms per frame at batch 8. The engine sizes the batch by how many cameras have
a fresh frame, so batching happens naturally without adding latency.

Precision path — the three candidates from the rebuild plan, evaluated on this
box (see ``docs/BENCHMARKS.md`` for the numbers and the full reasoning):

* **TensorRT FP16** — rejected *today*. The only TensorRT for CUDA 13 / aarch64 /
  cp312 is 11.2.1.2, which removed ``BuilderFlag.FP16``: engines take precision
  from the ONNX graph, so an FP32 export builds an FP32 engine (RF-DETR's own
  exporter detects the missing flag and silently downgrades). It is also
  licensed under NVIDIA's proprietary SLA, outside this project's permissive-only
  dependency rule.
* **ONNX Runtime GPU EP** — unavailable: ``onnxruntime-gpu`` publishes no
  aarch64 wheel (x86_64 and Windows only), so it would have to be built from
  source.
* **Torch FP16** — adopted. Eager for correctness, with a ``torch.jit`` trace
  cached per batch size (the traced graph shares the eager model's weight
  storage, so caching several shapes costs no extra memory) for the last ~15%.

The class is deliberately a thin object with a lazy :meth:`load`: constructing a
detector must stay cheap so config parsing, tests and ``argus status`` never
touch CUDA. (``Engine.start`` calls :meth:`load` eagerly all the same — see the
comment there for why a detector that fails should fail before the cameras do.)

**Weights.** COCO by default, on every appliance, forever: they are the only
weights that are not about one customer's factory. ``detection.model.checkpoint``
points at site-specific ones instead — produced by ``scripts/train_detector.py``,
measured by ``argus detector-eval --checkpoint`` — and the only thing that
changes in here is :attr:`person_class_index`, because RF-DETR's COCO head is
indexed by raw category id and a fine-tune is free to lay its head out
differently. ``argus/vision/checkpoint.py`` is the gate that makes that index
knowable rather than guessed.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

# COCO category id for "person" — RF-DETR's COCO checkpoints emit raw COCO ids,
# so class slot 1 of 91 is the one we want. Everything downstream of the
# detector speaks Argus's own single-class space (0 = person).
COCO_PERSON_CLASS_ID = 1
PERSON_CLASS_ID = 0

_VARIANTS = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "base": "RFDETRBase",
    "large": "RFDETRLarge",
}

# ImageNet statistics RF-DETR was trained with.
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class Detector(Protocol):
    """What the engine needs from a detector — and all a fake needs to provide."""

    def detect(self, images: Sequence[np.ndarray]) -> list[Any]:
        """One ``supervision.Detections`` per input image (BGR uint8 HxWx3)."""
        ...


@dataclass(slots=True)
class DetectorStats:
    batches: int = 0
    frames: int = 0
    last_batch_size: int = 0
    last_ms: float = 0.0
    ema_ms_per_frame: float = 0.0   # inference only, excluding decode and analytics
    total_ms: float = 0.0

    def record(self, batch_size: int, elapsed_ms: float) -> None:
        self.batches += 1
        self.frames += batch_size
        self.last_batch_size = batch_size
        self.last_ms = elapsed_ms
        self.total_ms += elapsed_ms
        per_frame = elapsed_ms / max(1, batch_size)
        self.ema_ms_per_frame = (
            per_frame if self.ema_ms_per_frame == 0.0 else 0.1 * per_frame + 0.9 * self.ema_ms_per_frame
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "batches": self.batches,
            "frames": self.frames,
            "last_batch_size": self.last_batch_size,
            "last_ms": round(self.last_ms, 2),
            "ms_per_frame": round(self.ema_ms_per_frame, 2),
        }


class PersonDetector:
    """Batched RF-DETR person detector.

    Thread safety: :meth:`detect` holds a lock. One engine thread drives it, and
    serialising any stray caller is cheaper than debugging two CUDA streams
    fighting over one model.
    """

    def __init__(
        self,
        *,
        variant: str = "nano",
        precision: str = "fp16",
        device: str = "auto",
        threshold: float = 0.5,
        track_floor: float = 0.0,
        nms_iou: float = 0.8,
        resolution: int = 0,
        trace: bool = True,
        max_traced_shapes: int = 4,
        weights_dir: str | Path | None = None,
        checkpoint: str | Path | None = None,
        verify_checkpoint: bool = True,
    ):
        if variant not in _VARIANTS:
            raise ValueError(f"unknown detector variant {variant!r} (expected one of {sorted(_VARIANTS)})")
        if precision not in ("fp16", "fp32"):
            raise ValueError(f"unknown precision {precision!r} (expected fp16 or fp32)")
        self.variant = variant
        self.precision = precision
        self.device_preference = device
        self.threshold = float(threshold)
        self.track_floor = float(track_floor)
        self.nms_iou = float(nms_iou)
        self.requested_resolution = int(resolution)
        self.trace = bool(trace)
        self.max_traced_shapes = int(max_traced_shapes)
        self.weights_dir = Path(weights_dir) if weights_dir else None
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.verify_checkpoint = bool(verify_checkpoint)

        self.stats = DetectorStats()
        self.resolution: int = 0
        self.device: Any = None
        self.dtype: Any = None
        self.backend: str = "unloaded"
        # Which output slot carries the person score. The COCO default is the
        # raw category id; a fine-tune's own provenance may say otherwise, and
        # nothing else in the file could tell us. See argus/vision/checkpoint.py.
        self.person_class_index: int = COCO_PERSON_CLASS_ID
        self.provenance: Any = None

        self._model: Any = None
        self._traced: dict[int, Any] = {}
        self._trace_failed = False
        self._mean: Any = None
        self._std: Any = None
        self._lock = threading.Lock()

    @property
    def emit_floor(self) -> float:
        """The lowest score that leaves this detector.

        ``threshold`` is the detector's operating point. ``track_floor`` lowers
        what is *emitted* below it, for a consumer that knows what to do with a
        doubtful box — the tracker, which lets one continue an existing track
        and never start a new one. A floor of 0, or one at or above the
        threshold, means "emit at the threshold", which is every caller that
        never asked for this (and the eval harness, which sets ``threshold`` to
        0.01 and must get exactly that).
        """
        if 0.0 < self.track_floor < self.threshold:
            return self.track_floor
        return self.threshold

    # --- loading ----------------------------------------------------------

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> "PersonDetector":
        """Build the model. Called on first :meth:`detect` unless done earlier."""
        if self._model is not None:
            return self
        import torch

        if self.weights_dir is not None:
            # RF_HOME is RF-DETR's weight cache. Pointing it inside the appliance
            # data dir keeps every downloaded artefact under one directory that
            # backup and retention already know about.
            self.weights_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("RF_HOME", str(self.weights_dir))

        import rfdetr

        device = self._resolve_device(torch)
        model_cls = getattr(rfdetr, _VARIANTS[self.variant])
        # A fine-tuned checkpoint is handed to RF-DETR as `pretrain_weights` and
        # nothing else: `num_classes` is deliberately *not* passed, because
        # leaving it unset is what makes `load_pretrain_weights` treat the
        # checkpoint's own head width as authoritative and resize the head to
        # match it. Passing a number here would pin the head and silently
        # reinitialise the rows the fine-tune spent its epochs on.
        wrapper = model_cls(**self._checkpoint_kwargs())
        core = wrapper.model.model
        self.resolution = self.requested_resolution or int(wrapper.model.resolution)
        self._check_person_slot(core)

        core.eval().to(device)
        # export() swaps forward() for the tuple-returning inference path
        # (boxes, logits) — the same one RF-DETR's own optimizer traces.
        core.export()
        dtype = torch.float16 if self.precision == "fp16" else torch.float32
        core.to(dtype=dtype)

        self._model = core
        self.device = device
        self.dtype = dtype
        self._mean = torch.tensor(_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
        self._std = torch.tensor(_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
        self.backend = f"torch-{self.precision}" + ("-traced" if self.trace else "-eager")
        # The wrapper holds a postprocessor and class-name tables we do not use;
        # dropping it now keeps only the module we actually run.
        del wrapper
        return self

    def _checkpoint_kwargs(self) -> dict[str, Any]:
        """``{}`` for the stock COCO weights, or the fine-tune plus its provenance.

        Reading the sidecar here rather than in ``__init__`` keeps construction
        free of I/O — ``argus status`` and every test build a detector without
        touching the disk — while still failing before a single weight is loaded.
        """
        if self.checkpoint is None:
            return {}
        from argus.vision.checkpoint import load_provenance

        self.provenance = load_provenance(
            self.checkpoint, verify_digest=self.verify_checkpoint
        )
        self.person_class_index = self.provenance.person_class_index
        return {"pretrain_weights": str(self.checkpoint.resolve())}

    def _check_person_slot(self, core: Any) -> None:
        """The head is wide enough for the slot ``_postprocess`` will index.

        The one failure this catches is the expensive one: a head narrower than
        the person index reads as an ordinary IndexError deep inside inference,
        thousands of frames after start-up, on a box nobody is watching.
        """
        width = getattr(getattr(core, "class_embed", None), "out_features", None)
        if isinstance(width, int) and self.person_class_index >= width:
            from argus.vision.checkpoint import CheckpointError

            raise CheckpointError(
                f"{self.checkpoint or 'the detector'} has a {width}-wide class head, "
                f"but person is recorded at index {self.person_class_index}. "
                f"These weights and this provenance are not the same model."
            )

    def _resolve_device(self, torch: Any) -> Any:
        if self.device_preference != "auto":
            return torch.device(self.device_preference)
        if torch.cuda.is_available():
            return torch.device("cuda")
        # CPU inference is ~50x slower and is not a deployment mode; it exists so
        # the detector smoke test can run on a machine without a GPU.
        return torch.device("cpu")

    # --- inference --------------------------------------------------------

    def detect(self, images: Sequence[np.ndarray]) -> list[Any]:
        """Detect people in a batch of BGR frames.

        Returns one ``supervision.Detections`` per image, with boxes in that
        image's own pixel coordinates (the frames in a batch do not have to be
        the same size).
        """
        import supervision as sv

        if not images:
            return []
        self.load()
        with self._lock:
            boxes, logits = self._forward(images)
            results = self._postprocess(boxes, logits, images)
        return [
            sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
            for xyxy, confidence, class_id in results
        ]

    def _forward(self, images: Sequence[np.ndarray]) -> tuple[Any, Any]:
        import torch

        batch = self._preprocess(images)
        started = time.perf_counter()
        with torch.inference_mode():
            model = self._model_for(batch.shape[0], batch)
            outputs = model(batch)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
        self.stats.record(len(images), (time.perf_counter() - started) * 1000.0)
        boxes, logits = outputs[0], outputs[1]
        return boxes.float(), logits.float()

    def _model_for(self, batch_size: int, example: Any) -> Any:
        """The eager module, or a trace specialised to this batch size.

        A trace is shape-specialised, so it is cached per batch size; the engine
        only ever presents "number of cameras with a fresh frame", which is a
        handful of distinct values. Tracing shares parameter storage with the
        eager model, so the cache costs no extra weight memory. Any failure to
        trace disables tracing permanently rather than retrying every batch.
        """
        if not self.trace or self._trace_failed:
            return self._model
        traced = self._traced.get(batch_size)
        if traced is not None:
            return traced
        if len(self._traced) >= self.max_traced_shapes:
            return self._model
        import torch

        try:
            with torch.inference_mode():
                traced = torch.jit.trace(self._model, example.clone())
                traced(example)      # prove the trace runs before adopting it
        except Exception:  # noqa: BLE001 - tracing is an optimisation, never a requirement
            self._trace_failed = True
            self.backend = f"torch-{self.precision}-eager"
            return self._model
        self._traced[batch_size] = traced
        return traced

    def _preprocess(self, images: Sequence[np.ndarray]) -> Any:
        """BGR uint8 frames -> normalized NCHW tensor on the device.

        Everything after the host-to-device copy runs on the GPU: the CPU only
        stacks the raw bytes, which matters because the Grace cores are also
        running ffmpeg for every camera.
        """
        import torch

        heights = {image.shape[0] for image in images}
        widths = {image.shape[1] for image in images}
        if len(heights) == 1 and len(widths) == 1:
            stacked = np.stack(images) if len(images) > 1 else images[0][None]
            batch = self._to_model_space(self._upload(stacked))
        else:
            # Mixed frame sizes: resize each separately, then stack. Cameras
            # normally share the detection resolution, so this is the rare path.
            batch = torch.cat([self._to_model_space(self._upload(image[None])) for image in images])
        return batch

    def _upload(self, array: np.ndarray) -> Any:
        import torch

        array = np.ascontiguousarray(array)
        if not array.flags.writeable:
            # torch.from_numpy refuses to share storage it cannot write to.
            # Decoder frames are writable, so this copy only happens for
            # read-only callers (memory-mapped or frombuffer(bytes) images).
            array = array.copy()
        return torch.from_numpy(array).to(self.device, non_blocking=True)

    def _to_model_space(self, tensor: Any) -> Any:
        import torch
        import torch.nn.functional as F

        # NHWC BGR uint8 -> NCHW RGB float in [0, 1]
        tensor = tensor[..., [2, 1, 0]].permute(0, 3, 1, 2).to(self.dtype).div_(255.0)
        size = (self.resolution, self.resolution)
        if tensor.shape[-2:] != size:
            # antialias=False matches RF-DETR's own resize (and the antialias-free
            # bilinear used in its training pipeline); turning it on shifts scores.
            tensor = F.interpolate(tensor, size=size, mode="bilinear", align_corners=False, antialias=False)
        return (tensor - self._mean) / self._std

    def _postprocess(
        self, boxes: Any, logits: Any, images: Sequence[np.ndarray]
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Person boxes in source-frame pixels, per image.

        RF-DETR's own postprocessor takes the top-k over *all* classes; we only
        ever want people, so this takes the person logit per query directly —
        same arithmetic, one class wide, and it cannot spend the k budget on
        forklifts in a busy frame.

        ``person_class_index`` is the COCO category id on the stock weights and
        whatever a fine-tune's provenance records otherwise; ``load`` has
        already proved the head is that wide.
        """
        import torch
        from torchvision.ops import nms

        scores = logits[:, :, self.person_class_index].sigmoid()
        cx, cy, w, h = boxes.unbind(-1)
        xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)

        results = []
        for index, image in enumerate(images):
            keep = scores[index] > self.emit_floor
            image_scores = scores[index][keep]
            image_boxes = xyxy[index][keep]          # advanced indexing copies
            height, width = image.shape[:2]
            # Normalized -> this image's pixels, column-wise so no scale tensor
            # has to be built and uploaded per frame.
            image_boxes[:, 0::2] = image_boxes[:, 0::2].mul(width).clamp_(min=0, max=width)
            image_boxes[:, 1::2] = image_boxes[:, 1::2].mul(height).clamp_(min=0, max=height)
            if self.nms_iou < 1.0 and len(image_boxes) > 1:
                # DETR-family models need no NMS in principle; this is a cheap
                # guard against two queries locking onto one person and inflating
                # a headcount, at an IoU high enough not to merge two people.
                order = nms(image_boxes, image_scores, self.nms_iou)
                image_boxes, image_scores = image_boxes[order], image_scores[order]
            results.append((
                image_boxes.cpu().numpy().astype(np.float32).reshape(-1, 4),
                image_scores.cpu().numpy().astype(np.float32),
                np.full(len(image_scores), PERSON_CLASS_ID, dtype=int),
            ))
        return results

    # --- introspection ----------------------------------------------------

    def describe(self) -> dict[str, Any]:
        described: dict[str, Any] = {
            "model": f"rf-detr-{self.variant}",
            "backend": self.backend,
            "device": str(self.device) if self.device is not None else self.device_preference,
            "resolution": self.resolution,
            "threshold": self.threshold,
            "traced_batch_sizes": sorted(self._traced),
            **self.stats.as_dict(),
        }
        if self.provenance is not None:
            # "stock" is the absence of this key. Anything reading `describe()`
            # — the engine banner, `argus status`, the API's system route —
            # therefore says which weights are running without being changed.
            described["checkpoint"] = self.provenance.as_dict()
            described["model"] = f"rf-detr-{self.variant}+finetune"
        return described

    def weights_summary(self) -> str:
        """One line naming the weights, for a startup log.

        Named rather than implied: the whole risk of an optional checkpoint is a
        site running one and nobody knowing, or a site *not* running one and
        everybody assuming it is.
        """
        if self.provenance is None:
            return f"rf-detr-{self.variant} stock COCO weights"
        return f"rf-detr-{self.variant} fine-tune: {self.provenance.summary()}"


def detector_from_config(config: Any) -> PersonDetector:
    """Build the detector described by ``detection.model`` in the config."""
    model = config.detection.model
    return PersonDetector(
        variant=model.variant,
        precision=model.precision,
        device=model.device,
        threshold=model.threshold,
        track_floor=model.track_floor,
        nms_iou=model.nms_iou,
        resolution=model.resolution,
        trace=model.trace,
        weights_dir=config.models_dir,
        checkpoint=config.detector_checkpoint_path,
    )


__all__ = [
    "COCO_PERSON_CLASS_ID",
    "PERSON_CLASS_ID",
    "Detector",
    "DetectorStats",
    "PersonDetector",
    "detector_from_config",
]
