"""SigLIP 2 — the one model that turns pixels and English into the same numbers.

Semantic search on this appliance is not a language model reading captions. It
is a single dual-tower encoder: the vision tower embeds a frame, the text tower
embeds the operator's sentence, and the answer to "person near conveyor" is a
dot product against every frame this box has ever indexed. That is why a query
costs milliseconds rather than the ~6 s a VLM policy evaluation costs — the
expensive half was paid once, when the frame was recorded.

**Why SigLIP 2 and not the VLM we already run.** NVIDIA's VSS 3.2 blueprint uses
SigLIP 2 for exactly this role, and the arithmetic is the reason: asking
Qwen3-VL-8B about one frame takes 6.1 s on this box with the detector running
(docs/PHASE2_NOTES.md), so a day of one camera at 10 s intervals — 8 640 frames
— would be 14.6 hours of inference. The same day costs **29 seconds** here.
Search has to index everything; policies only have to look at now.

**The checkpoint: ``google/siglip2-base-patch16-384``.** Both candidates were
downloaded and measured on this GB10 against the other one, ``so400m-patch14-384``,
which is the quality leader of the family (2026-08-15, 74 real 960x544 frames):

===========================  ========  ====  =============  ============  ===========
checkpoint                   params    dim   fp16 on GPU    ms/frame @b8  best batch
===========================  ========  ====  =============  ============  ===========
siglip2-base-patch16-384     375 M     768   0.77 GB        **3.35**      3.35 @b8
siglip2-so400m-patch14-384   1 136 M   1152  2.33 GB        30.0          16.3 @b4
===========================  ========  ====  =============  ============  ===========

Both are Apache-2.0 and ungated (checked against the HF API before either was
downloaded — see docs/LICENSES.md). so400m costs **4.9x** as much per frame even
when each model is given the batch size it likes best, and on the two clips
available here it did not retrieve better: both put the packing line above the
aerial-ocean clip on every factory probe and the ocean above the packing line on
the ocean probe, with absolute scores within a few hundredths of each other in
*both* directions. The footprint that decides this is not disk — it is SMs
shared with RF-DETR *and* vLLM, and 3.35 ms/frame means a 48-camera fleet at a
10 s sample interval costs about **1.6% of one GPU**.

Two caveats that belong next to those numbers, both measured and both in
docs/PHASE3_NOTES.md: the cost curve is **not** monotonic in batch size (base is
3.35 ms/frame at 8 and 5.32 at 16, which is why ``batch_size`` defaults to 8 and
should not be raised without re-measuring), and neither model does negation —
"an empty factory floor with no people" retrieves the *most crowded* frames.

The id is configuration (``index.model``), the store records which model wrote
each vector, and mixing two is refused rather than silently ranked — so
revisiting this on real NVR footage is a re-backfill, not a migration.

**Local, like everything else at runtime.** Weights load from ``models/`` on
disk (``scripts/fetch_siglip.sh`` puts them there), never from the hub. The
embedder never opens a socket.

Two implementations behind one protocol: :class:`SiglipEmbedder` and
:class:`FakeEmbedder`, the latter deterministic and CPU-only so every test of
the store, the runner, the API and retention runs without a GPU or a download.
"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

# SigLIP was trained with a fixed 64-token text sequence and it is not an
# implementation detail: padding to the longest string in the batch instead
# changes the embedding of the *same* query depending on what it was batched
# with. Every text tower call in this file pads to exactly this.
TEXT_LENGTH = 64

# Prepended to a bare operator query before it reaches the text tower. SigLIP's
# training captions are alt-text sentences, so a naked noun phrase ("forklift")
# sits further from the image manifold than the same phrase in a caption frame.
# Applied in one place so the query stored in a saved search stays the
# operator's own words.
QUERY_TEMPLATE = "a photo of {}"


class Embedder(Protocol):
    """What the index needs from a model. Both towers, one vector space."""

    name: str
    dim: int

    def embed_images(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """BGR uint8 frames -> ``(n, dim)`` float32, L2-normalized."""

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Query strings -> ``(n, dim)`` float32, L2-normalized."""


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize rows, so cosine similarity is a plain dot product.

    Done here rather than at query time because it is what lets the store use
    sqlite-vec's ``cosine`` metric on ready-made unit vectors, and what makes a
    score comparable between two rows written months apart.
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


class SiglipEmbedder:
    """The real thing: a SigLIP 2 checkpoint on the GPU (or CPU).

    Thread-safe by lock rather than by luck. In the shipped layout the writer
    and the query tower are different instances in different processes, so the
    lock never contends; it is here because nothing in the class *enforces*
    that layout, and two concurrent forward passes on one module is not a
    documented guarantee in torch.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "auto",
        precision: str = "fp16",
        text_only: bool = False,
        name: str | None = None,
    ):
        import torch
        from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

        path = Path(model_path)
        if not (path / "config.json").is_file():
            raise FileNotFoundError(
                f"SigLIP checkpoint not found at {path} — run scripts/fetch_siglip.sh"
            )
        self._torch = torch
        self.name = name or str(model_path)
        self.text_only = text_only
        self.device = _resolve_device(device, torch)
        # fp16 on the GPU, fp32 on the CPU: half precision on Grace cores is
        # emulated and measurably slower than the float32 path it replaces.
        self.dtype = (
            torch.float16
            if precision == "fp16" and self.device.startswith("cuda")
            else torch.float32
        )

        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        if text_only:
            # The API process only ever embeds queries, so it loads the text
            # tower alone: 282 M parameters (196 M of which is the 256k-token
            # embedding table SigLIP 2 inherited from Gemma), 1.0 GB of RSS,
            # 2.8 s to load. Loading the whole checkpoint there would add the
            # 93 M-parameter vision tower — and, on `query_device: cuda`, a
            # second CUDA context — for weights no request can reach.
            from transformers import SiglipTextModel

            self.model = SiglipTextModel.from_pretrained(str(path), dtype=self.dtype)
            self.processor = None
        else:
            self.processor = AutoImageProcessor.from_pretrained(str(path))
            self.model = AutoModel.from_pretrained(str(path), dtype=self.dtype)
        self.model = self.model.eval().to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        config = self.model.config
        text_config = getattr(config, "text_config", config)
        self.dim = int(getattr(text_config, "hidden_size", 0) or 0)
        if not self.dim:  # pragma: no cover - every published checkpoint has it
            raise ValueError(f"cannot determine embedding width for {path}")
        self._lock = threading.Lock()

    # --- towers -----------------------------------------------------------

    @staticmethod
    def _pooled(output: Any) -> Any:
        """transformers 5 returns a ``BaseModelOutputWithPooling`` where 4
        returned a tensor. Accept both rather than pin the major version."""
        return getattr(output, "pooler_output", output)

    # --- preprocessing ----------------------------------------------------
    #
    # Resizing 960x544 frames to 384x384 is not a rounding error in this
    # pipeline's cost. Measured on this box at batch 8 (2026-08-15):
    # HuggingFace's image processor takes **11.8 ms** on the Grace cores
    # against **24.4 ms** for the forward pass itself, and the cores are
    # precisely the resource the vision loop is already short of
    # (docs/BENCHMARKS.md: the ceiling is the GIL, not the GPU). Doing the same
    # arithmetic on the GPU takes **2.1 ms** and takes it off the cores
    # entirely — 36 -> 26.8 ms end to end per batch, all of the saving in
    # Python-visible CPU time.
    #
    # The catch is that PIL's bilinear downscale antialiases and torch's does
    # not by default, which is a real difference at a 2.5x reduction:
    # `antialias=False` agrees with the processor to a cosine of only 0.972,
    # while `antialias=True` agrees to **0.9999** (0.99985 minimum over 16 real
    # frames, re-measured). So the GPU path is used with antialiasing on, and
    # the processor stays as the CPU path — where there is no GPU to move the
    # work to, and where being byte-identical to the reference implementation
    # is worth more than speed.

    def _pixels_gpu(self, images: Sequence[np.ndarray]) -> Any:
        """BGR uint8 frames -> the normalized NCHW tensor SigLIP expects."""
        torch = self._torch
        size = int(self.model.config.vision_config.image_size)
        mean, std = 0.5, 0.5  # every published SigLIP checkpoint; see preprocessor_config

        def one(image: np.ndarray) -> Any:
            tensor = torch.from_numpy(np.ascontiguousarray(image)).to(self.device)
            # HWC BGR uint8 -> 1CHW RGB float. float32 for the resize: an
            # fp16 bilinear kernel loses enough precision on a 2.5x reduction
            # to show up in the third decimal of a score.
            tensor = tensor.permute(2, 0, 1)[[2, 1, 0]].unsqueeze(0).to(torch.float32).div_(255.0)
            return torch.nn.functional.interpolate(
                tensor, size=(size, size), mode="bilinear",
                align_corners=False, antialias=True,
            )

        shapes = {image.shape for image in images}
        if len(shapes) == 1:
            # The common case — one camera resolution, or a whole batch of
            # them. Resize once for the batch rather than once per frame.
            stacked = torch.from_numpy(np.ascontiguousarray(np.stack(images))).to(self.device)
            stacked = stacked.permute(0, 3, 1, 2)[:, [2, 1, 0]].to(torch.float32).div_(255.0)
            pixels = torch.nn.functional.interpolate(
                stacked, size=(size, size), mode="bilinear",
                align_corners=False, antialias=True,
            )
        else:
            # Backfill reads snapshots written at whatever resolution their
            # camera used, so a mixed batch is normal there.
            pixels = torch.cat([one(image) for image in images])
        return pixels.sub_(mean).div_(std).to(self.dtype)

    def embed_images(self, images: Sequence[np.ndarray]) -> np.ndarray:
        if self.text_only:
            raise RuntimeError("this embedder was loaded text-only")
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        torch = self._torch
        with self._lock:
            if self.device.startswith("cuda"):
                pixels = self._pixels_gpu(images)
            else:
                # Frames are BGR from the decoder; the processor wants RGB, and
                # getting this backwards is invisible — it degrades ranking
                # rather than raising.
                assert self.processor is not None
                rgb = [np.ascontiguousarray(image[:, :, ::-1]) for image in images]
                pixels = self.processor(images=rgb, return_tensors="pt")["pixel_values"]
                pixels = pixels.to(self.device, self.dtype)
            with torch.no_grad():
                features = self._pooled(self.model.get_image_features(pixel_values=pixels))
            return _normalize(features.float().cpu().numpy())

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        torch = self._torch
        encoded = self.tokenizer(
            list(texts),
            padding="max_length",
            max_length=TEXT_LENGTH,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self._lock:
            with torch.no_grad():
                if self.text_only:
                    features = self._pooled(self.model(**encoded))
                else:
                    features = self._pooled(self.model.get_text_features(**encoded))
            return _normalize(features.float().cpu().numpy())

    def describe(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "dim": self.dim,
            "device": self.device,
            "precision": str(self.dtype).replace("torch.", ""),
            "text_only": self.text_only,
        }


def _resolve_device(device: str, torch: Any) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


class FakeEmbedder:
    """A deterministic stand-in with no model, no GPU and no download.

    Every test of the *plumbing* — the store's kNN, the runner's queue,
    retention's pruning, the API's filters, the promote flow — runs against
    this, so the whole of Phase 3 is testable on a CI box with no CUDA and no
    17 GB of weights. Tests about retrieval *quality* are not unit tests: they
    are the live verification recorded in docs/PHASE3_NOTES.md.

    It is not only noise. Both towers reduce their input to the same three-byte
    token, so an image :meth:`paint`ed for a phrase embeds to **exactly** the
    vector that phrase embeds to. A test can therefore assert that searching
    for "conveyor" ranks the conveyor frame first and the ocean frame last,
    with real cosine scores and no semantics anywhere — which is the only way
    to test a ranking endpoint without shipping a model to CI.

    The image token is the mean pixel quantised to buckets of 8, so it survives
    the JPEG round trip a backfilled frame makes on its way back off disk.
    """

    _BUCKET = 8

    def __init__(self, dim: int = 64, name: str = "fake-embedder"):
        self.dim = dim
        self.name = name

    # --- the shared token -------------------------------------------------

    @classmethod
    def _text_token(cls, text: str) -> bytes:
        digest = hashlib.sha256(text.strip().lower().encode()).digest()
        # Land in the middle of a bucket, so ±1 of JPEG error cannot move it.
        return bytes((value // cls._BUCKET) * cls._BUCKET + cls._BUCKET // 2
                     for value in digest[:3])

    @classmethod
    def _image_token(cls, image: np.ndarray) -> bytes:
        # Floor division, matching `_text_token` exactly: a painted colour sits
        # in the middle of its bucket, so ±1 of JPEG error cannot move it into
        # the next one, and the two towers agree on the byte.
        mean = np.asarray(image, dtype=np.float32).mean(axis=(0, 1))
        return bytes(
            int(value) // cls._BUCKET * cls._BUCKET + cls._BUCKET // 2 for value in mean
        )

    @classmethod
    def paint(cls, text: str, shape: tuple[int, int] = (32, 32)) -> np.ndarray:
        """A solid BGR frame that embeds to the same vector as ``text``."""
        colour = cls._text_token(text)
        return np.full((shape[0], shape[1], 3), list(colour), dtype=np.uint8)

    def _vector(self, token: bytes) -> np.ndarray:
        digest = hashlib.sha256(token).digest()
        state = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        return state.standard_normal(self.dim, dtype=np.float32)

    # --- towers -----------------------------------------------------------

    def embed_images(self, images: Sequence[np.ndarray]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize(np.stack([self._vector(self._image_token(i)) for i in images]))

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize(np.stack([self._vector(self._text_token(t)) for t in texts]))

    def describe(self) -> dict[str, Any]:
        return {"model": self.name, "dim": self.dim, "device": "cpu", "precision": "none"}


def embedder_from_config(
    config: Any, *, text_only: bool = False, device: str | None = None
) -> Embedder:
    """Build the configured embedder. ``index.model: fake`` selects the stub.

    ``device`` overrides ``index.device`` for one caller — the search API uses
    it to keep the query tower on the CPU while the engine's writer runs on the
    GPU, without the two having to share a setting they disagree about.

    The fake is reachable from configuration on purpose: it is how the API, the
    engine and the CLI are exercised end to end on a box with no checkpoint —
    and how a demo appliance can show the *screen* working before the weights
    are installed, without any of them faking a result.
    """
    settings = config.index
    if settings.model == "fake":
        return FakeEmbedder()
    return SiglipEmbedder(
        config.index_model_path,
        device=device or settings.device,
        precision=settings.precision,
        text_only=text_only,
        name=settings.model,
    )


__all__ = [
    "QUERY_TEMPLATE",
    "TEXT_LENGTH",
    "Embedder",
    "FakeEmbedder",
    "SiglipEmbedder",
    "embedder_from_config",
]
