"""``argus index backfill`` — make the evidence already on disk searchable.

Turning the index on should not mean the appliance can only find what happened
after lunch. Every event this box has ever raised has an annotated snapshot
under ``data/snapshots``, kept for 30 days by retention, and every one of them
is a moment somebody may need to search for. This walks them.

Two things it deliberately does *not* do:

* **It does not re-embed what is already indexed.** The store knows which event
  ids it holds, so a second run is cheap and a run interrupted at 40% resumes.
  ``--reset`` is the escape hatch for the one case where that is wrong: the
  configured model changed, so every existing vector is in a different space.
* **It does not read the ring buffer.** The rolling footage is 300 seconds deep
  and is the *live* window; sampling it here would produce five minutes of index
  and the impression that history had been recovered.

Backfill embeds the annotated snapshot rather than the source frame, because
the source frame is gone — it existed in RAM at the moment of the event. That
is a real difference from the live path, which embeds the clean frame, and it
is not a small one: **SigLIP reads the burned-in headline.**

Measured on this box, and the demo makes it unmissable because both demo
cameras play the same video file — so the *only* systematic difference between
their snapshots is the zone name in the banner:

=========================  ======================  ==================
mean score over 3 748 ev.  camera "Conveyor side"  camera "Packing floor"
=========================  ======================  ==================
"person near the conveyor"        0.1403                 0.1096
"packing floor"                   0.1256                 0.1387
"a person carrying a box"         0.0913                 0.0940
=========================  ======================  ==================

Each query wins on the camera whose banner contains its words, by ~0.03 on a
scale where a good match is 0.10-0.17 — a quarter of the signal — while a query
naming neither zone splits evenly. The scene contributed nothing; the caption
we drew on it did.

So backfill **crops the banner strip off before embedding**
(:func:`argus.media.annotate.banner_height` is the same formula that drew it).
The box labels and the zone shading survive the crop and are left alone: they
are pixels-worth of "Worker 7 · 88%" rather than a sentence naming the place,
and cropping the whole overlay away would mean cropping the boxes that are the
evidence. The residual difference after cropping is recorded in
docs/PHASE3_NOTES.md — it is small, and it is not zero.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from argus.config import ArgusConfig
from argus.db import Database
from argus.index.embedder import Embedder, embedder_from_config
from argus.index.store import IndexedFrame, VectorStore, store_from_config


@dataclass
class BackfillResult:
    scanned: int = 0
    indexed: int = 0
    skipped_indexed: int = 0
    missing_snapshot: int = 0
    failed: int = 0
    duration_sec: float = 0.0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "scanned": self.scanned,
            "indexed": self.indexed,
            "skipped_indexed": self.skipped_indexed,
            "missing_snapshot": self.missing_snapshot,
            "failed": self.failed,
            "duration_sec": round(self.duration_sec, 2),
            "errors": self.errors,
        }

    def __str__(self) -> str:
        rate = f"{self.indexed / self.duration_sec:.1f}/s" if self.duration_sec else "-"
        return (
            f"backfill: {self.indexed} event frame(s) indexed from {self.scanned} scanned "
            f"({self.skipped_indexed} already indexed, {self.missing_snapshot} without a "
            f"snapshot, {self.failed} failed) in {self.duration_sec:.1f}s — {rate}"
        )


def load_bgr(path: Path, *, crop_banner: bool = False) -> np.ndarray:
    """A JPEG on disk as the BGR array the embedder expects.

    BGR because that is what the decoder produces and what the live path
    embeds; a backfilled vector has to land in the same space as a live one,
    and a channel swap between the two would be invisible until somebody
    noticed search worked better for recent footage.

    ``crop_banner`` removes the headline strip an evidence snapshot carries at
    the top — the module docstring has the measurement that makes this
    necessary. It is off by default because this function is also how tests and
    tools read an ordinary picture off disk.
    """
    from PIL import Image

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        if crop_banner:
            from argus.media.annotate import banner_height

            top = banner_height(rgb.width)
            # Never crop into the scene proper: a pathologically small snapshot
            # would otherwise come back as a sliver or an empty array.
            if top < rgb.height // 4:
                rgb = rgb.crop((0, top, rgb.width, rgb.height))
        return np.ascontiguousarray(np.asarray(rgb)[:, :, ::-1])


def image_size(path: Path) -> tuple[int, int]:
    """``(width, height)`` from the JPEG header, without decoding the pixels."""
    from PIL import Image

    with Image.open(path) as image:
        return int(image.width), int(image.height)


def backfill_events(
    config: ArgusConfig,
    database: Database,
    *,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    reset: bool = False,
    limit: int | None = None,
    batch_size: int | None = None,
    log: Callable[..., None] = print,
    progress_every: int = 200,
) -> BackfillResult:
    """Embed every event snapshot that is on disk and not yet in the store."""
    started = time.monotonic()
    result = BackfillResult()

    embedder = embedder or embedder_from_config(config)
    store = store or store_from_config(config, dim=embedder.dim, model=embedder.name)
    if reset:
        log("index: resetting the vector store")
        store.reset()

    already = set() if reset else store.indexed_event_ids()
    batch_size = batch_size or config.index.batch_size

    # Newest first: an operator who interrupts a long backfill keeps the window
    # they are most likely to search.
    events = database.recent_events(limit=limit or 1_000_000)
    # (event row, cropped frame to embed, size of the file on disk). The size is
    # the *uncropped* one because `thumb_path` points at the uncropped file:
    # what the store records has to describe what the browser will fetch.
    pending: list[tuple[dict[str, Any], np.ndarray, tuple[int, int]]] = []

    def flush() -> None:
        if not pending:
            return
        try:
            vectors = embedder.embed_images([image for _, image, _ in pending])
        except Exception as exc:  # noqa: BLE001 - a bad batch is not a bad run
            result.failed += len(pending)
            if len(result.errors) < 10:
                result.errors.append(f"embed failed: {type(exc).__name__}: {exc}")
            pending.clear()
            return
        rows = [
            IndexedFrame(
                camera_id=str(event["camera_id"]),
                ts=float(event["ts"]),
                kind="event",
                embedding=vector,
                thumb_path=str(event["snapshot_path"]),
                thumb_kind="snapshot",
                event_id=int(event["id"]),
                width=size[0],
                height=size[1],
            )
            for (event, _image, size), vector in zip(pending, vectors)
        ]
        result.indexed += store.add(rows)
        pending.clear()

    for event in events:
        result.scanned += 1
        event_id = int(event["id"])
        if event_id in already:
            result.skipped_indexed += 1
            continue
        snapshot = event.get("snapshot_path")
        if not snapshot:
            result.missing_snapshot += 1
            continue
        path = config.snapshots_dir / str(snapshot)
        if not path.is_file():
            # Retention deleted it (30-day window) but kept the event row for
            # 180. There is nothing to embed and nothing to show.
            result.missing_snapshot += 1
            continue
        try:
            size = image_size(path)
            image = load_bgr(path, crop_banner=True)
        except Exception as exc:  # noqa: BLE001
            result.failed += 1
            if len(result.errors) < 10:
                result.errors.append(f"{snapshot}: {type(exc).__name__}: {exc}")
            continue
        pending.append((event, image, size))
        if len(pending) >= batch_size:
            flush()
            if progress_every and result.indexed and result.indexed % progress_every < batch_size:
                log(f"index: {result.indexed} indexed…")
    flush()

    result.duration_sec = time.monotonic() - started
    return result


__all__ = ["BackfillResult", "backfill_events", "image_size", "load_bgr"]
