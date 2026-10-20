"""The index service (Phase 3): natural-language search over recorded footage.

Four pieces, each with its own module docstring for the *why*:

============  ==========================================================
module        role
============  ==========================================================
`embedder`    SigLIP 2 — the dual tower that puts frames and English in
              one vector space. Also the deterministic fake every test
              runs against.
`store`       sqlite-vec in its own SQLite file: kNN with camera and
              time pushed into the scan, plus the prunes retention calls.
`runner`      the writer inside ``argus run`` — periodic per-camera
              samples and event frames, batched onto the GPU behind a
              bounded queue that drops rather than blocks.
`backfill`    ``argus index backfill``: the evidence already on disk.
============  ==========================================================

Nothing here is imported by ``argus.engine`` at module scope — a run with
``index.enabled: false`` never pays for transformers or torch on this path.
"""

from __future__ import annotations

__all__ = ["backfill", "embedder", "runner", "store"]
