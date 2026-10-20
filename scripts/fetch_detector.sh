#!/usr/bin/env bash
# Pre-stage the RF-DETR person-detector checkpoint, once, at install time.
#
# Why this script exists
# ----------------------
# It is the last runtime network fetch in the product, and until now it was an
# invisible one. RF-DETR downloads its weights on the FIRST call to `load()` —
# which, on a fresh appliance, is the first frame of the first camera, on a
# plant's uplink, minutes after somebody left the room. Every other artefact
# (go2rtc, ffmpeg, SigLIP, the VLM) is fetched by a named script; this one was
# fetched by the vision engine.
#
# Two consequences, both fixed here:
#
#   * An air-gapped site could pass every install step and still have an engine
#     that never produces a frame. `scripts/install.sh` now runs this, and
#     `argus doctor` FAILS when the checkpoint is absent rather than warning.
#   * The systemd units set HF_HUB_OFFLINE=1 to enforce "no network at
#     runtime", and that flag does NOT cover this download — RF-DETR fetches
#     from storage.googleapis.com, not from HuggingFace. Pre-staging is what
#     actually makes the claim true.
#
# The weights land in data/models (RF_HOME), which is where argus/config.py
# points them and where `argus doctor` looks. That is deliberately inside
# data_dir rather than beside the 17 GB VLM checkpoint: it is 366 MB, it is
# what the appliance's own detector config selects, and a support bundle
# listing "which detector is this box running" should see it.
#
#   ./scripts/fetch_detector.sh            # the variant in config/argus.yaml
#   ./scripts/fetch_detector.sh small      # a specific variant
#
# Idempotent: an existing checkpoint is verified as loadable and left alone.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VARIANT="${1:-}"

# The load happens through argus's own detector wrapper, not through rfdetr
# directly, so this fetches exactly the checkpoint the engine will ask for —
# including the variant chosen in config/argus.yaml. Anything else would be a
# second source of truth for "which detector".
uv run python - "$VARIANT" <<'PY'
import sys
from pathlib import Path

from argus.config import load_config
from argus.vision.detector import PersonDetector

variant = sys.argv[1] or None
config = load_config()
config.ensure_dirs()
chosen = variant or config.detection.model.variant

existing = sorted(config.models_dir.glob(f"rf-detr-{chosen}*.pth"))
if existing:
    size = existing[0].stat().st_size / 1e6
    print(f"already present: {existing[0]} ({size:.1f} MB)")
    raise SystemExit(0)

print(f"fetching rf-detr-{chosen} into {config.models_dir}")
# CPU: this is a download, and building the CUDA graph would only add seconds
# and claim GPU memory an installer has no reason to hold.
PersonDetector(variant=chosen, device="cpu", precision="fp32", trace=False,
               weights_dir=config.models_dir).load()

for path in sorted(Path(config.models_dir).glob(f"rf-detr-{chosen}*.pth")):
    print(f"installed {path} ({path.stat().st_size / 1e6:.1f} MB)")
PY

echo "done. Check it with: argus doctor"
