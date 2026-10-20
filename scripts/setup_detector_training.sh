#!/usr/bin/env bash
# Build the virtualenv that fine-tunes the person detector. Run once, before
# `scripts/train_detector.py`.
#
#   ./scripts/setup_detector_training.sh
#
# This env is deliberately NOT the argus env, for the same reason `.venv-vlm`
# is not: training is a lab activity, inference is the product. Three
# consequences, all of them the point.
#
# 1. **The product tree stays what it was.** `scripts/license_audit.py --check`
#    audits `.venv` (enforced) and `.venv-vlm` (reported). Training pulls
#    ~15 packages the appliance never imports; none of them belong in the
#    dependency list a customer's lawyer reads. `.venv-train` is reported by
#    the audit for the same reason `.venv-vlm` is — it exists on this disk, so
#    it is accounted for — and enforced for neither.
#
# 2. **`rfdetr[train]` is not what gets installed.** That extra pulls
#    `roboflow` and `rf100vl`: SDKs whose job is to talk to a cloud dataset
#    host. `rfdetr` imports `roboflow` in exactly one function (a
#    dataset-download helper this project never calls) and never imports
#    `rf100vl` at all. On an appliance whose whole claim is that nothing calls
#    a cloud, installing a cloud SDK to train a local model on local footage is
#    the wrong default even though it would sit inert. The list below is the
#    train extra minus those two, verified against the import graph.
#
# 3. **torch comes from the same index the appliance uses** (cu130, aarch64).
#    A second resolution could quietly pick a different build and then the
#    checkpoint would be trained by a torch the engine does not run.
#
# The training deps are pinned to no more than rfdetr's own declared bounds:
# this venv is rebuilt from scratch when it drifts, and `provenance.json` in
# every training run records the versions that produced those weights.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${ARGUS_TRAIN_VENV:-$ROOT/.venv-train}"
TORCH_INDEX="${ARGUS_TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"

echo "venv:  $VENV"
echo "torch: $TORCH_INDEX (the appliance's index — same build the engine runs)"

if ! command -v uv >/dev/null; then
    echo "uv is required (https://docs.astral.sh/uv/)" >&2
    exit 1
fi

# Match the appliance's interpreter: the checkpoint is loaded by `.venv`, and
# a pickle written by a different minor is a needless risk.
uv venv --python "$(cat "$ROOT/.python-version")" "$VENV"

VIRTUAL_ENV="$VENV" uv pip install \
    --extra-index-url "$TORCH_INDEX" \
    --index-strategy unsafe-best-match \
    "torch==2.12.1" \
    "torchvision==0.27.1" \
    "rfdetr==1.9.1" \
    "pytorch_lightning>=2.6,<3,!=2.6.2,!=2.6.3" \
    "torchmetrics[detection]>=1.2" \
    "faster-coco-eval>=1.7.2" \
    "pycocotools" \
    "scipy" \
    "pyyaml"

echo
"$VENV/bin/python" - <<'PY'
import torch

import pytorch_lightning
import rfdetr
from rfdetr.training import RFDETRDataModule  # the import that proves the extras are complete

print(f"rfdetr             {rfdetr.__version__ if hasattr(rfdetr, '__version__') else '1.9.1'}")
print(f"pytorch_lightning  {pytorch_lightning.__version__}")
print(f"torch              {torch.__version__} (cuda {torch.version.cuda})")
print(f"gpu                {torch.cuda.get_device_name(0)}")
PY

echo
echo "licences in this venv (nothing is enforced here — see the header):"
"$ROOT/.venv/bin/python" "$ROOT/scripts/license_audit.py" --tree "$VENV" || true

echo
echo "done. Next: uv run python scripts/train_detector.py --help"
