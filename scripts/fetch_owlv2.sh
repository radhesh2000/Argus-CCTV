#!/usr/bin/env bash
# Fetch the OWLv2 open-vocabulary detector used by the hairnet label bootstrap.
#
# This is a DEV-TIME tool and it is deliberately not wired into the appliance.
# `argus run` never imports it, `scripts/install.sh` never fetches it, and
# nothing in `argus/` outside `scripts/bootstrap_hairnet.py` mentions it. It
# exists to turn a directory of plant frames into *candidate* labels a human
# then corrects — the "auto-label with a slow zero-shot model, train a fast
# closed-set one" loop in docs/REBUILD_PLAN.md, step 4 of the detector roadmap.
#
#   ./scripts/fetch_owlv2.sh                                   # the default below
#   ./scripts/fetch_owlv2.sh google/owlv2-large-patch14-ensemble
#   ARGUS_MODELS_DIR=/mnt/big/models ./scripts/fetch_owlv2.sh
#
# Weights land in <repo>/models/<org>/<name>, beside SigLIP's and the VLM's,
# which is gitignored and is deliberately not data/models — see
# scripts/fetch_siglip.sh for why.
#
# ---------------------------------------------------------------------------
# Which checkpoint, and why
# ---------------------------------------------------------------------------
# OWLv2 (Minderer et al., "Scaling Open-Vocabulary Object Detection", 2023) is
# Apache-2.0 and ungated, which is the whole reason it is here rather than
# Grounding-DINO: the roadmap names "mG-DINO/YOLOE", and the two obvious
# Grounding-DINO checkpoints are Apache in the paper and non-commercial in the
# weights' own repo card. `scripts/license_audit.py --check` is a rule at every
# commit, and a bootstrap tool whose weights cannot be redistributed is a tool
# that cannot be part of a commercial appliance's training loop.
#
# `-base-patch16-ensemble` is the default because this runs offline in a batch,
# once, over a few hundred frames: the large variant is ~4x the compute for a
# recall gain that a human correction pass absorbs anyway. Swap it by argument
# if a scene defeats the base model.
#
# NOTE ON WHAT THIS PRODUCES. OWLv2 answers "where is a <phrase>" — it is not a
# hairnet detector and this script does not create one. It proposes head
# regions and a first guess at covered/uncovered, and every one of them is a
# candidate until somebody looks. docs/HAIRNET_BOOTSTRAP.md is the labelling
# guide that says so at more length.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-google/owlv2-base-patch16-ensemble}"
MODELS_DIR="${ARGUS_MODELS_DIR:-$ROOT/models}"
TARGET="$MODELS_DIR/$MODEL"

echo "model:  $MODEL"
echo "target: $TARGET"

if [[ -f "$TARGET/config.json" ]] && compgen -G "$TARGET/*.safetensors" >/dev/null; then
    echo "already present — nothing to do (delete the directory to force a refetch)"
    exit 0
fi

mkdir -p "$TARGET"

export HF_HUB_ENABLE_HF_TRANSFER=1

uv run --no-project --with "huggingface_hub[hf_transfer]>=0.36" python - "$MODEL" "$TARGET" <<'PY'
import sys

from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError

model, target = sys.argv[1], sys.argv[2]

try:
    path = snapshot_download(
        repo_id=model,
        local_dir=target,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
        max_workers=8,
    )
except GatedRepoError:
    print(
        f"\n{model} is a gated repository. OWLv2's own repos are not — check the\n"
        "id for a typo before accepting a licence for something else.\n",
        file=sys.stderr,
    )
    raise SystemExit(1)
print(f"\ndownloaded to {path}")
PY

echo
du -sh "$TARGET"
echo "done. Use it with: uv run python scripts/bootstrap_hairnet.py --help"
