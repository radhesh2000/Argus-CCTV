#!/usr/bin/env bash
# Fetch the SigLIP 2 checkpoint the semantic index embeds with, once, at
# install time.
#
# Argus is fully local at runtime: nothing here runs while the appliance is
# serving, and neither the index writer nor the search API ever opens a socket
# to a model host. This script and scripts/fetch_vlm.sh are the only moments
# weights cross the network. Idempotent — re-running resumes.
#
#   ./scripts/fetch_siglip.sh                                 # the configured default
#   ./scripts/fetch_siglip.sh google/siglip2-so400m-patch14-384
#   ARGUS_MODELS_DIR=/mnt/big/models ./scripts/fetch_siglip.sh
#
# Weights land in <repo>/models/<org>/<name>, beside the VLM's, which is
# gitignored and is deliberately not data/models: data/ is what retention
# sweeps and what a support bundle tars up, and an install artefact belongs in
# neither.
#
# ---------------------------------------------------------------------------
# Which checkpoint, and why this one
# ---------------------------------------------------------------------------
# SigLIP 2 is what NVIDIA's VSS 3.2 blueprint uses for the same job
# (docs/REBUILD_PLAN.md, "index-service"). The whole family is Apache-2.0 and
# ungated — verified against the HuggingFace API before anything was
# downloaded, because a gated repo is what stopped Cosmos-Reason2 in Phase 2.
#
# Both realistic candidates were fetched and measured on this GB10 (2026-08-15):
#
#   google/siglip2-base-patch16-384      375 M params, 768-d,   3.35 ms/frame @ b8
#   google/siglip2-so400m-patch14-384  1 136 M params, 1152-d, 30.0  ms/frame @ b8
#                                                             (16.3 at its own best, b4)
#
# base is the default: 4.9x cheaper per frame on a GPU it shares with RF-DETR
# and vLLM even after giving each model its best batch size, and on the 74
# frames of footage available here it retrieved at least as well
# (docs/PHASE3_NOTES.md has the probes). so400m is one argument away for a site
# with GPU headroom and a harder retrieval problem — set `index.model` in
# config/argus.yaml to match, then `argus index backfill --reset`, because
# vectors from two models are not comparable and the store refuses to mix them.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-${ARGUS_INDEX_MODEL:-google/siglip2-base-patch16-384}}"
MODELS_DIR="${ARGUS_MODELS_DIR:-$ROOT/models}"
TARGET="$MODELS_DIR/$MODEL"

echo "model:  $MODEL"
echo "target: $TARGET"

if [[ -f "$TARGET/config.json" ]] && compgen -G "$TARGET/*.safetensors" >/dev/null; then
    echo "already present — nothing to do (delete the directory to force a refetch)"
    exit 0
fi

mkdir -p "$TARGET"

# huggingface_hub is fetched on demand rather than added to the argus package:
# downloading weights is an install-time concern and the runtime dependency
# tree stays clean.
export HF_HUB_ENABLE_HF_TRANSFER=1

uv run --no-project --with "huggingface_hub[hf_transfer]>=0.36" python - "$MODEL" "$TARGET" <<'PY'
import sys

from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError

model, target = sys.argv[1], sys.argv[2]

# Weights, config and tokenizer only — the repos also carry README figures.
# `*.model` is the sentencepiece file; transformers prefers the fast
# tokenizer.json that ships alongside it, so this never needs sentencepiece
# installed, but a checkpoint without it would be unloadable elsewhere.
try:
    path = snapshot_download(
        repo_id=model,
        local_dir=target,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
        max_workers=8,
    )
except GatedRepoError:
    print(
        f"\n{model} is a gated repository. Open https://huggingface.co/{model}\n"
        "while signed in as the owner of HF_TOKEN, accept the licence, and re-run.\n"
        "Or fall back to the Apache-2.0 default:\n\n"
        "    ./scripts/fetch_siglip.sh google/siglip2-base-patch16-384\n",
        file=sys.stderr,
    )
    raise SystemExit(1)
print(f"\ndownloaded to {path}")
PY

echo
du -sh "$TARGET"
echo "done. Check it with: argus index status"
echo "Make existing evidence searchable with: argus index backfill"
