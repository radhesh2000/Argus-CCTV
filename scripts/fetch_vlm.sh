#!/usr/bin/env bash
# Fetch the VLM weights the policy engine serves, once, at install time.
#
# Argus is fully local at runtime: nothing here runs while the appliance is
# serving. This script is the *only* moment weights cross the network, and it is
# idempotent — re-running it resumes rather than re-downloads.
#
#   ./scripts/fetch_vlm.sh                    # the configured default model
#   ./scripts/fetch_vlm.sh Qwen/Qwen3-VL-8B-Instruct
#   ./scripts/fetch_vlm.sh Qwen/Qwen3.8-27B-FP8   # the 27B candidate, trialled
#   ARGUS_MODELS_DIR=/mnt/big/models ./scripts/fetch_vlm.sh
#
# Weights land in <repo>/models/<org>/<name>, which is gitignored. That is
# deliberately *not* data/models: data/ is what retention sweeps and what a
# support bundle tars up, and a 17 GB checkpoint belongs in neither.
#
# ---------------------------------------------------------------------------
# Which model, and why this one
# ---------------------------------------------------------------------------
# The rebuild plan names NVIDIA Cosmos-Reason2-8B (docs/REBUILD_PLAN.md, "Model
# generation"). Its HuggingFace repo is **gated** (`gated: auto` — accept the
# NVIDIA Open Model License on the model page with the account that owns
# HF_TOKEN), so the default here is its Apache-2.0 base model instead:
#
#     Qwen/Qwen3-VL-8B-Instruct
#
# Cosmos-Reason2-8B is a fine-tune *of that exact model* — same `qwen3_vl`
# architecture, same chat template shape, same vLLM code path. Swapping to it
# once the licence is accepted is a model id change and nothing else:
#
#     ./scripts/fetch_vlm.sh nvidia/Cosmos-Reason2-8B
#     # then set vlm.model in config/argus.yaml to the same id
#
# There is no FP8 Cosmos-Reason2 checkpoint published (only 2B / 8B / 32B in
# bf16), so the "FP8 if available" branch of the plan is currently not
# available. bf16 8B is ~17.6 GB, which is nothing against 128 GB unified.
#
# ---------------------------------------------------------------------------
# Qwen3.8-27B: FP8, and why not the other four quantisations
# ---------------------------------------------------------------------------
# Trialled 2026-08-15 (docs/VLM_EVAL.md, docs/BENCHMARKS.md "Phase 4"). Fetch it
# with `./scripts/fetch_vlm.sh Qwen/Qwen3.8-27B-FP8`; it is NOT the default and
# the measurements say why.
#
# 27.78B dense parameters is 55.6 GB in bf16. This box moves ~273 GB/s, and
# token decode is bandwidth-bound, so bf16 would decode at ~4-5 tok/s before
# anything else went wrong. The quantisation is not an optimisation here, it is
# the difference between a servable model and a demo.
#
#   Qwen/Qwen3.8-27B-FP8   **chosen**  28.75 GiB. Published by Qwen, Apache-2.0,
#                          ungated, block-scaled e4m3 (weight_block_size
#                          [128,128], dynamic activation scale). vLLM selects
#                          `CutlassFp8BlockScaledMMKernel` on this GB10 — native
#                          FP8 tensor-core matmul on sm_121, not an emulation.
#                          Measured 8.3 tok/s decode against 14.3 for the 8B.
#   unsloth/…-NVFP4,       W4 would be ~15 GB and roughly twice the decode rate,
#   RadixArk/…-NVFP4,      and every published NVFP4/AWQ/int4 checkpoint for this
#   philbert440/…-W4A16    model is third-party. Two reasons that was not
#                          followed: the DGX Spark playbook records no native FP4
#                          compute on GB10 (so the win would be bandwidth only,
#                          through a dequant path), and a third-party checkpoint
#                          means an unaudited licence and unpublished calibration
#                          data in a product whose licence audit is enforced in
#                          CI. If the 27B ever ships, W4 is the first thing to
#                          re-measure — and the first thing to re-license.
#   *-GGUF, *-mlx, *-exl3  Other runtimes. vLLM does not serve them here.
#
# The MTP module (`mtp.*`, one layer) comes with the checkpoint and vLLM
# registers `Qwen3_5MTP` for speculative decoding. It was NOT enabled for the
# trial: the harness measures the deployed serving path, and turning on
# speculation would have measured a configuration `argus vlm start` does not
# produce. It is the obvious next experiment on the decode rate.
#
# ---------------------------------------------------------------------------
# Qwen/Qwen3-VL-8B-Instruct-FP8 — fetchable, and it needs one env var
# ---------------------------------------------------------------------------
# The official FP8 of the *shipping* model (9.9 GB against bf16's 17 GB). It
# decodes at 25.8 tok/s against 14.3 and answers headcount claims measurably
# worse — the full numbers are in docs/PHASE2_NOTES.md §2, and it is not the
# default for that reason, not for a packaging one.
#
# It also does not load here on defaults. vLLM picks the DeepGEMM scaled-mm
# kernel for its block-scaled weights and dies in process_weights_after_loading
# with `Unknown SF transformation` (deepgemm layout.hpp:60). Start the server
# with VLLM_USE_DEEP_GEMM=0 and it takes the same CUTLASS path the 27B chose on
# its own. That env var is not in config/argus.yaml on purpose: a default that
# needs one is not the one-line model swap `vlm.model` is meant to be.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-${ARGUS_VLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct}}"
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
# downloading weights is an install-time concern, and the runtime dependency
# tree stays clean. hf_transfer is a Rust-backed parallel downloader (Apache-2.0)
# and is worth ~3x on a 17 GB checkpoint.
export HF_HUB_ENABLE_HF_TRANSFER=1

uv run --no-project --with "huggingface_hub[hf_transfer]>=0.36" python - "$MODEL" "$TARGET" <<'PY'
import sys

from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError

model, target = sys.argv[1], sys.argv[2]

# Weights, config and tokenizer only. The repos carry README images, benchmark
# PNGs and (for Cosmos) .pth originals that vLLM never opens.
#
# `*.jinja` is not decoration: transformers 5.x moved the chat template out of
# tokenizer_config.json into a standalone chat_template.jinja, and Qwen3.8-27B
# ships it that way. Without it vLLM serves the model but /v1/chat/completions
# has no template to render — the failure is at request time, not at load, so it
# is worth naming here rather than rediscovering.
try:
    path = snapshot_download(
        repo_id=model,
        local_dir=target,
        allow_patterns=[
            "*.json", "*.safetensors", "*.txt", "*.model", "*.py", "*.jinja",
        ],
        max_workers=8,
    )
except GatedRepoError:
    print(
        f"\n{model} is a gated repository.\n"
        f"Open https://huggingface.co/{model} while signed in as the owner of\n"
        "HF_TOKEN, accept the licence, then re-run this script. Or fall back to\n"
        "the Apache-2.0 base model:\n\n"
        "    ./scripts/fetch_vlm.sh Qwen/Qwen3-VL-8B-Instruct\n",
        file=sys.stderr,
    )
    raise SystemExit(1)
print(f"\ndownloaded to {path}")
PY

echo
du -sh "$TARGET"
echo "done. Serve it with: argus vlm start"
