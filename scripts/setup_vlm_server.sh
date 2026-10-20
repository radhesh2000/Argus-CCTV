#!/usr/bin/env bash
# Build the virtualenv that serves the VLM. Run once, at install time.
#
#   ./scripts/setup_vlm_server.sh
#
# This env is deliberately NOT the argus env — see argus/vlm/server.py for why.
# It holds vLLM and its own torch build; `uv run pytest` never sees it.
#
# ---------------------------------------------------------------------------
# Why these exact choices on this exact box (DGX Spark, GB10, aarch64, CUDA 13)
# ---------------------------------------------------------------------------
#
# 1. NOT the NGC container. `nvcr.io/nvidia/vllm` (and vLLM's own
#    `vllm/vllm-openai:cu130-nightly`, which the vLLM DGX Spark guide
#    recommends) is the smoothest path *if you can use docker*. On this
#    appliance the invoking user is not in the `docker` group and has no
#    passwordless sudo, so the socket is unreachable. Wheels it is. If a
#    deployment does have docker, the container is the better answer and this
#    script is not needed — the endpoint is the only contract.
#
# 2. NOT stable PyPI vLLM. Stable publishes aarch64 wheels, but built for
#    CUDA 12.x — they link `libcudart.so.12`, which does not exist on a CUDA-13
#    Spark, so they fail at import before any kernel is reached. The
#    `wheels.vllm.ai/nightly/cu130` index is the aarch64 + CUDA 13 build.
#
# 3. GB10 is sm_121 and the wheels carry sm_120 kernels. Those are binary
#    compatible within Blackwell 12.x, so this works; there is no sm_121-
#    specific build to wait for.
#
# 4. **A uv-managed CPython, not the distro one.** This is the non-obvious one
#    and it cost a debugging cycle: vLLM's Triton JIT compiles `cuda_utils.c` at
#    *runtime* and needs `Python.h`. Ubuntu's `python3.12` ships no headers
#    without `python3.12-dev`, which needs root. A uv-managed interpreter
#    carries its own headers, so the whole install needs no privileges at all —
#    which an appliance installer should not require anyway.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${ARGUS_VLM_VENV:-$ROOT/.venv-vlm}"
PYTHON_VERSION="${ARGUS_VLM_PYTHON:-3.12.13}"
INDEX="${ARGUS_VLM_INDEX:-https://wheels.vllm.ai/nightly/cu130}"

echo "venv:   $VENV"
echo "python: $PYTHON_VERSION (uv-managed — needed for Triton's runtime headers)"
echo "index:  $INDEX"

if ! command -v uv >/dev/null; then
    echo "uv is required (https://docs.astral.sh/uv/)" >&2
    exit 1
fi

# A managed interpreter, so Python.h exists without a system package.
uv python install "$PYTHON_VERSION"

uv venv --python "$PYTHON_VERSION" "$VENV"
VIRTUAL_ENV="$VENV" uv pip install vllm \
    --prerelease=allow \
    --torch-backend=cu130 \
    --extra-index-url "$INDEX" \
    --index-strategy unsafe-best-match

echo
"$VENV/bin/python" - <<'PY'
import torch
import vllm

print(f"vllm  {vllm.__version__}")
print(f"torch {torch.__version__} (cuda {torch.version.cuda})")
print(f"gpu   {torch.cuda.get_device_name(0)} "
      f"sm_{''.join(str(v) for v in torch.cuda.get_device_capability(0))}")
print(f"built for: {', '.join(torch.cuda.get_arch_list())}")
PY

echo
echo "done. Next: ./scripts/fetch_vlm.sh && argus vlm start"
