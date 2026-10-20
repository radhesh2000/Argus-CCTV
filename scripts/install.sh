#!/usr/bin/env bash
# Turn a fresh DGX Spark into an Argus appliance.
#
#   ./scripts/install.sh                 # everything except the VLM and systemd
#   ./scripts/install.sh --vlm           # + the 17 GB VLM and its serving venv
#   ./scripts/install.sh --services      # + write, enable and start systemd units
#   ./scripts/install.sh --offline       # verify pre-staged artefacts, fetch nothing
#   ./scripts/install.sh --list          # print the steps and their state, do nothing
#
# ---------------------------------------------------------------------------
# What this is, and what it deliberately is not
# ---------------------------------------------------------------------------
# It is an ORCHESTRATOR. Every network fetch already lives in its own pinned,
# idempotent script — fetch_gateway.sh (sha256-pinned), build_ffmpeg.sh,
# fetch_detector.sh, fetch_siglip.sh, fetch_vlm.sh, setup_vlm_server.sh. This
# adds no new download, no new URL and no new hash; it decides the order, skips
# what is already done, and says what is left.
#
# That is what makes it resumable and air-gappable at the same time. Every step
# is "check, then do", so an install interrupted at step 6 of 9 is re-run and
# resumes at 6. And because each step's check is separate from its fetch,
# --offline runs the checks alone: pre-stage bin/, models/ and .venv by any
# means you like (rsync from a staging box, a USB disk, an image), then prove
# the appliance is complete without a single packet leaving it.
#
# It assumes NO ROOT. Nothing here needs sudo, including the systemd step,
# which installs user units. The one command in the whole install that does
# need root — `loginctl enable-linger` — is printed, never run, because a box
# that is meant to survive its operator logging out should have somebody
# deliberately decide that.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WITH_VLM=0
WITH_SERVICES=0
OFFLINE=0
LIST_ONLY=0
SKIP_UI=0
ADMIN_USER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --vlm)       WITH_VLM=1 ;;
        --services)  WITH_SERVICES=1 ;;
        --offline)   OFFLINE=1 ;;
        --list)      LIST_ONLY=1 ;;
        --no-ui)     SKIP_UI=1 ;;
        --admin)     ADMIN_USER="${2:?--admin needs a username}"; shift ;;
        -h|--help)   sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
    esac
    shift
done

# --- output -----------------------------------------------------------------

if [[ -t 1 ]]; then
    BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GREEN=$'\033[32m'
    YELLOW=$'\033[33m'; RESET=$'\033[0m'
else
    BOLD=""; DIM=""; RED=""; GREEN=""; YELLOW=""; RESET=""
fi

STEP=0
step()  { STEP=$((STEP + 1)); printf '\n%s[%d/%d] %s%s\n' "$BOLD" "$STEP" "$TOTAL" "$1" "$RESET"; }
have()  { printf '      %s✓%s %s\n' "$GREEN" "$RESET" "$1"; }
doing() { printf '      %s→%s %s\n' "$YELLOW" "$RESET" "$1"; }
warn()  { printf '      %s!%s %s\n' "$YELLOW" "$RESET" "$1"; }
fail()  { printf '      %s✗%s %s\n' "$RED" "$RESET" "$1" >&2; }

# Refuse to fetch when --offline was asked for: the whole point of the flag is
# that this box cannot, or must not, reach the network.
needs_network() {
    if [[ "$OFFLINE" == 1 ]]; then
        fail "$1"
        fail "pre-stage it and re-run, or drop --offline"
        exit 1
    fi
}

TOTAL=9
[[ "$WITH_VLM" == 1 ]]      && TOTAL=$((TOTAL + 1))
[[ "$WITH_SERVICES" == 1 ]] && TOTAL=$((TOTAL + 1))

# --- inventory (--list) -----------------------------------------------------

if [[ "$LIST_ONLY" == 1 ]]; then
    printf '%sArgus install state — %s%s\n\n' "$BOLD" "$ROOT" "$RESET"
    check() { if eval "$2"; then have "$1"; else printf '      %s·%s %s\n' "$DIM" "$RESET" "$1"; fi; }
    check "python environment (.venv)"      '[[ -x .venv/bin/argus ]]'
    check "go2rtc (bin/go2rtc)"             '[[ -x bin/go2rtc ]]'
    check "ffmpeg with NVDEC (bin/ffmpeg)"  '[[ -x bin/ffmpeg ]]'
    check "detector weights"                'compgen -G "data/models/rf-detr-*.pth" >/dev/null'
    check "SigLIP index model"              'compgen -G "models/google/*/*.safetensors" >/dev/null'
    check "VLM weights"                     'compgen -G "models/Qwen/*/*.safetensors" >/dev/null'
    check "VLM serving venv (.venv-vlm)"    '[[ -x .venv-vlm/bin/vllm ]]'
    check "built console (ui/dist)"         '[[ -f ui/dist/index.html ]]'
    check "database (data/db/argus.sqlite3)" '[[ -f data/db/argus.sqlite3 ]]'
    check "systemd units"                   '[[ -f "${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/argus-engine.service" ]]'
    exit 0
fi

printf '%sInstalling Argus into %s%s\n' "$BOLD" "$ROOT" "$RESET"
[[ "$OFFLINE" == 1 ]] && printf '%soffline: verifying pre-staged artefacts, fetching nothing%s\n' "$DIM" "$RESET"

# --- 1. python environment --------------------------------------------------

step "Python environment"
if ! command -v uv >/dev/null; then
    fail "uv is not installed — https://docs.astral.sh/uv/"
    exit 1
fi
if [[ "$OFFLINE" == 1 ]]; then
    if [[ -x .venv/bin/argus ]]; then
        have "$(.venv/bin/argus --version)"
    else
        needs_network "no .venv — uv sync needs the network"
    fi
else
    doing "uv sync --frozen"
    # --frozen: install exactly what uv.lock says. An installer that is allowed
    # to re-resolve can install a different tree from the one that was tested
    # and licence-audited, which is the whole argument for a lock file.
    uv sync --frozen
    have "$(.venv/bin/argus --version)"
fi
ARGUS=".venv/bin/argus"

# --- 2. stream gateway ------------------------------------------------------

step "Stream gateway (go2rtc)"
if [[ -x bin/go2rtc ]]; then
    have "$(bin/go2rtc --version 2>&1 | head -1)"
else
    needs_network "bin/go2rtc is missing"
    doing "scripts/fetch_gateway.sh"
    ./scripts/fetch_gateway.sh
fi

# --- 3. ffmpeg --------------------------------------------------------------

step "ffmpeg with NVDEC (LGPL)"
if [[ -x bin/ffmpeg ]] && bin/ffmpeg -hide_banner -decoders 2>/dev/null | grep -q h264_cuvid; then
    have "$(bin/ffmpeg -version | head -1)"
    if bin/ffmpeg -version | grep -q -- --enable-gpl; then
        fail "this ffmpeg is GPL and cannot ship in the appliance — rebuild it"
        exit 1
    fi
elif [[ -x bin/ffmpeg ]]; then
    warn "bin/ffmpeg has no NVDEC — every camera will decode on the CPU"
    warn "rebuild with ./scripts/build_ffmpeg.sh"
else
    needs_network "bin/ffmpeg is missing (built from source, needs git + a toolchain)"
    doing "scripts/build_ffmpeg.sh — this takes a while"
    ./scripts/build_ffmpeg.sh
fi

# --- 4. detector weights ----------------------------------------------------

step "Detector weights (RF-DETR)"
if compgen -G "data/models/rf-detr-*.pth" >/dev/null; then
    have "$(ls -1 data/models/rf-detr-*.pth | head -1)"
else
    needs_network "no detector checkpoint in data/models — the engine downloads it on first use, which is exactly what an air-gapped site cannot do"
    doing "scripts/fetch_detector.sh"
    ./scripts/fetch_detector.sh
fi

# --- 5. semantic index model ------------------------------------------------

step "Semantic index model (SigLIP 2)"
if compgen -G "models/google/*/*.safetensors" >/dev/null; then
    have "$(dirname "$(ls -1 models/google/*/*.safetensors | head -1)")"
else
    needs_network "no SigLIP checkpoint in models/ — semantic search will not start"
    doing "scripts/fetch_siglip.sh"
    ./scripts/fetch_siglip.sh
fi

# --- 6. VLM (optional) ------------------------------------------------------

if [[ "$WITH_VLM" == 1 ]]; then
    step "VLM policy engine (optional, ~17 GB + a serving venv)"
    if [[ -x .venv-vlm/bin/vllm ]]; then
        have ".venv-vlm is built"
    else
        needs_network ".venv-vlm is missing"
        doing "scripts/setup_vlm_server.sh"
        ./scripts/setup_vlm_server.sh
    fi
    if compgen -G "models/Qwen/*/*.safetensors" >/dev/null; then
        have "$(dirname "$(ls -1 models/Qwen/*/*.safetensors | head -1)")"
    else
        needs_network "no VLM weights in models/"
        doing "scripts/fetch_vlm.sh"
        ./scripts/fetch_vlm.sh
    fi
fi

# --- 7. console -------------------------------------------------------------

step "Operator console"
if [[ "$SKIP_UI" == 1 ]]; then
    warn "skipped (--no-ui); argus api will serve the JSON API alone"
elif [[ -f ui/dist/index.html ]] && [[ "$OFFLINE" == 1 ]]; then
    have "prebuilt ui/dist ($(find ui/dist -type f | wc -l) files)"
elif ! command -v npm >/dev/null; then
    if [[ -f ui/dist/index.html ]]; then
        have "prebuilt ui/dist, and no npm on this box — using it as staged"
    else
        warn "no npm and no ui/dist — the console will not be served"
        warn "build it elsewhere and copy ui/dist here, or install node"
    fi
else
    doing "npm ci && npm run build"
    (cd ui && npm ci --no-audit --no-fund && npm run build)
    have "ui/dist ($(find ui/dist -type f | wc -l) files)"
fi

# --- 8. database ------------------------------------------------------------

step "Database and fleet"
# `seed` is idempotent — it upserts by id — so re-running an install does not
# duplicate a camera or reset a policy's lifecycle.
doing "argus seed"
"$ARGUS" seed
have "schema v$("$ARGUS" status | grep -oP '(?<=schema v)\d+' | head -1)"

# --- 9. first admin ---------------------------------------------------------

step "First account"
if "$ARGUS" user list >/dev/null 2>&1; then
    have "accounts already exist — \`argus user list\` shows the roster"
elif [[ -n "$ADMIN_USER" ]]; then
    # Piped, not --password: a flag would put the plant's administrator
    # password into shell history and into the process table.
    if [[ -n "${ARGUS_ADMIN_PASSWORD:-}" ]]; then
        doing "creating $ADMIN_USER from ARGUS_ADMIN_PASSWORD"
        printf '%s\n' "$ARGUS_ADMIN_PASSWORD" | "$ARGUS" user add "$ADMIN_USER" --role ADMIN
    else
        doing "creating $ADMIN_USER"
        "$ARGUS" user add "$ADMIN_USER" --role ADMIN
    fi
elif [[ -t 0 ]]; then
    printf '      A fresh appliance has no accounts and refuses every request.\n'
    read -r -p "      username for the first ADMIN (blank to skip): " name
    if [[ -n "$name" ]]; then
        "$ARGUS" user add "$name" --role ADMIN
    else
        warn "skipped — run: argus user add <name> --role ADMIN"
    fi
else
    warn "no TTY and no --admin: run \`argus user add <name> --role ADMIN\` on the box"
fi

# --- 10. first backup -------------------------------------------------------

step "First backup"
if "$ARGUS" backup now --reason install >/dev/null 2>&1; then
    have "$("$ARGUS" backup list | sed -n '2p')"
else
    warn "backup failed — run \`argus backup now\` and read the error"
fi

# --- 11. services (optional) ------------------------------------------------

if [[ "$WITH_SERVICES" == 1 ]]; then
    step "systemd user units"
    VLM_FLAG=()
    [[ "$WITH_VLM" == 1 ]] && VLM_FLAG=(--vlm)
    "$ARGUS" install-services --start "${VLM_FLAG[@]}"
fi

# --- summary ----------------------------------------------------------------

printf '\n%s%s%s\n' "$BOLD" "Installed. What runs where:" "$RESET"
# Asked of the config loader, not grepped out of the YAML: `port:` appears
# under `smtp:` at the same indentation and appears first, so a grep for it
# printed 587 and sent an operator to a mail port.
API_PORT="$(.venv/bin/python -c \
    'from argus.config import load_config; print(load_config().api.port)' 2>/dev/null \
    || echo 8080)"
HOST_ADDR="$(hostname -I 2>/dev/null | awk '{print $1}')"
HOST_ADDR="${HOST_ADDR:-<appliance>}"
cat <<EOF

  vision engine   argus run        decode, detect, track, evidence, housekeeping
  console + API   argus api        http://${HOST_ADDR}:${API_PORT}/  (sign in first)
  VLM (optional)  argus vlm start  local model for plain-English policies
  gateway         (started by the engine)  RTSP :8554 on the LAN,
                                           HTTP API :1984 on loopback only

  data            $ROOT/data        evidence, database, backups, logs
  backups         nightly, keeping 7, inside the housekeeping thread
  logs            journald when run as services; data/logs for go2rtc and vLLM

EOF

if [[ "$WITH_SERVICES" != 1 ]]; then
    cat <<EOF
  Not installed as services. To have this box come up on its own:

      argus install-services --start
      sudo loginctl enable-linger $USER    # the one command that needs root

EOF
fi

printf '%sChecking the result%s\n\n' "$BOLD" "$RESET"
"$ARGUS" doctor || {
    printf '\n%sInstall finished with failures above.%s\n' "$RED" "$RESET"
    exit 1
}
