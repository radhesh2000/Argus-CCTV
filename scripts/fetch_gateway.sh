#!/usr/bin/env bash
# Fetch the pinned go2rtc stream-gateway binary into bin/.
#
# go2rtc (https://github.com/AlexxIT/go2rtc) is MIT licensed and is vendored as a
# prebuilt binary rather than a dependency: it is a standalone Go process that
# Argus supervises, so it never links into the Python package.
#
# bin/ is gitignored (binaries don't belong in the repo); this script plus the
# pinned version + sha256 below make the install reproducible and tamper-evident.
#
# Usage: scripts/fetch_gateway.sh [--force]
set -euo pipefail

GO2RTC_VERSION="v1.9.14"

# sha256 of each published asset for this version (recorded at vendoring time).
declare -A GO2RTC_SHA256=(
  ["linux_arm64"]="359fabade8a7a51e81a55fe6df6b0ef81764a5e1d63179577534eaaa71904b50"
  ["linux_amd64"]="32d616af226bd731678ffde328b94cfb94e30339bfefc469cfb76323144615a6"
)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="${ROOT}/bin"
DEST="${BIN_DIR}/go2rtc"

case "$(uname -s)/$(uname -m)" in
  Linux/aarch64|Linux/arm64) ASSET_KEY="linux_arm64" ;;
  Linux/x86_64)              ASSET_KEY="linux_amd64" ;;
  *) echo "unsupported platform: $(uname -s)/$(uname -m) (Argus targets linux/arm64)" >&2; exit 1 ;;
esac

ASSET="go2rtc_${ASSET_KEY}"
URL="https://github.com/AlexxIT/go2rtc/releases/download/${GO2RTC_VERSION}/${ASSET}"
EXPECTED="${GO2RTC_SHA256[$ASSET_KEY]}"

mkdir -p "${BIN_DIR}"

if [[ -x "${DEST}" && "${1:-}" != "--force" ]]; then
  actual="$(sha256sum "${DEST}" | cut -d' ' -f1)"
  if [[ "${actual}" == "${EXPECTED}" ]]; then
    echo "go2rtc ${GO2RTC_VERSION} already present and verified: ${DEST}"
    exit 0
  fi
  echo "existing ${DEST} has unexpected sha256 (${actual}); re-downloading" >&2
fi

tmp="$(mktemp "${BIN_DIR}/.go2rtc.XXXXXX")"
trap 'rm -f "${tmp}"' EXIT

echo "downloading ${URL}"
curl -fsSL --retry 3 --connect-timeout 20 -o "${tmp}" "${URL}"

actual="$(sha256sum "${tmp}" | cut -d' ' -f1)"
if [[ "${actual}" != "${EXPECTED}" ]]; then
  echo "sha256 mismatch for ${ASSET}" >&2
  echo "  expected ${EXPECTED}" >&2
  echo "  actual   ${actual}" >&2
  exit 1
fi

chmod +x "${tmp}"
mv "${tmp}" "${DEST}"
trap - EXIT

echo "installed go2rtc ${GO2RTC_VERSION} -> ${DEST}"
"${DEST}" --version 2>/dev/null || true
