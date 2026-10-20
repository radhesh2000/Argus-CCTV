#!/usr/bin/env bash
# Build an LGPL ffmpeg with NVDEC/NVENC into bin/, the way go2rtc is vendored.
#
# Why this exists
# ---------------
# The ffmpeg most easily installed on this box (johnvansickle static, on PATH)
# is built --enable-gpl --enable-version3 and *without* CUDA: `ffmpeg -hwaccels`
# lists only vdpau. That costs the appliance twice.
#
#   1. Every camera is decoded in software. The GB10's single NVDEC — which the
#      rebuild plan names as the real capacity ceiling — is never touched, so the
#      stream-capacity number is a CPU number wearing a GPU's clothes.
#   2. A GPL binary cannot ship inside a closed commercial appliance.
#
# This builds FFmpeg from source with no --enable-gpl and no --enable-nonfree, so
# the result is LGPL-2.1+ and redistributable, plus:
#
#   * nvdec/cuvid — hardware H.264/H.265 decode on the GB10's NVDEC engine,
#   * nvenc       — hardware H.264 encode, which is how the ring buffer can
#                   re-encode a non-remuxable source without burning Grace cores
#                   (libx264 is GPL and deliberately absent).
#
# nv-codec-headers is MIT. Neither it nor the CUDA toolkit is needed at runtime:
# ffmpeg dlopens libcuda.so.1 / libnvcuvid.so.1 / libnvidia-encode.so.1 from the
# driver.
#
# argus/ingest/decoder.py prefers bin/ffmpeg over PATH, so the whole engine
# (decoders, ring recorder, go2rtc's exec: sources) picks this up automatically
# once it exists. Nothing else has to change.
#
#     ./scripts/build_ffmpeg.sh          # build into bin/
#     ./scripts/build_ffmpeg.sh --check  # just report what bin/ffmpeg can do
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="$ROOT/bin"
BUILD_DIR="${ARGUS_FFMPEG_BUILD_DIR:-$ROOT/.build/ffmpeg}"
PREFIX="$BUILD_DIR/prefix"

# Pinned. ffnvcodec 13.x matches the 580.x driver and FFmpeg 8.x.
NV_CODEC_TAG="n13.0.19.1"
FFMPEG_TAG="n8.1.2"

report() {
    local bin="$BIN_DIR/ffmpeg"
    if [[ ! -x "$bin" ]]; then
        echo "bin/ffmpeg: not built"
        return 1
    fi
    echo "bin/ffmpeg: $("$bin" -version | head -1)"
    echo "  hwaccels: $("$bin" -hide_banner -hwaccels | tail -n +2 | tr '\n' ' ')"
    echo "  h264 decoders: $("$bin" -hide_banner -decoders 2>/dev/null | grep -c 'h264_cuvid' || true) cuvid"
    echo "  license: $("$bin" -version | grep -o 'configuration:.*' | grep -q -- '--enable-gpl' && echo GPL || echo 'LGPL (no --enable-gpl)')"
}

if [[ "${1:-}" == "--check" ]]; then
    report
    exit $?
fi

for tool in gcc make pkg-config git; do
    command -v "$tool" >/dev/null || { echo "missing build tool: $tool" >&2; exit 1; }
done

mkdir -p "$BUILD_DIR" "$BIN_DIR"
cd "$BUILD_DIR"

# --- nv-codec-headers (MIT) -------------------------------------------------
if [[ ! -d nv-codec-headers ]]; then
    git clone --depth 1 --branch "$NV_CODEC_TAG" \
        https://github.com/FFmpeg/nv-codec-headers.git
fi
make -C nv-codec-headers install "PREFIX=$PREFIX" >/dev/null

# --- ffmpeg (LGPL) ----------------------------------------------------------
if [[ ! -d FFmpeg ]]; then
    git clone --depth 1 --branch "$FFMPEG_TAG" https://github.com/FFmpeg/FFmpeg.git
fi
cd FFmpeg

export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

configure_args=(
    "--prefix=$PREFIX"
    --disable-doc
    --disable-ffplay
    --disable-debug
    --disable-autodetect        # no surprise GPL/nonfree library gets linked in
    --enable-ffnvcodec
    --enable-nvdec
    --enable-cuvid
    --enable-nvenc
    --enable-cuda
    "--extra-cflags=-I$PREFIX/include"
    "--extra-ldflags=-L$PREFIX/lib"
)

# scale_cuda/hwupload_cuda keep a resize on the GPU. Optional: it needs nvcc, and
# a toolchain mismatch must not cost us hardware decode.
if command -v nvcc >/dev/null; then
    if ./configure "${configure_args[@]}" --enable-cuda-nvcc --nvcc="$(command -v nvcc)" \
        >"$BUILD_DIR/configure.log" 2>&1; then
        echo "configured with cuda-nvcc (GPU scaling filters available)"
    else
        echo "cuda-nvcc configure failed, retrying without it (see $BUILD_DIR/configure.log)"
        ./configure "${configure_args[@]}" >"$BUILD_DIR/configure.log" 2>&1
    fi
else
    ./configure "${configure_args[@]}" >"$BUILD_DIR/configure.log" 2>&1
fi

make -j"$(nproc)" >"$BUILD_DIR/build.log" 2>&1
make install >>"$BUILD_DIR/build.log" 2>&1

install -m 0755 "$PREFIX/bin/ffmpeg" "$BIN_DIR/ffmpeg"
install -m 0755 "$PREFIX/bin/ffprobe" "$BIN_DIR/ffprobe"

report
