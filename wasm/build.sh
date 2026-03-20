#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${REPO_DIR}/build-wasm"
BUILD_TYPE="${1:-Release}"

# Check for emcmake
if ! command -v emcmake &> /dev/null; then
    echo "Error: emcmake not found. Install Emscripten SDK first:"
    echo "  git clone https://github.com/emscripten-core/emsdk.git"
    echo "  cd emsdk && ./emsdk install latest && ./emsdk activate latest"
    echo "  source emsdk_env.sh"
    echo ""
    echo "Or use Docker instead: ./wasm/docker-build.sh"
    exit 1
fi

echo "=== Building Spectra WASM (${BUILD_TYPE}) ==="

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

emcmake cmake "${REPO_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DBUILD_WASM=ON \
    -DCPM_DOWNLOAD_ALL=ON

emmake make -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)" spectra-wasm

echo ""
echo "=== Build complete ==="
ls -lh "${BUILD_DIR}/wasm/spectra.js" "${BUILD_DIR}/wasm/spectra.wasm" 2>/dev/null
