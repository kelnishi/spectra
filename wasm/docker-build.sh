#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_TYPE="${1:-Release}"

echo "=== Building Spectra WASM via Docker (${BUILD_TYPE}) ==="

docker run --rm \
    -v "${REPO_DIR}":/src \
    -w /src \
    emscripten/emsdk:3.1.51 \
    bash -c "
        mkdir -p build-wasm && cd build-wasm &&
        emcmake cmake .. \
            -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
            -DBUILD_WASM=ON \
            -DCPM_DOWNLOAD_ALL=ON &&
        emmake make -j\$(nproc) spectra-wasm
    "

echo ""
echo "=== Build complete ==="
ls -lh "${REPO_DIR}/build-wasm/wasm/spectra.js" "${REPO_DIR}/build-wasm/wasm/spectra.wasm" 2>/dev/null
