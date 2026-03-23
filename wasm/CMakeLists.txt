# WASM module — intended to be added via add_subdirectory() from root CMakeLists.txt
# Build with: emcmake cmake .. -DBUILD_WASM=ON

add_executable(spectra-wasm bindings.cpp)
target_link_libraries(spectra-wasm PRIVATE Spectra)
set_target_properties(spectra-wasm PROPERTIES OUTPUT_NAME spectra)

# SIMD: enable WebAssembly SIMD (128-bit, maps to SSE2 intrinsics in Eigen)
target_compile_options(spectra-wasm PRIVATE -msimd128)

# Emscripten flags — use "SHELL:" so CMake passes "-s KEY=VALUE" as one argument
target_link_options(spectra-wasm PRIVATE
    --bind
    "SHELL:-s MODULARIZE=1"
    "SHELL:-s EXPORT_NAME=createSpectra"
    "SHELL:-s ALLOW_MEMORY_GROWTH=1"
    "SHELL:-s ENVIRONMENT=web,node"
    "SHELL:-s WASM=1"
    # Export malloc/free for zero-copy TypedArray → WASM heap transfers
    "SHELL:-s EXPORTED_FUNCTIONS=['_malloc','_free']"
    "SHELL:-s EXPORTED_RUNTIME_METHODS=['HEAP32','HEAPF64']"
)
