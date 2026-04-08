# WASM module — intended to be added via add_subdirectory() from root CMakeLists.txt
# Build with: emcmake cmake .. -DBUILD_WASM=ON

add_executable(spectra-wasm bindings.cpp)
target_link_libraries(spectra-wasm PRIVATE Spectra)
set_target_properties(spectra-wasm PROPERTIES
    OUTPUT_NAME spectra
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# SIMD: enable WebAssembly SIMD (128-bit, maps to SSE2 intrinsics in Eigen)
# -fwasm-exceptions: native WASM exception handling (lighter than JS-based -fexceptions)
target_compile_options(spectra-wasm PRIVATE -msimd128 -fwasm-exceptions)
target_link_options(spectra-wasm PRIVATE -fwasm-exceptions)

# Emscripten flags — use "SHELL:" so CMake passes "-s KEY=VALUE" as one argument
target_link_options(spectra-wasm PRIVATE
    -lembind
    "SHELL:-s MODULARIZE=1"
    "SHELL:-s EXPORT_NAME=createSpectra"
    "SHELL:-s ALLOW_MEMORY_GROWTH=1"
    "SHELL:-s ENVIRONMENT=web,node"
    "SHELL:-s WASM=1"
    "SHELL:-s WASM_BIGINT=1"
    "SHELL:-s STACK_SIZE=1048576"
    # Export malloc/free for zero-copy TypedArray → WASM heap transfers
    "SHELL:-s EXPORTED_FUNCTIONS=['_malloc','_free']"
    # Export heap views for CSR zero-copy allocators in spectra.mjs
    "SHELL:-s EXPORTED_RUNTIME_METHODS=['HEAP32','HEAPF64']"
    # Export exception decoding so JS can read C++ exception messages
    "SHELL:-s EXPORT_EXCEPTION_HANDLING_HELPERS=1"
)
