# WASM module — intended to be added via add_subdirectory() from root CMakeLists.txt
# Build with: emcmake cmake .. -DBUILD_WASM=ON

add_executable(spectra-wasm bindings.cpp)
target_link_libraries(spectra-wasm PRIVATE Spectra)
set_target_properties(spectra-wasm PROPERTIES OUTPUT_NAME spectra)

# Emscripten flags — use "SHELL:" so CMake passes "-s KEY=VALUE" as one argument
target_link_options(spectra-wasm PRIVATE
    --bind
    "SHELL:-s MODULARIZE=1"
    "SHELL:-s EXPORT_NAME=createSpectra"
    "SHELL:-s ALLOW_MEMORY_GROWTH=1"
    "SHELL:-s ENVIRONMENT=web,node"
    "SHELL:-s WASM=1"
)
