cmake_minimum_required(VERSION 3.25)
project(ShadowSpill VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# Keep editable-build artifacts at one stable build-root location.  Installed
# wheels still place the same targets under shadowspill/lib via install().
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

include(CTest)
find_package(Threads REQUIRED)

add_library(shadowspill_build_options INTERFACE)
target_compile_options(
  shadowspill_build_options
  INTERFACE
    $<$<COMPILE_LANG_AND_ID:C,GNU,Clang>:-Wall;-Wextra;-Werror;-Wpedantic>
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall;-Wextra;-Werror;-Wpedantic>
)

# The dependency order is simulator/runtime -> planner -> PyTorch adapter.
add_subdirectory(csrc/simulator)
add_subdirectory(csrc/runtime)
add_subdirectory(csrc/planner)
add_subdirectory(csrc/pytorch_adapter)

if(BUILD_TESTING)
  add_subdirectory(tests/csrc)
  add_subdirectory(tests/integration)
endif()
