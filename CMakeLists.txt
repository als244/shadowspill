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
# The tree compiles as strict ISO C11, which hides POSIX threads unless asked
# for. Setting it here holds for every translation unit, so no header has to be
# included before any system header for pthreads to exist.
target_compile_definitions(
  shadowspill_build_options INTERFACE _POSIX_C_SOURCE=200809L
)
target_compile_options(
  shadowspill_build_options
  INTERFACE
    $<$<COMPILE_LANG_AND_ID:C,GNU,Clang>:-Wall;-Wextra;-Werror;-Wpedantic>
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall;-Wextra;-Werror;-Wpedantic>
)

# One library for planning and execution, plus the pluggable backends and
# the PyTorch adapter, which stay separate for the reasons csrc/CMakeLists.txt
# gives.
add_subdirectory(csrc)

if(BUILD_TESTING)
  add_subdirectory(tests/csrc)
  add_subdirectory(tests/integration)
endif()
