

add_library(
  shadowspill_pytorch
  SHARED
  adapter.c
  allocator/allocator.c
  allocator/exception.cpp
  failure/latch.c
  failure/message.c
  lifecycle/backend.c
  lifecycle/bootstrap.c
  lifecycle/close.c
  lifecycle/library.c
  lifecycle/physical.c
  profiling/spill.c
  tasks/boundaries.c
  tasks/scopes.c
  storage/objects.c
  tasks/task_range.c
)
target_include_directories(
  shadowspill_pytorch
  PRIVATE
    include
    ${PROJECT_SOURCE_DIR}/csrc/include
)
# on_exit(), which hands the exit handler the process's exit status, is a
# GNU extension. Asking for it is a property of the compilation, not of a
# file, so no source has to define _GNU_SOURCE ahead of its first include.
target_compile_definitions(
  shadowspill_pytorch
  PRIVATE SHADOWSPILL_PYTORCH_BUILDING $<$<PLATFORM_ID:Linux>:_GNU_SOURCE>
)
target_link_libraries(
  shadowspill_pytorch
  PRIVATE
    shadowspill
    ${CMAKE_DL_LIBS}
    Threads::Threads
    shadowspill_build_options
)
if(UNIX AND NOT APPLE)
  target_link_options(
    shadowspill_pytorch
    PRIVATE "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/shadowspill_pytorch.map"
  )
endif()

# A CUDA build of PyTorch enables CMake's CUDA language as its package loads,
# and CMake looks for that compiler only through CUDACXX, on PATH, and under
# CUDA_PATH. A toolkit in its default place with none of those set is found by
# FindCUDAToolkit, which also searches /usr/local/cuda, yet not by the
# language, and the configure stops. When the language's own search comes up
# empty, it is handed the compiler FindCUDAToolkit found; wherever that search
# succeeds, nothing changes.
if(NOT CMAKE_CUDA_COMPILER AND "$ENV{CUDACXX}" STREQUAL "")
  find_program(shadowspill_language_nvcc nvcc PATHS "$ENV{CUDA_PATH}/bin" NO_CACHE)
  if(NOT shadowspill_language_nvcc)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_NVCC_EXECUTABLE)
      message(STATUS "CUDA compiler not on PATH; using ${CUDAToolkit_NVCC_EXECUTABLE}")
      set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}")
    endif()
  endif()
endif()
find_package(Torch QUIET)
if(Torch_FOUND)
  # CUDA toolkits do not own NCCL. Prefer the headers shipped beside the torch
  # wheel over a system toolkit's unconfigured nccl.h template.
  find_path(SHADOWSPILL_NCCL_INCLUDE_DIR nccl.h
    HINTS "${TORCH_INSTALL_PREFIX}/../nvidia/nccl/include" "$ENV{NCCL_ROOT}/include"
    REQUIRED)
  file(GLOB SHADOWSPILL_TORCH_CUDA_HEADERS "${TORCH_INSTALL_PREFIX}/../nvidia/*/include")
  target_include_directories(shadowspill_pytorch SYSTEM BEFORE PRIVATE
    ${SHADOWSPILL_NCCL_INCLUDE_DIR} ${SHADOWSPILL_TORCH_CUDA_HEADERS})
  target_sources(shadowspill_pytorch PRIVATE storage/cpu.cpp storage/device.cpp storage/sharded.cpp telemetry/collectives.cpp profiling/collectives.cpp profiling/allocations.cpp)
  target_link_libraries(shadowspill_pytorch PRIVATE ${TORCH_LIBRARIES})
  target_compile_definitions(
    shadowspill_pytorch
    PRIVATE SHADOWSPILL_PYTORCH_STORAGE_ADAPTER=1 USE_C10D_NCCL
  )
  if(TORCH_CXX_FLAGS)
    separate_arguments(
      SHADOWSPILL_TORCH_CXX_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}"
    )
    target_compile_options(
      shadowspill_pytorch PRIVATE ${SHADOWSPILL_TORCH_CXX_FLAGS}
    )
  endif()
else()
  message(
    STATUS
    "PyTorch CMake package not found; building allocator connector without storage rebinding"
  )
endif()

set_target_properties(
  shadowspill_pytorch
  PROPERTIES
    OUTPUT_NAME shadowspill_pytorch
    INSTALL_RPATH "$ORIGIN"
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)
install(
  TARGETS shadowspill_pytorch
  LIBRARY DESTINATION shadowspill/lib
  RUNTIME DESTINATION shadowspill/lib
)
