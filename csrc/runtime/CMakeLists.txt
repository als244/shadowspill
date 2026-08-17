add_library(
  shadowspill_runtime
  SHARED
  src/admission_replay.c
  src/allocations.c
  src/completion_tracker.c
  src/event_pool.c
  src/execution_table.c
  src/failure_state.c
  src/fixed_layout.c
  src/idle_wakeup.c
  src/lifecycle.c
  src/memory_pool.c
  src/object_actions.c
  src/object_table.c
  src/objects.c
  src/profiler.c
  src/ranges.c
  src/retirement_queue.c
  src/runtime.c
  src/status.c
  src/task_boundaries.c
  src/telemetry.c
  src/trace.c
  src/transfer_lane.c
  src/transfer_profiles.c
  src/worker.c
)
target_include_directories(
  shadowspill_runtime
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:shadowspill/include>
  PRIVATE src
)
target_link_libraries(
  shadowspill_runtime
  PRIVATE Threads::Threads shadowspill_build_options
)
set_target_properties(
  shadowspill_runtime
  PROPERTIES
    OUTPUT_NAME shadowspill_runtime
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
)

add_library(
  shadowspill_backend_mock
  SHARED
  backends/mock/src/backend_mock.c
)
target_include_directories(
  shadowspill_backend_mock
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/backends/mock/include>
    $<INSTALL_INTERFACE:shadowspill/include>
)
target_link_libraries(
  shadowspill_backend_mock
  PRIVATE Threads::Threads shadowspill_build_options
)
set_target_properties(
  shadowspill_backend_mock
  PROPERTIES
    OUTPUT_NAME shadowspill_backend_mock
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
)

install(
  TARGETS shadowspill_runtime shadowspill_backend_mock
  LIBRARY DESTINATION shadowspill/lib
  RUNTIME DESTINATION shadowspill/lib
)
install(
  FILES
    include/shadowspill/backend.h
    include/shadowspill/admission_replay.h
    include/shadowspill/profiler.h
    include/shadowspill/runtime.h
    backends/mock/include/shadowspill/backend_mock.h
  DESTINATION shadowspill/include/shadowspill
)

option(SHADOWSPILL_BUILD_CUDA "Build the CUDA slab backend" ON)
if(SHADOWSPILL_BUILD_CUDA)
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    find_library(
      SHADOWSPILL_NVML_LIBRARY
      NAMES nvidia-ml libnvidia-ml.so.1
      REQUIRED
    )
    add_library(
      shadowspill_backend_cuda
      SHARED
      backends/cuda/src/backend_cuda.c
      backends/cuda/src/profiler_nvtx.c
    )
    target_include_directories(
      shadowspill_backend_cuda
      PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/backends/cuda/include>
        $<INSTALL_INTERFACE:shadowspill/include>
    )
    target_link_libraries(
      shadowspill_backend_cuda
      PRIVATE
        CUDA::cuda_driver
        CUDA::nvtx3
        ${SHADOWSPILL_NVML_LIBRARY}
        Threads::Threads
        shadowspill_build_options
    )
    set_target_properties(
      shadowspill_backend_cuda
      PROPERTIES
        OUTPUT_NAME shadowspill_backend_cuda
        VERSION ${PROJECT_VERSION}
        SOVERSION 0
    )
    install(
      TARGETS shadowspill_backend_cuda
      LIBRARY DESTINATION shadowspill/lib
      RUNTIME DESTINATION shadowspill/lib
    )
    install(
      FILES backends/cuda/include/shadowspill/backend_cuda.h
      DESTINATION shadowspill/include/shadowspill
    )
  else()
    message(STATUS "CUDA Toolkit not found; skipping CUDA backend")
  endif()
endif()
