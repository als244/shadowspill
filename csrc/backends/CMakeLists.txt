# Pluggable memory-pool backends. Each is dlopened at runtime and compiled
# against the backend contract alone, which is why they stay separate shared
# objects while everything that plans and executes a step is one.
add_library(
  shadowspill_backend_mock
  SHARED
  mock/src/backend_mock.c
)
target_include_directories(
  shadowspill_backend_mock
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/mock/include>
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
  TARGETS shadowspill_backend_mock
  LIBRARY DESTINATION shadowspill/lib
  RUNTIME DESTINATION shadowspill/lib
)
install(
  FILES mock/include/shadowspill/backend_mock.h
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
      cuda/src/backend_cuda.c
      cuda/src/profiler_nvtx.c
    )
    target_include_directories(
      shadowspill_backend_cuda
      PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cuda/include>
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
      FILES cuda/include/shadowspill/backend_cuda.h
      DESTINATION shadowspill/include/shadowspill
    )
  else()
    message(STATUS "CUDA Toolkit not found; skipping CUDA backend")
  endif()
endif()
