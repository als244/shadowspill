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
target_compile_definitions(
  shadowspill_backend_mock PRIVATE SHADOWSPILL_BACKEND_MOCK_BUILDING
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

# Which provider backends to build. "auto" builds every provider whose
# toolchain is installed; a semicolon list names them explicitly, for example
# -DSHADOWSPILL_BACKENDS=cuda or, through pip,
# --config-settings=cmake.define.SHADOWSPILL_BACKENDS=cuda. The mock backend
# is always built. A named provider whose toolchain is missing fails the
# configure step instead of silently disappearing.
set(SHADOWSPILL_BACKENDS "auto" CACHE STRING "Provider backends to build: auto, or a list such as cuda")

set(shadowspill_known_backends cuda)
if(SHADOWSPILL_BACKENDS STREQUAL "auto")
  set(shadowspill_requested_backends "")
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    list(APPEND shadowspill_requested_backends cuda)
  endif()
  if(NOT shadowspill_requested_backends)
    message(STATUS "No provider toolchain found; building the mock backend only")
  endif()
else()
  set(shadowspill_requested_backends ${SHADOWSPILL_BACKENDS})
  foreach(backend IN LISTS shadowspill_requested_backends)
    if(NOT backend IN_LIST shadowspill_known_backends)
      message(FATAL_ERROR "Unknown backend '${backend}'; known: ${shadowspill_known_backends}")
    endif()
  endforeach()
endif()
message(STATUS "ShadowSpill backends: mock ${shadowspill_requested_backends}")

if("cuda" IN_LIST shadowspill_requested_backends)
  find_package(CUDAToolkit REQUIRED)
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
        $<INSTALL_INTERFACE:shadowspill/include>
    )
    target_compile_definitions(
      shadowspill_backend_cuda PRIVATE SHADOWSPILL_BACKEND_CUDA_BUILDING
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
  endif()
endif()
