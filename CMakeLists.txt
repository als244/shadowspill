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

include(CTest)

add_library(shadowspill_build_options INTERFACE)
target_compile_options(
  shadowspill_build_options
  INTERFACE
    $<$<COMPILE_LANG_AND_ID:C,GNU,Clang>:-Wall;-Wextra;-Werror;-Wpedantic>
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall;-Wextra;-Werror;-Wpedantic>
)

add_library(
  shadowspill_simulator
  SHARED
  simulator/src/diagnostics.c
  simulator/src/events.c
  simulator/src/memory.c
  simulator/src/simulator.c
  simulator/src/status.c
  simulator/src/tasks.c
  simulator/src/transfers.c
  simulator/src/validation.c
)
target_include_directories(
  shadowspill_simulator
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/simulator/include>
    $<INSTALL_INTERFACE:shadowspill/include>
  PRIVATE simulator/src
)
target_link_libraries(shadowspill_simulator PRIVATE shadowspill_build_options)
set_target_properties(
  shadowspill_simulator
  PROPERTIES
    OUTPUT_NAME shadowspill_simulator
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
)

install(
  TARGETS shadowspill_simulator
  LIBRARY DESTINATION shadowspill/lib
  RUNTIME DESTINATION shadowspill/lib
)
install(
  FILES simulator/include/shadowspill/simulator.h
  DESTINATION shadowspill/include/shadowspill
)

add_library(
  shadowspill_planner
  SHARED
  planner/src/residency.c
  planner/src/selector.c
  planner/src/status.c
)
target_include_directories(
  shadowspill_planner
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/planner/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/simulator/include>
    $<INSTALL_INTERFACE:shadowspill/include>
  PRIVATE planner/src
)
target_link_libraries(
  shadowspill_planner
  PRIVATE shadowspill_simulator shadowspill_build_options
)
set_target_properties(
  shadowspill_planner
  PROPERTIES
    OUTPUT_NAME shadowspill_planner
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
    INSTALL_RPATH "$ORIGIN"
)

install(
  TARGETS shadowspill_planner
  LIBRARY DESTINATION shadowspill/lib
  RUNTIME DESTINATION shadowspill/lib
)
install(
  FILES planner/include/shadowspill/planner.h
  DESTINATION shadowspill/include/shadowspill
)

find_package(Threads REQUIRED)

add_library(
  shadowspill_runtime
  SHARED
  runtime/src/allocations.c
  runtime/src/objects.c
  runtime/src/progress.c
  runtime/src/ranges.c
  runtime/src/runtime.c
  runtime/src/status.c
  runtime/src/telemetry.c
)
target_include_directories(
  shadowspill_runtime
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/runtime/include>
    $<INSTALL_INTERFACE:shadowspill/include>
  PRIVATE runtime/src
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
  runtime/backends/mock/src/backend_mock.c
)
target_include_directories(
  shadowspill_backend_mock
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/runtime/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/runtime/backends/mock/include>
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
    runtime/include/shadowspill/backend.h
    runtime/include/shadowspill/runtime.h
    runtime/backends/mock/include/shadowspill/backend_mock.h
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
      runtime/backends/cuda/src/backend_cuda.c
    )
    target_include_directories(
      shadowspill_backend_cuda
      PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/runtime/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/runtime/backends/cuda/include>
        $<INSTALL_INTERFACE:shadowspill/include>
    )
    target_link_libraries(
      shadowspill_backend_cuda
      PRIVATE
        CUDA::cuda_driver
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
      FILES runtime/backends/cuda/include/shadowspill/backend_cuda.h
      DESTINATION shadowspill/include/shadowspill
    )

    add_library(
      shadowspill_pytorch
      SHARED
      pytorch/csrc/allocator.c
    )
    target_include_directories(
      shadowspill_pytorch
      PRIVATE
        pytorch/csrc/include
        runtime/include
        runtime/backends/cuda/include
    )
    target_link_libraries(
      shadowspill_pytorch
      PRIVATE
        shadowspill_runtime
        shadowspill_backend_cuda
        Threads::Threads
        shadowspill_build_options
    )
    if(TARGET CUDA::nvtx3)
      target_link_libraries(shadowspill_pytorch PRIVATE CUDA::nvtx3)
    endif()
    if(UNIX AND NOT APPLE)
      target_link_options(
        shadowspill_pytorch
        PRIVATE
          "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/pytorch/csrc/shadowspill_pytorch.map"
      )
    endif()
    find_package(Torch QUIET)
    if(Torch_FOUND)
      target_sources(shadowspill_pytorch PRIVATE pytorch/csrc/storage.cpp)
      target_link_libraries(shadowspill_pytorch PRIVATE ${TORCH_LIBRARIES})
      target_compile_definitions(
        shadowspill_pytorch
        PRIVATE SHADOWSPILL_PYTORCH_STORAGE_ADAPTER=1
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
  else()
    message(STATUS "CUDA Toolkit not found; skipping CUDA backend")
  endif()
endif()

if(BUILD_TESTING)
  add_executable(shadowspill_build_canary tests/native/build_canary.c)
  target_link_libraries(shadowspill_build_canary PRIVATE shadowspill_build_options)
  add_test(NAME shadowspill_build_canary COMMAND shadowspill_build_canary)

  add_executable(
    shadowspill_simulator_canary
    tests/simulator/simulator_canary.c
  )
  target_link_libraries(
    shadowspill_simulator_canary
    PRIVATE shadowspill_simulator shadowspill_build_options
  )
  add_test(NAME shadowspill_simulator_canary COMMAND shadowspill_simulator_canary)

  add_executable(shadowspill_planner_canary tests/planner/planner_canary.c)
  target_link_libraries(
    shadowspill_planner_canary
    PRIVATE shadowspill_planner shadowspill_build_options
  )
  add_test(NAME shadowspill_planner_canary COMMAND shadowspill_planner_canary)

  add_executable(shadowspill_runtime_canary tests/runtime/runtime_canary.c)
  target_link_libraries(
    shadowspill_runtime_canary
    PRIVATE
      shadowspill_runtime
      shadowspill_backend_mock
      shadowspill_build_options
  )
  add_test(NAME shadowspill_runtime_canary COMMAND shadowspill_runtime_canary)

  foreach(runtime_test IN ITEMS blocking failure telemetry transitions)
    add_executable(
      shadowspill_runtime_${runtime_test}_canary
      tests/runtime/runtime_${runtime_test}_canary.c
    )
    target_link_libraries(
      shadowspill_runtime_${runtime_test}_canary
      PRIVATE
        shadowspill_runtime
        shadowspill_backend_mock
        shadowspill_build_options
        Threads::Threads
    )
    add_test(
      NAME shadowspill_runtime_${runtime_test}_canary
      COMMAND shadowspill_runtime_${runtime_test}_canary
    )
  endforeach()

  if(TARGET shadowspill_backend_cuda)
    add_executable(
      shadowspill_cuda_backend_canary
      tests/runtime/cuda_backend_canary.c
    )
    target_link_libraries(
      shadowspill_cuda_backend_canary
      PRIVATE
        shadowspill_runtime
        shadowspill_backend_cuda
        shadowspill_build_options
    )
    add_test(
      NAME shadowspill_cuda_backend_canary
      COMMAND shadowspill_cuda_backend_canary
    )
    set_tests_properties(
      shadowspill_cuda_backend_canary
      PROPERTIES LABELS "cuda"
    )

    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(
      NAME shadowspill_pytorch_allocator_canary
      COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/pytorch/allocator_canary.py
        $<TARGET_FILE:shadowspill_pytorch>
    )
    set_tests_properties(
      shadowspill_pytorch_allocator_canary
      PROPERTIES
        LABELS "cuda;pytorch"
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src"
    )
    add_test(
      NAME shadowspill_pytorch_overlap_canary
      COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/pytorch/overlap_canary.py
        $<TARGET_FILE:shadowspill_pytorch>
    )
    set_tests_properties(
      shadowspill_pytorch_overlap_canary
      PROPERTIES
        LABELS "cuda;pytorch"
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src"
        TIMEOUT 20
    )
    add_test(
      NAME shadowspill_pytorch_physical_failure_canary
      COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/pytorch/physical_failure_canary.py
        $<TARGET_FILE:shadowspill_pytorch>
    )
    set_tests_properties(
      shadowspill_pytorch_physical_failure_canary
      PROPERTIES
        LABELS "cuda;pytorch"
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src"
    )
    add_test(
      NAME shadowspill_pytorch_allocator_oom_canary
      COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/pytorch/allocator_oom_canary.py
        $<TARGET_FILE:shadowspill_pytorch>
    )
    set_tests_properties(
      shadowspill_pytorch_allocator_oom_canary
      PROPERTIES
        LABELS "cuda;pytorch"
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src"
    )
    add_test(
      NAME shadowspill_pytorch_profiler_canary
      COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/pytorch/profiler_canary.py
        $<TARGET_FILE:shadowspill_pytorch>
    )
    set_tests_properties(
      shadowspill_pytorch_profiler_canary
      PROPERTIES
        LABELS "cuda;pytorch"
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src"
        TIMEOUT 60
    )
    add_test(
      NAME shadowspill_pytorch_forward_canary
      COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/pytorch/forward_canary.py
        $<TARGET_FILE:shadowspill_pytorch>
    )
    set_tests_properties(
      shadowspill_pytorch_forward_canary
      PROPERTIES
        LABELS "cuda;pytorch"
        ENVIRONMENT
          "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src;SHADOWSPILL_SIMULATOR_LIBRARY=$<TARGET_FILE:shadowspill_simulator>;SHADOWSPILL_PLANNER_LIBRARY=$<TARGET_FILE:shadowspill_planner>"
        TIMEOUT 90
    )
    add_test(
      NAME shadowspill_pytorch_training_canary
      COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/pytorch/training_canary.py
        $<TARGET_FILE:shadowspill_pytorch>
    )
    set_tests_properties(
      shadowspill_pytorch_training_canary
      PROPERTIES
        LABELS "cuda;pytorch"
        ENVIRONMENT
          "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src;SHADOWSPILL_SIMULATOR_LIBRARY=$<TARGET_FILE:shadowspill_simulator>;SHADOWSPILL_PLANNER_LIBRARY=$<TARGET_FILE:shadowspill_planner>"
        TIMEOUT 120
    )
  endif()
endif()
