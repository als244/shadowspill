add_executable(shadowspill_build_canary build_canary.c)
target_link_libraries(shadowspill_build_canary PRIVATE shadowspill_build_options)
add_test(NAME shadowspill_build_canary COMMAND shadowspill_build_canary)

if(TARGET shadowspill_pytorch)
  add_executable(shadowspill_abi_sizes_canary abi_sizes_canary.c)
  target_include_directories(
    shadowspill_abi_sizes_canary
    PRIVATE
      ${PROJECT_SOURCE_DIR}/csrc/include
      ${PROJECT_SOURCE_DIR}/csrc/adapter/pytorch/include
  )
  target_link_libraries(
    shadowspill_abi_sizes_canary PRIVATE shadowspill_build_options
  )
endif()

add_executable(
  shadowspill_backend_contract_canary backends/backend_contract_canary.c
)
target_include_directories(
  shadowspill_backend_contract_canary PRIVATE ${PROJECT_SOURCE_DIR}/csrc/include
)
target_link_libraries(
  shadowspill_backend_contract_canary
  PRIVATE ${CMAKE_DL_LIBS} shadowspill_build_options
)
add_test(
  NAME shadowspill_backend_contract_canary
  COMMAND shadowspill_backend_contract_canary $<TARGET_FILE:shadowspill_backend_mock>
)
# The value-wait sequence blocks until a word arrives, so a backend that stops
# honouring it hangs here rather than failing. Bound it.
set_tests_properties(shadowspill_backend_contract_canary PROPERTIES TIMEOUT 30)

add_executable(shadowspill_simulator_canary simulator/simulator_canary.c)
target_link_libraries(
  shadowspill_simulator_canary
  PRIVATE shadowspill shadowspill_build_options
)
add_test(NAME shadowspill_simulator_canary COMMAND shadowspill_simulator_canary)

add_executable(
  shadowspill_pressurefit_canary
  planner/search/algorithms/pressurefit/pressurefit_canary.c
)
target_link_libraries(
  shadowspill_pressurefit_canary
  PRIVATE shadowspill shadowspill_build_options
)
add_test(
  NAME shadowspill_pressurefit_canary
  COMMAND shadowspill_pressurefit_canary
)

add_executable(shadowspill_runtime_canary runtime/runtime_canary.c)
target_link_libraries(
  shadowspill_runtime_canary
  PRIVATE
    shadowspill
    shadowspill_backend_mock
    shadowspill_build_options
)
add_test(NAME shadowspill_runtime_canary COMMAND shadowspill_runtime_canary)

add_executable(shadowspill_runtime_plan_canary runtime/runtime_plan_canary.c)
target_link_libraries(
  shadowspill_runtime_plan_canary
  PRIVATE
    shadowspill
    shadowspill_backend_mock
    shadowspill_build_options
    Threads::Threads
)
add_test(NAME shadowspill_runtime_plan_canary COMMAND shadowspill_runtime_plan_canary)

add_executable(
  shadowspill_memory_pool_canary
  runtime/memory_pool_canary.c
  ${PROJECT_SOURCE_DIR}/csrc/src/runtime/memory/memory_pool/memory.c
  ${PROJECT_SOURCE_DIR}/csrc/src/runtime/memory/memory_pool/records.c
  ${PROJECT_SOURCE_DIR}/csrc/src/runtime/memory/memory_pool/locks.c
  ${PROJECT_SOURCE_DIR}/csrc/src/runtime/memory/memory_pool/leases.c
  ${PROJECT_SOURCE_DIR}/csrc/src/runtime/memory/memory_pool/causal.c
  ${PROJECT_SOURCE_DIR}/csrc/src/runtime/memory/ranges.c
)
target_include_directories(
  shadowspill_memory_pool_canary
  PRIVATE
    ${PROJECT_SOURCE_DIR}/csrc/include
    ${PROJECT_SOURCE_DIR}/csrc/src/runtime
)
target_compile_definitions(shadowspill_memory_pool_canary PRIVATE SHADOWSPILL_BUILDING)
target_link_libraries(
  shadowspill_memory_pool_canary
  PRIVATE shadowspill_build_options Threads::Threads
)
add_test(NAME shadowspill_memory_pool_canary COMMAND shadowspill_memory_pool_canary)

add_executable(shadowspill_fixed_layout_canary runtime/fixed_layout_canary.c)
target_link_libraries(
  shadowspill_fixed_layout_canary
  PRIVATE
    shadowspill
    shadowspill_backend_mock
    shadowspill_build_options
    Threads::Threads
)
add_test(NAME shadowspill_fixed_layout_canary COMMAND shadowspill_fixed_layout_canary)
set_tests_properties(shadowspill_fixed_layout_canary PROPERTIES TIMEOUT 10)

add_executable(
  shadowspill_admission_replay_canary
  runtime/admission_replay_canary.c
)
target_include_directories(
  shadowspill_admission_replay_canary
  PRIVATE ${PROJECT_SOURCE_DIR}/csrc/include
)
target_link_libraries(
  shadowspill_admission_replay_canary
  PRIVATE shadowspill shadowspill_build_options
)
add_test(
  NAME shadowspill_admission_replay_canary
  COMMAND shadowspill_admission_replay_canary
)

foreach(runtime_test IN ITEMS abandon blocking calibration completion failure telemetry transitions)
  add_executable(
    shadowspill_runtime_${runtime_test}_canary
    runtime/runtime_${runtime_test}_canary.c
  )
  target_link_libraries(
    shadowspill_runtime_${runtime_test}_canary
    PRIVATE
      shadowspill
      shadowspill_backend_mock
      shadowspill_build_options
      Threads::Threads
  )
  add_test(
    NAME shadowspill_runtime_${runtime_test}_canary
    COMMAND shadowspill_runtime_${runtime_test}_canary
  )
endforeach()
target_include_directories(
  shadowspill_runtime_transitions_canary
  PRIVATE ${PROJECT_SOURCE_DIR}/csrc/src/runtime
)
# The blocking canary drives a pool's own release accounting, which is the
# state a waiting allocator watches and is not public.
target_include_directories(
  shadowspill_runtime_blocking_canary
  PRIVATE ${PROJECT_SOURCE_DIR}/csrc/src/runtime
)

add_executable(
  shadowspill_transfer_queue_canary
  runtime/transfer_queue_canary.c
  ${PROJECT_SOURCE_DIR}/csrc/src/runtime/transfers/transfer_queue.c
)
target_include_directories(
  shadowspill_transfer_queue_canary
  PRIVATE
    ${PROJECT_SOURCE_DIR}/csrc/include
    ${PROJECT_SOURCE_DIR}/csrc/src/runtime
)
target_compile_definitions(shadowspill_transfer_queue_canary PRIVATE SHADOWSPILL_BUILDING)
target_link_libraries(
  shadowspill_transfer_queue_canary
  PRIVATE shadowspill_build_options Threads::Threads
)
add_test(NAME shadowspill_transfer_queue_canary COMMAND shadowspill_transfer_queue_canary)
set_tests_properties(
  shadowspill_runtime_completion_canary
  shadowspill_runtime_failure_canary
  PROPERTIES TIMEOUT 10
)

# A pool whose memory is on another machine. Labelled `network`, which at this
# stage means "a daemon is reachable" -- the canary skips without one, and the
# label is what keeps it out of the default suite.
add_executable(shadowspill_remote_pool_canary network/remote_pool_canary.c)
target_link_libraries(
  shadowspill_remote_pool_canary
  PRIVATE shadowspill shadowspill_backend_mock shadowspill_build_options
)
add_test(NAME shadowspill_remote_pool_canary COMMAND shadowspill_remote_pool_canary)
set_tests_properties(
  shadowspill_remote_pool_canary
  PROPERTIES
    LABELS "network"
    TIMEOUT 30
    ENVIRONMENT
      "SHADOWSPILL_NETWORK_LIBRARY=$<TARGET_FILE:shadowspill_network>"
)

if(TARGET shadowspill_backend_cuda)
  add_executable(shadowspill_cuda_backend_canary runtime/cuda_backend_canary.c)
  target_link_libraries(
    shadowspill_cuda_backend_canary
    PRIVATE
      shadowspill
      shadowspill_backend_cuda
      shadowspill_build_options
  )
  add_test(NAME shadowspill_cuda_backend_canary COMMAND shadowspill_cuda_backend_canary)
  set_tests_properties(shadowspill_cuda_backend_canary PROPERTIES LABELS "cuda")
endif()

# The adapter's C entry points against the mock backend, loaded by path the
# way the frontend loads a provider. Needs no accelerator and no libtorch; the
# storage operators are the Python canaries' to cover.
if(TARGET shadowspill_pytorch)
  add_executable(
    shadowspill_pytorch_adapter_canary adapter/pytorch_adapter_canary.c
  )
  target_include_directories(
    shadowspill_pytorch_adapter_canary
    PRIVATE
      ${PROJECT_SOURCE_DIR}/csrc/adapter/pytorch/include
      ${CMAKE_CURRENT_SOURCE_DIR}/runtime
  )
  target_link_libraries(
    shadowspill_pytorch_adapter_canary
    PRIVATE
      shadowspill_pytorch
      shadowspill
      shadowspill_backend_mock
      shadowspill_build_options
      Threads::Threads
  )
  add_test(
    NAME shadowspill_pytorch_adapter_canary
    COMMAND
      shadowspill_pytorch_adapter_canary $<TARGET_FILE:shadowspill_backend_mock>
  )
  set_tests_properties(
    shadowspill_pytorch_adapter_canary
    PROPERTIES
      TIMEOUT 30
      # The extension library the canary loads through the adapter. Not
      # verbs-gated, so it is built wherever this canary is.
      ENVIRONMENT
        "SHADOWSPILL_NETWORK_LIBRARY=$<TARGET_FILE:shadowspill_network>"
  )
endif()
