add_executable(shadowspill_build_canary build_canary.c)
target_link_libraries(shadowspill_build_canary PRIVATE shadowspill_build_options)
add_test(NAME shadowspill_build_canary COMMAND shadowspill_build_canary)

add_executable(shadowspill_simulator_canary simulator_canary.c)
target_link_libraries(
  shadowspill_simulator_canary
  PRIVATE shadowspill shadowspill_build_options
)
add_test(NAME shadowspill_simulator_canary COMMAND shadowspill_simulator_canary)

add_executable(shadowspill_planner_canary planner_canary.c)
target_link_libraries(
  shadowspill_planner_canary
  PRIVATE shadowspill shadowspill_build_options
)
add_test(NAME shadowspill_planner_canary COMMAND shadowspill_planner_canary)

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
  ${PROJECT_SOURCE_DIR}/csrc/src/runtime/memory/memory_pool.c
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

foreach(runtime_test IN ITEMS blocking calibration completion failure telemetry transitions)
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

add_executable(
  shadowspill_transfer_lane_canary
  runtime/transfer_lane_canary.c
  ${PROJECT_SOURCE_DIR}/csrc/src/runtime/transfers/transfer_lane.c
)
target_include_directories(
  shadowspill_transfer_lane_canary
  PRIVATE
    ${PROJECT_SOURCE_DIR}/csrc/include
    ${PROJECT_SOURCE_DIR}/csrc/src/runtime
)
target_compile_definitions(shadowspill_transfer_lane_canary PRIVATE SHADOWSPILL_BUILDING)
target_link_libraries(
  shadowspill_transfer_lane_canary
  PRIVATE shadowspill_build_options Threads::Threads
)
add_test(NAME shadowspill_transfer_lane_canary COMMAND shadowspill_transfer_lane_canary)
set_tests_properties(
  shadowspill_runtime_completion_canary
  shadowspill_runtime_failure_canary
  PROPERTIES TIMEOUT 10
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
