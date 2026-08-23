# One library for everything that plans and executes a step.
#
# The simulator, the planner and the runtime were three shared objects with a
# strict dependency order between them, and nothing else ever linked them
# apart. Separating them bought no independent deployment and cost a second
# and third ABI to keep in sync, a status vocabulary per component, and a
# library-load path per component on the Python side.
#
# Two things stay separate because they are genuinely pluggable: a backend is
# dlopened and compiled against CUDA or nothing, and the PyTorch adapter links
# libtorch, which planning-only callers must not be made to carry.
add_library(
  shadowspill
  SHARED
  src/common/platform.c
  src/common/status.c

  src/simulator/admission.c
  src/simulator/diagnostics.c
  src/simulator/events.c
  src/simulator/memory.c
  src/simulator/simulator.c
  src/simulator/tasks.c
  src/simulator/transfers.c
  src/simulator/validation.c

  src/planner/admission/candidate.c
  src/planner/admission/lifetimes.c
  src/planner/admission/operations.c
  src/planner/admission/placement.c
  src/planner/admission/facts.c
  src/planner/problem.c
  src/planner/digest.c
  src/planner/candidates.c
  src/planner/residency.c
  src/planner/schedule.c
  src/planner/selector.c
  src/planner/result.c

  src/runtime/failure_state.c
  src/runtime/runtime.c
  src/runtime/failure_reason.c
  src/runtime/worker.c
  src/runtime/memory/allocation_scope.c
  src/runtime/memory/allocations.c
  src/runtime/memory/memory_pool.c
  src/runtime/memory/ranges.c
  src/runtime/memory/retirement_queue.c
  src/runtime/objects/object_acquisition.c
  src/runtime/objects/object_actions.c
  src/runtime/objects/object_table.c
  src/runtime/objects/objects.c
  src/runtime/plan/admission_replay.c
  src/runtime/plan/fixed_layout.c
  src/runtime/plan/plan.c
  src/runtime/plan/plan_object_bindings.c
  src/runtime/sync/completion_tracker.c
  src/runtime/sync/event_pool.c
  src/runtime/sync/idle_wakeup.c
  src/runtime/tasks/task_boundaries.c
  src/runtime/tasks/task_scope.c
  src/runtime/tasks/task_table.c
  src/runtime/telemetry/allocation_events.c
  src/runtime/telemetry/profiler.c
  src/runtime/telemetry/trace.c
  src/runtime/transfers/transfer_lane.c
  src/runtime/transfers/transfer_profiles.c
)
target_include_directories(
  shadowspill
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:shadowspill/include>
)
target_compile_definitions(shadowspill PRIVATE SHADOWSPILL_BUILDING)
target_link_libraries(
  shadowspill
  PRIVATE Threads::Threads shadowspill_build_options
)
set_property(
  TARGET shadowspill
  PROPERTY OUTPUT_NAME shadowspill
)
set_property(TARGET shadowspill PROPERTY VERSION ${PROJECT_VERSION})
set_property(TARGET shadowspill PROPERTY SOVERSION 0)

install(
  TARGETS shadowspill
  LIBRARY DESTINATION shadowspill/lib
  RUNTIME DESTINATION shadowspill/lib
)
install(
  DIRECTORY include/shadowspill
  DESTINATION shadowspill/include
)

add_subdirectory(backends)
add_subdirectory(adapter/pytorch)
