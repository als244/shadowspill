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
  simulator/src/admission.c
  simulator/src/diagnostics.c
  simulator/src/events.c
  simulator/src/memory.c
  simulator/src/simulator.c
  simulator/src/status.c
  simulator/src/tasks.c
  simulator/src/transfers.c
  simulator/src/validation.c

  runtime/src/admission_replay.c
  runtime/src/allocation_scope.c
  runtime/src/allocations.c
  runtime/src/completion_tracker.c
  runtime/src/event_pool.c
  runtime/src/failure_state.c
  runtime/src/fixed_layout.c
  runtime/src/idle_wakeup.c
  runtime/src/memory_pool.c
  runtime/src/object_acquisition.c
  runtime/src/object_actions.c
  runtime/src/object_table.c
  runtime/src/objects.c
  runtime/src/plan.c
  runtime/src/plan_object_bindings.c
  runtime/src/profiler.c
  runtime/src/ranges.c
  runtime/src/retirement_queue.c
  runtime/src/runtime.c
  runtime/src/status.c
  runtime/src/task_boundaries.c
  runtime/src/task_table.c
  runtime/src/telemetry.c
  runtime/src/trace.c
  runtime/src/transfer_lane.c
  runtime/src/transfer_profiles.c
  runtime/src/worker.c

  planner/src/admission/candidate.c
  planner/src/admission/lifetimes.c
  planner/src/admission/operations.c
  planner/src/admission/placement.c
  planner/src/admission/topology.c
  planner/src/context.c
  planner/src/digest.c
  planner/src/portfolio.c
  planner/src/residency.c
  planner/src/schedule.c
  planner/src/selector.c
  planner/src/status.c
)
target_include_directories(
  shadowspill
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/simulator/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/runtime/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/planner/include>
    $<INSTALL_INTERFACE:shadowspill/include>
  PRIVATE
    simulator/src
    runtime/src
    planner/src
)
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
  FILES
    simulator/include/shadowspill/simulator.h
    runtime/include/shadowspill/runtime.h
    runtime/include/shadowspill/admission_replay.h
    runtime/include/shadowspill/backend.h
    runtime/include/shadowspill/profiler.h
    planner/include/shadowspill/planner.h
  DESTINATION shadowspill/include/shadowspill
)

add_subdirectory(runtime/backends)
add_subdirectory(pytorch_adapter)
