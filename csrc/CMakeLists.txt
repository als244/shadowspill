# One library for everything that plans and executes a step.
#
# Two things stay separate because they are genuinely pluggable: a backend is
# dlopened and compiled against a provider toolchain alone, and the PyTorch
# adapter links libtorch, which planning-only callers must not be made to
# carry. README.md has the rest of the rationale.
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

  # Search-agnostic: admitting a plan physically, and naming or choosing one.
  src/planner/admission/candidate.c
  src/planner/admission/facts.c
  src/planner/admission/lifetimes.c
  src/planner/admission/operations.c
  src/planner/admission/placement.c
  src/planner/abi.c
  src/planner/search/toolkit/digest.c
  src/planner/search/toolkit/schedule.c

  # One search.
  src/planner/search/algorithms/pressurefit/best_placed.c
  src/planner/search/algorithms/pressurefit/candidates/candidates.c
  src/planner/search/algorithms/pressurefit/candidates/search.c
  src/planner/search/algorithms/pressurefit/candidates/workers.c
  src/planner/search/algorithms/pressurefit/candidates/plan.c
  src/planner/search/algorithms/pressurefit/candidates/repair.c
  src/planner/search/algorithms/pressurefit/candidates/memo.c
  src/planner/search/algorithms/pressurefit/candidates/workspace.c
  src/planner/search/algorithms/pressurefit/candidates/diagnostics.c
  src/planner/search/algorithms/pressurefit/problem/buffers.c
  src/planner/search/algorithms/pressurefit/problem/facts.c
  src/planner/search/algorithms/pressurefit/problem/floor.c
  src/planner/search/algorithms/pressurefit/problem/placement.c
  src/planner/search/algorithms/pressurefit/problem/search.c
  src/planner/search/algorithms/pressurefit/residency/workspace.c
  src/planner/search/algorithms/pressurefit/residency/spans.c
  src/planner/search/algorithms/pressurefit/residency/cuts.c
  src/planner/search/algorithms/pressurefit/residency/index.c
  src/planner/search/algorithms/pressurefit/residency/canonical.c
  src/planner/search/algorithms/pressurefit/residency/tree.c
  src/planner/search/algorithms/pressurefit/residency/heap.c
  src/planner/search/algorithms/pressurefit/residency/reduce.c
  src/planner/search/algorithms/pressurefit/residency/sparse.c
  src/planner/search/algorithms/pressurefit/schedule/facts.c
  src/planner/search/algorithms/pressurefit/schedule/storage.c
  src/planner/search/algorithms/pressurefit/schedule/pressure.c
  src/planner/search/algorithms/pressurefit/schedule/triggers.c
  src/planner/search/algorithms/pressurefit/schedule/clamp.c
  src/planner/search/algorithms/pressurefit/schedule/fetches.c
  src/planner/search/algorithms/pressurefit/schedule/emit.c

  src/runtime/failure_state.c
  src/runtime/runtime.c
  src/runtime/runtime_close.c
  src/runtime/runtime_pools.c
  src/runtime/runtime_statistics.c
  src/runtime/failure_reason.c
  src/runtime/worker/worker.c
  src/runtime/worker/actions.c
  src/runtime/worker/dispatch.c
  src/runtime/worker/completion.c
  src/runtime/memory/allocation_scope.c
  src/runtime/memory/allocations/index.c
  src/runtime/memory/allocations/records.c
  src/runtime/memory/allocations/leases.c
  src/runtime/memory/allocations/reuse.c
  src/runtime/memory/allocations/allocate.c
  src/runtime/memory/allocations/free.c
  src/runtime/memory/memory_pool/arena.c
  src/runtime/memory/memory_pool/records.c
  src/runtime/memory/memory_pool/locks.c
  src/runtime/memory/memory_pool/leases.c
  src/runtime/memory/memory_pool/causal.c
  src/runtime/memory/ranges.c
  src/runtime/memory/retirement_queue.c
  src/runtime/objects/object_acquisition.c
  src/runtime/objects/object_actions.c
  src/runtime/objects/object_table.c
  src/runtime/objects/object_registry.c
  src/runtime/objects/object_owners.c
  src/runtime/objects/object_allocations.c
  src/runtime/objects/object_transfers.c
  src/runtime/plan/admission_replay.c
  src/runtime/plan/fixed_layout.c
  src/runtime/plan/plan.c
  src/runtime/plan/registry.c
  src/runtime/plan/plan_object_bindings.c
  src/runtime/sync/completion_tracker.c
  src/runtime/sync/event_pool.c
  src/runtime/sync/idle_wakeup.c
  src/runtime/sync/stream_interval.c
  src/runtime/tasks/task_boundaries.c
  src/runtime/tasks/task_scope.c
  src/runtime/tasks/task_table.c
  src/runtime/tasks/task_records.c
  src/runtime/tasks/task_admission.c
  src/runtime/tasks/task_handles.c
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
