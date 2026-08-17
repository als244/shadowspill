# Runtime C API

Include `<shadowspill/runtime.h>`. The runtime owns memory pools, object
residency, allocation records, transfer lanes, completion frontiers, the
worker, trace buffers, and first-failure state.

## Lifecycle and calibration

- `shadowspill_runtime_create()` validates the backend/configuration, allocates
  pool arenas, creates lanes and event inventory, and starts the worker.
- `shadowspill_runtime_close()` stops new work, drains or reports failure,
  stops and joins the worker, closes lanes and pools, and is idempotent.
- `shadowspill_runtime_destroy()` performs close and releases the handle.
- `shadowspill_runtime_wait_idle()` waits at an explicit lifecycle boundary.
- `shadowspill_runtime_calibrate_transfer_capabilities()` measures selected
  directions.
- `shadowspill_runtime_transfer_profiles()` reads the published immutable
  transfer matrix.
- `shadowspill_runtime_resize_spill_pool()` changes the configured spill arena
  only when the header's idle-state preconditions hold.

## Allocation API

- `shadowspill_allocate()` leases a compatible range for the active task.
- `shadowspill_allocation_for_pointer()` resolves the stable allocation record
  that owns a pointer.
- `shadowspill_free()` performs logical release and records causal retirement.
- `shadowspill_record_stream()` adds a stream use that must complete before
  reuse.

No backend operation runs while the pool lock is held. A request waits only
when a known pending transition can satisfy it; otherwise it returns a
structured no-progress status.

## Object API

- `shadowspill_register_object()` and `shadowspill_unregister_object()` manage
  public object-table membership.
- `shadowspill_rekey_object()` changes the public identity without changing
  the retained object record.
- `shadowspill_write_spill_object()` and `shadowspill_read_spill_object()` copy
  bytes through a declared pool route.
- `shadowspill_bind_object()` publishes a lease as an object residency.
- `shadowspill_replace_object_allocation()` atomically publishes a mutation or
  output replacement generation.
- `shadowspill_transfer_object_to_caller()` hands a terminal allocation to
  caller ownership while preserving stream readiness.
- `shadowspill_object_snapshot()` returns a lock-consistent diagnostic view.

Object pointers retained by execution records and queued actions stay valid
after table removal until their own references are released.

## Task and execution API

The direct task boundary is `shadowspill_before_task()` /
`shadowspill_after_task()`, with `shadowspill_abort_task()` for rollback.

Resolved execution plans use:

- `shadowspill_admit_execution()`
- `shadowspill_clear_execution_plan()`
- `shadowspill_resolve_execution()`
- `shadowspill_before_execution()` and
  `shadowspill_before_execution_handle()`
- `shadowspill_after_execution()` and
  `shadowspill_after_execution_handle()`

Handle variants bypass repeated task-ID lookup. Admission retains direct
object references and predecoded actions for the complete plan lifetime.

Physical placement is installed with `shadowspill_admit_fixed_layout()` and
made immutable by `shadowspill_seal_fixed_layout()`. Allocation callbacks then
validate task/ordinal/size/ownership before returning the admitted offset.
See [Physical admission and offset handling](../architecture/physical-admission.md)
for the layout certificate and offset coordinate systems.

## Telemetry and failure

Allocation profiling uses:

- `shadowspill_allocation_telemetry_start()`
- `shadowspill_allocation_telemetry_stop()`
- `shadowspill_allocation_telemetry_read()`

Runtime tracing uses:

- `shadowspill_trace_prepare()`
- `shadowspill_trace_begin()`
- `shadowspill_trace_end()`
- `shadowspill_trace_read()`

`shadowspill_runtime_statistics()` returns aggregate pool and action counters.
`shadowspill_runtime_failure()` returns the first latched failure.
`shadowspill_runtime_recover_no_progress()` performs the explicit recovery
operation defined by the header; it does not hide an infeasible request.

`shadowspill_runtime_abi_version()` and
`shadowspill_runtime_status_string()` support loading and error reporting.

## Threading

Frontend task calls and allocator callbacks may run concurrently with the
worker. Object, pool, lane, completion, trace, and lifecycle owners provide
their own synchronization. Backend calls are made after snapshotting and
retaining the necessary records, outside unrelated locks.

## Admission replay

Include `<shadowspill/admission_replay.h>` for deterministic, backend-free
replay of `MemoryPool` ownership transitions.

`shadowspill_admission_replay_run()` allocates temporary replay state for one
call. Repeated evaluations use
`shadowspill_admission_replay_workspace_create()`,
`shadowspill_admission_replay_run_reusing()`, and
`shadowspill_admission_replay_workspace_destroy()` to avoid heap work.

Operations cover acquire, retirement begin/completion, dependency publication,
reservation, reserved acquisition, and release. Results include every
allocator decision, reuse dependency, peak allocation/reservation/
fragmentation, and the first infeasible live-lease ledger.

Use `shadowspill_admission_replay_abi_version()` and
`shadowspill_admission_replay_status_string()` at the boundary.
