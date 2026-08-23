# Runtime C API

Include `<shadowspill/runtime.h>`. The runtime owns memory pools, object
residency, allocation records, transfer lanes, completion frontiers, the
worker, trace buffers, and first-failure state.

## Lifecycle and calibration

- `shadowspill_runtime_create()` validates the backend/configuration, allocates
  pool arenas, creates lanes, and starts the worker.
- `shadowspill_runtime_reserve_event_leases()` grows and seals the neutral
  event-record inventory at an idle cold-plan boundary. Repeated calls support
  additional callables sharing the same runtime without racing existing work.
- `shadowspill_runtime_reserve_retirement_records()` does the same for the
  immutable records queued between logical release and physical reclamation.
  A sealed inventory never falls back to `malloc` on the task or worker path.
- `shadowspill_runtime_reserve_memory_lease_records()` grows and seals one
  pool's reusable `MemoryLease` metadata and lease-use inventories. A
  lease-use record begins as one distinct stream attribution and is converted
  in place into its retirement requirement; neither task release nor the
  worker copies that list. Physical range release returns both record types to
  the pool owner. The same cold call reserves the pool's prospective-release
  frontier and range workspace, so a pressure-driven destination reservation
  can test coalescing pending ranges without allocating under the pool lock.
  Exhaustion after sealing fails closed instead of allocating process-heap
  metadata.
- `shadowspill_runtime_close()` stops new work, drains or reports failure,
  stops and joins the worker, closes lanes and pools, and is idempotent.
- `shadowspill_runtime_destroy()` performs close and releases the handle.
- `shadowspill_runtime_wait_idle()` waits at an explicit lifecycle boundary.
- `shadowspill_runtime_calibrate_transfer_capabilities()` measures selected
  directions.
- `shadowspill_runtime_transfer_profiles()` reads the published immutable
  transfer matrix.
- `shadowspill_memory_pool_grow()` grows one explicitly selected pool only when
  the header's idle-state preconditions hold.

Calibration first measures each available directed route alone. When reverse
routes exist, it then measures both directions simultaneously on independent
lanes and publishes the concurrent per-direction rates as the effective
`bandwidth_bytes_per_second`. Each `ShadowSpillTransferProfile` retains solo
and concurrent bandwidth, measurement duration, latency, copy geometry,
generation, mode, timestamp, and provenance. Planning consumes the immutable
matrix; it does not benchmark routes itself.

## Allocation API

- `shadowspill_memory_pool_allocate()` leases a compatible range from an
  explicitly selected pool for the active allocation scope.
- `shadowspill_memory_pool_allocation_for_pointer()` resolves the stable
  allocation record that owns a pointer within one pool.
- `shadowspill_memory_pool_free()` performs logical release and records causal
  retirement in that pool.
- `shadowspill_memory_pool_record_stream()` adds a stream use that must
  complete before the range can be reused.

No backend operation runs while the pool lock is held. A request waits only
when a known pending transition can satisfy it; otherwise it returns a
structured no-progress status.

## Object API

- `shadowspill_register_object()` and `shadowspill_unregister_object()` manage
  public object-table membership.
- `shadowspill_rekey_object()` changes the public identity without changing
  the retained object record.
- `shadowspill_write_object()` and `shadowspill_read_object()` copy
  bytes through a declared pool route.
- `shadowspill_plan_publish_initial_allocation()` publishes cold residency
  through one plan-local object binding.
- `shadowspill_task_publish_allocation()` atomically publishes a task output
  or replacement generation through a predecoded publication ordinal.
- `shadowspill_transfer_acquired_object_to_caller()` hands an acquired terminal
  generation to caller ownership while preserving stream readiness.
- `shadowspill_object_snapshot()` returns a lock-consistent diagnostic view.
- `shadowspill_object_location_snapshot()` returns the same object's current
  lease state in one explicitly selected pool.
- `shadowspill_object_handle_acquire()` creates an opaque retained owner for a
  runtime-global logical object.
- `shadowspill_object_handle_release()` releases that owner. The object is
  reclaimed only after registration, plans, and public handles have all
  released ownership.
- `shadowspill_object_release_generation()` releases one exact completed
  residency generation while preserving the logical object and its plan
  bindings. A later execution may publish a replacement generation into the
  same object.

Object pointers retained by task records and queued actions stay valid
after table removal until their own references are released.

## Task and execution API

`ShadowSpillPlan` owns one callable's immutable topology while sharing the
runtime's pool, route, event, and object owners:

- `shadowspill_plan_create()` creates a plan from explicit pool and route IDs.
- `shadowspill_plan_bind_object()` maps a Program-local object identity to a
  retained `ShadowSpillObjectHandle` with causal or explicitly unordered
  consistency. The plan owns an independent reference after the call returns.
- `shadowspill_plan_admit_task()` copies one immutable task topology and
  returns its direct repeated-path handle in the same cold-path call.
- `shadowspill_task_id()` and `shadowspill_task_trace_label()` expose the
  handle's immutable diagnostic identity without a table lookup. The returned
  label is borrowed from the handle and remains valid until its plan is
  cleared or destroyed.
- `shadowspill_plan_publish_initial_allocation()` publishes cold
  materialization through a plan-local object binding and the plan's selected
  execution pool; it does not create a fake task boundary.
- `shadowspill_task_publish_allocation()` updates one predecoded logical
  object by task-owned publication ordinal. Bind and replacement publication
  preserve the same logical object identity.
- `shadowspill_task_validate_replacement_binding()` validates that a frontend
  view names the replacement publication's exact retired lease while its
  successor tensor names the current lease.
- `shadowspill_before_task_handle()` and `shadowspill_after_task_handle()` are
  the sole production execution boundary. For an action-bearing task, the
  after boundary publishes its preallocated batch and actively waits only for
  worker submission acknowledgement, never for route completion.
- `shadowspill_abort_task_handle()` closes that same handle-bound task scope
  when frontend execution raises before `after_task`; it does not cancel work
  already submitted to the device.
- `shadowspill_plan_admit_action_batch()` creates an action-only trigger
  handle; `shadowspill_submit_action_batch_handle()` publishes it without
  opening a task boundary.
- `shadowspill_plan_admit_object_acquisition()` creates an immutable direct
  object set; `shadowspill_acquire_objects_handle()` snapshots its current
  generations and inserts readiness waits without opening a task boundary.
- `shadowspill_transfer_acquired_object_to_caller()` transfers one acquired
  ordinal after atomically validating its expected address and generation.
- `shadowspill_plan_admit_fixed_layout()` and
  `shadowspill_plan_seal_fixed_layout()` install the plan's physical layout.
- `shadowspill_plan_clear_tasks()` discards admitted records and bindings.
- `shadowspill_plan_wait_idle()` actively waits for only that plan's claimed
  task scopes, submitted actions, and task-owned retirements. Other plans on
  the same runtime do not participate.
- `shadowspill_plan_close()` and `shadowspill_plan_destroy()` release plan-owned
  references without closing the shared runtime.

Task handles bypass repeated task-ID and profiler-label lookup. Admission
retains the semantic label, direct object references, and predecoded actions
for the complete plan lifetime.
It also allocates the exact byte-state workspace used to validate that task's
allocation contract, so `before_task()` never grows a thread-local matcher.
The handle owns its exact expanded input-binding array as well. A successful
`shadowspill_before_task_handle()` returns a borrowed immutable view of that
array; the view remains valid through the matching `after_task()` or abort and
requires no caller allocation or binding copy.
One task handle is deliberately non-reentrant because its admitted action and
validation records are reused in place; concurrent callables use distinct
plan-owned handles and may remain active on the same runtime. Plan-local idle
waiting uses monotonic atomics and `cpu_relax`, not the runtime-global lifecycle
condition variable.
Initial placement and caller-output acquisition use their dedicated handles;
they never impersonate execution tasks or allocate per-invocation identities.

Physical placement is installed with `shadowspill_plan_admit_fixed_layout()`
and made immutable by `shadowspill_plan_seal_fixed_layout()`. Allocation
callbacks then validate task/ordinal/size/ownership before returning the
admitted offset.
See [Physical admission and offset handling](../architecture/physical-admission.md)
for the layout certificate and offset coordinate systems.

## Telemetry and failure

Structural profiling attributes allocator activity through a dedicated,
non-execution boundary:

- `shadowspill_allocation_scope_begin()` opens one allocator-attribution scope
  against an explicitly selected pool.
- `shadowspill_allocation_scope_end()` retires its anonymous allocations behind
  the supplied stream fence and closes the scope.
- `shadowspill_allocation_scope_abort()` rolls back an interrupted scope.

Allocation scopes do not resolve task records, publish object mutations,
decode actions, or enter the task API. They exist only where isolated
compilation/profiling needs the runtime allocator and its causal retirement
rules.

Allocation profiling uses:

- `shadowspill_allocation_telemetry_start()`
- `shadowspill_allocation_telemetry_stop()`
- `shadowspill_allocation_telemetry_read()`

Runtime tracing uses:

- `shadowspill_trace_prepare()`
- `shadowspill_trace_begin()`
- `shadowspill_trace_end()`
- `shadowspill_trace_read()`

`shadowspill_runtime_statistics()` returns aggregate pool and action counters,
including capacity, current/peak use, and rejected growth for neutral event
leases, retirement records, memory-lease records, and lease-use records.
`shadowspill_runtime_failure()` returns the first latched failure.
`shadowspill_runtime_recover_no_progress()` performs the explicit recovery
operation defined by the header; it does not hide an infeasible request.

`shadowspill_abi_version()` and `shadowspill_status_string()` cover loading
and error reporting for this boundary as for every other; see the
[C API guide](README.md#abi-use).

`shadowspill_failure_reason_string()` names the condition behind a status in
one sentence. Every site that latches a failure supplies one: the reason is a
required argument, not an option, so a report can always say what was
attempted and refused. The status is the class a caller acts on; the reason is what a
reader needs. Several reasons share one status on purpose - a lease that
cannot be released and a process allocator that refuses a record are both
internal failures a caller treats alike, but a reader must be able to tell
them apart.

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

Replay statuses occupy 80-89 of the one status vocabulary, so
`shadowspill_status_string()` names them like any other.
