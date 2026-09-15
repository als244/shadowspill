# Runtime C API

Include `<shadowspill/runtime.h>`, which is the umbrella over one header per
subsystem in `<shadowspill/runtime/>`; the section below each heading names the
part it comes from. The runtime owns memory pools, object residency,
allocation records, transfer lanes, completion frontiers, the worker, trace
buffers, and first-failure state.

## Lifecycle and calibration

- `shadowspill_backend_is_valid()` checks a backend table for the contract
  version and every required entry; the runtime applies it at create and the
  adapter after loading a backend library.
- `shadowspill_runtime_create()` takes a `ShadowSpillRuntimeConfig`: the
  backend table it copies, pools as `ShadowSpillMemoryPoolDescription`
  (`pool_id`, a `ShadowSpillPoolKind` of device or pinned host, capacity,
  alignment), and routes as
  `ShadowSpillTransferRouteDescription` (`route_id`, name, source and
  destination pool ids, whose kinds must differ), the worker's poll interval,
  and `background_transfer_window_bytes`, how far a lane may run ahead with
  transfers the plan did not schedule (zero removes the bound). It validates
  the table and the topology, allocates or maps each pool's arena, creates one
  lane per route, and starts the worker; see
  [memory pools](../architecture/memory-pools.md) and
  [transfers](../architecture/transfers.md).
- `shadowspill_runtime_reserve_event_leases()` grows and seals the event-lease
  inventory at an idle cold-plan boundary, creating the backend events up
  front so a steady-state step makes no driver calls; see
  [events](../architecture/events.md). Repeated calls support additional
  plans sharing the same runtime without racing existing work.
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
- `shadowspill_runtime_calibrate_transfer_capabilities()` measures the routes
  named by a `ShadowSpillTransferRouteKey` array, or every configured route
  when that array is NULL and its count zero, under a
  `ShadowSpillTransferCalibrationConfig` giving the two copy sizes and the
  warmup and measured counts. The runtime must be locally idle, and one
  successful call publishes one new matrix generation atomically.
- `shadowspill_runtime_transfer_profiles()` copies the published immutable
  transfer matrix: the complete row-major N-by-N grid, so `capacity` must be
  at least N*N, with the generation and count it was consistent at.
- `shadowspill_memory_pool_grow()` grows one explicitly selected pool's arena,
  copying the old arena into the new one and rebasing every live lease, so
  offsets and payloads survive. It waits for idle first and then refuses
  unless the runtime is quiet -- not closing, no queued action, no pending
  retirement -- and refuses a capacity below the current one. A capacity equal
  to the current one is accepted and does nothing.

`shadowspill_runtime_abandon()` closes without waiting for anything: no drain,
no lane synchronization, and no cleanup that could block on a lock the worker
holds. It is for a process that is already exiting, where waiting prevents the
exit rather than delaying it, and where everything a drain protects is
reclaimed at process exit anyway. It still stops the worker and releases every
route, event pool, and memory pool, which is what unregisters pinned host
memory and frees device memory. The two counts it fills in, either of which may
be NULL, report what was still outstanding, so a caller can say so. See
[failure, abort, and process exit](../architecture/failure-and-exit.md).

Calibration first measures each available directed route alone. When reverse
routes exist, it then measures both directions simultaneously on independent
lanes and publishes the concurrent per-direction rates as the effective
`bandwidth_bytes_per_second`. Each `ShadowSpillTransferProfile` retains solo
and concurrent bandwidth, measurement duration, latency, copy geometry,
generation, its `ShadowSpillTransferCalibrationMode`, timestamp, and its
`ShadowSpillTransferProfileProvenance` -- whether the cell came from
initialization or from a later recalibration. An identity cell is available at
zero latency and needs no copy. Planning consumes the immutable
matrix; it does not benchmark routes itself.

## Allocation API

- `shadowspill_memory_pool_allocate()` leases a compatible range from an
  explicitly selected pool for the active allocation scope, filling a
  `ShadowSpillAllocation` with the pool, id, generation, requested and charged
  bytes, and the address. It leases from the existing arena and never grows
  it.
- `shadowspill_memory_pool_allocation_for_pointer()` resolves the same record
  from an exact live address, which is what an allocator callback carrying an
  address rather than an id needs.
- `shadowspill_memory_pool_free()` performs logical release and records causal
  retirement in that pool.
- `shadowspill_memory_pool_record_stream()` adds a stream use that must
  complete before the range can be reused.

No backend operation runs while the pool lock is held. A request waits only
when a known pending transition can satisfy it; otherwise it returns a
structured no-progress status.

## Object API

- `shadowspill_register_object()` admits one `ShadowSpillObjectDescription` --
  id, size, initial version, initial pool, whether a spill copy is retained
  and whether it starts resident -- and `shadowspill_unregister_object()`
  removes an object that is spill-only or released, with no live allocation
  and no queued action.
- `shadowspill_rekey_object()` changes the public identity without changing
  the retained object record.
- `shadowspill_write_object()` and `shadowspill_read_object()` copy
  bytes through a declared pool route.
- `shadowspill_object_snapshot()` returns a lock-consistent diagnostic view as
  a `ShadowSpillObjectSnapshot`, whose `residency` is a
  `ShadowSpillObjectResidency`: spill-only, execution-ready, fetching,
  evicting, or released.
- `shadowspill_object_location_snapshot()` returns the same object's current
  lease state in one explicitly selected pool, as a
  `ShadowSpillObjectLocationSnapshot`, which assigns that pool no execution or
  spill meaning.
- `shadowspill_object_handle_acquire()` creates a `ShadowSpillObjectHandle`,
  an opaque retained owner for a runtime-global logical object that stays
  valid across generation and residency changes.
- `shadowspill_object_handle_release()` releases that owner. The object is
  reclaimed only after registration, plans, and public handles have all
  released ownership.
- `shadowspill_object_release_generation()` releases one exact completed
  residency generation while preserving the logical object and its plan
  bindings. A later execution may publish a replacement generation into the
  same object.

Object pointers retained by task records and queued actions stay valid
after table removal until their own references are released.

## Plan identity

A pool outlives any one plan and more than one plan may share it, so a lease
records which plan's scope made it. Task ids cannot serve: the frontend mints
them from the program's numbering, so task 1112 exists in every plan.

- `shadowspill_runtime_next_plan_id()` hands out the ids. It only counts up, so
  an id names one plan for the life of the runtime and is never reissued, not
  even after the plan holding it is destroyed. Taken before the plan is created,
  because allocation scopes opened for the plan name it too and the runtime has
  no task there to read a plan from.
- `shadowspill_plan_create()` requires that id in its description. An id the
  runtime did not issue is refused, and so is one some plan has already been
  created with. Zero is not a plan id: it is what a lease made outside any plan
  reports.
- `shadowspill_plan_id()` reports the id a plan holds.
- `shadowspill_plan_reclaim_scoped_leases()` takes back every range the plan's own
  scopes allocated, whatever still points at it, and reports how many. This is the
  forcing path for a plan that is closing: it works in leases and bytes, does not
  consult the framework, and so a caller that may still read an object backed by
  one of those ranges must drop it first -- the PyTorch frontend detaches the
  storages before calling it. A lease the framework has not freed keeps its pointer
  indexed, so the free that eventually arrives still resolves. Ranges carrying no
  plan are left alone.
- `shadowspill_runtime_plan()` is the inverse, mapping an id back to its record,
  or `NULL` once that record is gone. The pointer is valid only while the caller
  knows the plan is live.
- `shadowspill_runtime_plan_state()` says what became of an id without touching
  the record: `UNKNOWN` for one no plan was created with, `LIVE` for an open
  plan, `CLOSED` for one that admits nothing more while its record is still
  present, and `DESTROYED` once that record is freed.

The state outlives the record deliberately. A closing plan releases the ranges
its own scopes made, so a live allocation naming a `CLOSED` or `DESTROYED` plan
is a defect rather than an expected state, and the id is what makes that defect
visible instead of invisible. The one intended exception is an object shared
between plans, which outlives any one of them by design.

Allocations taken outside any plan's scope are a separate case: they carry no
plan at all, report `SHADOWSPILL_RUNTIME_NO_ID` for the scope, and are not a
plan's to reclaim.

`shadowspill_allocation_scope_begin()` takes the same id, and refuses one that
names no live plan.

The registry behind these is `csrc/src/runtime/plan/registry.c`, with its own
lock so asking what an id means never waits behind plan creation or teardown.
See [plan identity](../architecture/plan-identity.md) for the reasoning.

## Task and execution API

`ShadowSpillPlan` owns one plan's immutable topology while sharing the
runtime's pool, route, event, and object owners:

- `shadowspill_plan_create()` creates a plan from a
  `ShadowSpillPlanDescription`: the plan id, then the execution and spill pool
  ids and the fetch and evict route ids, all explicit, none inferred from a
  runtime-wide role.
- `shadowspill_plan_bind_object()` maps a program-local object identity to a
  retained `ShadowSpillObjectHandle` with a `ShadowSpillObjectConsistency` of
  causal or explicitly unordered.
  The plan owns an independent reference after the call returns.
- `shadowspill_plan_admit_task()` copies one `ShadowSpillTaskDescription` --
  the task's inputs, its `ShadowSpillObjectUpdate` mutations, its
  `ShadowSpillTaskPublicationDescription` outputs, its actions, its
  `ShadowSpillTaskAllocationContractStep` sequence, each a
  `ShadowSpillTaskAllocationOperation` of allocate or free, and the envelope
  bounding it -- and returns the `ShadowSpillTaskHandle` every later
  invocation uses, in the same cold-path call.
- `shadowspill_task_id()` and `shadowspill_task_trace_label()` expose the
  handle's immutable diagnostic identity without a table lookup. The returned
  label is borrowed from the handle and remains valid until its plan is
  cleared or destroyed.
- `shadowspill_plan_publish_initial_allocation()` publishes cold
  materialization through a plan-local object binding and the plan's selected
  execution pool; it does not create a fake task boundary.
- `shadowspill_task_publish_allocation()` updates one predecoded logical
  object by task-owned publication ordinal and fills a
  `ShadowSpillObjectBinding` with the generation, allocation and address it
  published. Both `ShadowSpillTaskPublicationKind` values -- bind and
  replacement -- preserve the same logical object identity; replacement
  changes only its physical lease and generation.
- `shadowspill_task_validate_replacement_binding()` validates that a frontend
  view names the replacement publication's exact retired lease while its
  successor tensor names the current lease.
- `shadowspill_before_task_handle()` and `shadowspill_after_task_handle()` are
  the sole production execution boundary. For an action-bearing task, the
  after boundary publishes its preallocated batch and actively waits only for
  worker submission acknowledgement, never for route completion.
- `shadowspill_wait_task_allocations_handle()` resolves, at the boundary, the
  range-reuse dependency of every allocation the plan pinned to this task. The
  allocator resolves the same dependency where the allocation happens, which
  is inside the task; doing it here as well means the wait falls in an
  interval the boundary owns and can be measured apart from compute. It is
  idempotent with respect to the allocator's own call, which then finds every
  dependency already published.
- `shadowspill_abort_task_handle()` closes that same handle-bound task scope
  when frontend execution raises before `after_task`; it does not cancel work
  already submitted to the device.
- A task's actions are `ShadowSpillRuntimeAction` records of one of the four
  `ShadowSpillRuntimeActionKind` values: release, evict, fetch and
  write-back. A write-back copies the execution copy to the spill pool and
  keeps it; one scheduled while the spill copy is already current completes
  without a copy; a release scheduled behind a pending write-back of its
  object frees the execution copy once that copy has landed.
- `shadowspill_plan_admit_action_batch()` creates a
  `ShadowSpillActionBatchHandle`, an action-only trigger with no task;
  `shadowspill_submit_action_batch_handle()` publishes it without
  opening a task boundary.
- `shadowspill_plan_admit_object_acquisition()` creates a
  `ShadowSpillObjectAcquisitionHandle` over an immutable ordered object set;
  `shadowspill_acquire_objects_handle()` snapshots its current
  generations into caller-owned `ShadowSpillObjectBinding` entries and inserts
  readiness waits, without opening a task or allocation scope.
- `shadowspill_transfer_acquired_object_to_caller()` transfers one acquired
  ordinal after atomically validating its expected address and generation.
- `shadowspill_plan_admit_fixed_layout()` copies and validates one
  `ShadowSpillFixedLayoutDescription` -- the slice, its
  `ShadowSpillFixedPlacementDescription` entries, each carrying a
  `ShadowSpillFixedPlacementKind`, and the
  `ShadowSpillFixedDependencyDescription` proofs behind every reused address --
  and reserves the single parent slice.
  `shadowspill_plan_seal_fixed_layout()` resolves the task and action
  identities after task admission and makes the layout immutable. Allocation
  callbacks then validate task, ordinal, size and ownership before returning
  the admitted offset. See [physical admission and offset
  handling](../architecture/physical-admission.md) for the layout certificate
  and the offset coordinate systems.
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
validation records are reused in place; plans running at the same time use
handles of their own and may remain active on the same runtime. Plan-local idle
waiting uses monotonic atomics and `cpu_relax`, not the runtime-global lifecycle
condition variable.
Initial placement and caller-output acquisition use their dedicated handles;
they never impersonate execution tasks or allocate per-invocation identities.

## Telemetry and failure

Structural profiling attributes allocator activity through a dedicated,
non-execution boundary:

- `shadowspill_allocation_scope_begin()` opens one allocator-attribution scope
  against an explicitly selected pool, for the live plan whose id it is given
  (see [Plan identity](#plan-identity)).
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

It records `ShadowSpillAllocationEvent` entries, each a
`ShadowSpillAllocationEventKind` (created, released, promoted, logically
freed) against a `ShadowSpillAllocationCategory` (anonymous, planned object,
caller owned). Passing a null buffer and zero capacity to the read queries the
count.

Runtime tracing uses:

- `shadowspill_trace_prepare()`, which allocates the two rings from a
  `ShadowSpillTraceConfig` and does not enable tracing
- `shadowspill_trace_begin()`
- `shadowspill_trace_end()`
- `shadowspill_trace_read()`, which fills a `ShadowSpillTraceSummary` beside
  the events and, with null arrays, reports the counts alone

Neither ring grows from a hot path, and neither may grow while a session is
active. Both are diagnostic and a step never depends on either: a ring that
fills stops recording and lets the step continue, and says so through
`event_overflow` and `allocation_event_overflow` on the summary, so a caller
can tell an incomplete record from a complete one.

`shadowspill_trace_begin()` takes the caller's origin event: a timing event
the caller has already recorded on its compute stream and keeps alive for
the trace. While the trace is active the worker brackets every copy it
dispatches with two timing events from the runtime's timing pool on the lane,
and the transfer's
`SHADOWSPILL_TRACE_TRANSFER_COMPLETED` event carries the copy's interval
from that origin in `lane_started_at_ns` and `lane_finished_at_ns`. Every other
event kind, and a completion the backend could not measure, carries
`SHADOWSPILL_TRACE_NO_STREAM_TIME` in both. A zero origin token records no
intervals. `timestamp_ns` on every event is the host clock; the two stream
fields are the only device-clock values in the trace.

### What a trace event carries

Every `ShadowSpillTraceEvent` carries its sequence, host timestamp, the step
id the trace was begun with, and whichever of task, object and allocation id
apply, with `SHADOWSPILL_RUNTIME_NO_ID` where one does not. `bytes` is the
object or allocation size the event is about, and zero at a task boundary.

`detail_0` and `detail_1` are two words the `ShadowSpillTraceEventKind` gives
meaning to. Kinds below are named without their `SHADOWSPILL_TRACE_` prefix.

| Kind | `detail_0` | `detail_1` |
|---|---|---|
| `SESSION_BEGIN`, `SESSION_END` | zero | zero |
| `BEFORE_TASK` | the task's declared input count | actions queued runtime-wide at that moment |
| `AFTER_TASK` | the `ShadowSpillStatus` the boundary is returning | the task's admitted action count |
| `READINESS_WAIT` | 1 when a wait was inserted on the consumer stream; 0 on the refusal, where the object still had an unpublished fetch and there was no readiness event to wait on | wait events inserted so far, or, on the refusal, the queued action count |
| `ACTION_QUEUED` | the `ShadowSpillRuntimeActionKind` queued | actions this boundary published |
| `DESTINATION_RESERVED` | the `ShadowSpillRuntimeActionKind` the destination is for | the reserved lease's slab offset |
| `TRANSFER_DISPATCHED`, `TRANSFER_COMPLETED` | the `ShadowSpillTransferDirection`; a write-back reports evict, since it shares that lane | actions queued runtime-wide at that moment |
| `ALLOCATION_WAIT_BEGIN`, `ALLOCATION_WAIT_END` | the pool's free bytes | its largest free range |
| `RETIREMENT_COMPLETED` | the retired lease's slab offset | its charged bytes |
| `FAILURE_LATCHED` | the `ShadowSpillStatus` being latched | the pool's free bytes at that moment |

`shadowspill_runtime_statistics()` copies a lock-consistent
`ShadowSpillRuntimeStatistics`: what the runtime holds that no pool does -- the
work in flight, the records it owns, and `pool_count`, the number of pools there
are to ask about. The records are the event leases and the retirement records,
each with capacity, current and peak use, and rejected growth. Event leases add
`event_lease_driver_creates` and `event_lease_sealed`, and the timing pool
contributes `timing_event_capacity`, `timing_event_in_use`,
`timing_event_peak_in_use`, and `timing_event_driver_creates`; a create after
sealing is a driver call the plan did not reserve for.

`shadowspill_memory_pool_statistics()` reports one pool's own numbers -- its
capacity, what is allocated and free in it, its largest free range and the
fragmentation that follows, its live allocation count, and its memory-lease and
lease-use record reserves.

The split follows ownership. A runtime may own any number of pools, and which of
them a plan uses as its execution and spill pools is that plan's choice, so a
pool's numbers belong to the pool rather than to named fields for two of them.
The PyTorch adapter's statistics carry the pool the allocator is bound to
alongside the runtime's, since that is the one a caller on the allocation path
wants.

`shadowspill_memory_pool_live_allocations()` answers the question statistics
cannot: not how many allocations a pool holds but *which*, and where.

It covers the ranges the pool has *published*, and only the execution-lease paths
publish. That makes it an execution-pool question in practice, which is what it
exists for: a contiguous-range refusal is an execution-arena problem, and position
is what explains one. Two things follow, and neither is an error to be reported.

Storage the runtime holds for a registered object is reserved without being
published, so it is absent from this list in either pool. Its lease carries
`SHADOWSPILL_RUNTIME_OBJECT_SCOPE_ID` as its scope, its bytes are in the pool's
statistics, and the storage itself is reached through the object registry.

Asked of a spill pool, the call succeeds and reports nothing, because a spill
copy -- an object's retained copy, or an eviction's destination -- is reserved
rather than published. Spill-pool occupancy is a statistics question.

It copies one `ShadowSpillLiveAllocation` per published live allocation into
caller-owned storage: allocation id, byte offset into the arena, charged and
requested bytes, the plan whose scope made it and the scope itself with that
scope's invocation and ordinal, the object it is bound to or
`SHADOWSPILL_RUNTIME_NO_ID` when it is bound to none, its reference count, and
five flags: `scratch` for task workspace, `plan_owned` for a range the plan
placed, `ever_plan_owned` which stays set after ownership moves on,
`logical_freed` for one the frontend has given up and is awaiting retirement,
and `framework_free_seen`.

`ever_plan_owned` without `plan_owned` is the signature of a range promoted out
to a named owner -- an output the caller holds now -- which is why the two are
reported separately. It is what separates a scope's leftover workspace, which a
closing plan may take back, from a range whose owner outlives the plan.

The count written is always the total live count, so a caller may pass a null
buffer with zero capacity to size one, and a buffer too small is reported by a
count greater than the capacity rather than by an error.

The object id travels rather than a role. The runtime does not know what an
object is *for* -- that is the program's to say -- so a frontend holding the
program resolves the id and reports a parameter or an activation, while the
runtime reports only that the range is bound.

The offsets are the point. A fixed layout refused for want of a contiguous
range is not explained by a count, because a small allocation in the wrong place
costs the largest free range while leaving the free total almost untouched, and
because an allocation still held after its scope ended can only be attributed if
something names the scope that made it.

`shadowspill_runtime_failure()` returns the first latched failure as a
`ShadowSpillRuntimeFailure`: the status, a `ShadowSpillFailureReason`, the
pool, task, object and allocation it names, and, for an allocation-contract
break, the expected and actual operation at that step.
`shadowspill_runtime_recover_no_progress()` clears a latched NO_PROGRESS
allocation failure -- and nothing else, since every other failure stays
latched -- so the worker can drain what it already owns and objects and pool
leases can be reclaimed. It exists for deterministic fault teardown and does
not hide an infeasible request: the caller must first synchronize every
external producer stream and let the failed allocator caller return.

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
`shadowspill_admission_replay_workspace_destroy()` to avoid heap work: a
`ShadowSpillAdmissionReplayWorkspace` sized once allocates nothing on a later
run whose program fits it.

`ShadowSpillAdmissionReplayProgram` is the pool's capacity and minimum
alignment, the lease and dependency counts the ids are bounded by, the
threshold at or above which a request splits its free range from the high end
rather than the low, and the ordered `ShadowSpillAdmissionReplayOperation`
entries themselves. Their `ShadowSpillAdmissionReplayOperationKind` covers
acquire, retirement begin/completion, dependency publication,
reservation, reserved acquisition, and release; these are ownership
transitions, not transfer semantics.

`ShadowSpillAdmissionReplayResult` reports one
`ShadowSpillAdmissionReplayDecision` per operation, carrying its offset, its
charged bytes, the physical delta it caused and the
`ShadowSpillAdmissionReplayLeaseState` it left the lease in; every
`ShadowSpillAdmissionReuseDependency` a consumer must wait on; the allocation,
reservation and fragmentation peaks; the final allocated, reserved and
largest-free figures; a digest over the decisions; and, as
`ShadowSpillAdmissionReplayLiveLease` entries, the exact ledger at the first
infeasible operation. Every output array is caller-owned and sized by its
capacity field.

Replay statuses occupy 80-89 of the one status vocabulary, so
`shadowspill_status_string()` names them like any other.
