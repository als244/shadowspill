# PyTorch adapter C API

Include `<shadowspill/pytorch_adapter.h>`. This is the one library that knows
PyTorch allocator and storage conventions. It does four things:
installs PyTorch's allocator, bootstraps the process-global neutral runtime and
publishes its handle, wraps provider streams and profiler ranges at task
boundaries, and validates PyTorch storage views.

It deliberately does not restate the neutral runtime. Anything reachable with a
handle the neutral runtime already owns is called there directly, so this
header carries only what needs PyTorch.

## What the adapter exposes

The header is ordered the way this page is, with a section banner at each
heading below: vocabulary and descriptions; bootstrap, physical admission and
close; the allocator callbacks; objects and storage; task boundaries and
allocation scopes; failure and recovery. Every symbol is prefixed
`shadowspill_pytorch_`. The one heading with no banner behind it is
[Profiling](#profiling), which is here to say the adapter exposes none.

## What the adapter requires of a backend

The [backend contract](backends.md) and nothing else.
`ShadowSpillPytorchAdapterConfig.backend_library` names the shared object to
load; bootstrap opens it with `dlopen()`, resolves
`shadowspill_backend_create()` and `shadowspill_backend_destroy()`, validates
the table with `shadowspill_backend_is_valid()`, and keeps it for the life of
the runtime. The adapter links no provider library and includes no provider
header; see [backends](../architecture/backends.md).

## Vocabulary and descriptions

`ShadowSpillPytorchAdapterConfig` is what bootstrap takes: the pools and
directed routes as `ShadowSpillPytorchPoolConfig` and
`ShadowSpillPytorchRouteConfig` arrays, which id and name what the runtime
will build, the device budget and the provider's headroom, the worker's
poll interval and the background transfer window it passes through to the
runtime, and the backend library by path. The adapter hands back
`ShadowSpillPytorchPhysicalAdmission` (the ledger as sealed),
`ShadowSpillPytorchAdapterCapabilities` (the three contract versions and
whether the storage operators were built), `ShadowSpillPytorchAdapterStatistics`
(the callback counters, with the runtime's statistics, the backend's, and
`allocator_pool`, the statistics of the pool the allocator is bound to, inside)
and `ShadowSpillPytorchAdapterFailure` (the first failure, with the runtime's
record inside). `SHADOWSPILL_PYTORCH_ADAPTER_ABI_VERSION` versions all of it.

Two ids the frontend synthesises for work that is not a planned task are
defined here so the failure report can decode them: allocation scopes take
ids at or above `SHADOWSPILL_PYTORCH_PROFILING_SCOPE_BASE`, and the pre-task
placement batch runs as `SHADOWSPILL_PYTORCH_INITIAL_ACTIONS_TASK_ID`. The
frontend copies both.

## Bootstrap, physical admission and close

- `shadowspill_pytorch_allocator_bootstrap()` installs the allocator and
  process-owned runtime from explicit pool and directed-route registries.
- `shadowspill_pytorch_allocator_close()` permanently closes the installed
  runtime, joins its worker, releases its routes, pools, and backend, and
  closes the backend library. It refuses while any caller-owned allocation is
  still outstanding. The PyTorch allocator shim remains installed and
  rejects future allocations. Close is deterministic and idempotent.
  Bootstrap also registers a process-exit handler as a last resort, which
  closes the same way except that it waits for nothing and refuses nothing:
  outstanding transfers cannot complete once exit handlers run, and it reports
  on `stderr` what was still outstanding. Python callables must still be
  closed explicitly so errors and ownership violations are reported at the
  correct boundary.
- `shadowspill_pytorch_adapter_capabilities()` reports the adapter contract.
- `shadowspill_pytorch_runtime_handle()` publishes the neutral runtime this
  process bound. Everything reachable with that handle alone is called on the
  neutral library; what remains here needs something only this library has.
- `shadowspill_pytorch_physical_memory()`,
  `shadowspill_pytorch_physical_admission()`,
  `shadowspill_pytorch_check_physical_budget()`, and
  `shadowspill_pytorch_seal_physical_budget()` expose and seal physical limits.
  Sealing confirms the profiled provider reserve fits the bootstrap
  reservation; it never resizes or weakens the budget. Its second argument is
  a record reserve it passes straight through to the neutral runtime, sealing
  the event leases, the retirement records, and every pool's memory-lease
  records in one call, so no steady-state step allocates one. A later callable
  may grow any of those inventories again during plan adoption.
- Transfer calibration is the neutral runtime's:
  `shadowspill_runtime_calibrate_transfer_capabilities()` and
  `shadowspill_runtime_transfer_profiles()`, called with the handle.

## The allocator callbacks

- `shadowspill_pytorch_backend_malloc()`
- `shadowspill_pytorch_backend_free()`
- `shadowspill_pytorch_backend_record_stream()`
- `shadowspill_pytorch_allocation_for_pointer()`

The first three are the symbols PyTorch's pluggable allocator is pointed at.
A nonzero allocation failure is surfaced as a typed frontend exception before
compiled code can use an invalid address. The fourth is the read-only lookup
that says which allocation a pointer belongs to, used to classify profiled
task outputs.

## Objects and storage

Only the four `shadowspill_pytorch_` entries below are this library's; they
are here because each wraps a provider stream or a PyTorch storage view. The
rest of the object vocabulary is the neutral runtime's, listed here for the
shape of the workflow and specified in the [Runtime API](runtime.md#object-api);
the frontend calls those with the handle from
`shadowspill_pytorch_runtime_handle()`.

- `shadowspill_register_object()` creates runtime objects, resident in a pool
  or as placeholders, and `shadowspill_write_object()` populates them.
- `shadowspill_unregister_object()` and `shadowspill_rekey_object()` manage
  identity.
- `shadowspill_plan_publish_initial_allocation()` and
  `shadowspill_task_publish_allocation()` publish initial and repeated task
  storages through immutable plan/task records.
- `shadowspill_pytorch_validate_object_binding()` rejects stale imported CPU
  storage views. Device storage acquisition is validated by its admitted task
  or object-acquisition handle before the adapter installs the returned
  address.
- `shadowspill_write_object()` and `shadowspill_read_object()` move persistent
  state through an explicitly selected pool.
- `shadowspill_acquire_objects_handle()` acquires the objects an admitted
  acquisition names for a consumer stream. It is the neutral runtime's: it
  takes the integer its caller names the stream by and asks the backend which
  stream that is, so no adapter entry point is needed to reach it.
  `shadowspill_pytorch_transfer_acquired_object_to_caller()` and
  `shadowspill_pytorch_release_caller_allocation()` hand one to the caller and
  take it back; they are the storage operators' own and have no ctypes caller.
- `shadowspill_object_snapshot()` returns diagnostic state.
- `shadowspill_object_location_snapshot()` returns one explicit
  pool-location view without assigning execution or spill meaning to it.
- `shadowspill_object_handle_acquire()` and
  `shadowspill_object_handle_release()` retain and release opaque
  runtime-global object ownership across callable boundaries.
- `shadowspill_object_release_generation()` releases one exact completed
  residency generation without destroying its logical object or plan
  binding.

## Task boundaries and allocation scopes

The pre-task action batch is `shadowspill_submit_action_batch_handle()`, the
neutral runtime's. It used to be mirrored here, because a stream had to be
turned into a backend token and only this library could do that; the runtime
resolves the caller's stream itself now, so the mirror is gone. The same is true
of object acquisition under [Objects and storage](#objects-and-storage).

Everything else in plan admission needs nothing but handles the neutral runtime
already owns, so the frontend calls those on the neutral library, passing the
runtime from `shadowspill_pytorch_runtime_handle()`:

- `shadowspill_plan_bind_object()`, `shadowspill_plan_admit_task()`,
  `shadowspill_plan_publish_initial_allocation()`
- `shadowspill_plan_admit_fixed_layout()`,
  `shadowspill_plan_seal_fixed_layout()`
- `shadowspill_plan_admit_object_acquisition()`,
  `shadowspill_plan_admit_action_batch()`, `shadowspill_plan_create()`
- `shadowspill_object_handle_acquire()`,
  `shadowspill_task_publish_allocation()`
- `shadowspill_plan_close()`, `shadowspill_plan_destroy()`,
  `shadowspill_plan_clear_tasks()`, `shadowspill_plan_wait_idle()`
- `shadowspill_object_handle_release()`,
  `shadowspill_object_release_generation()`
- `shadowspill_register_object()`, `shadowspill_write_object()`,
  `shadowspill_runtime_wait_idle()`, and, from the storage operators,
  `shadowspill_task_validate_replacement_binding()`

Acquiring an object handle stays on the adapter while releasing one does not:
acquiring resolves an id against the bound runtime, releasing needs only the
handle.

Plan creation receives the plan id together with explicit execution/spill pool
IDs and fetch/evict route IDs. The adapter does not infer routes from global
runtime roles.

Task calls mirror the neutral runtime:

- `shadowspill_pytorch_before_task_handle()` and
  `shadowspill_pytorch_after_task_handle()` are the production task boundary.
  The before boundary exposes the task-owned borrowed binding array instead
  of copying bindings into caller storage. The storage operators consume that
  view in place and return no per-task generation container to Python. Both
  boundaries derive task identity and the semantic profiler label from the
  admitted handle; no parallel task ID or mutable label table exists. The
  after boundary returns once the continuously active worker acknowledges
  submission of eligible actions, not when their asynchronous copies finish.
- The allocation wait between them is an operator rather than an entry point
  here: `torch.ops.shadowspill._wait_task_allocations` forwards the
  boundary's range-reuse resolution to the neutral runtime on the caller's
  current compute stream. It carries no tensor, so it is registered against
  the schema rather than against a dispatch key.
- `shadowspill_pytorch_abort_task_handle()` closes the matching admitted task
  scope and its profiler range after frontend execution aborts.
- `shadowspill_pytorch_allocation_scope_begin()`,
  `shadowspill_pytorch_allocation_scope_end()`, and
  `shadowspill_pytorch_allocation_scope_abort()` attribute isolated profiling
  allocations without creating a fake execution task. `begin` takes the plan the
  measurements are for alongside the scope id, and refuses one that names no
  live plan: a scope runs outside any task, so the runtime has none to read the
  plan from, and anything a probe leaves behind would otherwise be attributable
  to nothing. See [plan identity](../architecture/plan-identity.md).

Fixed placement uses the plan-owned admission and sealing calls above. The
certificate and its runtime projection are described in [Physical admission
and offset handling](../architecture/physical-admission.md).

## Profiling

The adapter has none of its own. Ranges and the annotation flag are the
runtime's -- `shadowspill_profiler_annotations_set()`,
`shadowspill_profiler_range_begin()` and `shadowspill_profiler_range_end()` in
[runtime.md](runtime.md#profiler-annotations) -- because the runtime owns the
backend the ranges go to. The adapter opens ranges on the runtime it is bound
to, like any other caller, and exports nothing for it.

Structured runtime tracing (`shadowspill_trace_prepare()`,
`shadowspill_trace_begin()`, `shadowspill_trace_end()`,
`shadowspill_trace_read()`) and allocation profiling
(`shadowspill_allocation_telemetry_start()`,
`shadowspill_allocation_telemetry_stop()`,
`shadowspill_allocation_telemetry_read()`) are neutral runtime calls the
frontend makes directly, passing the handle from
`shadowspill_pytorch_runtime_handle()`.

## Failure and recovery

`shadowspill_pytorch_allocator_failure()` and
`shadowspill_pytorch_allocator_statistics()` return structured state: the
first failure, which is what stopped the runtime, and the counters around it.
`shadowspill_runtime_wait_idle()`, called with the handle, is the explicit
lifecycle barrier; `shadowspill_plan_wait_idle()` is the plan-local
active-poll boundary used for callable recurrence and teardown, and ignores
unrelated plans. `shadowspill_pytorch_recover_no_progress()` performs the one
documented recovery: the frontend synchronizes the execution device first,
and only a latched NO_PROGRESS failure can be cleared.
