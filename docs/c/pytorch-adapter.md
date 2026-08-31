# PyTorch adapter C API

Include `<shadowspill/pytorch_adapter.h>`. This is the only compiled component
that knows PyTorch allocator and storage conventions. It does four things:
installs PyTorch's allocator, bootstraps the process-global neutral runtime and
publishes its handle, wraps provider streams and profiler ranges at task
boundaries, and validates PyTorch storage views.

It deliberately does not restate the neutral runtime. Anything reachable with a
handle the neutral runtime already owns is called there directly, so this
header carries only what needs PyTorch or the provider.

## Bootstrap and capabilities

- `shadowspill_pytorch_allocator_bootstrap()` installs the allocator and
  process-owned runtime from explicit pool and directed-route registries.
- `shadowspill_pytorch_allocator_close()` permanently closes the installed
  runtime, joins its worker, and releases its routes, pools, and backend. The
  PyTorch allocator shim remains installed and rejects future allocations.
- `shadowspill_pytorch_adapter_capabilities()` reports the adapter contract.
- `shadowspill_pytorch_physical_memory()`,
  `shadowspill_pytorch_physical_admission()`,
  `shadowspill_pytorch_check_physical_budget()`, and
  `shadowspill_pytorch_seal_physical_budget()` expose and seal physical limits.
  Sealing also reserves both neutral event records and backend event handles;
  a later callable may explicitly grow both inventories during plan adoption.
- `shadowspill_pytorch_calibrate_transfer_capabilities()` and
  `shadowspill_pytorch_transfer_profiles()` manage transfer calibration.

## Allocator callbacks

- `shadowspill_pytorch_cuda_malloc()`
- `shadowspill_pytorch_cuda_free()`
- `shadowspill_pytorch_cuda_record_stream()`
- `shadowspill_pytorch_allocation_for_pointer()`

The provider-specific spelling on these callback symbols matches PyTorch's
allocator hook. The neutral runtime and pool API do not use provider names. A
nonzero allocation failure is surfaced as a typed frontend exception before
compiled code can use an invalid address.

## Objects and storage

- `shadowspill_pytorch_register_object()` and
  `shadowspill_pytorch_register_placeholder_object()` create runtime objects.
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
- `shadowspill_pytorch_transfer_acquired_object_to_caller()` and
  `shadowspill_pytorch_release_caller_allocation()` manage public outputs.
- `shadowspill_object_snapshot()` returns diagnostic state.
- `shadowspill_object_location_snapshot()` returns one explicit
  pool-location view without assigning execution or spill meaning to it.
- `shadowspill_object_handle_acquire()` and
  `shadowspill_object_handle_release()` retain and release opaque
  runtime-global object ownership across callable boundaries.
- `shadowspill_object_release_generation()` releases one exact completed
  residency generation without destroying its logical object or plan
  binding.

## Execution boundaries

These publish the neutral plan owner without introducing frontend object
semantics:

- `shadowspill_pytorch_validate_task_replacement_binding()`
- `shadowspill_pytorch_submit_action_batch_handle()`
- `shadowspill_pytorch_acquire_objects_handle()`
- `shadowspill_pytorch_transfer_acquired_object_to_caller()`

Each of those either wraps a provider stream or validates a frontend storage
view, which is work only this library can do. Everything else in plan
admission needs nothing but handles the neutral runtime already owns, so the
frontend calls those on the neutral library, passing the runtime from
`shadowspill_pytorch_runtime_handle()`:

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

Acquiring an object handle stays on the adapter while releasing one does not:
acquiring resolves an id against the bound runtime, releasing needs only the
handle.

`shadowspill_pytorch_runtime_handle()` publishes the neutral runtime this
process bound. Everything reachable with that handle alone is called on the
neutral library; what remains here needs something only this library has.

Plan creation receives explicit execution/spill pool IDs and fetch/evict route
IDs. The adapter does not infer routes from global runtime roles.

Task calls mirror the neutral runtime:

- `shadowspill_pytorch_wait_task_allocations()` forwards the boundary's
  range-reuse resolution to the neutral runtime on the caller's current
  compute stream. Its operator carries no tensor, so it is registered against
  the schema rather than against a dispatch key.
- `shadowspill_pytorch_before_task_handle()` and
  `shadowspill_pytorch_after_task_handle()` are the production task boundary.
  The before boundary exposes the task-owned borrowed binding array instead
  of copying bindings into caller storage. The storage operators consume that
  view in place and return no per-task generation container to Python. Both
  boundaries derive task identity and the semantic profiler label from the
  admitted handle; no parallel task ID or mutable label table exists. The
  after boundary returns once the continuously active worker acknowledges
  submission of eligible actions, not when their asynchronous copies finish.

Fixed placement uses the plan-owned admission and sealing calls above. The
certificate and its runtime projection are described in [Physical admission
and offset handling](../architecture/physical-admission.md).

## Profiling and tracing

- `shadowspill_pytorch_allocation_scope_begin()`,
  `shadowspill_pytorch_allocation_scope_end()`, and
  `shadowspill_pytorch_allocation_scope_abort()` attribute isolated profiling
  allocations without creating a fake execution task.
- `shadowspill_pytorch_profiler_annotations_set()` enables the profiler
  backend.
- `shadowspill_pytorch_profile_range_begin()` and
  `shadowspill_pytorch_profile_range_end()` manage explicit ranges.
- `shadowspill_pytorch_abort_task_handle()` closes the matching admitted task
  scope and its profiler range after frontend execution aborts.

Structured runtime tracing (`shadowspill_trace_prepare()`,
`shadowspill_trace_begin()`, `shadowspill_trace_end()`,
`shadowspill_trace_read()`) and allocation profiling
(`shadowspill_allocation_telemetry_start()`,
`shadowspill_allocation_telemetry_stop()`,
`shadowspill_allocation_telemetry_read()`) are neutral runtime calls the
frontend makes directly, passing the handle from
`shadowspill_pytorch_runtime_handle()`.

## Failure and lifecycle

`shadowspill_pytorch_allocator_failure()` and
`shadowspill_pytorch_allocator_statistics()` return structured state.
`shadowspill_pytorch_allocator_wait_idle()` is an explicit lifecycle barrier;
`shadowspill_plan_wait_idle()` is the plan-local active-poll boundary
used for callable recurrence and teardown. It ignores unrelated plans;
`shadowspill_pytorch_recover_no_progress()` performs the documented recovery
operation.

`shadowspill_pytorch_allocator_close()` performs deterministic teardown and is
idempotent. The adapter also registers the same cleanup at process exit as a
last resort. Python callables must still be closed explicitly so errors and
ownership violations are reported at the correct boundary.
