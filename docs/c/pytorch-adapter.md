# PyTorch adapter C API

Include `<shadowspill/pytorch_adapter.h>`. The adapter is the only compiled
component that knows PyTorch allocator/storage conventions. It bootstraps the
neutral runtime and translates Python/ATen-facing values into runtime handles.

## Bootstrap and capabilities

- `shadowspill_pytorch_allocator_bootstrap()` installs the allocator and
  process-owned runtime.
- `shadowspill_pytorch_adapter_capabilities()` reports the adapter contract.
- `shadowspill_pytorch_physical_memory()`,
  `shadowspill_pytorch_physical_admission()`,
  `shadowspill_pytorch_check_physical_budget()`, and
  `shadowspill_pytorch_seal_physical_budget()` expose and seal physical limits.
  Sealing also reserves both neutral event records and backend event handles;
  a later callable may explicitly grow both inventories during plan adoption.
- `shadowspill_pytorch_calibrate_transfer_capabilities()` and
  `shadowspill_pytorch_transfer_profiles()` manage transfer calibration.
- `shadowspill_pytorch_resize_spill_pool()` performs the supported idle resize.

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

- `shadowspill_pytorch_register_host_object()` and
  `shadowspill_pytorch_register_placeholder_object()` create runtime objects.
- `shadowspill_pytorch_unregister_object()` and
  `shadowspill_pytorch_rekey_object()` manage identity.
- `shadowspill_pytorch_plan_publish_initial_allocation()` and
  `shadowspill_pytorch_task_publish_allocation()` publish initial and repeated
  task storages through immutable plan/task records.
- `shadowspill_pytorch_validate_spill_binding()` rejects stale imported CPU
  storage views. Device storage acquisition is validated by its admitted task
  or object-acquisition handle before the adapter installs the returned
  address.
- `shadowspill_pytorch_write_spill_object()` and
  `shadowspill_pytorch_read_spill_object()` move persistent state.
- `shadowspill_pytorch_transfer_acquired_object_to_caller()` and
  `shadowspill_pytorch_release_caller_allocation()` manage public outputs.
- `shadowspill_pytorch_object_snapshot()` returns diagnostic state.
- `shadowspill_pytorch_object_handle_acquire()` and
  `shadowspill_pytorch_object_handle_release()` retain and release opaque
  runtime-global object ownership across callable boundaries.
- `shadowspill_pytorch_object_release_generation()` releases one exact
  completed residency generation without destroying its logical object or
  plan binding.

## Execution boundaries

The adapter exposes the neutral plan owner without introducing frontend object
semantics:

- `shadowspill_pytorch_plan_create()`
- `shadowspill_pytorch_plan_close()`
- `shadowspill_pytorch_plan_destroy()`
- `shadowspill_pytorch_plan_bind_object()`
- `shadowspill_pytorch_plan_admit_task()`
- `shadowspill_pytorch_plan_publish_initial_allocation()`
- `shadowspill_pytorch_task_publish_allocation()` and
  `shadowspill_pytorch_validate_task_replacement_binding()`
- `shadowspill_pytorch_plan_admit_action_batch()` and
  `shadowspill_pytorch_submit_action_batch_handle()`
- `shadowspill_pytorch_plan_admit_object_acquisition()` and
  `shadowspill_pytorch_acquire_objects_handle()`
- `shadowspill_pytorch_transfer_acquired_object_to_caller()`
- `shadowspill_pytorch_plan_admit_fixed_layout()`
- `shadowspill_pytorch_plan_seal_fixed_layout()`
- `shadowspill_pytorch_plan_clear_tasks()`

Task calls mirror the neutral runtime:

- `shadowspill_pytorch_before_task_handle()` and
  `shadowspill_pytorch_after_task_handle()` are the production task boundary.
  The before boundary exposes the task-owned borrowed binding array instead
  of copying bindings into caller storage. The storage operators consume that
  view in place and return no per-task generation container to Python. Both
  boundaries derive task identity and the semantic profiler label from the
  admitted handle; no parallel task ID or mutable label table exists.

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
- `shadowspill_pytorch_task_labels_configure()` installs semantic execution
  labels.
- `shadowspill_pytorch_debug_task_timing_enable()`,
  `shadowspill_pytorch_debug_task_timing_read()`, and
  `shadowspill_pytorch_debug_task_timing_disable()` manage task events.
- `shadowspill_pytorch_trace_prepare()`, `shadowspill_pytorch_trace_begin()`,
  `shadowspill_pytorch_trace_end()`, and `shadowspill_pytorch_trace_read()`
  manage structured runtime tracing.
- `shadowspill_pytorch_allocation_telemetry_start()`,
  `shadowspill_pytorch_allocation_telemetry_stop()`, and
  `shadowspill_pytorch_allocation_telemetry_read()` manage allocation profiling.

## Failure and lifecycle

`shadowspill_pytorch_allocator_failure()` and
`shadowspill_pytorch_allocator_statistics()` return structured state.
`shadowspill_pytorch_allocator_wait_idle()` is an explicit lifecycle barrier;
`shadowspill_pytorch_recover_no_progress()` performs the documented recovery
operation.

The adapter registers process-exit cleanup that destroys the neutral runtime,
joins its worker, and closes all pool backends. Python callables must still be
closed explicitly so errors and ownership violations are reported at the
correct boundary.
