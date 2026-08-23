#ifndef SHADOWSPILL_PYTORCH_ADAPTER_H
#define SHADOWSPILL_PYTORCH_ADAPTER_H

#include <stddef.h>
#include <stdint.h>

#include <shadowspill/backend_cuda.h>
#include <shadowspill/runtime.h>

#if defined(_WIN32)
#if defined(SHADOWSPILL_PYTORCH_BUILDING)
#define SHADOWSPILL_PYTORCH_API __declspec(dllexport)
#else
#define SHADOWSPILL_PYTORCH_API __declspec(dllimport)
#endif
#else
#define SHADOWSPILL_PYTORCH_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SHADOWSPILL_PYTORCH_ADAPTER_ABI_VERSION 56U

typedef enum ShadowSpillPytorchPoolBackendKind {
    SHADOWSPILL_PYTORCH_POOL_DEVICE = 0,
    SHADOWSPILL_PYTORCH_POOL_PINNED_HOST = 1,
} ShadowSpillPytorchPoolBackendKind;

typedef struct ShadowSpillPytorchPoolConfig {
    uint32_t pool_id;
    uint8_t backend_kind;
    uint64_t capacity_bytes;
} ShadowSpillPytorchPoolConfig;

typedef struct ShadowSpillPytorchRouteConfig {
    uint32_t route_id;
    uint32_t source_pool_id;
    uint32_t destination_pool_id;
    const char *name;
} ShadowSpillPytorchRouteConfig;

typedef struct ShadowSpillPytorchAdapterConfig {
    uint32_t abi_version;
    int32_t device_ordinal;
    uint64_t device_budget_bytes;
    uint64_t provider_headroom_bytes;
    uint32_t allocator_pool_id;
    const ShadowSpillPytorchPoolConfig *pools;
    uint32_t pool_count;
    const ShadowSpillPytorchRouteConfig *routes;
    uint32_t route_count;
    uint64_t worker_poll_nanoseconds;
} ShadowSpillPytorchAdapterConfig;

typedef struct ShadowSpillPytorchPhysicalAdmission {
    uint32_t abi_version;
    int32_t device_ordinal;
    uint64_t device_budget_bytes;
    uint64_t baseline_bytes;
    uint64_t provider_headroom_bytes;
    uint32_t allocator_pool_id;
    uint32_t pool_count;
    uint64_t allocator_pool_bytes;
    uint64_t bootstrap_process_bytes;
    uint64_t device_used_bytes;
    uint64_t device_total_bytes;
} ShadowSpillPytorchPhysicalAdmission;

typedef struct ShadowSpillPytorchAdapterCapabilities {
    uint32_t abi_version;
    uint32_t runtime_abi_version;
    uint32_t backend_abi_version;
    uint8_t slab_memory_strategy;
    uint8_t record_stream_callback;
    uint8_t storage_rebinding;
    uint8_t debug_task_dispatch_timing;
    uint8_t runtime_trace;
} ShadowSpillPytorchAdapterCapabilities;

/*
 * Optional task-boundary dispatch timestamps. The four fields use
 * CLOCK_MONOTONIC. The six compute-stream compatibility fields are reserved
 * and zero; the PyTorch frontend records non-invasive preallocated CUDA events
 * for those boundaries instead of executing dispatch callbacks on the compute
 * stream.
 */
typedef struct ShadowSpillPytorchTaskDispatchTiming {
    uint64_t task_id;
    uint64_t before_readiness_waits_timestamp_ns;
    uint64_t before_task_compute_timestamp_ns;
    uint64_t after_task_compute_timestamp_ns;
    uint64_t before_readiness_waits_sequence;
    uint64_t before_task_compute_sequence;
    uint64_t after_task_compute_sequence;
    uint64_t before_task_enter_timestamp_ns;
    uint64_t before_task_exit_timestamp_ns;
    uint64_t after_task_enter_timestamp_ns;
    uint64_t after_task_exit_timestamp_ns;
} ShadowSpillPytorchTaskDispatchTiming;

typedef struct ShadowSpillPytorchAdapterStatistics {
    uint64_t allocation_callbacks;
    uint64_t zero_size_allocation_callbacks;
    uint64_t free_callbacks;
    uint64_t record_stream_callbacks;
    uint64_t pointer_lookup_failures;
    uint64_t callback_failures;
    uint64_t physical_checks;
    uint64_t peak_process_physical_bytes;
    uint64_t observed_external_high_water_bytes;
    uint64_t physical_budget_sealed;
    ShadowSpillRuntimeStatistics runtime;
    ShadowSpillCudaBackendStatistics cuda;
} ShadowSpillPytorchAdapterStatistics;

typedef struct ShadowSpillPytorchAdapterFailure {
    uint32_t status;
    int32_t device_ordinal;
    uint64_t address;
    uint64_t requested_bytes;
    ShadowSpillRuntimeFailure runtime;
} ShadowSpillPytorchAdapterFailure;

/*
 * Creates and permanently binds one process-global CUDA slab runtime. Call
 * before installing the callbacks and before PyTorch initializes CUDA. The
 * connector owns the runtime/backend for process lifetime because PyTorch's
 * selected allocator cannot safely be replaced after initialization.
 */
SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_allocator_bootstrap(
    const ShadowSpillPytorchAdapterConfig *config
);

/*
 * Idempotently reject new allocator work, stop and join the runtime worker,
 * close every route and pool, and release the concrete backend. PyTorch's
 * allocator shim remains installed; subsequent nonzero allocations fail with
 * SHADOWSPILL_STATUS_CLOSED instead of accessing released backend state.
 */
SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_allocator_close(void);

SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_adapter_capabilities(
    ShadowSpillPytorchAdapterCapabilities *capabilities
);

/*
 * Backend-neutral profiling ranges used by the Python task orchestrator.
 * These are no-ops when the configured runtime profiler has no provider.
 */
SHADOWSPILL_PYTORCH_API ShadowSpillProfilerRange
shadowspill_pytorch_profile_range_begin(const char *name);

SHADOWSPILL_PYTORCH_API void shadowspill_pytorch_profile_range_end(
    ShadowSpillProfilerRange range
);

/* Enable or disable provider annotations independently of runtime tracing. */
SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_profiler_annotations_set(uint8_t enabled);

/* Copies immutable bootstrap admission and physical-accounting evidence. */
SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_physical_admission(
    ShadowSpillPytorchPhysicalAdmission *admission
);

/* Queries current per-process physical use for seal/diagnostic boundaries. */
SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_physical_memory(ShadowSpillCudaPhysicalMemory *memory);

/*
 * Confirms the profiled provider reserve fits the bootstrap reservation and
 * seals the physical ledger. This call does not resize or weaken the budget.
 */
SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_seal_physical_budget(
    uint64_t required_provider_headroom_bytes,
    uint64_t runtime_record_reserve
);

/* Cold-path immutable execution admission and hot predecoded boundaries. */
SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_plan_create(
    uint32_t execution_pool_id,
    uint32_t spill_pool_id,
    uint32_t fetch_route_id,
    uint32_t evict_route_id,
    uintptr_t *plan_handle
);

SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_plan_close(uintptr_t plan_handle);

SHADOWSPILL_PYTORCH_API void shadowspill_pytorch_plan_destroy(
    uintptr_t plan_handle
);

SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_object_handle_acquire(
    uint64_t runtime_object_id,
    uintptr_t *object_handle
);

SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_object_handle_release(
    uintptr_t object_handle
);

SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_object_release_generation(
    uintptr_t object_handle,
    uint64_t expected_generation
);

SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_plan_bind_object(
    uintptr_t plan_handle,
    uint64_t plan_object_id,
    uintptr_t object_handle,
    uint8_t consistency
);

SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_plan_admit_task(
    uintptr_t plan_handle,
    const ShadowSpillTaskDescription *description,
    uintptr_t *task_handle
);

SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_plan_publish_initial_allocation(
    uintptr_t plan_handle,
    uint64_t plan_object_id,
    uint64_t address,
    ShadowSpillObjectBinding *binding
);

SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_plan_admit_fixed_layout(
    uintptr_t plan_handle,
    const ShadowSpillFixedLayoutDescription *description
);

SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_plan_seal_fixed_layout(uintptr_t plan_handle);

SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_plan_clear_tasks(uintptr_t plan_handle);
SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_plan_wait_idle(uintptr_t plan_handle);

SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_plan_admit_object_acquisition(
    uintptr_t plan_handle,
    const uint64_t *object_ids,
    uint32_t object_count,
    uintptr_t *acquisition_handle
);

SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_plan_admit_action_batch(
    uintptr_t plan_handle,
    uint64_t batch_id,
    const ShadowSpillRuntimeAction *actions,
    uint32_t action_count,
    uintptr_t *action_batch_handle
);

SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_submit_action_batch_handle(
    uintptr_t action_batch_handle,
    uintptr_t trigger_stream_address
);

SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_acquire_objects_handle(
    uintptr_t acquisition_handle,
    uintptr_t consumer_stream_address,
    ShadowSpillObjectBinding *bindings,
    uint32_t binding_capacity
);

SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_transfer_acquired_object_to_caller(
    uintptr_t acquisition_handle,
    uint32_t object_ordinal,
    uintptr_t consumer_stream,
    uint64_t expected_address,
    uint64_t expected_generation,
    uint64_t expected_allocation_id,
    ShadowSpillAllocation *allocation
);

SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_before_task_handle(
    uintptr_t task_handle,
    uintptr_t compute_stream_address,
    const ShadowSpillObjectBinding **bindings,
    uint32_t *binding_count
);

SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_after_task_handle(
    uintptr_t task_handle,
    uintptr_t compute_stream_address
);

/* Publish one task-owned output without a runtime object-ID lookup. */
SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_task_publish_allocation(
    uintptr_t task_handle,
    uint32_t publication_ordinal,
    uint64_t address,
    ShadowSpillObjectBinding *binding
);

/* Validate one replacement's retired and successor addresses by direct record. */
SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_validate_task_replacement_binding(
    uintptr_t task_handle,
    uint32_t publication_ordinal,
    uint64_t retired_address,
    uint64_t successor_address
);

/* Reconciles current process bytes against the sealed or provisional cap. */
SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_check_physical_budget(void);

SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_allocator_statistics(
    ShadowSpillPytorchAdapterStatistics *statistics
);

SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_allocator_failure(
    ShadowSpillPytorchAdapterFailure *failure
);

/*
 * Fault-teardown helper. The frontend must first synchronize the execution
 * device. Only a latched NO_PROGRESS allocator failure can be cleared.
 */
SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_recover_no_progress(void);

/* Explicitly synchronizing qualification/checkpoint helper. */
SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_allocator_wait_idle(void);

/* Public frontend bridge for runtime transfer-route calibration and snapshots. */
SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_calibrate_transfer_capabilities(
    const ShadowSpillTransferCalibrationConfig *config,
    const ShadowSpillTransferRouteKey *routes,
    uint32_t route_count
);

SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_transfer_profiles(
    ShadowSpillTransferProfile *profiles,
    uint32_t capacity,
    uint32_t *count,
    uint64_t *generation
);

/* Bounded task-scoped allocation telemetry used by structural profiling. */
SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_allocation_telemetry_start(uint64_t capacity);

SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_allocation_telemetry_stop(void);

SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_allocation_telemetry_read(
    ShadowSpillAllocationEvent *events,
    uint64_t capacity,
    uint64_t *count
);

/*
 * Optional bounded runtime tracing. Preparation allocates reusable CPU-side
 * buffers but does not enable tracing. Begin/end are per-step operations;
 * callers establish the desired completion boundary before ending a trace.
 */
SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_trace_prepare(const ShadowSpillTraceConfig *config);

SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_trace_begin(uint64_t step_id);

SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_trace_end(void);

SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_trace_read(
    ShadowSpillTraceSummary *summary,
    ShadowSpillTraceEvent *events,
    uint64_t event_capacity,
    ShadowSpillAllocationEvent *allocation_events,
    uint64_t allocation_event_capacity
);

/* Read-only exact pointer lookup used to classify profiled task outputs. */
SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_allocation_for_pointer(
    uint64_t address,
    ShadowSpillAllocation *allocation
);

/* Register and populate one object in an explicitly selected pool. */
SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_register_object(
    uint32_t pool_id,
    uint64_t object_id,
    uint64_t size_bytes,
    uint8_t retain_spill_copy,
    uint64_t source_address
);

/* Register a logical object without allocating spill or device residency. */
SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_register_placeholder_object(
    uint64_t object_id,
    uint64_t size_bytes,
    uint8_t retain_spill_copy
);

/* Replace one existing object's current payload in an explicit pool. */
SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_write_object(
    uint32_t pool_id,
    uint64_t object_id,
    uint64_t size_bytes,
    uint64_t source_address
);

/* Copy one current pool payload into borrowed caller memory. */
SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_read_object(
    uint32_t pool_id,
    uint64_t object_id,
    uint64_t size_bytes,
    uint64_t destination_address
);

/* Remove one terminal object and reclaim any retained spill range. */
SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_unregister_object(uint64_t object_id);

/* Retarget one idle spill-resident object without moving its lease. */
SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_rekey_object(
    uint64_t object_id,
    uint64_t replacement_object_id
);

/* Release one caller-owned allocation from the private owning DataPtr. */
SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_release_caller_allocation(
    uint64_t allocation_id,
    uintptr_t stream
);

/* Validate one CPU-addressable pool lease before rebinding CPU storage. */
SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_validate_object_binding(
    uint32_t pool_id,
    uint64_t object_id,
    uint64_t address,
    uint64_t size_bytes
);

/* Attribute isolated profiling allocations without opening a fake task. */
SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_allocation_scope_begin(uint64_t scope_id);

SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_allocation_scope_end(
    uint64_t scope_id,
    uintptr_t compute_stream_address
);

SHADOWSPILL_PYTORCH_API void
shadowspill_pytorch_allocation_scope_abort(void);

SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_object_snapshot(
    uint64_t object_id,
    ShadowSpillObjectSnapshot *snapshot
);

SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_object_location_snapshot(
    uint64_t object_id,
    uint32_t pool_id,
    ShadowSpillObjectLocationSnapshot *snapshot
);

/* Allocate/reset pre-sized callback records before a measured invocation. */
SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_debug_task_timing_enable(uint32_t task_capacity);

/* Read completed records after synchronizing the measured compute stream. */
SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_debug_task_timing_read(
    ShadowSpillPytorchTaskDispatchTiming *records,
    uint32_t record_capacity,
    uint32_t *record_count
);

/* Disable and release records; rejected while a callback remains in flight. */
SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_debug_task_timing_disable(void);

/* Closes a task NVTX range when frontend execution raises before after_task. */
SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_abort_task_handle(
    uintptr_t task_handle
);

/*
 * Exact callback ABI consumed by torch.cuda.memory.CUDAPluggableAllocator.
 * The C runtime never throws and latches complete first-failure diagnostics.
 * The private C++ PyTorch adapter raises a structured, task-attributed
 * exception for a failed nonzero request because CUDAPluggableAllocator does
 * not validate a null pointer before constructing a DataPtr. OOM statuses use
 * PyTorch's typed OutOfMemoryError; contract failures use RuntimeError.
 */
SHADOWSPILL_PYTORCH_API void *shadowspill_pytorch_cuda_malloc(
    ptrdiff_t bytes,
    int32_t device_ordinal,
    void *stream
);

SHADOWSPILL_PYTORCH_API void shadowspill_pytorch_cuda_free(
    void *address,
    size_t bytes,
    int32_t device_ordinal,
    void *stream
);

SHADOWSPILL_PYTORCH_API void shadowspill_pytorch_cuda_record_stream(
    void *address,
    void *stream
);

#ifdef __cplusplus
}
#endif

#endif
