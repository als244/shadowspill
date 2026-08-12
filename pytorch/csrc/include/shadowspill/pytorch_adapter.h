#ifndef SHADOWSPILL_PYTORCH_ADAPTER_H
#define SHADOWSPILL_PYTORCH_ADAPTER_H

#include <stddef.h>
#include <stdint.h>

#include <shadowspill/backend_cuda.h>
#include <shadowspill/runtime.h>

#if defined(_WIN32)
#define SHADOWSPILL_PYTORCH_API __declspec(dllexport)
#else
#define SHADOWSPILL_PYTORCH_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SHADOWSPILL_PYTORCH_ADAPTER_ABI_VERSION 27U

typedef struct ShadowSpillPytorchAdapterConfig {
    uint32_t abi_version;
    int32_t device_ordinal;
    uint64_t device_budget_bytes;
    uint64_t provider_headroom_bytes;
    uint64_t spill_pool_bytes;
    uint64_t worker_poll_nanoseconds;
} ShadowSpillPytorchAdapterConfig;

typedef struct ShadowSpillPytorchPhysicalAdmission {
    uint32_t abi_version;
    int32_t device_ordinal;
    uint64_t device_budget_bytes;
    uint64_t context_bytes;
    uint64_t provider_headroom_bytes;
    uint64_t execution_pool_bytes;
    uint64_t bootstrap_process_bytes;
    uint64_t device_used_bytes;
    uint64_t device_total_bytes;
    uint64_t spill_pool_bytes;
} ShadowSpillPytorchPhysicalAdmission;

typedef struct ShadowSpillPytorchAdapterCapabilities {
    uint32_t abi_version;
    uint32_t runtime_abi_version;
    uint32_t backend_abi_version;
    uint8_t slab_memory_strategy;
    uint8_t record_stream_callback;
    uint8_t storage_rebinding;
    uint8_t debug_task_host_timing;
    uint8_t runtime_trace;
} ShadowSpillPytorchAdapterCapabilities;

/*
 * Optional task-boundary host timestamps. The four host fields use
 * CLOCK_MONOTONIC. The six legacy compute-stream fields are reserved and zero;
 * the PyTorch frontend records non-invasive preallocated CUDA events for those
 * boundaries instead of executing host callbacks on the compute stream.
 */
typedef struct ShadowSpillPytorchTaskHostTiming {
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
} ShadowSpillPytorchTaskHostTiming;

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
SHADOWSPILL_PYTORCH_API ShadowSpillRuntimeStatus
shadowspill_pytorch_allocator_bootstrap(
    const ShadowSpillPytorchAdapterConfig *config
);

SHADOWSPILL_PYTORCH_API ShadowSpillRuntimeStatus
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
SHADOWSPILL_PYTORCH_API ShadowSpillRuntimeStatus
shadowspill_pytorch_profiler_annotations_set(uint8_t enabled);

/* Copies immutable bootstrap admission and physical-accounting evidence. */
SHADOWSPILL_PYTORCH_API ShadowSpillRuntimeStatus
shadowspill_pytorch_physical_admission(
    ShadowSpillPytorchPhysicalAdmission *admission
);

/* Queries current per-process physical use for seal/diagnostic boundaries. */
SHADOWSPILL_PYTORCH_API ShadowSpillRuntimeStatus
shadowspill_pytorch_physical_memory(ShadowSpillCudaPhysicalMemory *memory);

/*
 * Confirms the profiled provider reserve fits the bootstrap reservation and
 * seals the physical ledger. This call does not resize or weaken the budget.
 */
SHADOWSPILL_PYTORCH_API ShadowSpillRuntimeStatus
shadowspill_pytorch_seal_physical_budget(
    uint64_t required_provider_headroom_bytes,
    uint64_t event_pool_reserve
);

/* Cold-path immutable execution admission and hot predecoded boundaries. */
SHADOWSPILL_PYTORCH_API ShadowSpillRuntimeStatus
shadowspill_pytorch_admit_execution(
    const ShadowSpillExecutionDescription *description
);

/* Clear the completed plan's immutable execution records. */
SHADOWSPILL_PYTORCH_API ShadowSpillRuntimeStatus
shadowspill_pytorch_clear_execution_plan(void);

SHADOWSPILL_PYTORCH_API ShadowSpillRuntimeStatus
shadowspill_pytorch_resolve_execution(
    uint64_t task_id,
    uintptr_t *execution_handle
);

SHADOWSPILL_PYTORCH_API ShadowSpillRuntimeStatus
shadowspill_pytorch_before_execution(
    uint64_t task_id,
    uintptr_t compute_stream_address,
    ShadowSpillObjectBinding *bindings,
    uint32_t binding_capacity
);

SHADOWSPILL_PYTORCH_API ShadowSpillRuntimeStatus
shadowspill_pytorch_after_execution(
    uint64_t task_id,
    uintptr_t compute_stream_address
);

SHADOWSPILL_PYTORCH_API ShadowSpillRuntimeStatus
shadowspill_pytorch_before_execution_handle(
    uintptr_t execution_handle,
    uint64_t task_id,
    uintptr_t compute_stream_address,
    ShadowSpillObjectBinding *bindings,
    uint32_t binding_capacity
);

SHADOWSPILL_PYTORCH_API ShadowSpillRuntimeStatus
shadowspill_pytorch_after_execution_handle(
    uintptr_t execution_handle,
    uint64_t task_id,
    uintptr_t compute_stream_address
);

/* Reconciles current process bytes against the sealed or provisional cap. */
SHADOWSPILL_PYTORCH_API ShadowSpillRuntimeStatus
shadowspill_pytorch_check_physical_budget(void);

SHADOWSPILL_PYTORCH_API ShadowSpillRuntimeStatus
shadowspill_pytorch_allocator_statistics(
    ShadowSpillPytorchAdapterStatistics *statistics
);

SHADOWSPILL_PYTORCH_API ShadowSpillRuntimeStatus
shadowspill_pytorch_allocator_failure(
    ShadowSpillPytorchAdapterFailure *failure
);

/* Explicitly synchronizing qualification/checkpoint helper. */
SHADOWSPILL_PYTORCH_API ShadowSpillRuntimeStatus
shadowspill_pytorch_allocator_wait_idle(void);

/* Public frontend bridge for runtime transfer-route calibration and snapshots. */
SHADOWSPILL_PYTORCH_API ShadowSpillRuntimeStatus
shadowspill_pytorch_calibrate_transfer_capabilities(
    const ShadowSpillTransferCalibrationConfig *config,
    const ShadowSpillTransferRouteKey *routes,
    uint32_t route_count
);

SHADOWSPILL_PYTORCH_API ShadowSpillRuntimeStatus
shadowspill_pytorch_transfer_profiles(
    ShadowSpillTransferProfile *profiles,
    uint32_t capacity,
    uint32_t *count,
    uint64_t *generation
);

/*
 * Planning-only pinned-host growth before physical sealing. Existing payloads
 * and object offsets are preserved. The caller must admit the brief overlap of
 * old and replacement arenas within its public host budget.
 */
SHADOWSPILL_PYTORCH_API ShadowSpillRuntimeStatus
shadowspill_pytorch_resize_spill_pool(uint64_t spill_pool_bytes);

/* Bounded task-scoped allocation telemetry used by structural profiling. */
SHADOWSPILL_PYTORCH_API ShadowSpillRuntimeStatus
shadowspill_pytorch_allocation_telemetry_start(uint64_t capacity);

SHADOWSPILL_PYTORCH_API ShadowSpillRuntimeStatus
shadowspill_pytorch_allocation_telemetry_stop(void);

SHADOWSPILL_PYTORCH_API ShadowSpillRuntimeStatus
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
SHADOWSPILL_PYTORCH_API ShadowSpillRuntimeStatus
shadowspill_pytorch_trace_prepare(const ShadowSpillTraceConfig *config);

SHADOWSPILL_PYTORCH_API ShadowSpillRuntimeStatus
shadowspill_pytorch_trace_begin(uint64_t step_id);

SHADOWSPILL_PYTORCH_API ShadowSpillRuntimeStatus
shadowspill_pytorch_trace_end(void);

SHADOWSPILL_PYTORCH_API ShadowSpillRuntimeStatus
shadowspill_pytorch_trace_read(
    ShadowSpillTraceSummary *summary,
    ShadowSpillTraceEvent *events,
    uint64_t event_capacity,
    ShadowSpillAllocationEvent *allocation_events,
    uint64_t allocation_event_capacity
);

/* Read-only exact pointer lookup used to classify profiled task outputs. */
SHADOWSPILL_PYTORCH_API ShadowSpillRuntimeStatus
shadowspill_pytorch_allocation_for_pointer(
    uint64_t address,
    ShadowSpillAllocation *allocation
);

/* Register and populate one initially host-resident alias-group object. */
SHADOWSPILL_PYTORCH_API ShadowSpillRuntimeStatus
shadowspill_pytorch_register_host_object(
    uint64_t object_id,
    uint64_t size_bytes,
    uint8_t retain_spill_copy,
    uint64_t source_address
);

/* Register a logical object without allocating host or device residency. */
SHADOWSPILL_PYTORCH_API ShadowSpillRuntimeStatus
shadowspill_pytorch_register_placeholder_object(
    uint64_t object_id,
    uint64_t size_bytes,
    uint8_t retain_spill_copy
);

/* Replace one existing SPILL_ONLY object's current pinned payload. */
SHADOWSPILL_PYTORCH_API ShadowSpillRuntimeStatus
shadowspill_pytorch_write_spill_object(
    uint64_t object_id,
    uint64_t size_bytes,
    uint64_t source_address
);

/* Copy one current SPILL_ONLY payload into borrowed caller memory. */
SHADOWSPILL_PYTORCH_API ShadowSpillRuntimeStatus
shadowspill_pytorch_read_spill_object(
    uint64_t object_id,
    uint64_t size_bytes,
    uint64_t destination_address
);

/* Remove one terminal object and reclaim any retained host range. */
SHADOWSPILL_PYTORCH_API ShadowSpillRuntimeStatus
shadowspill_pytorch_unregister_object(uint64_t object_id);

/* Bind an already registered object to one ordinary framework allocation. */
SHADOWSPILL_PYTORCH_API ShadowSpillRuntimeStatus
shadowspill_pytorch_bind_registered_allocation(
    uint64_t object_id,
    uint64_t address,
    uint64_t size_bytes,
    ShadowSpillObjectBinding *binding
);

/* Replace a registered object's lease with a fresh functional output. */
SHADOWSPILL_PYTORCH_API ShadowSpillRuntimeStatus
shadowspill_pytorch_replace_registered_allocation(
    uint64_t object_id,
    uint64_t address,
    uint64_t size_bytes,
    ShadowSpillObjectBinding *binding
);

/* Transfer one final execution allocation from plan to caller ownership. */
SHADOWSPILL_PYTORCH_API ShadowSpillRuntimeStatus
shadowspill_pytorch_transfer_output_to_caller(
    uint64_t object_id,
    ShadowSpillAllocation *allocation
);

/* Release one caller-owned allocation from the private owning DataPtr. */
SHADOWSPILL_PYTORCH_API ShadowSpillRuntimeStatus
shadowspill_pytorch_release_caller_allocation(uint64_t allocation_id);

/*
 * Converts an existing ordinary PyTorch allocation into one plan-owned object
 * and returns its current address generation. This is used only after graph
 * output allocation and before the owning DataPtr is replaced.
 */
SHADOWSPILL_PYTORCH_API ShadowSpillRuntimeStatus
shadowspill_pytorch_promote_allocation(
    uint64_t object_id,
    uint64_t address,
    uint64_t size_bytes,
    ShadowSpillObjectBinding *binding
);

/* Private storage-operator guard over object identity/address/generation. */
SHADOWSPILL_PYTORCH_API ShadowSpillRuntimeStatus
shadowspill_pytorch_validate_object_binding(
    uint64_t object_id,
    uint64_t address,
    uint64_t generation
);

/*
 * Private frontend bridge for exact runtime task boundaries. CUDA stream
 * addresses are borrowed for the duration of each call and wrapped without
 * transferring ownership.
 */
SHADOWSPILL_PYTORCH_API ShadowSpillRuntimeStatus
shadowspill_pytorch_before_task(
    uint64_t task_id,
    uintptr_t compute_stream_address,
    const uint64_t *input_object_ids,
    uint32_t input_count,
    ShadowSpillObjectBinding *bindings,
    uint32_t binding_capacity
);

SHADOWSPILL_PYTORCH_API ShadowSpillRuntimeStatus
shadowspill_pytorch_after_task(
    uint64_t task_id,
    uintptr_t compute_stream_address,
    const ShadowSpillObjectUpdate *updates,
    uint32_t update_count,
    const ShadowSpillRuntimeAction *actions,
    uint32_t action_count
);

SHADOWSPILL_PYTORCH_API ShadowSpillRuntimeStatus
shadowspill_pytorch_object_snapshot(
    uint64_t object_id,
    ShadowSpillObjectSnapshot *snapshot
);

/*
 * Copies one optional human-readable label per dense canonical task ID.
 * Configure labels only while no task is executing. Task NVTX ranges use the
 * label (normally execution_XXXXXX plus its semantic stage name) and retain
 * the canonical task ID only as fallback correlation metadata.
 */
SHADOWSPILL_PYTORCH_API ShadowSpillRuntimeStatus
shadowspill_pytorch_task_labels_configure(
    const char *const *task_labels,
    uint32_t task_label_count
);

/* Allocate/reset pre-sized callback records before a measured invocation. */
SHADOWSPILL_PYTORCH_API ShadowSpillRuntimeStatus
shadowspill_pytorch_debug_task_timing_enable(uint32_t task_capacity);

/* Read completed records after synchronizing the measured compute stream. */
SHADOWSPILL_PYTORCH_API ShadowSpillRuntimeStatus
shadowspill_pytorch_debug_task_timing_read(
    ShadowSpillPytorchTaskHostTiming *records,
    uint32_t record_capacity,
    uint32_t *record_count
);

/* Disable and release records; rejected while a callback remains in flight. */
SHADOWSPILL_PYTORCH_API ShadowSpillRuntimeStatus
shadowspill_pytorch_debug_task_timing_disable(void);

/* Closes a task NVTX range when frontend execution raises before after_task. */
SHADOWSPILL_PYTORCH_API void shadowspill_pytorch_abort_task_range(void);

/*
 * Exact callback ABI consumed by torch.cuda.memory.CUDAPluggableAllocator.
 * These functions never throw across the C boundary. Failures are latched and
 * malloc returns NULL so PyTorch raises through its ordinary OOM path.
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
