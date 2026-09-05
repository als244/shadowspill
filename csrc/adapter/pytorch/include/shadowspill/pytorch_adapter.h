#ifndef SHADOWSPILL_PYTORCH_ADAPTER_H
#define SHADOWSPILL_PYTORCH_ADAPTER_H

#include <stddef.h>
#include <stdint.h>

#include <shadowspill/backend.h>
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

#define SHADOWSPILL_PYTORCH_ADAPTER_ABI_VERSION 2U

/* Ids the frontend synthesises for work that is not a planned task: the
   profiling allocation scopes, and the pre-task placement batch. The
   failure report decodes them, so both sides read one definition. */
#define SHADOWSPILL_PYTORCH_PROFILING_SCOPE_BASE (UINT64_C(1) << 62)
#define SHADOWSPILL_PYTORCH_INITIAL_ACTIONS_TASK_ID (UINT64_C(1) << 60)

/* ------------------------------------------------------------------------
 * Vocabulary and descriptions
 *
 * What a caller fills in -- the bootstrap config -- and what the adapter
 * hands back: the physical ledger, the capabilities, statistics, and
 * the failure record. Nothing here allocates or holds state.
 */

typedef struct ShadowSpillPytorchPoolConfig {
    uint32_t pool_id;
    /* A ShadowSpillPoolKind value. */
    uint8_t kind;
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
    /* Path of the backend shared object to load: the library exporting
       shadowspill_backend_create() and shadowspill_backend_destroy(). */
    const char *backend_library;
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

/* The three contracts this build was compiled against, and the one thing
   that varies between builds: whether libtorch was found, so the storage
   operators exist. */
typedef struct ShadowSpillPytorchAdapterCapabilities {
    uint32_t abi_version;
    uint32_t runtime_abi_version;
    uint32_t backend_abi_version;
    uint8_t storage_rebinding;
} ShadowSpillPytorchAdapterCapabilities;

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
    ShadowSpillBackendStatistics backend;
} ShadowSpillPytorchAdapterStatistics;

typedef struct ShadowSpillPytorchAdapterFailure {
    uint32_t status;
    int32_t device_ordinal;
    uint64_t address;
    uint64_t requested_bytes;
    ShadowSpillRuntimeFailure runtime;
} ShadowSpillPytorchAdapterFailure;

/* ------------------------------------------------------------------------
 * Bootstrap, physical admission and close
 *
 * One runtime per process, bound before PyTorch touches the device;
 * the physical-memory ledger it is admitted against; its close.
 */

/*
 * Creates and permanently binds one process-global slab runtime on the backend
 * the config names. Call before installing the callbacks and before PyTorch
 * initializes the accelerator. The
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
 * Publishes the process-global runtime this library bound, so the frontend can
 * make the neutral runtime calls that need nothing else directly rather than
 * through an entry point here that would only fetch this pointer and forward.
 * Returns SHADOWSPILL_STATUS_CLOSED, and a null handle, once the runtime is
 * released.
 */
SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_runtime_handle(uintptr_t *runtime_handle);

/* Copies immutable bootstrap admission and physical-accounting evidence. */
SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_physical_admission(
    ShadowSpillPytorchPhysicalAdmission *admission
);

/* Queries current per-process physical use for seal/diagnostic boundaries. */
SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_physical_memory(ShadowSpillBackendPhysicalMemory *memory);

/* Reconciles current process bytes against the sealed or provisional cap. */
SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_check_physical_budget(void);

/*
 * Confirms the profiled provider reserve fits the bootstrap reservation and
 * seals the physical ledger. This call does not resize or weaken the budget.
 */
SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_seal_physical_budget(
    uint64_t required_provider_headroom_bytes,
    uint64_t runtime_record_reserve
);

/* ------------------------------------------------------------------------
 * The allocator callbacks
 *
 * The three calls PyTorch's pluggable allocator makes, and the query
 * that says which allocation a pointer belongs to.
 */

/*
 * Exact callback ABI consumed by torch.cuda.memory.CUDAPluggableAllocator.
 * The C runtime never throws and latches complete first-failure diagnostics.
 * The private C++ PyTorch adapter raises a structured, task-attributed
 * exception for a failed nonzero request because CUDAPluggableAllocator does
 * not validate a null pointer before constructing a DataPtr. OOM statuses use
 * PyTorch's typed OutOfMemoryError; contract failures use RuntimeError.
 */
SHADOWSPILL_PYTORCH_API void *shadowspill_pytorch_backend_malloc(
    ptrdiff_t bytes,
    int32_t device_ordinal,
    void *stream
);

SHADOWSPILL_PYTORCH_API void shadowspill_pytorch_backend_free(
    void *address,
    size_t bytes,
    int32_t device_ordinal,
    void *stream
);

SHADOWSPILL_PYTORCH_API void shadowspill_pytorch_backend_record_stream(
    void *address,
    void *stream
);

/* Read-only exact pointer lookup used to classify profiled task outputs. */
SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_allocation_for_pointer(
    uint64_t address,
    ShadowSpillAllocation *allocation
);

/* ------------------------------------------------------------------------
 * Objects and storage
 *
 * Validating a CPU storage view against its lease, acquiring objects
 * for a stream, and handing one to the caller and taking it back.
 */

/* Validate one CPU-addressable pool lease before rebinding CPU storage. */
SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_validate_object_binding(
    uint32_t pool_id,
    uint64_t object_id,
    uint64_t address,
    uint64_t size_bytes
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

/* Release one caller-owned allocation from the private owning DataPtr. */
SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_release_caller_allocation(
    uint64_t allocation_id,
    uintptr_t stream
);

/* ------------------------------------------------------------------------
 * Task boundaries and allocation scopes
 *
 * The pre-task action batch, the two calls every planned task runs
 * between, the abort, and the scopes profiling opens outside a task.
 */

SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_submit_action_batch_handle(
    uintptr_t action_batch_handle,
    uintptr_t trigger_stream_address
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

/* Closes a task profiler range when frontend execution raises before after_task. */
SHADOWSPILL_PYTORCH_API ShadowSpillStatus
shadowspill_pytorch_abort_task_handle(
    uintptr_t task_handle
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

/* ------------------------------------------------------------------------
 * Profiling
 *
 * Ranges on the backend's profiler, no-ops when it has none.
 */

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

/* ------------------------------------------------------------------------
 * Failure and recovery
 *
 * What was latched, the counters around it, and the one recovery.
 */

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

#ifdef __cplusplus
}
#endif

#endif
