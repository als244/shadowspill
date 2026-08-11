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

#define SHADOWSPILL_PYTORCH_ADAPTER_ABI_VERSION 2U

typedef struct ShadowSpillPytorchAdapterConfig {
    uint32_t abi_version;
    int32_t device_ordinal;
    uint64_t device_budget_bytes;
    uint64_t provider_headroom_bytes;
    uint64_t host_arena_bytes;
    uint64_t progress_poll_nanoseconds;
} ShadowSpillPytorchAdapterConfig;

typedef struct ShadowSpillPytorchPhysicalAdmission {
    uint32_t abi_version;
    int32_t device_ordinal;
    uint64_t device_budget_bytes;
    uint64_t context_bytes;
    uint64_t provider_headroom_bytes;
    uint64_t slab_bytes;
    uint64_t bootstrap_process_bytes;
    uint64_t device_used_bytes;
    uint64_t device_total_bytes;
    uint64_t host_arena_bytes;
} ShadowSpillPytorchPhysicalAdmission;

typedef struct ShadowSpillPytorchAdapterCapabilities {
    uint32_t abi_version;
    uint32_t runtime_abi_version;
    uint32_t backend_abi_version;
    uint8_t slab_memory_strategy;
    uint8_t record_stream_callback;
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
    uint64_t required_provider_headroom_bytes
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
