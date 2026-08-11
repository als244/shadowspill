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

#define SHADOWSPILL_PYTORCH_ADAPTER_ABI_VERSION 1U

typedef struct ShadowSpillPytorchAdapterConfig {
    uint32_t abi_version;
    int32_t device_ordinal;
    uint64_t device_slab_bytes;
    uint64_t host_arena_bytes;
    uint64_t progress_poll_nanoseconds;
} ShadowSpillPytorchAdapterConfig;

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
