#ifndef SHADOWSPILL_BACKEND_CUDA_H
#define SHADOWSPILL_BACKEND_CUDA_H

#include <stdint.h>

#include <shadowspill/backend.h>
#include <shadowspill/profiler.h>

#if defined(_WIN32)
#define SHADOWSPILL_BACKEND_CUDA_API __declspec(dllexport)
#else
#define SHADOWSPILL_BACKEND_CUDA_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SHADOWSPILL_CUDA_BACKEND_ABI_VERSION 1U

typedef struct ShadowSpillCudaBackend ShadowSpillCudaBackend;

/* Provider state used through the framework-neutral profiler vtable. */
SHADOWSPILL_BACKEND_CUDA_API void shadowspill_cuda_backend_profiler_enable(
    ShadowSpillCudaBackend *backend, uint8_t enabled
);
SHADOWSPILL_BACKEND_CUDA_API uint8_t
shadowspill_cuda_backend_profiler_is_enabled(
    const ShadowSpillCudaBackend *backend
);

typedef struct ShadowSpillCudaBackendConfig {
    uint32_t abi_version;
    int32_t device_ordinal;
} ShadowSpillCudaBackendConfig;

typedef struct ShadowSpillCudaBackendCapabilities {
    uint32_t abi_version;
    uint32_t driver_version;
    int32_t device_ordinal;
    int32_t compute_major;
    int32_t compute_minor;
    int32_t asynchronous_engine_count;
    uint64_t total_device_bytes;
    uint64_t recommended_minimum_alignment;
    uint8_t concurrent_kernels;
    uint8_t unified_addressing;
    uint8_t process_memory_accounting;
} ShadowSpillCudaBackendCapabilities;

typedef struct ShadowSpillCudaPhysicalMemory {
    uint32_t abi_version;
    uint64_t process_bytes;
    uint64_t device_used_bytes;
    uint64_t device_total_bytes;
} ShadowSpillCudaPhysicalMemory;

typedef struct ShadowSpillCudaBackendStatistics {
    uint64_t device_allocations;
    uint64_t device_frees;
    uint64_t pinned_host_allocations;
    uint64_t pinned_host_frees;
    uint64_t streams_created;
    uint64_t streams_destroyed;
    uint64_t events_created;
    uint64_t events_destroyed;
    uint64_t fetch_copies;
    uint64_t evict_copies;
    uint64_t bytes_fetched;
    uint64_t bytes_evicted;
    uint64_t event_queries;
    uint64_t stream_waits;
    uint64_t stream_synchronizations;
    uint64_t context_activations;
    uint64_t event_pool_capacity;
    uint64_t event_pool_in_use;
    uint64_t event_pool_peak_in_use;
    uint64_t event_pool_driver_creates;
    uint64_t event_pool_growth_rejections;
    uint8_t event_pool_sealed;
} ShadowSpillCudaBackendStatistics;

/*
 * Retains the selected device's primary context. An already-current different
 * context is rejected. The returned backend owns only its primary-context
 * reference and must outlive every runtime borrowing its vtable.
 */
SHADOWSPILL_BACKEND_CUDA_API int shadowspill_cuda_backend_create(
    const ShadowSpillCudaBackendConfig *config,
    ShadowSpillCudaBackend **backend
);

/* Releases the primary-context reference. Destroy runtimes first. */
SHADOWSPILL_BACKEND_CUDA_API void shadowspill_cuda_backend_destroy(
    ShadowSpillCudaBackend *backend
);

/* Returns a copied neutral vtable borrowing backend as its context. */
SHADOWSPILL_BACKEND_CUDA_API ShadowSpillBackend
shadowspill_cuda_backend_vtable(ShadowSpillCudaBackend *backend);

/* Returns the NVIDIA profiler implementation as a neutral profiler vtable. */
SHADOWSPILL_BACKEND_CUDA_API ShadowSpillProfiler
shadowspill_cuda_backend_profiler(ShadowSpillCudaBackend *backend);

/* Copies immutable device/backend capabilities into caller-owned storage. */
SHADOWSPILL_BACKEND_CUDA_API int shadowspill_cuda_backend_capabilities(
    ShadowSpillCudaBackend *backend,
    ShadowSpillCudaBackendCapabilities *capabilities
);

/* Copies a lock-consistent operation ledger into caller-owned storage. */
SHADOWSPILL_BACKEND_CUDA_API void shadowspill_cuda_backend_statistics(
    ShadowSpillCudaBackend *backend,
    ShadowSpillCudaBackendStatistics *statistics
);

/*
 * Ensures the requested free-event reserve, then forbids later cuEventCreate
 * calls. Returned event leases continue to recycle through the same pool.
 */
SHADOWSPILL_BACKEND_CUDA_API int shadowspill_cuda_backend_seal_event_pool(
    ShadowSpillCudaBackend *backend,
    uint64_t minimum_free_events
);

/*
 * Reports NVML physical memory for the current process and whole device.
 * Process bytes include the CUDA context, the slab, and provider allocations;
 * they do not depend on allocator-visible logical occupancy.
 */
SHADOWSPILL_BACKEND_CUDA_API int shadowspill_cuda_physical_memory(
    ShadowSpillCudaBackend *backend,
    ShadowSpillCudaPhysicalMemory *memory
);

/* Last CUDA Driver API result observed by this backend; zero means success. */
SHADOWSPILL_BACKEND_CUDA_API uint32_t shadowspill_cuda_backend_last_error(
    ShadowSpillCudaBackend *backend
);

/* Last NVML result observed by physical accounting; zero means success. */
SHADOWSPILL_BACKEND_CUDA_API uint32_t shadowspill_cuda_backend_last_nvml_error(
    ShadowSpillCudaBackend *backend
);

SHADOWSPILL_BACKEND_CUDA_API const char *shadowspill_cuda_error_name(
    uint32_t error_code
);

SHADOWSPILL_BACKEND_CUDA_API const char *shadowspill_cuda_error_string(
    uint32_t error_code
);

/* Wraps an existing CUDA stream address without transferring ownership. */
SHADOWSPILL_BACKEND_CUDA_API ShadowSpillBackendStream
shadowspill_cuda_wrap_stream(uintptr_t stream_address);

#ifdef __cplusplus
}
#endif

#endif
