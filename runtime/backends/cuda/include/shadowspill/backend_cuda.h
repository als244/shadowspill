#ifndef SHADOWSPILL_BACKEND_CUDA_H
#define SHADOWSPILL_BACKEND_CUDA_H

#include <stdint.h>

#include <shadowspill/backend.h>

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
} ShadowSpillCudaBackendCapabilities;

typedef struct ShadowSpillCudaBackendStatistics {
    uint64_t device_allocations;
    uint64_t device_frees;
    uint64_t pinned_host_allocations;
    uint64_t pinned_host_frees;
    uint64_t streams_created;
    uint64_t streams_destroyed;
    uint64_t events_created;
    uint64_t events_destroyed;
    uint64_t copies_to_device;
    uint64_t copies_to_host;
    uint64_t bytes_to_device;
    uint64_t bytes_to_host;
    uint64_t event_queries;
    uint64_t stream_waits;
    uint64_t stream_synchronizations;
    uint64_t context_activations;
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

/* Last CUDA Driver API result observed by this backend; zero means success. */
SHADOWSPILL_BACKEND_CUDA_API uint32_t shadowspill_cuda_backend_last_error(
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
