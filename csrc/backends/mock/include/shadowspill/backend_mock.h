#ifndef SHADOWSPILL_BACKEND_MOCK_H
#define SHADOWSPILL_BACKEND_MOCK_H

#include <stdint.h>

#include <shadowspill/runtime.h>

#if defined(_WIN32)
#if defined(SHADOWSPILL_BACKEND_MOCK_BUILDING)
#define SHADOWSPILL_BACKEND_MOCK_API __declspec(dllexport)
#else
#define SHADOWSPILL_BACKEND_MOCK_API __declspec(dllimport)
#endif
#else
#define SHADOWSPILL_BACKEND_MOCK_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ShadowSpillMockBackend ShadowSpillMockBackend;

#define SHADOWSPILL_MOCK_BACKEND_ABI_VERSION 1U

/*
 * Deterministic qualification backend. Delay values extend opaque stream
 * timelines; copies use ordinary host memory so payload assertions remain
 * possible. The backend is thread-safe and must outlive every runtime borrowing
 * its vtable.
 */

typedef struct ShadowSpillMockBackendConfig {
    uint32_t abi_version;
    uint64_t fetch_delay_nanoseconds;
    uint64_t evict_delay_nanoseconds;
    uint64_t event_delay_nanoseconds;
} ShadowSpillMockBackendConfig;

typedef struct ShadowSpillMockBackendStatistics {
    uint64_t operation_count;
    uint64_t execution_allocations;
    uint64_t spill_allocations;
    uint64_t fetch_copies;
    uint64_t evict_copies;
    uint64_t stream_waits;
    uint64_t stream_synchronizations;
    uint64_t events_created;
    uint64_t events_destroyed;
    uint64_t event_queries;
} ShadowSpillMockBackendStatistics;

/* Convenient owned storage for one two-pool mock runtime configuration. */
typedef struct ShadowSpillMockRuntimeTopology {
    ShadowSpillMemoryPoolDescription pools[2];
    ShadowSpillTransferRouteDescription routes[2];
    ShadowSpillRuntimeConfig runtime;
} ShadowSpillMockRuntimeTopology;

SHADOWSPILL_BACKEND_MOCK_API int shadowspill_mock_backend_create(
    const ShadowSpillMockBackendConfig *config,
    ShadowSpillMockBackend **backend
);

SHADOWSPILL_BACKEND_MOCK_API void shadowspill_mock_backend_destroy(
    ShadowSpillMockBackend *backend
);

SHADOWSPILL_BACKEND_MOCK_API ShadowSpillMemoryPoolBackend
shadowspill_mock_execution_pool_backend(
    ShadowSpillMockBackend *backend
);

SHADOWSPILL_BACKEND_MOCK_API ShadowSpillMemoryPoolBackend
shadowspill_mock_spill_pool_backend(ShadowSpillMockBackend *backend);

SHADOWSPILL_BACKEND_MOCK_API ShadowSpillTransferRoute
shadowspill_mock_fetch_route(
    ShadowSpillMockBackend *backend,
    uint32_t source_pool_id,
    uint32_t destination_pool_id
);

SHADOWSPILL_BACKEND_MOCK_API ShadowSpillTransferRoute
shadowspill_mock_evict_route(
    ShadowSpillMockBackend *backend,
    uint32_t source_pool_id,
    uint32_t destination_pool_id
);

SHADOWSPILL_BACKEND_MOCK_API ShadowSpillSynchronizationBackend
shadowspill_mock_synchronization_backend(ShadowSpillMockBackend *backend);

SHADOWSPILL_BACKEND_MOCK_API void shadowspill_mock_runtime_topology(
    ShadowSpillMockBackend *backend,
    uint64_t execution_pool_bytes,
    uint64_t spill_pool_bytes,
    uint64_t minimum_alignment,
    uint64_t worker_poll_nanoseconds,
    ShadowSpillMockRuntimeTopology *topology
);

SHADOWSPILL_BACKEND_MOCK_API int shadowspill_mock_create_compute_stream(
    ShadowSpillMockBackend *backend,
    ShadowSpillBackendStream *stream
);

SHADOWSPILL_BACKEND_MOCK_API int shadowspill_mock_destroy_compute_stream(
    ShadowSpillMockBackend *backend,
    ShadowSpillBackendStream stream
);

SHADOWSPILL_BACKEND_MOCK_API int shadowspill_mock_enqueue_compute(
    ShadowSpillMockBackend *backend,
    ShadowSpillBackendStream stream,
    uint64_t duration_nanoseconds
);

/* Fail exactly the absolute backend operation number; zero disables failure. */
SHADOWSPILL_BACKEND_MOCK_API void shadowspill_mock_fail_operation(
    ShadowSpillMockBackend *backend,
    uint64_t operation_number
);

/* Atomically selects the next backend operation for failure. */
SHADOWSPILL_BACKEND_MOCK_API void shadowspill_mock_fail_next_operation(
    ShadowSpillMockBackend *backend
);

SHADOWSPILL_BACKEND_MOCK_API void shadowspill_mock_backend_statistics(
    ShadowSpillMockBackend *backend,
    ShadowSpillMockBackendStatistics *statistics
);

#ifdef __cplusplus
}
#endif

#endif
