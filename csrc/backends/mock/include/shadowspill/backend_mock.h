#ifndef SHADOWSPILL_BACKEND_MOCK_H
#define SHADOWSPILL_BACKEND_MOCK_H

#include <stdint.h>

#include <shadowspill/backend.h>
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

/*
 * The accelerator-free backend: host memory stands in for the device, and
 * streams, events, and copies are simulated with delays on a monotonic clock.
 * It implements the backend contract like any provider, and adds the test
 * hooks below for driving streams and injecting failures.
 */

typedef struct ShadowSpillMockBackend ShadowSpillMockBackend;

typedef struct ShadowSpillMockBackendConfig {
    uint64_t fetch_delay_nanoseconds;
    uint64_t evict_delay_nanoseconds;
    uint64_t event_delay_nanoseconds;
} ShadowSpillMockBackendConfig;

/* Every operation the mock performed, in order, for failure injection. */
typedef struct ShadowSpillMockBackendStatistics {
    uint64_t operation_count;
} ShadowSpillMockBackendStatistics;

typedef struct ShadowSpillMockRuntimeTopology {
    ShadowSpillBackend backend;
    ShadowSpillMemoryPoolDescription pools[2];
    ShadowSpillTransferRouteDescription routes[2];
    ShadowSpillRuntimeConfig runtime;
} ShadowSpillMockRuntimeTopology;

/* A mock with configured delays; shadowspill_backend_create() makes one with
   no delays. Either way shadowspill_backend_destroy() releases it. */
SHADOWSPILL_BACKEND_MOCK_API int shadowspill_mock_backend_create(
    const ShadowSpillMockBackendConfig *config,
    ShadowSpillBackend *backend
);

/* One device pool, one pinned-host pool, and the fetch and evict routes
   between them, ready for shadowspill_runtime_create(). */
SHADOWSPILL_BACKEND_MOCK_API void shadowspill_mock_runtime_topology(
    const ShadowSpillBackend *backend,
    uint64_t execution_pool_bytes,
    uint64_t spill_pool_bytes,
    uint64_t minimum_alignment,
    uint64_t worker_poll_nanoseconds,
    ShadowSpillMockRuntimeTopology *topology
);

/* Test hooks. Compute work on a stream is a delay; failures are counted in
   operations and fire once. */
SHADOWSPILL_BACKEND_MOCK_API int shadowspill_mock_enqueue_compute(
    const ShadowSpillBackend *backend,
    ShadowSpillBackendStream stream,
    uint64_t duration_nanoseconds
);
SHADOWSPILL_BACKEND_MOCK_API void shadowspill_mock_fail_operation(
    const ShadowSpillBackend *backend,
    uint64_t operation_number
);
SHADOWSPILL_BACKEND_MOCK_API void shadowspill_mock_fail_next_operation(
    const ShadowSpillBackend *backend
);
SHADOWSPILL_BACKEND_MOCK_API void shadowspill_mock_backend_statistics(
    const ShadowSpillBackend *backend,
    ShadowSpillMockBackendStatistics *statistics
);

#ifdef __cplusplus
}
#endif

#endif
