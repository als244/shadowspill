#ifndef SHADOWSPILL_BACKEND_MOCK_H
#define SHADOWSPILL_BACKEND_MOCK_H

#include <stdint.h>

#include <shadowspill/backend.h>

#if defined(_WIN32)
#define SHADOWSPILL_BACKEND_MOCK_API __declspec(dllexport)
#else
#define SHADOWSPILL_BACKEND_MOCK_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ShadowSpillMockBackend ShadowSpillMockBackend;

/*
 * Deterministic qualification backend. Delay values extend opaque stream
 * timelines; copies use ordinary host memory so payload assertions remain
 * possible. The backend is thread-safe and must outlive every runtime borrowing
 * its vtable.
 */

typedef struct ShadowSpillMockBackendConfig {
    uint32_t abi_version;
    uint64_t h2d_delay_nanoseconds;
    uint64_t d2h_delay_nanoseconds;
    uint64_t event_delay_nanoseconds;
} ShadowSpillMockBackendConfig;

typedef struct ShadowSpillMockBackendStatistics {
    uint64_t operation_count;
    uint64_t device_allocations;
    uint64_t host_allocations;
    uint64_t copies_to_device;
    uint64_t copies_to_host;
    uint64_t stream_waits;
    uint64_t stream_synchronizations;
    uint64_t events_created;
    uint64_t events_destroyed;
    uint64_t event_queries;
} ShadowSpillMockBackendStatistics;

SHADOWSPILL_BACKEND_MOCK_API int shadowspill_mock_backend_create(
    const ShadowSpillMockBackendConfig *config,
    ShadowSpillMockBackend **backend
);

SHADOWSPILL_BACKEND_MOCK_API void shadowspill_mock_backend_destroy(
    ShadowSpillMockBackend *backend
);

SHADOWSPILL_BACKEND_MOCK_API ShadowSpillBackend shadowspill_mock_backend_vtable(
    ShadowSpillMockBackend *backend
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
