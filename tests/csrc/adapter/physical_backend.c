/* Mock storage with explicit physical-memory snapshots for startup tests. */
#include <shadowspill/backend_mock.h>

#include <stdint.h>

static uint64_t process_bytes;

SHADOWSPILL_BACKEND_MOCK_API void shadowspill_test_process_bytes(uint64_t value) {
    process_bytes = value;
}

static int physical(void *state, ShadowSpillBackendPhysicalMemory *out) {
    if (state == NULL || out == NULL) {
        return -1;
    }
    *out = (ShadowSpillBackendPhysicalMemory){
        .process_bytes = process_bytes,
        .device_used_bytes = process_bytes,
        .device_total_bytes = UINT64_C(1) << 40U,
    };
    return 0;
}

SHADOWSPILL_BACKEND_MOCK_API int shadowspill_backend_create(
    const ShadowSpillBackendConfig *config, ShadowSpillBackend *backend
) {
    if (config == NULL || backend == NULL ||
        config->abi_version != SHADOWSPILL_BACKEND_ABI_VERSION) {
        return -1;
    }
    const ShadowSpillMockBackendConfig mock = {0};
    if (shadowspill_mock_backend_create(&mock, backend) != 0) {
        return -1;
    }
    backend->physical_memory = physical;
    return 0;
}

/* The dependency's shadowspill_backend_destroy releases this unchanged mock
   state. dlsym on this library searches that dependency, just as the adapter
   searches an ordinary backend's exported symbols. */
