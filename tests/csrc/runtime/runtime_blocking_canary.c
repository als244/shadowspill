#define _POSIX_C_SOURCE 200809L

#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>

#include <shadowspill/backend_mock.h>
#include <shadowspill/runtime.h>

typedef struct AllocationRequest {
    ShadowSpillRuntime *runtime;
    ShadowSpillBackendStream stream;
    ShadowSpillRuntimeStatus status;
    ShadowSpillAllocation allocation;
} AllocationRequest;

static void *allocate_from_thread(void *pointer) {
    AllocationRequest *request = pointer;
    request->status = shadowspill_memory_pool_allocate(request->runtime, 0U,
        128U,
        1U,
        request->stream,
        &request->allocation
    );
    return NULL;
}

int main(void) {
    ShadowSpillMockBackend *mock = NULL;
    const ShadowSpillMockBackendConfig mock_config = {
        .abi_version = SHADOWSPILL_MOCK_BACKEND_ABI_VERSION,
        .event_delay_nanoseconds = 2000000U,
    };
    if (shadowspill_mock_backend_create(&mock_config, &mock) != 0) {
        return EXIT_FAILURE;
    }
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillMockRuntimeTopology topology;
    shadowspill_mock_runtime_topology(
        mock, 128U, 1U, 1U, 10000U, &topology
    );
    if (shadowspill_runtime_create(&topology.runtime, &runtime) !=
        SHADOWSPILL_RUNTIME_OK) {
        return EXIT_FAILURE;
    }
    ShadowSpillBackendStream first_stream = {{0U, 0U}};
    ShadowSpillBackendStream second_stream = {{0U, 0U}};
    ShadowSpillAllocation first = {0};
    if (shadowspill_mock_create_compute_stream(mock, &first_stream) != 0 ||
        shadowspill_mock_create_compute_stream(mock, &second_stream) != 0 ||
        shadowspill_memory_pool_allocate(runtime, 0U, 128U, 1U, first_stream, &first) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_memory_pool_record_stream(runtime, 0U, first.allocation_id, second_stream
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_memory_pool_free(runtime, 0U, first.allocation_id, first_stream) !=
            SHADOWSPILL_RUNTIME_OK) {
        return EXIT_FAILURE;
    }
    AllocationRequest request = {
        .runtime = runtime,
        .stream = first_stream,
        .status = SHADOWSPILL_RUNTIME_INVALID_STATE,
    };
    pthread_t thread;
    if (pthread_create(&thread, NULL, allocate_from_thread, &request) != 0 ||
        pthread_join(thread, NULL) != 0 ||
        request.status != SHADOWSPILL_RUNTIME_OK ||
        request.allocation.pointer == NULL ||
        shadowspill_memory_pool_free(runtime, 0U, request.allocation.allocation_id, first_stream
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_wait_idle(runtime) != SHADOWSPILL_RUNTIME_OK) {
        return EXIT_FAILURE;
    }
    ShadowSpillRuntimeStatistics statistics = {0};
    if (shadowspill_runtime_statistics(runtime, &statistics) !=
            SHADOWSPILL_RUNTIME_OK ||
        statistics.free_bytes != 128U ||
        statistics.largest_free_range_bytes != 128U) {
        return EXIT_FAILURE;
    }
    if (shadowspill_runtime_close(runtime) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_mock_destroy_compute_stream(mock, first_stream) != 0 ||
        shadowspill_mock_destroy_compute_stream(mock, second_stream) != 0) {
        return EXIT_FAILURE;
    }
    shadowspill_runtime_destroy(runtime);
    shadowspill_mock_backend_destroy(mock);
    return EXIT_SUCCESS;
}
