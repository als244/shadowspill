#define _POSIX_C_SOURCE 200809L

#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#include <shadowspill/backend_mock.h>
#include <shadowspill/runtime.h>

typedef struct WaitingAllocation {
    ShadowSpillRuntime *runtime;
    ShadowSpillBackendStream stream;
    ShadowSpillRuntimeStatus status;
} WaitingAllocation;

static void *allocate_while_pending(void *pointer) {
    WaitingAllocation *waiting = pointer;
    ShadowSpillAllocation allocation = {0};
    waiting->status = shadowspill_allocate(
        waiting->runtime, 128U, 1U, waiting->stream, &allocation
    );
    return NULL;
}

static int impossible_oom(void) {
    ShadowSpillMockBackend *mock = NULL;
    const ShadowSpillMockBackendConfig mock_config = {
        .abi_version = SHADOWSPILL_BACKEND_ABI_VERSION,
    };
    if (shadowspill_mock_backend_create(&mock_config, &mock) != 0) {
        return -1;
    }
    ShadowSpillRuntime *runtime = NULL;
    const ShadowSpillRuntimeConfig config = {
        .abi_version = SHADOWSPILL_RUNTIME_ABI_VERSION,
        .execution_pool_bytes = 128U,
        .spill_pool_bytes = 1U,
        .minimum_alignment = 1U,
        .backend = shadowspill_mock_backend_vtable(mock),
    };
    ShadowSpillBackendStream stream = {{0U, 0U}};
    ShadowSpillAllocation full = {0};
    ShadowSpillAllocation impossible = {0};
    int result = 0;
    if (shadowspill_runtime_create(&config, &runtime) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_mock_create_compute_stream(mock, &stream) != 0 ||
        shadowspill_allocate(runtime, 128U, 1U, stream, &full) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_allocate(runtime, 1U, 1U, stream, &impossible) !=
            SHADOWSPILL_RUNTIME_NO_PROGRESS) {
        result = -1;
    }
    ShadowSpillRuntimeFailure failure = {0};
    if (shadowspill_runtime_failure(runtime, &failure) !=
            SHADOWSPILL_RUNTIME_OK ||
        failure.status != SHADOWSPILL_RUNTIME_NO_PROGRESS ||
        failure.requested_bytes != 1U || failure.free_bytes != 0U ||
        failure.largest_free_range_bytes != 0U) {
        result = -1;
    }
    shadowspill_runtime_destroy(runtime);
    (void)shadowspill_mock_destroy_compute_stream(mock, stream);
    shadowspill_mock_backend_destroy(mock);
    return result;
}

static int worker_failure(void) {
    ShadowSpillMockBackend *mock = NULL;
    const ShadowSpillMockBackendConfig mock_config = {
        .abi_version = SHADOWSPILL_BACKEND_ABI_VERSION,
        .event_delay_nanoseconds = 1000000000U,
    };
    if (shadowspill_mock_backend_create(&mock_config, &mock) != 0) {
        return -1;
    }
    ShadowSpillRuntime *runtime = NULL;
    const ShadowSpillRuntimeConfig config = {
        .abi_version = SHADOWSPILL_RUNTIME_ABI_VERSION,
        .execution_pool_bytes = 128U,
        .spill_pool_bytes = 1U,
        .minimum_alignment = 1U,
        .worker_poll_nanoseconds = 10000U,
        .backend = shadowspill_mock_backend_vtable(mock),
    };
    ShadowSpillBackendStream stream = {{0U, 0U}};
    ShadowSpillBackendStream other_stream = {{0U, 0U}};
    ShadowSpillAllocation allocation = {0};
    if (shadowspill_runtime_create(&config, &runtime) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_mock_create_compute_stream(mock, &stream) != 0 ||
        shadowspill_mock_create_compute_stream(mock, &other_stream) != 0 ||
        shadowspill_allocate(runtime, 128U, 1U, stream, &allocation) !=
            SHADOWSPILL_RUNTIME_OK) {
        return -1;
    }
    if (shadowspill_record_stream(
            runtime, allocation.allocation_id, other_stream
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_free(runtime, allocation.allocation_id, stream) !=
            SHADOWSPILL_RUNTIME_OK) {
        return -1;
    }
    WaitingAllocation waiting = {
        .runtime = runtime,
        .stream = stream,
        .status = SHADOWSPILL_RUNTIME_INVALID_STATE,
    };
    pthread_t waiter;
    if (pthread_create(&waiter, NULL, allocate_while_pending, &waiting) != 0) {
        return -1;
    }
    ShadowSpillRuntimeStatistics statistics = {0};
    for (uint32_t attempt = 0U; attempt < 1000U; ++attempt) {
        if (shadowspill_runtime_statistics(runtime, &statistics) !=
            SHADOWSPILL_RUNTIME_OK) {
            return -1;
        }
        if (statistics.blocked_allocators == 1U) {
            break;
        }
        const struct timespec delay = {.tv_nsec = 1000000L};
        (void)nanosleep(&delay, NULL);
    }
    if (statistics.blocked_allocators != 1U) {
        return -1;
    }
    shadowspill_mock_fail_next_operation(mock);
    if (pthread_join(waiter, NULL) != 0 ||
        waiting.status != SHADOWSPILL_RUNTIME_BACKEND_FAILURE) {
        return -1;
    }
    ShadowSpillRuntimeStatus wait_status = shadowspill_runtime_wait_idle(runtime);
    ShadowSpillRuntimeFailure failure = {0};
    int result = 0;
    if (wait_status != SHADOWSPILL_RUNTIME_BACKEND_FAILURE ||
        shadowspill_runtime_failure(runtime, &failure) !=
            SHADOWSPILL_RUNTIME_OK ||
        failure.status != SHADOWSPILL_RUNTIME_BACKEND_FAILURE ||
        failure.allocation_id != allocation.allocation_id) {
        result = -1;
    }
    shadowspill_runtime_destroy(runtime);
    (void)shadowspill_mock_destroy_compute_stream(mock, stream);
    (void)shadowspill_mock_destroy_compute_stream(mock, other_stream);
    shadowspill_mock_backend_destroy(mock);
    return result;
}

static int fragmented_oom(void) {
    ShadowSpillMockBackend *mock = NULL;
    const ShadowSpillMockBackendConfig mock_config = {
        .abi_version = SHADOWSPILL_BACKEND_ABI_VERSION,
    };
    if (shadowspill_mock_backend_create(&mock_config, &mock) != 0) {
        return -1;
    }
    ShadowSpillRuntime *runtime = NULL;
    const ShadowSpillRuntimeConfig config = {
        .abi_version = SHADOWSPILL_RUNTIME_ABI_VERSION,
        .execution_pool_bytes = 128U,
        .spill_pool_bytes = 1U,
        .minimum_alignment = 1U,
        .worker_poll_nanoseconds = 10000U,
        .backend = shadowspill_mock_backend_vtable(mock),
    };
    ShadowSpillBackendStream stream = {{0U, 0U}};
    ShadowSpillAllocation blocks[4] = {{0}};
    int result = 0;
    if (shadowspill_runtime_create(&config, &runtime) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_mock_create_compute_stream(mock, &stream) != 0) {
        result = -1;
        goto done;
    }
    for (uint32_t index = 0U; index < 4U; ++index) {
        if (shadowspill_allocate(runtime, 32U, 1U, stream, &blocks[index]) !=
            SHADOWSPILL_RUNTIME_OK) {
            result = -1;
            goto done;
        }
    }
    if (shadowspill_free(runtime, blocks[0].allocation_id, stream) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_free(runtime, blocks[2].allocation_id, stream) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_wait_idle(runtime) != SHADOWSPILL_RUNTIME_OK) {
        result = -1;
        goto done;
    }
    ShadowSpillRuntimeStatistics statistics = {0};
    ShadowSpillAllocation impossible = {0};
    if (shadowspill_runtime_statistics(runtime, &statistics) !=
            SHADOWSPILL_RUNTIME_OK ||
        statistics.free_bytes != 64U ||
        statistics.largest_free_range_bytes != 32U ||
        statistics.external_fragmentation_bytes != 32U ||
        shadowspill_allocate(runtime, 48U, 1U, stream, &impossible) !=
            SHADOWSPILL_RUNTIME_NO_PROGRESS) {
        result = -1;
    }
    ShadowSpillRuntimeFailure failure = {0};
    if (shadowspill_runtime_failure(runtime, &failure) !=
            SHADOWSPILL_RUNTIME_OK ||
        failure.status != SHADOWSPILL_RUNTIME_NO_PROGRESS ||
        failure.requested_bytes != 48U || failure.free_bytes != 64U ||
        failure.largest_free_range_bytes != 32U) {
        result = -1;
    }

done:
    shadowspill_runtime_destroy(runtime);
    if (stream.words[0] != 0U) {
        (void)shadowspill_mock_destroy_compute_stream(mock, stream);
    }
    shadowspill_mock_backend_destroy(mock);
    return result;
}

int main(void) {
    return impossible_oom() == 0 && fragmented_oom() == 0 &&
            worker_failure() == 0
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}
