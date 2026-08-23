#define _GNU_SOURCE

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <shadowspill/backend_mock.h>
#include <shadowspill/runtime.h>

#include "runtime_test.h"

typedef struct WaitingAllocation {
    ShadowSpillRuntime *runtime;
    ShadowSpillBackendStream stream;
    ShadowSpillStatus status;
} WaitingAllocation;

static void *allocate_while_pending(void *pointer) {
    WaitingAllocation *waiting = pointer;
    ShadowSpillAllocation allocation = {0};
    waiting->status = shadowspill_memory_pool_allocate(waiting->runtime, 0U, 128U, 1U, waiting->stream, &allocation
    );
    return NULL;
}

static int impossible_oom(void) {
    ShadowSpillMockBackend *mock = NULL;
    const ShadowSpillMockBackendConfig mock_config = {
        .abi_version = SHADOWSPILL_MOCK_BACKEND_ABI_VERSION,
    };
    if (shadowspill_mock_backend_create(&mock_config, &mock) != 0) {
        return -1;
    }
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillBackendStream stream = {{0U, 0U}};
    ShadowSpillAllocation full = {0};
    ShadowSpillAllocation impossible = {0};
    int result = 0;
    if (shadowspill_test_create_runtime(
            mock, 128U, 1U, 1U, 0U, &runtime
        ) != SHADOWSPILL_STATUS_OK ||
        shadowspill_mock_create_compute_stream(mock, &stream) != 0 ||
        shadowspill_memory_pool_allocate(runtime, 0U, 128U, 1U, stream, &full) !=
            SHADOWSPILL_STATUS_OK ||
        shadowspill_memory_pool_allocate(runtime, 0U, 1U, 1U, stream, &impossible) !=
            SHADOWSPILL_STATUS_NO_PROGRESS) {
        result = -1;
    }
    ShadowSpillRuntimeFailure failure = {0};
    if (shadowspill_runtime_failure(runtime, &failure) !=
            SHADOWSPILL_STATUS_OK ||
        failure.status != SHADOWSPILL_STATUS_NO_PROGRESS ||
        failure.requested_bytes != 1U || failure.free_bytes != 0U ||
        failure.largest_free_range_bytes != 0U) {
        result = -1;
    }
    if (shadowspill_runtime_recover_no_progress(runtime) !=
            SHADOWSPILL_STATUS_OK ||
        shadowspill_runtime_failure(runtime, &failure) !=
            SHADOWSPILL_STATUS_OK ||
        failure.status != SHADOWSPILL_STATUS_OK ||
        shadowspill_memory_pool_free(runtime, 0U, full.allocation_id, stream) !=
            SHADOWSPILL_STATUS_OK ||
        shadowspill_runtime_wait_idle(runtime) != SHADOWSPILL_STATUS_OK ||
        shadowspill_memory_pool_allocate(runtime, 0U, 1U, 1U, stream, &impossible) !=
            SHADOWSPILL_STATUS_OK) {
        result = -1;
    }
    shadowspill_test_destroy_runtime(runtime);
    (void)shadowspill_mock_destroy_compute_stream(mock, stream);
    shadowspill_mock_backend_destroy(mock);
    return result;
}

static int worker_failure(void) {
    ShadowSpillMockBackend *mock = NULL;
    const ShadowSpillMockBackendConfig mock_config = {
        .abi_version = SHADOWSPILL_MOCK_BACKEND_ABI_VERSION,
        .event_delay_nanoseconds = 1000000000U,
    };
    if (shadowspill_mock_backend_create(&mock_config, &mock) != 0) {
        return -1;
    }
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillBackendStream stream = {{0U, 0U}};
    ShadowSpillBackendStream other_stream = {{0U, 0U}};
    ShadowSpillAllocation allocation = {0};
    if (shadowspill_test_create_runtime(
            mock, 128U, 1U, 1U, 10000U, &runtime
        ) != SHADOWSPILL_STATUS_OK ||
        shadowspill_mock_create_compute_stream(mock, &stream) != 0 ||
        shadowspill_mock_create_compute_stream(mock, &other_stream) != 0 ||
        shadowspill_memory_pool_allocate(runtime, 0U, 128U, 1U, stream, &allocation) !=
            SHADOWSPILL_STATUS_OK) {
        return -1;
    }
    if (shadowspill_memory_pool_record_stream(runtime, 0U, allocation.allocation_id, other_stream
        ) != SHADOWSPILL_STATUS_OK ||
        shadowspill_memory_pool_free(runtime, 0U, allocation.allocation_id, stream) !=
            SHADOWSPILL_STATUS_OK) {
        return -1;
    }
    WaitingAllocation waiting = {
        .runtime = runtime,
        .stream = stream,
        .status = SHADOWSPILL_STATUS_INVALID_STATE,
    };
    pthread_t waiter;
    if (pthread_create(&waiter, NULL, allocate_while_pending, &waiting) != 0) {
        return -1;
    }
    ShadowSpillRuntimeStatistics statistics = {0};
    for (uint32_t attempt = 0U; attempt < 1000U; ++attempt) {
        if (shadowspill_runtime_statistics(runtime, &statistics) !=
            SHADOWSPILL_STATUS_OK) {
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
        waiting.status != SHADOWSPILL_STATUS_BACKEND_FAILURE) {
        return -1;
    }
    ShadowSpillStatus wait_status = shadowspill_runtime_wait_idle(runtime);
    ShadowSpillRuntimeFailure failure = {0};
    int result = 0;
    if (wait_status != SHADOWSPILL_STATUS_BACKEND_FAILURE ||
        shadowspill_runtime_failure(runtime, &failure) !=
            SHADOWSPILL_STATUS_OK ||
        failure.status != SHADOWSPILL_STATUS_BACKEND_FAILURE ||
        failure.allocation_id != allocation.allocation_id) {
        result = -1;
    }
    shadowspill_test_destroy_runtime(runtime);
    (void)shadowspill_mock_destroy_compute_stream(mock, stream);
    (void)shadowspill_mock_destroy_compute_stream(mock, other_stream);
    shadowspill_mock_backend_destroy(mock);
    return result;
}

static int worker_submission_failure_reaches_dispatcher(void) {
    ShadowSpillMockBackend *mock = NULL;
    const ShadowSpillMockBackendConfig mock_config = {
        .abi_version = SHADOWSPILL_MOCK_BACKEND_ABI_VERSION,
        .event_delay_nanoseconds = 1000000000U,
    };
    if (shadowspill_mock_backend_create(&mock_config, &mock) != 0) {
        return -1;
    }
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillBackendStream stream = {{0U, 0U}};
    const ShadowSpillObjectDescription object = {
        .object_id = 19U,
        .size_bytes = 32U,
        .initial_pool_id = 1U,
        .initially_resident = 1U,
    };
    const ShadowSpillRuntimeAction fetch = {
        .object_id = object.object_id,
        .kind = SHADOWSPILL_RUNTIME_PREFETCH,
    };
    const ShadowSpillTaskDescription trigger = {
        .task_id = 90U,
        .actions = &fetch,
        .action_count = 1U,
    };
    int result = 0;
    if (shadowspill_test_create_runtime(
            mock, 128U, 128U, 1U, 1000U, &runtime
        ) !=
            SHADOWSPILL_STATUS_OK ||
        shadowspill_mock_create_compute_stream(mock, &stream) != 0 ||
        shadowspill_register_object(runtime, &object) !=
            SHADOWSPILL_STATUS_OK ||
        shadowspill_test_admit_task(runtime, &trigger) !=
            SHADOWSPILL_STATUS_OK ||
        shadowspill_test_before_task(
            runtime, trigger.task_id, stream, NULL, 0U
        ) != SHADOWSPILL_STATUS_OK) {
        result = -1;
        goto done;
    }

    shadowspill_mock_fail_next_operation(mock);
    const ShadowSpillStatus submission_status =
        shadowspill_test_after_task(runtime, trigger.task_id, stream);
    ShadowSpillRuntimeFailure failure = {0};
    if (submission_status != SHADOWSPILL_STATUS_BACKEND_FAILURE ||
        shadowspill_runtime_failure(runtime, &failure) !=
            SHADOWSPILL_STATUS_OK ||
        failure.status != SHADOWSPILL_STATUS_BACKEND_FAILURE) {
        result = -1;
    }

done:
    shadowspill_test_destroy_runtime(runtime);
    if (stream.words[0] != 0U) {
        (void)shadowspill_mock_destroy_compute_stream(mock, stream);
    }
    shadowspill_mock_backend_destroy(mock);
    return result;
}

static int fragmented_oom(void) {
    ShadowSpillMockBackend *mock = NULL;
    const ShadowSpillMockBackendConfig mock_config = {
        .abi_version = SHADOWSPILL_MOCK_BACKEND_ABI_VERSION,
    };
    if (shadowspill_mock_backend_create(&mock_config, &mock) != 0) {
        return -1;
    }
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillBackendStream stream = {{0U, 0U}};
    ShadowSpillAllocation blocks[4] = {{0}};
    int result = 0;
    if (shadowspill_test_create_runtime(
            mock, 128U, 1U, 1U, 10000U, &runtime
        ) != SHADOWSPILL_STATUS_OK ||
        shadowspill_mock_create_compute_stream(mock, &stream) != 0) {
        result = -1;
        goto done;
    }
    for (uint32_t index = 0U; index < 4U; ++index) {
        if (shadowspill_memory_pool_allocate(runtime, 0U, 32U, 1U, stream, &blocks[index]) !=
            SHADOWSPILL_STATUS_OK) {
            result = -1;
            goto done;
        }
    }
    if (shadowspill_memory_pool_free(runtime, 0U, blocks[0].allocation_id, stream) !=
            SHADOWSPILL_STATUS_OK ||
        shadowspill_memory_pool_free(runtime, 0U, blocks[2].allocation_id, stream) !=
            SHADOWSPILL_STATUS_OK ||
        shadowspill_runtime_wait_idle(runtime) != SHADOWSPILL_STATUS_OK) {
        result = -1;
        goto done;
    }
    ShadowSpillRuntimeStatistics statistics = {0};
    ShadowSpillAllocation impossible = {0};
    if (shadowspill_runtime_statistics(runtime, &statistics) !=
            SHADOWSPILL_STATUS_OK ||
        statistics.free_bytes != 64U ||
        statistics.largest_free_range_bytes != 32U ||
        statistics.external_fragmentation_bytes != 32U ||
        shadowspill_memory_pool_allocate(runtime, 0U, 48U, 1U, stream, &impossible) !=
            SHADOWSPILL_STATUS_NO_PROGRESS) {
        result = -1;
    }
    ShadowSpillRuntimeFailure failure = {0};
    if (shadowspill_runtime_failure(runtime, &failure) !=
            SHADOWSPILL_STATUS_OK ||
        failure.status != SHADOWSPILL_STATUS_NO_PROGRESS ||
        failure.requested_bytes != 48U || failure.free_bytes != 64U ||
        failure.largest_free_range_bytes != 32U) {
        result = -1;
    }

done:
    shadowspill_test_destroy_runtime(runtime);
    if (stream.words[0] != 0U) {
        (void)shadowspill_mock_destroy_compute_stream(mock, stream);
    }
    shadowspill_mock_backend_destroy(mock);
    return result;
}

static int failed_task_retirement_recovery(void) {
    ShadowSpillMockBackend *mock = NULL;
    const ShadowSpillMockBackendConfig mock_config = {
        .abi_version = SHADOWSPILL_MOCK_BACKEND_ABI_VERSION,
    };
    if (shadowspill_mock_backend_create(&mock_config, &mock) != 0) {
        return -1;
    }
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillBackendStream stream = {{0U, 0U}};
    ShadowSpillAllocation live = {0};
    ShadowSpillAllocation impossible = {0};
    ShadowSpillAllocation recovered = {0};
    const ShadowSpillTaskDescription task = {.task_id = 41U};
    int result = 0;
    if (shadowspill_test_create_runtime(
            mock, 128U, 1U, 1U, 1000U, &runtime
        ) !=
            SHADOWSPILL_STATUS_OK ||
        shadowspill_mock_create_compute_stream(mock, &stream) != 0 ||
        shadowspill_test_admit_task(runtime, &task) !=
            SHADOWSPILL_STATUS_OK || shadowspill_test_before_task(
            runtime, task.task_id, stream, NULL, 0U
        ) != SHADOWSPILL_STATUS_OK ||
        shadowspill_memory_pool_allocate(runtime, 0U, 128U, 1U, stream, &live) !=
            SHADOWSPILL_STATUS_OK ||
        shadowspill_memory_pool_allocate(runtime, 0U, 1U, 1U, stream, &impossible) !=
            SHADOWSPILL_STATUS_NO_PROGRESS ||
        shadowspill_memory_pool_free(runtime, 0U, live.allocation_id, stream) !=
            SHADOWSPILL_STATUS_NO_PROGRESS) {
        result = -1;
        goto done;
    }
    ShadowSpillRuntimeStatistics statistics = {0};
    if (shadowspill_runtime_statistics(runtime, &statistics) !=
            SHADOWSPILL_STATUS_OK ||
        statistics.pending_retirements != 1U ||
        statistics.retirement_records_unfenced != 1U ||
        shadowspill_runtime_recover_no_progress(runtime) !=
            SHADOWSPILL_STATUS_INVALID_STATE ||
        shadowspill_test_after_task(
            runtime, task.task_id, stream
        ) != SHADOWSPILL_STATUS_NO_PROGRESS ||
        shadowspill_runtime_statistics(runtime, &statistics) !=
            SHADOWSPILL_STATUS_OK ||
        statistics.pending_retirements != 1U ||
        statistics.retirement_records_fenced != 1U ||
        statistics.retirement_records_unfenced != 0U ||
        shadowspill_runtime_recover_no_progress(runtime) !=
            SHADOWSPILL_STATUS_OK ||
        shadowspill_runtime_wait_idle(runtime) != SHADOWSPILL_STATUS_OK ||
        shadowspill_runtime_statistics(runtime, &statistics) !=
            SHADOWSPILL_STATUS_OK ||
        statistics.pending_retirements != 0U ||
        statistics.allocated_bytes != 0U ||
        shadowspill_memory_pool_allocate(runtime, 0U, 1U, 1U, stream, &recovered) !=
            SHADOWSPILL_STATUS_OK ||
        shadowspill_memory_pool_free(runtime, 0U, recovered.allocation_id, stream) !=
            SHADOWSPILL_STATUS_OK ||
        shadowspill_runtime_wait_idle(runtime) != SHADOWSPILL_STATUS_OK) {
        result = -1;
    }

done:
    shadowspill_test_destroy_runtime(runtime);
    if (stream.words[0] != 0U) {
        (void)shadowspill_mock_destroy_compute_stream(mock, stream);
    }
    shadowspill_mock_backend_destroy(mock);
    return result;
}

static int failed_prefetch_reports_trigger_reservation_oom(void) {
    ShadowSpillMockBackend *mock = NULL;
    const ShadowSpillMockBackendConfig mock_config = {
        .abi_version = SHADOWSPILL_MOCK_BACKEND_ABI_VERSION,
    };
    if (shadowspill_mock_backend_create(&mock_config, &mock) != 0) {
        return -1;
    }
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillBackendStream stream = {{0U, 0U}};
    ShadowSpillAllocation temporary = {0};
    const ShadowSpillObjectDescription object = {
        .object_id = 7U,
        .size_bytes = 32U,
        .initial_pool_id = 1U,
        .initially_resident = 1U,
    };
    const ShadowSpillRuntimeAction prefetch = {
        .object_id = object.object_id,
        .kind = SHADOWSPILL_RUNTIME_PREFETCH,
    };
    const ShadowSpillTaskDescription execution = {
        .task_id = 81U,
        .actions = &prefetch,
        .action_count = 1U,
        .maximum_requested_allocation_bytes = 128U,
        .maximum_charged_allocation_bytes = 128U,
        .live_requested_allocation_limit_bytes = 128U,
        .live_charged_allocation_limit_bytes = 128U,
    };
    int result = 0;
    if (shadowspill_test_create_runtime(
            mock, 128U, 128U, 1U, 1000U, &runtime
        ) !=
            SHADOWSPILL_STATUS_OK ||
        shadowspill_mock_create_compute_stream(mock, &stream) != 0 ||
        shadowspill_register_object(runtime, &object) !=
            SHADOWSPILL_STATUS_OK ||
        shadowspill_test_admit_task(runtime, &execution) !=
            SHADOWSPILL_STATUS_OK ||
        shadowspill_test_before_task(
            runtime, execution.task_id, stream, NULL, 0U
        ) != SHADOWSPILL_STATUS_OK ||
        shadowspill_memory_pool_allocate(runtime, 0U, 128U, 1U, stream, &temporary) !=
            SHADOWSPILL_STATUS_OK ||
        shadowspill_test_after_task(runtime, execution.task_id, stream) !=
            SHADOWSPILL_STATUS_NO_PROGRESS) {
        result = -1;
        goto done;
    }

    ShadowSpillRuntimeFailure failure = {0};
    if (shadowspill_runtime_failure(runtime, &failure) !=
        SHADOWSPILL_STATUS_OK) {
        result = -1;
        goto done;
    }
    const ShadowSpillStatus free_status = shadowspill_memory_pool_free(runtime, 0U, temporary.allocation_id, stream
    );
    if (failure.status != SHADOWSPILL_STATUS_NO_PROGRESS ||
        failure.task_id != execution.task_id ||
        failure.object_id != object.object_id ||
        failure.requested_bytes != object.size_bytes ||
        free_status != SHADOWSPILL_STATUS_NO_PROGRESS) {
        fprintf(
            stderr,
            "failed prefetch recovery: status=%u task=%llu object=%llu bytes=%llu free_status=%u\n",
            (unsigned)failure.status,
            (unsigned long long)failure.task_id,
            (unsigned long long)failure.object_id,
            (unsigned long long)failure.requested_bytes,
            (unsigned)free_status
        );
        result = -1;
    }

done:
    shadowspill_test_destroy_runtime(runtime);
    if (stream.words[0] != 0U) {
        (void)shadowspill_mock_destroy_compute_stream(mock, stream);
    }
    shadowspill_mock_backend_destroy(mock);
    return result;
}

int main(void) {
#define REQUIRE_FAILURE_CANARY(call)                                        \
    do {                                                                    \
        if ((call) != 0) {                                                  \
            fprintf(stderr, "runtime failure canary failed: %s\n", #call); \
            return EXIT_FAILURE;                                            \
        }                                                                   \
    } while (0)
    REQUIRE_FAILURE_CANARY(impossible_oom());
    REQUIRE_FAILURE_CANARY(fragmented_oom());
    REQUIRE_FAILURE_CANARY(worker_failure());
    REQUIRE_FAILURE_CANARY(worker_submission_failure_reaches_dispatcher());
    REQUIRE_FAILURE_CANARY(failed_task_retirement_recovery());
    REQUIRE_FAILURE_CANARY(failed_prefetch_reports_trigger_reservation_oom());
    return EXIT_SUCCESS;
}
