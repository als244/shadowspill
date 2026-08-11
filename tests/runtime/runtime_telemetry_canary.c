#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <shadowspill/backend_mock.h>
#include <shadowspill/runtime.h>

static int create_runtime(
    ShadowSpillMockBackend **mock,
    ShadowSpillRuntime **runtime,
    ShadowSpillBackendStream *compute
) {
    const ShadowSpillMockBackendConfig mock_config = {
        .abi_version = SHADOWSPILL_BACKEND_ABI_VERSION,
    };
    if (shadowspill_mock_backend_create(&mock_config, mock) != 0) {
        return -1;
    }
    const ShadowSpillRuntimeConfig runtime_config = {
        .abi_version = SHADOWSPILL_RUNTIME_ABI_VERSION,
        .device_slab_bytes = 256U,
        .host_arena_bytes = 0U,
        .minimum_alignment = 1U,
        .progress_poll_nanoseconds = 1000U,
        .backend = shadowspill_mock_backend_vtable(*mock),
    };
    if (shadowspill_runtime_create(&runtime_config, runtime) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_mock_create_compute_stream(*mock, compute) != 0) {
        shadowspill_runtime_destroy(*runtime);
        shadowspill_mock_backend_destroy(*mock);
        return -1;
    }
    return 0;
}

static void destroy_runtime(
    ShadowSpillMockBackend *mock,
    ShadowSpillRuntime *runtime,
    ShadowSpillBackendStream compute
) {
    (void)shadowspill_runtime_close(runtime);
    (void)shadowspill_mock_destroy_compute_stream(mock, compute);
    shadowspill_runtime_destroy(runtime);
    shadowspill_mock_backend_destroy(mock);
}

static int ordered_task_capture(void) {
    ShadowSpillMockBackend *mock = NULL;
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillBackendStream compute = {{0U, 0U}};
    if (create_runtime(&mock, &runtime, &compute) != 0) {
        return -1;
    }
    ShadowSpillAllocation first = {0};
    ShadowSpillAllocation second = {0};
    const ShadowSpillObjectDescription object = {
        .object_id = 9U,
        .size_bytes = 96U,
    };
    int failed = shadowspill_allocation_telemetry_start(runtime, 8U) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_before_task(
            runtime, 42U, compute, NULL, 0U, NULL, 0U
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_allocate(runtime, 64U, 1U, compute, &first) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_free(runtime, first.allocation_id, compute) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_allocate(runtime, 96U, 1U, compute, &second) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_register_object(runtime, &object) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_bind_object(runtime, object.object_id, second.allocation_id) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_after_task(runtime, 42U, compute, NULL, 0U, NULL, 0U) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_allocation_telemetry_stop(runtime) !=
            SHADOWSPILL_RUNTIME_OK;

    uint64_t count = 0U;
    ShadowSpillAllocationEvent events[8] = {{0}};
    ShadowSpillRuntimeStatistics statistics = {0};
    failed = failed || shadowspill_allocation_telemetry_read(
            runtime, NULL, 0U, &count
        ) != SHADOWSPILL_RUNTIME_OK || (count != 4U && count != 5U) ||
        shadowspill_allocation_telemetry_read(
            runtime, events, 8U, &count
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_statistics(runtime, &statistics) !=
            SHADOWSPILL_RUNTIME_OK ||
        statistics.allocation_events != count ||
        statistics.allocation_event_capacity != 8U ||
        statistics.allocation_event_overflow != 0U;
    const uint8_t expected_logical_kinds[] = {
        SHADOWSPILL_ALLOCATION_CREATED,
        SHADOWSPILL_ALLOCATION_LOGICAL_FREED,
        SHADOWSPILL_ALLOCATION_CREATED,
        SHADOWSPILL_ALLOCATION_PROMOTED,
    };
    const uint64_t expected_logical_bytes[] = {64U, 64U, 96U, 96U};
    ShadowSpillAllocationEvent logical_events[4] = {{0}};
    uint64_t logical_count = 0U;
    for (uint64_t index = 0U; !failed && index < count; ++index) {
        if (events[index].sequence != index || events[index].task_id != 42U ||
            (events[index].kind == SHADOWSPILL_ALLOCATION_RELEASED &&
             events[index].charged_bytes != 64U)) {
            failed = 1;
        } else if (events[index].kind != SHADOWSPILL_ALLOCATION_RELEASED) {
            if (logical_count >= 4U) {
                failed = 1;
            } else {
                logical_events[logical_count++] = events[index];
            }
        }
    }
    for (uint64_t index = 0U; !failed && index < logical_count; ++index) {
        if (logical_events[index].kind != expected_logical_kinds[index] ||
            logical_events[index].charged_bytes != expected_logical_bytes[index]) {
            failed = 1;
        }
    }
    if (!failed &&
        (logical_count != 4U ||
         logical_events[0].slab_offset + logical_events[0].charged_bytes != 256U ||
         logical_events[2].slab_offset + logical_events[2].charged_bytes != 192U ||
         logical_events[0].category != SHADOWSPILL_ALLOCATION_ANONYMOUS ||
         logical_events[3].category != SHADOWSPILL_ALLOCATION_PLANNED_OBJECT)) {
            failed = 1;
    }
    if (failed) {
        fprintf(stderr, "telemetry count=%llu logical=%llu\n",
            (unsigned long long)count,
            (unsigned long long)logical_count);
        for (uint64_t index = 0U; index < count && index < 8U; ++index) {
            fprintf(stderr,
                "event[%llu] sequence=%llu task=%llu kind=%u bytes=%llu "
                "offset=%llu category=%u\n",
                (unsigned long long)index,
                (unsigned long long)events[index].sequence,
                (unsigned long long)events[index].task_id,
                (unsigned int)events[index].kind,
                (unsigned long long)events[index].charged_bytes,
                (unsigned long long)events[index].slab_offset,
                (unsigned int)events[index].category);
        }
    }
    destroy_runtime(mock, runtime, compute);
    return failed ? -1 : 0;
}

static int overflow_is_failure(void) {
    ShadowSpillMockBackend *mock = NULL;
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillBackendStream compute = {{0U, 0U}};
    if (create_runtime(&mock, &runtime, &compute) != 0) {
        return -1;
    }
    ShadowSpillAllocation allocation = {0};
    if (shadowspill_allocation_telemetry_start(runtime, 1U) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_allocate(runtime, 8U, 1U, compute, &allocation) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_free(runtime, allocation.allocation_id, compute) !=
            SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE) {
        destroy_runtime(mock, runtime, compute);
        return -1;
    }
    ShadowSpillRuntimeFailure failure = {0};
    ShadowSpillRuntimeStatistics statistics = {0};
    const int failed = shadowspill_runtime_failure(runtime, &failure) !=
            SHADOWSPILL_RUNTIME_OK ||
        failure.status != SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE ||
        shadowspill_runtime_statistics(runtime, &statistics) !=
            SHADOWSPILL_RUNTIME_OK ||
        statistics.allocation_event_overflow != 1U;
    destroy_runtime(mock, runtime, compute);
    return failed ? -1 : 0;
}

int main(void) {
    return ordered_task_capture() == 0 && overflow_is_failure() == 0
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}
