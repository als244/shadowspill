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
        .execution_pool_bytes = 256U,
        .spill_pool_bytes = 0U,
        .minimum_alignment = 1U,
        .worker_poll_nanoseconds = 1000U,
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

static int same_stream_retirement_is_task_batched(void) {
    ShadowSpillMockBackend *mock = NULL;
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillBackendStream compute = {{0U, 0U}};
    if (create_runtime(&mock, &runtime, &compute) != 0) {
        return -1;
    }
    ShadowSpillMockBackendStatistics before = {0};
    ShadowSpillMockBackendStatistics during = {0};
    ShadowSpillMockBackendStatistics after = {0};
    shadowspill_mock_backend_statistics(mock, &before);
    int failed = shadowspill_before_task(
            runtime, 77U, compute, NULL, 0U, NULL, 0U
        ) != SHADOWSPILL_RUNTIME_OK;
    void *first_pointer = NULL;
    for (uint32_t index = 0U; !failed && index < 2500U; ++index) {
        ShadowSpillAllocation allocation = {0};
        failed = shadowspill_allocate(
                runtime, 64U, 1U, compute, &allocation
            ) != SHADOWSPILL_RUNTIME_OK;
        if (index == 0U) {
            first_pointer = allocation.pointer;
        } else if (allocation.pointer != first_pointer) {
            failed = 1;
        }
        failed = failed || shadowspill_free(
                runtime, allocation.allocation_id, compute
            ) != SHADOWSPILL_RUNTIME_OK;
    }
    shadowspill_mock_backend_statistics(mock, &during);
    failed = failed || during.operation_count != before.operation_count ||
        shadowspill_after_task(
            runtime, 77U, compute, NULL, 0U, NULL, 0U
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_wait_idle(runtime) != SHADOWSPILL_RUNTIME_OK;
    shadowspill_mock_backend_statistics(mock, &after);
    ShadowSpillRuntimeStatistics runtime_statistics = {0};
    failed = failed || shadowspill_runtime_statistics(
            runtime, &runtime_statistics
        ) != SHADOWSPILL_RUNTIME_OK ||
        runtime_statistics.pending_retirements != 0U ||
        runtime_statistics.allocated_bytes != 0U ||
        after.operation_count - during.operation_count >= 32U;
    if (failed) {
        fprintf(
            stderr,
            "task retirement batching before=%llu during=%llu after=%llu "
            "pending=%llu allocated=%llu\n",
            (unsigned long long)before.operation_count,
            (unsigned long long)during.operation_count,
            (unsigned long long)after.operation_count,
            (unsigned long long)runtime_statistics.pending_retirements,
            (unsigned long long)runtime_statistics.allocated_bytes
        );
    }
    destroy_runtime(mock, runtime, compute);
    return failed ? -1 : 0;
}

static int queued_transfers_survive_retirement_only_task(void) {
    ShadowSpillMockBackend *mock = NULL;
    const ShadowSpillMockBackendConfig mock_config = {
        .abi_version = SHADOWSPILL_BACKEND_ABI_VERSION,
        .fetch_delay_nanoseconds = 100000000U,
    };
    if (shadowspill_mock_backend_create(&mock_config, &mock) != 0) {
        return -1;
    }
    const ShadowSpillRuntimeConfig runtime_config = {
        .abi_version = SHADOWSPILL_RUNTIME_ABI_VERSION,
        .execution_pool_bytes = 256U,
        .spill_pool_bytes = 128U,
        .minimum_alignment = 1U,
        .worker_poll_nanoseconds = 1000U,
        .backend = shadowspill_mock_backend_vtable(mock),
    };
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillBackendStream compute = {{0U, 0U}};
    int failed = shadowspill_runtime_create(&runtime_config, &runtime) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_mock_create_compute_stream(mock, &compute) != 0;
    const ShadowSpillObjectDescription objects[] = {
        {.object_id = 20U, .size_bytes = 32U, .initially_spill_resident = 1U},
        {.object_id = 21U, .size_bytes = 32U, .initially_spill_resident = 1U},
        {.object_id = 22U, .size_bytes = 32U, .initially_spill_resident = 1U},
    };
    const ShadowSpillRuntimeAction initial[] = {
        {.object_id = 20U, .kind = SHADOWSPILL_RUNTIME_PREFETCH},
        {.object_id = 21U, .kind = SHADOWSPILL_RUNTIME_PREFETCH},
    };
    const ShadowSpillRuntimeAction appended = {
        .object_id = 22U,
        .kind = SHADOWSPILL_RUNTIME_PREFETCH,
    };
    for (uint32_t index = 0U; !failed && index < 3U; ++index) {
        failed = shadowspill_register_object(runtime, &objects[index]) !=
            SHADOWSPILL_RUNTIME_OK;
    }
    failed = failed || shadowspill_after_task(
            runtime, 80U, compute, NULL, 0U, initial, 2U
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_before_task(
            runtime, 81U, compute, NULL, 0U, NULL, 0U
        ) != SHADOWSPILL_RUNTIME_OK;
    ShadowSpillAllocation temporary = {0};
    failed = failed || shadowspill_allocate(
            runtime, 16U, 1U, compute, &temporary
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_free(runtime, temporary.allocation_id, compute) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_after_task(
            runtime, 81U, compute, NULL, 0U, NULL, 0U
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_after_task(
            runtime, 82U, compute, NULL, 0U, &appended, 1U
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_wait_idle(runtime) != SHADOWSPILL_RUNTIME_OK;
    for (uint32_t index = 0U; !failed && index < 3U; ++index) {
        ShadowSpillObjectSnapshot snapshot = {0};
        failed = shadowspill_object_snapshot(
                runtime, objects[index].object_id, &snapshot
            ) != SHADOWSPILL_RUNTIME_OK ||
            snapshot.residency != SHADOWSPILL_OBJECT_EXECUTION_READY ||
            snapshot.execution_pointer == NULL;
    }
    if (failed) {
        fprintf(stderr, "retirement-only task dropped a queued transfer\n");
    }
    destroy_runtime(mock, runtime, compute);
    return failed ? -1 : 0;
}

static int all_completed_retirements_precede_action_admission(void) {
    ShadowSpillMockBackend *mock = NULL;
    const ShadowSpillMockBackendConfig mock_config = {
        .abi_version = SHADOWSPILL_BACKEND_ABI_VERSION,
        .event_delay_nanoseconds = 100000000U,
    };
    if (shadowspill_mock_backend_create(&mock_config, &mock) != 0) {
        return -1;
    }
    const ShadowSpillRuntimeConfig runtime_config = {
        .abi_version = SHADOWSPILL_RUNTIME_ABI_VERSION,
        .execution_pool_bytes = 128U,
        .spill_pool_bytes = 128U,
        .minimum_alignment = 1U,
        .worker_poll_nanoseconds = 1000U,
        .backend = shadowspill_mock_backend_vtable(mock),
    };
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillBackendStream compute = {{0U, 0U}};
    int failed = shadowspill_runtime_create(&runtime_config, &runtime) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_mock_create_compute_stream(mock, &compute) != 0;
    const ShadowSpillObjectDescription object = {
        .object_id = 30U,
        .size_bytes = 96U,
        .initially_spill_resident = 1U,
    };
    ShadowSpillAllocation first = {0};
    ShadowSpillAllocation second = {0};
    failed = failed || shadowspill_register_object(runtime, &object) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_before_task(
            runtime, 90U, compute, NULL, 0U, NULL, 0U
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_allocate(runtime, 64U, 1U, compute, &first) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_allocate(runtime, 64U, 1U, compute, &second) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_free(runtime, first.allocation_id, compute) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_free(runtime, second.allocation_id, compute) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_after_task(
            runtime, 90U, compute, NULL, 0U, NULL, 0U
        ) != SHADOWSPILL_RUNTIME_OK;
    const ShadowSpillRuntimeAction prefetch = {
        .object_id = object.object_id,
        .kind = SHADOWSPILL_RUNTIME_PREFETCH,
    };
    failed = failed || shadowspill_after_task(
            runtime, 91U, compute, NULL, 0U, &prefetch, 1U
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_wait_idle(runtime) != SHADOWSPILL_RUNTIME_OK;
    ShadowSpillObjectSnapshot snapshot = {0};
    failed = failed || shadowspill_object_snapshot(
            runtime, object.object_id, &snapshot
        ) != SHADOWSPILL_RUNTIME_OK ||
        snapshot.residency != SHADOWSPILL_OBJECT_EXECUTION_READY ||
        snapshot.execution_pointer == NULL;
    if (failed) {
        fprintf(stderr, "action admission ran before all completed retirements\n");
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

static int bounded_runtime_trace_is_opt_in(void) {
    ShadowSpillMockBackend *mock = NULL;
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillBackendStream compute = {{0U, 0U}};
    const ShadowSpillMockBackendConfig mock_config = {
        .abi_version = SHADOWSPILL_BACKEND_ABI_VERSION,
    };
    const ShadowSpillRuntimeConfig runtime_config = {
        .abi_version = SHADOWSPILL_RUNTIME_ABI_VERSION,
        .execution_pool_bytes = 256U,
        .spill_pool_bytes = 128U,
        .minimum_alignment = 1U,
        .worker_poll_nanoseconds = 1000U,
    };
    if (shadowspill_mock_backend_create(&mock_config, &mock) != 0) {
        return -1;
    }
    ShadowSpillRuntimeConfig configured_runtime = runtime_config;
    configured_runtime.backend = shadowspill_mock_backend_vtable(mock);
    if (shadowspill_runtime_create(&configured_runtime, &runtime) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_mock_create_compute_stream(mock, &compute) != 0) {
        destroy_runtime(mock, runtime, compute);
        return -1;
    }
    const ShadowSpillTraceConfig trace_config = {
        .abi_version = SHADOWSPILL_TRACE_ABI_VERSION,
        .event_capacity = 64U,
        .allocation_event_capacity = 64U,
    };
    ShadowSpillTraceSummary summary = {0};
    const ShadowSpillObjectDescription object = {
        .object_id = 50U,
        .size_bytes = 32U,
        .initially_spill_resident = 1U,
    };
    const ShadowSpillRuntimeAction prefetch = {
        .object_id = object.object_id,
        .kind = SHADOWSPILL_RUNTIME_PREFETCH,
    };
    const ShadowSpillRuntimeAction release = {
        .object_id = object.object_id,
        .kind = SHADOWSPILL_RUNTIME_RELEASE,
    };
    int failed = shadowspill_trace_prepare(runtime, &trace_config) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_trace_read(
            runtime, &summary, NULL, 0U, NULL, 0U
        ) != SHADOWSPILL_RUNTIME_OK ||
        summary.event_count != 0U || summary.active != 0U ||
        shadowspill_register_object(runtime, &object) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_trace_begin(runtime, 7U) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_after_task(
            runtime, 100U, compute, NULL, 0U, &prefetch, 1U
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_wait_idle(runtime) != SHADOWSPILL_RUNTIME_OK;
    ShadowSpillObjectBinding binding = {0};
    const uint64_t input = object.object_id;
    failed = failed || shadowspill_before_task(
            runtime, 101U, compute, &input, 1U, &binding, 1U
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_after_task(
            runtime, 101U, compute, NULL, 0U, &release, 1U
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_wait_idle(runtime) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_trace_end(runtime) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_trace_read(
            runtime, &summary, NULL, 0U, NULL, 0U
        ) != SHADOWSPILL_RUNTIME_OK ||
        summary.abi_version != SHADOWSPILL_TRACE_ABI_VERSION ||
        summary.step_id != 7U || summary.active != 0U ||
        summary.event_count < 9U || summary.allocation_event_count == 0U ||
        summary.event_overflow != 0U ||
        summary.allocation_event_overflow != 0U;
    ShadowSpillTraceEvent events[64] = {{0}};
    ShadowSpillAllocationEvent allocations[64] = {{0}};
    failed = failed || shadowspill_trace_read(
            runtime,
            &summary,
            events,
            64U,
            allocations,
            64U
        ) != SHADOWSPILL_RUNTIME_OK;
    int saw_queued = 0;
    int saw_reserved = 0;
    int saw_dispatched = 0;
    int saw_completed = 0;
    int saw_before = 0;
    int saw_after = 0;
    for (uint64_t index = 0U; !failed && index < summary.event_count; ++index) {
        failed = events[index].sequence != index ||
            events[index].step_id != 7U || events[index].timestamp_ns == 0U;
        saw_queued |= events[index].kind == SHADOWSPILL_TRACE_ACTION_QUEUED;
        saw_reserved |=
            events[index].kind == SHADOWSPILL_TRACE_DESTINATION_RESERVED;
        saw_dispatched |=
            events[index].kind == SHADOWSPILL_TRACE_TRANSFER_DISPATCHED;
        saw_completed |=
            events[index].kind == SHADOWSPILL_TRACE_TRANSFER_COMPLETED;
        saw_before |= events[index].kind == SHADOWSPILL_TRACE_BEFORE_TASK;
        saw_after |= events[index].kind == SHADOWSPILL_TRACE_AFTER_TASK;
    }
    failed = failed || events[0].kind != SHADOWSPILL_TRACE_SESSION_BEGIN ||
        events[summary.event_count - 1U].kind != SHADOWSPILL_TRACE_SESSION_END ||
        !saw_queued || !saw_reserved || !saw_dispatched || !saw_completed ||
        !saw_before || !saw_after ||
        shadowspill_unregister_object(runtime, object.object_id) !=
            SHADOWSPILL_RUNTIME_OK;
    if (failed) {
        fprintf(
            stderr,
            "runtime trace events=%llu allocations=%llu overflow=%u/%u\n",
            (unsigned long long)summary.event_count,
            (unsigned long long)summary.allocation_event_count,
            (unsigned int)summary.event_overflow,
            (unsigned int)summary.allocation_event_overflow
        );
    }
    destroy_runtime(mock, runtime, compute);
    return failed ? -1 : 0;
}

int main(void) {
    return ordered_task_capture() == 0 &&
        same_stream_retirement_is_task_batched() == 0 &&
        queued_transfers_survive_retirement_only_task() == 0 &&
        all_completed_retirements_precede_action_admission() == 0 &&
        bounded_runtime_trace_is_opt_in() == 0 &&
        overflow_is_failure() == 0
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}
