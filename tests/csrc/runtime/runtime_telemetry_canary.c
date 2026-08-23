#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <shadowspill/backend_mock.h>
#include <shadowspill/runtime.h>

#include "runtime_test.h"

static int create_runtime(
    ShadowSpillMockBackend **mock,
    ShadowSpillRuntime **runtime,
    ShadowSpillBackendStream *compute
) {
    const ShadowSpillMockBackendConfig mock_config = {
        .abi_version = SHADOWSPILL_MOCK_BACKEND_ABI_VERSION,
    };
    if (shadowspill_mock_backend_create(&mock_config, mock) != 0) {
        return -1;
    }
    if (shadowspill_test_create_runtime(
            *mock, 256U, 0U, 1U, 1000U, runtime
        ) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_mock_create_compute_stream(*mock, compute) != 0) {
        shadowspill_test_destroy_runtime(*runtime);
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
    shadowspill_test_destroy_runtime(runtime);
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
        shadowspill_allocation_scope_begin(runtime, 0U, 42U) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_memory_pool_allocate(runtime, 0U, 64U, 1U, compute, &first) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_memory_pool_free(runtime, 0U, first.allocation_id, compute) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_memory_pool_allocate(runtime, 0U, 96U, 1U, compute, &second) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_register_object(runtime, &object) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_test_publish_initial(runtime, object.object_id, second.pointer, NULL) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_allocation_scope_end(runtime, 42U, compute) !=
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
         logical_events[0].slab_offset != 0U ||
         logical_events[2].slab_offset != 64U ||
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
    const ShadowSpillTaskDescription task = {.task_id = 77U};
    shadowspill_mock_backend_statistics(mock, &before);
    int failed = shadowspill_test_admit_task(runtime, &task) !=
            SHADOWSPILL_RUNTIME_OK || shadowspill_test_before_task(
            runtime, task.task_id, compute, NULL, 0U
        ) != SHADOWSPILL_RUNTIME_OK;
    for (uint32_t index = 0U; !failed && index < 2500U; ++index) {
        ShadowSpillAllocation allocation = {0};
        failed = shadowspill_memory_pool_allocate(runtime, 0U, 64U, 1U, compute, &allocation
            ) != SHADOWSPILL_RUNTIME_OK;
        failed = failed || shadowspill_memory_pool_free(runtime, 0U, allocation.allocation_id, compute
        ) != SHADOWSPILL_RUNTIME_OK;
    }
    shadowspill_mock_backend_statistics(mock, &during);
    ShadowSpillRuntimeStatistics during_runtime = {0};
    failed = failed || shadowspill_runtime_statistics(
            runtime, &during_runtime
        ) != SHADOWSPILL_RUNTIME_OK ||
        during.operation_count != before.operation_count ||
        during_runtime.pending_retirements != 1U ||
        during_runtime.allocated_bytes != 64U ||
        during_runtime.live_allocations != 1U ||
        shadowspill_test_after_task(
            runtime, task.task_id, compute
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
            "during_pending=%llu during_live=%llu during_allocated=%llu "
            "pending=%llu actions=%llu allocated=%llu\n",
            (unsigned long long)before.operation_count,
            (unsigned long long)during.operation_count,
            (unsigned long long)after.operation_count,
            (unsigned long long)during_runtime.pending_retirements,
            (unsigned long long)during_runtime.live_allocations,
            (unsigned long long)during_runtime.allocated_bytes,
            (unsigned long long)runtime_statistics.pending_retirements,
            (unsigned long long)runtime_statistics.queued_actions,
            (unsigned long long)runtime_statistics.allocated_bytes
        );
    }
    destroy_runtime(mock, runtime, compute);
    return failed ? -1 : 0;
}

static int queued_transfers_survive_retirement_only_task(void) {
    ShadowSpillMockBackend *mock = NULL;
    const ShadowSpillMockBackendConfig mock_config = {
        .abi_version = SHADOWSPILL_MOCK_BACKEND_ABI_VERSION,
        .fetch_delay_nanoseconds = 100000000U,
    };
    if (shadowspill_mock_backend_create(&mock_config, &mock) != 0) {
        return -1;
    }
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillBackendStream compute = {{0U, 0U}};
    int failed = shadowspill_test_create_runtime(
            mock, 256U, 128U, 1U, 1000U, &runtime
        ) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_mock_create_compute_stream(mock, &compute) != 0;
    const ShadowSpillObjectDescription objects[] = {
        {
            .object_id = 20U,
            .size_bytes = 32U,
            .initial_pool_id = 1U,
            .initially_resident = 1U,
        },
        {
            .object_id = 21U,
            .size_bytes = 32U,
            .initial_pool_id = 1U,
            .initially_resident = 1U,
        },
        {
            .object_id = 22U,
            .size_bytes = 32U,
            .initial_pool_id = 1U,
            .initially_resident = 1U,
        },
    };
    const ShadowSpillRuntimeAction initial[] = {
        {.object_id = 20U, .kind = SHADOWSPILL_RUNTIME_PREFETCH},
        {.object_id = 21U, .kind = SHADOWSPILL_RUNTIME_PREFETCH},
    };
    const ShadowSpillRuntimeAction appended = {
        .object_id = 22U,
        .kind = SHADOWSPILL_RUNTIME_PREFETCH,
    };
    const ShadowSpillTaskDescription temporary_task = {.task_id = 81U};
    for (uint32_t index = 0U; !failed && index < 3U; ++index) {
        failed = shadowspill_register_object(runtime, &objects[index]) !=
            SHADOWSPILL_RUNTIME_OK;
    }
    failed = failed || shadowspill_test_submit_actions(
            runtime, 80U, compute, initial, 2U
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_test_admit_task(
            runtime, &temporary_task
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_test_before_task(
            runtime, temporary_task.task_id, compute, NULL, 0U
        ) != SHADOWSPILL_RUNTIME_OK;
    ShadowSpillAllocation temporary = {0};
    failed = failed || shadowspill_memory_pool_allocate(runtime, 0U, 16U, 1U, compute, &temporary
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_memory_pool_free(runtime, 0U, temporary.allocation_id, compute) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_test_after_task(
            runtime, temporary_task.task_id, compute
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_test_submit_actions(
            runtime, 82U, compute, &appended, 1U
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
        .abi_version = SHADOWSPILL_MOCK_BACKEND_ABI_VERSION,
        .event_delay_nanoseconds = 100000000U,
    };
    if (shadowspill_mock_backend_create(&mock_config, &mock) != 0) {
        return -1;
    }
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillBackendStream compute = {{0U, 0U}};
    int failed = shadowspill_test_create_runtime(
            mock, 128U, 128U, 1U, 1000U, &runtime
        ) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_mock_create_compute_stream(mock, &compute) != 0 ||
        shadowspill_runtime_reserve_memory_lease_records(
            runtime, 0U, 8U
        ) != SHADOWSPILL_RUNTIME_OK;
    const ShadowSpillObjectDescription object = {
        .object_id = 30U,
        .size_bytes = 96U,
        .initial_pool_id = 1U,
        .initially_resident = 1U,
    };
    ShadowSpillAllocation first = {0};
    ShadowSpillAllocation second = {0};
    const ShadowSpillTaskDescription allocator_task = {.task_id = 90U};
    failed = failed || shadowspill_register_object(runtime, &object) !=
            SHADOWSPILL_RUNTIME_OK || shadowspill_test_admit_task(
            runtime, &allocator_task
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_test_before_task(
            runtime, allocator_task.task_id, compute, NULL, 0U
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_memory_pool_allocate(runtime, 0U, 64U, 1U, compute, &first) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_memory_pool_allocate(runtime, 0U, 64U, 1U, compute, &second) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_memory_pool_free(runtime, 0U, first.allocation_id, compute) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_memory_pool_free(runtime, 0U, second.allocation_id, compute) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_test_after_task(
            runtime, allocator_task.task_id, compute
        ) != SHADOWSPILL_RUNTIME_OK;
    const ShadowSpillRuntimeAction prefetch = {
        .object_id = object.object_id,
        .kind = SHADOWSPILL_RUNTIME_PREFETCH,
    };
    failed = failed || shadowspill_test_submit_actions(
            runtime, 91U, compute, &prefetch, 1U
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_wait_idle(runtime) != SHADOWSPILL_RUNTIME_OK;
    ShadowSpillObjectSnapshot snapshot = {0};
    failed = failed || shadowspill_object_snapshot(
            runtime, object.object_id, &snapshot
        ) != SHADOWSPILL_RUNTIME_OK ||
        snapshot.residency != SHADOWSPILL_OBJECT_EXECUTION_READY ||
        snapshot.execution_pointer == NULL;
    if (failed) {
        fprintf(stderr, "causal reservation did not await completed retirements\n");
    }
    destroy_runtime(mock, runtime, compute);
    return failed ? -1 : 0;
}

/*
 * A full event record stops recording; it does not stop the runtime. Events
 * describe what happened, so running out of room to describe an allocation
 * says nothing about whether it succeeded. Latching a failure here used to
 * make every later allocation in the process fail, reported with the operands
 * of whichever allocation took the last slot.
 */
static int overflow_stops_recording_not_the_runtime(void) {
    ShadowSpillMockBackend *mock = NULL;
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillBackendStream compute = {{0U, 0U}};
    if (create_runtime(&mock, &runtime, &compute) != 0) {
        return -1;
    }
    ShadowSpillAllocation allocation = {0};
    ShadowSpillAllocation second = {0};
    /* One slot: the allocation fills it and the free overflows. */
    if (shadowspill_allocation_telemetry_start(runtime, 1U) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_memory_pool_allocate(runtime, 0U, 8U, 1U, compute, &allocation) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_memory_pool_free(runtime, 0U, allocation.allocation_id, compute) !=
            SHADOWSPILL_RUNTIME_OK) {
        destroy_runtime(mock, runtime, compute);
        return -1;
    }
    ShadowSpillRuntimeFailure failure = {0};
    ShadowSpillRuntimeStatistics statistics = {0};
    const int failed =
        /* The overflow is reported ... */
        shadowspill_runtime_statistics(runtime, &statistics) !=
            SHADOWSPILL_RUNTIME_OK ||
        statistics.allocation_event_overflow != 1U ||
        /* ... and nothing was latched ... */
        shadowspill_runtime_failure(runtime, &failure) !=
            SHADOWSPILL_RUNTIME_OK ||
        failure.status != SHADOWSPILL_RUNTIME_OK ||
        /* ... so the runtime still serves allocations. */
        shadowspill_memory_pool_allocate(runtime, 0U, 8U, 1U, compute, &second) !=
            SHADOWSPILL_RUNTIME_OK;
    destroy_runtime(mock, runtime, compute);
    return failed ? -1 : 0;
}

static int bounded_runtime_trace_is_opt_in(void) {
    ShadowSpillMockBackend *mock = NULL;
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillBackendStream compute = {{0U, 0U}};
    const ShadowSpillMockBackendConfig mock_config = {
        .abi_version = SHADOWSPILL_MOCK_BACKEND_ABI_VERSION,
    };
    if (shadowspill_mock_backend_create(&mock_config, &mock) != 0) {
        return -1;
    }
    if (shadowspill_test_create_runtime(
            mock, 256U, 128U, 1U, 1000U, &runtime
        ) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_mock_create_compute_stream(mock, &compute) != 0) {
        destroy_runtime(mock, runtime, compute);
        return -1;
    }
    const ShadowSpillTraceConfig trace_config = {
        .abi_version = SHADOWSPILL_ABI_VERSION,
        .event_capacity = 64U,
        .allocation_event_capacity = 64U,
    };
    ShadowSpillTraceSummary summary = {0};
    const ShadowSpillObjectDescription object = {
        .object_id = 50U,
        .size_bytes = 32U,
        .initial_pool_id = 1U,
        .initially_resident = 1U,
    };
    const ShadowSpillRuntimeAction prefetch = {
        .object_id = object.object_id,
        .kind = SHADOWSPILL_RUNTIME_PREFETCH,
    };
    const ShadowSpillRuntimeAction release = {
        .object_id = object.object_id,
        .kind = SHADOWSPILL_RUNTIME_RELEASE,
    };
    const uint64_t input = object.object_id;
    const ShadowSpillTaskDescription consumer = {
        .task_id = 101U,
        .input_object_ids = &input,
        .input_count = 1U,
        .actions = &release,
        .action_count = 1U,
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
        shadowspill_test_submit_actions(
            runtime, 100U, compute, &prefetch, 1U
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_wait_idle(runtime) != SHADOWSPILL_RUNTIME_OK;
    ShadowSpillObjectBinding binding = {0};
    failed = failed || shadowspill_test_admit_task(runtime, &consumer) !=
            SHADOWSPILL_RUNTIME_OK || shadowspill_test_before_task(
            runtime, consumer.task_id, compute, &binding, 1U
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_test_after_task(
            runtime, consumer.task_id, compute
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_wait_idle(runtime) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_trace_end(runtime) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_trace_read(
            runtime, &summary, NULL, 0U, NULL, 0U
        ) != SHADOWSPILL_RUNTIME_OK ||
        summary.abi_version != SHADOWSPILL_ABI_VERSION ||
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
#define REQUIRE_TELEMETRY_CANARY(call)                                      \
    do {                                                                    \
        fprintf(stderr, "runtime telemetry canary starting: %s\n", #call);  \
        if ((call) != 0) {                                                  \
            fprintf(stderr, "runtime telemetry canary failed: %s\n", #call); \
            return EXIT_FAILURE;                                            \
        }                                                                   \
    } while (0)
    REQUIRE_TELEMETRY_CANARY(ordered_task_capture());
    REQUIRE_TELEMETRY_CANARY(same_stream_retirement_is_task_batched());
    REQUIRE_TELEMETRY_CANARY(queued_transfers_survive_retirement_only_task());
    REQUIRE_TELEMETRY_CANARY(all_completed_retirements_precede_action_admission());
    REQUIRE_TELEMETRY_CANARY(bounded_runtime_trace_is_opt_in());
    REQUIRE_TELEMETRY_CANARY(overflow_stops_recording_not_the_runtime());
    return EXIT_SUCCESS;
}
