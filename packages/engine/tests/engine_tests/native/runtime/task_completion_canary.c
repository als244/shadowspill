#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <shadowspill/backend_mock.h>
#include <shadowspill/runtime.h>

#include "runtime_test.h"

static int cross_stream_completion(uint8_t successor_kind) {
    ShadowSpillBackend mock = {0};
    const ShadowSpillMockBackendConfig config = {0};
    if (shadowspill_mock_backend_create(&config, &mock) != 0) {
        return -1;
    }
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillBackendStream producer = 0U, consumer = 0U, independent = 0U;
    ShadowSpillBackendEvent consumer_done = 0U, independent_done = 0U;
    int failed = shadowspill_test_create_runtime(
        &mock, 64U, 64U, 16U, 1000U, &runtime
    ) != SHADOWSPILL_STATUS_OK ||
        mock.create_stream(mock.state, &producer) != 0 ||
        mock.create_stream(mock.state, &consumer) != 0 ||
        mock.create_stream(mock.state, &independent) != 0 ||
        mock.create_event(mock.state, &consumer_done, 0U) != 0 ||
        mock.create_event(mock.state, &independent_done, 0U) != 0;
    const ShadowSpillFixedPlacementDescription placements[2] = {
        {.task_id = 1U, .ordinal = 0U, .object_id = SHADOWSPILL_RUNTIME_NO_ID,
         .bytes = 64U, .alignment_bytes = 16U, .kind = SHADOWSPILL_FIXED_TASK_ALLOCATION},
        {.task_id = 2U, .ordinal = 0U, .object_id = SHADOWSPILL_RUNTIME_NO_ID,
         .bytes = 64U, .alignment_bytes = 16U, .kind = SHADOWSPILL_FIXED_TASK_ALLOCATION},
    };
    const ShadowSpillFixedDependencyDescription dependency = {
        .predecessor_task_id = 1U,
        .predecessor_action_ordinal = SHADOWSPILL_RUNTIME_NO_ID,
        .successor_task_id = 2U,
        .successor_ordinal = successor_kind == SHADOWSPILL_FIXED_TASK_START
            ? SHADOWSPILL_RUNTIME_NO_ID : 0U,
        .successor_kind = successor_kind,
    };
    const ShadowSpillFixedLayoutDescription layout = {
        .abi_version = SHADOWSPILL_ABI_VERSION, .slice_bytes = 64U,
        .placements = placements, .placement_count = 2U,
        .dependencies = &dependency, .dependency_count = 1U,
    };
    const ShadowSpillTaskAllocationContractStep contract[2] = {
        {.allocation_ordinal = 0U, .requested_bytes = 64U, .charged_bytes = 64U,
         .alignment_bytes = 16U, .operation = SHADOWSPILL_TASK_ALLOCATION_ALLOCATE},
        {.allocation_ordinal = 0U, .requested_bytes = 64U, .charged_bytes = 64U,
         .alignment_bytes = 16U, .operation = SHADOWSPILL_TASK_ALLOCATION_FREE},
    };
    const ShadowSpillTaskDescription first = {
        .task_id = 1U, .allocation_contract_steps = contract,
        .allocation_contract_step_count = 2U, .enforce_allocation_contract = 1U,
    };
    ShadowSpillTaskDescription second = first;
    second.task_id = 2U;
    failed = failed ||
        shadowspill_test_admit_fixed_layout(runtime, &layout) != SHADOWSPILL_STATUS_OK ||
        shadowspill_test_admit_task(runtime, &first) != SHADOWSPILL_STATUS_OK ||
        shadowspill_test_admit_task(runtime, &second) != SHADOWSPILL_STATUS_OK ||
        shadowspill_test_seal_fixed_layout(runtime) != SHADOWSPILL_STATUS_OK;

    for (int iteration = 0; iteration < 3 && !failed; ++iteration) {
        if (successor_kind == SHADOWSPILL_FIXED_TASK_START) {
            /* Neither a missing completion nor the previous invocation's event
               can authorize the consumer. Failure must leave it retryable. */
            failed = shadowspill_test_before_task(runtime, 2U, consumer, NULL, 0U) !=
                SHADOWSPILL_STATUS_INVALID_STATE;
        }
        ShadowSpillAllocation a = {0}, b = {0};
        failed = failed ||
        shadowspill_test_before_task(runtime, 1U, producer, NULL, 0U) != SHADOWSPILL_STATUS_OK ||
            shadowspill_memory_pool_allocate(runtime, 0U, 64U, 16U, producer, &a) != SHADOWSPILL_STATUS_OK ||
            shadowspill_mock_enqueue_compute(&mock, producer, 100000000U) != 0 ||
            shadowspill_memory_pool_free(runtime, 0U, a.allocation_id, producer) != SHADOWSPILL_STATUS_OK ||
            shadowspill_test_after_task(runtime, 1U, producer) != SHADOWSPILL_STATUS_OK ||
            shadowspill_test_before_task(runtime, 2U, consumer, NULL, 0U) != SHADOWSPILL_STATUS_OK ||
            shadowspill_memory_pool_allocate(runtime, 0U, 64U, 16U, consumer, &b) != SHADOWSPILL_STATUS_OK ||
            a.pointer != b.pointer ||
            shadowspill_memory_pool_free(runtime, 0U, b.allocation_id, consumer) != SHADOWSPILL_STATUS_OK ||
            shadowspill_test_after_task(runtime, 2U, consumer) != SHADOWSPILL_STATUS_OK ||
            mock.record_event(mock.state, consumer_done, consumer) != 0 ||
            mock.record_event(mock.state, independent_done, independent) != 0;
        int complete = 0;
        failed = failed ||
        mock.query_event(mock.state, independent_done, &complete) != 0 || !complete ||
            mock.query_event(mock.state, consumer_done, &complete) != 0 || complete ||
            shadowspill_runtime_wait_idle(runtime) != SHADOWSPILL_STATUS_OK ||
            mock.query_event(mock.state, consumer_done, &complete) != 0 || !complete;
    }
    ShadowSpillTestStatistics stats = {0};
    failed = failed ||
        shadowspill_test_statistics(runtime, &stats) != SHADOWSPILL_STATUS_OK ||
        stats.runtime.wait_events_inserted < 3U;
    shadowspill_test_destroy_runtime(runtime);
    if (consumer_done != 0U) mock.destroy_event(mock.state, consumer_done);
    if (independent_done != 0U) mock.destroy_event(mock.state, independent_done);
    if (producer != 0U) mock.destroy_stream(mock.state, producer);
    if (consumer != 0U) mock.destroy_stream(mock.state, consumer);
    if (independent != 0U) mock.destroy_stream(mock.state, independent);
    shadowspill_backend_destroy(&mock);
    return failed ? -1 : 0;
}

int main(void) {
    if (cross_stream_completion(SHADOWSPILL_FIXED_TASK_START) != 0 ||
        cross_stream_completion(SHADOWSPILL_FIXED_TASK_ALLOCATION) != 0) {
        fprintf(stderr, "cross-stream task completion failed\n");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
