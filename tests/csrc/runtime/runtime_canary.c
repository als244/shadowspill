#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <shadowspill/backend_mock.h>
#include <shadowspill/runtime.h>

#include "runtime_test.h"

static int best_fit_preserves_largest_range(void) {
    ShadowSpillMockBackend *mock = NULL;
    const ShadowSpillMockBackendConfig mock_config = {
        .abi_version = SHADOWSPILL_MOCK_BACKEND_ABI_VERSION,
    };
    if (shadowspill_mock_backend_create(&mock_config, &mock) != 0) {
        return -1;
    }
    ShadowSpillMockRuntimeTopology topology;
    shadowspill_mock_runtime_topology(mock, 256U, 0U, 1U, 1000U, &topology);
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillBackendStream compute = {{0U, 0U}};
    int failed = shadowspill_runtime_create(&topology.runtime, &runtime) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_mock_create_compute_stream(mock, &compute) != 0;
    ShadowSpillAllocation allocations[5] = {{0}};
    const uint64_t sizes[] = {64U, 32U, 96U, 32U};
    for (uint32_t index = 0U; !failed && index < 4U; ++index) {
        failed = shadowspill_allocate(
            runtime, sizes[index], 1U, compute, &allocations[index]
        ) != SHADOWSPILL_RUNTIME_OK;
    }
    failed = failed || shadowspill_free(
            runtime, allocations[1].allocation_id, compute
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_free(
            runtime, allocations[0].allocation_id, compute
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_free(
            runtime, allocations[3].allocation_id, compute
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_allocate(
            runtime, 48U, 1U, compute, &allocations[4]
        ) != SHADOWSPILL_RUNTIME_OK;
    ShadowSpillRuntimeStatistics statistics = {0};
    failed = failed || shadowspill_runtime_statistics(runtime, &statistics) !=
            SHADOWSPILL_RUNTIME_OK ||
        allocations[4].pointer != allocations[0].pointer ||
        statistics.largest_free_range_bytes != 32U ||
        allocations[4].charged_bytes != 48U ||
        statistics.pending_retirements != 3U;
    if (failed) {
        fprintf(
            stderr,
            "best-fit dynamic mismatch: chosen=%p expected=%p largest=%llu pending=%llu\n",
            allocations[4].pointer,
            allocations[0].pointer,
            (unsigned long long)statistics.largest_free_range_bytes,
            (unsigned long long)statistics.pending_retirements
        );
    }
    shadowspill_runtime_destroy(runtime);
    if (compute.words[0] != 0U) {
        (void)shadowspill_mock_destroy_compute_stream(mock, compute);
    }
    shadowspill_mock_backend_destroy(mock);
    return failed ? -1 : 0;
}

static int same_stream_split_retires_cleanly(void) {
    ShadowSpillMockBackend *mock = NULL;
    const ShadowSpillMockBackendConfig mock_config = {
        .abi_version = SHADOWSPILL_MOCK_BACKEND_ABI_VERSION,
        .event_delay_nanoseconds = 100000000U,
    };
    if (shadowspill_mock_backend_create(&mock_config, &mock) != 0) {
        return -1;
    }
    ShadowSpillMockRuntimeTopology topology;
    shadowspill_mock_runtime_topology(mock, 128U, 0U, 1U, 1000U, &topology);
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillBackendStream compute = {{0U, 0U}};
    ShadowSpillAllocation original = {0};
    ShadowSpillAllocation split = {0};
    int failed = shadowspill_runtime_create(&topology.runtime, &runtime) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_mock_create_compute_stream(mock, &compute) != 0 ||
        shadowspill_allocate(runtime, 128U, 1U, compute, &original) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_free(runtime, original.allocation_id, compute) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_allocate(runtime, 64U, 1U, compute, &split) !=
            SHADOWSPILL_RUNTIME_OK ||
        split.pointer != original.pointer ||
        shadowspill_free(runtime, split.allocation_id, compute) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_wait_idle(runtime) != SHADOWSPILL_RUNTIME_OK;
    ShadowSpillRuntimeStatistics statistics = {0};
    failed = failed || shadowspill_runtime_statistics(runtime, &statistics) !=
            SHADOWSPILL_RUNTIME_OK ||
        statistics.live_allocations != 0U ||
        statistics.allocated_bytes != 0U ||
        statistics.free_bytes != 128U ||
        statistics.free_prefix_bytes != 128U ||
        statistics.largest_free_range_bytes != 128U;
    shadowspill_runtime_destroy(runtime);
    if (compute.words[0] != 0U) {
        (void)shadowspill_mock_destroy_compute_stream(mock, compute);
    }
    shadowspill_mock_backend_destroy(mock);
    return failed ? -1 : 0;
}

static int repeated_nested_splits_reclaim_the_pool(void) {
    enum {
        POOL_BYTES = 4096,
        ITERATIONS = 128,
        SPLITS_PER_ITERATION = 8,
    };
    ShadowSpillMockBackend *mock = NULL;
    const ShadowSpillMockBackendConfig mock_config = {
        .abi_version = SHADOWSPILL_MOCK_BACKEND_ABI_VERSION,
        .event_delay_nanoseconds = 100000U,
    };
    if (shadowspill_mock_backend_create(&mock_config, &mock) != 0) {
        return -1;
    }
    ShadowSpillMockRuntimeTopology topology;
    shadowspill_mock_runtime_topology(
        mock, POOL_BYTES, 0U, 16U, 1000U, &topology
    );
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillBackendStream compute = {{0U, 0U}};
    int failed = shadowspill_runtime_create(&topology.runtime, &runtime) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_mock_create_compute_stream(mock, &compute) != 0;
    uint32_t pseudo_random = 0x6d2b79f5U;
    for (uint32_t iteration = 0U;
         !failed && iteration < ITERATIONS;
         ++iteration) {
        ShadowSpillAllocation whole = {0};
        ShadowSpillAllocation splits[SPLITS_PER_ITERATION] = {{0}};
        failed = shadowspill_allocate(
                runtime, POOL_BYTES, 16U, compute, &whole
            ) != SHADOWSPILL_RUNTIME_OK ||
            shadowspill_mock_enqueue_compute(mock, compute, 1000U) != 0 ||
            shadowspill_free(runtime, whole.allocation_id, compute) !=
                SHADOWSPILL_RUNTIME_OK;
        uint64_t remaining = POOL_BYTES;
        for (uint32_t index = 0U;
             !failed && index < SPLITS_PER_ITERATION;
             ++index) {
            pseudo_random = pseudo_random * 1664525U + 1013904223U;
            const uint64_t slots_left = SPLITS_PER_ITERATION - index;
            const uint64_t maximum = remaining - 16U * (slots_left - 1U);
            const uint64_t slot_count = maximum / 16U;
            const uint64_t bytes = index + 1U == SPLITS_PER_ITERATION
                ? remaining
                : 16U * (1U + pseudo_random % slot_count);
            failed = shadowspill_allocate(
                    runtime, bytes, 16U, compute, &splits[index]
                ) != SHADOWSPILL_RUNTIME_OK;
            remaining -= bytes;
        }
        for (uint32_t index = 0U;
             !failed && index < SPLITS_PER_ITERATION;
             ++index) {
            failed = shadowspill_free(
                    runtime, splits[index].allocation_id, compute
                ) != SHADOWSPILL_RUNTIME_OK;
        }
        failed = failed || shadowspill_runtime_wait_idle(runtime) !=
                SHADOWSPILL_RUNTIME_OK;
        ShadowSpillRuntimeStatistics statistics = {0};
        failed = failed || shadowspill_runtime_statistics(
                runtime, &statistics
            ) != SHADOWSPILL_RUNTIME_OK ||
            statistics.live_allocations != 0U ||
            statistics.allocated_bytes != 0U ||
            statistics.free_bytes != POOL_BYTES ||
            statistics.largest_free_range_bytes != POOL_BYTES;
    }
    shadowspill_runtime_destroy(runtime);
    if (compute.words[0] != 0U) {
        (void)shadowspill_mock_destroy_compute_stream(mock, compute);
    }
    shadowspill_mock_backend_destroy(mock);
    return failed ? -1 : 0;
}

int main(void) {
    if (best_fit_preserves_largest_range() != 0) {
        fprintf(stderr, "runtime canary failed: best_fit_preserves_largest_range\n");
        return EXIT_FAILURE;
    }
    if (same_stream_split_retires_cleanly() != 0) {
        fprintf(stderr, "runtime canary failed: same_stream_split_retires_cleanly\n");
        return EXIT_FAILURE;
    }
    if (repeated_nested_splits_reclaim_the_pool() != 0) {
        fprintf(stderr, "runtime canary failed: repeated_nested_splits_reclaim_the_pool\n");
        return EXIT_FAILURE;
    }
    ShadowSpillMockBackend *mock = NULL;
    const ShadowSpillMockBackendConfig mock_config = {
        .abi_version = SHADOWSPILL_MOCK_BACKEND_ABI_VERSION,
        .fetch_delay_nanoseconds = 100000000U,
        .evict_delay_nanoseconds = 100000000U,
    };
    if (shadowspill_mock_backend_create(&mock_config, &mock) != 0) {
        return EXIT_FAILURE;
    }
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillMockRuntimeTopology topology;
    shadowspill_mock_runtime_topology(
        mock, 256U, 256U, 1U, 10000U, &topology
    );
    if (shadowspill_runtime_create(&topology.runtime, &runtime) !=
        SHADOWSPILL_RUNTIME_OK) {
        shadowspill_mock_backend_destroy(mock);
        return EXIT_FAILURE;
    }
    ShadowSpillBackendStream compute = {{0U, 0U}};
    if (shadowspill_mock_create_compute_stream(mock, &compute) != 0) {
        shadowspill_runtime_destroy(runtime);
        shadowspill_mock_backend_destroy(mock);
        return EXIT_FAILURE;
    }
    ShadowSpillAllocation allocation = {0};
    ShadowSpillAllocation resolved = {0};
    if (shadowspill_allocate(runtime, 128U, 16U, compute, &allocation) !=
        SHADOWSPILL_RUNTIME_OK || allocation.pointer == NULL ||
        allocation.charged_bytes != 128U ||
        shadowspill_allocation_for_pointer(
            runtime, allocation.pointer, &resolved
        ) != SHADOWSPILL_RUNTIME_OK ||
        resolved.allocation_id != allocation.allocation_id ||
        resolved.generation != allocation.generation) {
        return EXIT_FAILURE;
    }
    if (shadowspill_record_stream(runtime, allocation.allocation_id, compute) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_mock_enqueue_compute(mock, compute, 100000U) != 0 ||
        shadowspill_free(runtime, allocation.allocation_id, compute) !=
            SHADOWSPILL_RUNTIME_OK) {
        return EXIT_FAILURE;
    }
    ShadowSpillAllocation same_stream_reuse = {0};
    ShadowSpillRuntimeStatistics statistics = {0};
    if (shadowspill_allocate(
            runtime, 128U, 16U, compute, &same_stream_reuse
        ) != SHADOWSPILL_RUNTIME_OK ||
        same_stream_reuse.pointer == allocation.pointer ||
        shadowspill_runtime_statistics(runtime, &statistics) !=
            SHADOWSPILL_RUNTIME_OK ||
        statistics.pending_retirements != 1U ||
        shadowspill_free(runtime, same_stream_reuse.allocation_id, compute) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_wait_idle(runtime) != SHADOWSPILL_RUNTIME_OK) {
        return EXIT_FAILURE;
    }
    if (shadowspill_allocation_for_pointer(
            runtime, allocation.pointer, &resolved
        ) != SHADOWSPILL_RUNTIME_INVALID_STATE) {
        return EXIT_FAILURE;
    }
    if (shadowspill_runtime_statistics(runtime, &statistics) !=
            SHADOWSPILL_RUNTIME_OK ||
        statistics.allocated_bytes != 0U || statistics.free_bytes != 256U ||
        statistics.largest_free_range_bytes != 256U ||
        statistics.pending_retirements != 0U) {
        return EXIT_FAILURE;
    }

    const ShadowSpillObjectDescription object = {
        .object_id = 7U,
        .size_bytes = 128U,
        .initial_version = 4U,
        .retain_spill_copy = 1U,
        .initially_spill_resident = 1U,
    };
    ShadowSpillAllocation first_generation = {0};
    unsigned char original_payload[128];
    unsigned char restored_payload[128] = {0};
    for (uint32_t index = 0U; index < 128U; ++index) {
        original_payload[index] = (unsigned char)(index ^ 0x5aU);
    }
    if (shadowspill_register_object(runtime, &object) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_write_spill_object(
            runtime,
            object.object_id,
            original_payload,
            sizeof(original_payload)
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_read_spill_object(
            runtime,
            object.object_id,
            restored_payload,
            sizeof(restored_payload)
        ) != SHADOWSPILL_RUNTIME_OK ||
        memcmp(
            original_payload, restored_payload, sizeof(original_payload)
        ) != 0 ||
        shadowspill_runtime_resize_spill_pool(runtime, 512U) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_resize_spill_pool(runtime, 255U) !=
            SHADOWSPILL_RUNTIME_INVALID_ARGUMENT ||
        shadowspill_read_spill_object(
            runtime,
            object.object_id,
            restored_payload,
            sizeof(restored_payload)
        ) != SHADOWSPILL_RUNTIME_OK ||
        memcmp(
            original_payload, restored_payload, sizeof(original_payload)
        ) != 0 ||
        shadowspill_runtime_statistics(runtime, &statistics) !=
            SHADOWSPILL_RUNTIME_OK ||
        statistics.spill_pool_bytes != 512U ||
        statistics.spill_allocated_bytes != 128U ||
        shadowspill_allocate(runtime, 128U, 16U, compute, &first_generation) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_bind_object(
            runtime, object.object_id, first_generation.allocation_id
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_mock_enqueue_compute(mock, compute, 100000U) != 0) {
        return EXIT_FAILURE;
    }
    const ShadowSpillRuntimeAction offload = {
        .object_id = object.object_id,
        .kind = SHADOWSPILL_RUNTIME_OFFLOAD,
    };
    const ShadowSpillObjectUpdate update = {
        .object_id = object.object_id,
        .version_delta = 1U,
    };
    const ShadowSpillTaskDescription first_task = {
        .task_id = 1U,
        .updates = &update,
        .update_count = 1U,
        .actions = &offload,
        .action_count = 1U,
    };
    if (shadowspill_test_admit_task(runtime, &first_task) !=
            SHADOWSPILL_RUNTIME_OK || shadowspill_test_before_task(
            runtime, first_task.task_id, compute, NULL, 0U
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_test_after_task(
            runtime, first_task.task_id, compute
        ) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_wait_idle(runtime) != SHADOWSPILL_RUNTIME_OK) {
        return EXIT_FAILURE;
    }
    ShadowSpillAllocation blocker = {0};
    if (shadowspill_allocate(runtime, 64U, 16U, compute, &blocker) !=
        SHADOWSPILL_RUNTIME_OK) {
        return EXIT_FAILURE;
    }
    const ShadowSpillRuntimeAction prefetch = {
        .object_id = object.object_id,
        .kind = SHADOWSPILL_RUNTIME_PREFETCH,
    };
    if (shadowspill_test_submit_actions(
            runtime, 2U, compute, &prefetch, 1U
        ) !=
        SHADOWSPILL_RUNTIME_OK) {
        return EXIT_FAILURE;
    }
    const uint64_t input_ids[] = {object.object_id, object.object_id};
    const ShadowSpillObjectUpdate post_prefetch_update = {
        .object_id = object.object_id,
        .version_delta = 1U,
    };
    const ShadowSpillTaskDescription consumer = {
        .task_id = 3U,
        .input_object_ids = input_ids,
        .input_count = 2U,
        .updates = &post_prefetch_update,
        .update_count = 1U,
    };
    ShadowSpillObjectBinding bindings[2] = {{0}};
    if (shadowspill_test_admit_task(runtime, &consumer) !=
            SHADOWSPILL_RUNTIME_OK || shadowspill_test_before_task(
            runtime, consumer.task_id, compute, bindings, 2U
        ) != SHADOWSPILL_RUNTIME_OK ||
        bindings[0].pointer == first_generation.pointer ||
        bindings[0].pointer != bindings[1].pointer ||
        bindings[0].pointer == blocker.pointer ||
        bindings[0].generation <= first_generation.generation ||
        bindings[0].authoritative_version != 5U) {
        return EXIT_FAILURE;
    }
    ShadowSpillObjectSnapshot snapshot = {0};
    if (shadowspill_test_after_task(runtime, consumer.task_id, compute) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_wait_idle(runtime) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_object_snapshot(runtime, object.object_id, &snapshot) !=
            SHADOWSPILL_RUNTIME_OK ||
        snapshot.residency != SHADOWSPILL_OBJECT_EXECUTION_READY ||
        snapshot.authoritative_version != 6U || snapshot.execution_version != 6U ||
        snapshot.spill_current) {
        return EXIT_FAILURE;
    }
    const ShadowSpillRuntimeAction release = {
        .object_id = object.object_id,
        .kind = SHADOWSPILL_RUNTIME_RELEASE,
    };
    if (shadowspill_mock_enqueue_compute(mock, compute, 100000U) != 0 ||
        shadowspill_test_submit_actions(
            runtime, 6U, compute, &release, 1U
        ) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_wait_idle(runtime) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_free(runtime, blocker.allocation_id, compute) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_wait_idle(runtime) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_statistics(runtime, &statistics) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_object_snapshot(runtime, object.object_id, &snapshot) !=
            SHADOWSPILL_RUNTIME_OK ||
        statistics.evict_transfers != 1U ||
        statistics.fetch_transfers != 1U ||
        statistics.bytes_evicted != 128U ||
        statistics.bytes_fetched != 128U ||
        statistics.wait_events_inserted != 1U ||
        statistics.allocated_bytes != 0U ||
        snapshot.authoritative_version != 6U || snapshot.spill_current ||
        snapshot.residency != SHADOWSPILL_OBJECT_RELEASED ||
        snapshot.execution_pointer != NULL) {
        return EXIT_FAILURE;
    }
    const ShadowSpillObjectDescription pair[] = {
        {
            .object_id = 8U,
            .size_bytes = 64U,
            .initially_spill_resident = 1U,
        },
        {
            .object_id = 9U,
            .size_bytes = 64U,
            .initially_spill_resident = 1U,
        },
    };
    const ShadowSpillRuntimeAction pair_prefetch[] = {
        {.object_id = 8U, .kind = SHADOWSPILL_RUNTIME_PREFETCH},
        {.object_id = 9U, .kind = SHADOWSPILL_RUNTIME_PREFETCH},
    };
    if (shadowspill_register_object(runtime, &pair[0]) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_register_object(runtime, &pair[1]) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_test_submit_actions(
            runtime, 4U, compute, pair_prefetch, 2U
        ) !=
            SHADOWSPILL_RUNTIME_OK) {
        return EXIT_FAILURE;
    }
    const uint64_t pair_inputs[] = {8U, 9U};
    const ShadowSpillRuntimeAction pair_release[] = {
        {.object_id = 8U, .kind = SHADOWSPILL_RUNTIME_RELEASE},
        {.object_id = 9U, .kind = SHADOWSPILL_RUNTIME_RELEASE},
    };
    const ShadowSpillTaskDescription pair_consumer = {
        .task_id = 5U,
        .input_object_ids = pair_inputs,
        .input_count = 2U,
        .actions = pair_release,
        .action_count = 2U,
    };
    ShadowSpillObjectBinding pair_bindings[2] = {{0}};
    if (shadowspill_test_admit_task(runtime, &pair_consumer) !=
            SHADOWSPILL_RUNTIME_OK || shadowspill_test_before_task(
            runtime, pair_consumer.task_id, compute, pair_bindings, 2U
        ) != SHADOWSPILL_RUNTIME_OK ||
        pair_bindings[0].pointer == pair_bindings[1].pointer) {
        return EXIT_FAILURE;
    }
    if (shadowspill_mock_enqueue_compute(mock, compute, 100000U) != 0 ||
        shadowspill_test_after_task(
            runtime, pair_consumer.task_id, compute
        ) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_wait_idle(runtime) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_statistics(runtime, &statistics) !=
            SHADOWSPILL_RUNTIME_OK ||
        statistics.wait_events_inserted != 3U ||
        statistics.fetch_transfers != 3U ||
        statistics.allocated_bytes != 0U) {
        return EXIT_FAILURE;
    }
    if (shadowspill_runtime_close(runtime) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_close(runtime) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_mock_destroy_compute_stream(mock, compute) != 0) {
        return EXIT_FAILURE;
    }
    shadowspill_test_destroy_runtime(runtime);
    shadowspill_mock_backend_destroy(mock);
    return EXIT_SUCCESS;
}
