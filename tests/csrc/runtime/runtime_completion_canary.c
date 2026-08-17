#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <shadowspill/backend_mock.h>
#include <shadowspill/runtime.h>

enum {
    COMPLETION_COUNT = 64,
    WAIT_IDLE_ROUNDS = 256,
    ALLOCATION_BYTES = 16,
    /*
     * One delayed FIFO head may be queried repeatedly; successors on the same
     * stream are queried only after that head completes. This bound detects a
     * return to full-population scans without constraining the intentional
     * low-latency head polling cadence.
     */
    MAX_HEAD_POLL_QUERIES = 4096,
};

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
        mock,
        COMPLETION_COUNT * ALLOCATION_BYTES,
        0U,
        1U,
        10000U,
        &topology
    );
    ShadowSpillBackendStream compute = {{0U, 0U}};
    if (shadowspill_runtime_create(&topology.runtime, &runtime) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_mock_create_compute_stream(mock, &compute) != 0 ||
        shadowspill_runtime_reserve_event_leases(runtime, 32U) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_reserve_event_leases(runtime, COMPLETION_COUNT) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_reserve_retirement_records(
            runtime, COMPLETION_COUNT
        ) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_reserve_memory_lease_records(
            runtime, 0U, COMPLETION_COUNT
        ) !=
            SHADOWSPILL_RUNTIME_OK) {
        return EXIT_FAILURE;
    }

    ShadowSpillAllocation allocations[COMPLETION_COUNT] = {{0}};
    for (uint64_t index = 0U; index < COMPLETION_COUNT; ++index) {
        if (shadowspill_memory_pool_allocate(runtime, 0U,
                ALLOCATION_BYTES,
                1U,
                compute,
                &allocations[index]
            ) != SHADOWSPILL_RUNTIME_OK) {
            return EXIT_FAILURE;
        }
    }
    if (shadowspill_mock_enqueue_compute(mock, compute, 1000000U) != 0) {
        return EXIT_FAILURE;
    }
    for (uint64_t index = 0U; index < COMPLETION_COUNT; ++index) {
        if (shadowspill_memory_pool_free(runtime, 0U, allocations[index].allocation_id, compute
            ) != SHADOWSPILL_RUNTIME_OK) {
            return EXIT_FAILURE;
        }
    }
    if (shadowspill_runtime_wait_idle(runtime) != SHADOWSPILL_RUNTIME_OK) {
        return EXIT_FAILURE;
    }
    ShadowSpillMockBackendStatistics batch_statistics = {0};
    shadowspill_mock_backend_statistics(mock, &batch_statistics);
    if (batch_statistics.event_queries >
            COMPLETION_COUNT + MAX_HEAD_POLL_QUERIES) {
        return EXIT_FAILURE;
    }

    /*
     * Exercise the exact final-retirement notification boundary repeatedly.
     * Each wait begins while one event-backed retirement is pending and must
     * return even when the worker clears it immediately.
     */
    for (uint64_t round = 0U; round < WAIT_IDLE_ROUNDS; ++round) {
        ShadowSpillAllocation allocation = {0};
        if (shadowspill_memory_pool_allocate(runtime, 0U,
                ALLOCATION_BYTES,
                1U,
                compute,
                &allocation
            ) != SHADOWSPILL_RUNTIME_OK || shadowspill_memory_pool_free(runtime, 0U, allocation.allocation_id, compute
            ) != SHADOWSPILL_RUNTIME_OK || shadowspill_runtime_wait_idle(
                runtime
            ) != SHADOWSPILL_RUNTIME_OK) {
            return EXIT_FAILURE;
        }
    }

    ShadowSpillMockBackendStatistics backend_statistics = {0};
    ShadowSpillRuntimeStatistics runtime_statistics = {0};
    shadowspill_mock_backend_statistics(mock, &backend_statistics);
    (void)printf(
        "completion_count=%u event_queries=%llu queries_per_completion=%.3f\n",
        COMPLETION_COUNT + WAIT_IDLE_ROUNDS,
        (unsigned long long)backend_statistics.event_queries,
        (double)backend_statistics.event_queries /
            (double)(COMPLETION_COUNT + WAIT_IDLE_ROUNDS)
    );
    if (shadowspill_runtime_statistics(runtime, &runtime_statistics) !=
            SHADOWSPILL_RUNTIME_OK ||
        backend_statistics.events_created !=
            COMPLETION_COUNT + WAIT_IDLE_ROUNDS ||
        backend_statistics.events_destroyed !=
            COMPLETION_COUNT + WAIT_IDLE_ROUNDS ||
        runtime_statistics.pending_retirements != 0U ||
        runtime_statistics.live_allocations != 0U ||
        runtime_statistics.event_lease_capacity != COMPLETION_COUNT ||
        runtime_statistics.event_lease_in_use != 0U ||
        runtime_statistics.event_lease_peak_in_use == 0U ||
        runtime_statistics.event_lease_peak_in_use > COMPLETION_COUNT ||
        runtime_statistics.event_lease_growth_rejections != 0U ||
        runtime_statistics.retirement_record_capacity != COMPLETION_COUNT ||
        runtime_statistics.retirement_record_in_use != 0U ||
        runtime_statistics.retirement_record_peak_in_use == 0U ||
        runtime_statistics.retirement_record_peak_in_use > COMPLETION_COUNT ||
        runtime_statistics.retirement_record_growth_rejections != 0U ||
        runtime_statistics.memory_lease_record_capacity !=
            COMPLETION_COUNT ||
        runtime_statistics.memory_lease_record_in_use != 0U ||
        runtime_statistics.memory_lease_record_peak_in_use !=
            COMPLETION_COUNT ||
        runtime_statistics.memory_lease_record_growth_rejections != 0U ||
        runtime_statistics.free_bytes !=
            COMPLETION_COUNT * ALLOCATION_BYTES) {
        return EXIT_FAILURE;
    }

    if (shadowspill_runtime_close(runtime) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_mock_destroy_compute_stream(mock, compute) != 0) {
        return EXIT_FAILURE;
    }
    shadowspill_runtime_destroy(runtime);
    shadowspill_mock_backend_destroy(mock);
    return EXIT_SUCCESS;
}
