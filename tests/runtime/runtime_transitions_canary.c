#define _POSIX_C_SOURCE 200809L

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <shadowspill/backend_mock.h>
#include <shadowspill/runtime.h>

#include "internal.h"

typedef struct Fixture {
    ShadowSpillMockBackend *mock;
    ShadowSpillRuntime *runtime;
    ShadowSpillBackendStream compute;
} Fixture;

static int fixture_create(Fixture *fixture) {
    const ShadowSpillMockBackendConfig backend_config = {
        .abi_version = SHADOWSPILL_BACKEND_ABI_VERSION,
        .fetch_delay_nanoseconds = 1000U,
        .evict_delay_nanoseconds = 1000U,
    };
    if (shadowspill_mock_backend_create(
            &backend_config, &fixture->mock
        ) != 0) {
        return -1;
    }
    const ShadowSpillRuntimeConfig runtime_config = {
        .abi_version = SHADOWSPILL_RUNTIME_ABI_VERSION,
        .execution_pool_bytes = 128U,
        .spill_pool_bytes = 256U,
        .minimum_alignment = 1U,
        .worker_poll_nanoseconds = 1000U,
        .backend = shadowspill_mock_backend_vtable(fixture->mock),
    };
    if (shadowspill_runtime_create(
            &runtime_config, &fixture->runtime
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_mock_create_compute_stream(
            fixture->mock, &fixture->compute
        ) != 0) {
        return -1;
    }
    return 0;
}

static void fixture_destroy(Fixture *fixture) {
    shadowspill_runtime_destroy(fixture->runtime);
    if (fixture->compute.words[0] != 0U) {
        (void)shadowspill_mock_destroy_compute_stream(
            fixture->mock, fixture->compute
        );
    }
    shadowspill_mock_backend_destroy(fixture->mock);
}

static void sleep_milliseconds(uint64_t milliseconds) {
    const struct timespec delay = {
        .tv_sec = (time_t)(milliseconds / 1000U),
        .tv_nsec = (long)((milliseconds % 1000U) * 1000000U),
    };
    (void)nanosleep(&delay, NULL);
}

static int spill_object_rekey_preserves_authoritative_lease(void) {
    Fixture fixture = {0};
    if (fixture_create(&fixture) != 0) {
        return -1;
    }
    const uint64_t persistent_id = 1001U;
    const uint64_t plan_id = 17U;
    const ShadowSpillObjectDescription description = {
        .object_id = persistent_id,
        .size_bytes = 32U,
        .initial_version = 9U,
        .initially_spill_resident = 1U,
        .retain_spill_copy = 1U,
    };
    uint8_t payload[32];
    uint8_t restored[32] = {0};
    for (uint32_t index = 0U; index < sizeof(payload); ++index) {
        payload[index] = (uint8_t)(index * 7U + 3U);
    }
    ShadowSpillObjectSnapshot before = {0};
    ShadowSpillObjectSnapshot after = {0};
    ShadowSpillObjectSnapshot fetched = {0};
    ShadowSpillRuntimeStatistics before_statistics = {0};
    ShadowSpillRuntimeStatistics after_statistics = {0};
    const ShadowSpillRuntimeAction fetch = {
        .object_id = plan_id,
        .kind = SHADOWSPILL_RUNTIME_PREFETCH,
    };
    int failed = shadowspill_register_object(fixture.runtime, &description) !=
            SHADOWSPILL_RUNTIME_OK || shadowspill_write_spill_object(
            fixture.runtime, persistent_id, payload, sizeof(payload)
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_object_snapshot(
            fixture.runtime, persistent_id, &before
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_runtime_statistics(
            fixture.runtime, &before_statistics
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_rekey_object(
            fixture.runtime, persistent_id, plan_id
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_object_snapshot(
            fixture.runtime, persistent_id, &after
        ) != SHADOWSPILL_RUNTIME_INVALID_STATE || shadowspill_object_snapshot(
            fixture.runtime, plan_id, &after
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_runtime_statistics(
            fixture.runtime, &after_statistics
        ) != SHADOWSPILL_RUNTIME_OK || after.object_id != plan_id ||
        after.spill_pointer != before.spill_pointer ||
        after.spill_version != before.spill_version ||
        after.authoritative_version != before.authoritative_version ||
        after_statistics.spill_allocated_bytes !=
            before_statistics.spill_allocated_bytes ||
        after_statistics.registered_objects !=
            before_statistics.registered_objects || shadowspill_read_spill_object(
            fixture.runtime, plan_id, restored, sizeof(restored)
        ) != SHADOWSPILL_RUNTIME_OK ||
        memcmp(payload, restored, sizeof(payload)) != 0 || shadowspill_after_task(
            fixture.runtime, 1U, fixture.compute, NULL, 0U, &fetch, 1U
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_runtime_wait_idle(
            fixture.runtime
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_object_snapshot(
            fixture.runtime, plan_id, &fetched
        ) != SHADOWSPILL_RUNTIME_OK ||
        fetched.residency != SHADOWSPILL_OBJECT_EXECUTION_READY ||
        fetched.execution_pointer == NULL ||
        memcmp(payload, fetched.execution_pointer, sizeof(payload)) != 0;
    fixture_destroy(&fixture);
    return failed ? -1 : 0;
}

static int prefetch_window_is_enqueued_without_host_blocking(void) {
    ShadowSpillMockBackend *mock = NULL;
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillBackendStream compute = {{0U, 0U}};
    const ShadowSpillMockBackendConfig backend_config = {
        .abi_version = SHADOWSPILL_BACKEND_ABI_VERSION,
        .fetch_delay_nanoseconds = 100000000U,
        .event_delay_nanoseconds = 50000000U,
    };
    const ShadowSpillRuntimeConfig runtime_config = {
        .abi_version = SHADOWSPILL_RUNTIME_ABI_VERSION,
        .execution_pool_bytes = 128U,
        .spill_pool_bytes = 128U,
        .minimum_alignment = 1U,
        .worker_poll_nanoseconds = 100000U,
    };
    if (shadowspill_mock_backend_create(&backend_config, &mock) != 0) {
        return -1;
    }
    ShadowSpillRuntimeConfig configured = runtime_config;
    configured.backend = shadowspill_mock_backend_vtable(mock);
    if (shadowspill_runtime_create(&configured, &runtime) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_mock_create_compute_stream(mock, &compute) != 0) {
        shadowspill_runtime_destroy(runtime);
        shadowspill_mock_backend_destroy(mock);
        return -1;
    }
    const ShadowSpillObjectDescription objects[] = {
        {.object_id = 1U, .size_bytes = 32U, .initially_spill_resident = 1U},
        {.object_id = 2U, .size_bytes = 32U, .initially_spill_resident = 1U},
    };
    const ShadowSpillRuntimeAction actions[] = {
        {.object_id = 1U, .kind = SHADOWSPILL_RUNTIME_PREFETCH},
        {.object_id = 2U, .kind = SHADOWSPILL_RUNTIME_PREFETCH},
    };
    int failed = shadowspill_register_object(runtime, &objects[0]) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_register_object(runtime, &objects[1]) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_after_task(
            runtime, 1U, compute, NULL, 0U, actions, 2U
        ) != SHADOWSPILL_RUNTIME_OK;
    sleep_milliseconds(5U);
    ShadowSpillRuntimeStatistics statistics = {0};
    failed = failed || shadowspill_runtime_statistics(runtime, &statistics) !=
            SHADOWSPILL_RUNTIME_OK ||
        statistics.live_allocations != 2U ||
        statistics.allocated_bytes != 64U ||
        statistics.fetch_transfers != 0U;
    sleep_milliseconds(60U);
    failed = failed || shadowspill_runtime_statistics(runtime, &statistics) !=
            SHADOWSPILL_RUNTIME_OK ||
        statistics.live_allocations != 2U ||
        statistics.allocated_bytes != 64U ||
        statistics.fetch_transfers != 2U;
    const uint64_t input = 2U;
    ShadowSpillObjectBinding binding = {0};
    struct timespec started = {0};
    struct timespec finished = {0};
    (void)clock_gettime(CLOCK_MONOTONIC, &started);
    failed = failed || shadowspill_before_task(
            runtime, 2U, compute, &input, 1U, &binding, 1U
        ) != SHADOWSPILL_RUNTIME_OK;
    (void)clock_gettime(CLOCK_MONOTONIC, &finished);
    const uint64_t elapsed_nanoseconds =
        ((uint64_t)finished.tv_sec - (uint64_t)started.tv_sec) * 1000000000U +
        (uint64_t)finished.tv_nsec - (uint64_t)started.tv_nsec;
    failed = failed || elapsed_nanoseconds > 30000000U ||
        binding.object_id != 2U;
    shadowspill_abort_task(runtime);
    failed = failed || shadowspill_runtime_wait_idle(runtime) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_statistics(runtime, &statistics) !=
            SHADOWSPILL_RUNTIME_OK ||
        statistics.live_allocations != 2U ||
        statistics.fetch_transfers != 2U;
    if (failed) {
        fprintf(
            stderr,
            "prefetch window mismatch: live=%llu, allocated=%llu, fetches=%llu\n",
            (unsigned long long)statistics.live_allocations,
            (unsigned long long)statistics.allocated_bytes,
            (unsigned long long)statistics.fetch_transfers
        );
    }
    shadowspill_runtime_destroy(runtime);
    (void)shadowspill_mock_destroy_compute_stream(mock, compute);
    shadowspill_mock_backend_destroy(mock);
    return failed ? -1 : 0;
}

static int inflight_prefetch_transfers_to_caller(void) {
    ShadowSpillMockBackend *mock = NULL;
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillBackendStream compute = {{0U, 0U}};
    const ShadowSpillMockBackendConfig backend_config = {
        .abi_version = SHADOWSPILL_BACKEND_ABI_VERSION,
        .fetch_delay_nanoseconds = 100000000U,
        .event_delay_nanoseconds = 50000000U,
    };
    if (shadowspill_mock_backend_create(&backend_config, &mock) != 0) {
        return -1;
    }
    const ShadowSpillRuntimeConfig runtime_config = {
        .abi_version = SHADOWSPILL_RUNTIME_ABI_VERSION,
        .execution_pool_bytes = 128U,
        .spill_pool_bytes = 128U,
        .minimum_alignment = 1U,
        .worker_poll_nanoseconds = 100000U,
        .backend = shadowspill_mock_backend_vtable(mock),
    };
    if (shadowspill_runtime_create(&runtime_config, &runtime) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_mock_create_compute_stream(mock, &compute) != 0) {
        shadowspill_runtime_destroy(runtime);
        shadowspill_mock_backend_destroy(mock);
        return -1;
    }
    const ShadowSpillObjectDescription object = {
        .object_id = 71U,
        .size_bytes = 32U,
        .initially_spill_resident = 1U,
    };
    const ShadowSpillRuntimeAction fetch = {
        .object_id = object.object_id,
        .kind = SHADOWSPILL_RUNTIME_PREFETCH,
    };
    ShadowSpillObjectBinding binding = {0};
    ShadowSpillAllocation caller = {0};
    ShadowSpillObjectSnapshot snapshot = {0};
    ShadowSpillRuntimeStatistics statistics = {0};
    int failed = shadowspill_register_object(runtime, &object) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_after_task(
            runtime, 1U, compute, NULL, 0U, &fetch, 1U
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_before_task(
            runtime, 2U, compute, &object.object_id, 1U, &binding, 1U
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_after_task(
            runtime, 2U, compute, NULL, 0U, NULL, 0U
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_transfer_object_to_caller(
            runtime, object.object_id, compute, &caller
        ) != SHADOWSPILL_RUNTIME_OK ||
        caller.allocation_id != binding.allocation_id ||
        shadowspill_object_snapshot(
            runtime, object.object_id, &snapshot
        ) != SHADOWSPILL_RUNTIME_OK ||
        snapshot.residency != SHADOWSPILL_OBJECT_RELEASED ||
        snapshot.execution_pointer != NULL ||
        shadowspill_free(runtime, caller.allocation_id, compute) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_wait_idle(runtime) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_statistics(runtime, &statistics) !=
            SHADOWSPILL_RUNTIME_OK ||
        statistics.fetch_transfers != 1U ||
        statistics.live_allocations != 0U;
    shadowspill_runtime_destroy(runtime);
    (void)shadowspill_mock_destroy_compute_stream(mock, compute);
    shadowspill_mock_backend_destroy(mock);
    return failed ? -1 : 0;
}

static int offload_window_is_enqueued_without_host_serialization(void) {
    ShadowSpillMockBackend *mock = NULL;
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillBackendStream compute = {{0U, 0U}};
    const ShadowSpillMockBackendConfig backend_config = {
        .abi_version = SHADOWSPILL_BACKEND_ABI_VERSION,
        .evict_delay_nanoseconds = 100000000U,
        .event_delay_nanoseconds = 50000000U,
    };
    ShadowSpillRuntimeConfig runtime_config = {
        .abi_version = SHADOWSPILL_RUNTIME_ABI_VERSION,
        .execution_pool_bytes = 128U,
        .spill_pool_bytes = 128U,
        .minimum_alignment = 1U,
        .worker_poll_nanoseconds = 100000U,
    };
    if (shadowspill_mock_backend_create(&backend_config, &mock) != 0) {
        return -1;
    }
    runtime_config.backend = shadowspill_mock_backend_vtable(mock);
    if (shadowspill_runtime_create(&runtime_config, &runtime) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_mock_create_compute_stream(mock, &compute) != 0) {
        shadowspill_runtime_destroy(runtime);
        shadowspill_mock_backend_destroy(mock);
        return -1;
    }
    const ShadowSpillObjectDescription objects[] = {
        {.object_id = 1U, .size_bytes = 32U},
        {.object_id = 2U, .size_bytes = 32U},
    };
    ShadowSpillAllocation allocations[2] = {{0}};
    const ShadowSpillRuntimeAction actions[] = {
        {.object_id = 1U, .kind = SHADOWSPILL_RUNTIME_OFFLOAD},
        {.object_id = 2U, .kind = SHADOWSPILL_RUNTIME_OFFLOAD},
    };
    int failed = 0;
    for (uint32_t index = 0U; index < 2U; ++index) {
        failed = failed || shadowspill_register_object(runtime, &objects[index]) !=
                SHADOWSPILL_RUNTIME_OK ||
            shadowspill_allocate(
                runtime, 32U, 1U, compute, &allocations[index]
            ) != SHADOWSPILL_RUNTIME_OK ||
            shadowspill_bind_object(
                runtime, objects[index].object_id,
                allocations[index].allocation_id
            ) != SHADOWSPILL_RUNTIME_OK;
    }
    failed = failed || shadowspill_after_task(
            runtime, 1U, compute, NULL, 0U, actions, 2U
        ) != SHADOWSPILL_RUNTIME_OK;
    sleep_milliseconds(5U);
    ShadowSpillRuntimeStatistics statistics = {0};
    failed = failed || shadowspill_runtime_statistics(runtime, &statistics) !=
            SHADOWSPILL_RUNTIME_OK ||
        statistics.spill_allocated_bytes != 64U ||
        statistics.evict_transfers != 0U;
    sleep_milliseconds(60U);
    failed = failed || shadowspill_runtime_statistics(runtime, &statistics) !=
            SHADOWSPILL_RUNTIME_OK ||
        statistics.spill_allocated_bytes != 64U ||
        statistics.evict_transfers != 2U;
    failed = failed || shadowspill_runtime_wait_idle(runtime) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_statistics(runtime, &statistics) !=
            SHADOWSPILL_RUNTIME_OK ||
        statistics.spill_allocated_bytes != 64U ||
        statistics.evict_transfers != 2U;
    if (failed) {
        fprintf(
            stderr,
            "evict window mismatch: spill=%llu, evicts=%llu\n",
            (unsigned long long)statistics.spill_allocated_bytes,
            (unsigned long long)statistics.evict_transfers
        );
    }
    shadowspill_runtime_destroy(runtime);
    (void)shadowspill_mock_destroy_compute_stream(mock, compute);
    shadowspill_mock_backend_destroy(mock);
    return failed ? -1 : 0;
}

static int trigger_reservation_failure_reports_no_progress(void) {
    Fixture fixture = {0};
    if (fixture_create(&fixture) != 0) {
        return -1;
    }
    const ShadowSpillObjectDescription objects[] = {
        {.object_id = 1U, .size_bytes = 80U, .initially_spill_resident = 1U},
        {.object_id = 2U, .size_bytes = 80U, .initially_spill_resident = 1U},
    };
    const ShadowSpillRuntimeAction actions[] = {
        {.object_id = 1U, .kind = SHADOWSPILL_RUNTIME_PREFETCH},
        {.object_id = 2U, .kind = SHADOWSPILL_RUNTIME_PREFETCH},
    };
    int failed = shadowspill_register_object(fixture.runtime, &objects[0]) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_register_object(fixture.runtime, &objects[1]) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_after_task(
            fixture.runtime, 1U, fixture.compute, NULL, 0U, actions, 2U
        ) != SHADOWSPILL_RUNTIME_NO_PROGRESS;
    ShadowSpillRuntimeFailure failure = {0};
    failed = failed || shadowspill_runtime_failure(
            fixture.runtime, &failure
        ) != SHADOWSPILL_RUNTIME_OK ||
        failure.status != SHADOWSPILL_RUNTIME_NO_PROGRESS ||
        failure.object_id != 2U || failure.requested_bytes != 80U ||
        failure.free_bytes != 48U ||
        failure.largest_free_range_bytes != 48U;
    if (failed) {
        fprintf(
            stderr,
            "trigger reservation mismatch: status=%u, object=%llu, requested=%llu, free=%llu, largest=%llu\n",
            (unsigned)failure.status,
            (unsigned long long)failure.object_id,
            (unsigned long long)failure.requested_bytes,
            (unsigned long long)failure.free_bytes,
            (unsigned long long)failure.largest_free_range_bytes
        );
    }
    fixture_destroy(&fixture);
    return failed ? -1 : 0;
}

static int invalid_action(
    uint8_t initially_spill_resident,
    uint8_t action_kind
) {
    Fixture fixture = {0};
    if (fixture_create(&fixture) != 0) {
        return -1;
    }
    const ShadowSpillObjectDescription object = {
        .object_id = 1U,
        .size_bytes = 32U,
        .initially_spill_resident = initially_spill_resident,
    };
    const ShadowSpillRuntimeAction action = {
        .object_id = object.object_id,
        .kind = action_kind,
    };
    int result = 0;
    if (shadowspill_register_object(fixture.runtime, &object) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_after_task(
            fixture.runtime, 1U, fixture.compute, NULL, 0U, &action, 1U
        ) != SHADOWSPILL_RUNTIME_PLAN_VIOLATION) {
        result = -1;
    }
    ShadowSpillRuntimeFailure failure = {0};
    if (shadowspill_runtime_failure(fixture.runtime, &failure) !=
            SHADOWSPILL_RUNTIME_OK ||
        failure.object_id != object.object_id) {
        result = -1;
    }
    fixture_destroy(&fixture);
    return result;
}

static int invalid_before_task(uint8_t initially_spill_resident) {
    Fixture fixture = {0};
    if (fixture_create(&fixture) != 0) {
        return -1;
    }
    const ShadowSpillObjectDescription object = {
        .object_id = 1U,
        .size_bytes = 32U,
        .initially_spill_resident = initially_spill_resident,
    };
    const uint64_t input = object.object_id;
    ShadowSpillObjectBinding binding = {0};
    int result = 0;
    if (shadowspill_register_object(fixture.runtime, &object) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_before_task(
            fixture.runtime, 1U, fixture.compute, &input, 1U, &binding, 1U
        ) != SHADOWSPILL_RUNTIME_PLAN_VIOLATION) {
        result = -1;
    }
    fixture_destroy(&fixture);
    return result;
}

static int duplicate_action(void) {
    Fixture fixture = {0};
    if (fixture_create(&fixture) != 0) {
        return -1;
    }
    const ShadowSpillObjectDescription object = {
        .object_id = 1U,
        .size_bytes = 32U,
        .initially_spill_resident = 1U,
    };
    ShadowSpillAllocation allocation = {0};
    const ShadowSpillRuntimeAction actions[] = {
        {.object_id = 1U, .kind = SHADOWSPILL_RUNTIME_RELEASE},
        {.object_id = 1U, .kind = SHADOWSPILL_RUNTIME_OFFLOAD},
    };
    int result = 0;
    if (shadowspill_register_object(fixture.runtime, &object) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_allocate(
            fixture.runtime, 32U, 1U, fixture.compute, &allocation
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_bind_object(
            fixture.runtime, object.object_id, allocation.allocation_id
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_after_task(
            fixture.runtime, 1U, fixture.compute, NULL, 0U, actions, 2U
        ) != SHADOWSPILL_RUNTIME_PLAN_VIOLATION) {
        result = -1;
    }
    fixture_destroy(&fixture);
    return result;
}

static int output_allocation_handoff(void) {
    Fixture fixture = {0};
    if (fixture_create(&fixture) != 0) {
        return -1;
    }
    const ShadowSpillObjectDescription source = {
        .object_id = 1U,
        .size_bytes = 32U,
    };
    const ShadowSpillObjectDescription target = {
        .object_id = 2U,
        .size_bytes = 32U,
    };
    ShadowSpillAllocation allocation = {0};
    ShadowSpillObjectBinding input = {0};
    const uint64_t input_id = source.object_id;
    const ShadowSpillRuntimeAction release = {
        .object_id = source.object_id,
        .kind = SHADOWSPILL_RUNTIME_RELEASE,
    };
    int result = 0;
    if (shadowspill_register_object(fixture.runtime, &source) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_register_object(fixture.runtime, &target) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_allocate(
            fixture.runtime, 32U, 1U, fixture.compute, &allocation
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_bind_object(
            fixture.runtime, source.object_id, allocation.allocation_id
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_before_task(
            fixture.runtime, 9U, fixture.compute, &input_id, 1U, &input, 1U
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_bind_object(
            fixture.runtime, target.object_id, allocation.allocation_id
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_after_task(
            fixture.runtime, 9U, fixture.compute, NULL, 0U, &release, 1U
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_wait_idle(fixture.runtime) !=
            SHADOWSPILL_RUNTIME_OK) {
        result = -1;
        goto done;
    }
    ShadowSpillObjectSnapshot source_snapshot = {0};
    ShadowSpillObjectSnapshot target_snapshot = {0};
    if (shadowspill_object_snapshot(
            fixture.runtime, source.object_id, &source_snapshot
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_object_snapshot(
            fixture.runtime, target.object_id, &target_snapshot
        ) != SHADOWSPILL_RUNTIME_OK ||
        source_snapshot.residency != SHADOWSPILL_OBJECT_RELEASED ||
        source_snapshot.execution_pointer != NULL ||
        target_snapshot.residency != SHADOWSPILL_OBJECT_EXECUTION_READY ||
        target_snapshot.execution_pointer != allocation.pointer ||
        target_snapshot.allocation_id != allocation.allocation_id) {
        result = -1;
    }

done:
    fixture_destroy(&fixture);
    return result;
}

static int chained_output_allocation_handoff(void) {
    ShadowSpillMockBackend *mock = NULL;
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillBackendStream compute = {{0U, 0U}};
    const ShadowSpillMockBackendConfig backend_config = {
        .abi_version = SHADOWSPILL_BACKEND_ABI_VERSION,
        /* Keep the first release pending while the dispatcher submits the
         * second zero-copy handoff on the same ordered compute stream. */
        .event_delay_nanoseconds = 50000000U,
    };
    ShadowSpillRuntimeConfig runtime_config = {
        .abi_version = SHADOWSPILL_RUNTIME_ABI_VERSION,
        .execution_pool_bytes = 128U,
        .spill_pool_bytes = 128U,
        .minimum_alignment = 1U,
        .worker_poll_nanoseconds = 1000U,
    };
    if (shadowspill_mock_backend_create(&backend_config, &mock) != 0) {
        return -1;
    }
    runtime_config.backend = shadowspill_mock_backend_vtable(mock);
    if (shadowspill_runtime_create(&runtime_config, &runtime) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_mock_create_compute_stream(mock, &compute) != 0) {
        shadowspill_runtime_destroy(runtime);
        shadowspill_mock_backend_destroy(mock);
        return -1;
    }

    const ShadowSpillObjectDescription objects[] = {
        {.object_id = 1U, .size_bytes = 32U},
        {.object_id = 2U, .size_bytes = 32U},
        {.object_id = 3U, .size_bytes = 32U},
    };
    ShadowSpillAllocation allocation = {0};
    ShadowSpillObjectBinding binding = {0};
    const uint64_t first_input = 1U;
    const uint64_t second_input = 2U;
    const ShadowSpillRuntimeAction first_release = {
        .object_id = 1U,
        .kind = SHADOWSPILL_RUNTIME_RELEASE,
    };
    const ShadowSpillRuntimeAction second_release = {
        .object_id = 2U,
        .kind = SHADOWSPILL_RUNTIME_RELEASE,
    };
    const ShadowSpillExecutionDescription first_execution = {
        .task_id = 9U,
        .input_object_ids = &first_input,
        .input_count = 1U,
        .actions = &first_release,
        .action_count = 1U,
    };
    const ShadowSpillExecutionDescription second_execution = {
        .task_id = 10U,
        .input_object_ids = &second_input,
        .input_count = 1U,
        .actions = &second_release,
        .action_count = 1U,
    };
    int failed = 0;
    for (uint32_t index = 0U; index < 3U; ++index) {
        failed = failed || shadowspill_register_object(runtime, &objects[index]) !=
            SHADOWSPILL_RUNTIME_OK;
    }
    failed = failed || shadowspill_allocate(
            runtime, 32U, 1U, compute, &allocation
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_bind_object(
            runtime, 1U, allocation.allocation_id
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_admit_execution(
            runtime, &first_execution
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_admit_execution(
            runtime, &second_execution
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_before_execution(
            runtime, 9U, compute, &binding, 1U
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_bind_object(
            runtime, 2U, allocation.allocation_id
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_after_execution(
            runtime, 9U, compute
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_before_execution(
            runtime, 10U, compute, &binding, 1U
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_bind_object(
            runtime, 3U, allocation.allocation_id
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_after_execution(
            runtime, 10U, compute
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_runtime_wait_idle(runtime) !=
            SHADOWSPILL_RUNTIME_OK;

    ShadowSpillObjectSnapshot snapshots[3] = {{0}};
    for (uint32_t index = 0U; index < 3U; ++index) {
        failed = failed || shadowspill_object_snapshot(
                runtime, objects[index].object_id, &snapshots[index]
            ) != SHADOWSPILL_RUNTIME_OK;
    }
    failed = failed || snapshots[0].residency != SHADOWSPILL_OBJECT_RELEASED ||
        snapshots[1].residency != SHADOWSPILL_OBJECT_RELEASED ||
        snapshots[2].residency != SHADOWSPILL_OBJECT_EXECUTION_READY ||
        snapshots[2].execution_pointer != allocation.pointer ||
        snapshots[2].allocation_id != allocation.allocation_id;

    shadowspill_runtime_destroy(runtime);
    if (compute.words[0] != 0U) {
        (void)shadowspill_mock_destroy_compute_stream(mock, compute);
    }
    shadowspill_mock_backend_destroy(mock);
    return failed ? -1 : 0;
}

static int valid_transition_paths(void) {
    Fixture fixture = {0};
    if (fixture_create(&fixture) != 0) {
        return -1;
    }
    const ShadowSpillObjectDescription retained = {
        .object_id = 1U,
        .size_bytes = 32U,
        .retain_spill_copy = 1U,
        .initially_spill_resident = 1U,
    };
    const ShadowSpillObjectDescription temporary_host = {
        .object_id = 2U,
        .size_bytes = 32U,
        .initially_spill_resident = 1U,
    };
    const ShadowSpillObjectDescription device_created = {
        .object_id = 3U,
        .size_bytes = 32U,
    };
    ShadowSpillAllocation first = {0};
    ShadowSpillAllocation third = {0};
    int result = 0;
    if (shadowspill_register_object(fixture.runtime, &retained) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_register_object(fixture.runtime, &temporary_host) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_register_object(fixture.runtime, &device_created) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_allocate(
            fixture.runtime, 32U, 1U, fixture.compute, &first
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_bind_object(
            fixture.runtime, retained.object_id, first.allocation_id
        ) != SHADOWSPILL_RUNTIME_OK) {
        result = -1;
        goto done;
    }
    const ShadowSpillRuntimeAction release = {
        .object_id = retained.object_id,
        .kind = SHADOWSPILL_RUNTIME_RELEASE,
    };
    if (shadowspill_after_task(
            fixture.runtime, 1U, fixture.compute, NULL, 0U, &release, 1U
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_wait_idle(fixture.runtime) != SHADOWSPILL_RUNTIME_OK) {
        result = -1;
        goto done;
    }
    ShadowSpillObjectSnapshot snapshot = {0};
    if (shadowspill_object_snapshot(
            fixture.runtime, retained.object_id, &snapshot
        ) != SHADOWSPILL_RUNTIME_OK ||
        snapshot.residency != SHADOWSPILL_OBJECT_SPILL_ONLY ||
        !snapshot.spill_current) {
        result = -1;
        goto done;
    }
    ShadowSpillAllocation retired = {0};
    if (shadowspill_allocation_for_pointer(
            fixture.runtime, first.pointer, &retired
        ) != SHADOWSPILL_RUNTIME_OK ||
        retired.allocation_id != first.allocation_id ||
        shadowspill_free(
            fixture.runtime, first.allocation_id, fixture.compute
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_free(
            fixture.runtime, first.allocation_id, fixture.compute
        ) != SHADOWSPILL_RUNTIME_INVALID_STATE) {
        result = -1;
        goto done;
    }
    if (shadowspill_unregister_object(
            fixture.runtime, retained.object_id
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_object_snapshot(
            fixture.runtime, retained.object_id, &snapshot
        ) != SHADOWSPILL_RUNTIME_INVALID_STATE ||
        shadowspill_register_object(fixture.runtime, &retained) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_register_object(fixture.runtime, &retained) !=
            SHADOWSPILL_RUNTIME_INVALID_STATE ||
        shadowspill_object_snapshot(
            fixture.runtime, retained.object_id, &snapshot
        ) != SHADOWSPILL_RUNTIME_OK ||
        snapshot.residency != SHADOWSPILL_OBJECT_SPILL_ONLY ||
        shadowspill_unregister_object(
            fixture.runtime, retained.object_id
        ) != SHADOWSPILL_RUNTIME_OK) {
        result = -1;
        goto done;
    }
    const ShadowSpillRuntimeAction prefetch = {
        .object_id = temporary_host.object_id,
        .kind = SHADOWSPILL_RUNTIME_PREFETCH,
    };
    if (shadowspill_after_task(
            fixture.runtime, 2U, fixture.compute, NULL, 0U, &prefetch, 1U
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_wait_idle(fixture.runtime) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_object_snapshot(
            fixture.runtime, temporary_host.object_id, &snapshot
        ) != SHADOWSPILL_RUNTIME_OK ||
        snapshot.residency != SHADOWSPILL_OBJECT_EXECUTION_READY ||
        snapshot.has_spill_lease) {
        result = -1;
        goto done;
    }
    if (shadowspill_allocate(
            fixture.runtime, 32U, 1U, fixture.compute, &third
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_bind_object(
            fixture.runtime, device_created.object_id, third.allocation_id
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_object_snapshot(
            fixture.runtime, device_created.object_id, &snapshot
        ) != SHADOWSPILL_RUNTIME_OK ||
        snapshot.residency != SHADOWSPILL_OBJECT_EXECUTION_READY) {
        result = -1;
        goto done;
    }
    ShadowSpillAllocation caller = {0};
    if (shadowspill_transfer_object_to_caller(
            fixture.runtime, device_created.object_id, fixture.compute, &caller
        ) != SHADOWSPILL_RUNTIME_OK ||
        caller.allocation_id != third.allocation_id ||
        shadowspill_object_snapshot(
            fixture.runtime, device_created.object_id, &snapshot
        ) != SHADOWSPILL_RUNTIME_OK ||
        snapshot.residency != SHADOWSPILL_OBJECT_RELEASED ||
        shadowspill_register_object(fixture.runtime, &device_created) !=
            SHADOWSPILL_RUNTIME_INVALID_STATE ||
        shadowspill_object_snapshot(
            fixture.runtime, device_created.object_id, &snapshot
        ) != SHADOWSPILL_RUNTIME_OK ||
        snapshot.residency != SHADOWSPILL_OBJECT_RELEASED ||
        shadowspill_unregister_object(
            fixture.runtime, device_created.object_id
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_free(
            fixture.runtime, caller.allocation_id, fixture.compute
        ) != SHADOWSPILL_RUNTIME_OK) {
        result = -1;
    }

done:
    fixture_destroy(&fixture);
    return result;
}

static int immutable_execution_admission(void) {
    Fixture fixture = {0};
    if (fixture_create(&fixture) != 0) {
        return -1;
    }
    const ShadowSpillObjectDescription description = {
        .object_id = 91U,
        .size_bytes = 32U,
    };
    ShadowSpillAllocation allocation = {0};
    const uint64_t input = description.object_id;
    const ShadowSpillExecutionDescription execution = {
        .task_id = 17U,
        .input_object_ids = &input,
        .input_count = 1U,
    };
    const ShadowSpillExecutionDescription conflict = {
        .task_id = execution.task_id,
    };
    ShadowSpillObjectBinding binding = {0};
    const ShadowSpillExecutionHandle *handle = NULL;
    int failed = shadowspill_register_object(
            fixture.runtime, &description
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_allocate(
            fixture.runtime, description.size_bytes, 1U, fixture.compute,
            &allocation
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_bind_object(
            fixture.runtime, description.object_id, allocation.allocation_id
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_admit_execution(
            fixture.runtime, &execution
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_admit_execution(
            fixture.runtime, &execution
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_admit_execution(
            fixture.runtime, &conflict
        ) != SHADOWSPILL_RUNTIME_INVALID_STATE || shadowspill_before_execution(
            fixture.runtime, execution.task_id, fixture.compute, &binding, 1U
        ) != SHADOWSPILL_RUNTIME_OK || binding.object_id != description.object_id ||
        binding.allocation_id != allocation.allocation_id ||
        shadowspill_after_execution(
            fixture.runtime, execution.task_id, fixture.compute
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_resolve_execution(
            fixture.runtime, execution.task_id, &handle
        ) != SHADOWSPILL_RUNTIME_OK || handle == NULL ||
        shadowspill_before_execution_handle(
            fixture.runtime, handle, fixture.compute, &binding, 1U
        ) != SHADOWSPILL_RUNTIME_OK || binding.object_id != description.object_id ||
        binding.allocation_id != allocation.allocation_id ||
        shadowspill_after_execution_handle(
            fixture.runtime, handle, fixture.compute
        ) != SHADOWSPILL_RUNTIME_OK;
    fixture_destroy(&fixture);
    return failed ? -1 : 0;
}

static int caller_handoff_preserves_recurrent_object_identity(void) {
    Fixture fixture = {0};
    if (fixture_create(&fixture) != 0) {
        return -1;
    }
    const ShadowSpillObjectDescription object = {
        .object_id = 92U,
        .size_bytes = 32U,
    };
    const ShadowSpillRuntimeAction evict = {
        .object_id = object.object_id,
        .kind = SHADOWSPILL_RUNTIME_OFFLOAD,
    };
    const ShadowSpillRuntimeAction fetch = {
        .object_id = object.object_id,
        .kind = SHADOWSPILL_RUNTIME_PREFETCH,
    };
    const ShadowSpillExecutionDescription evict_task = {
        .task_id = 18U,
        .actions = &evict,
        .action_count = 1U,
    };
    const ShadowSpillExecutionDescription fetch_task = {
        .task_id = 19U,
        .actions = &fetch,
        .action_count = 1U,
    };
    int failed = shadowspill_register_object(fixture.runtime, &object) !=
            SHADOWSPILL_RUNTIME_OK || shadowspill_admit_execution(
            fixture.runtime, &evict_task
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_admit_execution(
            fixture.runtime, &fetch_task
        ) != SHADOWSPILL_RUNTIME_OK;
    for (uint32_t invocation = 0U; !failed && invocation < 2U; ++invocation) {
        ShadowSpillAllocation produced = {0};
        ShadowSpillAllocation caller = {0};
        ShadowSpillObjectSnapshot snapshot = {0};
        failed = shadowspill_allocate(
                fixture.runtime, object.size_bytes, 1U, fixture.compute, &produced
            ) != SHADOWSPILL_RUNTIME_OK || shadowspill_bind_object(
                fixture.runtime, object.object_id, produced.allocation_id
            ) != SHADOWSPILL_RUNTIME_OK || shadowspill_before_execution(
                fixture.runtime, evict_task.task_id, fixture.compute, NULL, 0U
            ) != SHADOWSPILL_RUNTIME_OK || shadowspill_after_execution(
                fixture.runtime, evict_task.task_id, fixture.compute
            ) != SHADOWSPILL_RUNTIME_OK || shadowspill_before_execution(
                fixture.runtime, fetch_task.task_id, fixture.compute, NULL, 0U
            ) != SHADOWSPILL_RUNTIME_OK || shadowspill_after_execution(
                fixture.runtime, fetch_task.task_id, fixture.compute
            ) != SHADOWSPILL_RUNTIME_OK || shadowspill_transfer_object_to_caller(
                fixture.runtime, object.object_id, fixture.compute, &caller
            ) != SHADOWSPILL_RUNTIME_OK || shadowspill_object_snapshot(
                fixture.runtime, object.object_id, &snapshot
            ) != SHADOWSPILL_RUNTIME_OK ||
            snapshot.residency != SHADOWSPILL_OBJECT_RELEASED ||
            snapshot.execution_pointer != NULL ||
            caller.allocation_id == SHADOWSPILL_RUNTIME_NO_ID ||
            shadowspill_free(
                fixture.runtime, caller.allocation_id, fixture.compute
            ) != SHADOWSPILL_RUNTIME_OK || shadowspill_runtime_wait_idle(
                fixture.runtime
            ) != SHADOWSPILL_RUNTIME_OK;
    }
    ShadowSpillRuntimeStatistics statistics = {0};
    failed = failed || shadowspill_runtime_statistics(
            fixture.runtime, &statistics
        ) != SHADOWSPILL_RUNTIME_OK || statistics.evict_transfers != 2U ||
        statistics.fetch_transfers != 2U;
    fixture_destroy(&fixture);
    return failed ? -1 : 0;
}

static int admitted_task_allocates_dynamic_ranges(void) {
    Fixture fixture = {0};
    if (fixture_create(&fixture) != 0) {
        return -1;
    }
    const ShadowSpillExecutionDescription execution = {
        .task_id = 51U,
        .maximum_requested_allocation_bytes = 40U,
        .maximum_charged_allocation_bytes = 40U,
        .live_requested_allocation_limit_bytes = 112U,
        .live_charged_allocation_limit_bytes = 112U,
    };
    ShadowSpillAllocation first_workspace = {0};
    ShadowSpillAllocation output = {0};
    ShadowSpillAllocation second_workspace = {0};
    int failed = shadowspill_admit_execution(
            fixture.runtime, &execution
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_before_execution(
            fixture.runtime, execution.task_id, fixture.compute, NULL, 0U
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_allocate(
            fixture.runtime, 40U, 1U, fixture.compute, &first_workspace
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_allocate(
            fixture.runtime, 32U, 1U, fixture.compute, &output
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_allocate(
            fixture.runtime, 40U, 1U, fixture.compute, &second_workspace
        ) != SHADOWSPILL_RUNTIME_OK ||
        (uintptr_t)first_workspace.pointer >= (uintptr_t)output.pointer ||
        (uintptr_t)output.pointer >= (uintptr_t)second_workspace.pointer ||
        shadowspill_after_execution(
            fixture.runtime, execution.task_id, fixture.compute
        ) != SHADOWSPILL_RUNTIME_OK;
    fixture_destroy(&fixture);
    return failed ? -1 : 0;
}

static int admitted_reuse_reacquires_retired_dynamic_range(void) {
    Fixture fixture = {0};
    if (fixture_create(&fixture) != 0) {
        return -1;
    }
    const ShadowSpillExecutionDescription first_execution = {
        .task_id = 61U,
        .maximum_requested_allocation_bytes = 64U,
        .maximum_charged_allocation_bytes = 64U,
        .live_requested_allocation_limit_bytes = 64U,
        .live_charged_allocation_limit_bytes = 64U,
    };
    const ShadowSpillExecutionDescription reuse_execution = {
        .task_id = 62U,
        .maximum_requested_allocation_bytes = 64U,
        .maximum_charged_allocation_bytes = 64U,
        .live_requested_allocation_limit_bytes = 64U,
        .live_charged_allocation_limit_bytes = 64U,
    };
    ShadowSpillAllocation first = {0};
    ShadowSpillAllocation reacquired = {0};
    ShadowSpillRuntimeStatistics retired = {0};
    int failed = shadowspill_admit_execution(
            fixture.runtime, &first_execution
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_admit_execution(
            fixture.runtime, &reuse_execution
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_before_execution(
            fixture.runtime, first_execution.task_id, fixture.compute, NULL, 0U
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_allocate(
            fixture.runtime, 64U, 1U, fixture.compute, &first
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_free(
            fixture.runtime, first.allocation_id, fixture.compute
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_after_execution(
            fixture.runtime, first_execution.task_id, fixture.compute
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_runtime_wait_idle(
            fixture.runtime
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_runtime_statistics(
            fixture.runtime, &retired
        ) != SHADOWSPILL_RUNTIME_OK || retired.allocated_bytes != 0U ||
        retired.pending_retirements != 0U || shadowspill_before_execution(
            fixture.runtime, reuse_execution.task_id, fixture.compute, NULL, 0U
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_allocate(
            fixture.runtime, 64U, 1U, fixture.compute, &reacquired
        ) != SHADOWSPILL_RUNTIME_OK || reacquired.pointer != first.pointer ||
        reacquired.allocation_id == first.allocation_id || shadowspill_free(
            fixture.runtime, reacquired.allocation_id, fixture.compute
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_after_execution(
            fixture.runtime, reuse_execution.task_id, fixture.compute
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_runtime_wait_idle(
            fixture.runtime
        ) != SHADOWSPILL_RUNTIME_OK;
    fixture_destroy(&fixture);
    return failed ? -1 : 0;
}

static int admitted_allocation_without_progress_reports_no_progress(void) {
    Fixture fixture = {0};
    if (fixture_create(&fixture) != 0) {
        return -1;
    }
    const ShadowSpillExecutionDescription first_execution = {
        .task_id = 63U,
        .maximum_requested_allocation_bytes = 128U,
        .maximum_charged_allocation_bytes = 128U,
        .live_requested_allocation_limit_bytes = 128U,
        .live_charged_allocation_limit_bytes = 128U,
    };
    const ShadowSpillExecutionDescription reuse_execution = {
        .task_id = 64U,
        .maximum_requested_allocation_bytes = 64U,
        .maximum_charged_allocation_bytes = 64U,
        .live_requested_allocation_limit_bytes = 64U,
        .live_charged_allocation_limit_bytes = 64U,
    };
    ShadowSpillAllocation live = {0};
    ShadowSpillAllocation blocked = {0};
    ShadowSpillRuntimeFailure failure = {0};
    int failed = shadowspill_admit_execution(
            fixture.runtime, &first_execution
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_admit_execution(
            fixture.runtime, &reuse_execution
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_before_execution(
            fixture.runtime, first_execution.task_id, fixture.compute, NULL, 0U
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_allocate(
            fixture.runtime, 128U, 1U, fixture.compute, &live
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_after_execution(
            fixture.runtime, first_execution.task_id, fixture.compute
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_before_execution(
            fixture.runtime, reuse_execution.task_id, fixture.compute, NULL, 0U
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_allocate(
            fixture.runtime, 64U, 1U, fixture.compute, &blocked
        ) != SHADOWSPILL_RUNTIME_NO_PROGRESS || shadowspill_runtime_failure(
            fixture.runtime, &failure
        ) != SHADOWSPILL_RUNTIME_OK ||
        failure.status != SHADOWSPILL_RUNTIME_NO_PROGRESS ||
        failure.task_id != reuse_execution.task_id ||
        failure.requested_bytes != 64U;
    shadowspill_abort_task(fixture.runtime);
    fixture_destroy(&fixture);
    return failed ? -1 : 0;
}

static int admitted_dynamic_allocations_use_deterministic_low_ranges(void) {
    Fixture fixture = {0};
    if (fixture_create(&fixture) != 0) {
        return -1;
    }
    const ShadowSpillExecutionDescription execution = {
        .task_id = 53U,
        .maximum_requested_allocation_bytes = 32U,
        .maximum_charged_allocation_bytes = 32U,
        .live_requested_allocation_limit_bytes = 64U,
        .live_charged_allocation_limit_bytes = 64U,
    };
    ShadowSpillAllocation dynamic = {0};
    ShadowSpillAllocation exact = {0};
    int failed = shadowspill_admit_execution(
            fixture.runtime, &execution
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_before_execution(
            fixture.runtime, execution.task_id, fixture.compute, NULL, 0U
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_allocate(
            fixture.runtime, 32U, 1U, fixture.compute, &dynamic
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_allocate(
            fixture.runtime, 32U, 1U, fixture.compute, &exact
        ) != SHADOWSPILL_RUNTIME_OK ||
        (uintptr_t)exact.pointer != (uintptr_t)dynamic.pointer + 32U ||
        shadowspill_after_execution(
            fixture.runtime, execution.task_id, fixture.compute
        ) != SHADOWSPILL_RUNTIME_OK;
    fixture_destroy(&fixture);
    return failed ? -1 : 0;
}

static int admitted_task_rejects_envelope_excess(void) {
    Fixture fixture = {0};
    if (fixture_create(&fixture) != 0) {
        return -1;
    }
    const ShadowSpillExecutionDescription execution = {
        .task_id = 54U,
        .maximum_requested_allocation_bytes = 32U,
        .maximum_charged_allocation_bytes = 32U,
        .live_requested_allocation_limit_bytes = 32U,
        .live_charged_allocation_limit_bytes = 32U,
    };
    ShadowSpillAllocation expected = {0};
    ShadowSpillAllocation unexpected = {0};
    ShadowSpillRuntimeFailure failure = {0};
    const int failed = shadowspill_admit_execution(
            fixture.runtime, &execution
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_before_execution(
            fixture.runtime, execution.task_id, fixture.compute, NULL, 0U
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_allocate(
            fixture.runtime, 32U, 1U, fixture.compute, &expected
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_allocate(
            fixture.runtime, 16U, 1U, fixture.compute, &unexpected
        ) != SHADOWSPILL_RUNTIME_TASK_ALLOCATION_ENVELOPE_EXCEEDED ||
        shadowspill_runtime_failure(
            fixture.runtime, &failure
        ) != SHADOWSPILL_RUNTIME_OK ||
        failure.status !=
            SHADOWSPILL_RUNTIME_TASK_ALLOCATION_ENVELOPE_EXCEEDED ||
        failure.task_id != execution.task_id ||
        failure.requested_bytes != 16U ||
        failure.task_live_requested_bytes != 48U ||
        failure.task_live_charged_bytes != 48U ||
        failure.task_live_requested_limit_bytes != 32U ||
        failure.task_live_charged_limit_bytes != 32U;
    fixture_destroy(&fixture);
    return failed ? -1 : 0;
}

static int admitted_task_accepts_exact_allocation_abi(void) {
    Fixture fixture = {0};
    if (fixture_create(&fixture) != 0) {
        return -1;
    }
    const ShadowSpillTaskAllocationABIStep steps[] = {
        {
            .allocation_ordinal = 0U,
            .requested_bytes = 32U,
            .charged_bytes = 32U,
            .alignment_bytes = 1U,
            .operation = SHADOWSPILL_TASK_ALLOCATION_ALLOCATE,
        },
        {
            .allocation_ordinal = 0U,
            .requested_bytes = 32U,
            .charged_bytes = 32U,
            .alignment_bytes = 1U,
            .operation = SHADOWSPILL_TASK_ALLOCATION_FREE,
        },
    };
    const ShadowSpillExecutionDescription execution = {
        .task_id = 55U,
        .allocation_abi_steps = steps,
        .allocation_abi_step_count = 2U,
        .enforce_allocation_abi = 1U,
        .maximum_requested_allocation_bytes = 32U,
        .maximum_charged_allocation_bytes = 32U,
        .live_requested_allocation_limit_bytes = 32U,
        .live_charged_allocation_limit_bytes = 32U,
    };
    ShadowSpillAllocation allocation = {0};
    const ShadowSpillRuntimeStatus admit_status = shadowspill_admit_execution(
        fixture.runtime, &execution
    );
    const ShadowSpillRuntimeStatus before_status = shadowspill_before_execution(
        fixture.runtime, execution.task_id, fixture.compute, NULL, 0U
    );
    const ShadowSpillRuntimeStatus allocate_status = shadowspill_allocate(
        fixture.runtime, 32U, 1U, fixture.compute, &allocation
    );
    const ShadowSpillRuntimeStatus free_status = shadowspill_free(
        fixture.runtime, allocation.allocation_id, fixture.compute
    );
    const ShadowSpillRuntimeStatus after_status = shadowspill_after_execution(
        fixture.runtime, execution.task_id, fixture.compute
    );
    const ShadowSpillRuntimeStatus idle_status = shadowspill_runtime_wait_idle(
        fixture.runtime
    );
    const int failed = admit_status != SHADOWSPILL_RUNTIME_OK ||
        before_status != SHADOWSPILL_RUNTIME_OK ||
        allocate_status != SHADOWSPILL_RUNTIME_OK ||
        free_status != SHADOWSPILL_RUNTIME_OK ||
        after_status != SHADOWSPILL_RUNTIME_OK ||
        idle_status != SHADOWSPILL_RUNTIME_OK;
    if (failed) {
        fprintf(
            stderr,
            "allocation ABI exact mismatch: admit=%u before=%u allocate=%u free=%u after=%u idle=%u\n",
            (unsigned)admit_status,
            (unsigned)before_status,
            (unsigned)allocate_status,
            (unsigned)free_status,
            (unsigned)after_status,
            (unsigned)idle_status
        );
    }
    fixture_destroy(&fixture);
    return failed ? -1 : 0;
}

static int delayed_free_is_not_charged_to_later_task_invocation(void) {
    Fixture fixture = {0};
    if (fixture_create(&fixture) != 0) {
        return -1;
    }
    const ShadowSpillTaskAllocationABIStep step = {
        .allocation_ordinal = 0U,
        .requested_bytes = 32U,
        .charged_bytes = 32U,
        .alignment_bytes = 1U,
        .operation = SHADOWSPILL_TASK_ALLOCATION_ALLOCATE,
    };
    const ShadowSpillExecutionDescription execution = {
        .task_id = 56U,
        .allocation_abi_steps = &step,
        .allocation_abi_step_count = 1U,
        .enforce_allocation_abi = 1U,
        .maximum_requested_allocation_bytes = 32U,
        .maximum_charged_allocation_bytes = 32U,
        .live_requested_allocation_limit_bytes = 32U,
        .live_charged_allocation_limit_bytes = 32U,
    };
    ShadowSpillAllocation first = {0};
    ShadowSpillAllocation second = {0};
    int failed = shadowspill_admit_execution(
            fixture.runtime, &execution
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_before_execution(
            fixture.runtime, execution.task_id, fixture.compute, NULL, 0U
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_allocate(
            fixture.runtime, 32U, 1U, fixture.compute, &first
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_after_execution(
            fixture.runtime, execution.task_id, fixture.compute
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_before_execution(
            fixture.runtime, execution.task_id, fixture.compute, NULL, 0U
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_allocate(
            fixture.runtime, 32U, 1U, fixture.compute, &second
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_free(
            fixture.runtime, first.allocation_id, fixture.compute
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_after_execution(
            fixture.runtime, execution.task_id, fixture.compute
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_free(
            fixture.runtime, second.allocation_id, fixture.compute
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_wait_idle(fixture.runtime) != SHADOWSPILL_RUNTIME_OK;
    fixture_destroy(&fixture);
    return failed ? -1 : 0;
}

static int admitted_task_rejects_allocation_abi_geometry_mismatch(void) {
    Fixture fixture = {0};
    if (fixture_create(&fixture) != 0) {
        return -1;
    }
    const ShadowSpillTaskAllocationABIStep step = {
        .allocation_ordinal = 0U,
        .requested_bytes = 32U,
        .charged_bytes = 32U,
        .alignment_bytes = 1U,
        .operation = SHADOWSPILL_TASK_ALLOCATION_ALLOCATE,
    };
    const ShadowSpillExecutionDescription execution = {
        .task_id = 56U,
        .allocation_abi_steps = &step,
        .allocation_abi_step_count = 1U,
        .enforce_allocation_abi = 1U,
        .maximum_requested_allocation_bytes = 64U,
        .maximum_charged_allocation_bytes = 64U,
        .live_requested_allocation_limit_bytes = 64U,
        .live_charged_allocation_limit_bytes = 64U,
    };
    ShadowSpillAllocation allocation = {0};
    ShadowSpillRuntimeFailure failure = {0};
    const int failed = shadowspill_admit_execution(
            fixture.runtime, &execution
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_before_execution(
            fixture.runtime, execution.task_id, fixture.compute, NULL, 0U
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_allocate(
            fixture.runtime, 16U, 1U, fixture.compute, &allocation
        ) != SHADOWSPILL_RUNTIME_TASK_ALLOCATION_ABI_MISMATCH ||
        shadowspill_runtime_failure(
            fixture.runtime, &failure
        ) != SHADOWSPILL_RUNTIME_OK ||
        failure.status != SHADOWSPILL_RUNTIME_TASK_ALLOCATION_ABI_MISMATCH ||
        failure.task_id != execution.task_id ||
        failure.task_allocation_operation_index != 0U ||
        failure.task_allocation_expected_ordinal != 0U ||
        failure.task_allocation_actual_ordinal != 0U ||
        failure.task_allocation_expected_requested_bytes != 32U ||
        failure.task_allocation_actual_requested_bytes != 16U ||
        failure.task_allocation_expected_operation !=
            SHADOWSPILL_TASK_ALLOCATION_ALLOCATE ||
        failure.task_allocation_actual_operation !=
            SHADOWSPILL_TASK_ALLOCATION_ALLOCATE;
    shadowspill_abort_task(fixture.runtime);
    fixture_destroy(&fixture);
    return failed ? -1 : 0;
}

static int admitted_task_rejects_incomplete_allocation_abi(void) {
    Fixture fixture = {0};
    if (fixture_create(&fixture) != 0) {
        return -1;
    }
    const ShadowSpillTaskAllocationABIStep steps[] = {
        {
            .allocation_ordinal = 0U,
            .requested_bytes = 32U,
            .charged_bytes = 32U,
            .alignment_bytes = 1U,
            .operation = SHADOWSPILL_TASK_ALLOCATION_ALLOCATE,
        },
        {
            .allocation_ordinal = 0U,
            .requested_bytes = 32U,
            .charged_bytes = 32U,
            .alignment_bytes = 1U,
            .operation = SHADOWSPILL_TASK_ALLOCATION_FREE,
        },
    };
    const ShadowSpillExecutionDescription execution = {
        .task_id = 57U,
        .allocation_abi_steps = steps,
        .allocation_abi_step_count = 2U,
        .enforce_allocation_abi = 1U,
        .maximum_requested_allocation_bytes = 32U,
        .maximum_charged_allocation_bytes = 32U,
        .live_requested_allocation_limit_bytes = 32U,
        .live_charged_allocation_limit_bytes = 32U,
    };
    ShadowSpillAllocation allocation = {0};
    ShadowSpillRuntimeFailure failure = {0};
    const int failed = shadowspill_admit_execution(
            fixture.runtime, &execution
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_before_execution(
            fixture.runtime, execution.task_id, fixture.compute, NULL, 0U
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_allocate(
            fixture.runtime, 32U, 1U, fixture.compute, &allocation
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_after_execution(
            fixture.runtime, execution.task_id, fixture.compute
        ) != SHADOWSPILL_RUNTIME_TASK_ALLOCATION_ABI_MISMATCH ||
        shadowspill_runtime_failure(
            fixture.runtime, &failure
        ) != SHADOWSPILL_RUNTIME_OK ||
        failure.status != SHADOWSPILL_RUNTIME_TASK_ALLOCATION_ABI_MISMATCH ||
        failure.task_allocation_operation_index != 1U ||
        failure.task_allocation_expected_operation !=
            SHADOWSPILL_TASK_ALLOCATION_FREE ||
        failure.task_allocation_actual_operation != UINT8_MAX;
    fixture_destroy(&fixture);
    return failed ? -1 : 0;
}

static int admitted_prefetch_reserves_dynamic_capacity(void) {
    Fixture fixture = {0};
    if (fixture_create(&fixture) != 0) {
        return -1;
    }
    const ShadowSpillObjectDescription description = {
        .object_id = 52U,
        .size_bytes = 32U,
        .initially_spill_resident = 1U,
    };
    const ShadowSpillRuntimeAction prefetch = {
        .object_id = description.object_id,
        .kind = SHADOWSPILL_RUNTIME_PREFETCH,
    };
    ShadowSpillObjectSnapshot snapshot = {0};
    ShadowSpillAllocation following = {0};
    int failed = shadowspill_register_object(
            fixture.runtime, &description
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_after_task(
            fixture.runtime, 52U, fixture.compute, NULL, 0U, &prefetch, 1U
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_runtime_wait_idle(
            fixture.runtime
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_object_snapshot(
            fixture.runtime, description.object_id, &snapshot
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_allocate(
            fixture.runtime, 32U, 1U, fixture.compute, &following
        ) != SHADOWSPILL_RUNTIME_OK ||
        (uintptr_t)following.pointer !=
            (uintptr_t)snapshot.execution_pointer + 32U;
    fixture_destroy(&fixture);
    return failed ? -1 : 0;
}

static int execution_plan_lifecycle(void) {
    Fixture fixture = {0};
    if (fixture_create(&fixture) != 0) {
        return -1;
    }
    const ShadowSpillObjectDescription object = {
        .object_id = 92U,
        .size_bytes = 32U,
    };
    const uint64_t input = object.object_id;
    const ShadowSpillExecutionDescription first_plan = {
        .task_id = 18U,
        .input_object_ids = &input,
        .input_count = 1U,
    };
    const ShadowSpillExecutionDescription second_plan = {
        .task_id = first_plan.task_id,
    };
    const ShadowSpillExecutionHandle *handle = NULL;
    int failed = shadowspill_register_object(fixture.runtime, &object) !=
            SHADOWSPILL_RUNTIME_OK || shadowspill_admit_execution(
            fixture.runtime, &first_plan
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_clear_execution_plan(
            fixture.runtime
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_resolve_execution(
            fixture.runtime, first_plan.task_id, &handle
        ) != SHADOWSPILL_RUNTIME_INVALID_STATE || shadowspill_unregister_object(
            fixture.runtime, object.object_id
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_admit_execution(
            fixture.runtime, &second_plan
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_clear_execution_plan(
            fixture.runtime
        ) != SHADOWSPILL_RUNTIME_OK;
    fixture_destroy(&fixture);
    return failed ? -1 : 0;
}

static int functional_mutation_replaces_lease_without_copy(void) {
    Fixture fixture = {0};
    if (fixture_create(&fixture) != 0) {
        return -1;
    }
    const ShadowSpillObjectDescription description = {
        .object_id = 101U,
        .size_bytes = 32U,
        .initial_version = 7U,
    };
    ShadowSpillAllocation prior = {0};
    ShadowSpillAllocation replacement = {0};
    ShadowSpillAllocation probe = {0};
    const uint64_t input = description.object_id;
    const ShadowSpillObjectUpdate update = {
        .object_id = description.object_id,
        .version_delta = 1U,
    };
    const ShadowSpillExecutionDescription execution = {
        .task_id = 41U,
        .input_object_ids = &input,
        .input_count = 1U,
        .updates = &update,
        .update_count = 1U,
        .maximum_requested_allocation_bytes = 32U,
        .maximum_charged_allocation_bytes = 32U,
        .live_requested_allocation_limit_bytes = 64U,
        .live_charged_allocation_limit_bytes = 64U,
    };
    ShadowSpillObjectBinding acquired = {0};
    ShadowSpillObjectBinding replaced = {0};
    ShadowSpillObjectSnapshot snapshot = {0};
    ShadowSpillRuntimeStatistics statistics = {0};
    int failed = shadowspill_register_object(
            fixture.runtime, &description
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_allocate(
            fixture.runtime, description.size_bytes, 1U, fixture.compute, &prior
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_bind_object(
            fixture.runtime, description.object_id, prior.allocation_id
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_admit_execution(
            fixture.runtime, &execution
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_before_execution(
            fixture.runtime, execution.task_id, fixture.compute, &acquired, 1U
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_allocate(
            fixture.runtime, description.size_bytes, 1U, fixture.compute,
            &replacement
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_replace_object_allocation(
            fixture.runtime,
            description.object_id,
            replacement.allocation_id,
            &replaced
        ) != SHADOWSPILL_RUNTIME_OK || replaced.pointer != replacement.pointer ||
        replaced.generation != replacement.generation ||
        replaced.authoritative_version != 7U || shadowspill_runtime_statistics(
            fixture.runtime, &statistics
        ) != SHADOWSPILL_RUNTIME_OK || statistics.allocated_bytes != 64U ||
        statistics.pending_retirements != 1U || shadowspill_allocate(
            fixture.runtime, description.size_bytes, 1U, fixture.compute, &probe
        ) != SHADOWSPILL_RUNTIME_OK || probe.pointer == prior.pointer ||
        probe.pointer == replacement.pointer || shadowspill_free(
            fixture.runtime, probe.allocation_id, fixture.compute
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_mock_enqueue_compute(
            fixture.mock, fixture.compute, 100000U
        ) != 0 || shadowspill_after_execution(
            fixture.runtime, execution.task_id, fixture.compute
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_runtime_wait_idle(
            fixture.runtime
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_object_snapshot(
            fixture.runtime, description.object_id, &snapshot
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_runtime_statistics(
            fixture.runtime, &statistics
        ) != SHADOWSPILL_RUNTIME_OK ||
        snapshot.execution_pointer != replacement.pointer ||
        snapshot.generation != replacement.generation ||
        snapshot.retired_generation != prior.generation ||
        snapshot.retired_execution_pointer != prior.pointer ||
        snapshot.authoritative_version != 8U ||
        statistics.allocated_bytes != description.size_bytes ||
        statistics.pending_retirements != 0U;
    fixture_destroy(&fixture);
    return failed ? -1 : 0;
}

static int functional_mutation_supersedes_inflight_prefetch(void) {
    ShadowSpillMockBackend *mock = NULL;
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillBackendStream compute = {{0U, 0U}};
    const ShadowSpillMockBackendConfig backend_config = {
        .abi_version = SHADOWSPILL_BACKEND_ABI_VERSION,
        .fetch_delay_nanoseconds = 100000000U,
        .event_delay_nanoseconds = 50000000U,
    };
    if (shadowspill_mock_backend_create(&backend_config, &mock) != 0) {
        return -1;
    }
    ShadowSpillRuntimeConfig runtime_config = {
        .abi_version = SHADOWSPILL_RUNTIME_ABI_VERSION,
        .execution_pool_bytes = 96U,
        .spill_pool_bytes = 96U,
        .minimum_alignment = 1U,
        .worker_poll_nanoseconds = 1000U,
        .backend = shadowspill_mock_backend_vtable(mock),
    };
    if (shadowspill_runtime_create(&runtime_config, &runtime) !=
            SHADOWSPILL_RUNTIME_OK || shadowspill_mock_create_compute_stream(
            mock, &compute
        ) != 0) {
        shadowspill_runtime_destroy(runtime);
        shadowspill_mock_backend_destroy(mock);
        return -1;
    }

    const ShadowSpillObjectDescription description = {
        .object_id = 102U,
        .size_bytes = 32U,
        .initial_version = 3U,
        .retain_spill_copy = 1U,
        .initially_spill_resident = 1U,
    };
    const ShadowSpillRuntimeAction fetch = {
        .object_id = description.object_id,
        .kind = SHADOWSPILL_RUNTIME_PREFETCH,
    };
    const uint64_t input = description.object_id;
    const ShadowSpillObjectUpdate update = {
        .object_id = description.object_id,
        .version_delta = 1U,
    };
    const ShadowSpillExecutionDescription execution = {
        .task_id = 42U,
        .input_object_ids = &input,
        .input_count = 1U,
        .updates = &update,
        .update_count = 1U,
        .maximum_requested_allocation_bytes = 32U,
        .maximum_charged_allocation_bytes = 32U,
        .live_requested_allocation_limit_bytes = 32U,
        .live_charged_allocation_limit_bytes = 32U,
    };
    ShadowSpillObjectSnapshot during_fetch = {0};
    ShadowSpillObjectSnapshot completed = {0};
    ShadowSpillObjectBinding acquired = {0};
    ShadowSpillObjectBinding replaced = {0};
    ShadowSpillAllocation replacement = {0};
    ShadowSpillRuntimeStatistics statistics = {0};
    int failed = shadowspill_register_object(runtime, &description) !=
            SHADOWSPILL_RUNTIME_OK || shadowspill_after_task(
            runtime, 1U, compute, NULL, 0U, &fetch, 1U
        ) != SHADOWSPILL_RUNTIME_OK;
    sleep_milliseconds(75U);
    failed = failed || shadowspill_object_snapshot(
            runtime, description.object_id, &during_fetch
        ) != SHADOWSPILL_RUNTIME_OK ||
        during_fetch.residency != SHADOWSPILL_OBJECT_PREFETCHING ||
        shadowspill_admit_execution(runtime, &execution) !=
            SHADOWSPILL_RUNTIME_OK || shadowspill_before_execution(
            runtime, execution.task_id, compute, &acquired, 1U
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_allocate(
            runtime, description.size_bytes, 1U, compute, &replacement
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_replace_object_allocation(
            runtime,
            description.object_id,
            replacement.allocation_id,
            &replaced
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_mock_enqueue_compute(
            mock, compute, 1000000U
        ) != 0 || shadowspill_after_execution(
            runtime, execution.task_id, compute
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_runtime_wait_idle(runtime) !=
            SHADOWSPILL_RUNTIME_OK || shadowspill_object_snapshot(
            runtime, description.object_id, &completed
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_runtime_statistics(
            runtime, &statistics
        ) != SHADOWSPILL_RUNTIME_OK ||
        completed.execution_pointer != replacement.pointer ||
        completed.generation != replacement.generation ||
        completed.authoritative_version != 4U ||
        statistics.allocated_bytes != description.size_bytes ||
        statistics.pending_retirements != 0U ||
        statistics.wait_events_inserted != 1U;

    shadowspill_runtime_destroy(runtime);
    if (compute.words[0] != 0U) {
        (void)shadowspill_mock_destroy_compute_stream(mock, compute);
    }
    shadowspill_mock_backend_destroy(mock);
    return failed ? -1 : 0;
}

static int queued_release_causally_precedes_fetch(void) {
    ShadowSpillMockBackend *mock = NULL;
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillBackendStream compute = {{0U, 0U}};
    const ShadowSpillMockBackendConfig backend_config = {
        .abi_version = SHADOWSPILL_BACKEND_ABI_VERSION,
        .fetch_delay_nanoseconds = 1000000U,
        .event_delay_nanoseconds = 50000000U,
    };
    if (shadowspill_mock_backend_create(&backend_config, &mock) != 0) {
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
    if (shadowspill_runtime_create(&runtime_config, &runtime) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_mock_create_compute_stream(mock, &compute) != 0) {
        shadowspill_runtime_destroy(runtime);
        shadowspill_mock_backend_destroy(mock);
        return -1;
    }

    const ShadowSpillObjectDescription description = {
        .object_id = 103U,
        .size_bytes = 32U,
        .initial_version = 7U,
        .retain_spill_copy = 1U,
        .initially_spill_resident = 1U,
    };
    const ShadowSpillRuntimeAction release = {
        .object_id = description.object_id,
        .kind = SHADOWSPILL_RUNTIME_RELEASE,
    };
    const ShadowSpillRuntimeAction fetch = {
        .object_id = description.object_id,
        .kind = SHADOWSPILL_RUNTIME_PREFETCH,
    };
    const uint64_t input = description.object_id;
    const ShadowSpillExecutionDescription consumer = {
        .task_id = 3U,
        .input_object_ids = &input,
        .input_count = 1U,
    };
    ShadowSpillAllocation allocation = {0};
    ShadowSpillObjectBinding binding = {0};
    ShadowSpillObjectSnapshot completed = {0};
    ShadowSpillRuntimeStatistics statistics = {0};
    int failed = shadowspill_register_object(runtime, &description) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_allocate(
            runtime, description.size_bytes, 1U, compute, &allocation
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_bind_object(
            runtime, description.object_id, allocation.allocation_id
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_admit_execution(runtime, &consumer) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_mock_enqueue_compute(mock, compute, 100000000U) != 0 ||
        shadowspill_after_task(
            runtime, 1U, compute, NULL, 0U, &release, 1U
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_after_task(
            runtime, 2U, compute, NULL, 0U, &fetch, 1U
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_before_execution(
            runtime, consumer.task_id, compute, &binding, 1U
        ) != SHADOWSPILL_RUNTIME_OK ||
        binding.allocation_id == allocation.allocation_id ||
        shadowspill_after_execution(runtime, consumer.task_id, compute) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_wait_idle(runtime) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_object_snapshot(
            runtime, description.object_id, &completed
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_statistics(runtime, &statistics) !=
            SHADOWSPILL_RUNTIME_OK ||
        completed.residency != SHADOWSPILL_OBJECT_EXECUTION_READY ||
        completed.execution_pointer == NULL ||
        completed.authoritative_version != description.initial_version ||
        statistics.fetch_transfers != 1U;

    if (failed) {
        ShadowSpillRuntimeFailure failure = {0};
        (void)shadowspill_runtime_failure(runtime, &failure);
        fprintf(
            stderr,
            "queued release/fetch mismatch: status=%u object=%llu "
            "allocation=%llu binding=%llu residency=%u execution=%p "
            "version=%llu fetches=%llu\n",
            failure.status,
            (unsigned long long)failure.object_id,
            (unsigned long long)failure.allocation_id,
            (unsigned long long)binding.allocation_id,
            (unsigned)completed.residency,
            completed.execution_pointer,
            (unsigned long long)completed.authoritative_version,
            (unsigned long long)statistics.fetch_transfers
        );
    }

    shadowspill_runtime_destroy(runtime);
    if (compute.words[0] != 0U) {
        (void)shadowspill_mock_destroy_compute_stream(mock, compute);
    }
    shadowspill_mock_backend_destroy(mock);
    return failed ? -1 : 0;
}

static int nonretained_fetch_then_offload_reserves_fresh_spill(void) {
    ShadowSpillMockBackend *mock = NULL;
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillBackendStream compute = {{0U, 0U}};
    const ShadowSpillMockBackendConfig backend_config = {
        .abi_version = SHADOWSPILL_BACKEND_ABI_VERSION,
        .fetch_delay_nanoseconds = 100000000U,
        .evict_delay_nanoseconds = 1000000U,
        .event_delay_nanoseconds = 10000000U,
    };
    const ShadowSpillRuntimeConfig runtime_config = {
        .abi_version = SHADOWSPILL_RUNTIME_ABI_VERSION,
        .execution_pool_bytes = 128U,
        .spill_pool_bytes = 128U,
        .minimum_alignment = 1U,
        .worker_poll_nanoseconds = 1000U,
    };
    if (shadowspill_mock_backend_create(&backend_config, &mock) != 0) {
        return -1;
    }
    ShadowSpillRuntimeConfig configured = runtime_config;
    configured.backend = shadowspill_mock_backend_vtable(mock);
    if (shadowspill_runtime_create(&configured, &runtime) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_mock_create_compute_stream(mock, &compute) != 0) {
        shadowspill_runtime_destroy(runtime);
        shadowspill_mock_backend_destroy(mock);
        return -1;
    }

    const ShadowSpillObjectDescription description = {
        .object_id = 104U,
        .size_bytes = 32U,
        .initial_version = 1U,
        .retain_spill_copy = 0U,
        .initially_spill_resident = 1U,
    };
    const ShadowSpillRuntimeAction fetch = {
        .object_id = description.object_id,
        .kind = SHADOWSPILL_RUNTIME_PREFETCH,
    };
    const ShadowSpillRuntimeAction offload = {
        .object_id = description.object_id,
        .kind = SHADOWSPILL_RUNTIME_OFFLOAD,
    };
    const ShadowSpillObjectUpdate update = {
        .object_id = description.object_id,
        .version_delta = 1U,
    };
    const uint64_t input = description.object_id;
    ShadowSpillObjectBinding binding = {0};
    ShadowSpillObjectSnapshot snapshot = {0};
    ShadowSpillRuntimeStatistics statistics = {0};
    int failed = shadowspill_register_object(runtime, &description) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_after_task(
            runtime, 1U, compute, NULL, 0U, &fetch, 1U
        ) != SHADOWSPILL_RUNTIME_OK;

    for (uint32_t attempt = 0U; !failed && attempt < 500U; ++attempt) {
        failed = shadowspill_object_snapshot(
            runtime, description.object_id, &snapshot
        ) != SHADOWSPILL_RUNTIME_OK;
        if (!failed && snapshot.residency == SHADOWSPILL_OBJECT_PREFETCHING) {
            break;
        }
        sleep_milliseconds(1U);
    }
    failed = failed || snapshot.residency != SHADOWSPILL_OBJECT_PREFETCHING ||
        shadowspill_before_task(
            runtime, 2U, compute, &input, 1U, &binding, 1U
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_mock_enqueue_compute(mock, compute, 1000000U) != 0 ||
        shadowspill_after_task(
            runtime, 2U, compute, &update, 1U, &offload, 1U
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_wait_idle(runtime) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_object_snapshot(
            runtime, description.object_id, &snapshot
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_statistics(runtime, &statistics) !=
            SHADOWSPILL_RUNTIME_OK ||
        snapshot.residency != SHADOWSPILL_OBJECT_SPILL_ONLY ||
        !snapshot.spill_current || !snapshot.has_spill_lease ||
        snapshot.authoritative_version != 2U ||
        statistics.fetch_transfers != 1U ||
        statistics.evict_transfers != 1U ||
        statistics.allocated_bytes != 0U ||
        statistics.spill_allocated_bytes != description.size_bytes;

    if (failed) {
        fprintf(
            stderr,
            "nonretained fetch/offload mismatch: residency=%u spill=%u "
            "lease=%u version=%llu fetches=%llu evicts=%llu "
            "execution_bytes=%llu spill_bytes=%llu\n",
            (unsigned)snapshot.residency,
            (unsigned)snapshot.spill_current,
            (unsigned)snapshot.has_spill_lease,
            (unsigned long long)snapshot.authoritative_version,
            (unsigned long long)statistics.fetch_transfers,
            (unsigned long long)statistics.evict_transfers,
            (unsigned long long)statistics.allocated_bytes,
            (unsigned long long)statistics.spill_allocated_bytes
        );
    }
    shadowspill_runtime_destroy(runtime);
    if (compute.words[0] != 0U) {
        (void)shadowspill_mock_destroy_compute_stream(mock, compute);
    }
    shadowspill_mock_backend_destroy(mock);
    return failed ? -1 : 0;
}

static int completed_offload_preserves_later_queued_fetch(void) {
    ShadowSpillMockBackend *mock = NULL;
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillBackendStream compute = {{0U, 0U}};
    const ShadowSpillMockBackendConfig backend_config = {
        .abi_version = SHADOWSPILL_BACKEND_ABI_VERSION,
        .evict_delay_nanoseconds = 1000000U,
        .event_delay_nanoseconds = 10000000U,
    };
    const ShadowSpillRuntimeConfig runtime_config = {
        .abi_version = SHADOWSPILL_RUNTIME_ABI_VERSION,
        .execution_pool_bytes = 128U,
        .spill_pool_bytes = 128U,
        .minimum_alignment = 1U,
        .worker_poll_nanoseconds = 1000U,
    };
    if (shadowspill_mock_backend_create(&backend_config, &mock) != 0) {
        return -1;
    }
    ShadowSpillRuntimeConfig configured = runtime_config;
    configured.backend = shadowspill_mock_backend_vtable(mock);
    if (shadowspill_runtime_create(&configured, &runtime) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_mock_create_compute_stream(mock, &compute) != 0) {
        shadowspill_runtime_destroy(runtime);
        shadowspill_mock_backend_destroy(mock);
        return -1;
    }

    const ShadowSpillObjectDescription target = {
        .object_id = 106U,
        .size_bytes = 32U,
        .initial_version = 1U,
        .retain_spill_copy = 1U,
    };
    const ShadowSpillRuntimeAction target_offload = {
        .object_id = target.object_id,
        .kind = SHADOWSPILL_RUNTIME_OFFLOAD,
    };
    const ShadowSpillRuntimeAction target_fetch = {
        .object_id = target.object_id,
        .kind = SHADOWSPILL_RUNTIME_PREFETCH,
    };
    ShadowSpillAllocation target_allocation = {0};
    int failed = shadowspill_register_object(runtime, &target) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_allocate(
            runtime, target.size_bytes, 1U, compute, &target_allocation
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_bind_object(
            runtime, target.object_id, target_allocation.allocation_id
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_after_task(
            runtime, 1U, compute, NULL, 0U, &target_offload, 1U
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_after_task(
            runtime, 2U, compute, NULL, 0U, &target_fetch, 1U
        ) != SHADOWSPILL_RUNTIME_OK;

    ShadowSpillQueuedAction *queued_fetch = NULL;
    pthread_mutex_lock(&runtime->actions.lock);
    for (ShadowSpillQueuedAction *action = runtime->actions.head;
         action != NULL; action = action->next) {
        if (action->task_id == 2U &&
            action->kind == SHADOWSPILL_RUNTIME_PREFETCH) {
            queued_fetch = action;
            action->processing = 1U;
            break;
        }
    }
    pthread_mutex_unlock(&runtime->actions.lock);
    failed = failed || queued_fetch == NULL;

    sleep_milliseconds(50U);
    uint32_t unpublished_fetches = 0U;
    if (queued_fetch != NULL) {
        pthread_mutex_lock(&queued_fetch->object->lock);
        unpublished_fetches = atomic_load_explicit(
            &queued_fetch->object->unpublished_fetch_count,
            memory_order_acquire
        );
        pthread_mutex_unlock(&queued_fetch->object->lock);
        pthread_mutex_lock(&runtime->actions.lock);
        queued_fetch->processing = 0U;
        pthread_mutex_unlock(&runtime->actions.lock);
        pthread_mutex_lock(&runtime->mutex);
        pthread_cond_broadcast(&runtime->condition);
        pthread_mutex_unlock(&runtime->mutex);
    }
    failed = failed || unpublished_fetches == 0U ||
        shadowspill_runtime_wait_idle(runtime) != SHADOWSPILL_RUNTIME_OK;
    if (failed) {
        fprintf(
            stderr,
            "queued fetch marker mismatch: unpublished=%u\n",
            (unsigned)unpublished_fetches
        );
    }

    shadowspill_runtime_destroy(runtime);
    if (compute.words[0] != 0U) {
        (void)shadowspill_mock_destroy_compute_stream(mock, compute);
    }
    shadowspill_mock_backend_destroy(mock);
    return failed ? -1 : 0;
}

static int consumer_waits_for_latest_queued_fetch_generation(void) {
    ShadowSpillMockBackend *mock = NULL;
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillBackendStream compute = {{0U, 0U}};
    const ShadowSpillMockBackendConfig backend_config = {
        .abi_version = SHADOWSPILL_BACKEND_ABI_VERSION,
        .fetch_delay_nanoseconds = 50000000U,
        .event_delay_nanoseconds = 1000000U,
    };
    const ShadowSpillRuntimeConfig runtime_config = {
        .abi_version = SHADOWSPILL_RUNTIME_ABI_VERSION,
        .execution_pool_bytes = 128U,
        .spill_pool_bytes = 128U,
        .minimum_alignment = 1U,
        .worker_poll_nanoseconds = 1000U,
    };
    if (shadowspill_mock_backend_create(&backend_config, &mock) != 0) {
        return -1;
    }
    ShadowSpillRuntimeConfig configured = runtime_config;
    configured.backend = shadowspill_mock_backend_vtable(mock);
    if (shadowspill_runtime_create(&configured, &runtime) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_mock_create_compute_stream(mock, &compute) != 0) {
        shadowspill_runtime_destroy(runtime);
        shadowspill_mock_backend_destroy(mock);
        return -1;
    }

    const ShadowSpillObjectDescription object = {
        .object_id = 107U,
        .size_bytes = 32U,
        .initial_version = 1U,
        .retain_spill_copy = 1U,
        .initially_spill_resident = 1U,
    };
    const ShadowSpillRuntimeAction fetch = {
        .object_id = object.object_id,
        .kind = SHADOWSPILL_RUNTIME_PREFETCH,
    };
    const ShadowSpillRuntimeAction release = {
        .object_id = object.object_id,
        .kind = SHADOWSPILL_RUNTIME_RELEASE,
    };
    const uint64_t input = object.object_id;
    const ShadowSpillExecutionDescription consumer = {
        .task_id = 73U,
        .input_object_ids = &input,
        .input_count = 1U,
    };
    ShadowSpillObjectSnapshot first_fetch = {0};
    ShadowSpillObjectSnapshot completed = {0};
    ShadowSpillObjectBinding binding = {0};
    ShadowSpillRuntimeStatistics statistics = {0};
    int failed = shadowspill_register_object(runtime, &object) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_admit_execution(runtime, &consumer) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_after_task(
            runtime, 70U, compute, NULL, 0U, &fetch, 1U
        ) != SHADOWSPILL_RUNTIME_OK;

    for (uint32_t attempt = 0U; !failed && attempt < 500U; ++attempt) {
        failed = shadowspill_object_snapshot(
            runtime, object.object_id, &first_fetch
        ) != SHADOWSPILL_RUNTIME_OK;
        if (!failed &&
            first_fetch.residency == SHADOWSPILL_OBJECT_PREFETCHING) {
            break;
        }
        sleep_milliseconds(1U);
    }
    failed = failed ||
        first_fetch.residency != SHADOWSPILL_OBJECT_PREFETCHING ||
        shadowspill_after_task(
            runtime, 71U, compute, NULL, 0U, &release, 1U
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_after_task(
            runtime, 72U, compute, NULL, 0U, &fetch, 1U
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_before_execution(
            runtime, consumer.task_id, compute, &binding, 1U
        ) != SHADOWSPILL_RUNTIME_OK ||
        binding.allocation_id == first_fetch.allocation_id ||
        shadowspill_after_execution(runtime, consumer.task_id, compute) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_wait_idle(runtime) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_object_snapshot(
            runtime, object.object_id, &completed
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_statistics(runtime, &statistics) !=
            SHADOWSPILL_RUNTIME_OK ||
        completed.residency != SHADOWSPILL_OBJECT_EXECUTION_READY ||
        completed.allocation_id != binding.allocation_id ||
        statistics.fetch_transfers != 2U;

    if (failed) {
        fprintf(
            stderr,
            "latest fetch generation mismatch: first=%llu binding=%llu "
            "completed=%llu residency=%u fetches=%llu\n",
            (unsigned long long)first_fetch.allocation_id,
            (unsigned long long)binding.allocation_id,
            (unsigned long long)completed.allocation_id,
            (unsigned)completed.residency,
            (unsigned long long)statistics.fetch_transfers
        );
    }
    shadowspill_runtime_destroy(runtime);
    if (compute.words[0] != 0U) {
        (void)shadowspill_mock_destroy_compute_stream(mock, compute);
    }
    shadowspill_mock_backend_destroy(mock);
    return failed ? -1 : 0;
}

#define REQUIRE_CANARY(expression)                                           \
    do {                                                                     \
        if ((expression) != 0) {                                             \
            fprintf(stderr, "runtime transition failed: %s\n", #expression); \
            return EXIT_FAILURE;                                            \
        }                                                                    \
    } while (0)

int main(void) {
    REQUIRE_CANARY(spill_object_rekey_preserves_authoritative_lease());
    REQUIRE_CANARY(invalid_action(0U, SHADOWSPILL_RUNTIME_PREFETCH));
    REQUIRE_CANARY(invalid_action(1U, SHADOWSPILL_RUNTIME_RELEASE));
    REQUIRE_CANARY(invalid_action(1U, SHADOWSPILL_RUNTIME_OFFLOAD));
    REQUIRE_CANARY(invalid_before_task(0U));
    REQUIRE_CANARY(invalid_before_task(1U));
    REQUIRE_CANARY(duplicate_action());
    REQUIRE_CANARY(output_allocation_handoff());
    REQUIRE_CANARY(chained_output_allocation_handoff());
    REQUIRE_CANARY(valid_transition_paths());
    REQUIRE_CANARY(prefetch_window_is_enqueued_without_host_blocking());
    REQUIRE_CANARY(inflight_prefetch_transfers_to_caller());
    REQUIRE_CANARY(offload_window_is_enqueued_without_host_serialization());
    REQUIRE_CANARY(trigger_reservation_failure_reports_no_progress());
    REQUIRE_CANARY(immutable_execution_admission());
    REQUIRE_CANARY(caller_handoff_preserves_recurrent_object_identity());
    REQUIRE_CANARY(admitted_task_allocates_dynamic_ranges());
    REQUIRE_CANARY(admitted_reuse_reacquires_retired_dynamic_range());
    REQUIRE_CANARY(admitted_allocation_without_progress_reports_no_progress());
    REQUIRE_CANARY(admitted_dynamic_allocations_use_deterministic_low_ranges());
    REQUIRE_CANARY(admitted_task_rejects_envelope_excess());
    REQUIRE_CANARY(admitted_task_accepts_exact_allocation_abi());
    REQUIRE_CANARY(delayed_free_is_not_charged_to_later_task_invocation());
    REQUIRE_CANARY(admitted_task_rejects_allocation_abi_geometry_mismatch());
    REQUIRE_CANARY(admitted_task_rejects_incomplete_allocation_abi());
    REQUIRE_CANARY(admitted_prefetch_reserves_dynamic_capacity());
    REQUIRE_CANARY(execution_plan_lifecycle());
    REQUIRE_CANARY(functional_mutation_replaces_lease_without_copy());
    REQUIRE_CANARY(functional_mutation_supersedes_inflight_prefetch());
    REQUIRE_CANARY(queued_release_causally_precedes_fetch());
    REQUIRE_CANARY(nonretained_fetch_then_offload_reserves_fresh_spill());
    REQUIRE_CANARY(completed_offload_preserves_later_queued_fetch());
    REQUIRE_CANARY(consumer_waits_for_latest_queued_fetch_generation());
    return EXIT_SUCCESS;
}
