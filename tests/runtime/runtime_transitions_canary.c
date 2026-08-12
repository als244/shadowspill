#define _POSIX_C_SOURCE 200809L

#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#include <shadowspill/backend_mock.h>
#include <shadowspill/runtime.h>

typedef struct Fixture {
    ShadowSpillMockBackend *mock;
    ShadowSpillRuntime *runtime;
    ShadowSpillBackendStream compute;
} Fixture;

static int fixture_create(Fixture *fixture) {
    const ShadowSpillMockBackendConfig backend_config = {
        .abi_version = SHADOWSPILL_BACKEND_ABI_VERSION,
        .h2d_delay_nanoseconds = 1000U,
        .d2h_delay_nanoseconds = 1000U,
    };
    if (shadowspill_mock_backend_create(
            &backend_config, &fixture->mock
        ) != 0) {
        return -1;
    }
    const ShadowSpillRuntimeConfig runtime_config = {
        .abi_version = SHADOWSPILL_RUNTIME_ABI_VERSION,
        .device_slab_bytes = 128U,
        .host_arena_bytes = 256U,
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

static int prefetch_window_is_enqueued_without_host_blocking(void) {
    ShadowSpillMockBackend *mock = NULL;
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillBackendStream compute = {{0U, 0U}};
    const ShadowSpillMockBackendConfig backend_config = {
        .abi_version = SHADOWSPILL_BACKEND_ABI_VERSION,
        .h2d_delay_nanoseconds = 100000000U,
        .event_delay_nanoseconds = 50000000U,
    };
    const ShadowSpillRuntimeConfig runtime_config = {
        .abi_version = SHADOWSPILL_RUNTIME_ABI_VERSION,
        .device_slab_bytes = 128U,
        .host_arena_bytes = 128U,
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
        statistics.live_allocations != 0U ||
        statistics.allocated_bytes != 0U ||
        statistics.transfers_to_device != 0U;
    sleep_milliseconds(60U);
    failed = failed || shadowspill_runtime_statistics(runtime, &statistics) !=
            SHADOWSPILL_RUNTIME_OK ||
        statistics.live_allocations != 2U ||
        statistics.allocated_bytes != 64U ||
        statistics.transfers_to_device != 2U;
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
        statistics.transfers_to_device != 2U;
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
        .d2h_delay_nanoseconds = 100000000U,
        .event_delay_nanoseconds = 50000000U,
    };
    ShadowSpillRuntimeConfig runtime_config = {
        .abi_version = SHADOWSPILL_RUNTIME_ABI_VERSION,
        .device_slab_bytes = 128U,
        .host_arena_bytes = 128U,
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
    ShadowSpillRuntimeStatistics statistics = {0};
    failed = failed || shadowspill_runtime_statistics(runtime, &statistics) !=
            SHADOWSPILL_RUNTIME_OK ||
        statistics.host_allocated_bytes != 0U ||
        statistics.transfers_to_host != 0U;
    sleep_milliseconds(60U);
    failed = failed || shadowspill_runtime_statistics(runtime, &statistics) !=
            SHADOWSPILL_RUNTIME_OK ||
        statistics.host_allocated_bytes != 64U ||
        statistics.transfers_to_host != 2U;
    failed = failed || shadowspill_runtime_wait_idle(runtime) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_statistics(runtime, &statistics) !=
            SHADOWSPILL_RUNTIME_OK ||
        statistics.host_allocated_bytes != 64U ||
        statistics.transfers_to_host != 2U;
    shadowspill_runtime_destroy(runtime);
    (void)shadowspill_mock_destroy_compute_stream(mock, compute);
    shadowspill_mock_backend_destroy(mock);
    return failed ? -1 : 0;
}

static int trigger_reservation_failure_is_a_plan_violation(void) {
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
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_wait_idle(fixture.runtime) !=
            SHADOWSPILL_RUNTIME_PLAN_VIOLATION;
    ShadowSpillRuntimeFailure failure = {0};
    failed = failed || shadowspill_runtime_failure(
            fixture.runtime, &failure
        ) != SHADOWSPILL_RUNTIME_OK ||
        failure.status != SHADOWSPILL_RUNTIME_PLAN_VIOLATION ||
        failure.object_id != 2U || failure.requested_bytes != 80U ||
        failure.free_bytes != 48U ||
        failure.largest_free_range_bytes != 48U;
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
        target_snapshot.residency != SHADOWSPILL_OBJECT_DEVICE_READY ||
        target_snapshot.execution_pointer != allocation.pointer ||
        target_snapshot.allocation_id != allocation.allocation_id) {
        result = -1;
    }

done:
    fixture_destroy(&fixture);
    return result;
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
        snapshot.residency != SHADOWSPILL_OBJECT_HOST_ONLY ||
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
        snapshot.residency != SHADOWSPILL_OBJECT_HOST_ONLY ||
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
        snapshot.residency != SHADOWSPILL_OBJECT_DEVICE_READY ||
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
        snapshot.residency != SHADOWSPILL_OBJECT_DEVICE_READY) {
        result = -1;
        goto done;
    }
    ShadowSpillAllocation caller = {0};
    if (shadowspill_transfer_object_to_caller(
            fixture.runtime, device_created.object_id, &caller
        ) != SHADOWSPILL_RUNTIME_OK ||
        caller.allocation_id != third.allocation_id ||
        shadowspill_object_snapshot(
            fixture.runtime, device_created.object_id, &snapshot
        ) != SHADOWSPILL_RUNTIME_INVALID_STATE ||
        shadowspill_register_object(fixture.runtime, &device_created) !=
            SHADOWSPILL_RUNTIME_OK ||
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

int main(void) {
    return invalid_action(0U, SHADOWSPILL_RUNTIME_PREFETCH) == 0 &&
            invalid_action(1U, SHADOWSPILL_RUNTIME_RELEASE) == 0 &&
            invalid_action(1U, SHADOWSPILL_RUNTIME_OFFLOAD) == 0 &&
            invalid_before_task(0U) == 0 &&
            invalid_before_task(1U) == 0 && duplicate_action() == 0 &&
            output_allocation_handoff() == 0 &&
            valid_transition_paths() == 0 &&
            prefetch_window_is_enqueued_without_host_blocking() == 0 &&
            offload_window_is_enqueued_without_host_serialization() == 0 &&
            trigger_reservation_failure_is_a_plan_violation() == 0
            && immutable_execution_admission() == 0
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}
