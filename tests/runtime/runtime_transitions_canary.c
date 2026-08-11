#include <stdint.h>
#include <stdlib.h>

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
        .host_arena_bytes = 128U,
        .minimum_alignment = 1U,
        .progress_poll_nanoseconds = 1000U,
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

static int invalid_action(
    uint8_t initially_host_resident,
    uint8_t action_kind
) {
    Fixture fixture = {0};
    if (fixture_create(&fixture) != 0) {
        return -1;
    }
    const ShadowSpillObjectDescription object = {
        .object_id = 1U,
        .size_bytes = 32U,
        .initially_host_resident = initially_host_resident,
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
    fixture_destroy(&fixture);
    return result;
}

static int invalid_before_task(uint8_t initially_host_resident) {
    Fixture fixture = {0};
    if (fixture_create(&fixture) != 0) {
        return -1;
    }
    const ShadowSpillObjectDescription object = {
        .object_id = 1U,
        .size_bytes = 32U,
        .initially_host_resident = initially_host_resident,
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
        .initially_host_resident = 1U,
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

static int valid_transition_paths(void) {
    Fixture fixture = {0};
    if (fixture_create(&fixture) != 0) {
        return -1;
    }
    const ShadowSpillObjectDescription retained = {
        .object_id = 1U,
        .size_bytes = 32U,
        .retain_host_backing = 1U,
        .initially_host_resident = 1U,
    };
    const ShadowSpillObjectDescription temporary_host = {
        .object_id = 2U,
        .size_bytes = 32U,
        .initially_host_resident = 1U,
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
        !snapshot.host_current) {
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
        ) != SHADOWSPILL_RUNTIME_INVALID_STATE) {
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
        snapshot.has_host_range) {
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
        shadowspill_free(
            fixture.runtime, caller.allocation_id, fixture.compute
        ) != SHADOWSPILL_RUNTIME_OK) {
        result = -1;
    }

done:
    fixture_destroy(&fixture);
    return result;
}

int main(void) {
    return invalid_action(0U, SHADOWSPILL_RUNTIME_PREFETCH) == 0 &&
            invalid_action(1U, SHADOWSPILL_RUNTIME_RELEASE) == 0 &&
            invalid_action(1U, SHADOWSPILL_RUNTIME_OFFLOAD) == 0 &&
            invalid_before_task(0U) == 0 &&
            invalid_before_task(1U) == 0 && duplicate_action() == 0 &&
            valid_transition_paths() == 0
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}
