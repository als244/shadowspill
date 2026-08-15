#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <shadowspill/backend_mock.h>
#include <shadowspill/runtime.h>

static int layout_lifecycle_preserves_dynamic_allocations(void) {
    ShadowSpillMockBackend *mock = NULL;
    const ShadowSpillMockBackendConfig mock_config = {
        .abi_version = SHADOWSPILL_BACKEND_ABI_VERSION,
    };
    if (shadowspill_mock_backend_create(&mock_config, &mock) != 0) {
        return -1;
    }
    const ShadowSpillRuntimeConfig runtime_config = {
        .abi_version = SHADOWSPILL_RUNTIME_ABI_VERSION,
        .execution_pool_bytes = 256U,
        .spill_pool_bytes = 128U,
        .minimum_alignment = 16U,
        .worker_poll_nanoseconds = 1000U,
        .backend = shadowspill_mock_backend_vtable(mock),
    };
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillBackendStream compute = {{0U, 0U}};
    int failed = shadowspill_runtime_create(&runtime_config, &runtime) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_mock_create_compute_stream(mock, &compute) != 0;

    const ShadowSpillObjectDescription object = {
        .object_id = 42U,
        .size_bytes = 64U,
        .initially_spill_resident = 1U,
    };
    failed = failed || shadowspill_register_object(runtime, &object) !=
        SHADOWSPILL_RUNTIME_OK;
    const ShadowSpillFixedPlacementDescription placements[3] = {
        {
            .task_id = SHADOWSPILL_RUNTIME_NO_ID,
            .ordinal = SHADOWSPILL_RUNTIME_NO_ID,
            .object_id = object.object_id,
            .bytes = 64U,
            .alignment_bytes = 16U,
            .kind = SHADOWSPILL_FIXED_INITIAL_OBJECT,
        },
        {
            .task_id = 7U,
            .ordinal = 0U,
            .object_id = SHADOWSPILL_RUNTIME_NO_ID,
            .offset = 64U,
            .bytes = 64U,
            .alignment_bytes = 16U,
            .kind = SHADOWSPILL_FIXED_TASK_ALLOCATION,
        },
        {
            .task_id = 9U,
            .ordinal = 0U,
            .object_id = object.object_id,
            .bytes = 64U,
            .alignment_bytes = 16U,
            .kind = SHADOWSPILL_FIXED_ACTION_DESTINATION,
        },
    };
    const ShadowSpillFixedDependencyDescription dependency = {
        .predecessor_task_id = 8U,
        .predecessor_action_ordinal = 0U,
        .successor_task_id = 9U,
        .successor_ordinal = 0U,
        .successor_kind = SHADOWSPILL_FIXED_ACTION_DESTINATION,
    };
    const ShadowSpillFixedLayoutDescription layout = {
        .abi_version = SHADOWSPILL_FIXED_LAYOUT_ABI_VERSION,
        .slice_bytes = 128U,
        .placements = placements,
        .placement_count = 3U,
        .dependencies = &dependency,
        .dependency_count = 1U,
    };
    const ShadowSpillTaskAllocationABIStep allocation_steps[2] = {
        {
            .allocation_ordinal = 0U,
            .requested_bytes = 64U,
            .charged_bytes = 64U,
            .alignment_bytes = 16U,
            .operation = SHADOWSPILL_TASK_ALLOCATION_ALLOCATE,
        },
        {
            .allocation_ordinal = 0U,
            .requested_bytes = 64U,
            .charged_bytes = 64U,
            .alignment_bytes = 16U,
            .operation = SHADOWSPILL_TASK_ALLOCATION_FREE,
        },
    };
    const ShadowSpillExecutionDescription execution = {
        .task_id = 7U,
        .allocation_abi_steps = allocation_steps,
        .allocation_abi_step_count = 2U,
        .enforce_allocation_abi = 1U,
    };
    const ShadowSpillRuntimeAction eviction = {
        .object_id = object.object_id,
        .kind = SHADOWSPILL_RUNTIME_OFFLOAD,
    };
    const ShadowSpillExecutionDescription eviction_task = {
        .task_id = 8U,
        .actions = &eviction,
        .action_count = 1U,
    };
    const ShadowSpillRuntimeAction fetch = {
        .object_id = object.object_id,
        .kind = SHADOWSPILL_RUNTIME_PREFETCH,
    };
    const ShadowSpillExecutionDescription fetch_task = {
        .task_id = 9U,
        .actions = &fetch,
        .action_count = 1U,
    };
    failed = failed || shadowspill_admit_fixed_layout(runtime, &layout) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_admit_execution(runtime, &execution) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_admit_execution(runtime, &eviction_task) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_admit_execution(runtime, &fetch_task) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_seal_fixed_layout(runtime) != SHADOWSPILL_RUNTIME_OK;

    ShadowSpillAllocation caller_owned = {0};
    failed = failed || shadowspill_allocate(
            runtime, 64U, 16U, compute, &caller_owned
        ) != SHADOWSPILL_RUNTIME_OK;
    ShadowSpillRuntimeStatistics statistics = {0};
    failed = failed || shadowspill_runtime_statistics(runtime, &statistics) !=
            SHADOWSPILL_RUNTIME_OK ||
        statistics.allocated_bytes != 192U;

    failed = failed || shadowspill_clear_execution_plan(runtime) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_statistics(runtime, &statistics) !=
            SHADOWSPILL_RUNTIME_OK ||
        statistics.allocated_bytes != 64U ||
        statistics.largest_free_range_bytes != 128U;

    failed = failed || shadowspill_free(
            runtime, caller_owned.allocation_id, compute
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_wait_idle(runtime) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_unregister_object(runtime, object.object_id) !=
            SHADOWSPILL_RUNTIME_OK;
    shadowspill_runtime_destroy(runtime);
    if (compute.words[0] != 0U) {
        failed = failed ||
            shadowspill_mock_destroy_compute_stream(mock, compute) != 0;
    }
    shadowspill_mock_backend_destroy(mock);
    return failed ? -1 : 0;
}

static int duplicate_placement_is_rejected(void) {
    ShadowSpillMockBackend *mock = NULL;
    const ShadowSpillMockBackendConfig mock_config = {
        .abi_version = SHADOWSPILL_BACKEND_ABI_VERSION,
    };
    if (shadowspill_mock_backend_create(&mock_config, &mock) != 0) {
        return -1;
    }
    const ShadowSpillRuntimeConfig runtime_config = {
        .abi_version = SHADOWSPILL_RUNTIME_ABI_VERSION,
        .execution_pool_bytes = 128U,
        .minimum_alignment = 16U,
        .worker_poll_nanoseconds = 1000U,
        .backend = shadowspill_mock_backend_vtable(mock),
    };
    ShadowSpillRuntime *runtime = NULL;
    int failed = shadowspill_runtime_create(&runtime_config, &runtime) !=
        SHADOWSPILL_RUNTIME_OK;
    const ShadowSpillFixedPlacementDescription placements[2] = {
        {
            .task_id = 5U,
            .ordinal = 0U,
            .object_id = SHADOWSPILL_RUNTIME_NO_ID,
            .bytes = 16U,
            .alignment_bytes = 16U,
            .kind = SHADOWSPILL_FIXED_TASK_ALLOCATION,
        },
        {
            .task_id = 5U,
            .ordinal = 0U,
            .object_id = SHADOWSPILL_RUNTIME_NO_ID,
            .offset = 32U,
            .bytes = 16U,
            .alignment_bytes = 16U,
            .kind = SHADOWSPILL_FIXED_TASK_ALLOCATION,
        },
    };
    const ShadowSpillFixedLayoutDescription layout = {
        .abi_version = SHADOWSPILL_FIXED_LAYOUT_ABI_VERSION,
        .slice_bytes = 64U,
        .placements = placements,
        .placement_count = 2U,
    };
    failed = failed || shadowspill_admit_fixed_layout(runtime, &layout) !=
        SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    shadowspill_runtime_destroy(runtime);
    shadowspill_mock_backend_destroy(mock);
    return failed ? -1 : 0;
}

int main(void) {
    if (layout_lifecycle_preserves_dynamic_allocations() != 0) {
        fprintf(stderr, "fixed layout lifecycle failed\n");
        return EXIT_FAILURE;
    }
    if (duplicate_placement_is_rejected() != 0) {
        fprintf(stderr, "duplicate fixed placement was accepted\n");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
