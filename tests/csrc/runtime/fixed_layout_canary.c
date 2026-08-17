#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <shadowspill/backend_mock.h>
#include <shadowspill/runtime.h>

#include "runtime_test.h"

static int layout_lifecycle_preserves_dynamic_allocations(void) {
    ShadowSpillMockBackend *mock = NULL;
    const ShadowSpillMockBackendConfig mock_config = {
        .abi_version = SHADOWSPILL_MOCK_BACKEND_ABI_VERSION,
    };
    if (shadowspill_mock_backend_create(&mock_config, &mock) != 0) {
        return -1;
    }
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillBackendStream compute = {{0U, 0U}};
    int failed = shadowspill_test_create_runtime(
            mock, 256U, 128U, 16U, 1000U, &runtime
        ) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_mock_create_compute_stream(mock, &compute) != 0;

    const ShadowSpillObjectDescription object = {
        .object_id = 42U,
        .size_bytes = 64U,
        .initially_spill_resident = 1U,
    };
    failed = failed || shadowspill_register_object(runtime, &object) !=
        SHADOWSPILL_RUNTIME_OK;
    const ShadowSpillFixedPlacementDescription placements[4] = {
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
        {
            .task_id = 10U,
            .ordinal = 0U,
            .object_id = SHADOWSPILL_RUNTIME_NO_ID,
            .offset = SHADOWSPILL_RUNTIME_NO_ID,
            .bytes = 32U,
            .alignment_bytes = 16U,
            .kind = SHADOWSPILL_DYNAMIC_TASK_ALLOCATION,
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
        .placement_count = 4U,
        .dependencies = &dependency,
        .dependency_count = 1U,
    };
    const ShadowSpillTaskAllocationContractStep allocation_steps[2] = {
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
    const ShadowSpillTaskDescription execution = {
        .task_id = 7U,
        .allocation_contract_steps = allocation_steps,
        .allocation_contract_step_count = 2U,
        .enforce_allocation_contract = 1U,
    };
    const ShadowSpillTaskAllocationContractStep dynamic_steps[2] = {
        {
            .allocation_ordinal = 0U,
            .requested_bytes = 32U,
            .charged_bytes = 32U,
            .alignment_bytes = 16U,
            .operation = SHADOWSPILL_TASK_ALLOCATION_ALLOCATE,
        },
        {
            .allocation_ordinal = 0U,
            .requested_bytes = 32U,
            .charged_bytes = 32U,
            .alignment_bytes = 16U,
            .operation = SHADOWSPILL_TASK_ALLOCATION_FREE,
        },
    };
    const ShadowSpillTaskDescription dynamic_execution = {
        .task_id = 10U,
        .allocation_contract_steps = dynamic_steps,
        .allocation_contract_step_count = 2U,
        .enforce_allocation_contract = 1U,
    };
    const ShadowSpillRuntimeAction eviction = {
        .object_id = object.object_id,
        .kind = SHADOWSPILL_RUNTIME_OFFLOAD,
    };
    const ShadowSpillTaskDescription eviction_task = {
        .task_id = 8U,
        .actions = &eviction,
        .action_count = 1U,
    };
    const ShadowSpillRuntimeAction fetch = {
        .object_id = object.object_id,
        .kind = SHADOWSPILL_RUNTIME_PREFETCH,
    };
    const ShadowSpillTaskDescription fetch_task = {
        .task_id = 9U,
        .actions = &fetch,
        .action_count = 1U,
    };
    failed = failed || shadowspill_test_admit_fixed_layout(runtime, &layout) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_test_admit_task(runtime, &execution) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_test_admit_task(runtime, &dynamic_execution) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_test_admit_task(runtime, &eviction_task) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_test_admit_task(runtime, &fetch_task) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_test_seal_fixed_layout(runtime) != SHADOWSPILL_RUNTIME_OK;

    ShadowSpillAllocation caller_owned = {0};
    failed = failed || shadowspill_allocate(
            runtime, 64U, 16U, compute, &caller_owned
        ) != SHADOWSPILL_RUNTIME_OK;
    failed = failed || shadowspill_test_before_task(
            runtime, execution.task_id, compute, NULL, 0U
        ) != SHADOWSPILL_RUNTIME_OK;
    ShadowSpillAllocation fixed = {0};
    failed = failed || shadowspill_allocate(
            runtime, 64U, 16U, compute, &fixed
        ) != SHADOWSPILL_RUNTIME_OK ||
        (uintptr_t)fixed.pointer + 64U != (uintptr_t)caller_owned.pointer ||
        shadowspill_free(runtime, fixed.allocation_id, compute) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_test_after_task(runtime, execution.task_id, compute) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_wait_idle(runtime) != SHADOWSPILL_RUNTIME_OK;
    failed = failed || shadowspill_test_before_task(
            runtime, dynamic_execution.task_id, compute, NULL, 0U
        ) != SHADOWSPILL_RUNTIME_OK;
    ShadowSpillAllocation dynamic = {0};
    failed = failed || shadowspill_allocate(
            runtime, 32U, 16U, compute, &dynamic
        ) != SHADOWSPILL_RUNTIME_OK ||
        (uintptr_t)dynamic.pointer != (uintptr_t)caller_owned.pointer + 64U ||
        shadowspill_free(runtime, dynamic.allocation_id, compute) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_test_after_task(
            runtime, dynamic_execution.task_id, compute
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_wait_idle(runtime) != SHADOWSPILL_RUNTIME_OK;
    ShadowSpillRuntimeStatistics statistics = {0};
    failed = failed || shadowspill_runtime_statistics(runtime, &statistics) !=
            SHADOWSPILL_RUNTIME_OK ||
        statistics.allocated_bytes != 192U;

    failed = failed || shadowspill_test_clear_plan(runtime) !=
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
    shadowspill_test_destroy_runtime(runtime);
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
        .abi_version = SHADOWSPILL_MOCK_BACKEND_ABI_VERSION,
    };
    if (shadowspill_mock_backend_create(&mock_config, &mock) != 0) {
        return -1;
    }
    ShadowSpillRuntime *runtime = NULL;
    int failed = shadowspill_test_create_runtime(
            mock, 128U, 0U, 16U, 1000U, &runtime
        ) !=
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
    failed = failed || shadowspill_test_admit_fixed_layout(runtime, &layout) !=
        SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    shadowspill_test_destroy_runtime(runtime);
    shadowspill_mock_backend_destroy(mock);
    return failed ? -1 : 0;
}

static int empty_fixed_slice_allows_dynamic_task(void) {
    ShadowSpillMockBackend *mock = NULL;
    const ShadowSpillMockBackendConfig mock_config = {
        .abi_version = SHADOWSPILL_MOCK_BACKEND_ABI_VERSION,
    };
    if (shadowspill_mock_backend_create(&mock_config, &mock) != 0) {
        return -1;
    }
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillBackendStream compute = {{0U, 0U}};
    int failed = shadowspill_test_create_runtime(
            mock, 64U, 0U, 16U, 1000U, &runtime
        ) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_mock_create_compute_stream(mock, &compute) != 0;
    const ShadowSpillFixedPlacementDescription placement = {
        .task_id = 15U,
        .ordinal = 0U,
        .object_id = SHADOWSPILL_RUNTIME_NO_ID,
        .offset = SHADOWSPILL_RUNTIME_NO_ID,
        .bytes = 32U,
        .alignment_bytes = 16U,
        .kind = SHADOWSPILL_DYNAMIC_TASK_ALLOCATION,
    };
    const ShadowSpillFixedLayoutDescription layout = {
        .abi_version = SHADOWSPILL_FIXED_LAYOUT_ABI_VERSION,
        .placements = &placement,
        .placement_count = 1U,
    };
    const ShadowSpillTaskAllocationContractStep steps[2] = {
        {
            .allocation_ordinal = 0U,
            .requested_bytes = 32U,
            .charged_bytes = 32U,
            .alignment_bytes = 16U,
            .operation = SHADOWSPILL_TASK_ALLOCATION_ALLOCATE,
        },
        {
            .allocation_ordinal = 0U,
            .requested_bytes = 32U,
            .charged_bytes = 32U,
            .alignment_bytes = 16U,
            .operation = SHADOWSPILL_TASK_ALLOCATION_FREE,
        },
    };
    const ShadowSpillTaskDescription execution = {
        .task_id = 15U,
        .allocation_contract_steps = steps,
        .allocation_contract_step_count = 2U,
        .enforce_allocation_contract = 1U,
    };
    failed = failed || shadowspill_test_admit_fixed_layout(runtime, &layout) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_test_admit_task(runtime, &execution) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_test_seal_fixed_layout(runtime) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_test_before_task(
            runtime, execution.task_id, compute, NULL, 0U
        ) != SHADOWSPILL_RUNTIME_OK;
    ShadowSpillAllocation allocation = {0};
    failed = failed || shadowspill_allocate(
            runtime, 32U, 16U, compute, &allocation
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_free(runtime, allocation.allocation_id, compute) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_test_after_task(runtime, execution.task_id, compute) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_wait_idle(runtime) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_test_clear_plan(runtime) != SHADOWSPILL_RUNTIME_OK;
    ShadowSpillRuntimeStatistics statistics = {0};
    failed = failed || shadowspill_runtime_statistics(runtime, &statistics) !=
            SHADOWSPILL_RUNTIME_OK ||
        statistics.allocated_bytes != 0U ||
        statistics.largest_free_range_bytes != 64U;
    shadowspill_test_destroy_runtime(runtime);
    if (compute.words[0] != 0U) {
        failed = failed ||
            shadowspill_mock_destroy_compute_stream(mock, compute) != 0;
    }
    shadowspill_mock_backend_destroy(mock);
    return failed ? -1 : 0;
}

static int empty_fixed_slice_allows_dynamic_fetch(void) {
    ShadowSpillMockBackend *mock = NULL;
    const ShadowSpillMockBackendConfig mock_config = {
        .abi_version = SHADOWSPILL_MOCK_BACKEND_ABI_VERSION,
    };
    if (shadowspill_mock_backend_create(&mock_config, &mock) != 0) {
        return -1;
    }
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillBackendStream compute = {{0U, 0U}};
    int failed = shadowspill_test_create_runtime(
            mock, 64U, 64U, 16U, 1000U, &runtime
        ) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_mock_create_compute_stream(mock, &compute) != 0;
    const ShadowSpillObjectDescription object = {
        .object_id = 71U,
        .size_bytes = 64U,
        .retain_spill_copy = 1U,
        .initially_spill_resident = 1U,
    };
    failed = failed || shadowspill_register_object(runtime, &object) !=
        SHADOWSPILL_RUNTIME_OK;
    const ShadowSpillFixedPlacementDescription placement = {
        .task_id = 21U,
        .ordinal = 0U,
        .object_id = object.object_id,
        .offset = SHADOWSPILL_RUNTIME_NO_ID,
        .bytes = 64U,
        .alignment_bytes = 16U,
        .kind = SHADOWSPILL_DYNAMIC_ACTION_DESTINATION,
    };
    const ShadowSpillFixedLayoutDescription layout = {
        .abi_version = SHADOWSPILL_FIXED_LAYOUT_ABI_VERSION,
        .placements = &placement,
        .placement_count = 1U,
    };
    const ShadowSpillRuntimeAction fetch = {
        .object_id = object.object_id,
        .kind = SHADOWSPILL_RUNTIME_PREFETCH,
    };
    const ShadowSpillTaskDescription fetch_task = {
        .task_id = 21U,
        .actions = &fetch,
        .action_count = 1U,
    };
    const ShadowSpillRuntimeAction release = {
        .object_id = object.object_id,
        .kind = SHADOWSPILL_RUNTIME_RELEASE,
    };
    const ShadowSpillTaskDescription release_task = {
        .task_id = 22U,
        .actions = &release,
        .action_count = 1U,
    };
    failed = failed || shadowspill_test_admit_fixed_layout(runtime, &layout) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_test_admit_task(runtime, &fetch_task) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_test_admit_task(runtime, &release_task) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_test_seal_fixed_layout(runtime) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_test_before_task(
            runtime, fetch_task.task_id, compute, NULL, 0U
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_test_after_task(runtime, fetch_task.task_id, compute) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_wait_idle(runtime) != SHADOWSPILL_RUNTIME_OK;
    ShadowSpillObjectSnapshot snapshot = {0};
    ShadowSpillRuntimeStatistics statistics = {0};
    failed = failed || shadowspill_object_snapshot(
            runtime, object.object_id, &snapshot
        ) != SHADOWSPILL_RUNTIME_OK ||
        snapshot.residency != SHADOWSPILL_OBJECT_EXECUTION_READY ||
        snapshot.execution_pointer == NULL ||
        shadowspill_runtime_statistics(runtime, &statistics) !=
            SHADOWSPILL_RUNTIME_OK ||
        statistics.allocated_bytes != 64U;
    failed = failed || shadowspill_test_before_task(
            runtime, release_task.task_id, compute, NULL, 0U
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_test_after_task(runtime, release_task.task_id, compute) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_wait_idle(runtime) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_unregister_object(runtime, object.object_id) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_test_clear_plan(runtime) !=
            SHADOWSPILL_RUNTIME_OK;
    shadowspill_test_destroy_runtime(runtime);
    if (compute.words[0] != 0U) {
        failed = failed ||
            shadowspill_mock_destroy_compute_stream(mock, compute) != 0;
    }
    shadowspill_mock_backend_destroy(mock);
    return failed ? -1 : 0;
}

static int eviction_completion_orders_fixed_reuse(void) {
    ShadowSpillMockBackend *mock = NULL;
    const ShadowSpillMockBackendConfig mock_config = {
        .abi_version = SHADOWSPILL_MOCK_BACKEND_ABI_VERSION,
        .evict_delay_nanoseconds = 100000000U,
    };
    if (shadowspill_mock_backend_create(&mock_config, &mock) != 0) {
        return -1;
    }
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillBackendStream compute = {{0U, 0U}};
    int failed = shadowspill_test_create_runtime(
            mock, 128U, 128U, 16U, 1000U, &runtime
        ) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_mock_create_compute_stream(mock, &compute) != 0;
    const ShadowSpillObjectDescription object = {
        .object_id = 51U,
        .size_bytes = 64U,
        .retain_spill_copy = 1U,
        .initially_spill_resident = 1U,
    };
    failed = failed || shadowspill_register_object(runtime, &object) !=
        SHADOWSPILL_RUNTIME_OK;

    const ShadowSpillFixedPlacementDescription placements[2] = {
        {
            .task_id = 6U,
            .ordinal = 0U,
            .object_id = object.object_id,
            .bytes = 64U,
            .alignment_bytes = 16U,
            .kind = SHADOWSPILL_FIXED_ACTION_DESTINATION,
        },
        {
            .task_id = 7U,
            .ordinal = 0U,
            .object_id = SHADOWSPILL_RUNTIME_NO_ID,
            .bytes = 64U,
            .alignment_bytes = 16U,
            .kind = SHADOWSPILL_FIXED_TASK_ALLOCATION,
        },
    };
    const ShadowSpillFixedDependencyDescription dependency = {
        .predecessor_task_id = 8U,
        .predecessor_action_ordinal = 0U,
        .successor_task_id = 7U,
        .successor_ordinal = 0U,
        .successor_kind = SHADOWSPILL_FIXED_TASK_ALLOCATION,
    };
    const ShadowSpillFixedLayoutDescription layout = {
        .abi_version = SHADOWSPILL_FIXED_LAYOUT_ABI_VERSION,
        .slice_bytes = 64U,
        .placements = placements,
        .placement_count = 2U,
        .dependencies = &dependency,
        .dependency_count = 1U,
    };
    const ShadowSpillRuntimeAction fetch = {
        .object_id = object.object_id,
        .kind = SHADOWSPILL_RUNTIME_PREFETCH,
    };
    const ShadowSpillTaskDescription fetch_task = {
        .task_id = 6U,
        .actions = &fetch,
        .action_count = 1U,
    };
    const ShadowSpillTaskAllocationContractStep allocation_steps[2] = {
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
    const ShadowSpillTaskDescription allocation_task = {
        .task_id = 7U,
        .allocation_contract_steps = allocation_steps,
        .allocation_contract_step_count = 2U,
        .enforce_allocation_contract = 1U,
    };
    const ShadowSpillRuntimeAction eviction = {
        .object_id = object.object_id,
        .kind = SHADOWSPILL_RUNTIME_OFFLOAD,
    };
    const ShadowSpillTaskDescription eviction_task = {
        .task_id = 8U,
        .actions = &eviction,
        .action_count = 1U,
    };
    failed = failed || shadowspill_test_admit_fixed_layout(runtime, &layout) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_test_admit_task(runtime, &fetch_task) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_test_admit_task(runtime, &allocation_task) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_test_admit_task(runtime, &eviction_task) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_test_seal_fixed_layout(runtime) != SHADOWSPILL_RUNTIME_OK;

    failed = failed || shadowspill_test_before_task(
            runtime, fetch_task.task_id, compute, NULL, 0U
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_test_after_task(runtime, fetch_task.task_id, compute) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_wait_idle(runtime) != SHADOWSPILL_RUNTIME_OK;
    ShadowSpillObjectSnapshot snapshot = {0};
    failed = failed || shadowspill_object_snapshot(
            runtime, object.object_id, &snapshot
        ) != SHADOWSPILL_RUNTIME_OK ||
        snapshot.residency != SHADOWSPILL_OBJECT_EXECUTION_READY;

    failed = failed || shadowspill_test_before_task(
            runtime, eviction_task.task_id, compute, NULL, 0U
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_test_after_task(runtime, eviction_task.task_id, compute) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_test_before_task(
            runtime, allocation_task.task_id, compute, NULL, 0U
        ) != SHADOWSPILL_RUNTIME_OK;
    ShadowSpillAllocation reused = {0};
    failed = failed || shadowspill_allocate(
            runtime, 64U, 16U, compute, &reused
        ) != SHADOWSPILL_RUNTIME_OK ||
        reused.pointer != snapshot.execution_pointer ||
        shadowspill_free(runtime, reused.allocation_id, compute) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_test_after_task(
            runtime, allocation_task.task_id, compute
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_wait_idle(runtime) != SHADOWSPILL_RUNTIME_OK;
    ShadowSpillRuntimeStatistics statistics = {0};
    failed = failed || shadowspill_runtime_statistics(runtime, &statistics) !=
            SHADOWSPILL_RUNTIME_OK ||
        statistics.wait_events_inserted == 0U ||
        statistics.allocated_bytes != 64U ||
        shadowspill_unregister_object(runtime, object.object_id) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_test_clear_plan(runtime) != SHADOWSPILL_RUNTIME_OK;

    shadowspill_test_destroy_runtime(runtime);
    if (compute.words[0] != 0U) {
        failed = failed ||
            shadowspill_mock_destroy_compute_stream(mock, compute) != 0;
    }
    shadowspill_mock_backend_destroy(mock);
    return failed ? -1 : 0;
}

static int eviction_completion_orders_fixed_fetch_reuse(int same_object) {
    ShadowSpillMockBackend *mock = NULL;
    const ShadowSpillMockBackendConfig mock_config = {
        .abi_version = SHADOWSPILL_MOCK_BACKEND_ABI_VERSION,
        .fetch_delay_nanoseconds = 1000000U,
        .evict_delay_nanoseconds = same_object ? 1000000U : 100000000U,
    };
    if (shadowspill_mock_backend_create(&mock_config, &mock) != 0) {
        return -1;
    }
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillBackendStream compute = {{0U, 0U}};
    int failed = shadowspill_test_create_runtime(
            mock, 64U, 128U, 16U, 1000U, &runtime
        ) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_mock_create_compute_stream(mock, &compute) != 0;
    const ShadowSpillObjectDescription objects[2] = {
        {
            .object_id = 61U,
            .size_bytes = 64U,
            .retain_spill_copy = 1U,
            .initially_spill_resident = 1U,
        },
        {
            .object_id = 62U,
            .size_bytes = 64U,
            .retain_spill_copy = 1U,
            .initially_spill_resident = 1U,
        },
    };
    failed = failed || shadowspill_register_object(runtime, &objects[0]) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_register_object(runtime, &objects[1]) !=
            SHADOWSPILL_RUNTIME_OK;
    const uint64_t successor_object_id = same_object
        ? objects[0].object_id
        : objects[1].object_id;

    const ShadowSpillFixedPlacementDescription placements[2] = {
        {
            .task_id = 11U,
            .ordinal = 0U,
            .object_id = objects[0].object_id,
            .bytes = 64U,
            .alignment_bytes = 16U,
            .kind = SHADOWSPILL_FIXED_ACTION_DESTINATION,
        },
        {
            .task_id = 13U,
            .ordinal = 0U,
            .object_id = successor_object_id,
            .bytes = 64U,
            .alignment_bytes = 16U,
            .kind = SHADOWSPILL_FIXED_ACTION_DESTINATION,
        },
    };
    const ShadowSpillFixedDependencyDescription dependency = {
        .predecessor_task_id = 12U,
        .predecessor_action_ordinal = 0U,
        .successor_task_id = 13U,
        .successor_ordinal = 0U,
        .successor_kind = SHADOWSPILL_FIXED_ACTION_DESTINATION,
    };
    const ShadowSpillFixedLayoutDescription layout = {
        .abi_version = SHADOWSPILL_FIXED_LAYOUT_ABI_VERSION,
        .slice_bytes = 64U,
        .placements = placements,
        .placement_count = 2U,
        .dependencies = &dependency,
        .dependency_count = 1U,
    };
    const ShadowSpillRuntimeAction fetch_first = {
        .object_id = objects[0].object_id,
        .kind = SHADOWSPILL_RUNTIME_PREFETCH,
    };
    const ShadowSpillTaskDescription first_fetch_task = {
        .task_id = 11U,
        .actions = &fetch_first,
        .action_count = 1U,
    };
    const ShadowSpillObjectUpdate mutate_first = {
        .object_id = objects[0].object_id,
        .version_delta = 1U,
    };
    const ShadowSpillRuntimeAction evict_first = {
        .object_id = objects[0].object_id,
        .kind = SHADOWSPILL_RUNTIME_OFFLOAD,
    };
    const ShadowSpillTaskDescription eviction_task = {
        .task_id = 12U,
        .updates = &mutate_first,
        .update_count = 1U,
        .actions = &evict_first,
        .action_count = 1U,
    };
    const ShadowSpillRuntimeAction fetch_second = {
        .object_id = successor_object_id,
        .kind = SHADOWSPILL_RUNTIME_PREFETCH,
    };
    const ShadowSpillTaskDescription second_fetch_task = {
        .task_id = 13U,
        .actions = &fetch_second,
        .action_count = 1U,
    };
    const ShadowSpillRuntimeAction release_second = {
        .object_id = successor_object_id,
        .kind = SHADOWSPILL_RUNTIME_RELEASE,
    };
    const ShadowSpillTaskDescription release_task = {
        .task_id = 14U,
        .actions = &release_second,
        .action_count = 1U,
    };
    failed = failed || shadowspill_test_admit_fixed_layout(runtime, &layout) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_test_admit_task(runtime, &first_fetch_task) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_test_admit_task(runtime, &eviction_task) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_test_admit_task(runtime, &second_fetch_task) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_test_admit_task(runtime, &release_task) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_test_seal_fixed_layout(runtime) != SHADOWSPILL_RUNTIME_OK;

    const uint32_t invocation_count = same_object ? 128U : 1U;
    for (uint32_t invocation = 0U;
         !failed && invocation < invocation_count; ++invocation) {
        failed = shadowspill_test_before_task(
                runtime, first_fetch_task.task_id, compute, NULL, 0U
            ) != SHADOWSPILL_RUNTIME_OK ||
            shadowspill_test_after_task(
                runtime, first_fetch_task.task_id, compute
            ) != SHADOWSPILL_RUNTIME_OK ||
            shadowspill_runtime_wait_idle(runtime) != SHADOWSPILL_RUNTIME_OK;
        ShadowSpillObjectSnapshot first = {0};
        failed = failed || shadowspill_object_snapshot(
                runtime, objects[0].object_id, &first
            ) != SHADOWSPILL_RUNTIME_OK ||
            first.residency != SHADOWSPILL_OBJECT_EXECUTION_READY;

        failed = failed || shadowspill_test_before_task(
                runtime, eviction_task.task_id, compute, NULL, 0U
            ) != SHADOWSPILL_RUNTIME_OK ||
            shadowspill_test_after_task(
                runtime, eviction_task.task_id, compute
            ) != SHADOWSPILL_RUNTIME_OK ||
            shadowspill_test_before_task(
                runtime, second_fetch_task.task_id, compute, NULL, 0U
            ) != SHADOWSPILL_RUNTIME_OK ||
            shadowspill_test_after_task(
                runtime, second_fetch_task.task_id, compute
            ) != SHADOWSPILL_RUNTIME_OK ||
            shadowspill_runtime_wait_idle(runtime) != SHADOWSPILL_RUNTIME_OK;
        ShadowSpillObjectSnapshot second = {0};
        ShadowSpillRuntimeStatistics statistics = {0};
        failed = failed || shadowspill_object_snapshot(
                runtime, successor_object_id, &second
            ) != SHADOWSPILL_RUNTIME_OK ||
            second.residency != SHADOWSPILL_OBJECT_EXECUTION_READY ||
            second.execution_pointer != first.execution_pointer ||
            shadowspill_runtime_statistics(runtime, &statistics) !=
                SHADOWSPILL_RUNTIME_OK ||
            (!same_object && statistics.wait_events_inserted == 0U);

        failed = failed || shadowspill_test_before_task(
                runtime, release_task.task_id, compute, NULL, 0U
            ) != SHADOWSPILL_RUNTIME_OK ||
            shadowspill_test_after_task(
                runtime, release_task.task_id, compute
            ) != SHADOWSPILL_RUNTIME_OK ||
            shadowspill_runtime_wait_idle(runtime) != SHADOWSPILL_RUNTIME_OK;
    }
    failed = failed ||
        shadowspill_unregister_object(runtime, objects[0].object_id) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_unregister_object(runtime, objects[1].object_id) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_test_clear_plan(runtime) != SHADOWSPILL_RUNTIME_OK;

    shadowspill_test_destroy_runtime(runtime);
    if (compute.words[0] != 0U) {
        failed = failed ||
            shadowspill_mock_destroy_compute_stream(mock, compute) != 0;
    }
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
    if (empty_fixed_slice_allows_dynamic_task() != 0) {
        fprintf(stderr, "empty fixed slice rejected a dynamic task\n");
        return EXIT_FAILURE;
    }
    if (empty_fixed_slice_allows_dynamic_fetch() != 0) {
        fprintf(stderr, "empty fixed slice rejected a dynamic fetch\n");
        return EXIT_FAILURE;
    }
    if (eviction_completion_orders_fixed_reuse() != 0) {
        fprintf(stderr, "fixed eviction dependency failed\n");
        return EXIT_FAILURE;
    }
    if (eviction_completion_orders_fixed_fetch_reuse(0) != 0) {
        fprintf(stderr, "fixed fetch dependency failed\n");
        return EXIT_FAILURE;
    }
    if (eviction_completion_orders_fixed_fetch_reuse(1) != 0) {
        fprintf(stderr, "same-object fixed fetch dependency failed\n");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
