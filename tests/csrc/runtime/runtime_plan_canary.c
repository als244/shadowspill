#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <shadowspill/backend_mock.h>
#include <shadowspill/runtime.h>

static int runtime_accepts_generic_and_sparse_topologies(void) {
    ShadowSpillMockBackend *mock = NULL;
    const ShadowSpillMockBackendConfig mock_config = {
        .abi_version = SHADOWSPILL_MOCK_BACKEND_ABI_VERSION,
    };
    if (shadowspill_mock_backend_create(&mock_config, &mock) != 0) {
        return -1;
    }

    ShadowSpillMockRuntimeTopology single_pool;
    shadowspill_mock_runtime_topology(
        mock, 4096U, 4096U, 16U, 1000U, &single_pool
    );
    single_pool.runtime.pool_count = 1U;
    single_pool.runtime.routes = NULL;
    single_pool.runtime.route_count = 0U;
    ShadowSpillRuntime *runtime = NULL;
    int failed = shadowspill_runtime_create(&single_pool.runtime, &runtime) !=
        SHADOWSPILL_RUNTIME_OK;
    shadowspill_runtime_destroy(runtime);
    runtime = NULL;

    ShadowSpillMemoryPoolDescription pools[3] = {
        {
            .pool_id = 0U,
            .capacity_bytes = 4096U,
            .minimum_alignment = 16U,
            .backend = shadowspill_mock_execution_pool_backend(mock),
        },
        {
            .pool_id = 1U,
            .capacity_bytes = 4096U,
            .minimum_alignment = 16U,
            .backend = shadowspill_mock_spill_pool_backend(mock),
        },
        {
            .pool_id = 2U,
            .capacity_bytes = 4096U,
            .minimum_alignment = 16U,
            .backend = shadowspill_mock_spill_pool_backend(mock),
        },
    };
    const ShadowSpillTransferRouteDescription route = {
        .route_id = 0U,
        .name = "sparse_fetch",
        .route = shadowspill_mock_fetch_route(mock, 2U, 0U),
    };
    const ShadowSpillRuntimeConfig sparse = {
        .abi_version = SHADOWSPILL_RUNTIME_ABI_VERSION,
        .pools = pools,
        .pool_count = 3U,
        .routes = &route,
        .route_count = 1U,
        .worker_poll_nanoseconds = 1000U,
        .synchronization = shadowspill_mock_synchronization_backend(mock),
    };
    ShadowSpillPlan *plan = NULL;
    ShadowSpillBackendStream compute = {{0U, 0U}};
    ShadowSpillAllocation allocation = {0};
    failed = failed || shadowspill_runtime_create(&sparse, &runtime) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_mock_create_compute_stream(mock, &compute) != 0 ||
        shadowspill_memory_pool_allocate(
            runtime, 2U, 64U, 16U, compute, &allocation
        ) != SHADOWSPILL_RUNTIME_OK ||
        allocation.pool_id != 2U ||
        shadowspill_memory_pool_free(
            runtime, 2U, allocation.allocation_id, compute
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_wait_idle(runtime) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_plan_create_for_pools(runtime, 0U, 2U, &plan) !=
            SHADOWSPILL_RUNTIME_INVALID_ARGUMENT ||
        plan != NULL;
    if (compute.words[0] != 0U) {
        (void)shadowspill_mock_destroy_compute_stream(mock, compute);
    }
    shadowspill_runtime_destroy(runtime);
    shadowspill_mock_backend_destroy(mock);
    return failed ? -1 : 0;
}

static int shared_runtime_accepts_overlapping_plan_tasks(void) {
    ShadowSpillMockBackend *mock = NULL;
    const ShadowSpillMockBackendConfig mock_config = {
        .abi_version = SHADOWSPILL_MOCK_BACKEND_ABI_VERSION,
    };
    if (shadowspill_mock_backend_create(&mock_config, &mock) != 0) {
        return -1;
    }
    ShadowSpillMockRuntimeTopology topology;
    shadowspill_mock_runtime_topology(mock, 4096U, 4096U, 16U, 1000U, &topology);
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillPlan *first = NULL;
    ShadowSpillPlan *second = NULL;
    ShadowSpillBackendStream compute = {{0U, 0U}};
    const ShadowSpillPlanDescription roles = {
        .execution_pool_id = 0U,
        .spill_pool_id = 1U,
        .fetch_route_id = 0U,
        .evict_route_id = 1U,
    };
    const ShadowSpillTaskDescription task = {.task_id = 7U};
    const ShadowSpillTaskHandle *first_handle = NULL;
    const ShadowSpillTaskHandle *second_handle = NULL;
    int failed = shadowspill_runtime_create(&topology.runtime, &runtime) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_plan_create(runtime, &roles, &first) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_plan_create(runtime, &roles, &second) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_plan_admit_task(first, &task, &first_handle) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_plan_admit_task(second, &task, &second_handle) !=
            SHADOWSPILL_RUNTIME_OK ||
        first_handle == NULL || second_handle == NULL ||
        first_handle == second_handle ||
        shadowspill_mock_create_compute_stream(mock, &compute) != 0;
    if (!failed) {
        failed = shadowspill_before_task_handle(
                runtime, first_handle, compute, NULL, 0U
            ) != SHADOWSPILL_RUNTIME_OK ||
            shadowspill_after_task_handle(
                runtime, first_handle, compute
            ) != SHADOWSPILL_RUNTIME_OK ||
            shadowspill_before_task_handle(
                runtime, second_handle, compute, NULL, 0U
            ) != SHADOWSPILL_RUNTIME_OK ||
            shadowspill_after_task_handle(
                runtime, second_handle, compute
            ) != SHADOWSPILL_RUNTIME_OK ||
            shadowspill_runtime_wait_idle(runtime) != SHADOWSPILL_RUNTIME_OK;
    }
    if (first != NULL) {
        failed = shadowspill_plan_close(first) != SHADOWSPILL_RUNTIME_OK || failed;
        shadowspill_plan_destroy(first);
    }
    if (second != NULL) {
        failed = shadowspill_plan_close(second) != SHADOWSPILL_RUNTIME_OK || failed;
        shadowspill_plan_destroy(second);
    }
    if (compute.words[0] != 0U) {
        (void)shadowspill_mock_destroy_compute_stream(mock, compute);
    }
    shadowspill_runtime_destroy(runtime);
    shadowspill_mock_backend_destroy(mock);
    return failed ? -1 : 0;
}

static int plan_selects_nondefault_pool_pair(void) {
    ShadowSpillMockBackend *mock = NULL;
    const ShadowSpillMockBackendConfig mock_config = {
        .abi_version = SHADOWSPILL_MOCK_BACKEND_ABI_VERSION,
    };
    if (shadowspill_mock_backend_create(&mock_config, &mock) != 0) {
        return -1;
    }
    ShadowSpillMemoryPoolDescription pools[3] = {
        {
            .pool_id = 0U,
            .capacity_bytes = 4096U,
            .minimum_alignment = 16U,
            .backend = shadowspill_mock_execution_pool_backend(mock),
        },
        {
            .pool_id = 1U,
            .capacity_bytes = 4096U,
            .minimum_alignment = 16U,
            .backend = shadowspill_mock_spill_pool_backend(mock),
        },
        {
            .pool_id = 2U,
            .capacity_bytes = 4096U,
            .minimum_alignment = 16U,
            .backend = shadowspill_mock_spill_pool_backend(mock),
        },
    };
    ShadowSpillTransferRouteDescription routes[4] = {
        {
            .route_id = 0U,
            .name = "default_fetch",
            .route = shadowspill_mock_fetch_route(mock, 1U, 0U),
        },
        {
            .route_id = 1U,
            .name = "default_evict",
            .route = shadowspill_mock_evict_route(mock, 0U, 1U),
        },
        {
            .route_id = 2U,
            .name = "alternate_fetch",
            .route = shadowspill_mock_fetch_route(mock, 2U, 0U),
        },
        {
            .route_id = 3U,
            .name = "alternate_evict",
            .route = shadowspill_mock_evict_route(mock, 0U, 2U),
        },
    };
    const ShadowSpillRuntimeConfig config = {
        .abi_version = SHADOWSPILL_RUNTIME_ABI_VERSION,
        .pools = pools,
        .pool_count = 3U,
        .routes = routes,
        .route_count = 4U,
        .worker_poll_nanoseconds = 1000U,
        .synchronization = shadowspill_mock_synchronization_backend(mock),
    };
    const ShadowSpillPlanDescription alternate = {
        .execution_pool_id = 0U,
        .spill_pool_id = 2U,
        .fetch_route_id = 2U,
        .evict_route_id = 3U,
    };
    const ShadowSpillPlanDescription mismatched = {
        .execution_pool_id = 0U,
        .spill_pool_id = 2U,
        .fetch_route_id = 0U,
        .evict_route_id = 3U,
    };
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillPlan *plan = NULL;
    ShadowSpillPlan *invalid = NULL;
    ShadowSpillObjectHandle *object_handle = NULL;
    ShadowSpillBackendStream compute = {{0U, 0U}};
    const ShadowSpillObjectDescription object = {
        .object_id = 7007U,
        .size_bytes = 64U,
        .initial_pool_id = 2U,
        .initially_resident = 1U,
    };
    uint8_t payload[64];
    uint8_t restored[64] = {0};
    for (uint32_t index = 0U; index < sizeof(payload); ++index) {
        payload[index] = (uint8_t)(index * 11U + 5U);
    }
    const ShadowSpillRuntimeAction fetch = {
        .object_id = 9U,
        .kind = SHADOWSPILL_RUNTIME_PREFETCH,
    };
    const ShadowSpillRuntimeAction evict = {
        .object_id = 9U,
        .kind = SHADOWSPILL_RUNTIME_OFFLOAD,
    };
    const ShadowSpillActionBatchHandle *initial_actions = NULL;
    const uint64_t input = 9U;
    const ShadowSpillTaskDescription task = {
        .task_id = 1U,
        .input_object_ids = &input,
        .input_count = 1U,
        .actions = &evict,
        .action_count = 1U,
    };
    const ShadowSpillTaskHandle *task_handle = NULL;
    ShadowSpillObjectBinding binding = {0};
    ShadowSpillAllocation reclaimed = {0};
    int failed = shadowspill_runtime_create(&config, &runtime) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_plan_create(runtime, &alternate, &plan) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_plan_create(runtime, &mismatched, &invalid) !=
            SHADOWSPILL_RUNTIME_INVALID_ARGUMENT ||
        invalid != NULL ||
        shadowspill_register_object(runtime, &object) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_write_object(
            runtime, object.object_id, 2U, payload, sizeof(payload)
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_object_handle_acquire(
            runtime, object.object_id, &object_handle
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_plan_bind_object(
            plan, 9U, object_handle, SHADOWSPILL_OBJECT_CAUSAL
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_plan_admit_action_batch(
            plan, 0U, &fetch, 1U, &initial_actions
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_plan_admit_task(plan, &task, &task_handle) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_mock_create_compute_stream(mock, &compute) != 0;
    if (!failed) {
        failed = shadowspill_submit_action_batch_handle(
                runtime, initial_actions, compute
            ) != SHADOWSPILL_RUNTIME_OK ||
            shadowspill_before_task_handle(
                runtime, task_handle, compute, &binding, 1U
            ) != SHADOWSPILL_RUNTIME_OK ||
            binding.pointer == NULL ||
            memcmp(binding.pointer, payload, sizeof(payload)) != 0 ||
            shadowspill_after_task_handle(runtime, task_handle, compute) !=
                SHADOWSPILL_RUNTIME_OK ||
            shadowspill_runtime_wait_idle(runtime) !=
                SHADOWSPILL_RUNTIME_OK ||
            shadowspill_read_object(
                runtime, object.object_id, 2U, restored, sizeof(restored)
            ) != SHADOWSPILL_RUNTIME_OK ||
            memcmp(restored, payload, sizeof(payload)) != 0;
    }
    if (object_handle != NULL) {
        (void)shadowspill_object_handle_release(object_handle);
    }
    shadowspill_plan_destroy(plan);
    if (runtime != NULL && !failed) {
        failed = shadowspill_unregister_object(runtime, object.object_id) !=
                SHADOWSPILL_RUNTIME_OK ||
            shadowspill_memory_pool_allocate(
                runtime, 2U, 4096U, 16U, compute, &reclaimed
            ) != SHADOWSPILL_RUNTIME_OK ||
            shadowspill_memory_pool_free(
                runtime, 2U, reclaimed.allocation_id, compute
            ) != SHADOWSPILL_RUNTIME_OK ||
            shadowspill_runtime_wait_idle(runtime) !=
                SHADOWSPILL_RUNTIME_OK;
    } else if (runtime != NULL) {
        (void)shadowspill_unregister_object(runtime, object.object_id);
    }
    if (compute.words[0] != 0U) {
        (void)shadowspill_mock_destroy_compute_stream(mock, compute);
    }
    shadowspill_runtime_destroy(runtime);
    shadowspill_mock_backend_destroy(mock);
    return failed ? -1 : 0;
}

static int plans_bind_local_ids_to_explicit_runtime_objects(void) {
    ShadowSpillMockBackend *mock = NULL;
    const ShadowSpillMockBackendConfig mock_config = {
        .abi_version = SHADOWSPILL_MOCK_BACKEND_ABI_VERSION,
    };
    if (shadowspill_mock_backend_create(&mock_config, &mock) != 0) {
        return -1;
    }
    ShadowSpillMockRuntimeTopology topology;
    shadowspill_mock_runtime_topology(mock, 4096U, 4096U, 16U, 1000U, &topology);
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillPlan *first = NULL;
    ShadowSpillPlan *second = NULL;
    ShadowSpillObjectHandle *first_object_handle = NULL;
    ShadowSpillObjectHandle *second_object_handle = NULL;
    const ShadowSpillPlanDescription roles = {
        .execution_pool_id = 0U,
        .spill_pool_id = 1U,
        .fetch_route_id = 0U,
        .evict_route_id = 1U,
    };
    const ShadowSpillObjectDescription first_object = {
        .object_id = 1001U,
        .size_bytes = 64U,
        .initial_pool_id = 1U,
        .initially_resident = 1U,
    };
    const ShadowSpillObjectDescription second_object = {
        .object_id = 2002U,
        .size_bytes = 64U,
        .initial_pool_id = 1U,
        .initially_resident = 1U,
    };
    const uint64_t input = 7U;
    const ShadowSpillTaskDescription task = {
        .task_id = 11U,
        .input_object_ids = &input,
        .input_count = 1U,
    };
    const ShadowSpillTaskHandle *first_task = NULL;
    const ShadowSpillTaskHandle *second_task = NULL;
    int failed = shadowspill_runtime_create(&topology.runtime, &runtime) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_register_object(runtime, &first_object) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_register_object(runtime, &second_object) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_plan_create(runtime, &roles, &first) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_plan_create(runtime, &roles, &second) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_object_handle_acquire(
            runtime, 1001U, &first_object_handle
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_object_handle_acquire(
            runtime, 2002U, &second_object_handle
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_plan_bind_object(
            first, 7U, first_object_handle, SHADOWSPILL_OBJECT_CAUSAL
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_plan_bind_object(
            second, 7U, second_object_handle, SHADOWSPILL_OBJECT_CAUSAL
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_plan_admit_task(first, &task, &first_task) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_plan_admit_task(second, &task, &second_task) !=
            SHADOWSPILL_RUNTIME_OK ||
        first_task == NULL || second_task == NULL ||
        shadowspill_plan_bind_object(
            first, 7U, second_object_handle, SHADOWSPILL_OBJECT_CAUSAL
        ) != SHADOWSPILL_RUNTIME_INVALID_STATE;
    shadowspill_object_handle_release(first_object_handle);
    shadowspill_object_handle_release(second_object_handle);
    shadowspill_plan_destroy(first);
    shadowspill_plan_destroy(second);
    shadowspill_runtime_destroy(runtime);
    shadowspill_mock_backend_destroy(mock);
    return failed ? -1 : 0;
}

static int task_publications_resolve_plan_local_objects_once(void) {
    ShadowSpillMockBackend *mock = NULL;
    const ShadowSpillMockBackendConfig mock_config = {
        .abi_version = SHADOWSPILL_MOCK_BACKEND_ABI_VERSION,
    };
    if (shadowspill_mock_backend_create(&mock_config, &mock) != 0) {
        return -1;
    }
    ShadowSpillMockRuntimeTopology topology;
    shadowspill_mock_runtime_topology(mock, 4096U, 4096U, 16U, 1000U, &topology);
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillPlan *first = NULL;
    ShadowSpillPlan *second = NULL;
    ShadowSpillObjectHandle *first_object_handle = NULL;
    ShadowSpillObjectHandle *second_object_handle = NULL;
    ShadowSpillBackendStream compute = {{0U, 0U}};
    const ShadowSpillPlanDescription roles = {
        .execution_pool_id = 0U,
        .spill_pool_id = 1U,
        .fetch_route_id = 0U,
        .evict_route_id = 1U,
    };
    const ShadowSpillObjectDescription first_object = {
        .object_id = 3101U,
        .size_bytes = 64U,
        .initial_pool_id = 1U,
    };
    const ShadowSpillObjectDescription second_object = {
        .object_id = 3202U,
        .size_bytes = 64U,
        .initial_pool_id = 1U,
    };
    const ShadowSpillTaskPublicationDescription publication = {
        .object_id = 5U,
        .kind = SHADOWSPILL_TASK_PUBLICATION_BIND,
    };
    const ShadowSpillTaskDescription task = {
        .task_id = 17U,
        .publications = &publication,
        .publication_count = 1U,
    };
    const ShadowSpillTaskHandle *first_task = NULL;
    const ShadowSpillTaskHandle *second_task = NULL;
    ShadowSpillAllocation first_allocation = {0};
    ShadowSpillAllocation second_allocation = {0};
    ShadowSpillObjectBinding first_binding = {0};
    ShadowSpillObjectBinding second_binding = {0};
    int failed = shadowspill_runtime_create(&topology.runtime, &runtime) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_register_object(runtime, &first_object) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_register_object(runtime, &second_object) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_object_handle_acquire(
            runtime, first_object.object_id, &first_object_handle
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_object_handle_acquire(
            runtime, second_object.object_id, &second_object_handle
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_plan_create(runtime, &roles, &first) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_plan_create(runtime, &roles, &second) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_plan_bind_object(
            first, 5U, first_object_handle, SHADOWSPILL_OBJECT_CAUSAL
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_plan_bind_object(
            second, 5U, second_object_handle, SHADOWSPILL_OBJECT_CAUSAL
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_plan_admit_task(first, &task, &first_task) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_plan_admit_task(second, &task, &second_task) !=
            SHADOWSPILL_RUNTIME_OK ||
        first_task == NULL || second_task == NULL || first_task == second_task ||
        shadowspill_mock_create_compute_stream(mock, &compute) != 0;
    if (!failed) {
        failed = shadowspill_before_task_handle(
                runtime, first_task, compute, NULL, 0U
            ) != SHADOWSPILL_RUNTIME_OK ||
            shadowspill_memory_pool_allocate(
                runtime, 0U, 64U, 16U, compute, &first_allocation
            ) != SHADOWSPILL_RUNTIME_OK ||
            shadowspill_task_publish_allocation(
                runtime,
                first_task,
                0U,
                first_allocation.pointer,
                &first_binding
            ) != SHADOWSPILL_RUNTIME_OK ||
            first_binding.object_id != first_object.object_id ||
            shadowspill_task_publish_allocation(
                runtime,
                first_task,
                1U,
                first_allocation.pointer,
                &first_binding
            ) != SHADOWSPILL_RUNTIME_INVALID_ARGUMENT ||
            shadowspill_after_task_handle(runtime, first_task, compute) !=
                SHADOWSPILL_RUNTIME_OK ||
            shadowspill_before_task_handle(
                runtime, second_task, compute, NULL, 0U
            ) != SHADOWSPILL_RUNTIME_OK ||
            shadowspill_memory_pool_allocate(
                runtime, 0U, 64U, 16U, compute, &second_allocation
            ) != SHADOWSPILL_RUNTIME_OK ||
            shadowspill_task_publish_allocation(
                runtime,
                second_task,
                0U,
                second_allocation.pointer,
                &second_binding
            ) != SHADOWSPILL_RUNTIME_OK ||
            second_binding.object_id != second_object.object_id ||
            second_binding.object_id == first_binding.object_id ||
            shadowspill_after_task_handle(runtime, second_task, compute) !=
                SHADOWSPILL_RUNTIME_OK ||
            shadowspill_runtime_wait_idle(runtime) != SHADOWSPILL_RUNTIME_OK;
    }
    if (first_object_handle != NULL) {
        (void)shadowspill_object_handle_release(first_object_handle);
    }
    if (second_object_handle != NULL) {
        (void)shadowspill_object_handle_release(second_object_handle);
    }
    shadowspill_plan_destroy(first);
    shadowspill_plan_destroy(second);
    if (runtime != NULL) {
        (void)shadowspill_unregister_object(runtime, first_object.object_id);
        (void)shadowspill_unregister_object(runtime, second_object.object_id);
    }
    if (compute.words[0] != 0U) {
        (void)shadowspill_mock_destroy_compute_stream(mock, compute);
    }
    shadowspill_runtime_destroy(runtime);
    shadowspill_mock_backend_destroy(mock);
    return failed ? -1 : 0;
}

static int runtime_objects_survive_until_their_final_owner_closes(void) {
    ShadowSpillMockBackend *mock = NULL;
    const ShadowSpillMockBackendConfig mock_config = {
        .abi_version = SHADOWSPILL_MOCK_BACKEND_ABI_VERSION,
    };
    if (shadowspill_mock_backend_create(&mock_config, &mock) != 0) {
        return -1;
    }
    ShadowSpillMockRuntimeTopology topology;
    shadowspill_mock_runtime_topology(mock, 4096U, 4096U, 16U, 1000U, &topology);
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillPlan *first = NULL;
    ShadowSpillPlan *second = NULL;
    ShadowSpillObjectHandle *public_reference = NULL;
    const ShadowSpillPlanDescription roles = {
        .execution_pool_id = 0U,
        .spill_pool_id = 1U,
        .fetch_route_id = 0U,
        .evict_route_id = 1U,
    };
    const ShadowSpillObjectDescription object = {
        .object_id = 9001U,
        .size_bytes = 64U,
        .initial_pool_id = 1U,
        .initially_resident = 1U,
    };
    ShadowSpillObjectSnapshot snapshot = {0};
    ShadowSpillRuntimeStatistics statistics = {0};
    int failed = shadowspill_runtime_create(&topology.runtime, &runtime) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_register_object(runtime, &object) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_object_handle_acquire(
            runtime, object.object_id, &public_reference
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_plan_create(runtime, &roles, &first) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_plan_create(runtime, &roles, &second) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_plan_bind_object(
            first, 1U, public_reference, SHADOWSPILL_OBJECT_CAUSAL
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_plan_bind_object(
            second, 2U, public_reference, SHADOWSPILL_OBJECT_CAUSAL
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_unregister_object(runtime, object.object_id) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_object_snapshot(
            runtime, object.object_id, &snapshot
        ) != SHADOWSPILL_RUNTIME_OK;
    if (!failed) {
        failed = shadowspill_object_release_generation(
                public_reference, snapshot.generation + 1U
            ) != SHADOWSPILL_RUNTIME_INVALID_STATE ||
            shadowspill_object_release_generation(
                public_reference, snapshot.generation
            ) != SHADOWSPILL_RUNTIME_OK ||
            shadowspill_object_snapshot(
                runtime, object.object_id, &snapshot
            ) != SHADOWSPILL_RUNTIME_OK ||
            snapshot.residency != SHADOWSPILL_OBJECT_RELEASED ||
            shadowspill_runtime_statistics(runtime, &statistics) !=
                SHADOWSPILL_RUNTIME_OK ||
            statistics.spill_allocated_bytes != 0U;
    }
    if (!failed) {
        shadowspill_plan_destroy(first);
        first = NULL;
        failed = shadowspill_object_snapshot(
                runtime, object.object_id, &snapshot
            ) != SHADOWSPILL_RUNTIME_OK;
    }
    if (!failed) {
        shadowspill_plan_destroy(second);
        second = NULL;
        failed = shadowspill_object_snapshot(
                runtime, object.object_id, &snapshot
            ) != SHADOWSPILL_RUNTIME_OK ||
            shadowspill_object_handle_release(public_reference) !=
                SHADOWSPILL_RUNTIME_OK;
        public_reference = NULL;
    }
    if (!failed) {
        failed = shadowspill_object_snapshot(
                runtime, object.object_id, &snapshot
            ) != SHADOWSPILL_RUNTIME_INVALID_STATE ||
            shadowspill_runtime_statistics(runtime, &statistics) !=
                SHADOWSPILL_RUNTIME_OK ||
            statistics.registered_objects != 0U ||
            statistics.spill_allocated_bytes != 0U;
    }
    if (public_reference != NULL) {
        (void)shadowspill_object_handle_release(public_reference);
    }
    shadowspill_plan_destroy(first);
    shadowspill_plan_destroy(second);
    shadowspill_runtime_destroy(runtime);
    shadowspill_mock_backend_destroy(mock);
    return failed ? -1 : 0;
}

static int dedicated_action_and_acquisition_handles_are_not_tasks(void) {
    ShadowSpillMockBackend *mock = NULL;
    const ShadowSpillMockBackendConfig mock_config = {
        .abi_version = SHADOWSPILL_MOCK_BACKEND_ABI_VERSION,
    };
    if (shadowspill_mock_backend_create(&mock_config, &mock) != 0) {
        return -1;
    }
    ShadowSpillMockRuntimeTopology topology;
    shadowspill_mock_runtime_topology(mock, 4096U, 4096U, 16U, 1000U, &topology);
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillPlan *plan = NULL;
    ShadowSpillObjectHandle *object_handle = NULL;
    ShadowSpillBackendStream compute = {{0U, 0U}};
    const ShadowSpillPlanDescription roles = {
        .execution_pool_id = 0U,
        .spill_pool_id = 1U,
        .fetch_route_id = 0U,
        .evict_route_id = 1U,
    };
    const ShadowSpillObjectDescription object = {
        .object_id = 3003U,
        .size_bytes = 64U,
        .initial_pool_id = 1U,
        .initially_resident = 1U,
    };
    const ShadowSpillRuntimeAction fetch = {
        .object_id = 9U,
        .kind = SHADOWSPILL_RUNTIME_PREFETCH,
    };
    const uint64_t requested_objects[2] = {9U, 9U};
    const ShadowSpillActionBatchHandle *batch = NULL;
    const ShadowSpillObjectAcquisitionHandle *acquisition = NULL;
    ShadowSpillObjectBinding bindings[2] = {{0}};
    ShadowSpillAllocation caller_allocation = {0};
    int failed = shadowspill_runtime_create(&topology.runtime, &runtime) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_register_object(runtime, &object) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_object_handle_acquire(
            runtime, object.object_id, &object_handle
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_plan_create(runtime, &roles, &plan) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_plan_bind_object(
            plan, 9U, object_handle, SHADOWSPILL_OBJECT_CAUSAL
        ) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_plan_admit_action_batch(
            plan, 77U, &fetch, 1U, &batch
        ) != SHADOWSPILL_RUNTIME_OK ||
        batch == NULL ||
        shadowspill_plan_admit_object_acquisition(
            plan, requested_objects, 2U, &acquisition
        ) != SHADOWSPILL_RUNTIME_OK ||
        acquisition == NULL ||
        shadowspill_mock_create_compute_stream(mock, &compute) != 0;
    if (!failed) {
        failed = shadowspill_submit_action_batch_handle(
                runtime, batch, compute
            ) != SHADOWSPILL_RUNTIME_OK ||
            shadowspill_acquire_objects_handle(
                runtime, acquisition, compute, bindings, 2U
            ) != SHADOWSPILL_RUNTIME_OK ||
            bindings[0].object_id != object.object_id ||
            bindings[0].pointer == NULL ||
            bindings[0].pointer != bindings[1].pointer ||
            bindings[0].generation != bindings[1].generation ||
            shadowspill_before_task_handle(
                runtime,
                (const ShadowSpillTaskHandle *)batch,
                compute,
                NULL,
                0U
            ) != SHADOWSPILL_RUNTIME_INVALID_ARGUMENT ||
            shadowspill_runtime_wait_idle(runtime) != SHADOWSPILL_RUNTIME_OK ||
            shadowspill_transfer_acquired_object_to_caller(
                runtime,
                acquisition,
                0U,
                compute,
                bindings[0].pointer,
                bindings[0].generation + 1U,
                bindings[0].allocation_id,
                &caller_allocation
            ) != SHADOWSPILL_RUNTIME_INVALID_STATE ||
            shadowspill_transfer_acquired_object_to_caller(
                runtime,
                acquisition,
                0U,
                compute,
                bindings[0].pointer,
                bindings[0].generation,
                bindings[0].allocation_id,
                &caller_allocation
            ) != SHADOWSPILL_RUNTIME_OK ||
            caller_allocation.pointer != bindings[0].pointer ||
            shadowspill_memory_pool_free(
                runtime,
                caller_allocation.pool_id,
                caller_allocation.allocation_id,
                compute
            ) != SHADOWSPILL_RUNTIME_OK ||
            shadowspill_runtime_wait_idle(runtime) != SHADOWSPILL_RUNTIME_OK;
    }
    if (object_handle != NULL) {
        (void)shadowspill_object_handle_release(object_handle);
    }
    shadowspill_plan_destroy(plan);
    if (runtime != NULL) {
        (void)shadowspill_unregister_object(runtime, object.object_id);
    }
    if (compute.words[0] != 0U) {
        (void)shadowspill_mock_destroy_compute_stream(mock, compute);
    }
    shadowspill_runtime_destroy(runtime);
    shadowspill_mock_backend_destroy(mock);
    return failed ? -1 : 0;
}

int main(void) {
    if (runtime_accepts_generic_and_sparse_topologies() != 0) {
        fprintf(stderr, "runtime plan canary failed: generic topology\n");
        return EXIT_FAILURE;
    }
    if (shared_runtime_accepts_overlapping_plan_tasks() != 0) {
        fprintf(stderr, "runtime plan canary failed: shared runtime\n");
        return EXIT_FAILURE;
    }
    if (plan_selects_nondefault_pool_pair() != 0) {
        fprintf(stderr, "runtime plan canary failed: alternate pool pair\n");
        return EXIT_FAILURE;
    }
    if (plans_bind_local_ids_to_explicit_runtime_objects() != 0) {
        fprintf(stderr, "runtime plan canary failed: object bindings\n");
        return EXIT_FAILURE;
    }
    if (task_publications_resolve_plan_local_objects_once() != 0) {
        fprintf(stderr, "runtime plan canary failed: task publications\n");
        return EXIT_FAILURE;
    }
    if (runtime_objects_survive_until_their_final_owner_closes() != 0) {
        fprintf(stderr, "runtime plan canary failed: object ownership\n");
        return EXIT_FAILURE;
    }
    if (dedicated_action_and_acquisition_handles_are_not_tasks() != 0) {
        fprintf(stderr, "runtime plan canary failed: dedicated boundaries\n");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
