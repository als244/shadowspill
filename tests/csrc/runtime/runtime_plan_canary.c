#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <shadowspill/backend_mock.h>
#include <shadowspill/runtime.h>

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
    const ShadowSpillExecutionDescription task = {.task_id = 7U};
    const ShadowSpillExecutionHandle *first_handle = NULL;
    const ShadowSpillExecutionHandle *second_handle = NULL;
    int failed = shadowspill_runtime_create(&topology.runtime, &runtime) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_plan_create(runtime, &roles, &first) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_plan_create(runtime, &roles, &second) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_plan_admit_execution(first, &task) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_plan_admit_execution(second, &task) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_plan_resolve_execution(first, 7U, &first_handle) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_plan_resolve_execution(second, 7U, &second_handle) !=
            SHADOWSPILL_RUNTIME_OK ||
        first_handle == NULL || second_handle == NULL ||
        first_handle == second_handle ||
        shadowspill_mock_create_compute_stream(mock, &compute) != 0;
    if (!failed) {
        failed = shadowspill_before_execution_handle(
                runtime, first_handle, compute, NULL, 0U
            ) != SHADOWSPILL_RUNTIME_OK ||
            shadowspill_after_execution_handle(
                runtime, first_handle, compute
            ) != SHADOWSPILL_RUNTIME_OK ||
            shadowspill_before_execution_handle(
                runtime, second_handle, compute, NULL, 0U
            ) != SHADOWSPILL_RUNTIME_OK ||
            shadowspill_after_execution_handle(
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
    int failed = shadowspill_runtime_create(&config, &runtime) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_plan_create(runtime, &alternate, &plan) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_plan_create(runtime, &mismatched, &invalid) !=
            SHADOWSPILL_RUNTIME_INVALID_ARGUMENT ||
        invalid != NULL;
    shadowspill_plan_destroy(plan);
    shadowspill_runtime_destroy(runtime);
    shadowspill_mock_backend_destroy(mock);
    return failed ? -1 : 0;
}

int main(void) {
    if (shared_runtime_accepts_overlapping_plan_tasks() != 0) {
        fprintf(stderr, "runtime plan canary failed: shared runtime\n");
        return EXIT_FAILURE;
    }
    if (plan_selects_nondefault_pool_pair() != 0) {
        fprintf(stderr, "runtime plan canary failed: alternate pool pair\n");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
