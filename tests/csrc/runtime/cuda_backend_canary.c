#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <shadowspill/backend_cuda.h>
#include <shadowspill/runtime.h>

#define PAYLOAD_BYTES (4U << 20U)
#define FAIL(stage_)                                                         \
    do {                                                                     \
        fprintf(stderr, "cuda backend canary failed at %s\n", (stage_));   \
        return EXIT_FAILURE;                                                 \
    } while (0)

int main(void) {
    const ShadowSpillCudaBackendConfig backend_config = {
        .abi_version = SHADOWSPILL_CUDA_BACKEND_ABI_VERSION,
        .device_ordinal = 0,
    };
    ShadowSpillCudaBackend *cuda = NULL;
    if (shadowspill_cuda_backend_create(&backend_config, &cuda) != 0) {
        FAIL("backend creation");
    }
    ShadowSpillCudaBackendCapabilities capabilities = {0};
    if (shadowspill_cuda_backend_capabilities(cuda, &capabilities) != 0 ||
        capabilities.abi_version != SHADOWSPILL_CUDA_BACKEND_ABI_VERSION ||
        capabilities.total_device_bytes == 0U ||
        capabilities.recommended_minimum_alignment == 0U ||
        !capabilities.process_memory_accounting) {
        FAIL("capability validation");
    }
    ShadowSpillCudaPhysicalMemory context_memory = {0};
    if (shadowspill_cuda_physical_memory(cuda, &context_memory) != 0 ||
        context_memory.abi_version != SHADOWSPILL_CUDA_BACKEND_ABI_VERSION ||
        context_memory.process_bytes == 0U ||
        context_memory.device_used_bytes < context_memory.process_bytes ||
        context_memory.device_total_bytes < capabilities.total_device_bytes) {
        FAIL("initial physical-memory accounting");
    }
    const uint64_t execution_pool_bytes = 8U << 20U;
    const ShadowSpillMemoryPoolDescription pools[] = {
        {
            .pool_id = 0U,
            .capacity_bytes = execution_pool_bytes,
            .minimum_alignment = capabilities.recommended_minimum_alignment,
            .backend = shadowspill_cuda_device_pool_backend(cuda),
        },
        {
            .pool_id = 1U,
            .capacity_bytes = 8U << 20U,
            .minimum_alignment = 1U,
            .backend = shadowspill_cuda_pinned_pool_backend(cuda),
        },
    };
    const ShadowSpillTransferRouteDescription routes[] = {
        {
            .route_id = 0U,
            .name = "shadowspill_fetch",
            .route = shadowspill_cuda_fetch_route(cuda, 1U, 0U),
        },
        {
            .route_id = 1U,
            .name = "shadowspill_evict",
            .route = shadowspill_cuda_evict_route(cuda, 0U, 1U),
        },
    };
    const ShadowSpillRuntimeConfig runtime_config = {
        .abi_version = SHADOWSPILL_RUNTIME_ABI_VERSION,
        .pools = pools,
        .pool_count = 2U,
        .routes = routes,
        .route_count = 2U,
        .worker_poll_nanoseconds = 10000U,
        .synchronization = shadowspill_cuda_synchronization_backend(cuda),
    };
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillBackendStream compute = {{0U, 0U}};
    if (shadowspill_runtime_create(&runtime_config, &runtime) !=
            SHADOWSPILL_RUNTIME_OK ||
        routes[0].route.create_lane(routes[0].route.context, &compute) != 0 ||
        shadowspill_runtime_reserve_event_leases(runtime, 8U) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_cuda_backend_seal_event_pool(cuda, 8U) != 0 ||
        shadowspill_runtime_reserve_event_leases(runtime, 12U) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_cuda_backend_seal_event_pool(cuda, 12U) != 0) {
        FAIL("runtime or stream initialization");
    }
    ShadowSpillCudaPhysicalMemory admitted_memory = {0};
    if (shadowspill_cuda_physical_memory(cuda, &admitted_memory) != 0 ||
        admitted_memory.process_bytes <
            context_memory.process_bytes + execution_pool_bytes) {
        FAIL("admitted physical-memory accounting");
    }
    unsigned char *original = malloc(PAYLOAD_BYTES);
    unsigned char *restored = calloc(PAYLOAD_BYTES, 1U);
    if (original == NULL || restored == NULL) {
        FAIL("host payload allocation");
    }
    for (uint64_t index = 0U; index < PAYLOAD_BYTES; ++index) {
        original[index] = (unsigned char)((index * 17U + 29U) & 0xffU);
    }
    const ShadowSpillObjectDescription object = {
        .object_id = 1U,
        .size_bytes = PAYLOAD_BYTES,
        .initial_version = 1U,
        .retain_spill_copy = 1U,
        .initial_pool_id = 1U,
        .initially_resident = 1U,
    };
    const ShadowSpillRuntimeAction prefetch = {
        .object_id = object.object_id,
        .kind = SHADOWSPILL_RUNTIME_PREFETCH,
    };
    ShadowSpillPlan *plan = NULL;
    const ShadowSpillPlanDescription plan_description = {
        .execution_pool_id = 0U,
        .spill_pool_id = 1U,
        .fetch_route_id = 0U,
        .evict_route_id = 1U,
    };
    ShadowSpillObjectHandle *object_handle = NULL;
    const ShadowSpillActionBatchHandle *initial_actions = NULL;
    if (shadowspill_register_object(runtime, &object) !=
            SHADOWSPILL_RUNTIME_OK ||
        shadowspill_write_object(
            runtime, object.object_id, 1U, original, PAYLOAD_BYTES
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_plan_create(
            runtime, &plan_description, &plan
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_object_handle_acquire(
            runtime, object.object_id, &object_handle
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_plan_bind_object(
            plan, object.object_id, object_handle, SHADOWSPILL_OBJECT_CAUSAL
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_object_handle_release(
            object_handle
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_plan_admit_action_batch(
            plan, 0U, &prefetch, 1U, &initial_actions
        ) != SHADOWSPILL_RUNTIME_OK || shadowspill_submit_action_batch_handle(
            runtime, initial_actions, compute
        ) != SHADOWSPILL_RUNTIME_OK) {
        FAIL("initial object registration or fetch submission");
    }
    const uint64_t input_object_id = object.object_id;
    const ShadowSpillRuntimeAction offload = {
        .object_id = object.object_id,
        .kind = SHADOWSPILL_RUNTIME_OFFLOAD,
    };
    const ShadowSpillTaskDescription task = {
        .task_id = 1U,
        .input_object_ids = &input_object_id,
        .input_count = 1U,
        .actions = &offload,
        .action_count = 1U,
    };
    const ShadowSpillTaskHandle *task_handle = NULL;
    const ShadowSpillObjectBinding *bindings = NULL;
    uint32_t binding_count = 0U;
    if (shadowspill_plan_admit_task(plan, &task, &task_handle) !=
            SHADOWSPILL_RUNTIME_OK || shadowspill_before_task_handle(
            runtime,
            task_handle,
            compute,
            &bindings,
            &binding_count
        ) != SHADOWSPILL_RUNTIME_OK ||
        binding_count != 1U || bindings == NULL ||
        bindings[0].pointer == NULL ||
        bindings[0].authoritative_version != 1U) {
        FAIL("fetch readiness");
    }
    const ShadowSpillRuntimeStatus after_status = shadowspill_after_task_handle(
        runtime, task_handle, compute
    );
    const ShadowSpillRuntimeStatus idle_status = after_status ==
            SHADOWSPILL_RUNTIME_OK
        ? shadowspill_runtime_wait_idle(runtime)
        : after_status;
    const ShadowSpillRuntimeStatus read_status = idle_status ==
            SHADOWSPILL_RUNTIME_OK
        ? shadowspill_read_object(
            runtime, object.object_id, 1U, restored, PAYLOAD_BYTES
        )
        : idle_status;
    if (after_status != SHADOWSPILL_RUNTIME_OK ||
        idle_status != SHADOWSPILL_RUNTIME_OK ||
        read_status != SHADOWSPILL_RUNTIME_OK ||
        memcmp(original, restored, PAYLOAD_BYTES) != 0) {
        ShadowSpillRuntimeFailure failure = {0};
        (void)shadowspill_runtime_failure(runtime, &failure);
        fprintf(
            stderr,
            "after=%u idle=%u read=%u failure=%u task=%llu object=%llu "
            "allocation=%llu\n",
            (unsigned)after_status,
            (unsigned)idle_status,
            (unsigned)read_status,
            (unsigned)failure.status,
            (unsigned long long)failure.task_id,
            (unsigned long long)failure.object_id,
            (unsigned long long)failure.allocation_id
        );
        fprintf(
            stderr,
            "cuda_error=%u nvml_error=%u\n",
            shadowspill_cuda_backend_last_error(cuda),
            shadowspill_cuda_backend_last_nvml_error(cuda)
        );
        FAIL("eviction completion or data validation");
    }
    ShadowSpillRuntimeStatistics runtime_statistics = {0};
    ShadowSpillCudaBackendStatistics cuda_statistics = {0};
    if (shadowspill_runtime_statistics(runtime, &runtime_statistics) !=
            SHADOWSPILL_RUNTIME_OK ||
        runtime_statistics.fetch_transfers != 1U ||
        runtime_statistics.evict_transfers != 1U ||
        runtime_statistics.event_lease_capacity != 12U ||
        runtime_statistics.event_lease_in_use != 0U ||
        runtime_statistics.event_lease_peak_in_use == 0U ||
        runtime_statistics.event_lease_growth_rejections != 0U ||
        runtime_statistics.bytes_fetched != PAYLOAD_BYTES ||
        runtime_statistics.bytes_evicted != PAYLOAD_BYTES) {
        FAIL("runtime statistics");
    }
    shadowspill_cuda_backend_statistics(cuda, &cuda_statistics);
    if (cuda_statistics.device_allocations != 1U ||
        cuda_statistics.pinned_host_allocations != 1U ||
        cuda_statistics.fetch_copies != 1U ||
        cuda_statistics.evict_copies != 1U ||
        cuda_statistics.bytes_fetched != PAYLOAD_BYTES ||
        cuda_statistics.bytes_evicted != PAYLOAD_BYTES) {
        FAIL("backend transfer statistics");
    }
    if (shadowspill_plan_close(plan) != SHADOWSPILL_RUNTIME_OK ||
        shadowspill_runtime_close(runtime) != SHADOWSPILL_RUNTIME_OK ||
        routes[0].route.destroy_lane(routes[0].route.context, compute) != 0) {
        FAIL("runtime close");
    }
    shadowspill_cuda_backend_statistics(cuda, &cuda_statistics);
    if (cuda_statistics.device_frees != 1U ||
        cuda_statistics.pinned_host_frees != 1U ||
        cuda_statistics.streams_created != 3U ||
        cuda_statistics.streams_destroyed != 3U ||
        cuda_statistics.events_created != cuda_statistics.events_destroyed ||
        !cuda_statistics.event_pool_sealed ||
        cuda_statistics.event_pool_capacity < 12U ||
        cuda_statistics.event_pool_driver_creates !=
            cuda_statistics.event_pool_capacity ||
        cuda_statistics.event_pool_in_use != 0U ||
        cuda_statistics.event_pool_peak_in_use == 0U ||
        cuda_statistics.event_pool_growth_rejections != 0U ||
        shadowspill_cuda_backend_last_error(cuda) != 0U ||
        shadowspill_cuda_backend_last_nvml_error(cuda) != 0U) {
        FAIL("backend teardown statistics");
    }
    free(original);
    free(restored);
    shadowspill_plan_destroy(plan);
    shadowspill_runtime_destroy(runtime);
    shadowspill_cuda_backend_destroy(cuda);
    return EXIT_SUCCESS;
}
