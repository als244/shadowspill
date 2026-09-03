/* The accelerator backend end to end: the contract table over the real driver,
 * a runtime built on it, one fetch and one evict of a payload, and the byte
 * round trip that proves the copies. Needs a device. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <shadowspill/backend.h>
#include <shadowspill/runtime.h>

#define PAYLOAD_BYTES (4U << 20U)

#define FAIL(stage_)                                                         \
    do {                                                                     \
        fprintf(stderr, "device backend canary failed at %s\n", (stage_));  \
        return EXIT_FAILURE;                                                 \
    } while (0)

int main(void) {
    const ShadowSpillBackendConfig backend_config = {
        .abi_version = SHADOWSPILL_BACKEND_ABI_VERSION,
        .device_ordinal = 0,
    };
    ShadowSpillBackend backend = {0};
    if (shadowspill_backend_create(&backend_config, &backend) != 0) {
        FAIL("backend creation");
    }
    ShadowSpillBackendCapabilities capabilities = {0};
    if (backend.capabilities(backend.state, &capabilities) != 0 ||
        capabilities.minimum_alignment == 0U || capabilities.provider[0] == '\0') {
        FAIL("capabilities");
    }
    ShadowSpillBackendPhysicalMemory physical = {0};
    if (backend.physical_memory(backend.state, &physical) != 0 ||
        physical.device_total_bytes == 0U) {
        FAIL("physical memory");
    }
    const ShadowSpillMemoryPoolDescription pools[2] = {
        {
            .pool_id = 0U,
            .kind = SHADOWSPILL_POOL_DEVICE,
            .capacity_bytes = 16U << 20U,
            .minimum_alignment = capabilities.minimum_alignment,
        },
        {
            .pool_id = 1U,
            .kind = SHADOWSPILL_POOL_PINNED_HOST,
            .capacity_bytes = 16U << 20U,
            .minimum_alignment = 1U,
        },
    };
    const ShadowSpillTransferRouteDescription routes[2] = {
        {.route_id = 0U, .name = "fetch", .source_pool_id = 1U, .destination_pool_id = 0U},
        {.route_id = 1U, .name = "evict", .source_pool_id = 0U, .destination_pool_id = 1U},
    };
    const ShadowSpillRuntimeConfig runtime_config = {
        .abi_version = SHADOWSPILL_ABI_VERSION,
        .backend = &backend,
        .pools = pools,
        .pool_count = 2U,
        .routes = routes,
        .route_count = 2U,
        .worker_poll_nanoseconds = 1000U,
    };
    ShadowSpillRuntime *runtime = NULL;
    if (shadowspill_runtime_create(&runtime_config, &runtime) !=
            SHADOWSPILL_STATUS_OK ||
        shadowspill_runtime_reserve_event_leases(runtime, 8U) !=
            SHADOWSPILL_STATUS_OK ||
        shadowspill_runtime_reserve_event_leases(runtime, 12U) !=
            SHADOWSPILL_STATUS_OK) {
        FAIL("runtime creation or event reservation");
    }
    ShadowSpillBackendStream compute = {0};
    if (backend.create_stream(backend.state, &compute) != 0) {
        FAIL("compute stream");
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
    const ShadowSpillRuntimeAction fetch = {
        .object_id = object.object_id,
        .kind = SHADOWSPILL_RUNTIME_FETCH,
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
            SHADOWSPILL_STATUS_OK ||
        shadowspill_write_object(
            runtime, object.object_id, 1U, original, PAYLOAD_BYTES
        ) != SHADOWSPILL_STATUS_OK || shadowspill_plan_create(
            runtime, &plan_description, &plan
        ) != SHADOWSPILL_STATUS_OK || shadowspill_object_handle_acquire(
            runtime, object.object_id, &object_handle
        ) != SHADOWSPILL_STATUS_OK || shadowspill_plan_bind_object(
            plan, object.object_id, object_handle, SHADOWSPILL_OBJECT_CAUSAL
        ) != SHADOWSPILL_STATUS_OK || shadowspill_object_handle_release(
            object_handle
        ) != SHADOWSPILL_STATUS_OK || shadowspill_plan_admit_action_batch(
            plan, 0U, &fetch, 1U, &initial_actions
        ) != SHADOWSPILL_STATUS_OK || shadowspill_submit_action_batch_handle(
            runtime, initial_actions, compute
        ) != SHADOWSPILL_STATUS_OK) {
        FAIL("initial object registration or fetch submission");
    }
    const uint64_t input_object_id = object.object_id;
    const ShadowSpillRuntimeAction evict = {
        .object_id = object.object_id,
        .kind = SHADOWSPILL_RUNTIME_EVICT,
    };
    const ShadowSpillTaskDescription task = {
        .task_id = 1U,
        .input_object_ids = &input_object_id,
        .input_count = 1U,
        .actions = &evict,
        .action_count = 1U,
    };
    const ShadowSpillTaskHandle *task_handle = NULL;
    const ShadowSpillObjectBinding *bindings = NULL;
    uint32_t binding_count = 0U;
    if (shadowspill_plan_admit_task(plan, &task, &task_handle) !=
            SHADOWSPILL_STATUS_OK || shadowspill_before_task_handle(
            runtime,
            task_handle,
            compute,
            &bindings,
            &binding_count
        ) != SHADOWSPILL_STATUS_OK ||
        binding_count != 1U || bindings == NULL ||
        bindings[0].pointer == NULL ||
        bindings[0].authoritative_version != 1U) {
        FAIL("fetch readiness");
    }
    const ShadowSpillStatus after_status = shadowspill_after_task_handle(
        runtime, task_handle, compute
    );
    const ShadowSpillStatus idle_status = after_status ==
            SHADOWSPILL_STATUS_OK
        ? shadowspill_runtime_wait_idle(runtime)
        : after_status;
    const ShadowSpillStatus read_status = idle_status ==
            SHADOWSPILL_STATUS_OK
        ? shadowspill_read_object(
            runtime, object.object_id, 1U, restored, PAYLOAD_BYTES
        )
        : idle_status;
    if (after_status != SHADOWSPILL_STATUS_OK ||
        idle_status != SHADOWSPILL_STATUS_OK ||
        read_status != SHADOWSPILL_STATUS_OK ||
        memcmp(original, restored, PAYLOAD_BYTES) != 0) {
        ShadowSpillRuntimeFailure failure = {0};
        (void)shadowspill_runtime_failure(runtime, &failure);
        fprintf(
            stderr,
            "after=%u idle=%u read=%u failure=%u task=%llu object=%llu\n",
            (unsigned)after_status,
            (unsigned)idle_status,
            (unsigned)read_status,
            (unsigned)failure.status,
            (unsigned long long)failure.task_id,
            (unsigned long long)failure.object_id
        );
        FAIL("fetch, evict, and read round trip");
    }
    ShadowSpillRuntimeStatistics runtime_statistics = {0};
    ShadowSpillBackendStatistics statistics = {0};
    if (shadowspill_runtime_statistics(runtime, &runtime_statistics) !=
            SHADOWSPILL_STATUS_OK) {
        FAIL("runtime statistics");
    }
    backend.statistics(backend.state, &statistics);
    if (statistics.copies_host_to_device != 1U ||
        statistics.copies_device_to_host != 1U ||
        statistics.bytes_host_to_device != PAYLOAD_BYTES ||
        statistics.bytes_device_to_host != PAYLOAD_BYTES ||
        runtime_statistics.event_lease_driver_creates != 12U ||
        runtime_statistics.event_lease_sealed != 1U) {
        fprintf(
            stderr,
            "h2d=%llu d2h=%llu creates=%llu sealed=%llu\n",
            (unsigned long long)statistics.copies_host_to_device,
            (unsigned long long)statistics.copies_device_to_host,
            (unsigned long long)runtime_statistics.event_lease_driver_creates,
            (unsigned long long)runtime_statistics.event_lease_sealed
        );
        FAIL("copy and event accounting");
    }
    if (shadowspill_runtime_close(runtime) != SHADOWSPILL_STATUS_OK) {
        FAIL("runtime close");
    }
    shadowspill_runtime_destroy(runtime);
    if (backend.destroy_stream(backend.state, compute) != 0) {
        FAIL("compute stream teardown");
    }
    backend.statistics(backend.state, &statistics);
    if (statistics.events_created != statistics.events_destroyed ||
        statistics.streams_created != statistics.streams_destroyed ||
        statistics.device_allocations != statistics.device_frees ||
        statistics.pinned_host_registrations != statistics.pinned_host_unregistrations) {
        FAIL("every driver resource was returned");
    }
    shadowspill_backend_destroy(&backend);
    free(original);
    free(restored);
    printf("device backend canary: ok (%s)\n", capabilities.provider);
    return EXIT_SUCCESS;
}
