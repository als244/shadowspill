/* Independent route calibration. Never touches the adapter's global allocator. */
#include "../lifecycle/internal.h"

#include <stdlib.h>

typedef struct SpillProbe {
    ShadowSpillRuntime *runtime;
    ShadowSpillPytorchLoadedBackend backend;
    ShadowSpillPytorchLoadedLibrary *libraries;
    uint32_t library_count;
} SpillProbe;

void shadowspill_pytorch_spill_probe_destroy(void *handle) {
    SpillProbe *probe = handle;
    if (probe == NULL) return;
    if (probe->runtime != NULL) shadowspill_runtime_destroy(probe->runtime);
    shadowspill_pytorch_libraries_unload(probe->libraries, probe->library_count);
    free(probe->libraries);
    shadowspill_pytorch_backend_unload(&probe->backend);
    free(probe);
}

ShadowSpillStatus shadowspill_pytorch_spill_probe_create(
    const char *backend_path, int32_t device, const char *library_path,
    uint8_t kind, void *configuration, uint64_t capacity, void **out
) {
    if (out == NULL || backend_path == NULL || capacity == 0U)
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    *out = NULL;
    SpillProbe *probe = calloc(1U, sizeof(*probe));
    if (probe == NULL) return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    ShadowSpillPoolMemoryDescription *memory = NULL;
    ShadowSpillLaneDescription *lanes = NULL;
    uint32_t memory_count = 0U, lane_count = 0U;
    ShadowSpillStatus status = shadowspill_pytorch_backend_load(
        backend_path, device, &probe->backend
    );
    if (status != SHADOWSPILL_STATUS_OK) goto done;
    if (library_path != NULL) {
        status = shadowspill_pytorch_libraries_load(&library_path, 1U, &probe->libraries);
        if (status != SHADOWSPILL_STATUS_OK) goto done;
        probe->library_count = 1U;
        status = shadowspill_pytorch_library_entries(
            probe->libraries, 1U, &memory, &memory_count, &lanes, &lane_count
        );
        if (status != SHADOWSPILL_STATUS_OK) goto done;
    }
    ShadowSpillBackendCapabilities capabilities = {0};
    if (probe->backend.table.capabilities(probe->backend.table.state, &capabilities) != 0) {
        status = SHADOWSPILL_STATUS_BACKEND_FAILURE;
        goto done;
    }
    const ShadowSpillMemoryPoolDescription pools[] = {
        {.pool_id = 0U, .kind = SHADOWSPILL_POOL_DEVICE, .capacity_bytes = capacity,
         .minimum_alignment = capabilities.minimum_alignment},
        {.pool_id = 1U, .kind = kind, .capacity_bytes = capacity,
         .minimum_alignment = 1U, .configuration = configuration},
    };
    const ShadowSpillTransferRouteDescription routes[] = {
        {.route_id = 0U, .name = "fetch", .source_pool_id = 1U, .destination_pool_id = 0U},
        {.route_id = 1U, .name = "evict", .source_pool_id = 0U, .destination_pool_id = 1U},
    };
    const ShadowSpillRuntimeConfig config = {
        .abi_version = SHADOWSPILL_ABI_VERSION, .backend = &probe->backend.table,
        .pools = pools, .pool_count = 2U, .routes = routes, .route_count = 2U,
        .pool_memory = memory, .pool_memory_count = memory_count,
        .lanes = lanes, .lane_count = lane_count,
        .worker_poll_nanoseconds = 1000U,
        .disable_numa_binding = 1U,
    };
    status = shadowspill_runtime_create(&config, &probe->runtime);
done:
    free(memory);
    free(lanes);
    if (status == SHADOWSPILL_STATUS_OK) *out = probe;
    else shadowspill_pytorch_spill_probe_destroy(probe);
    return status;
}

ShadowSpillStatus shadowspill_pytorch_spill_probe_measure(
    void *handle, uint64_t small, uint64_t large,
    uint32_t warmup, uint32_t measured, uint64_t *values
) {
    SpillProbe *probe = handle;
    if (probe == NULL || values == NULL) return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    const ShadowSpillTransferCalibrationConfig config = {
        .abi_version = SHADOWSPILL_ABI_VERSION,
        .small_copy_bytes = small, .large_copy_bytes = large,
        .warmup_copies = warmup, .measured_copies = measured,
    };
    ShadowSpillStatus status = shadowspill_runtime_calibrate_transfer_capabilities(
        probe->runtime, &config, NULL, 0U
    );
    if (status != SHADOWSPILL_STATUS_OK) return status;
    ShadowSpillTransferProfile profiles[4] = {0};
    uint32_t count = 0U;
    uint64_t generation = 0U;
    status = shadowspill_runtime_transfer_profiles(
        probe->runtime, profiles, 4U, &count, &generation
    );
    if (status != SHADOWSPILL_STATUS_OK) return status;
    if (count != 4U) return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    for (uint32_t direction = 0U; direction < 2U; ++direction) {
        const ShadowSpillTransferProfile *p = &profiles[direction == 0U ? 2U : 1U];
        values[5U * direction] = p->latency_nanoseconds;
        values[5U * direction + 1U] = p->solo_bandwidth_bytes_per_second;
        values[5U * direction + 2U] = p->concurrent_bandwidth_bytes_per_second;
        values[5U * direction + 3U] = p->solo_measurement_nanoseconds;
        values[5U * direction + 4U] = p->concurrent_measurement_nanoseconds;
    }
    return SHADOWSPILL_STATUS_OK;
}
