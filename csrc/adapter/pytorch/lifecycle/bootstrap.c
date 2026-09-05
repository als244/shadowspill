#include "internal.h"
#include "../failure/internal.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>

/* Registered with on_exit once, on the first bootstrap of the process. */
static uint8_t process_exit_registered;

static int bootstrap_config_is_valid(
    const ShadowSpillPytorchAdapterConfig *config
) {
    if (config == NULL ||
        config->abi_version != SHADOWSPILL_PYTORCH_ADAPTER_ABI_VERSION ||
        config->backend_library == NULL || config->backend_library[0] == '\0' ||
        config->device_ordinal < 0 || config->device_budget_bytes == 0U ||
        config->provider_headroom_bytes >= config->device_budget_bytes ||
        config->pools == NULL || config->pool_count == 0U ||
        config->allocator_pool_id >= config->pool_count ||
        (config->routes == NULL && config->route_count != 0U)) {
        return 0;
    }
    uint32_t device_pool_count = 0U;
    for (uint32_t index = 0U; index < config->pool_count; ++index) {
        const ShadowSpillPytorchPoolConfig *pool = &config->pools[index];
        if (pool->pool_id != index ||
            pool->kind > SHADOWSPILL_POOL_PINNED_HOST) {
            return 0;
        }
        if (pool->kind == SHADOWSPILL_POOL_DEVICE) {
            ++device_pool_count;
            if (index != config->allocator_pool_id) {
                return 0;
            }
        } else if (pool->capacity_bytes == 0U) {
            return 0;
        }
    }
    if (device_pool_count != 1U) {
        return 0;
    }
    for (uint32_t index = 0U; index < config->route_count; ++index) {
        const ShadowSpillPytorchRouteConfig *route = &config->routes[index];
        if (route->route_id != index || route->name == NULL ||
            route->source_pool_id >= config->pool_count ||
            route->destination_pool_id >= config->pool_count ||
            route->source_pool_id == route->destination_pool_id) {
            return 0;
        }
        const uint8_t source_kind =
            config->pools[route->source_pool_id].kind;
        const uint8_t destination_kind =
            config->pools[route->destination_pool_id].kind;
        if (source_kind == destination_kind) {
            return 0;
        }
    }
    return 1;
}

/* The device pool the budget leaves after what the process already holds
   and the provider's headroom, in whole 2 MiB pages; 0 when nothing fits. */
static uint64_t allocator_pool_bytes(
    const ShadowSpillPytorchAdapterConfig *config,
    const ShadowSpillBackendPhysicalMemory *physical
) {
    const uint64_t physical_granularity = 2U << 20U;
    if (config->device_budget_bytes > physical->device_total_bytes ||
        physical->process_bytes >
            config->device_budget_bytes - config->provider_headroom_bytes) {
        return 0U;
    }
    const uint64_t available = config->device_budget_bytes -
        physical->process_bytes - config->provider_headroom_bytes;
    return available - available % physical_granularity;
}

static ShadowSpillStatus build_runtime_topology(
    const ShadowSpillPytorchAdapterConfig *config,
    const ShadowSpillBackend *backend,
    const ShadowSpillBackendCapabilities *capabilities,
    uint64_t allocator_pool_bytes,
    ShadowSpillRuntime **runtime
) {
    ShadowSpillMemoryPoolDescription *pools = calloc(
        config->pool_count, sizeof(*pools)
    );
    ShadowSpillTransferRouteDescription *routes = config->route_count == 0U
        ? NULL : calloc(config->route_count, sizeof(*routes));
    if (pools == NULL || (config->route_count != 0U && routes == NULL)) {
        free(routes);
        free(pools);
        return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    }
    for (uint32_t index = 0U; index < config->pool_count; ++index) {
        const ShadowSpillPytorchPoolConfig *source = &config->pools[index];
        const int is_device = source->kind == SHADOWSPILL_POOL_DEVICE;
        pools[index] = (ShadowSpillMemoryPoolDescription){
            .pool_id = index,
            .kind = source->kind,
            .capacity_bytes = is_device
                ? allocator_pool_bytes : source->capacity_bytes,
            .minimum_alignment = is_device
                ? capabilities->minimum_alignment : 1U,
        };
    }
    for (uint32_t index = 0U; index < config->route_count; ++index) {
        const ShadowSpillPytorchRouteConfig *source = &config->routes[index];
        routes[index] = (ShadowSpillTransferRouteDescription){
            .route_id = index,
            .name = source->name,
            .source_pool_id = source->source_pool_id,
            .destination_pool_id = source->destination_pool_id,
        };
    }
    const ShadowSpillRuntimeConfig runtime_config = {
        .abi_version = SHADOWSPILL_ABI_VERSION,
        .backend = backend,
        .pools = pools,
        .pool_count = config->pool_count,
        .routes = routes,
        .route_count = config->route_count,
        .worker_poll_nanoseconds = config->worker_poll_nanoseconds,
    };
    const ShadowSpillStatus status = shadowspill_runtime_create(
        &runtime_config, runtime
    );
    free(routes);
    free(pools);
    return status;
}

/* Size the pool against what the device holds now, and create the runtime
   over it. */
static ShadowSpillStatus create_runtime(
    const ShadowSpillPytorchAdapterConfig *config,
    const ShadowSpillBackend *backend,
    ShadowSpillBackendPhysicalMemory *baseline,
    uint64_t *pool_bytes,
    ShadowSpillRuntime **runtime
) {
    ShadowSpillBackendCapabilities capabilities = {0};
    if (backend->capabilities(backend->state, &capabilities) != 0 ||
        backend->physical_memory(backend->state, baseline) != 0) {
        return SHADOWSPILL_STATUS_BACKEND_FAILURE;
    }
    *pool_bytes = allocator_pool_bytes(config, baseline);
    if (*pool_bytes == 0U) {
        return SHADOWSPILL_STATUS_OUT_OF_MEMORY;
    }
    return build_runtime_topology(
        config, backend, &capabilities, *pool_bytes, runtime
    );
}

/* The pools exist now; the process must still fit the budget. */
static ShadowSpillStatus confirm_budget(
    const ShadowSpillPytorchAdapterConfig *config,
    const ShadowSpillBackend *backend,
    ShadowSpillBackendPhysicalMemory *bootstrapped
) {
    return backend->physical_memory(backend->state, bootstrapped) != 0 ||
            bootstrapped->process_bytes > config->device_budget_bytes
        ? SHADOWSPILL_STATUS_OUT_OF_MEMORY
        : SHADOWSPILL_STATUS_OK;
}

/* Everything the rest of the adapter reads, written under the lock, with the
   runtime published last so a reader that sees it sees all of it. */
static ShadowSpillStatus publish(
    const ShadowSpillPytorchAdapterConfig *config,
    const ShadowSpillPytorchLoadedBackend *backend,
    ShadowSpillRuntime *runtime,
    const ShadowSpillBackendPhysicalMemory *baseline,
    const ShadowSpillBackendPhysicalMemory *bootstrapped,
    uint64_t pool_bytes
) {
    pthread_mutex_lock(&adapter.mutex);
    if (adapter.runtime != NULL) {
        pthread_mutex_unlock(&adapter.mutex);
        return SHADOWSPILL_STATUS_INVALID_STATE;
    }
    if (!process_exit_registered) {
        if (on_exit(shadowspill_pytorch_process_exit, NULL) != 0) {
            pthread_mutex_unlock(&adapter.mutex);
            return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
        }
        process_exit_registered = 1U;
    }
    adapter.backend = *backend;
    adapter.runtime = runtime;
    adapter.bootstrapped = 1U;
    adapter.closed = 0U;
    adapter.device_ordinal = config->device_ordinal;
    adapter.allocator_pool_id = config->allocator_pool_id;
    adapter.admission = (ShadowSpillPytorchPhysicalAdmission){
        .abi_version = SHADOWSPILL_PYTORCH_ADAPTER_ABI_VERSION,
        .device_ordinal = config->device_ordinal,
        .device_budget_bytes = config->device_budget_bytes,
        .baseline_bytes = baseline->process_bytes,
        .provider_headroom_bytes = config->provider_headroom_bytes,
        .allocator_pool_id = config->allocator_pool_id,
        .pool_count = config->pool_count,
        .allocator_pool_bytes = pool_bytes,
        .bootstrap_process_bytes = bootstrapped->process_bytes,
        .device_used_bytes = bootstrapped->device_used_bytes,
        .device_total_bytes = bootstrapped->device_total_bytes,
    };
    adapter.physical_checks = 1U;
    adapter.peak_process_physical_bytes = bootstrapped->process_bytes;
    adapter.observed_external_high_water_bytes =
        bootstrapped->process_bytes > baseline->process_bytes + pool_bytes
        ? bootstrapped->process_bytes - baseline->process_bytes - pool_bytes
        : 0U;
    adapter.physical_budget_sealed = 0U;
    shadowspill_pytorch_failure_clear_locked(config->device_ordinal);
    atomic_store_explicit(
        &adapter.published_device_ordinal,
        config->device_ordinal,
        memory_order_relaxed
    );
    atomic_store_explicit(
        &adapter.published_allocator_pool_id,
        config->allocator_pool_id,
        memory_order_relaxed
    );
    atomic_store_explicit(
        &adapter.published_runtime, runtime, memory_order_release
    );
    pthread_mutex_unlock(&adapter.mutex);
    return SHADOWSPILL_STATUS_OK;
}

ShadowSpillStatus shadowspill_pytorch_allocator_bootstrap(
    const ShadowSpillPytorchAdapterConfig *config
) {
    if (!bootstrap_config_is_valid(config)) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&adapter.mutex);
    const int bootstrapped_already = adapter.bootstrapped;
    pthread_mutex_unlock(&adapter.mutex);
    if (bootstrapped_already) {
        return SHADOWSPILL_STATUS_INVALID_STATE;
    }
    ShadowSpillPytorchLoadedBackend backend = {0};
    ShadowSpillStatus status = shadowspill_pytorch_backend_load(
        config->backend_library, config->device_ordinal, &backend
    );
    if (status != SHADOWSPILL_STATUS_OK) {
        return status;
    }
    ShadowSpillBackendPhysicalMemory baseline = {0};
    ShadowSpillBackendPhysicalMemory bootstrapped = {0};
    uint64_t pool_bytes = 0U;
    ShadowSpillRuntime *runtime = NULL;
    status = create_runtime(
        config, &backend.table, &baseline, &pool_bytes, &runtime
    );
    if (status == SHADOWSPILL_STATUS_OK) {
        status = confirm_budget(config, &backend.table, &bootstrapped);
    }
    if (status == SHADOWSPILL_STATUS_OK) {
        status = publish(
            config, &backend, runtime, &baseline, &bootstrapped, pool_bytes
        );
    }
    if (status != SHADOWSPILL_STATUS_OK) {
        if (runtime != NULL) {
            shadowspill_runtime_destroy(runtime);
        }
        shadowspill_pytorch_backend_unload(&backend);
    }
    return status;
}
