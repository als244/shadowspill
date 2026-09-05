#define _GNU_SOURCE

#include "internal.h"

#include <dlfcn.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

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

ShadowSpillStatus shadowspill_pytorch_allocator_bootstrap(
    const ShadowSpillPytorchAdapterConfig *config
) {
    if (!bootstrap_config_is_valid(config)) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&adapter.mutex);
    if (adapter.bootstrapped) {
        pthread_mutex_unlock(&adapter.mutex);
        return SHADOWSPILL_STATUS_INVALID_STATE;
    }
    pthread_mutex_unlock(&adapter.mutex);

    void *const library = dlopen(config->backend_library, RTLD_NOW | RTLD_LOCAL);
    if (library == NULL) {
        return SHADOWSPILL_STATUS_BACKEND_FAILURE;
    }
    union {
        void *object;
        ShadowSpillBackendCreate create;
        ShadowSpillBackendDestroy destroy;
    } create_symbol = {.object = dlsym(library, SHADOWSPILL_BACKEND_CREATE_SYMBOL)};
    union {
        void *object;
        ShadowSpillBackendDestroy destroy;
    } destroy_symbol = {.object = dlsym(library, SHADOWSPILL_BACKEND_DESTROY_SYMBOL)};
    if (create_symbol.object == NULL || destroy_symbol.object == NULL) {
        (void)dlclose(library);
        return SHADOWSPILL_STATUS_BACKEND_FAILURE;
    }
    const ShadowSpillBackendConfig backend_config = {
        .abi_version = SHADOWSPILL_BACKEND_ABI_VERSION,
        .device_ordinal = config->device_ordinal,
    };
    ShadowSpillBackend backend = {0};
    if (create_symbol.create(&backend_config, &backend) != 0 ||
        !shadowspill_backend_is_valid(&backend)) {
        if (backend.state != NULL) {
            destroy_symbol.destroy(&backend);
        }
        (void)dlclose(library);
        return SHADOWSPILL_STATUS_BACKEND_FAILURE;
    }
    ShadowSpillBackendCapabilities capabilities = {0};
    if (backend.capabilities(backend.state, &capabilities) != 0) {
        destroy_symbol.destroy(&backend);
        (void)dlclose(library);
        return SHADOWSPILL_STATUS_BACKEND_FAILURE;
    }
    ShadowSpillBackendPhysicalMemory physical = {0};
    if (backend.physical_memory(backend.state, &physical) != 0) {
        destroy_symbol.destroy(&backend);
        (void)dlclose(library);
        return SHADOWSPILL_STATUS_BACKEND_FAILURE;
    }
    const uint64_t physical_granularity = 2U << 20U;
    if (config->device_budget_bytes > physical.device_total_bytes ||
        physical.process_bytes >
            config->device_budget_bytes - config->provider_headroom_bytes) {
        destroy_symbol.destroy(&backend);
        (void)dlclose(library);
        return SHADOWSPILL_STATUS_OUT_OF_MEMORY;
    }
    uint64_t available = config->device_budget_bytes -
        physical.process_bytes - config->provider_headroom_bytes;
    const uint64_t allocator_pool_bytes =
        available - available % physical_granularity;
    if (allocator_pool_bytes == 0U) {
        destroy_symbol.destroy(&backend);
        (void)dlclose(library);
        return SHADOWSPILL_STATUS_OUT_OF_MEMORY;
    }
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillStatus status = build_runtime_topology(
        config,
        &backend,
        &capabilities,
        allocator_pool_bytes,
        &runtime
    );
    if (status != SHADOWSPILL_STATUS_OK) {
        destroy_symbol.destroy(&backend);
        (void)dlclose(library);
        return status;
    }
    ShadowSpillBackendPhysicalMemory bootstrap_memory = {0};
    if (backend.physical_memory(backend.state, &bootstrap_memory) != 0 ||
        bootstrap_memory.process_bytes > config->device_budget_bytes) {
        shadowspill_runtime_destroy(runtime);
        destroy_symbol.destroy(&backend);
        (void)dlclose(library);
        return SHADOWSPILL_STATUS_OUT_OF_MEMORY;
    }
    pthread_mutex_lock(&adapter.mutex);
    if (adapter.runtime != NULL) {
        pthread_mutex_unlock(&adapter.mutex);
        shadowspill_runtime_destroy(runtime);
        destroy_symbol.destroy(&backend);
        (void)dlclose(library);
        return SHADOWSPILL_STATUS_INVALID_STATE;
    }
    if (!process_exit_registered) {
        if (on_exit(shadowspill_pytorch_process_exit, NULL) != 0) {
            pthread_mutex_unlock(&adapter.mutex);
            shadowspill_runtime_destroy(runtime);
            destroy_symbol.destroy(&backend);
            (void)dlclose(library);
            return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
        }
        process_exit_registered = 1U;
    }
    adapter.backend = backend;
    adapter.backend_destroy = destroy_symbol.destroy;
    adapter.backend_library = library;
    adapter.runtime = runtime;
    adapter.bootstrapped = 1U;
    adapter.closed = 0U;
    adapter.device_ordinal = config->device_ordinal;
    adapter.allocator_pool_id = config->allocator_pool_id;
    adapter.admission = (ShadowSpillPytorchPhysicalAdmission){
        .abi_version = SHADOWSPILL_PYTORCH_ADAPTER_ABI_VERSION,
        .device_ordinal = config->device_ordinal,
        .device_budget_bytes = config->device_budget_bytes,
        .baseline_bytes = physical.process_bytes,
        .provider_headroom_bytes = config->provider_headroom_bytes,
        .allocator_pool_id = config->allocator_pool_id,
        .pool_count = config->pool_count,
        .allocator_pool_bytes = allocator_pool_bytes,
        .bootstrap_process_bytes = bootstrap_memory.process_bytes,
        .device_used_bytes = bootstrap_memory.device_used_bytes,
        .device_total_bytes = bootstrap_memory.device_total_bytes,
    };
    adapter.physical_checks = 1U;
    adapter.peak_process_physical_bytes = bootstrap_memory.process_bytes;
    adapter.observed_external_high_water_bytes =
        bootstrap_memory.process_bytes >
                physical.process_bytes + allocator_pool_bytes
        ? bootstrap_memory.process_bytes - physical.process_bytes -
            allocator_pool_bytes
        : 0U;
    adapter.physical_budget_sealed = 0U;
    memset(&adapter.failure, 0, sizeof(adapter.failure));
    adapter.failure_task_label[0] = '\0';
    adapter.failure.device_ordinal = config->device_ordinal;
    adapter.failure.runtime.task_id = SHADOWSPILL_RUNTIME_NO_ID;
    adapter.failure.runtime.object_id = SHADOWSPILL_RUNTIME_NO_ID;
    adapter.failure.runtime.allocation_id = SHADOWSPILL_RUNTIME_NO_ID;
    adapter.failure.runtime.pool_id = UINT32_MAX;
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
