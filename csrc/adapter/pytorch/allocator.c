#define _GNU_SOURCE


#include <shadowspill/pytorch_adapter.h>

#include "internal.h"

#include <shadowspill/backend.h>

#include <dlfcn.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdint.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


typedef struct ShadowSpillPytorchAdapterState {
    pthread_mutex_t mutex;
    ShadowSpillBackend backend;
    ShadowSpillBackendDestroy backend_destroy;
    void *backend_library;
    ShadowSpillRuntime *runtime;
    int32_t device_ordinal;
    uint32_t allocator_pool_id;
    uint64_t allocation_callbacks;
    uint64_t zero_size_allocation_callbacks;
    uint64_t free_callbacks;
    uint64_t record_stream_callbacks;
    uint64_t pointer_lookup_failures;
    uint64_t callback_failures;
    uint64_t physical_checks;
    uint64_t peak_process_physical_bytes;
    uint64_t observed_external_high_water_bytes;
    uint64_t physical_budget_sealed;
    _Atomic uint8_t profiler_annotations_enabled;
    _Atomic uint8_t shutdown_started;
    _Atomic uint64_t active_allocator_callbacks;
    _Atomic(ShadowSpillRuntime *) published_runtime;
    _Atomic int32_t published_device_ordinal;
    _Atomic uint32_t published_allocator_pool_id;
    char failure_task_label[SHADOWSPILL_RUNTIME_TRACE_LABEL_MAX_BYTES + 1U];
    ShadowSpillPytorchPhysicalAdmission admission;
    /*
     * `failure` is the first failure, which is what stopped the runtime and
     * what a caller asking "is this runtime usable" wants. `recent` is the
     * last one, which is what a caller asking "why did my call fail" wants.
     * They are different questions: a failure that was recovered from must
     * not describe a later one, and a first failure must not be overwritten
     * by the calls it subsequently causes to fail.
     */
    ShadowSpillPytorchAdapterFailure failure;
    ShadowSpillPytorchAdapterFailure recent;
    uint8_t recent_valid;
    uint8_t bootstrapped;
    uint8_t closed;
} ShadowSpillPytorchAdapterState;

static ShadowSpillPytorchAdapterState adapter = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .device_ordinal = -1,
    .published_device_ordinal = -1,
    .published_allocator_pool_id = UINT32_MAX,
};

static uint8_t process_exit_registered;

static _Thread_local int task_range_active;
static _Thread_local ShadowSpillProfilerRange task_range_id;
static _Thread_local const char *active_task_label;

static inline void adapter_cpu_relax(void) {
#if defined(__x86_64__) || defined(__i386__)
    __asm__ __volatile__("pause" ::: "memory");
#elif defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__("yield" ::: "memory");
#else
    atomic_signal_fence(memory_order_seq_cst);
#endif
}

ShadowSpillProfilerRange shadowspill_pytorch_profile_range_begin(
    const char *name
) {
    return atomic_load_explicit(
               &adapter.profiler_annotations_enabled, memory_order_relaxed
           ) == 0U || adapter.backend.range_begin == NULL
        ? 0U
        : adapter.backend.range_begin(adapter.backend.state, name);
}

void shadowspill_pytorch_profile_range_end(ShadowSpillProfilerRange range) {
    if (range != 0U && adapter.backend.range_end != NULL) {
        adapter.backend.range_end(adapter.backend.state, range);
    }
}

ShadowSpillStatus shadowspill_pytorch_profiler_annotations_set(
    uint8_t enabled
) {
    pthread_mutex_lock(&adapter.mutex);
    const int available = adapter.runtime != NULL &&
        adapter.backend.profiler_enable != NULL;
    const ShadowSpillBackend backend = adapter.backend;
    pthread_mutex_unlock(&adapter.mutex);
    if (!available) {
        return SHADOWSPILL_STATUS_CLOSED;
    }
    backend.profiler_enable(backend.state, enabled != 0U);
    atomic_store_explicit(
        &adapter.profiler_annotations_enabled,
        enabled != 0U,
        memory_order_release
    );
    return SHADOWSPILL_STATUS_OK;
}

static void end_task_range(void) {
    if (task_range_active) {
        shadowspill_pytorch_profile_range_end(task_range_id);
        task_range_active = 0;
        task_range_id = 0;
    }
    active_task_label = NULL;
}

static void format_task_range_name(
    char *destination,
    size_t destination_bytes,
    const char *operation,
    const ShadowSpillTaskHandle *handle
) {
    const uint64_t task_id = shadowspill_task_id(handle);
    const char *label = shadowspill_task_trace_label(handle);
    if (label != NULL && label[0] != '\0') {
        (void)snprintf(
            destination,
            destination_bytes,
            "shadowspill.pytorch.%s.%s",
            operation,
            label
        );
    } else {
        (void)snprintf(
            destination,
            destination_bytes,
            "shadowspill.pytorch.%s.canonical_%llu",
            operation,
            (unsigned long long)task_id
        );
    }
}

static void latch_failure(
    ShadowSpillStatus status,
    int32_t device_ordinal,
    const void *address,
    uint64_t requested_bytes
) {
    char task_label[SHADOWSPILL_RUNTIME_TRACE_LABEL_MAX_BYTES + 1U] = {0};
    if (active_task_label != NULL) {
        (void)snprintf(task_label, sizeof(task_label), "%s", active_task_label);
    }
    pthread_mutex_lock(&adapter.mutex);
    ++adapter.callback_failures;
    adapter.recent = (ShadowSpillPytorchAdapterFailure){
        .status = (uint32_t)status,
        .device_ordinal = device_ordinal,
        .address = (uint64_t)(uintptr_t)address,
        .requested_bytes = requested_bytes,
    };
    adapter.recent_valid = 1U;
    if (adapter.failure.status == SHADOWSPILL_STATUS_OK) {
        adapter.failure.status = (uint32_t)status;
        adapter.failure.device_ordinal = device_ordinal;
        adapter.failure.address = (uint64_t)(uintptr_t)address;
        adapter.failure.requested_bytes = requested_bytes;
        (void)snprintf(
            adapter.failure_task_label,
            sizeof(adapter.failure_task_label),
            "%s",
            task_label
        );
        if (adapter.runtime != NULL) {
            (void)shadowspill_runtime_failure(
                adapter.runtime, &adapter.failure.runtime
            );
        }
    }
    pthread_mutex_unlock(&adapter.mutex);
}

static ShadowSpillRuntime *bound_runtime(int32_t *device_ordinal) {
    *device_ordinal = atomic_load_explicit(
        &adapter.published_device_ordinal, memory_order_relaxed
    );
    return atomic_load_explicit(
        &adapter.published_runtime, memory_order_acquire
    );
}

ShadowSpillStatus shadowspill_pytorch_runtime_handle(
    uintptr_t *runtime_handle
) {
    if (runtime_handle == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    *runtime_handle = (uintptr_t)runtime;
    return runtime == NULL ? SHADOWSPILL_STATUS_CLOSED : SHADOWSPILL_STATUS_OK;
}

static uint32_t bound_allocator_pool_id(void) {
    return atomic_load_explicit(
        &adapter.published_allocator_pool_id, memory_order_relaxed
    );
}

static ShadowSpillRuntime *acquire_allocator_callback_runtime(
    int32_t *device_ordinal
) {
    if (atomic_load_explicit(
            &adapter.shutdown_started, memory_order_acquire
        ) != 0U) {
        *device_ordinal = atomic_load_explicit(
            &adapter.published_device_ordinal, memory_order_relaxed
        );
        return NULL;
    }
    (void)atomic_fetch_add_explicit(
        &adapter.active_allocator_callbacks, 1U, memory_order_acq_rel
    );
    if (atomic_load_explicit(
            &adapter.shutdown_started, memory_order_acquire
        ) != 0U) {
        (void)atomic_fetch_sub_explicit(
            &adapter.active_allocator_callbacks, 1U, memory_order_release
        );
        *device_ordinal = atomic_load_explicit(
            &adapter.published_device_ordinal, memory_order_relaxed
        );
        return NULL;
    }
    ShadowSpillRuntime *runtime = bound_runtime(device_ordinal);
    if (runtime == NULL) {
        (void)atomic_fetch_sub_explicit(
            &adapter.active_allocator_callbacks, 1U, memory_order_release
        );
    }
    return runtime;
}

static void release_allocator_callback_runtime(void) {
    (void)atomic_fetch_sub_explicit(
        &adapter.active_allocator_callbacks, 1U, memory_order_release
    );
}

/* The framework's stream handle as the token the backend's vtables accept. The
 * bundle is written once at bootstrap and cleared only after the runtime it
 * served is gone, so reading it here needs no lock. */
static ShadowSpillBackendStream adapter_stream(uint64_t framework_stream_handle) {
    return adapter.backend.wrap_stream(adapter.backend.state, framework_stream_handle);
}

static ShadowSpillStatus close_adapter_runtime(
    int require_no_caller_allocations,
    int wait_for_outstanding_work,
    uint64_t *outstanding_actions,
    uint64_t *outstanding_retirements
) {
    pthread_mutex_lock(&adapter.mutex);
    if (adapter.closed) {
        pthread_mutex_unlock(&adapter.mutex);
        return SHADOWSPILL_STATUS_OK;
    }
    if (atomic_load_explicit(
            &adapter.shutdown_started, memory_order_acquire
        ) != 0U) {
        pthread_mutex_unlock(&adapter.mutex);
        return SHADOWSPILL_STATUS_INVALID_STATE;
    }
    ShadowSpillRuntime *runtime = adapter.runtime;
    ShadowSpillBackend backend = adapter.backend;
    const ShadowSpillBackendDestroy backend_destroy = adapter.backend_destroy;
    void *const backend_library = adapter.backend_library;
    if (runtime == NULL) {
        pthread_mutex_unlock(&adapter.mutex);
        return SHADOWSPILL_STATUS_CLOSED;
    }
    atomic_store_explicit(
        &adapter.shutdown_started, 1U, memory_order_release
    );
    pthread_mutex_unlock(&adapter.mutex);

    while (atomic_load_explicit(
               &adapter.active_allocator_callbacks, memory_order_acquire
           ) != 0U) {
        adapter_cpu_relax();
    }

    if (require_no_caller_allocations) {
        ShadowSpillRuntimeStatistics statistics = {0};
        const ShadowSpillStatus statistics_status =
            shadowspill_runtime_statistics(runtime, &statistics);
        if (statistics_status != SHADOWSPILL_STATUS_OK ||
            statistics.caller_owned_allocations != 0U) {
            atomic_store_explicit(
                &adapter.shutdown_started, 0U, memory_order_release
            );
            return statistics_status != SHADOWSPILL_STATUS_OK
                ? statistics_status
                : SHADOWSPILL_STATUS_INVALID_STATE;
        }
    }

    pthread_mutex_lock(&adapter.mutex);
    if (adapter.runtime != runtime) {
        pthread_mutex_unlock(&adapter.mutex);
        atomic_store_explicit(
            &adapter.shutdown_started, 0U, memory_order_release
        );
        return SHADOWSPILL_STATUS_INVALID_STATE;
    }
    atomic_store_explicit(
        &adapter.published_runtime, NULL, memory_order_release
    );
    atomic_store_explicit(
        &adapter.published_allocator_pool_id, UINT32_MAX, memory_order_relaxed
    );
    adapter.runtime = NULL;
    adapter.backend = (ShadowSpillBackend){0};
    adapter.backend_destroy = NULL;
    adapter.backend_library = NULL;
    atomic_store_explicit(
        &adapter.profiler_annotations_enabled, 0U, memory_order_release
    );
    adapter.closed = 1U;
    pthread_mutex_unlock(&adapter.mutex);

    /*
     * runtime_destroy stops and joins the worker before releasing anything it
     * can observe. Keep the backend alive until all lanes, events, pinned
     * registrations, and pool arenas have been explicitly closed.
     */
    const ShadowSpillStatus status =
        wait_for_outstanding_work
            ? shadowspill_runtime_close(runtime)
            : shadowspill_runtime_abandon(
                  runtime, outstanding_actions, outstanding_retirements
              );
    shadowspill_runtime_destroy(runtime);
    if (backend_destroy != NULL) {
        backend_destroy(&backend);
    }
    if (backend_library != NULL) {
        (void)dlclose(backend_library);
    }
    return status;
}

static void shadowspill_pytorch_process_exit(int status, void *argument) {
    (void)argument;
    /*
     * The process is going away, so this never waits. Outstanding transfers
     * cannot complete once exit handlers are running, and everything a drain
     * protects is reclaimed at process exit regardless; waiting here is how a
     * third party calling exit() turns into a process that never dies.
     *
     * A nonzero status with nothing latched means something outside
     * ShadowSpill ended the process. Say so: it is the only evidence such a
     * caller leaves behind.
     */
    uint64_t actions = 0U;
    uint64_t retirements = 0U;
    const ShadowSpillStatus latched = close_adapter_runtime(
        0, 0, &actions, &retirements
    );
    if (status == 0 && actions == 0U && retirements == 0U) {
        return;
    }
    (void)fprintf(
        stderr,
        "ShadowSpill: the process is exiting with status %d and "
        "%llu action(s) and %llu retirement(s) outstanding; "
        "ShadowSpill did not wait for them. Latched failure: %s.\n",
        status,
        (unsigned long long)actions,
        (unsigned long long)retirements,
        latched == SHADOWSPILL_STATUS_OK
            ? "none, so the exit came from outside ShadowSpill"
            : shadowspill_status_string(latched)
    );
    (void)fflush(stderr);
}

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

ShadowSpillStatus shadowspill_pytorch_allocator_close(void) {
    return close_adapter_runtime(1, 1, NULL, NULL);
}

ShadowSpillStatus shadowspill_pytorch_physical_admission(
    ShadowSpillPytorchPhysicalAdmission *admission
) {
    if (admission == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&adapter.mutex);
    if (adapter.runtime == NULL) {
        pthread_mutex_unlock(&adapter.mutex);
        return SHADOWSPILL_STATUS_CLOSED;
    }
    *admission = adapter.admission;
    pthread_mutex_unlock(&adapter.mutex);
    return SHADOWSPILL_STATUS_OK;
}

ShadowSpillStatus shadowspill_pytorch_physical_memory(
    ShadowSpillBackendPhysicalMemory *memory
) {
    if (memory == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&adapter.mutex);
    const ShadowSpillBackend backend = adapter.backend;
    pthread_mutex_unlock(&adapter.mutex);
    if (backend.state == NULL) {
        return SHADOWSPILL_STATUS_CLOSED;
    }
    return backend.physical_memory(backend.state, memory) == 0
        ? SHADOWSPILL_STATUS_OK
        : SHADOWSPILL_STATUS_BACKEND_FAILURE;
}

ShadowSpillStatus shadowspill_pytorch_check_physical_budget(void) {
    ShadowSpillBackendPhysicalMemory memory = {0};
    pthread_mutex_lock(&adapter.mutex);
    const ShadowSpillBackend backend = adapter.backend;
    ShadowSpillPytorchPhysicalAdmission admission = adapter.admission;
    pthread_mutex_unlock(&adapter.mutex);
    if (backend.state == NULL) {
        return SHADOWSPILL_STATUS_CLOSED;
    }
    if (backend.physical_memory(backend.state, &memory) != 0) {
        return SHADOWSPILL_STATUS_BACKEND_FAILURE;
    }
    uint64_t base = admission.baseline_bytes + admission.allocator_pool_bytes;
    uint64_t external = memory.process_bytes > base
        ? memory.process_bytes - base
        : 0U;
    ShadowSpillStatus status =
        memory.process_bytes <= admission.device_budget_bytes &&
            external <= admission.provider_headroom_bytes
        ? SHADOWSPILL_STATUS_OK
        : SHADOWSPILL_STATUS_PLAN_VIOLATION;
    pthread_mutex_lock(&adapter.mutex);
    ++adapter.physical_checks;
    if (memory.process_bytes > adapter.peak_process_physical_bytes) {
        adapter.peak_process_physical_bytes = memory.process_bytes;
    }
    if (external > adapter.observed_external_high_water_bytes) {
        adapter.observed_external_high_water_bytes = external;
    }
    if (status != SHADOWSPILL_STATUS_OK &&
        adapter.failure.status == SHADOWSPILL_STATUS_OK) {
        adapter.failure.status = (uint32_t)status;
        adapter.failure.requested_bytes = memory.process_bytes;
        adapter.failure.runtime.status = (uint32_t)status;
        adapter.failure.runtime.requested_bytes = memory.process_bytes;
        adapter.failure.runtime.free_bytes =
            admission.device_budget_bytes > memory.process_bytes
            ? admission.device_budget_bytes - memory.process_bytes
            : 0U;
    }
    pthread_mutex_unlock(&adapter.mutex);
    return status;
}

ShadowSpillStatus shadowspill_pytorch_seal_physical_budget(
    uint64_t required_provider_headroom_bytes,
    uint64_t runtime_record_reserve
) {
    pthread_mutex_lock(&adapter.mutex);
    const ShadowSpillBackend backend = adapter.backend;
    ShadowSpillRuntime *runtime = adapter.runtime;
    pthread_mutex_unlock(&adapter.mutex);
    if (backend.state == NULL || runtime == NULL) {
        return SHADOWSPILL_STATUS_CLOSED;
    }
    ShadowSpillStatus reserve_status =
        shadowspill_runtime_reserve_event_leases(runtime, runtime_record_reserve);
    if (reserve_status != SHADOWSPILL_STATUS_OK) {
        return reserve_status;
    }
    reserve_status = shadowspill_runtime_reserve_retirement_records(
        runtime, runtime_record_reserve
    );
    if (reserve_status != SHADOWSPILL_STATUS_OK) {
        return reserve_status;
    }
    for (uint32_t pool_id = 0U;
         pool_id < adapter.admission.pool_count;
         ++pool_id) {
        reserve_status = shadowspill_runtime_reserve_memory_lease_records(
            runtime, pool_id, runtime_record_reserve
        );
        if (reserve_status != SHADOWSPILL_STATUS_OK) {
            return reserve_status;
        }
    }
    ShadowSpillStatus status =
        shadowspill_pytorch_check_physical_budget();
    if (status != SHADOWSPILL_STATUS_OK) {
        return status;
    }
    pthread_mutex_lock(&adapter.mutex);
    if (required_provider_headroom_bytes >
        adapter.admission.provider_headroom_bytes) {
        status = SHADOWSPILL_STATUS_PLAN_VIOLATION;
        if (adapter.failure.status == SHADOWSPILL_STATUS_OK) {
            adapter.failure.status = (uint32_t)status;
            adapter.failure.requested_bytes = required_provider_headroom_bytes;
            adapter.failure.runtime.status = (uint32_t)status;
            adapter.failure.runtime.requested_bytes =
                required_provider_headroom_bytes;
            adapter.failure.runtime.free_bytes =
                adapter.admission.provider_headroom_bytes;
        }
    } else {
        adapter.physical_budget_sealed = 1U;
    }
    pthread_mutex_unlock(&adapter.mutex);
    return status;
}

ShadowSpillStatus shadowspill_pytorch_adapter_capabilities(
    ShadowSpillPytorchAdapterCapabilities *capabilities
) {
    if (capabilities == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    *capabilities = (ShadowSpillPytorchAdapterCapabilities){
        .abi_version = SHADOWSPILL_PYTORCH_ADAPTER_ABI_VERSION,
        .runtime_abi_version = SHADOWSPILL_ABI_VERSION,
        .backend_abi_version = SHADOWSPILL_BACKEND_ABI_VERSION,
        .slab_memory_strategy = 1U,
        .record_stream_callback = 1U,
#ifdef SHADOWSPILL_PYTORCH_STORAGE_ADAPTER
        .storage_rebinding = 1U,
#else
        .storage_rebinding = 0U,
#endif
        .debug_task_dispatch_timing = 1U,
        .runtime_trace = 1U,
    };
    return SHADOWSPILL_STATUS_OK;
}

ShadowSpillStatus shadowspill_pytorch_allocator_statistics(
    ShadowSpillPytorchAdapterStatistics *statistics
) {
    if (statistics == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&adapter.mutex);
    ShadowSpillRuntime *runtime = adapter.runtime;
    const ShadowSpillBackend backend = adapter.backend;
    *statistics = (ShadowSpillPytorchAdapterStatistics){
        .allocation_callbacks = adapter.allocation_callbacks,
        .zero_size_allocation_callbacks =
            adapter.zero_size_allocation_callbacks,
        .free_callbacks = adapter.free_callbacks,
        .record_stream_callbacks = adapter.record_stream_callbacks,
        .pointer_lookup_failures = adapter.pointer_lookup_failures,
        .callback_failures = adapter.callback_failures,
        .physical_checks = adapter.physical_checks,
        .peak_process_physical_bytes = adapter.peak_process_physical_bytes,
        .observed_external_high_water_bytes =
            adapter.observed_external_high_water_bytes,
        .physical_budget_sealed = adapter.physical_budget_sealed,
    };
    pthread_mutex_unlock(&adapter.mutex);
    if (runtime == NULL || backend.state == NULL) {
        return SHADOWSPILL_STATUS_CLOSED;
    }
    ShadowSpillStatus status = shadowspill_runtime_statistics(
        runtime, &statistics->runtime
    );
    backend.statistics(backend.state, &statistics->backend);
    return status;
}

ShadowSpillStatus shadowspill_pytorch_allocator_failure(
    ShadowSpillPytorchAdapterFailure *failure
) {
    if (failure == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&adapter.mutex);
    *failure = adapter.failure;
    ShadowSpillRuntime *runtime = adapter.runtime;
    int32_t device_ordinal = adapter.device_ordinal;
    pthread_mutex_unlock(&adapter.mutex);
    if (failure->status == SHADOWSPILL_STATUS_OK && runtime != NULL) {
        ShadowSpillRuntimeFailure runtime_failure = {0};
        if (shadowspill_runtime_failure(runtime, &runtime_failure) ==
                SHADOWSPILL_STATUS_OK &&
            runtime_failure.status != SHADOWSPILL_STATUS_OK) {
            failure->status = runtime_failure.status;
            failure->device_ordinal = device_ordinal;
            failure->runtime = runtime_failure;
        }
    }
    return (ShadowSpillStatus)failure->status;
}

static void append_failure_message(
    char *destination,
    size_t destination_bytes,
    size_t *offset,
    const char *format,
    ...
) {
    if (destination == NULL || destination_bytes == 0U ||
        *offset >= destination_bytes) {
        return;
    }
    va_list arguments;
    va_start(arguments, format);
    const int written = vsnprintf(
        destination + *offset,
        destination_bytes - *offset,
        format,
        arguments
    );
    va_end(arguments);
    if (written < 0) {
        return;
    }
    const size_t available = destination_bytes - *offset;
    *offset += (size_t)written < available ? (size_t)written : available - 1U;
}

static void append_failure_task(
    char *destination,
    size_t destination_bytes,
    size_t *offset,
    uint64_t task_id
) {
    const uint64_t profiling_base = UINT64_C(1) << 62U;
    const uint64_t initial_actions_base = UINT64_C(1) << 60U;
    const uint64_t caller_handoff_base = UINT64_C(1) << 59U;
    if (task_id >= profiling_base) {
        append_failure_message(
            destination,
            destination_bytes,
            offset,
            "planning_task: structural_profile_%06llu\n",
            (unsigned long long)(task_id - profiling_base)
        );
        return;
    }
    if (task_id >= initial_actions_base) {
        append_failure_message(
            destination,
            destination_bytes,
            offset,
            "runtime_scope: initial_actions.invocation_%06llu\n",
            (unsigned long long)(task_id - initial_actions_base)
        );
        return;
    }
    if (task_id >= caller_handoff_base) {
        append_failure_message(
            destination,
            destination_bytes,
            offset,
            "runtime_scope: caller_handoff.invocation_%06llu\n",
            (unsigned long long)(task_id - caller_handoff_base)
        );
        return;
    }
    char label[SHADOWSPILL_RUNTIME_TRACE_LABEL_MAX_BYTES + 1U] = {0};
    pthread_mutex_lock(&adapter.mutex);
    if (adapter.failure_task_label[0] != '\0') {
        (void)snprintf(
            label, sizeof(label), "%s", adapter.failure_task_label
        );
    }
    pthread_mutex_unlock(&adapter.mutex);
    if (label[0] == '\0') {
        append_failure_message(
            destination,
            destination_bytes,
            offset,
            "canonical_task: task_%06llu\n",
            (unsigned long long)task_id
        );
        return;
    }
    char *const separator = strchr(label, '.');
    if (separator != NULL) {
        *separator = '\0';
        append_failure_message(
            destination,
            destination_bytes,
            offset,
            "execution_task: %s\nsemantic_task: %s\n",
            label,
            separator + 1
        );
    } else {
        append_failure_message(
            destination,
            destination_bytes,
            offset,
            "execution_task: %s\n",
            label
        );
    }
    append_failure_message(
        destination,
        destination_bytes,
        offset,
        "canonical_task: task_%06llu\n",
        (unsigned long long)task_id
    );
}

/* Bytes as a person reads them. Reports are read by people deciding what to
 * change, and "16273899520" does not tell them it exceeds a 16 GiB budget. */
static void append_bytes(
    char *destination,
    size_t destination_bytes,
    size_t *offset,
    const char *label,
    uint64_t value
) {
    static const char *const units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    size_t unit = 0U;
    double scaled = (double)value;
    while (scaled >= 1024.0 && unit + 1U < sizeof(units) / sizeof(units[0])) {
        scaled /= 1024.0;
        ++unit;
    }
    if (unit == 0U) {
        append_failure_message(
            destination, destination_bytes, offset, "%s: %llu B\n",
            label, (unsigned long long)value
        );
        return;
    }
    append_failure_message(
        destination, destination_bytes, offset, "%s: %.2f %s (%llu bytes)\n",
        label, scaled, units[unit], (unsigned long long)value
    );
}

static const char *allocation_operation_name(uint8_t operation) {
    if (operation == SHADOWSPILL_TASK_ALLOCATION_ALLOCATE) {
        return "allocate";
    }
    if (operation == SHADOWSPILL_TASK_ALLOCATION_FREE) {
        return "free";
    }
    if (operation == UINT8_MAX) {
        return "end_of_task";
    }
    return "unknown";
}

ShadowSpillStatus shadowspill_pytorch_backend_malloc_failure_message(
    char *destination,
    size_t destination_bytes
) {
    if (destination == NULL || destination_bytes == 0U) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    destination[0] = '\0';
    ShadowSpillPytorchAdapterFailure failure = {0};
    ShadowSpillStatus status =
        shadowspill_pytorch_allocator_failure(&failure);
    /*
     * Describe the call that failed, not the first failure the runtime ever
     * saw. A recovered failure latched earlier says nothing about this one,
     * and reporting its operands beside this call's makes every number in the
     * message unattributable.
     */
    pthread_mutex_lock(&adapter.mutex);
    const int have_recent = adapter.recent_valid != 0U;
    const ShadowSpillPytorchAdapterFailure recent = adapter.recent;
    pthread_mutex_unlock(&adapter.mutex);
    int reported_first_failure = 0;
    if (have_recent) {
        reported_first_failure =
            failure.status != SHADOWSPILL_STATUS_OK &&
            failure.status != recent.status;
        status = (ShadowSpillStatus)recent.status;
        failure.device_ordinal = recent.device_ordinal;
        failure.requested_bytes = recent.requested_bytes;
    }
    const ShadowSpillRuntimeFailure *runtime = &failure.runtime;
    const uint64_t requested_bytes = failure.requested_bytes != 0U
        ? failure.requested_bytes
        : runtime->requested_bytes;
    size_t offset = 0U;
    if (!have_recent && status == SHADOWSPILL_STATUS_OK) {
        /* Nothing recorded this call. Say so rather than inventing a status:
         * a made-up diagnosis is worse than an admitted absence. */
        append_failure_message(
            destination,
            destination_bytes,
            &offset,
            "ShadowSpill allocator returned no memory and recorded no reason\n"
        );
        return SHADOWSPILL_STATUS_INVALID_STATE;
    }
    /* Name the pool: "out of memory" means very different things for the
     * device execution pool and the spill pool, and an internal failure
     * belongs to neither. */
    const char *pool_name = runtime->pool_id == UINT32_MAX
        ? "no pool"
        : runtime->pool_id == bound_allocator_pool_id() ? "execution pool"
                                                        : "spill pool";
    if (status == SHADOWSPILL_STATUS_NO_PROGRESS) {
        append_failure_message(
            destination, destination_bytes, &offset,
            "ShadowSpill out of memory in the %s (device %d), with nothing "
            "left to release\n",
            pool_name, failure.device_ordinal
        );
    } else if (status == SHADOWSPILL_STATUS_OUT_OF_MEMORY) {
        append_failure_message(
            destination, destination_bytes, &offset,
            "ShadowSpill out of memory in the %s (device %d)\n",
            pool_name, failure.device_ordinal
        );
    } else {
        append_failure_message(
            destination, destination_bytes, &offset,
            "ShadowSpill %s (device %d)\nstatus: %u (%s)\n",
            shadowspill_status_string(status),
            failure.device_ordinal,
            (unsigned int)status,
            shadowspill_status_string(status)
        );
    }
    if (runtime->reason != SHADOWSPILL_FAILURE_REASON_UNSPECIFIED) {
        append_failure_message(
            destination, destination_bytes, &offset, "reason: %s\n",
            shadowspill_failure_reason_string(
                (ShadowSpillFailureReason)runtime->reason
            )
        );
    }
    if (reported_first_failure) {
        append_failure_message(
            destination, destination_bytes, &offset,
            "note: an earlier failure (%s) had already stopped this runtime; "
            "later calls fail because of it, not on their own\n",
            shadowspill_status_string(
                (ShadowSpillStatus)failure.status
            )
        );
    }
    if (runtime->task_id != UINT64_MAX) {
        append_failure_task(
            destination,
            destination_bytes,
            &offset,
            runtime->task_id
        );
    }
    append_bytes(destination, destination_bytes, &offset, "requested",
        requested_bytes);
    if (runtime->free_bytes != 0U || runtime->largest_free_range_bytes != 0U) {
        append_bytes(destination, destination_bytes, &offset, "pool free",
            runtime->free_bytes);
        append_bytes(destination, destination_bytes, &offset,
            "largest free range", runtime->largest_free_range_bytes);
    }
    if (status == SHADOWSPILL_STATUS_TASK_ALLOCATION_ENVELOPE_EXCEEDED) {
        append_failure_message(
            destination,
            destination_bytes,
            &offset,
            "reason: TASK_ALLOCATION_ENVELOPE_EXCEEDED\n"
            "task_live_requested: %llu\n"
            "task_live_charged: %llu\n"
            "task_live_requested_limit: %llu\n"
            "task_live_charged_limit: %llu\n"
            "task_maximum_requested_allocation: %llu\n"
            "task_maximum_charged_allocation: %llu\n",
            (unsigned long long)runtime->task_live_requested_bytes,
            (unsigned long long)runtime->task_live_charged_bytes,
            (unsigned long long)runtime->task_live_requested_limit_bytes,
            (unsigned long long)runtime->task_live_charged_limit_bytes,
            (unsigned long long)
                runtime->task_maximum_requested_allocation_bytes,
            (unsigned long long)
                runtime->task_maximum_charged_allocation_bytes
        );
    } else if (status == SHADOWSPILL_STATUS_TASK_ALLOCATION_CONTRACT_MISMATCH) {
        append_failure_message(
            destination,
            destination_bytes,
            &offset,
            "reason: TASK_ALLOCATION_CONTRACT_MISMATCH\n"
            "task_allocation_operation_index: %llu\n"
            "expected_operation: %s\nactual_operation: %s\n"
            "expected_ordinal: %llu\nactual_ordinal: %llu\n"
            "expected_requested: %llu\nactual_requested: %llu\n"
            "expected_charged: %llu\nactual_charged: %llu\n"
            "expected_alignment: %llu\nactual_alignment: %llu\n",
            (unsigned long long)runtime->task_allocation_operation_index,
            allocation_operation_name(
                runtime->task_allocation_expected_operation
            ),
            allocation_operation_name(
                runtime->task_allocation_actual_operation
            ),
            (unsigned long long)runtime->task_allocation_expected_ordinal,
            (unsigned long long)runtime->task_allocation_actual_ordinal,
            (unsigned long long)
                runtime->task_allocation_expected_requested_bytes,
            (unsigned long long)
                runtime->task_allocation_actual_requested_bytes,
            (unsigned long long)
                runtime->task_allocation_expected_charged_bytes,
            (unsigned long long)
                runtime->task_allocation_actual_charged_bytes,
            (unsigned long long)
                runtime->task_allocation_expected_alignment_bytes,
            (unsigned long long)
                runtime->task_allocation_actual_alignment_bytes
        );
    }
    return status;
}

ShadowSpillStatus shadowspill_pytorch_recover_no_progress(void) {
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    if (runtime == NULL) {
        return SHADOWSPILL_STATUS_CLOSED;
    }
    ShadowSpillStatus status =
        shadowspill_runtime_recover_no_progress(runtime);
    if (status != SHADOWSPILL_STATUS_OK) {
        return status;
    }
    pthread_mutex_lock(&adapter.mutex);
    if (adapter.failure.status == SHADOWSPILL_STATUS_NO_PROGRESS) {
        memset(&adapter.failure, 0, sizeof(adapter.failure));
        adapter.failure_task_label[0] = '\0';
        adapter.failure.device_ordinal = device_ordinal;
        adapter.failure.runtime.task_id = SHADOWSPILL_RUNTIME_NO_ID;
        adapter.failure.runtime.object_id = SHADOWSPILL_RUNTIME_NO_ID;
        adapter.failure.runtime.allocation_id = SHADOWSPILL_RUNTIME_NO_ID;
        adapter.failure.runtime.pool_id = UINT32_MAX;
    } else if (adapter.failure.status != SHADOWSPILL_STATUS_OK) {
        status = (ShadowSpillStatus)adapter.failure.status;
    }
    pthread_mutex_unlock(&adapter.mutex);
    return status;
}

ShadowSpillStatus shadowspill_pytorch_allocator_wait_idle(void) {
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    if (runtime == NULL) {
        return SHADOWSPILL_STATUS_CLOSED;
    }
    return shadowspill_runtime_wait_idle(runtime);
}

ShadowSpillStatus
shadowspill_pytorch_calibrate_transfer_capabilities(
    const ShadowSpillTransferCalibrationConfig *config,
    const ShadowSpillTransferRouteKey *routes,
    uint32_t route_count
) {
    int32_t device_ordinal = -1;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    if (runtime == NULL) {
        return SHADOWSPILL_STATUS_INVALID_STATE;
    }
    return shadowspill_runtime_calibrate_transfer_capabilities(
        runtime, config, routes, route_count
    );
}

ShadowSpillStatus shadowspill_pytorch_transfer_profiles(
    ShadowSpillTransferProfile *profiles,
    uint32_t capacity,
    uint32_t *count,
    uint64_t *generation
) {
    int32_t device_ordinal = -1;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    if (runtime == NULL) {
        return SHADOWSPILL_STATUS_INVALID_STATE;
    }
    return shadowspill_runtime_transfer_profiles(
        runtime, profiles, capacity, count, generation
    );
}

ShadowSpillStatus shadowspill_pytorch_allocation_for_pointer(
    uint64_t address,
    ShadowSpillAllocation *allocation
) {
    if (address == 0U || allocation == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    return runtime == NULL
        ? SHADOWSPILL_STATUS_CLOSED
        : shadowspill_memory_pool_allocation_for_pointer(
              runtime,
              bound_allocator_pool_id(),
              (const void *)(uintptr_t)address,
              allocation
          );
}

ShadowSpillStatus shadowspill_pytorch_register_object(
    uint32_t pool_id,
    uint64_t object_id,
    uint64_t size_bytes,
    uint8_t retain_spill_copy,
    uint64_t source_address
) {
    if (retain_spill_copy > 1U ||
        (size_bytes != 0U && source_address == 0U)) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    if (runtime == NULL) {
        return SHADOWSPILL_STATUS_CLOSED;
    }
    const ShadowSpillObjectDescription description = {
        .object_id = object_id,
        .size_bytes = size_bytes,
        .retain_spill_copy = retain_spill_copy,
        .initial_pool_id = pool_id,
        .initially_resident = 1U,
    };
    ShadowSpillStatus status = shadowspill_register_object(
        runtime, &description
    );
    if (status != SHADOWSPILL_STATUS_OK) {
        return status;
    }
    return shadowspill_write_object(
        runtime,
        object_id,
        pool_id,
        (const void *)(uintptr_t)source_address,
        size_bytes
    );
}

ShadowSpillStatus shadowspill_pytorch_register_placeholder_object(
    uint64_t object_id,
    uint64_t size_bytes,
    uint8_t retain_spill_copy
) {
    if (retain_spill_copy > 1U) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    if (runtime == NULL) {
        return SHADOWSPILL_STATUS_CLOSED;
    }
    const ShadowSpillObjectDescription description = {
        .object_id = object_id,
        .size_bytes = size_bytes,
        .retain_spill_copy = retain_spill_copy,
        .initially_resident = 0U,
    };
    return shadowspill_register_object(runtime, &description);
}

ShadowSpillStatus shadowspill_pytorch_release_caller_allocation(
    uint64_t allocation_id,
    uintptr_t stream
) {
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    return runtime == NULL
        ? SHADOWSPILL_STATUS_CLOSED
        : shadowspill_memory_pool_free(
              runtime,
              bound_allocator_pool_id(),
              allocation_id,
              adapter_stream(stream)
          );
}

ShadowSpillStatus
shadowspill_pytorch_validate_task_replacement_binding(
    uintptr_t task_handle,
    uint32_t publication_ordinal,
    uint64_t retired_address,
    uint64_t successor_address
) {
    if (task_handle == 0U || retired_address == 0U ||
        successor_address == 0U) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    return runtime == NULL
        ? SHADOWSPILL_STATUS_CLOSED
        : shadowspill_task_validate_replacement_binding(
              runtime,
              (const ShadowSpillTaskHandle *)task_handle,
              publication_ordinal,
              (const void *)(uintptr_t)retired_address,
              (const void *)(uintptr_t)successor_address
          );
}

ShadowSpillStatus shadowspill_pytorch_submit_action_batch_handle(
    uintptr_t action_batch_handle,
    uintptr_t trigger_stream_address
) {
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    if (runtime == NULL) {
        return SHADOWSPILL_STATUS_CLOSED;
    }
    if (action_batch_handle == 0U) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    const ShadowSpillProfilerRange range =
        shadowspill_pytorch_profile_range_begin(
            "shadowspill.pytorch.initial_actions"
        );
    const ShadowSpillStatus status =
        shadowspill_submit_action_batch_handle(
            runtime,
            (const ShadowSpillActionBatchHandle *)action_batch_handle,
            adapter_stream(trigger_stream_address)
        );
    shadowspill_pytorch_profile_range_end(range);
    return status;
}

ShadowSpillStatus shadowspill_pytorch_acquire_objects_handle(
    uintptr_t acquisition_handle,
    uintptr_t consumer_stream_address,
    ShadowSpillObjectBinding *bindings,
    uint32_t binding_capacity
) {
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    if (runtime == NULL) {
        return SHADOWSPILL_STATUS_CLOSED;
    }
    return acquisition_handle == 0U
        ? SHADOWSPILL_STATUS_INVALID_ARGUMENT
        : shadowspill_acquire_objects_handle(
            runtime,
            (const ShadowSpillObjectAcquisitionHandle *)acquisition_handle,
            adapter_stream(consumer_stream_address),
            bindings,
            binding_capacity
        );
}

ShadowSpillStatus
shadowspill_pytorch_transfer_acquired_object_to_caller(
    uintptr_t acquisition_handle,
    uint32_t object_ordinal,
    uintptr_t consumer_stream,
    uint64_t expected_address,
    uint64_t expected_generation,
    uint64_t expected_allocation_id,
    ShadowSpillAllocation *allocation
) {
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    return runtime == NULL
        ? SHADOWSPILL_STATUS_CLOSED
        : shadowspill_transfer_acquired_object_to_caller(
              runtime,
              (const ShadowSpillObjectAcquisitionHandle *)acquisition_handle,
              object_ordinal,
              adapter_stream(consumer_stream),
              (const void *)(uintptr_t)expected_address,
              expected_generation,
              expected_allocation_id,
              allocation
          );
}

ShadowSpillStatus shadowspill_pytorch_before_task_handle(
    uintptr_t task_handle,
    uintptr_t compute_stream_address,
    const ShadowSpillObjectBinding **bindings,
    uint32_t *binding_count
) {
    const ShadowSpillTaskHandle *handle =
        (const ShadowSpillTaskHandle *)task_handle;
    const uint64_t task_id = shadowspill_task_id(handle);
    if (task_range_active || task_handle == 0U ||
        task_id == SHADOWSPILL_RUNTIME_NO_ID) {
        return SHADOWSPILL_STATUS_INVALID_STATE;
    }
    if (atomic_load_explicit(
            &adapter.profiler_annotations_enabled, memory_order_relaxed
        ) != 0U) {
        char range_name[384];
        format_task_range_name(range_name, sizeof(range_name), "task", handle);
        task_range_id = shadowspill_pytorch_profile_range_begin(range_name);
    } else {
        task_range_id = 0U;
    }
    task_range_active = 1;
    active_task_label = shadowspill_task_trace_label(handle);
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    ShadowSpillStatus status = runtime == NULL
        ? SHADOWSPILL_STATUS_CLOSED
        : shadowspill_before_task_handle(
            runtime,
            handle,
            adapter_stream(compute_stream_address),
            bindings,
            binding_count
        );
    if (status != SHADOWSPILL_STATUS_OK) {
        end_task_range();
    }
    return status;
}

ShadowSpillStatus shadowspill_pytorch_wait_task_allocations(
    uintptr_t task_handle,
    uintptr_t compute_stream_address
) {
    const ShadowSpillTaskHandle *handle =
        (const ShadowSpillTaskHandle *)task_handle;
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    return runtime == NULL || task_handle == 0U
        ? SHADOWSPILL_STATUS_CLOSED
        : shadowspill_wait_task_allocations_handle(
            runtime,
            handle,
            adapter_stream(compute_stream_address)
        );
}

ShadowSpillStatus shadowspill_pytorch_after_task_handle(
    uintptr_t task_handle,
    uintptr_t compute_stream_address
) {
    const ShadowSpillTaskHandle *handle =
        (const ShadowSpillTaskHandle *)task_handle;
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    ShadowSpillStatus status =
        runtime == NULL || task_handle == 0U
        ? SHADOWSPILL_STATUS_CLOSED
        : shadowspill_after_task_handle(
            runtime,
            handle,
            adapter_stream(compute_stream_address)
        );
    end_task_range();
    return status;
}

ShadowSpillStatus shadowspill_pytorch_allocation_scope_begin(
    uint64_t scope_id
) {
    if (task_range_active) {
        return SHADOWSPILL_STATUS_INVALID_STATE;
    }
    const uint64_t profiling_base = UINT64_C(1) << 62U;
    char range_name[192];
    (void)snprintf(
        range_name,
        sizeof(range_name),
        "shadowspill.pytorch.profiling.allocation_scope_%06llu",
        (unsigned long long)(
            scope_id >= profiling_base ? scope_id - profiling_base : scope_id
        )
    );
    task_range_id = shadowspill_pytorch_profile_range_begin(range_name);
    task_range_active = 1;
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    const ShadowSpillStatus status = runtime == NULL
        ? SHADOWSPILL_STATUS_CLOSED
        : shadowspill_allocation_scope_begin(
              runtime, bound_allocator_pool_id(), scope_id
          );
    if (status != SHADOWSPILL_STATUS_OK) {
        end_task_range();
    }
    return status;
}

ShadowSpillStatus shadowspill_pytorch_allocation_scope_end(
    uint64_t scope_id,
    uintptr_t compute_stream_address
) {
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    const ShadowSpillStatus status = runtime == NULL
        ? SHADOWSPILL_STATUS_CLOSED
        : shadowspill_allocation_scope_end(
              runtime,
              scope_id,
              adapter_stream(compute_stream_address)
          );
    end_task_range();
    return status;
}

void shadowspill_pytorch_allocation_scope_abort(void) {
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    shadowspill_allocation_scope_abort(runtime);
    end_task_range();
}

ShadowSpillStatus shadowspill_pytorch_validate_object_binding(
    uint32_t pool_id,
    uint64_t object_id,
    uint64_t address,
    uint64_t size_bytes
) {
    if (address == 0U && size_bytes != 0U) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    if (runtime == NULL) {
        return SHADOWSPILL_STATUS_CLOSED;
    }
    ShadowSpillObjectLocationSnapshot snapshot = {0};
    ShadowSpillStatus status = shadowspill_object_location_snapshot(
        runtime, object_id, pool_id, &snapshot
    );
    if (status != SHADOWSPILL_STATUS_OK) {
        return status;
    }
    return snapshot.size_bytes == size_bytes && snapshot.has_lease &&
            snapshot.current &&
            snapshot.pointer == (void *)(uintptr_t)address
        ? SHADOWSPILL_STATUS_OK
        : SHADOWSPILL_STATUS_INVALID_STATE;
}

ShadowSpillStatus shadowspill_pytorch_abort_task_handle(
    uintptr_t task_handle
) {
    const ShadowSpillTaskHandle *handle =
        (const ShadowSpillTaskHandle *)task_handle;
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    const ShadowSpillStatus status = runtime == NULL
        ? SHADOWSPILL_STATUS_CLOSED
        : shadowspill_abort_task_handle(
              runtime, handle
          );
    end_task_range();
    return status;
}

void *shadowspill_pytorch_backend_malloc_impl(
    ptrdiff_t bytes,
    int32_t device_ordinal,
    void *stream
) {
    const ShadowSpillProfilerRange range =
        shadowspill_pytorch_profile_range_begin("shadowspill.runtime.allocate");
    pthread_mutex_lock(&adapter.mutex);
    ++adapter.allocation_callbacks;
    if (bytes == 0) {
        ++adapter.zero_size_allocation_callbacks;
    }
    pthread_mutex_unlock(&adapter.mutex);
    int32_t expected_device;
    ShadowSpillRuntime *runtime = acquire_allocator_callback_runtime(
        &expected_device
    );
    if (bytes == 0 && runtime == NULL) {
        shadowspill_pytorch_profile_range_end(range);
        return NULL;
    }
    if (bytes == 0 && runtime != NULL && device_ordinal == expected_device) {
        shadowspill_pytorch_profile_range_end(range);
        release_allocator_callback_runtime();
        return NULL;
    }
    if (runtime == NULL || bytes < 0 || device_ordinal != expected_device) {
        latch_failure(
            runtime == NULL ? SHADOWSPILL_STATUS_CLOSED
                            : SHADOWSPILL_STATUS_INVALID_ARGUMENT,
            device_ordinal,
            NULL,
            bytes < 0 ? 0U : (uint64_t)bytes
        );
        shadowspill_pytorch_profile_range_end(range);
        if (runtime != NULL) {
            release_allocator_callback_runtime();
        }
        return NULL;
    }
    ShadowSpillAllocation allocation = {0};
    ShadowSpillStatus status = shadowspill_memory_pool_allocate(
        runtime,
        bound_allocator_pool_id(),
        (uint64_t)bytes,
        256U,
        adapter_stream((uintptr_t)stream),
        &allocation
    );
    if (status != SHADOWSPILL_STATUS_OK) {
        latch_failure(status, device_ordinal, NULL, (uint64_t)bytes);
        shadowspill_pytorch_profile_range_end(range);
        release_allocator_callback_runtime();
        return NULL;
    }
    shadowspill_pytorch_profile_range_end(range);
    release_allocator_callback_runtime();
    return allocation.pointer;
}

void shadowspill_pytorch_backend_free(
    void *address,
    size_t bytes,
    int32_t device_ordinal,
    void *stream
) {
    pthread_mutex_lock(&adapter.mutex);
    ++adapter.free_callbacks;
    pthread_mutex_unlock(&adapter.mutex);
    if (address == NULL) {
        return;
    }
    int32_t expected_device;
    ShadowSpillRuntime *runtime = acquire_allocator_callback_runtime(
        &expected_device
    );
    if (runtime == NULL) {
        return;
    }
    if (device_ordinal != expected_device) {
        latch_failure(
            SHADOWSPILL_STATUS_INVALID_ARGUMENT,
            device_ordinal,
            address,
            (uint64_t)bytes
        );
        release_allocator_callback_runtime();
        return;
    }
    ShadowSpillAllocation allocation = {0};
    ShadowSpillStatus status =
        shadowspill_memory_pool_allocation_for_pointer(
            runtime, bound_allocator_pool_id(), address, &allocation
    );
    if (status != SHADOWSPILL_STATUS_OK) {
        pthread_mutex_lock(&adapter.mutex);
        ++adapter.pointer_lookup_failures;
        pthread_mutex_unlock(&adapter.mutex);
        latch_failure(status, device_ordinal, address, (uint64_t)bytes);
        release_allocator_callback_runtime();
        return;
    }
    status = shadowspill_memory_pool_free(
        runtime,
        bound_allocator_pool_id(),
        allocation.allocation_id,
        adapter_stream((uintptr_t)stream)
    );
    if (status != SHADOWSPILL_STATUS_OK) {
        latch_failure(status, device_ordinal, address, (uint64_t)bytes);
    }
    release_allocator_callback_runtime();
}

void shadowspill_pytorch_backend_record_stream(void *address, void *stream) {
    pthread_mutex_lock(&adapter.mutex);
    ++adapter.record_stream_callbacks;
    pthread_mutex_unlock(&adapter.mutex);
    if (address == NULL) {
        return;
    }
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = acquire_allocator_callback_runtime(
        &device_ordinal
    );
    if (runtime == NULL) {
        return;
    }
    ShadowSpillAllocation allocation = {0};
    ShadowSpillStatus status =
        shadowspill_memory_pool_allocation_for_pointer(
            runtime, bound_allocator_pool_id(), address, &allocation
    );
    if (status != SHADOWSPILL_STATUS_OK) {
        pthread_mutex_lock(&adapter.mutex);
        ++adapter.pointer_lookup_failures;
        pthread_mutex_unlock(&adapter.mutex);
        latch_failure(status, device_ordinal, address, 0U);
        release_allocator_callback_runtime();
        return;
    }
    status = shadowspill_memory_pool_record_stream(
        runtime,
        bound_allocator_pool_id(),
        allocation.allocation_id,
        adapter_stream((uintptr_t)stream)
    );
    if (status != SHADOWSPILL_STATUS_OK) {
        latch_failure(status, device_ordinal, address, 0U);
    }
    release_allocator_callback_runtime();
}
