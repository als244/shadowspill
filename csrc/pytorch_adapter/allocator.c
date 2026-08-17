#define _POSIX_C_SOURCE 200809L

#include <shadowspill/pytorch_adapter.h>

#include "allocator_internal.h"

#include <cuda.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdint.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct ShadowSpillDebugTaskRecord ShadowSpillDebugTaskRecord;

typedef struct ShadowSpillPytorchAdapterState {
    pthread_mutex_t mutex;
    ShadowSpillCudaBackend *cuda;
    ShadowSpillRuntime *runtime;
    ShadowSpillProfiler profiler;
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
    ShadowSpillDebugTaskRecord *debug_task_records;
    uint32_t debug_task_capacity;
    _Atomic uint8_t debug_task_timing_enabled;
    _Atomic uint8_t shutdown_started;
    _Atomic uint64_t active_allocator_callbacks;
    _Atomic(ShadowSpillRuntime *) published_runtime;
    _Atomic int32_t published_device_ordinal;
    _Atomic uint32_t published_allocator_pool_id;
    char failure_task_label[SHADOWSPILL_RUNTIME_TRACE_LABEL_MAX_BYTES + 1U];
    ShadowSpillPytorchPhysicalAdmission admission;
    ShadowSpillPytorchAdapterFailure failure;
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

struct ShadowSpillDebugTaskRecord {
    uint64_t task_id;
    _Atomic uint64_t before_task_enter_timestamp_ns;
    _Atomic uint64_t before_task_exit_timestamp_ns;
    _Atomic uint64_t after_task_enter_timestamp_ns;
    _Atomic uint64_t after_task_exit_timestamp_ns;
};

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
    return adapter.profiler.range_begin == NULL
        ? 0U
        : adapter.profiler.range_begin(adapter.profiler.context, name);
}

void shadowspill_pytorch_profile_range_end(ShadowSpillProfilerRange range) {
    if (adapter.profiler.range_end != NULL) {
        adapter.profiler.range_end(adapter.profiler.context, range);
    }
}

ShadowSpillRuntimeStatus shadowspill_pytorch_profiler_annotations_set(
    uint8_t enabled
) {
    pthread_mutex_lock(&adapter.mutex);
    const int available = adapter.runtime != NULL &&
        adapter.profiler.abi_version != 0U &&
        adapter.profiler.set_enabled != NULL;
    ShadowSpillProfiler profiler = adapter.profiler;
    pthread_mutex_unlock(&adapter.mutex);
    if (!available) {
        return SHADOWSPILL_RUNTIME_CLOSED;
    }
    profiler.set_enabled(profiler.context, enabled != 0U);
    return SHADOWSPILL_RUNTIME_OK;
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

static uint64_t monotonic_nanoseconds(void) {
    struct timespec value = {0};
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        return 0U;
    }
    return (uint64_t)value.tv_sec * 1000000000U + (uint64_t)value.tv_nsec;
}

static void record_debug_host_boundary(
    uint64_t task_id,
    uint8_t boundary
) {
    if (!atomic_load_explicit(
            &adapter.debug_task_timing_enabled, memory_order_acquire
        ) || task_id >= adapter.debug_task_capacity ||
        adapter.debug_task_records == NULL) {
        return;
    }
    ShadowSpillDebugTaskRecord *record = &adapter.debug_task_records[task_id];
    _Atomic uint64_t *destination = boundary == 0U
        ? &record->before_task_enter_timestamp_ns
        : boundary == 1U
            ? &record->before_task_exit_timestamp_ns
            : boundary == 2U
                ? &record->after_task_enter_timestamp_ns
                : &record->after_task_exit_timestamp_ns;
    atomic_store_explicit(
        destination, monotonic_nanoseconds(), memory_order_release
    );
}

static void latch_failure(
    ShadowSpillRuntimeStatus status,
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
    if (adapter.failure.status == SHADOWSPILL_RUNTIME_OK) {
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

static ShadowSpillRuntimeStatus close_adapter_runtime(
    int require_no_caller_allocations
) {
    pthread_mutex_lock(&adapter.mutex);
    if (adapter.closed) {
        pthread_mutex_unlock(&adapter.mutex);
        return SHADOWSPILL_RUNTIME_OK;
    }
    if (atomic_load_explicit(
            &adapter.shutdown_started, memory_order_acquire
        ) != 0U) {
        pthread_mutex_unlock(&adapter.mutex);
        return SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    ShadowSpillRuntime *runtime = adapter.runtime;
    ShadowSpillCudaBackend *cuda = adapter.cuda;
    ShadowSpillDebugTaskRecord *debug_records = adapter.debug_task_records;
    if (runtime == NULL) {
        pthread_mutex_unlock(&adapter.mutex);
        return SHADOWSPILL_RUNTIME_CLOSED;
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
        const ShadowSpillRuntimeStatus statistics_status =
            shadowspill_runtime_statistics(runtime, &statistics);
        if (statistics_status != SHADOWSPILL_RUNTIME_OK ||
            statistics.caller_owned_allocations != 0U) {
            atomic_store_explicit(
                &adapter.shutdown_started, 0U, memory_order_release
            );
            return statistics_status != SHADOWSPILL_RUNTIME_OK
                ? statistics_status
                : SHADOWSPILL_RUNTIME_INVALID_STATE;
        }
    }

    pthread_mutex_lock(&adapter.mutex);
    if (adapter.runtime != runtime) {
        pthread_mutex_unlock(&adapter.mutex);
        atomic_store_explicit(
            &adapter.shutdown_started, 0U, memory_order_release
        );
        return SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    atomic_store_explicit(
        &adapter.published_runtime, NULL, memory_order_release
    );
    atomic_store_explicit(
        &adapter.published_allocator_pool_id, UINT32_MAX, memory_order_relaxed
    );
    adapter.runtime = NULL;
    adapter.cuda = NULL;
    adapter.profiler = (ShadowSpillProfiler){0};
    adapter.debug_task_records = NULL;
    adapter.debug_task_capacity = 0U;
    atomic_store_explicit(
        &adapter.debug_task_timing_enabled, 0U, memory_order_release
    );
    adapter.closed = 1U;
    pthread_mutex_unlock(&adapter.mutex);

    /*
     * runtime_destroy stops and joins the worker before releasing anything it
     * can observe. Keep the CUDA backend alive until all lanes, events, pinned
     * registrations, and pool arenas have been explicitly closed.
     */
    const ShadowSpillRuntimeStatus status = shadowspill_runtime_close(runtime);
    shadowspill_runtime_destroy(runtime);
    shadowspill_cuda_backend_destroy(cuda);
    free(debug_records);
    return status;
}

static void shadowspill_pytorch_process_exit(void) {
    (void)close_adapter_runtime(0);
}

static int bootstrap_config_is_valid(
    const ShadowSpillPytorchAdapterConfig *config
) {
    if (config == NULL ||
        config->abi_version != SHADOWSPILL_PYTORCH_ADAPTER_ABI_VERSION ||
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
            pool->backend_kind > SHADOWSPILL_PYTORCH_POOL_PINNED_HOST) {
            return 0;
        }
        if (pool->backend_kind == SHADOWSPILL_PYTORCH_POOL_DEVICE) {
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
            config->pools[route->source_pool_id].backend_kind;
        const uint8_t destination_kind =
            config->pools[route->destination_pool_id].backend_kind;
        if (source_kind == destination_kind) {
            return 0;
        }
    }
    return 1;
}

static ShadowSpillRuntimeStatus build_runtime_topology(
    const ShadowSpillPytorchAdapterConfig *config,
    ShadowSpillCudaBackend *cuda,
    const ShadowSpillCudaBackendCapabilities *capabilities,
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
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    for (uint32_t index = 0U; index < config->pool_count; ++index) {
        const ShadowSpillPytorchPoolConfig *source = &config->pools[index];
        const int is_device =
            source->backend_kind == SHADOWSPILL_PYTORCH_POOL_DEVICE;
        pools[index] = (ShadowSpillMemoryPoolDescription){
            .pool_id = index,
            .capacity_bytes = is_device
                ? allocator_pool_bytes : source->capacity_bytes,
            .minimum_alignment = is_device
                ? capabilities->recommended_minimum_alignment : 1U,
            .backend = is_device
                ? shadowspill_cuda_device_pool_backend(cuda)
                : shadowspill_cuda_pinned_pool_backend(cuda),
        };
    }
    for (uint32_t index = 0U; index < config->route_count; ++index) {
        const ShadowSpillPytorchRouteConfig *source = &config->routes[index];
        const uint8_t source_kind =
            config->pools[source->source_pool_id].backend_kind;
        routes[index] = (ShadowSpillTransferRouteDescription){
            .route_id = index,
            .name = source->name,
            .route = source_kind == SHADOWSPILL_PYTORCH_POOL_PINNED_HOST
                ? shadowspill_cuda_fetch_route(
                      cuda,
                      source->source_pool_id,
                      source->destination_pool_id
                  )
                : shadowspill_cuda_evict_route(
                      cuda,
                      source->source_pool_id,
                      source->destination_pool_id
                  ),
        };
    }
    const ShadowSpillRuntimeConfig runtime_config = {
        .abi_version = SHADOWSPILL_RUNTIME_ABI_VERSION,
        .pools = pools,
        .pool_count = config->pool_count,
        .routes = routes,
        .route_count = config->route_count,
        .worker_poll_nanoseconds = config->worker_poll_nanoseconds,
        .synchronization = shadowspill_cuda_synchronization_backend(cuda),
        .profiler = shadowspill_cuda_backend_profiler(cuda),
    };
    const ShadowSpillRuntimeStatus status = shadowspill_runtime_create(
        &runtime_config, runtime
    );
    free(routes);
    free(pools);
    return status;
}

ShadowSpillRuntimeStatus shadowspill_pytorch_allocator_bootstrap(
    const ShadowSpillPytorchAdapterConfig *config
) {
    if (!bootstrap_config_is_valid(config)) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&adapter.mutex);
    if (adapter.bootstrapped) {
        pthread_mutex_unlock(&adapter.mutex);
        return SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    pthread_mutex_unlock(&adapter.mutex);

    const ShadowSpillCudaBackendConfig cuda_config = {
        .abi_version = SHADOWSPILL_CUDA_BACKEND_ABI_VERSION,
        .device_ordinal = config->device_ordinal,
    };
    ShadowSpillCudaBackend *cuda = NULL;
    if (shadowspill_cuda_backend_create(&cuda_config, &cuda) != 0) {
        return SHADOWSPILL_RUNTIME_BACKEND_FAILURE;
    }
    ShadowSpillCudaBackendCapabilities capabilities = {0};
    if (shadowspill_cuda_backend_capabilities(cuda, &capabilities) != 0) {
        shadowspill_cuda_backend_destroy(cuda);
        return SHADOWSPILL_RUNTIME_BACKEND_FAILURE;
    }
    ShadowSpillCudaPhysicalMemory context_memory = {0};
    if (shadowspill_cuda_physical_memory(cuda, &context_memory) != 0) {
        shadowspill_cuda_backend_destroy(cuda);
        return SHADOWSPILL_RUNTIME_BACKEND_FAILURE;
    }
    const uint64_t physical_granularity = 2U << 20U;
    if (config->device_budget_bytes > context_memory.device_total_bytes ||
        context_memory.process_bytes >
            config->device_budget_bytes - config->provider_headroom_bytes) {
        shadowspill_cuda_backend_destroy(cuda);
        return SHADOWSPILL_RUNTIME_OUT_OF_MEMORY;
    }
    uint64_t available = config->device_budget_bytes -
        context_memory.process_bytes - config->provider_headroom_bytes;
    const uint64_t allocator_pool_bytes =
        available - available % physical_granularity;
    if (allocator_pool_bytes == 0U) {
        shadowspill_cuda_backend_destroy(cuda);
        return SHADOWSPILL_RUNTIME_OUT_OF_MEMORY;
    }
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillRuntimeStatus status = build_runtime_topology(
        config,
        cuda,
        &capabilities,
        allocator_pool_bytes,
        &runtime
    );
    if (status != SHADOWSPILL_RUNTIME_OK) {
        shadowspill_cuda_backend_destroy(cuda);
        return status;
    }
    ShadowSpillCudaPhysicalMemory bootstrap_memory = {0};
    if (shadowspill_cuda_physical_memory(cuda, &bootstrap_memory) != 0 ||
        bootstrap_memory.process_bytes > config->device_budget_bytes) {
        shadowspill_runtime_destroy(runtime);
        shadowspill_cuda_backend_destroy(cuda);
        return SHADOWSPILL_RUNTIME_OUT_OF_MEMORY;
    }
    pthread_mutex_lock(&adapter.mutex);
    if (adapter.runtime != NULL) {
        pthread_mutex_unlock(&adapter.mutex);
        shadowspill_runtime_destroy(runtime);
        shadowspill_cuda_backend_destroy(cuda);
        return SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    if (!process_exit_registered) {
        if (atexit(shadowspill_pytorch_process_exit) != 0) {
            pthread_mutex_unlock(&adapter.mutex);
            shadowspill_runtime_destroy(runtime);
            shadowspill_cuda_backend_destroy(cuda);
            return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
        }
        process_exit_registered = 1U;
    }
    adapter.cuda = cuda;
    adapter.profiler = shadowspill_cuda_backend_profiler(cuda);
    adapter.runtime = runtime;
    adapter.bootstrapped = 1U;
    adapter.closed = 0U;
    adapter.device_ordinal = config->device_ordinal;
    adapter.allocator_pool_id = config->allocator_pool_id;
    adapter.admission = (ShadowSpillPytorchPhysicalAdmission){
        .abi_version = SHADOWSPILL_PYTORCH_ADAPTER_ABI_VERSION,
        .device_ordinal = config->device_ordinal,
        .device_budget_bytes = config->device_budget_bytes,
        .context_bytes = context_memory.process_bytes,
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
                context_memory.process_bytes + allocator_pool_bytes
        ? bootstrap_memory.process_bytes - context_memory.process_bytes -
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
    return SHADOWSPILL_RUNTIME_OK;
}

ShadowSpillRuntimeStatus shadowspill_pytorch_allocator_close(void) {
    return close_adapter_runtime(1);
}

ShadowSpillRuntimeStatus shadowspill_pytorch_physical_admission(
    ShadowSpillPytorchPhysicalAdmission *admission
) {
    if (admission == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&adapter.mutex);
    if (adapter.runtime == NULL) {
        pthread_mutex_unlock(&adapter.mutex);
        return SHADOWSPILL_RUNTIME_CLOSED;
    }
    *admission = adapter.admission;
    pthread_mutex_unlock(&adapter.mutex);
    return SHADOWSPILL_RUNTIME_OK;
}

ShadowSpillRuntimeStatus shadowspill_pytorch_physical_memory(
    ShadowSpillCudaPhysicalMemory *memory
) {
    if (memory == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&adapter.mutex);
    ShadowSpillCudaBackend *cuda = adapter.cuda;
    pthread_mutex_unlock(&adapter.mutex);
    if (cuda == NULL) {
        return SHADOWSPILL_RUNTIME_CLOSED;
    }
    return shadowspill_cuda_physical_memory(cuda, memory) == 0
        ? SHADOWSPILL_RUNTIME_OK
        : SHADOWSPILL_RUNTIME_BACKEND_FAILURE;
}

ShadowSpillRuntimeStatus shadowspill_pytorch_check_physical_budget(void) {
    ShadowSpillCudaPhysicalMemory memory = {0};
    pthread_mutex_lock(&adapter.mutex);
    ShadowSpillCudaBackend *cuda = adapter.cuda;
    ShadowSpillPytorchPhysicalAdmission admission = adapter.admission;
    pthread_mutex_unlock(&adapter.mutex);
    if (cuda == NULL) {
        return SHADOWSPILL_RUNTIME_CLOSED;
    }
    if (shadowspill_cuda_physical_memory(cuda, &memory) != 0) {
        return SHADOWSPILL_RUNTIME_BACKEND_FAILURE;
    }
    uint64_t base = admission.context_bytes + admission.allocator_pool_bytes;
    uint64_t external = memory.process_bytes > base
        ? memory.process_bytes - base
        : 0U;
    ShadowSpillRuntimeStatus status =
        memory.process_bytes <= admission.device_budget_bytes &&
            external <= admission.provider_headroom_bytes
        ? SHADOWSPILL_RUNTIME_OK
        : SHADOWSPILL_RUNTIME_PLAN_VIOLATION;
    pthread_mutex_lock(&adapter.mutex);
    ++adapter.physical_checks;
    if (memory.process_bytes > adapter.peak_process_physical_bytes) {
        adapter.peak_process_physical_bytes = memory.process_bytes;
    }
    if (external > adapter.observed_external_high_water_bytes) {
        adapter.observed_external_high_water_bytes = external;
    }
    if (status != SHADOWSPILL_RUNTIME_OK &&
        adapter.failure.status == SHADOWSPILL_RUNTIME_OK) {
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

ShadowSpillRuntimeStatus shadowspill_pytorch_seal_physical_budget(
    uint64_t required_provider_headroom_bytes,
    uint64_t runtime_record_reserve
) {
    pthread_mutex_lock(&adapter.mutex);
    ShadowSpillCudaBackend *cuda = adapter.cuda;
    ShadowSpillRuntime *runtime = adapter.runtime;
    pthread_mutex_unlock(&adapter.mutex);
    if (cuda == NULL || runtime == NULL) {
        return SHADOWSPILL_RUNTIME_CLOSED;
    }
    ShadowSpillRuntimeStatus reserve_status =
        shadowspill_runtime_reserve_event_leases(runtime, runtime_record_reserve);
    if (reserve_status != SHADOWSPILL_RUNTIME_OK) {
        return reserve_status;
    }
    reserve_status = shadowspill_runtime_reserve_retirement_records(
        runtime, runtime_record_reserve
    );
    if (reserve_status != SHADOWSPILL_RUNTIME_OK) {
        return reserve_status;
    }
    for (uint32_t pool_id = 0U;
         pool_id < adapter.admission.pool_count;
         ++pool_id) {
        reserve_status = shadowspill_runtime_reserve_memory_lease_records(
            runtime, pool_id, runtime_record_reserve
        );
        if (reserve_status != SHADOWSPILL_RUNTIME_OK) {
            return reserve_status;
        }
    }
    if (shadowspill_cuda_backend_seal_event_pool(
            cuda, runtime_record_reserve) != 0) {
        return SHADOWSPILL_RUNTIME_BACKEND_FAILURE;
    }
    ShadowSpillRuntimeStatus status =
        shadowspill_pytorch_check_physical_budget();
    if (status != SHADOWSPILL_RUNTIME_OK) {
        return status;
    }
    pthread_mutex_lock(&adapter.mutex);
    if (required_provider_headroom_bytes >
        adapter.admission.provider_headroom_bytes) {
        status = SHADOWSPILL_RUNTIME_PLAN_VIOLATION;
        if (adapter.failure.status == SHADOWSPILL_RUNTIME_OK) {
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

ShadowSpillRuntimeStatus shadowspill_pytorch_adapter_capabilities(
    ShadowSpillPytorchAdapterCapabilities *capabilities
) {
    if (capabilities == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    *capabilities = (ShadowSpillPytorchAdapterCapabilities){
        .abi_version = SHADOWSPILL_PYTORCH_ADAPTER_ABI_VERSION,
        .runtime_abi_version = SHADOWSPILL_RUNTIME_ABI_VERSION,
        .backend_abi_version = SHADOWSPILL_CUDA_BACKEND_ABI_VERSION,
        .slab_memory_strategy = 1U,
        .record_stream_callback = 1U,
#ifdef SHADOWSPILL_PYTORCH_STORAGE_ADAPTER
        .storage_rebinding = 1U,
#else
        .storage_rebinding = 0U,
#endif
        .debug_task_host_timing = 1U,
        .runtime_trace = 1U,
    };
    return SHADOWSPILL_RUNTIME_OK;
}

ShadowSpillRuntimeStatus shadowspill_pytorch_allocator_statistics(
    ShadowSpillPytorchAdapterStatistics *statistics
) {
    if (statistics == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&adapter.mutex);
    ShadowSpillRuntime *runtime = adapter.runtime;
    ShadowSpillCudaBackend *cuda = adapter.cuda;
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
    if (runtime == NULL || cuda == NULL) {
        return SHADOWSPILL_RUNTIME_CLOSED;
    }
    ShadowSpillRuntimeStatus status = shadowspill_runtime_statistics(
        runtime, &statistics->runtime
    );
    shadowspill_cuda_backend_statistics(cuda, &statistics->cuda);
    return status;
}

ShadowSpillRuntimeStatus shadowspill_pytorch_allocator_failure(
    ShadowSpillPytorchAdapterFailure *failure
) {
    if (failure == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&adapter.mutex);
    *failure = adapter.failure;
    ShadowSpillRuntime *runtime = adapter.runtime;
    int32_t device_ordinal = adapter.device_ordinal;
    pthread_mutex_unlock(&adapter.mutex);
    if (failure->status == SHADOWSPILL_RUNTIME_OK && runtime != NULL) {
        ShadowSpillRuntimeFailure runtime_failure = {0};
        if (shadowspill_runtime_failure(runtime, &runtime_failure) ==
                SHADOWSPILL_RUNTIME_OK &&
            runtime_failure.status != SHADOWSPILL_RUNTIME_OK) {
            failure->status = runtime_failure.status;
            failure->device_ordinal = device_ordinal;
            failure->runtime = runtime_failure;
        }
    }
    return (ShadowSpillRuntimeStatus)failure->status;
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

ShadowSpillRuntimeStatus shadowspill_pytorch_cuda_malloc_failure_message(
    char *destination,
    size_t destination_bytes
) {
    if (destination == NULL || destination_bytes == 0U) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    destination[0] = '\0';
    ShadowSpillPytorchAdapterFailure failure = {0};
    ShadowSpillRuntimeStatus status =
        shadowspill_pytorch_allocator_failure(&failure);
    if (status == SHADOWSPILL_RUNTIME_OK) {
        status = SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    const ShadowSpillRuntimeFailure *runtime = &failure.runtime;
    const uint64_t requested_bytes = runtime->requested_bytes != 0U
        ? runtime->requested_bytes
        : failure.requested_bytes;
    size_t offset = 0U;
    if (status == SHADOWSPILL_RUNTIME_NO_PROGRESS) {
        append_failure_message(
            destination,
            destination_bytes,
            &offset,
            "ShadowSpill no-progress OOM\n"
        );
    } else if (status == SHADOWSPILL_RUNTIME_OUT_OF_MEMORY) {
        append_failure_message(
            destination, destination_bytes, &offset, "ShadowSpill OOM\n"
        );
    } else {
        append_failure_message(
            destination,
            destination_bytes,
            &offset,
            "ShadowSpill allocator callback failed\nstatus: %u (%s)\n",
            (unsigned int)status,
            shadowspill_runtime_status_string(status)
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
    append_failure_message(
        destination,
        destination_bytes,
        &offset,
        "device: %d\nrequested: %llu\nfree: %llu\n"
        "largest_free_range: %llu\n",
        failure.device_ordinal,
        (unsigned long long)requested_bytes,
        (unsigned long long)runtime->free_bytes,
        (unsigned long long)runtime->largest_free_range_bytes
    );
    if (status == SHADOWSPILL_RUNTIME_TASK_ALLOCATION_ENVELOPE_EXCEEDED) {
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
    } else if (status == SHADOWSPILL_RUNTIME_TASK_ALLOCATION_CONTRACT_MISMATCH) {
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

ShadowSpillRuntimeStatus shadowspill_pytorch_recover_no_progress(void) {
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    if (runtime == NULL) {
        return SHADOWSPILL_RUNTIME_CLOSED;
    }
    ShadowSpillRuntimeStatus status =
        shadowspill_runtime_recover_no_progress(runtime);
    if (status != SHADOWSPILL_RUNTIME_OK) {
        return status;
    }
    pthread_mutex_lock(&adapter.mutex);
    if (adapter.failure.status == SHADOWSPILL_RUNTIME_NO_PROGRESS) {
        memset(&adapter.failure, 0, sizeof(adapter.failure));
        adapter.failure_task_label[0] = '\0';
        adapter.failure.device_ordinal = device_ordinal;
        adapter.failure.runtime.task_id = SHADOWSPILL_RUNTIME_NO_ID;
        adapter.failure.runtime.object_id = SHADOWSPILL_RUNTIME_NO_ID;
        adapter.failure.runtime.allocation_id = SHADOWSPILL_RUNTIME_NO_ID;
        adapter.failure.runtime.pool_id = UINT32_MAX;
    } else if (adapter.failure.status != SHADOWSPILL_RUNTIME_OK) {
        status = (ShadowSpillRuntimeStatus)adapter.failure.status;
    }
    pthread_mutex_unlock(&adapter.mutex);
    return status;
}

ShadowSpillRuntimeStatus shadowspill_pytorch_allocator_wait_idle(void) {
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    if (runtime == NULL) {
        return SHADOWSPILL_RUNTIME_CLOSED;
    }
    return shadowspill_runtime_wait_idle(runtime);
}

ShadowSpillRuntimeStatus
shadowspill_pytorch_calibrate_transfer_capabilities(
    const ShadowSpillTransferCalibrationConfig *config,
    const ShadowSpillTransferRouteKey *routes,
    uint32_t route_count
) {
    int32_t device_ordinal = -1;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    if (runtime == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    return shadowspill_runtime_calibrate_transfer_capabilities(
        runtime, config, routes, route_count
    );
}

ShadowSpillRuntimeStatus shadowspill_pytorch_transfer_profiles(
    ShadowSpillTransferProfile *profiles,
    uint32_t capacity,
    uint32_t *count,
    uint64_t *generation
) {
    int32_t device_ordinal = -1;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    if (runtime == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    return shadowspill_runtime_transfer_profiles(
        runtime, profiles, capacity, count, generation
    );
}

ShadowSpillRuntimeStatus shadowspill_pytorch_allocation_telemetry_start(
    uint64_t capacity
) {
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    return runtime == NULL
        ? SHADOWSPILL_RUNTIME_CLOSED
        : shadowspill_allocation_telemetry_start(runtime, capacity);
}

ShadowSpillRuntimeStatus shadowspill_pytorch_allocation_telemetry_stop(void) {
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    return runtime == NULL
        ? SHADOWSPILL_RUNTIME_CLOSED
        : shadowspill_allocation_telemetry_stop(runtime);
}

ShadowSpillRuntimeStatus shadowspill_pytorch_allocation_telemetry_read(
    ShadowSpillAllocationEvent *events,
    uint64_t capacity,
    uint64_t *count
) {
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    return runtime == NULL
        ? SHADOWSPILL_RUNTIME_CLOSED
        : shadowspill_allocation_telemetry_read(
              runtime, events, capacity, count
          );
}

ShadowSpillRuntimeStatus shadowspill_pytorch_trace_prepare(
    const ShadowSpillTraceConfig *config
) {
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    return runtime == NULL
        ? SHADOWSPILL_RUNTIME_CLOSED
        : shadowspill_trace_prepare(runtime, config);
}

ShadowSpillRuntimeStatus shadowspill_pytorch_trace_begin(uint64_t step_id) {
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    return runtime == NULL
        ? SHADOWSPILL_RUNTIME_CLOSED
        : shadowspill_trace_begin(runtime, step_id);
}

ShadowSpillRuntimeStatus shadowspill_pytorch_trace_end(void) {
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    return runtime == NULL
        ? SHADOWSPILL_RUNTIME_CLOSED
        : shadowspill_trace_end(runtime);
}

ShadowSpillRuntimeStatus shadowspill_pytorch_trace_read(
    ShadowSpillTraceSummary *summary,
    ShadowSpillTraceEvent *events,
    uint64_t event_capacity,
    ShadowSpillAllocationEvent *allocation_events,
    uint64_t allocation_event_capacity
) {
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    return runtime == NULL
        ? SHADOWSPILL_RUNTIME_CLOSED
        : shadowspill_trace_read(
              runtime,
              summary,
              events,
              event_capacity,
              allocation_events,
              allocation_event_capacity
          );
}

ShadowSpillRuntimeStatus shadowspill_pytorch_allocation_for_pointer(
    uint64_t address,
    ShadowSpillAllocation *allocation
) {
    if (address == 0U || allocation == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    return runtime == NULL
        ? SHADOWSPILL_RUNTIME_CLOSED
        : shadowspill_memory_pool_allocation_for_pointer(
              runtime,
              bound_allocator_pool_id(),
              (const void *)(uintptr_t)address,
              allocation
          );
}

ShadowSpillRuntimeStatus shadowspill_pytorch_register_object(
    uint32_t pool_id,
    uint64_t object_id,
    uint64_t size_bytes,
    uint8_t retain_spill_copy,
    uint64_t source_address
) {
    if (retain_spill_copy > 1U ||
        (size_bytes != 0U && source_address == 0U)) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    if (runtime == NULL) {
        return SHADOWSPILL_RUNTIME_CLOSED;
    }
    const ShadowSpillObjectDescription description = {
        .object_id = object_id,
        .size_bytes = size_bytes,
        .retain_spill_copy = retain_spill_copy,
        .initial_pool_id = pool_id,
        .initially_resident = 1U,
    };
    ShadowSpillRuntimeStatus status = shadowspill_register_object(
        runtime, &description
    );
    if (status != SHADOWSPILL_RUNTIME_OK) {
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

ShadowSpillRuntimeStatus shadowspill_pytorch_register_placeholder_object(
    uint64_t object_id,
    uint64_t size_bytes,
    uint8_t retain_spill_copy
) {
    if (retain_spill_copy > 1U) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    if (runtime == NULL) {
        return SHADOWSPILL_RUNTIME_CLOSED;
    }
    const ShadowSpillObjectDescription description = {
        .object_id = object_id,
        .size_bytes = size_bytes,
        .retain_spill_copy = retain_spill_copy,
        .initially_resident = 0U,
    };
    return shadowspill_register_object(runtime, &description);
}

ShadowSpillRuntimeStatus shadowspill_pytorch_rekey_object(
    uint64_t object_id,
    uint64_t replacement_object_id
) {
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    return runtime == NULL
        ? SHADOWSPILL_RUNTIME_CLOSED
        : shadowspill_rekey_object(runtime, object_id, replacement_object_id);
}

ShadowSpillRuntimeStatus shadowspill_pytorch_write_object(
    uint32_t pool_id,
    uint64_t object_id,
    uint64_t size_bytes,
    uint64_t source_address
) {
    if (size_bytes != 0U && source_address == 0U) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    return runtime == NULL
        ? SHADOWSPILL_RUNTIME_CLOSED
        : shadowspill_write_object(
              runtime,
              object_id,
              pool_id,
              (const void *)(uintptr_t)source_address,
              size_bytes
          );
}

ShadowSpillRuntimeStatus shadowspill_pytorch_read_object(
    uint32_t pool_id,
    uint64_t object_id,
    uint64_t size_bytes,
    uint64_t destination_address
) {
    if (size_bytes != 0U && destination_address == 0U) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    return runtime == NULL
        ? SHADOWSPILL_RUNTIME_CLOSED
        : shadowspill_read_object(
              runtime,
              object_id,
              pool_id,
              (void *)(uintptr_t)destination_address,
              size_bytes
          );
}

ShadowSpillRuntimeStatus shadowspill_pytorch_unregister_object(
    uint64_t object_id
) {
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    return runtime == NULL
        ? SHADOWSPILL_RUNTIME_CLOSED
        : shadowspill_unregister_object(runtime, object_id);
}

ShadowSpillRuntimeStatus shadowspill_pytorch_release_caller_allocation(
    uint64_t allocation_id,
    uintptr_t stream
) {
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    return runtime == NULL
        ? SHADOWSPILL_RUNTIME_CLOSED
        : shadowspill_memory_pool_free(
              runtime,
              bound_allocator_pool_id(),
              allocation_id,
              shadowspill_cuda_wrap_stream(stream)
          );
}

ShadowSpillRuntimeStatus shadowspill_pytorch_plan_create(
    uint32_t execution_pool_id,
    uint32_t spill_pool_id,
    uint32_t fetch_route_id,
    uint32_t evict_route_id,
    uintptr_t *plan_handle
) {
    if (plan_handle == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    *plan_handle = 0U;
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    if (runtime == NULL) {
        return SHADOWSPILL_RUNTIME_CLOSED;
    }
    ShadowSpillPlan *plan = NULL;
    const ShadowSpillPlanDescription description = {
        .execution_pool_id = execution_pool_id,
        .spill_pool_id = spill_pool_id,
        .fetch_route_id = fetch_route_id,
        .evict_route_id = evict_route_id,
    };
    const ShadowSpillRuntimeStatus status = shadowspill_plan_create(
        runtime, &description, &plan
    );
    if (status == SHADOWSPILL_RUNTIME_OK) {
        *plan_handle = (uintptr_t)plan;
    }
    return status;
}

ShadowSpillRuntimeStatus shadowspill_pytorch_plan_close(
    uintptr_t plan_handle
) {
    return plan_handle == 0U
        ? SHADOWSPILL_RUNTIME_INVALID_ARGUMENT
        : shadowspill_plan_close((ShadowSpillPlan *)plan_handle);
}

void shadowspill_pytorch_plan_destroy(uintptr_t plan_handle) {
    shadowspill_plan_destroy((ShadowSpillPlan *)plan_handle);
}

ShadowSpillRuntimeStatus shadowspill_pytorch_object_handle_acquire(
    uint64_t runtime_object_id,
    uintptr_t *object_handle
) {
    if (object_handle == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    *object_handle = 0U;
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    if (runtime == NULL) {
        return SHADOWSPILL_RUNTIME_CLOSED;
    }
    ShadowSpillObjectHandle *handle = NULL;
    const ShadowSpillRuntimeStatus status = shadowspill_object_handle_acquire(
        runtime, runtime_object_id, &handle
    );
    if (status == SHADOWSPILL_RUNTIME_OK) {
        *object_handle = (uintptr_t)handle;
    }
    return status;
}

ShadowSpillRuntimeStatus shadowspill_pytorch_object_handle_release(
    uintptr_t object_handle
) {
    return shadowspill_object_handle_release(
        (ShadowSpillObjectHandle *)object_handle
    );
}

ShadowSpillRuntimeStatus shadowspill_pytorch_object_release_generation(
    uintptr_t object_handle,
    uint64_t expected_generation
) {
    return object_handle == 0U
        ? SHADOWSPILL_RUNTIME_INVALID_ARGUMENT
        : shadowspill_object_release_generation(
              (const ShadowSpillObjectHandle *)object_handle,
              expected_generation
          );
}

ShadowSpillRuntimeStatus shadowspill_pytorch_plan_bind_object(
    uintptr_t plan_handle,
    uint64_t plan_object_id,
    uintptr_t object_handle,
    uint8_t consistency
) {
    return plan_handle == 0U || object_handle == 0U
        ? SHADOWSPILL_RUNTIME_INVALID_ARGUMENT
        : shadowspill_plan_bind_object(
              (ShadowSpillPlan *)plan_handle,
              plan_object_id,
              (const ShadowSpillObjectHandle *)object_handle,
              consistency
          );
}

ShadowSpillRuntimeStatus shadowspill_pytorch_plan_admit_task(
    uintptr_t plan_handle,
    const ShadowSpillTaskDescription *description,
    uintptr_t *task_handle
) {
    if (plan_handle == 0U || task_handle == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    *task_handle = 0U;
    const ShadowSpillTaskHandle *handle = NULL;
    const ShadowSpillRuntimeStatus status = shadowspill_plan_admit_task(
        (ShadowSpillPlan *)plan_handle, description, &handle
    );
    if (status == SHADOWSPILL_RUNTIME_OK) {
        *task_handle = (uintptr_t)handle;
    }
    return status;
}

ShadowSpillRuntimeStatus shadowspill_pytorch_plan_publish_initial_allocation(
    uintptr_t plan_handle,
    uint64_t plan_object_id,
    uint64_t address,
    ShadowSpillObjectBinding *binding
) {
    return plan_handle == 0U || address == 0U || binding == NULL
        ? SHADOWSPILL_RUNTIME_INVALID_ARGUMENT
        : shadowspill_plan_publish_initial_allocation(
              (ShadowSpillPlan *)plan_handle,
              plan_object_id,
              (const void *)(uintptr_t)address,
              binding
          );
}

ShadowSpillRuntimeStatus shadowspill_pytorch_task_publish_allocation(
    uintptr_t task_handle,
    uint32_t publication_ordinal,
    uint64_t address,
    ShadowSpillObjectBinding *binding
) {
    if (task_handle == 0U || address == 0U || binding == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    if (runtime == NULL) {
        return SHADOWSPILL_RUNTIME_CLOSED;
    }
    return shadowspill_task_publish_allocation(
        runtime,
        (const ShadowSpillTaskHandle *)task_handle,
        publication_ordinal,
        (const void *)(uintptr_t)address,
        binding
    );
}

ShadowSpillRuntimeStatus
shadowspill_pytorch_validate_task_replacement_binding(
    uintptr_t task_handle,
    uint32_t publication_ordinal,
    uint64_t retired_address,
    uint64_t successor_address
) {
    if (task_handle == 0U || retired_address == 0U ||
        successor_address == 0U) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    return runtime == NULL
        ? SHADOWSPILL_RUNTIME_CLOSED
        : shadowspill_task_validate_replacement_binding(
              runtime,
              (const ShadowSpillTaskHandle *)task_handle,
              publication_ordinal,
              (const void *)(uintptr_t)retired_address,
              (const void *)(uintptr_t)successor_address
          );
}

ShadowSpillRuntimeStatus shadowspill_pytorch_plan_admit_fixed_layout(
    uintptr_t plan_handle,
    const ShadowSpillFixedLayoutDescription *description
) {
    return plan_handle == 0U
        ? SHADOWSPILL_RUNTIME_INVALID_ARGUMENT
        : shadowspill_plan_admit_fixed_layout(
              (ShadowSpillPlan *)plan_handle, description
          );
}

ShadowSpillRuntimeStatus shadowspill_pytorch_plan_seal_fixed_layout(
    uintptr_t plan_handle
) {
    return plan_handle == 0U
        ? SHADOWSPILL_RUNTIME_INVALID_ARGUMENT
        : shadowspill_plan_seal_fixed_layout(
              (ShadowSpillPlan *)plan_handle
          );
}

ShadowSpillRuntimeStatus shadowspill_pytorch_plan_clear_tasks(
    uintptr_t plan_handle
) {
    return plan_handle == 0U
        ? SHADOWSPILL_RUNTIME_INVALID_ARGUMENT
        : shadowspill_plan_clear_tasks((ShadowSpillPlan *)plan_handle);
}

ShadowSpillRuntimeStatus shadowspill_pytorch_plan_wait_idle(
    uintptr_t plan_handle
) {
    return plan_handle == 0U
        ? SHADOWSPILL_RUNTIME_INVALID_ARGUMENT
        : shadowspill_plan_wait_idle((ShadowSpillPlan *)plan_handle);
}

ShadowSpillRuntimeStatus shadowspill_pytorch_plan_admit_object_acquisition(
    uintptr_t plan_handle,
    const uint64_t *object_ids,
    uint32_t object_count,
    uintptr_t *acquisition_handle
) {
    if (plan_handle == 0U || acquisition_handle == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    *acquisition_handle = 0U;
    const ShadowSpillObjectAcquisitionHandle *handle = NULL;
    const ShadowSpillRuntimeStatus status =
        shadowspill_plan_admit_object_acquisition(
            (ShadowSpillPlan *)plan_handle,
            object_ids,
            object_count,
            &handle
        );
    if (status == SHADOWSPILL_RUNTIME_OK) {
        *acquisition_handle = (uintptr_t)handle;
    }
    return status;
}

ShadowSpillRuntimeStatus shadowspill_pytorch_plan_admit_action_batch(
    uintptr_t plan_handle,
    uint64_t batch_id,
    const ShadowSpillRuntimeAction *actions,
    uint32_t action_count,
    uintptr_t *action_batch_handle
) {
    if (plan_handle == 0U || action_batch_handle == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    *action_batch_handle = 0U;
    const ShadowSpillActionBatchHandle *handle = NULL;
    const ShadowSpillRuntimeStatus status = shadowspill_plan_admit_action_batch(
        (ShadowSpillPlan *)plan_handle,
        batch_id,
        actions,
        action_count,
        &handle
    );
    if (status == SHADOWSPILL_RUNTIME_OK) {
        *action_batch_handle = (uintptr_t)handle;
    }
    return status;
}

ShadowSpillRuntimeStatus shadowspill_pytorch_submit_action_batch_handle(
    uintptr_t action_batch_handle,
    uintptr_t trigger_stream_address
) {
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    if (runtime == NULL) {
        return SHADOWSPILL_RUNTIME_CLOSED;
    }
    if (action_batch_handle == 0U) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    const ShadowSpillProfilerRange range =
        shadowspill_pytorch_profile_range_begin(
            "shadowspill.pytorch.initial_actions"
        );
    const ShadowSpillRuntimeStatus status =
        shadowspill_submit_action_batch_handle(
            runtime,
            (const ShadowSpillActionBatchHandle *)action_batch_handle,
            shadowspill_cuda_wrap_stream(trigger_stream_address)
        );
    shadowspill_pytorch_profile_range_end(range);
    return status;
}

ShadowSpillRuntimeStatus shadowspill_pytorch_acquire_objects_handle(
    uintptr_t acquisition_handle,
    uintptr_t consumer_stream_address,
    ShadowSpillObjectBinding *bindings,
    uint32_t binding_capacity
) {
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    if (runtime == NULL) {
        return SHADOWSPILL_RUNTIME_CLOSED;
    }
    return acquisition_handle == 0U
        ? SHADOWSPILL_RUNTIME_INVALID_ARGUMENT
        : shadowspill_acquire_objects_handle(
            runtime,
            (const ShadowSpillObjectAcquisitionHandle *)acquisition_handle,
            shadowspill_cuda_wrap_stream(consumer_stream_address),
            bindings,
            binding_capacity
        );
}

ShadowSpillRuntimeStatus
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
        ? SHADOWSPILL_RUNTIME_CLOSED
        : shadowspill_transfer_acquired_object_to_caller(
              runtime,
              (const ShadowSpillObjectAcquisitionHandle *)acquisition_handle,
              object_ordinal,
              shadowspill_cuda_wrap_stream(consumer_stream),
              (const void *)(uintptr_t)expected_address,
              expected_generation,
              expected_allocation_id,
              allocation
          );
}

ShadowSpillRuntimeStatus shadowspill_pytorch_before_task_handle(
    uintptr_t task_handle,
    uintptr_t compute_stream_address,
    const ShadowSpillObjectBinding **bindings,
    uint32_t *binding_count
) {
    const ShadowSpillTaskHandle *handle =
        (const ShadowSpillTaskHandle *)task_handle;
    const uint64_t task_id = shadowspill_task_id(handle);
    record_debug_host_boundary(task_id, 0U);
    if (task_range_active || task_handle == 0U ||
        task_id == SHADOWSPILL_RUNTIME_NO_ID) {
        record_debug_host_boundary(task_id, 1U);
        return SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    char range_name[384];
    format_task_range_name(range_name, sizeof(range_name), "task", handle);
    task_range_id = shadowspill_pytorch_profile_range_begin(range_name);
    task_range_active = 1;
    active_task_label = shadowspill_task_trace_label(handle);
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    ShadowSpillRuntimeStatus status = runtime == NULL
        ? SHADOWSPILL_RUNTIME_CLOSED
        : shadowspill_before_task_handle(
            runtime,
            handle,
            shadowspill_cuda_wrap_stream(compute_stream_address),
            bindings,
            binding_count
        );
    if (status != SHADOWSPILL_RUNTIME_OK) {
        end_task_range();
    }
    record_debug_host_boundary(task_id, 1U);
    return status;
}

ShadowSpillRuntimeStatus shadowspill_pytorch_after_task_handle(
    uintptr_t task_handle,
    uintptr_t compute_stream_address
) {
    const ShadowSpillTaskHandle *handle =
        (const ShadowSpillTaskHandle *)task_handle;
    const uint64_t task_id = shadowspill_task_id(handle);
    record_debug_host_boundary(task_id, 2U);
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    ShadowSpillRuntimeStatus status =
        runtime == NULL || task_handle == 0U
        ? SHADOWSPILL_RUNTIME_CLOSED
        : shadowspill_after_task_handle(
            runtime,
            handle,
            shadowspill_cuda_wrap_stream(compute_stream_address)
        );
    end_task_range();
    record_debug_host_boundary(task_id, 3U);
    return status;
}

ShadowSpillRuntimeStatus shadowspill_pytorch_allocation_scope_begin(
    uint64_t scope_id
) {
    if (task_range_active) {
        return SHADOWSPILL_RUNTIME_INVALID_STATE;
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
    const ShadowSpillRuntimeStatus status = runtime == NULL
        ? SHADOWSPILL_RUNTIME_CLOSED
        : shadowspill_allocation_scope_begin(
              runtime, bound_allocator_pool_id(), scope_id
          );
    if (status != SHADOWSPILL_RUNTIME_OK) {
        end_task_range();
    }
    return status;
}

ShadowSpillRuntimeStatus shadowspill_pytorch_allocation_scope_end(
    uint64_t scope_id,
    uintptr_t compute_stream_address
) {
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    const ShadowSpillRuntimeStatus status = runtime == NULL
        ? SHADOWSPILL_RUNTIME_CLOSED
        : shadowspill_allocation_scope_end(
              runtime,
              scope_id,
              shadowspill_cuda_wrap_stream(compute_stream_address)
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

ShadowSpillRuntimeStatus shadowspill_pytorch_object_snapshot(
    uint64_t object_id,
    ShadowSpillObjectSnapshot *snapshot
) {
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    if (runtime == NULL) {
        return SHADOWSPILL_RUNTIME_CLOSED;
    }
    return shadowspill_object_snapshot(runtime, object_id, snapshot);
}

ShadowSpillRuntimeStatus shadowspill_pytorch_object_location_snapshot(
    uint64_t object_id,
    uint32_t pool_id,
    ShadowSpillObjectLocationSnapshot *snapshot
) {
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    return runtime == NULL
        ? SHADOWSPILL_RUNTIME_CLOSED
        : shadowspill_object_location_snapshot(
              runtime, object_id, pool_id, snapshot
          );
}

ShadowSpillRuntimeStatus shadowspill_pytorch_validate_object_binding(
    uint32_t pool_id,
    uint64_t object_id,
    uint64_t address,
    uint64_t size_bytes
) {
    if (address == 0U && size_bytes != 0U) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    ShadowSpillObjectLocationSnapshot snapshot = {0};
    ShadowSpillRuntimeStatus status = shadowspill_pytorch_object_location_snapshot(
        object_id, pool_id, &snapshot
    );
    if (status != SHADOWSPILL_RUNTIME_OK) {
        return status;
    }
    return snapshot.size_bytes == size_bytes && snapshot.has_lease &&
            snapshot.current &&
            snapshot.pointer == (void *)(uintptr_t)address
        ? SHADOWSPILL_RUNTIME_OK
        : SHADOWSPILL_RUNTIME_INVALID_STATE;
}

ShadowSpillRuntimeStatus shadowspill_pytorch_debug_task_timing_enable(
    uint32_t task_capacity
) {
    if (task_capacity == 0U) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&adapter.mutex);
    const uint32_t existing_capacity = adapter.debug_task_capacity;
    pthread_mutex_unlock(&adapter.mutex);
    ShadowSpillDebugTaskRecord *records = NULL;
    if (existing_capacity < task_capacity) {
        records = calloc(task_capacity, sizeof(*records));
        if (records == NULL) {
            return SHADOWSPILL_RUNTIME_OUT_OF_MEMORY;
        }
    } else {
        records = adapter.debug_task_records;
        memset(records, 0, (size_t)existing_capacity * sizeof(*records));
        task_capacity = existing_capacity;
    }
    for (uint32_t index = 0U; index < task_capacity; ++index) {
        ShadowSpillDebugTaskRecord *record = &records[index];
        record->task_id = index;
        atomic_init(&record->before_task_enter_timestamp_ns, 0U);
        atomic_init(&record->before_task_exit_timestamp_ns, 0U);
        atomic_init(&record->after_task_enter_timestamp_ns, 0U);
        atomic_init(&record->after_task_exit_timestamp_ns, 0U);
    }
    pthread_mutex_lock(&adapter.mutex);
    ShadowSpillDebugTaskRecord *previous = records != adapter.debug_task_records
        ? adapter.debug_task_records
        : NULL;
    adapter.debug_task_records = records;
    adapter.debug_task_capacity = task_capacity;
    atomic_store_explicit(
        &adapter.debug_task_timing_enabled, 1U, memory_order_release
    );
    pthread_mutex_unlock(&adapter.mutex);
    free(previous);
    return SHADOWSPILL_RUNTIME_OK;
}

ShadowSpillRuntimeStatus shadowspill_pytorch_debug_task_timing_read(
    ShadowSpillPytorchTaskHostTiming *records,
    uint32_t record_capacity,
    uint32_t *record_count
) {
    if (record_count == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    uint32_t required = 0U;
    for (uint32_t index = 0U; index < adapter.debug_task_capacity; ++index) {
        ShadowSpillDebugTaskRecord *record = &adapter.debug_task_records[index];
        const uint64_t before_enter = atomic_load_explicit(
            &record->before_task_enter_timestamp_ns, memory_order_acquire
        );
        const uint64_t after_exit = atomic_load_explicit(
            &record->after_task_exit_timestamp_ns, memory_order_acquire
        );
        if (before_enter == 0U && after_exit == 0U) {
            continue;
        }
        if (records != NULL && required < record_capacity) {
            records[required] = (ShadowSpillPytorchTaskHostTiming){
                .task_id = record->task_id,
                .before_readiness_waits_timestamp_ns = 0U,
                .before_task_compute_timestamp_ns = 0U,
                .after_task_compute_timestamp_ns = 0U,
                .before_readiness_waits_sequence = 0U,
                .before_task_compute_sequence = 0U,
                .after_task_compute_sequence = 0U,
                .before_task_enter_timestamp_ns = atomic_load_explicit(
                    &record->before_task_enter_timestamp_ns,
                    memory_order_acquire
                ),
                .before_task_exit_timestamp_ns = atomic_load_explicit(
                    &record->before_task_exit_timestamp_ns,
                    memory_order_acquire
                ),
                .after_task_enter_timestamp_ns = atomic_load_explicit(
                    &record->after_task_enter_timestamp_ns,
                    memory_order_acquire
                ),
                .after_task_exit_timestamp_ns = atomic_load_explicit(
                    &record->after_task_exit_timestamp_ns,
                    memory_order_acquire
                ),
            };
        }
        ++required;
    }
    *record_count = required;
    if (required > record_capacity) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    return SHADOWSPILL_RUNTIME_OK;
}

ShadowSpillRuntimeStatus shadowspill_pytorch_debug_task_timing_disable(void) {
    atomic_store_explicit(
        &adapter.debug_task_timing_enabled, 0U, memory_order_release
    );
    pthread_mutex_lock(&adapter.mutex);
    pthread_mutex_unlock(&adapter.mutex);
    return SHADOWSPILL_RUNTIME_OK;
}

ShadowSpillRuntimeStatus shadowspill_pytorch_abort_task_handle(
    uintptr_t task_handle
) {
    const ShadowSpillTaskHandle *handle =
        (const ShadowSpillTaskHandle *)task_handle;
    const uint64_t task_id = shadowspill_task_id(handle);
    record_debug_host_boundary(task_id, 3U);
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    const ShadowSpillRuntimeStatus status = runtime == NULL
        ? SHADOWSPILL_RUNTIME_CLOSED
        : shadowspill_abort_task_handle(
              runtime, handle
          );
    end_task_range();
    return status;
}

void *shadowspill_pytorch_cuda_malloc_impl(
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
            runtime == NULL ? SHADOWSPILL_RUNTIME_CLOSED
                            : SHADOWSPILL_RUNTIME_INVALID_ARGUMENT,
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
    ShadowSpillRuntimeStatus status = shadowspill_memory_pool_allocate(
        runtime,
        bound_allocator_pool_id(),
        (uint64_t)bytes,
        256U,
        shadowspill_cuda_wrap_stream((uintptr_t)stream),
        &allocation
    );
    if (status != SHADOWSPILL_RUNTIME_OK) {
        latch_failure(status, device_ordinal, NULL, (uint64_t)bytes);
        shadowspill_pytorch_profile_range_end(range);
        release_allocator_callback_runtime();
        return NULL;
    }
    shadowspill_pytorch_profile_range_end(range);
    release_allocator_callback_runtime();
    return allocation.pointer;
}

void shadowspill_pytorch_cuda_free(
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
            SHADOWSPILL_RUNTIME_INVALID_ARGUMENT,
            device_ordinal,
            address,
            (uint64_t)bytes
        );
        release_allocator_callback_runtime();
        return;
    }
    ShadowSpillAllocation allocation = {0};
    ShadowSpillRuntimeStatus status =
        shadowspill_memory_pool_allocation_for_pointer(
            runtime, bound_allocator_pool_id(), address, &allocation
    );
    if (status != SHADOWSPILL_RUNTIME_OK) {
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
        shadowspill_cuda_wrap_stream((uintptr_t)stream)
    );
    if (status != SHADOWSPILL_RUNTIME_OK) {
        latch_failure(status, device_ordinal, address, (uint64_t)bytes);
    }
    release_allocator_callback_runtime();
}

void shadowspill_pytorch_cuda_record_stream(void *address, void *stream) {
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
    ShadowSpillRuntimeStatus status =
        shadowspill_memory_pool_allocation_for_pointer(
            runtime, bound_allocator_pool_id(), address, &allocation
    );
    if (status != SHADOWSPILL_RUNTIME_OK) {
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
        shadowspill_cuda_wrap_stream((uintptr_t)stream)
    );
    if (status != SHADOWSPILL_RUNTIME_OK) {
        latch_failure(status, device_ordinal, address, 0U);
    }
    release_allocator_callback_runtime();
}
