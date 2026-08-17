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
    char **task_labels;
    uint32_t task_label_count;
    ShadowSpillPytorchPhysicalAdmission admission;
    ShadowSpillPytorchAdapterFailure failure;
} ShadowSpillPytorchAdapterState;

static ShadowSpillPytorchAdapterState adapter = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .device_ordinal = -1,
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
}

static void free_task_labels(char **labels, uint32_t count) {
    if (labels == NULL) {
        return;
    }
    for (uint32_t index = 0U; index < count; ++index) {
        free(labels[index]);
    }
    free(labels);
}

static void format_task_range_name(
    char *destination,
    size_t destination_bytes,
    const char *operation,
    uint64_t task_id
) {
    const uint64_t caller_handoff_base = UINT64_C(1) << 59U;
    const uint64_t initial_actions_base = UINT64_C(1) << 60U;
    if (task_id >= initial_actions_base) {
        (void)snprintf(
            destination,
            destination_bytes,
            "shadowspill.pytorch.initial_actions.invocation_%06llu",
            (unsigned long long)(task_id - initial_actions_base)
        );
        return;
    }
    if (task_id >= caller_handoff_base) {
        (void)snprintf(
            destination,
            destination_bytes,
            "shadowspill.pytorch.caller_handoff.invocation_%06llu",
            (unsigned long long)(task_id - caller_handoff_base)
        );
        return;
    }
    pthread_mutex_lock(&adapter.mutex);
    const char *label = task_id < adapter.task_label_count
        ? adapter.task_labels[task_id]
        : NULL;
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
    pthread_mutex_unlock(&adapter.mutex);
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
    pthread_mutex_lock(&adapter.mutex);
    ++adapter.callback_failures;
    if (adapter.failure.status == SHADOWSPILL_RUNTIME_OK) {
        adapter.failure.status = (uint32_t)status;
        adapter.failure.device_ordinal = device_ordinal;
        adapter.failure.address = (uint64_t)(uintptr_t)address;
        adapter.failure.requested_bytes = requested_bytes;
        if (adapter.runtime != NULL) {
            (void)shadowspill_runtime_failure(
                adapter.runtime, &adapter.failure.runtime
            );
        }
    }
    pthread_mutex_unlock(&adapter.mutex);
}

static ShadowSpillRuntime *bound_runtime(int32_t *device_ordinal) {
    pthread_mutex_lock(&adapter.mutex);
    ShadowSpillRuntime *runtime = adapter.runtime;
    *device_ordinal = adapter.device_ordinal;
    pthread_mutex_unlock(&adapter.mutex);
    return runtime;
}

static void shadowspill_pytorch_process_exit(void) {
    pthread_mutex_lock(&adapter.mutex);
    ShadowSpillRuntime *runtime = adapter.runtime;
    ShadowSpillCudaBackend *cuda = adapter.cuda;
    char **task_labels = adapter.task_labels;
    const uint32_t task_label_count = adapter.task_label_count;
    ShadowSpillDebugTaskRecord *debug_records = adapter.debug_task_records;
    adapter.runtime = NULL;
    adapter.cuda = NULL;
    adapter.profiler = (ShadowSpillProfiler){0};
    adapter.task_labels = NULL;
    adapter.task_label_count = 0U;
    adapter.debug_task_records = NULL;
    adapter.debug_task_capacity = 0U;
    atomic_store_explicit(
        &adapter.debug_task_timing_enabled, 0U, memory_order_release
    );
    pthread_mutex_unlock(&adapter.mutex);

    /*
     * runtime_destroy stops and joins the worker before releasing anything it
     * can observe. Keep the CUDA backend alive until all lanes, events, pinned
     * registrations, and pool arenas have been explicitly closed.
     */
    shadowspill_runtime_destroy(runtime);
    shadowspill_cuda_backend_destroy(cuda);
    free_task_labels(task_labels, task_label_count);
    free(debug_records);
}

ShadowSpillRuntimeStatus shadowspill_pytorch_allocator_bootstrap(
    const ShadowSpillPytorchAdapterConfig *config
) {
    if (config == NULL ||
        config->abi_version != SHADOWSPILL_PYTORCH_ADAPTER_ABI_VERSION ||
        config->device_ordinal < 0 || config->device_budget_bytes == 0U ||
        config->provider_headroom_bytes >= config->device_budget_bytes) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&adapter.mutex);
    if (adapter.runtime != NULL) {
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
    uint64_t execution_pool_bytes = available - available % physical_granularity;
    if (execution_pool_bytes == 0U) {
        shadowspill_cuda_backend_destroy(cuda);
        return SHADOWSPILL_RUNTIME_OUT_OF_MEMORY;
    }
    const ShadowSpillMemoryPoolDescription pools[] = {
        {
            .pool_id = 0U,
            .capacity_bytes = execution_pool_bytes,
            .minimum_alignment = capabilities.recommended_minimum_alignment,
            .backend = shadowspill_cuda_device_pool_backend(cuda),
        },
        {
            .pool_id = 1U,
            .capacity_bytes = config->spill_pool_bytes,
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
        .pool_count = (uint32_t)(sizeof(pools) / sizeof(pools[0])),
        .routes = routes,
        .route_count = (uint32_t)(sizeof(routes) / sizeof(routes[0])),
        .worker_poll_nanoseconds = config->worker_poll_nanoseconds,
        .synchronization = shadowspill_cuda_synchronization_backend(cuda),
        .profiler = shadowspill_cuda_backend_profiler(cuda),
    };
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillRuntimeStatus status = shadowspill_runtime_create(
        &runtime_config, &runtime
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
    adapter.profiler = runtime_config.profiler;
    adapter.runtime = runtime;
    adapter.device_ordinal = config->device_ordinal;
    adapter.admission = (ShadowSpillPytorchPhysicalAdmission){
        .abi_version = SHADOWSPILL_PYTORCH_ADAPTER_ABI_VERSION,
        .device_ordinal = config->device_ordinal,
        .device_budget_bytes = config->device_budget_bytes,
        .context_bytes = context_memory.process_bytes,
        .provider_headroom_bytes = config->provider_headroom_bytes,
        .execution_pool_bytes = execution_pool_bytes,
        .bootstrap_process_bytes = bootstrap_memory.process_bytes,
        .device_used_bytes = bootstrap_memory.device_used_bytes,
        .device_total_bytes = bootstrap_memory.device_total_bytes,
        .spill_pool_bytes = config->spill_pool_bytes,
    };
    adapter.physical_checks = 1U;
    adapter.peak_process_physical_bytes = bootstrap_memory.process_bytes;
    adapter.observed_external_high_water_bytes =
        bootstrap_memory.process_bytes >
                context_memory.process_bytes + execution_pool_bytes
        ? bootstrap_memory.process_bytes - context_memory.process_bytes -
            execution_pool_bytes
        : 0U;
    adapter.physical_budget_sealed = 0U;
    memset(&adapter.failure, 0, sizeof(adapter.failure));
    adapter.failure.device_ordinal = config->device_ordinal;
    adapter.failure.runtime.object_id = SHADOWSPILL_RUNTIME_NO_ID;
    adapter.failure.runtime.allocation_id = SHADOWSPILL_RUNTIME_NO_ID;
    pthread_mutex_unlock(&adapter.mutex);
    return SHADOWSPILL_RUNTIME_OK;
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
    uint64_t base = admission.context_bytes + admission.execution_pool_bytes;
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
    uint64_t event_pool_reserve
) {
    pthread_mutex_lock(&adapter.mutex);
    ShadowSpillCudaBackend *cuda = adapter.cuda;
    pthread_mutex_unlock(&adapter.mutex);
    if (cuda == NULL) {
        return SHADOWSPILL_RUNTIME_CLOSED;
    }
    if (shadowspill_cuda_backend_seal_event_pool(cuda, event_pool_reserve) != 0) {
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
    char label[1024] = {0};
    pthread_mutex_lock(&adapter.mutex);
    if (task_id < adapter.task_label_count &&
        adapter.task_labels[task_id] != NULL) {
        (void)snprintf(
            label, sizeof(label), "%s", adapter.task_labels[task_id]
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
        adapter.failure.device_ordinal = device_ordinal;
        adapter.failure.runtime.object_id = SHADOWSPILL_RUNTIME_NO_ID;
        adapter.failure.runtime.allocation_id = SHADOWSPILL_RUNTIME_NO_ID;
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

ShadowSpillRuntimeStatus shadowspill_pytorch_resize_spill_pool(
    uint64_t spill_pool_bytes
) {
    pthread_mutex_lock(&adapter.mutex);
    if (adapter.runtime == NULL) {
        pthread_mutex_unlock(&adapter.mutex);
        return SHADOWSPILL_RUNTIME_CLOSED;
    }
    if (adapter.physical_budget_sealed ||
        spill_pool_bytes < adapter.admission.spill_pool_bytes) {
        pthread_mutex_unlock(&adapter.mutex);
        return SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    ShadowSpillRuntimeStatus status = shadowspill_memory_pool_grow(
        adapter.runtime, 1U, spill_pool_bytes
    );
    if (status == SHADOWSPILL_RUNTIME_OK) {
        adapter.admission.spill_pool_bytes = spill_pool_bytes;
    }
    pthread_mutex_unlock(&adapter.mutex);
    return status;
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
              runtime, 0U, (const void *)(uintptr_t)address, allocation
          );
}

ShadowSpillRuntimeStatus shadowspill_pytorch_register_host_object(
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
        .initial_pool_id = 1U,
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
        1U,
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

ShadowSpillRuntimeStatus shadowspill_pytorch_write_spill_object(
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
              1U,
              (const void *)(uintptr_t)source_address,
              size_bytes
          );
}

ShadowSpillRuntimeStatus shadowspill_pytorch_read_spill_object(
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
              1U,
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
              runtime, 0U, allocation_id, shadowspill_cuda_wrap_stream(stream)
          );
}

ShadowSpillRuntimeStatus shadowspill_pytorch_plan_create(
    uint32_t execution_pool_id,
    uint32_t spill_pool_id,
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
    const ShadowSpillRuntimeStatus status =
        shadowspill_plan_create_for_pools(
            runtime, execution_pool_id, spill_pool_id, &plan
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
shadowspill_pytorch_validate_task_publication_binding(
    uintptr_t task_handle,
    uint32_t publication_ordinal,
    uint64_t address,
    uint64_t generation
) {
    if (task_handle == 0U || address == 0U) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    return runtime == NULL
        ? SHADOWSPILL_RUNTIME_CLOSED
        : shadowspill_task_validate_publication_binding(
              runtime,
              (const ShadowSpillTaskHandle *)task_handle,
              publication_ordinal,
              (const void *)(uintptr_t)address,
              generation
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
    uint64_t task_id,
    uintptr_t compute_stream_address,
    ShadowSpillObjectBinding *bindings,
    uint32_t binding_capacity
) {
    record_debug_host_boundary(task_id, 0U);
    if (task_range_active || task_handle == 0U) {
        record_debug_host_boundary(task_id, 1U);
        return SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    char range_name[384];
    format_task_range_name(range_name, sizeof(range_name), "task", task_id);
    task_range_id = shadowspill_pytorch_profile_range_begin(range_name);
    task_range_active = 1;
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    ShadowSpillRuntimeStatus status = runtime == NULL
        ? SHADOWSPILL_RUNTIME_CLOSED
        : shadowspill_before_task_handle(
            runtime,
            (const ShadowSpillTaskHandle *)task_handle,
            shadowspill_cuda_wrap_stream(compute_stream_address),
            bindings,
            binding_capacity
        );
    if (status != SHADOWSPILL_RUNTIME_OK) {
        end_task_range();
    }
    record_debug_host_boundary(task_id, 1U);
    return status;
}

ShadowSpillRuntimeStatus shadowspill_pytorch_after_task_handle(
    uintptr_t task_handle,
    uint64_t task_id,
    uintptr_t compute_stream_address
) {
    record_debug_host_boundary(task_id, 2U);
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    ShadowSpillRuntimeStatus status =
        runtime == NULL || task_handle == 0U
        ? SHADOWSPILL_RUNTIME_CLOSED
        : shadowspill_after_task_handle(
            runtime,
            (const ShadowSpillTaskHandle *)task_handle,
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
        : shadowspill_allocation_scope_begin(runtime, 0U, scope_id);
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

ShadowSpillRuntimeStatus shadowspill_pytorch_validate_spill_binding(
    uint64_t object_id,
    uint64_t address,
    uint64_t size_bytes
) {
    if (address == 0U && size_bytes != 0U) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    ShadowSpillObjectSnapshot snapshot = {0};
    ShadowSpillRuntimeStatus status = shadowspill_pytorch_object_snapshot(
        object_id, &snapshot
    );
    if (status != SHADOWSPILL_RUNTIME_OK) {
        return status;
    }
    return snapshot.size_bytes == size_bytes && snapshot.has_spill_lease &&
            snapshot.spill_current &&
            snapshot.spill_pointer == (void *)(uintptr_t)address
        ? SHADOWSPILL_RUNTIME_OK
        : SHADOWSPILL_RUNTIME_INVALID_STATE;
}

ShadowSpillRuntimeStatus shadowspill_pytorch_task_labels_configure(
    const char *const *task_labels,
    uint32_t task_label_count
) {
    if ((task_labels == NULL) != (task_label_count == 0U)) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    char **labels = NULL;
    if (task_label_count != 0U) {
        labels = calloc(task_label_count, sizeof(*labels));
        if (labels == NULL) {
            return SHADOWSPILL_RUNTIME_OUT_OF_MEMORY;
        }
        for (uint32_t index = 0U; index < task_label_count; ++index) {
            if (task_labels[index] == NULL) {
                continue;
            }
            labels[index] = strdup(task_labels[index]);
            if (labels[index] == NULL) {
                free_task_labels(labels, task_label_count);
                return SHADOWSPILL_RUNTIME_OUT_OF_MEMORY;
            }
        }
    }
    pthread_mutex_lock(&adapter.mutex);
    char **previous = adapter.task_labels;
    uint32_t previous_count = adapter.task_label_count;
    adapter.task_labels = labels;
    adapter.task_label_count = task_label_count;
    pthread_mutex_unlock(&adapter.mutex);
    free_task_labels(previous, previous_count);
    return SHADOWSPILL_RUNTIME_OK;
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
    uintptr_t task_handle,
    uint64_t task_id
) {
    record_debug_host_boundary(task_id, 3U);
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    const ShadowSpillRuntimeStatus status = runtime == NULL
        ? SHADOWSPILL_RUNTIME_CLOSED
        : shadowspill_abort_task_handle(
              runtime, (const ShadowSpillTaskHandle *)task_handle
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
    ShadowSpillRuntime *runtime = bound_runtime(&expected_device);
    if (bytes == 0 && runtime != NULL && device_ordinal == expected_device) {
        shadowspill_pytorch_profile_range_end(range);
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
        return NULL;
    }
    ShadowSpillAllocation allocation = {0};
    ShadowSpillRuntimeStatus status = shadowspill_memory_pool_allocate(
        runtime,
        0U,
        (uint64_t)bytes,
        256U,
        shadowspill_cuda_wrap_stream((uintptr_t)stream),
        &allocation
    );
    if (status != SHADOWSPILL_RUNTIME_OK) {
        latch_failure(status, device_ordinal, NULL, (uint64_t)bytes);
        shadowspill_pytorch_profile_range_end(range);
        return NULL;
    }
    shadowspill_pytorch_profile_range_end(range);
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
    ShadowSpillRuntime *runtime = bound_runtime(&expected_device);
    if (runtime == NULL || device_ordinal != expected_device) {
        latch_failure(
            runtime == NULL ? SHADOWSPILL_RUNTIME_CLOSED
                            : SHADOWSPILL_RUNTIME_INVALID_ARGUMENT,
            device_ordinal,
            address,
            (uint64_t)bytes
        );
        return;
    }
    ShadowSpillAllocation allocation = {0};
    ShadowSpillRuntimeStatus status =
        shadowspill_memory_pool_allocation_for_pointer(
            runtime, 0U, address, &allocation
    );
    if (status != SHADOWSPILL_RUNTIME_OK) {
        pthread_mutex_lock(&adapter.mutex);
        ++adapter.pointer_lookup_failures;
        pthread_mutex_unlock(&adapter.mutex);
        latch_failure(status, device_ordinal, address, (uint64_t)bytes);
        return;
    }
    status = shadowspill_memory_pool_free(
        runtime,
        0U,
        allocation.allocation_id,
        shadowspill_cuda_wrap_stream((uintptr_t)stream)
    );
    if (status != SHADOWSPILL_RUNTIME_OK) {
        latch_failure(status, device_ordinal, address, (uint64_t)bytes);
    }
}

void shadowspill_pytorch_cuda_record_stream(void *address, void *stream) {
    pthread_mutex_lock(&adapter.mutex);
    ++adapter.record_stream_callbacks;
    pthread_mutex_unlock(&adapter.mutex);
    if (address == NULL) {
        return;
    }
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    if (runtime == NULL) {
        latch_failure(
            SHADOWSPILL_RUNTIME_CLOSED, device_ordinal, address, 0U
        );
        return;
    }
    ShadowSpillAllocation allocation = {0};
    ShadowSpillRuntimeStatus status =
        shadowspill_memory_pool_allocation_for_pointer(
            runtime, 0U, address, &allocation
    );
    if (status != SHADOWSPILL_RUNTIME_OK) {
        pthread_mutex_lock(&adapter.mutex);
        ++adapter.pointer_lookup_failures;
        pthread_mutex_unlock(&adapter.mutex);
        latch_failure(status, device_ordinal, address, 0U);
        return;
    }
    status = shadowspill_memory_pool_record_stream(
        runtime,
        0U,
        allocation.allocation_id,
        shadowspill_cuda_wrap_stream((uintptr_t)stream)
    );
    if (status != SHADOWSPILL_RUNTIME_OK) {
        latch_failure(status, device_ordinal, address, 0U);
    }
}
