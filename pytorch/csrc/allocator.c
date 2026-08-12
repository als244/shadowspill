#define _POSIX_C_SOURCE 200809L

#include <shadowspill/pytorch_adapter.h>

#include <cuda.h>
#include <pthread.h>
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
    const ShadowSpillRuntimeConfig runtime_config = {
        .abi_version = SHADOWSPILL_RUNTIME_ABI_VERSION,
        .execution_pool_bytes = execution_pool_bytes,
        .spill_pool_bytes = config->spill_pool_bytes,
        .minimum_alignment = capabilities.recommended_minimum_alignment,
        .worker_poll_nanoseconds = config->worker_poll_nanoseconds,
        .backend = shadowspill_cuda_backend_vtable(cuda),
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
    ShadowSpillRuntimeStatus status = shadowspill_runtime_resize_spill_pool(
        adapter.runtime, spill_pool_bytes
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
        : shadowspill_allocation_for_pointer(
              runtime, (const void *)(uintptr_t)address, allocation
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
        .initially_spill_resident = 1U,
    };
    ShadowSpillRuntimeStatus status = shadowspill_register_object(
        runtime, &description
    );
    if (status != SHADOWSPILL_RUNTIME_OK) {
        return status;
    }
    return shadowspill_write_spill_object(
        runtime,
        object_id,
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
        .initially_spill_resident = 0U,
    };
    return shadowspill_register_object(runtime, &description);
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
        : shadowspill_write_spill_object(
              runtime,
              object_id,
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
        : shadowspill_read_spill_object(
              runtime,
              object_id,
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

ShadowSpillRuntimeStatus shadowspill_pytorch_bind_registered_allocation(
    uint64_t object_id,
    uint64_t address,
    uint64_t size_bytes,
    ShadowSpillObjectBinding *binding
) {
    if (address == 0U || binding == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    if (runtime == NULL) {
        return SHADOWSPILL_RUNTIME_CLOSED;
    }
    ShadowSpillAllocation allocation = {0};
    ShadowSpillRuntimeStatus status = shadowspill_allocation_for_pointer(
        runtime, (const void *)(uintptr_t)address, &allocation
    );
    if (status != SHADOWSPILL_RUNTIME_OK ||
        allocation.requested_bytes < size_bytes) {
        return status == SHADOWSPILL_RUNTIME_OK
            ? SHADOWSPILL_RUNTIME_INVALID_STATE
            : status;
    }
    status = shadowspill_bind_object(runtime, object_id, allocation.allocation_id);
    if (status != SHADOWSPILL_RUNTIME_OK) {
        return status;
    }
    ShadowSpillObjectSnapshot snapshot = {0};
    status = shadowspill_object_snapshot(runtime, object_id, &snapshot);
    if (status != SHADOWSPILL_RUNTIME_OK) {
        return status;
    }
    *binding = (ShadowSpillObjectBinding){
        .object_id = object_id,
        .generation = allocation.generation,
        .allocation_id = allocation.allocation_id,
        .authoritative_version = snapshot.authoritative_version,
        .pointer = allocation.pointer,
    };
    return SHADOWSPILL_RUNTIME_OK;
}

ShadowSpillRuntimeStatus shadowspill_pytorch_replace_registered_allocation(
    uint64_t object_id,
    uint64_t address,
    uint64_t size_bytes,
    ShadowSpillObjectBinding *binding
) {
    if (address == 0U || binding == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    if (runtime == NULL) {
        return SHADOWSPILL_RUNTIME_CLOSED;
    }
    ShadowSpillAllocation allocation = {0};
    ShadowSpillRuntimeStatus status = shadowspill_allocation_for_pointer(
        runtime, (const void *)(uintptr_t)address, &allocation
    );
    if (status != SHADOWSPILL_RUNTIME_OK ||
        allocation.requested_bytes < size_bytes) {
        return status == SHADOWSPILL_RUNTIME_OK
            ? SHADOWSPILL_RUNTIME_INVALID_STATE
            : status;
    }
    return shadowspill_replace_object_allocation(
        runtime, object_id, allocation.allocation_id, binding
    );
}

ShadowSpillRuntimeStatus shadowspill_pytorch_transfer_output_to_caller(
    uint64_t object_id,
    ShadowSpillAllocation *allocation
) {
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    return runtime == NULL
        ? SHADOWSPILL_RUNTIME_CLOSED
        : shadowspill_transfer_object_to_caller(
              runtime, object_id, allocation
          );
}

ShadowSpillRuntimeStatus shadowspill_pytorch_release_caller_allocation(
    uint64_t allocation_id
) {
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    return runtime == NULL
        ? SHADOWSPILL_RUNTIME_CLOSED
        : shadowspill_free(
              runtime, allocation_id, shadowspill_cuda_wrap_stream(0U)
          );
}

ShadowSpillRuntimeStatus shadowspill_pytorch_promote_allocation(
    uint64_t object_id,
    uint64_t address,
    uint64_t size_bytes,
    ShadowSpillObjectBinding *binding
) {
    if (address == 0U || binding == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    if (runtime == NULL) {
        return SHADOWSPILL_RUNTIME_CLOSED;
    }
    ShadowSpillAllocation allocation = {0};
    ShadowSpillRuntimeStatus status = shadowspill_allocation_for_pointer(
        runtime, (const void *)(uintptr_t)address, &allocation
    );
    if (status != SHADOWSPILL_RUNTIME_OK ||
        allocation.requested_bytes < size_bytes) {
        return status == SHADOWSPILL_RUNTIME_OK
            ? SHADOWSPILL_RUNTIME_INVALID_STATE
            : status;
    }
    const ShadowSpillObjectDescription description = {
        .object_id = object_id,
        .size_bytes = size_bytes,
    };
    status = shadowspill_register_object(runtime, &description);
    if (status != SHADOWSPILL_RUNTIME_OK) {
        return status;
    }
    status = shadowspill_bind_object(
        runtime, object_id, allocation.allocation_id
    );
    if (status != SHADOWSPILL_RUNTIME_OK) {
        return status;
    }
    *binding = (ShadowSpillObjectBinding){
        .object_id = object_id,
        .generation = allocation.generation,
        .allocation_id = allocation.allocation_id,
        .authoritative_version = 0U,
        .pointer = allocation.pointer,
    };
    return SHADOWSPILL_RUNTIME_OK;
}

ShadowSpillRuntimeStatus shadowspill_pytorch_validate_object_binding(
    uint64_t object_id,
    uint64_t address,
    uint64_t generation
) {
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    if (runtime == NULL) {
        return SHADOWSPILL_RUNTIME_CLOSED;
    }
    ShadowSpillObjectSnapshot snapshot = {0};
    ShadowSpillRuntimeStatus status = shadowspill_object_snapshot(
        runtime, object_id, &snapshot
    );
    if (status != SHADOWSPILL_RUNTIME_OK) {
        return status;
    }
    const int current_matches = snapshot.generation == generation &&
        (address == 0U ||
         snapshot.execution_pointer == (void *)(uintptr_t)address);
    const int retired_matches = address != 0U &&
        snapshot.retired_generation == generation &&
        snapshot.retired_execution_pointer == (void *)(uintptr_t)address;
    if (!current_matches && !retired_matches) {
        return SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    return SHADOWSPILL_RUNTIME_OK;
}

ShadowSpillRuntimeStatus shadowspill_pytorch_before_task(
    uint64_t task_id,
    uintptr_t compute_stream_address,
    const uint64_t *input_object_ids,
    uint32_t input_count,
    ShadowSpillObjectBinding *bindings,
    uint32_t binding_capacity
) {
    record_debug_host_boundary(task_id, 0U);
    if (task_range_active) {
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
    if (runtime == NULL) {
        end_task_range();
        record_debug_host_boundary(task_id, 1U);
        return SHADOWSPILL_RUNTIME_CLOSED;
    }
    ShadowSpillRuntimeStatus status = shadowspill_before_task(
        runtime,
        task_id,
        shadowspill_cuda_wrap_stream(compute_stream_address),
        input_object_ids,
        input_count,
        bindings,
        binding_capacity
    );
    if (status != SHADOWSPILL_RUNTIME_OK) {
        end_task_range();
        record_debug_host_boundary(task_id, 1U);
        return status;
    }
    record_debug_host_boundary(task_id, 1U);
    return status;
}

ShadowSpillRuntimeStatus shadowspill_pytorch_admit_execution(
    const ShadowSpillExecutionDescription *description
) {
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    return runtime == NULL
        ? SHADOWSPILL_RUNTIME_CLOSED
        : shadowspill_admit_execution(runtime, description);
}

ShadowSpillRuntimeStatus shadowspill_pytorch_clear_execution_plan(void) {
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    return runtime == NULL
        ? SHADOWSPILL_RUNTIME_CLOSED
        : shadowspill_clear_execution_plan(runtime);
}

ShadowSpillRuntimeStatus shadowspill_pytorch_resolve_execution(
    uint64_t task_id,
    uintptr_t *execution_handle
) {
    if (execution_handle == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    if (runtime == NULL) {
        return SHADOWSPILL_RUNTIME_CLOSED;
    }
    const ShadowSpillExecutionHandle *handle = NULL;
    const ShadowSpillRuntimeStatus status = shadowspill_resolve_execution(
        runtime, task_id, &handle
    );
    if (status == SHADOWSPILL_RUNTIME_OK) {
        *execution_handle = (uintptr_t)handle;
    }
    return status;
}

ShadowSpillRuntimeStatus shadowspill_pytorch_before_execution(
    uint64_t task_id,
    uintptr_t compute_stream_address,
    ShadowSpillObjectBinding *bindings,
    uint32_t binding_capacity
) {
    record_debug_host_boundary(task_id, 0U);
    if (task_range_active) {
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
        : shadowspill_before_execution(
            runtime,
            task_id,
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

ShadowSpillRuntimeStatus shadowspill_pytorch_after_execution(
    uint64_t task_id,
    uintptr_t compute_stream_address
) {
    record_debug_host_boundary(task_id, 2U);
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    ShadowSpillRuntimeStatus status = runtime == NULL
        ? SHADOWSPILL_RUNTIME_CLOSED
        : shadowspill_after_execution(
            runtime,
            task_id,
            shadowspill_cuda_wrap_stream(compute_stream_address)
        );
    end_task_range();
    record_debug_host_boundary(task_id, 3U);
    return status;
}

ShadowSpillRuntimeStatus shadowspill_pytorch_before_execution_handle(
    uintptr_t execution_handle,
    uint64_t task_id,
    uintptr_t compute_stream_address,
    ShadowSpillObjectBinding *bindings,
    uint32_t binding_capacity
) {
    record_debug_host_boundary(task_id, 0U);
    if (task_range_active || execution_handle == 0U) {
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
        : shadowspill_before_execution_handle(
            runtime,
            (const ShadowSpillExecutionHandle *)execution_handle,
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

ShadowSpillRuntimeStatus shadowspill_pytorch_after_execution_handle(
    uintptr_t execution_handle,
    uint64_t task_id,
    uintptr_t compute_stream_address
) {
    record_debug_host_boundary(task_id, 2U);
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    ShadowSpillRuntimeStatus status =
        runtime == NULL || execution_handle == 0U
        ? SHADOWSPILL_RUNTIME_CLOSED
        : shadowspill_after_execution_handle(
            runtime,
            (const ShadowSpillExecutionHandle *)execution_handle,
            shadowspill_cuda_wrap_stream(compute_stream_address)
        );
    end_task_range();
    record_debug_host_boundary(task_id, 3U);
    return status;
}

ShadowSpillRuntimeStatus shadowspill_pytorch_after_task(
    uint64_t task_id,
    uintptr_t compute_stream_address,
    const ShadowSpillObjectUpdate *updates,
    uint32_t update_count,
    const ShadowSpillRuntimeAction *actions,
    uint32_t action_count
) {
    record_debug_host_boundary(task_id, 2U);
    if (!task_range_active) {
        char range_name[384];
        format_task_range_name(
            range_name, sizeof(range_name), "after_task", task_id
        );
        task_range_id = shadowspill_pytorch_profile_range_begin(range_name);
        task_range_active = 1;
    }
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    if (runtime == NULL) {
        end_task_range();
        record_debug_host_boundary(task_id, 3U);
        return SHADOWSPILL_RUNTIME_CLOSED;
    }
    ShadowSpillRuntimeStatus status = shadowspill_after_task(
        runtime,
        task_id,
        shadowspill_cuda_wrap_stream(compute_stream_address),
        updates,
        update_count,
        actions,
        action_count
    );
    end_task_range();
    record_debug_host_boundary(task_id, 3U);
    return status;
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

void shadowspill_pytorch_abort_task_range(void) {
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    shadowspill_abort_task(runtime);
    end_task_range();
}

void *shadowspill_pytorch_cuda_malloc(
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
    ShadowSpillRuntimeStatus status = shadowspill_allocate(
        runtime,
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
    ShadowSpillRuntimeStatus status = shadowspill_allocation_for_pointer(
        runtime, address, &allocation
    );
    if (status != SHADOWSPILL_RUNTIME_OK) {
        pthread_mutex_lock(&adapter.mutex);
        ++adapter.pointer_lookup_failures;
        pthread_mutex_unlock(&adapter.mutex);
        latch_failure(status, device_ordinal, address, (uint64_t)bytes);
        return;
    }
    status = shadowspill_free(
        runtime,
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
    ShadowSpillRuntimeStatus status = shadowspill_allocation_for_pointer(
        runtime, address, &allocation
    );
    if (status != SHADOWSPILL_RUNTIME_OK) {
        pthread_mutex_lock(&adapter.mutex);
        ++adapter.pointer_lookup_failures;
        pthread_mutex_unlock(&adapter.mutex);
        latch_failure(status, device_ordinal, address, 0U);
        return;
    }
    status = shadowspill_record_stream(
        runtime,
        allocation.allocation_id,
        shadowspill_cuda_wrap_stream((uintptr_t)stream)
    );
    if (status != SHADOWSPILL_RUNTIME_OK) {
        latch_failure(status, device_ordinal, address, 0U);
    }
}
