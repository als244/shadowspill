#include <shadowspill/pytorch_adapter.h>

#include <pthread.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include <nvtx3/nvToolsExt.h>

typedef struct ShadowSpillPytorchAdapterState {
    pthread_mutex_t mutex;
    ShadowSpillCudaBackend *cuda;
    ShadowSpillRuntime *runtime;
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
    ShadowSpillPytorchPhysicalAdmission admission;
    ShadowSpillPytorchAdapterFailure failure;
} ShadowSpillPytorchAdapterState;

static ShadowSpillPytorchAdapterState adapter = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .device_ordinal = -1,
};

static _Thread_local int task_range_active;

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
    uint64_t slab_bytes = available - available % physical_granularity;
    if (slab_bytes == 0U) {
        shadowspill_cuda_backend_destroy(cuda);
        return SHADOWSPILL_RUNTIME_OUT_OF_MEMORY;
    }
    const ShadowSpillRuntimeConfig runtime_config = {
        .abi_version = SHADOWSPILL_RUNTIME_ABI_VERSION,
        .device_slab_bytes = slab_bytes,
        .host_arena_bytes = config->host_arena_bytes,
        .minimum_alignment = capabilities.recommended_minimum_alignment,
        .progress_poll_nanoseconds = config->progress_poll_nanoseconds,
        .backend = shadowspill_cuda_backend_vtable(cuda),
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
    adapter.runtime = runtime;
    adapter.device_ordinal = config->device_ordinal;
    adapter.admission = (ShadowSpillPytorchPhysicalAdmission){
        .abi_version = SHADOWSPILL_PYTORCH_ADAPTER_ABI_VERSION,
        .device_ordinal = config->device_ordinal,
        .device_budget_bytes = config->device_budget_bytes,
        .context_bytes = context_memory.process_bytes,
        .provider_headroom_bytes = config->provider_headroom_bytes,
        .slab_bytes = slab_bytes,
        .bootstrap_process_bytes = bootstrap_memory.process_bytes,
        .device_used_bytes = bootstrap_memory.device_used_bytes,
        .device_total_bytes = bootstrap_memory.device_total_bytes,
        .host_arena_bytes = config->host_arena_bytes,
    };
    adapter.physical_checks = 1U;
    adapter.peak_process_physical_bytes = bootstrap_memory.process_bytes;
    adapter.observed_external_high_water_bytes =
        bootstrap_memory.process_bytes >
                context_memory.process_bytes + slab_bytes
        ? bootstrap_memory.process_bytes - context_memory.process_bytes -
            slab_bytes
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
    uint64_t base = admission.context_bytes + admission.slab_bytes;
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
    uint64_t required_provider_headroom_bytes
) {
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
    pthread_mutex_unlock(&adapter.mutex);
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
    uint8_t retain_host_backing,
    uint64_t source_address
) {
    if (retain_host_backing > 1U ||
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
        .retain_host_backing = retain_host_backing,
        .initially_host_resident = 1U,
    };
    ShadowSpillRuntimeStatus status = shadowspill_register_object(
        runtime, &description
    );
    if (status != SHADOWSPILL_RUNTIME_OK) {
        return status;
    }
    return shadowspill_write_host_object(
        runtime,
        object_id,
        (const void *)(uintptr_t)source_address,
        size_bytes
    );
}

ShadowSpillRuntimeStatus shadowspill_pytorch_write_host_object(
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
        : shadowspill_write_host_object(
              runtime,
              object_id,
              (const void *)(uintptr_t)source_address,
              size_bytes
          );
}

ShadowSpillRuntimeStatus shadowspill_pytorch_read_host_object(
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
        : shadowspill_read_host_object(
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
         snapshot.device_pointer == (void *)(uintptr_t)address);
    const int retired_matches = address != 0U &&
        snapshot.retired_generation == generation &&
        snapshot.retired_device_pointer == (void *)(uintptr_t)address;
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
    if (task_range_active) {
        return SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    char range_name[96];
    (void)snprintf(
        range_name,
        sizeof(range_name),
        "shadowspill.pytorch.task.%llu",
        (unsigned long long)task_id
    );
    (void)nvtxRangePushA(range_name);
    task_range_active = 1;
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    if (runtime == NULL) {
        (void)nvtxRangePop();
        task_range_active = 0;
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
        (void)nvtxRangePop();
        task_range_active = 0;
    }
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
    if (!task_range_active) {
        char range_name[96];
        (void)snprintf(
            range_name,
            sizeof(range_name),
            "shadowspill.pytorch.after_task.%llu",
            (unsigned long long)task_id
        );
        (void)nvtxRangePushA(range_name);
    }
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    if (runtime == NULL) {
        (void)nvtxRangePop();
        task_range_active = 0;
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
    (void)nvtxRangePop();
    task_range_active = 0;
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

void shadowspill_pytorch_abort_task_range(void) {
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    shadowspill_abort_task(runtime);
    if (task_range_active) {
        (void)nvtxRangePop();
        task_range_active = 0;
    }
}

void *shadowspill_pytorch_cuda_malloc(
    ptrdiff_t bytes,
    int32_t device_ordinal,
    void *stream
) {
    (void)nvtxRangePushA("shadowspill.runtime.allocate");
    pthread_mutex_lock(&adapter.mutex);
    ++adapter.allocation_callbacks;
    if (bytes == 0) {
        ++adapter.zero_size_allocation_callbacks;
    }
    pthread_mutex_unlock(&adapter.mutex);
    int32_t expected_device;
    ShadowSpillRuntime *runtime = bound_runtime(&expected_device);
    if (bytes == 0 && runtime != NULL && device_ordinal == expected_device) {
        (void)nvtxRangePop();
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
        (void)nvtxRangePop();
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
        (void)nvtxRangePop();
        return NULL;
    }
    (void)nvtxRangePop();
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
