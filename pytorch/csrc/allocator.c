#include <shadowspill/pytorch_adapter.h>

#include <pthread.h>
#include <stdint.h>
#include <string.h>

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
    ShadowSpillPytorchAdapterFailure failure;
} ShadowSpillPytorchAdapterState;

static ShadowSpillPytorchAdapterState adapter = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .device_ordinal = -1,
};

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
        config->device_ordinal < 0 || config->device_slab_bytes == 0U) {
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
    const ShadowSpillRuntimeConfig runtime_config = {
        .abi_version = SHADOWSPILL_RUNTIME_ABI_VERSION,
        .device_slab_bytes = config->device_slab_bytes,
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
    memset(&adapter.failure, 0, sizeof(adapter.failure));
    adapter.failure.device_ordinal = config->device_ordinal;
    adapter.failure.runtime.object_id = SHADOWSPILL_RUNTIME_NO_ID;
    adapter.failure.runtime.allocation_id = SHADOWSPILL_RUNTIME_NO_ID;
    pthread_mutex_unlock(&adapter.mutex);
    return SHADOWSPILL_RUNTIME_OK;
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
        .storage_rebinding = 0U,
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

void *shadowspill_pytorch_cuda_malloc(
    ptrdiff_t bytes,
    int32_t device_ordinal,
    void *stream
) {
    pthread_mutex_lock(&adapter.mutex);
    ++adapter.allocation_callbacks;
    if (bytes == 0) {
        ++adapter.zero_size_allocation_callbacks;
    }
    pthread_mutex_unlock(&adapter.mutex);
    int32_t expected_device;
    ShadowSpillRuntime *runtime = bound_runtime(&expected_device);
    if (bytes == 0 && runtime != NULL && device_ordinal == expected_device) {
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
        return NULL;
    }
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
