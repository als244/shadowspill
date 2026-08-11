#include <shadowspill/backend_cuda.h>

#include <cuda.h>
#include <nvtx3/nvToolsExt.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>

struct ShadowSpillCudaBackend {
    pthread_mutex_t mutex;
    CUdevice device;
    CUcontext context;
    CUcontext creator_previous_context;
    pthread_t creator_thread;
    ShadowSpillCudaBackendCapabilities capabilities;
    ShadowSpillCudaBackendStatistics statistics;
    CUresult last_error;
};

static _Thread_local ShadowSpillCudaBackend *attached_backend;

static int record_result(ShadowSpillCudaBackend *backend, CUresult result) {
    if (result == CUDA_SUCCESS) {
        return 0;
    }
    pthread_mutex_lock(&backend->mutex);
    if (backend->last_error == CUDA_SUCCESS) {
        backend->last_error = result;
    }
    pthread_mutex_unlock(&backend->mutex);
    return -1;
}

static int activate_context(ShadowSpillCudaBackend *backend) {
    if (attached_backend == backend) {
        return 0;
    }
    CUcontext current = NULL;
    CUresult result = cuCtxGetCurrent(&current);
    if (result != CUDA_SUCCESS) {
        return record_result(backend, result);
    }
    if (current != backend->context) {
        result = cuCtxSetCurrent(backend->context);
        if (result != CUDA_SUCCESS) {
            return record_result(backend, result);
        }
        pthread_mutex_lock(&backend->mutex);
        ++backend->statistics.context_activations;
        pthread_mutex_unlock(&backend->mutex);
    }
    attached_backend = backend;
    return 0;
}

static CUstream stream_value(ShadowSpillBackendStream stream) {
    return (CUstream)stream.words[0];
}

static CUevent event_value(ShadowSpillBackendEvent event) {
    return (CUevent)event.words[0];
}

static int allocate_device(void *context, uint64_t bytes, void **pointer) {
    ShadowSpillCudaBackend *backend = context;
    if (pointer == NULL || activate_context(backend) != 0) {
        return -1;
    }
    CUdeviceptr allocation = 0U;
    CUresult result = cuMemAlloc(&allocation, (size_t)bytes);
    if (record_result(backend, result) != 0) {
        return -1;
    }
    *pointer = (void *)(uintptr_t)allocation;
    pthread_mutex_lock(&backend->mutex);
    ++backend->statistics.device_allocations;
    pthread_mutex_unlock(&backend->mutex);
    return 0;
}

static int free_device(void *context, void *pointer) {
    ShadowSpillCudaBackend *backend = context;
    if (activate_context(backend) != 0) {
        return -1;
    }
    CUresult result = cuMemFree((CUdeviceptr)(uintptr_t)pointer);
    if (record_result(backend, result) != 0) {
        return -1;
    }
    pthread_mutex_lock(&backend->mutex);
    ++backend->statistics.device_frees;
    pthread_mutex_unlock(&backend->mutex);
    return 0;
}

static int allocate_host(void *context, uint64_t bytes, void **pointer) {
    ShadowSpillCudaBackend *backend = context;
    if (pointer == NULL || activate_context(backend) != 0) {
        return -1;
    }
    CUresult result = cuMemHostAlloc(
        pointer, (size_t)bytes, CU_MEMHOSTALLOC_PORTABLE
    );
    if (record_result(backend, result) != 0) {
        return -1;
    }
    pthread_mutex_lock(&backend->mutex);
    ++backend->statistics.pinned_host_allocations;
    pthread_mutex_unlock(&backend->mutex);
    return 0;
}

static int free_host(void *context, void *pointer) {
    ShadowSpillCudaBackend *backend = context;
    if (activate_context(backend) != 0) {
        return -1;
    }
    CUresult result = cuMemFreeHost(pointer);
    if (record_result(backend, result) != 0) {
        return -1;
    }
    pthread_mutex_lock(&backend->mutex);
    ++backend->statistics.pinned_host_frees;
    pthread_mutex_unlock(&backend->mutex);
    return 0;
}

static int create_stream(
    void *context,
    ShadowSpillTransferKind kind,
    ShadowSpillBackendStream *stream
) {
    (void)kind;
    ShadowSpillCudaBackend *backend = context;
    if (stream == NULL || activate_context(backend) != 0) {
        return -1;
    }
    CUstream created = NULL;
    CUresult result = cuStreamCreate(&created, CU_STREAM_NON_BLOCKING);
    if (record_result(backend, result) != 0) {
        return -1;
    }
    *stream = (ShadowSpillBackendStream){
        .words = {(uintptr_t)created, 0U},
    };
    pthread_mutex_lock(&backend->mutex);
    ++backend->statistics.streams_created;
    pthread_mutex_unlock(&backend->mutex);
    return 0;
}

static int destroy_stream(void *context, ShadowSpillBackendStream stream) {
    ShadowSpillCudaBackend *backend = context;
    if (activate_context(backend) != 0) {
        return -1;
    }
    CUresult result = cuStreamDestroy(stream_value(stream));
    if (record_result(backend, result) != 0) {
        return -1;
    }
    pthread_mutex_lock(&backend->mutex);
    ++backend->statistics.streams_destroyed;
    pthread_mutex_unlock(&backend->mutex);
    return 0;
}

static int create_event(void *context, ShadowSpillBackendEvent *event) {
    ShadowSpillCudaBackend *backend = context;
    if (event == NULL || activate_context(backend) != 0) {
        return -1;
    }
    CUevent created = NULL;
    CUresult result = cuEventCreate(&created, CU_EVENT_DISABLE_TIMING);
    if (record_result(backend, result) != 0) {
        return -1;
    }
    *event = (ShadowSpillBackendEvent){
        .words = {(uintptr_t)created, 0U},
    };
    pthread_mutex_lock(&backend->mutex);
    ++backend->statistics.events_created;
    pthread_mutex_unlock(&backend->mutex);
    return 0;
}

static int destroy_event(void *context, ShadowSpillBackendEvent event) {
    ShadowSpillCudaBackend *backend = context;
    if (activate_context(backend) != 0) {
        return -1;
    }
    CUresult result = cuEventDestroy(event_value(event));
    if (record_result(backend, result) != 0) {
        return -1;
    }
    pthread_mutex_lock(&backend->mutex);
    ++backend->statistics.events_destroyed;
    pthread_mutex_unlock(&backend->mutex);
    return 0;
}

static int record_event(
    void *context,
    ShadowSpillBackendEvent event,
    ShadowSpillBackendStream stream
) {
    ShadowSpillCudaBackend *backend = context;
    if (activate_context(backend) != 0) {
        return -1;
    }
    return record_result(
        backend, cuEventRecord(event_value(event), stream_value(stream))
    );
}

static int query_event(
    void *context,
    ShadowSpillBackendEvent event,
    int *complete
) {
    ShadowSpillCudaBackend *backend = context;
    if (complete == NULL || activate_context(backend) != 0) {
        return -1;
    }
    CUresult result = cuEventQuery(event_value(event));
    pthread_mutex_lock(&backend->mutex);
    ++backend->statistics.event_queries;
    pthread_mutex_unlock(&backend->mutex);
    if (result == CUDA_SUCCESS) {
        *complete = 1;
        return 0;
    }
    if (result == CUDA_ERROR_NOT_READY) {
        *complete = 0;
        return 0;
    }
    return record_result(backend, result);
}

static int wait_event(
    void *context,
    ShadowSpillBackendStream stream,
    ShadowSpillBackendEvent event
) {
    (void)nvtxRangePushA("shadowspill.runtime.wait_event");
    ShadowSpillCudaBackend *backend = context;
    if (activate_context(backend) != 0 || record_result(
            backend,
            cuStreamWaitEvent(stream_value(stream), event_value(event), 0U)
        ) != 0) {
        (void)nvtxRangePop();
        return -1;
    }
    pthread_mutex_lock(&backend->mutex);
    ++backend->statistics.stream_waits;
    pthread_mutex_unlock(&backend->mutex);
    (void)nvtxRangePop();
    return 0;
}

static int copy_async(
    void *context,
    void *destination,
    const void *source,
    uint64_t bytes,
    ShadowSpillTransferKind kind,
    ShadowSpillBackendStream stream
) {
    const char *range_name = kind == SHADOWSPILL_TRANSFER_TO_DEVICE
        ? "shadowspill.runtime.transfer.h2d"
        : "shadowspill.runtime.transfer.d2h";
    (void)nvtxRangePushA(range_name);
    ShadowSpillCudaBackend *backend = context;
    if ((bytes != 0U && (destination == NULL || source == NULL)) ||
        bytes > SIZE_MAX || activate_context(backend) != 0) {
        (void)nvtxRangePop();
        return -1;
    }
    CUresult result;
    if (kind == SHADOWSPILL_TRANSFER_TO_DEVICE) {
        result = cuMemcpyHtoDAsync(
            (CUdeviceptr)(uintptr_t)destination,
            source,
            (size_t)bytes,
            stream_value(stream)
        );
    } else {
        result = cuMemcpyDtoHAsync(
            destination,
            (CUdeviceptr)(uintptr_t)source,
            (size_t)bytes,
            stream_value(stream)
        );
    }
    if (record_result(backend, result) != 0) {
        (void)nvtxRangePop();
        return -1;
    }
    pthread_mutex_lock(&backend->mutex);
    if (kind == SHADOWSPILL_TRANSFER_TO_DEVICE) {
        ++backend->statistics.copies_to_device;
        backend->statistics.bytes_to_device += bytes;
    } else {
        ++backend->statistics.copies_to_host;
        backend->statistics.bytes_to_host += bytes;
    }
    pthread_mutex_unlock(&backend->mutex);
    (void)nvtxRangePop();
    return 0;
}

static int synchronize_stream(
    void *context,
    ShadowSpillBackendStream stream
) {
    ShadowSpillCudaBackend *backend = context;
    if (activate_context(backend) != 0 || record_result(
            backend, cuStreamSynchronize(stream_value(stream))
        ) != 0) {
        return -1;
    }
    pthread_mutex_lock(&backend->mutex);
    ++backend->statistics.stream_synchronizations;
    pthread_mutex_unlock(&backend->mutex);
    return 0;
}

static int read_attribute(CUdevice device, CUdevice_attribute attribute) {
    int value = 0;
    if (cuDeviceGetAttribute(&value, attribute, device) != CUDA_SUCCESS) {
        return 0;
    }
    return value;
}

int shadowspill_cuda_backend_create(
    const ShadowSpillCudaBackendConfig *config,
    ShadowSpillCudaBackend **output
) {
    if (config == NULL || output == NULL ||
        config->abi_version != SHADOWSPILL_CUDA_BACKEND_ABI_VERSION ||
        config->device_ordinal < 0) {
        return -1;
    }
    *output = NULL;
    if (cuInit(0U) != CUDA_SUCCESS) {
        return -1;
    }
    ShadowSpillCudaBackend *backend = calloc(1U, sizeof(*backend));
    if (backend == NULL) {
        return -1;
    }
    if (pthread_mutex_init(&backend->mutex, NULL) != 0) {
        free(backend);
        return -1;
    }
    backend->creator_thread = pthread_self();
    CUresult result = cuDeviceGet(&backend->device, config->device_ordinal);
    if (result != CUDA_SUCCESS) {
        goto fail;
    }
    result = cuCtxGetCurrent(&backend->creator_previous_context);
    if (result != CUDA_SUCCESS) {
        goto fail;
    }
    result = cuDevicePrimaryCtxRetain(&backend->context, backend->device);
    if (result != CUDA_SUCCESS) {
        goto fail;
    }
    if (backend->creator_previous_context != NULL &&
        backend->creator_previous_context != backend->context) {
        result = CUDA_ERROR_INVALID_CONTEXT;
        (void)cuDevicePrimaryCtxRelease(backend->device);
        backend->context = NULL;
        goto fail;
    }
    result = cuCtxSetCurrent(backend->context);
    if (result != CUDA_SUCCESS) {
        (void)cuDevicePrimaryCtxRelease(backend->device);
        backend->context = NULL;
        goto fail;
    }
    attached_backend = backend;
    size_t total_bytes = 0U;
    int driver_version = 0;
    if (cuDeviceTotalMem(&total_bytes, backend->device) != CUDA_SUCCESS ||
        cuDriverGetVersion(&driver_version) != CUDA_SUCCESS) {
        result = CUDA_ERROR_UNKNOWN;
        goto release_context;
    }
    backend->capabilities = (ShadowSpillCudaBackendCapabilities){
        .abi_version = SHADOWSPILL_CUDA_BACKEND_ABI_VERSION,
        .driver_version = (uint32_t)driver_version,
        .device_ordinal = config->device_ordinal,
        .compute_major = read_attribute(
            backend->device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
        ),
        .compute_minor = read_attribute(
            backend->device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
        ),
        .asynchronous_engine_count = read_attribute(
            backend->device, CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT
        ),
        .total_device_bytes = (uint64_t)total_bytes,
        .recommended_minimum_alignment = 256U,
        .concurrent_kernels = (uint8_t)read_attribute(
            backend->device, CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS
        ),
        .unified_addressing = (uint8_t)read_attribute(
            backend->device, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING
        ),
    };
    *output = backend;
    return 0;

release_context:
    (void)cuCtxSetCurrent(backend->creator_previous_context);
    attached_backend = NULL;
    (void)cuDevicePrimaryCtxRelease(backend->device);
    backend->context = NULL;
fail:
    backend->last_error = result;
    pthread_mutex_destroy(&backend->mutex);
    free(backend);
    return -1;
}

void shadowspill_cuda_backend_destroy(ShadowSpillCudaBackend *backend) {
    if (backend == NULL) {
        return;
    }
    if (pthread_equal(pthread_self(), backend->creator_thread) != 0) {
        (void)cuCtxSetCurrent(backend->creator_previous_context);
    }
    if (attached_backend == backend) {
        attached_backend = NULL;
    }
    if (backend->context != NULL) {
        (void)cuDevicePrimaryCtxRelease(backend->device);
    }
    pthread_mutex_destroy(&backend->mutex);
    free(backend);
}

ShadowSpillBackend shadowspill_cuda_backend_vtable(
    ShadowSpillCudaBackend *backend
) {
    return (ShadowSpillBackend){
        .abi_version = SHADOWSPILL_BACKEND_ABI_VERSION,
        .context = backend,
        .allocate_device = allocate_device,
        .free_device = free_device,
        .allocate_host = allocate_host,
        .free_host = free_host,
        .create_stream = create_stream,
        .destroy_stream = destroy_stream,
        .create_event = create_event,
        .destroy_event = destroy_event,
        .record_event = record_event,
        .query_event = query_event,
        .wait_event = wait_event,
        .copy_async = copy_async,
        .synchronize_stream = synchronize_stream,
    };
}

int shadowspill_cuda_backend_capabilities(
    ShadowSpillCudaBackend *backend,
    ShadowSpillCudaBackendCapabilities *capabilities
) {
    if (backend == NULL || capabilities == NULL) {
        return -1;
    }
    *capabilities = backend->capabilities;
    return 0;
}

void shadowspill_cuda_backend_statistics(
    ShadowSpillCudaBackend *backend,
    ShadowSpillCudaBackendStatistics *statistics
) {
    if (backend == NULL || statistics == NULL) {
        return;
    }
    pthread_mutex_lock(&backend->mutex);
    *statistics = backend->statistics;
    pthread_mutex_unlock(&backend->mutex);
}

uint32_t shadowspill_cuda_backend_last_error(ShadowSpillCudaBackend *backend) {
    if (backend == NULL) {
        return (uint32_t)CUDA_ERROR_INVALID_VALUE;
    }
    pthread_mutex_lock(&backend->mutex);
    uint32_t result = (uint32_t)backend->last_error;
    pthread_mutex_unlock(&backend->mutex);
    return result;
}

const char *shadowspill_cuda_error_name(uint32_t error_code) {
    const char *name = "CUDA_ERROR_UNKNOWN";
    (void)cuGetErrorName((CUresult)error_code, &name);
    return name;
}

const char *shadowspill_cuda_error_string(uint32_t error_code) {
    const char *description = "unknown CUDA driver error";
    (void)cuGetErrorString((CUresult)error_code, &description);
    return description;
}

ShadowSpillBackendStream shadowspill_cuda_wrap_stream(
    uintptr_t stream_address
) {
    return (ShadowSpillBackendStream){
        .words = {stream_address, 0U},
    };
}
