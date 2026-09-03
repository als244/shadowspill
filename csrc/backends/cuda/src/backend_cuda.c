/* The CUDA backend: the driver-level table over the CUDA driver API and NVML. */
#include "backend_cuda_internal.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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

static int record_nvml_result(ShadowSpillCudaBackend *backend, nvmlReturn_t result) {
    if (result == NVML_SUCCESS) {
        return 0;
    }
    pthread_mutex_lock(&backend->mutex);
    if (backend->last_nvml_error == NVML_SUCCESS) {
        backend->last_nvml_error = result;
    }
    pthread_mutex_unlock(&backend->mutex);
    return -1;
}

/* The provider's context must be current on the calling thread; the runtime
 * worker and the framework's threads all come through here. */
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
        ++backend->statistics.provider_activations;
        pthread_mutex_unlock(&backend->mutex);
    }
    attached_backend = backend;
    return 0;
}

static void count(ShadowSpillCudaBackend *backend, uint64_t *counter, uint64_t by) {
    pthread_mutex_lock(&backend->mutex);
    *counter += by;
    pthread_mutex_unlock(&backend->mutex);
}

static CUstream stream_value(ShadowSpillBackendStream stream) {
    return (CUstream)stream.words[0];
}

static CUevent event_value(ShadowSpillBackendEvent event) {
    return (CUevent)event.words[0];
}

/* ---------------------------------------------------------------- memory */

static int allocate_device(void *state, uint64_t bytes, void **address) {
    ShadowSpillCudaBackend *backend = state;
    if (address == NULL || bytes > SIZE_MAX || activate_context(backend) != 0) {
        return -1;
    }
    CUdeviceptr allocation = 0U;
    if (record_result(backend, cuMemAlloc(&allocation, (size_t)bytes)) != 0) {
        return -1;
    }
    *address = (void *)(uintptr_t)allocation;
    pthread_mutex_lock(&backend->mutex);
    ++backend->statistics.device_allocations;
    backend->statistics.bytes_device_allocated += bytes;
    pthread_mutex_unlock(&backend->mutex);
    return 0;
}

static int free_device(void *state, void *address, uint64_t bytes) {
    ShadowSpillCudaBackend *backend = state;
    if (activate_context(backend) != 0 ||
        record_result(backend, cuMemFree((CUdeviceptr)(uintptr_t)address)) != 0) {
        return -1;
    }
    pthread_mutex_lock(&backend->mutex);
    ++backend->statistics.device_frees;
    backend->statistics.bytes_device_freed += bytes;
    pthread_mutex_unlock(&backend->mutex);
    return 0;
}

static int register_host_memory(void *state, void *address, uint64_t bytes) {
    ShadowSpillCudaBackend *backend = state;
    if (address == NULL || bytes == 0U || bytes > SIZE_MAX ||
        activate_context(backend) != 0 ||
        record_result(
            backend,
            cuMemHostRegister(address, (size_t)bytes, CU_MEMHOSTREGISTER_PORTABLE)
        ) != 0) {
        return -1;
    }
    pthread_mutex_lock(&backend->mutex);
    ++backend->statistics.pinned_host_registrations;
    backend->statistics.bytes_pinned_host_registered += bytes;
    pthread_mutex_unlock(&backend->mutex);
    return 0;
}

static int unregister_host_memory(void *state, void *address, uint64_t bytes) {
    ShadowSpillCudaBackend *backend = state;
    if (address == NULL || activate_context(backend) != 0 ||
        record_result(backend, cuMemHostUnregister(address)) != 0) {
        return -1;
    }
    pthread_mutex_lock(&backend->mutex);
    ++backend->statistics.pinned_host_unregistrations;
    backend->statistics.bytes_pinned_host_unregistered += bytes;
    pthread_mutex_unlock(&backend->mutex);
    return 0;
}

/* --------------------------------------------------------------- streams */

static int create_stream(void *state, ShadowSpillBackendStream *stream) {
    ShadowSpillCudaBackend *backend = state;
    if (stream == NULL || activate_context(backend) != 0) {
        return -1;
    }
    CUstream created = NULL;
    if (record_result(backend, cuStreamCreate(&created, CU_STREAM_NON_BLOCKING)) != 0) {
        return -1;
    }
    *stream = (ShadowSpillBackendStream){.words = {(uintptr_t)created, 0U}};
    count(backend, &backend->statistics.streams_created, 1U);
    return 0;
}

static int destroy_stream(void *state, ShadowSpillBackendStream stream) {
    ShadowSpillCudaBackend *backend = state;
    if (activate_context(backend) != 0 ||
        record_result(backend, cuStreamDestroy(stream_value(stream))) != 0) {
        return -1;
    }
    count(backend, &backend->statistics.streams_destroyed, 1U);
    return 0;
}

static int synchronize_stream(void *state, ShadowSpillBackendStream stream) {
    ShadowSpillCudaBackend *backend = state;
    if (activate_context(backend) != 0 ||
        record_result(backend, cuStreamSynchronize(stream_value(stream))) != 0) {
        return -1;
    }
    count(backend, &backend->statistics.stream_synchronizations, 1U);
    return 0;
}

static ShadowSpillBackendStream wrap_stream(
    void *state, uint64_t framework_stream_handle
) {
    (void)state;
    return (ShadowSpillBackendStream){
        .words = {(uintptr_t)framework_stream_handle, 0U},
    };
}

/* ---------------------------------------------------------------- copies */

static int copy_host_to_device(
    void *state, void *device, const void *host, uint64_t bytes,
    ShadowSpillBackendStream stream
) {
    ShadowSpillCudaBackend *backend = state;
    if ((bytes != 0U && (device == NULL || host == NULL)) || bytes > SIZE_MAX ||
        activate_context(backend) != 0 ||
        record_result(
            backend,
            cuMemcpyHtoDAsync(
                (CUdeviceptr)(uintptr_t)device, host, (size_t)bytes,
                stream_value(stream)
            )
        ) != 0) {
        return -1;
    }
    pthread_mutex_lock(&backend->mutex);
    ++backend->statistics.copies_host_to_device;
    backend->statistics.bytes_host_to_device += bytes;
    pthread_mutex_unlock(&backend->mutex);
    return 0;
}

static int copy_device_to_host(
    void *state, void *host, const void *device, uint64_t bytes,
    ShadowSpillBackendStream stream
) {
    ShadowSpillCudaBackend *backend = state;
    if ((bytes != 0U && (device == NULL || host == NULL)) || bytes > SIZE_MAX ||
        activate_context(backend) != 0 ||
        record_result(
            backend,
            cuMemcpyDtoHAsync(
                host, (CUdeviceptr)(uintptr_t)device, (size_t)bytes,
                stream_value(stream)
            )
        ) != 0) {
        return -1;
    }
    pthread_mutex_lock(&backend->mutex);
    ++backend->statistics.copies_device_to_host;
    backend->statistics.bytes_device_to_host += bytes;
    pthread_mutex_unlock(&backend->mutex);
    return 0;
}

static int copy_device_to_device(
    void *state, void *destination, const void *source, uint64_t bytes,
    ShadowSpillBackendStream stream
) {
    ShadowSpillCudaBackend *backend = state;
    if ((bytes != 0U && (destination == NULL || source == NULL)) ||
        bytes > SIZE_MAX || activate_context(backend) != 0 ||
        record_result(
            backend,
            cuMemcpyDtoDAsync(
                (CUdeviceptr)(uintptr_t)destination,
                (CUdeviceptr)(uintptr_t)source, (size_t)bytes,
                stream_value(stream)
            )
        ) != 0) {
        return -1;
    }
    pthread_mutex_lock(&backend->mutex);
    ++backend->statistics.copies_device_to_device;
    backend->statistics.bytes_device_to_device += bytes;
    pthread_mutex_unlock(&backend->mutex);
    return 0;
}

/* ---------------------------------------------------------------- events */

static int create_event(void *state, ShadowSpillBackendEvent *event, uint8_t timing) {
    ShadowSpillCudaBackend *backend = state;
    if (event == NULL || activate_context(backend) != 0) {
        return -1;
    }
    CUevent created = NULL;
    if (record_result(
            backend,
            cuEventCreate(&created, timing ? CU_EVENT_DEFAULT : CU_EVENT_DISABLE_TIMING)
        ) != 0) {
        return -1;
    }
    *event = (ShadowSpillBackendEvent){.words = {(uintptr_t)created, 0U}};
    count(backend, &backend->statistics.events_created, 1U);
    return 0;
}

static int destroy_event(void *state, ShadowSpillBackendEvent event) {
    ShadowSpillCudaBackend *backend = state;
    if (activate_context(backend) != 0 ||
        record_result(backend, cuEventDestroy(event_value(event))) != 0) {
        return -1;
    }
    count(backend, &backend->statistics.events_destroyed, 1U);
    return 0;
}

static int record_event(
    void *state, ShadowSpillBackendEvent event, ShadowSpillBackendStream stream
) {
    ShadowSpillCudaBackend *backend = state;
    if (activate_context(backend) != 0) {
        return -1;
    }
    return record_result(
        backend, cuEventRecord(event_value(event), stream_value(stream))
    );
}

static int query_event(void *state, ShadowSpillBackendEvent event, int *complete) {
    ShadowSpillCudaBackend *backend = state;
    if (complete == NULL || activate_context(backend) != 0) {
        return -1;
    }
    const CUresult result = cuEventQuery(event_value(event));
    count(backend, &backend->statistics.event_queries, 1U);
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
    void *state, ShadowSpillBackendStream stream, ShadowSpillBackendEvent event
) {
    ShadowSpillCudaBackend *backend = state;
    const ShadowSpillProfilerRange range =
        shadowspill_cuda_range_begin(backend, "shadowspill.runtime.wait_event");
    if (activate_context(backend) != 0 ||
        record_result(
            backend, cuStreamWaitEvent(stream_value(stream), event_value(event), 0U)
        ) != 0) {
        shadowspill_cuda_range_end(backend, range);
        return -1;
    }
    count(backend, &backend->statistics.stream_waits, 1U);
    shadowspill_cuda_range_end(backend, range);
    return 0;
}

static int elapsed_nanoseconds(
    void *state, ShadowSpillBackendEvent from, ShadowSpillBackendEvent to,
    uint64_t *nanoseconds
) {
    ShadowSpillCudaBackend *backend = state;
    if (nanoseconds == NULL || activate_context(backend) != 0) {
        return -1;
    }
    float milliseconds = 0.0f;
    const CUresult result =
        cuEventElapsedTime(&milliseconds, event_value(from), event_value(to));
    if (result == CUDA_ERROR_NOT_READY) {
        return 1;
    }
    if (result != CUDA_SUCCESS) {
        return -1;
    }
    *nanoseconds = milliseconds <= 0.0f
        ? 0U
        : (uint64_t)((double)milliseconds * 1000000.0);
    return 0;
}

/* ----------------------------------------------------------------- facts */

static int capabilities(void *state, ShadowSpillBackendCapabilities *out) {
    ShadowSpillCudaBackend *backend = state;
    if (backend == NULL || out == NULL) {
        return -1;
    }
    *out = (ShadowSpillBackendCapabilities){
        .device_ordinal = backend->device_ordinal,
        .minimum_alignment = 256U,
        .provider = "cuda",
    };
    return 0;
}

static int physical_memory(void *state, ShadowSpillBackendPhysicalMemory *out) {
    ShadowSpillCudaBackend *backend = state;
    if (backend == NULL || out == NULL || !backend->nvml_initialized) {
        return -1;
    }
    nvmlMemory_t device_memory = {0};
    nvmlReturn_t result =
        nvmlDeviceGetMemoryInfo(backend->nvml_device, &device_memory);
    if (record_nvml_result(backend, result) != 0) {
        return -1;
    }
    unsigned int process_count = 0U;
    result = nvmlDeviceGetComputeRunningProcesses_v3(
        backend->nvml_device, &process_count, NULL
    );
    if (result != NVML_SUCCESS && result != NVML_ERROR_INSUFFICIENT_SIZE) {
        return record_nvml_result(backend, result);
    }
    nvmlProcessInfo_t *processes = NULL;
    if (process_count != 0U) {
        processes = calloc((size_t)process_count, sizeof(*processes));
        if (processes == NULL) {
            return -1;
        }
        result = nvmlDeviceGetComputeRunningProcesses_v3(
            backend->nvml_device, &process_count, processes
        );
        if (record_nvml_result(backend, result) != 0) {
            free(processes);
            return -1;
        }
    }
    uint64_t process_bytes = 0U;
    const unsigned int process_id = (unsigned int)getpid();
    for (unsigned int index = 0U; index < process_count; ++index) {
        if (processes[index].pid == process_id &&
            processes[index].usedGpuMemory !=
                (unsigned long long)NVML_VALUE_NOT_AVAILABLE) {
            process_bytes += (uint64_t)processes[index].usedGpuMemory;
        }
    }
    free(processes);
    *out = (ShadowSpillBackendPhysicalMemory){
        .process_bytes = process_bytes,
        .device_used_bytes = (uint64_t)device_memory.used,
        .device_total_bytes = (uint64_t)device_memory.total,
    };
    return 0;
}

static void statistics(void *state, ShadowSpillBackendStatistics *out) {
    ShadowSpillCudaBackend *backend = state;
    if (backend == NULL || out == NULL) {
        return;
    }
    pthread_mutex_lock(&backend->mutex);
    *out = backend->statistics;
    pthread_mutex_unlock(&backend->mutex);
}

/* -------------------------------------------------------------- lifetime */

SHADOWSPILL_BACKEND_CUDA_API int shadowspill_backend_create(
    const ShadowSpillBackendConfig *config,
    ShadowSpillBackend *table
) {
    if (config == NULL || table == NULL ||
        config->abi_version != SHADOWSPILL_BACKEND_ABI_VERSION ||
        config->device_ordinal < 0 || cuInit(0U) != CUDA_SUCCESS) {
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
    backend->device_ordinal = config->device_ordinal;
    CUresult result = CUDA_SUCCESS;
    nvmlReturn_t nvml_result = nvmlInit_v2();
    if (nvml_result != NVML_SUCCESS) {
        backend->last_nvml_error = nvml_result;
        goto fail;
    }
    backend->nvml_initialized = 1U;
    result = cuDeviceGet(&backend->device, config->device_ordinal);
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
    char pci_bus_id[32];
    result = cuDeviceGetPCIBusId(pci_bus_id, (int)sizeof(pci_bus_id), backend->device);
    if (result != CUDA_SUCCESS) {
        goto release_context;
    }
    nvml_result = nvmlDeviceGetHandleByPciBusId_v2(pci_bus_id, &backend->nvml_device);
    if (nvml_result != NVML_SUCCESS) {
        backend->last_nvml_error = nvml_result;
        goto release_context;
    }
    *table = (ShadowSpillBackend){
        .abi_version = SHADOWSPILL_BACKEND_ABI_VERSION,
        .state = backend,
        .allocate_device = allocate_device,
        .free_device = free_device,
        .register_host_memory = register_host_memory,
        .unregister_host_memory = unregister_host_memory,
        .create_stream = create_stream,
        .destroy_stream = destroy_stream,
        .synchronize_stream = synchronize_stream,
        .wrap_stream = wrap_stream,
        .copy_host_to_device = copy_host_to_device,
        .copy_device_to_host = copy_device_to_host,
        .copy_device_to_device = copy_device_to_device,
        .create_event = create_event,
        .destroy_event = destroy_event,
        .record_event = record_event,
        .query_event = query_event,
        .wait_event = wait_event,
        .elapsed_nanoseconds = elapsed_nanoseconds,
        .capabilities = capabilities,
        .physical_memory = physical_memory,
        .statistics = statistics,
        .name_thread = shadowspill_cuda_name_thread,
        .name_stream = shadowspill_cuda_name_stream,
        .profiler_enable = shadowspill_cuda_profiler_enable,
        .range_begin = shadowspill_cuda_range_begin,
        .range_end = shadowspill_cuda_range_end,
    };
    return 0;
release_context:
    (void)cuCtxSetCurrent(backend->creator_previous_context);
    attached_backend = NULL;
    (void)cuDevicePrimaryCtxRelease(backend->device);
    backend->context = NULL;
fail:
    backend->last_error = result;
    if (backend->nvml_initialized) {
        (void)nvmlShutdown();
    }
    pthread_mutex_destroy(&backend->mutex);
    free(backend);
    return -1;
}

SHADOWSPILL_BACKEND_CUDA_API void shadowspill_backend_destroy(
    ShadowSpillBackend *table
) {
    if (table == NULL || table->state == NULL) {
        return;
    }
    ShadowSpillCudaBackend *backend = table->state;
    if (pthread_equal(pthread_self(), backend->creator_thread) != 0) {
        (void)cuCtxSetCurrent(backend->creator_previous_context);
    }
    if (attached_backend == backend) {
        attached_backend = NULL;
    }
    if (backend->context != NULL) {
        (void)cuDevicePrimaryCtxRelease(backend->device);
    }
    if (backend->nvml_initialized) {
        (void)nvmlShutdown();
    }
    pthread_mutex_destroy(&backend->mutex);
    free(backend);
    memset(table, 0, sizeof(*table));
}
