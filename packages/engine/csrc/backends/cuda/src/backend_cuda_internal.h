/* Private to the CUDA backend: the provider object and the profiler entries
 * profiler_nvtx.c contributes to the table. */
#ifndef SHADOWSPILL_BACKEND_CUDA_INTERNAL_H
#define SHADOWSPILL_BACKEND_CUDA_INTERNAL_H

#include <shadowspill/backend.h>

#include <cuda.h>
#include <nvml.h>
#include <pthread.h>
#include <stdatomic.h>

#if defined(_WIN32)
#define SHADOWSPILL_BACKEND_CUDA_API __declspec(dllexport)
#else
#define SHADOWSPILL_BACKEND_CUDA_API __attribute__((visibility("default")))
#endif

typedef struct ShadowSpillCudaBackend {
    pthread_mutex_t mutex;
    CUdevice device;
    CUcontext context;
    CUcontext creator_previous_context;
    nvmlDevice_t nvml_device;
    uint8_t nvml_initialized;
    pthread_t creator_thread;
    int32_t device_ordinal;
    ShadowSpillBackendStatistics statistics;
    _Atomic uint8_t profiler_enabled;
    CUresult last_error;
    nvmlReturn_t last_nvml_error;
} ShadowSpillCudaBackend;

void shadowspill_cuda_name_thread(void *state, const char *name);
void shadowspill_cuda_name_stream(
    void *state,
    ShadowSpillBackendStream stream,
    const char *name
);
void shadowspill_cuda_profiler_enable(void *state, uint8_t enabled);
ShadowSpillProfilerRange shadowspill_cuda_range_begin(void *state, const char *name);
void shadowspill_cuda_range_end(void *state, ShadowSpillProfilerRange range);

#endif
