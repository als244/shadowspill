#define _GNU_SOURCE
#include "backend_cuda_internal.h"

#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtCuda.h>
#include <stdint.h>
#include <sys/syscall.h>
#include <unistd.h>

void shadowspill_cuda_name_thread(void *state, const char *name) {
    (void)state;
    if (name != NULL) {
        nvtxNameOsThreadA((uint32_t)syscall(SYS_gettid), name);
    }
}

void shadowspill_cuda_name_stream(
    void *state,
    ShadowSpillBackendStream stream,
    const char *name
) {
    (void)state;
    if (name != NULL) {
        nvtxNameCuStreamA((CUstream)stream.words[0], name);
    }
}

void shadowspill_cuda_profiler_enable(void *state, uint8_t enabled) {
    ShadowSpillCudaBackend *backend = state;
    if (backend != NULL) {
        atomic_store_explicit(
            &backend->profiler_enabled, enabled != 0U, memory_order_release
        );
    }
}

ShadowSpillProfilerRange shadowspill_cuda_range_begin(void *state, const char *name) {
    const ShadowSpillCudaBackend *backend = state;
    return name == NULL || backend == NULL ||
            !atomic_load_explicit(&backend->profiler_enabled, memory_order_acquire)
        ? 0U
        : (ShadowSpillProfilerRange)nvtxRangeStartA(name);
}

void shadowspill_cuda_range_end(void *state, ShadowSpillProfilerRange range) {
    (void)state;
    if (range != 0U) {
        nvtxRangeEnd((nvtxRangeId_t)range);
    }
}
