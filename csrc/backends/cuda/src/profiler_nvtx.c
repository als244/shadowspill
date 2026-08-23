#define _GNU_SOURCE

#include <shadowspill/backend_cuda.h>

#include <cuda.h>
#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtCuda.h>
#include <pthread.h>
#include <stdint.h>
#include <sys/syscall.h>
#include <unistd.h>

static void name_current_thread(void *state, const char *name) {
    (void)state;
    if (name == NULL) {
        return;
    }
    nvtxNameOsThreadA((uint32_t)syscall(SYS_gettid), name);
}

static void name_stream(
    void *state,
    ShadowSpillBackendStream stream,
    const char *name
) {
    (void)state;
    if (name != NULL) {
        nvtxNameCuStreamA((CUstream)stream.words[0], name);
    }
}

static void set_enabled(void *state, uint8_t enabled) {
    shadowspill_cuda_backend_profiler_enable(state, enabled);
}

static ShadowSpillProfilerRange range_begin(
    void *state, const char *name
) {
    return name == NULL ||
            !shadowspill_cuda_backend_profiler_is_enabled(state)
        ? 0U
        : (ShadowSpillProfilerRange)nvtxRangeStartA(name);
}

static void range_end(void *state, ShadowSpillProfilerRange range) {
    (void)state;
    if (range != 0U) {
        nvtxRangeEnd((nvtxRangeId_t)range);
    }
}

ShadowSpillProfiler shadowspill_cuda_backend_profiler(
    ShadowSpillCudaBackend *backend
) {
    return (ShadowSpillProfiler){
        .abi_version = SHADOWSPILL_PROFILER_ABI_VERSION,
        .state = backend,
        .name_current_thread = name_current_thread,
        .name_stream = name_stream,
        .set_enabled = set_enabled,
        .range_begin = range_begin,
        .range_end = range_end,
    };
}
