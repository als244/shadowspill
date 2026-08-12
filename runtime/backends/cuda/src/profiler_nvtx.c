#define _GNU_SOURCE

#include <shadowspill/backend_cuda.h>

#include <cuda.h>
#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtCuda.h>
#include <pthread.h>
#include <stdint.h>
#include <sys/syscall.h>
#include <unistd.h>

static void name_current_thread(void *context, const char *name) {
    (void)context;
    if (name == NULL) {
        return;
    }
    nvtxNameOsThreadA((uint32_t)syscall(SYS_gettid), name);
}

static void name_stream(
    void *context,
    ShadowSpillBackendStream stream,
    const char *name
) {
    (void)context;
    if (name != NULL) {
        nvtxNameCuStreamA((CUstream)stream.words[0], name);
    }
}

static ShadowSpillProfilerRange range_begin(
    void *context, const char *name
) {
    (void)context;
    return name == NULL ? 0U : (ShadowSpillProfilerRange)nvtxRangeStartA(name);
}

static void range_end(void *context, ShadowSpillProfilerRange range) {
    (void)context;
    if (range != 0U) {
        nvtxRangeEnd((nvtxRangeId_t)range);
    }
}

ShadowSpillProfiler shadowspill_cuda_backend_profiler(
    ShadowSpillCudaBackend *backend
) {
    return (ShadowSpillProfiler){
        .abi_version = SHADOWSPILL_PROFILER_ABI_VERSION,
        .context = backend,
        .name_current_thread = name_current_thread,
        .name_stream = name_stream,
        .range_begin = range_begin,
        .range_end = range_end,
    };
}
