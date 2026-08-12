#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "internal.h"

#include <pthread.h>
#include <stdint.h>

int shadowspill_profiler_is_valid(const ShadowSpillProfiler *profiler) {
    if (profiler == NULL || profiler->abi_version == 0U) {
        return 1;
    }
    return profiler->abi_version == SHADOWSPILL_PROFILER_ABI_VERSION &&
        profiler->name_current_thread != NULL &&
        profiler->name_stream != NULL && profiler->range_begin != NULL &&
        profiler->range_end != NULL;
}

void shadowspill_profiler_name_current_thread(
    const ShadowSpillProfiler *profiler, const char *name
) {
#if defined(__linux__)
    if (name != NULL) {
        (void)pthread_setname_np(pthread_self(), "shadowspill.wkr");
    }
#endif
    if (profiler != NULL && profiler->abi_version != 0U &&
        profiler->name_current_thread != NULL) {
        profiler->name_current_thread(profiler->context, name);
    }
}

void shadowspill_profiler_name_stream(
    const ShadowSpillProfiler *profiler,
    ShadowSpillBackendStream stream,
    const char *name
) {
    if (profiler != NULL && profiler->abi_version != 0U &&
        profiler->name_stream != NULL) {
        profiler->name_stream(profiler->context, stream, name);
    }
}

ShadowSpillProfilerRange shadowspill_profiler_range_begin(
    const ShadowSpillProfiler *profiler, const char *name
) {
    return profiler != NULL && profiler->abi_version != 0U &&
        profiler->range_begin != NULL
        ? profiler->range_begin(profiler->context, name)
        : 0U;
}

void shadowspill_profiler_range_end(
    const ShadowSpillProfiler *profiler, ShadowSpillProfilerRange range
) {
    if (profiler != NULL && profiler->abi_version != 0U &&
        profiler->range_end != NULL) {
        profiler->range_end(profiler->context, range);
    }
}
