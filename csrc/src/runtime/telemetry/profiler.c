#define _GNU_SOURCE

#include "internal.h"
#include "../../common/platform.h"

#include <pthread.h>
#include <stdint.h>

int shadowspill_profiler_is_valid(const ShadowSpillProfiler *profiler) {
    if (profiler == NULL || profiler->abi_version == 0U) {
        return 1;
    }
    return profiler->abi_version == SHADOWSPILL_PROFILER_ABI_VERSION &&
        profiler->name_current_thread != NULL &&
        profiler->name_stream != NULL && profiler->set_enabled != NULL &&
        profiler->range_begin != NULL && profiler->range_end != NULL;
}

void shadowspill_profiler_set_enabled(
    const ShadowSpillProfiler *profiler, uint8_t enabled
) {
    if (profiler != NULL && profiler->abi_version != 0U &&
        profiler->set_enabled != NULL) {
        profiler->set_enabled(profiler->state, enabled != 0U);
    }
}

void shadowspill_profiler_name_current_thread(
    const ShadowSpillProfiler *profiler, const char *name
) {
    shadowspill_name_current_thread(name);
    if (profiler != NULL && profiler->abi_version != 0U &&
        profiler->name_current_thread != NULL) {
        profiler->name_current_thread(profiler->state, name);
    }
}

void shadowspill_profiler_name_stream(
    const ShadowSpillProfiler *profiler,
    ShadowSpillBackendStream stream,
    const char *name
) {
    if (profiler != NULL && profiler->abi_version != 0U &&
        profiler->name_stream != NULL) {
        profiler->name_stream(profiler->state, stream, name);
    }
}

ShadowSpillProfilerRange shadowspill_profiler_range_begin(
    const ShadowSpillProfiler *profiler, const char *name
) {
    return profiler != NULL && profiler->abi_version != 0U &&
        profiler->range_begin != NULL
        ? profiler->range_begin(profiler->state, name)
        : 0U;
}

void shadowspill_profiler_range_end(
    const ShadowSpillProfiler *profiler, ShadowSpillProfilerRange range
) {
    if (profiler != NULL && profiler->abi_version != 0U &&
        profiler->range_end != NULL) {
        profiler->range_end(profiler->state, range);
    }
}
