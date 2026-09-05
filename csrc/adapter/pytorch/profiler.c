#include "internal.h"

#include <pthread.h>
#include <stdatomic.h>

ShadowSpillProfilerRange shadowspill_pytorch_profile_range_begin(
    const char *name
) {
    return atomic_load_explicit(
               &adapter.profiler_annotations_enabled, memory_order_relaxed
           ) == 0U || adapter.backend.range_begin == NULL
        ? 0U
        : adapter.backend.range_begin(adapter.backend.state, name);
}

void shadowspill_pytorch_profile_range_end(ShadowSpillProfilerRange range) {
    if (range != 0U && adapter.backend.range_end != NULL) {
        adapter.backend.range_end(adapter.backend.state, range);
    }
}

ShadowSpillStatus shadowspill_pytorch_profiler_annotations_set(
    uint8_t enabled
) {
    pthread_mutex_lock(&adapter.mutex);
    const int available = adapter.runtime != NULL &&
        adapter.backend.profiler_enable != NULL;
    const ShadowSpillBackend backend = adapter.backend;
    pthread_mutex_unlock(&adapter.mutex);
    if (!available) {
        return SHADOWSPILL_STATUS_CLOSED;
    }
    backend.profiler_enable(backend.state, enabled != 0U);
    atomic_store_explicit(
        &adapter.profiler_annotations_enabled,
        enabled != 0U,
        memory_order_release
    );
    return SHADOWSPILL_STATUS_OK;
}
