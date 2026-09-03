#define _GNU_SOURCE
#include "internal.h"
#include "../../common/platform.h"

#include <pthread.h>
#include <stdint.h>

/* The backend's profiler entries are optional: a NULL entry is a no-op. */

void shadowspill_profiler_name_current_thread(
    const ShadowSpillBackend *backend, const char *name
) {
    shadowspill_name_current_thread(name);
    if (backend != NULL && backend->name_thread != NULL) {
        backend->name_thread(backend->state, name);
    }
}

void shadowspill_profiler_name_stream(
    const ShadowSpillBackend *backend,
    ShadowSpillBackendStream stream,
    const char *name
) {
    if (backend != NULL && backend->name_stream != NULL) {
        backend->name_stream(backend->state, stream, name);
    }
}

ShadowSpillProfilerRange shadowspill_profiler_range_begin(
    const ShadowSpillBackend *backend, const char *name
) {
    return backend != NULL && backend->range_begin != NULL
        ? backend->range_begin(backend->state, name)
        : 0U;
}

void shadowspill_profiler_range_end(
    const ShadowSpillBackend *backend, ShadowSpillProfilerRange range
) {
    if (backend != NULL && backend->range_end != NULL) {
        backend->range_end(backend->state, range);
    }
}
