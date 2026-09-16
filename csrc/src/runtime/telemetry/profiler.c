#define _GNU_SOURCE
#include "internal.h"
#include "../internal.h"
#include "../../common/platform.h"

#include <pthread.h>
#include <stdatomic.h>
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

uint8_t shadowspill_profiler_annotations_enabled(ShadowSpillRuntime *runtime) {
    return runtime == NULL
        ? 0U
        : atomic_load_explicit(
              &runtime->profiler_annotations_enabled, memory_order_relaxed
          );
}

ShadowSpillProfilerRange shadowspill_profiler_range_begin(
    ShadowSpillRuntime *runtime, const char *name
) {
    if (runtime == NULL ||
        atomic_load_explicit(
            &runtime->profiler_annotations_enabled, memory_order_relaxed
        ) == 0U) {
        return 0U;
    }
    const ShadowSpillBackend *backend = &runtime->backend;
    return backend->range_begin != NULL
        ? backend->range_begin(backend->state, name)
        : 0U;
}

ShadowSpillStatus shadowspill_profiler_annotations_set(
    ShadowSpillRuntime *runtime, uint8_t enabled
) {
    if (runtime == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    /* A backend with no profiler is a no-op, not a failure: the whole
       facility is best-effort and never changes execution semantics. */
    const ShadowSpillBackend *backend = &runtime->backend;
    if (backend->profiler_enable != NULL) {
        backend->profiler_enable(backend->state, enabled);
    }
    atomic_store_explicit(
        &runtime->profiler_annotations_enabled,
        enabled != 0U ? 1U : 0U,
        memory_order_relaxed
    );
    return SHADOWSPILL_STATUS_OK;
}

void shadowspill_profiler_range_end(
    ShadowSpillRuntime *runtime, ShadowSpillProfilerRange range
) {
    const ShadowSpillBackend *backend = runtime == NULL ? NULL : &runtime->backend;
    if (backend != NULL && backend->range_end != NULL) {
        backend->range_end(backend->state, range);
    }
}
