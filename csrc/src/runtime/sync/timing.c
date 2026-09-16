/* Timing work on the device, against the runtime's own events. */

#include <stdlib.h>

#include "../internal.h"
#include "internal.h"

#include <shadowspill/runtime/timing.h>

struct ShadowSpillTimingMarker {
    ShadowSpillRuntime *runtime;
    ShadowSpillEventLease *lease;
};

ShadowSpillStatus shadowspill_timing_marker_create(
    ShadowSpillRuntime *runtime,
    ShadowSpillTimingMarker **marker
) {
    if (runtime == NULL || marker == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    *marker = NULL;
    ShadowSpillTimingMarker *record = calloc(1U, sizeof(*record));
    if (record == NULL) {
        return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    }
    ShadowSpillStatus status = shadowspill_event_lease_acquire(
        runtime, &runtime->timing_events, &record->lease
    );
    if (status != SHADOWSPILL_STATUS_OK) {
        /* The pool is sealed with nothing free: its reserve belongs to the
         * lanes a trace measures, and a marker must not eat it. Taking a
         * marker is cold -- a caller takes them before it times anything --
         * so grow the pool for this one, which is the further reservation
         * that may raise driver creates after sealing. */
        status = shadowspill_event_pool_reserve(
            runtime, &runtime->timing_events, 1U
        );
        if (status == SHADOWSPILL_STATUS_OK) {
            status = shadowspill_event_lease_acquire(
                runtime, &runtime->timing_events, &record->lease
            );
        }
    }
    if (status != SHADOWSPILL_STATUS_OK) {
        free(record);
        return status;
    }
    record->runtime = runtime;
    *marker = record;
    return SHADOWSPILL_STATUS_OK;
}

ShadowSpillStatus shadowspill_timing_marker_record(
    ShadowSpillTimingMarker *marker,
    uint64_t compute_stream
) {
    if (marker == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    ShadowSpillBackend *backend = &marker->runtime->backend;
    ShadowSpillBackendStream stream =
        backend->resolve_stream(backend->state, compute_stream);
    if (backend->record_event(backend->state, marker->lease->event, stream) != 0) {
        return SHADOWSPILL_STATUS_BACKEND_FAILURE;
    }
    return SHADOWSPILL_STATUS_OK;
}

ShadowSpillBackendEvent shadowspill_timing_marker_event(
    const ShadowSpillTimingMarker *marker
) {
    if (marker == NULL) {
        return (ShadowSpillBackendEvent){0};
    }
    return marker->lease->event;
}

ShadowSpillStatus shadowspill_timing_marker_query(
    const ShadowSpillTimingMarker *marker,
    uint8_t *reached
) {
    if (marker == NULL || reached == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    ShadowSpillBackend *backend = &marker->runtime->backend;
    int complete = 0;
    if (backend->query_event(backend->state, marker->lease->event, &complete) != 0) {
        return SHADOWSPILL_STATUS_BACKEND_FAILURE;
    }
    *reached = complete != 0 ? 1U : 0U;
    return SHADOWSPILL_STATUS_OK;
}

ShadowSpillStatus shadowspill_timing_marker_wait(
    const ShadowSpillTimingMarker *marker
) {
    if (marker == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    ShadowSpillBackend *backend = &marker->runtime->backend;
    if (backend->synchronize_event(backend->state, marker->lease->event) != 0) {
        return SHADOWSPILL_STATUS_BACKEND_FAILURE;
    }
    return SHADOWSPILL_STATUS_OK;
}

ShadowSpillStatus shadowspill_timing_elapsed(
    const ShadowSpillTimingMarker *from,
    const ShadowSpillTimingMarker *to,
    uint8_t *reached,
    uint64_t *nanoseconds
) {
    if (from == NULL || to == NULL || reached == NULL || nanoseconds == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    *reached = 0U;
    ShadowSpillBackend *backend = &from->runtime->backend;
    int complete = 0;
    if (backend->query_event(backend->state, to->lease->event, &complete) != 0) {
        return SHADOWSPILL_STATUS_BACKEND_FAILURE;
    }
    if (complete == 0) {
        /* The device has not reached the later marker yet. */
        return SHADOWSPILL_STATUS_OK;
    }
    uint64_t elapsed = 0U;
    if (backend->elapsed_nanoseconds(
            backend->state, from->lease->event, to->lease->event, &elapsed
        ) != 0) {
        return SHADOWSPILL_STATUS_BACKEND_FAILURE;
    }
    *nanoseconds = elapsed;
    *reached = 1U;
    return SHADOWSPILL_STATUS_OK;
}

ShadowSpillStatus shadowspill_timing_stream_wait(
    ShadowSpillRuntime *runtime,
    uint64_t compute_stream
) {
    if (runtime == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    ShadowSpillBackend *backend = &runtime->backend;
    ShadowSpillBackendStream stream =
        backend->resolve_stream(backend->state, compute_stream);
    if (backend->synchronize_stream(backend->state, stream) != 0) {
        return SHADOWSPILL_STATUS_BACKEND_FAILURE;
    }
    return SHADOWSPILL_STATUS_OK;
}

void shadowspill_timing_marker_release(ShadowSpillTimingMarker *marker) {
    if (marker == NULL) {
        return;
    }
    (void)shadowspill_event_lease_release(marker->runtime, marker->lease);
    free(marker);
}
