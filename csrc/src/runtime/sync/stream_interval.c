#include "internal.h"
#include "../internal.h"

/* Timing events come from the runtime's timing pool, reserved when a trace is
 * prepared; an exhausted pool leaves an interval unmeasured, never a failure
 * of the transfer it would have measured. */

int shadowspill_stream_interval_open(
    ShadowSpillRuntime *runtime,
    ShadowSpillStreamInterval *interval,
    ShadowSpillBackendStream stream
) {
    ShadowSpillBackend *backend = &runtime->backend;
    if (shadowspill_event_lease_acquire(
            runtime, &runtime->timing_events, &interval->start
        ) != SHADOWSPILL_STATUS_OK) {
        return -1;
    }
    if (shadowspill_event_lease_acquire(
            runtime, &runtime->timing_events, &interval->end
        ) != SHADOWSPILL_STATUS_OK) {
        (void)shadowspill_event_lease_release(runtime, interval->start);
        return -1;
    }
    if (backend->record_event(backend->state, interval->start->event, stream) != 0) {
        (void)shadowspill_event_lease_release(runtime, interval->start);
        (void)shadowspill_event_lease_release(runtime, interval->end);
        return -1;
    }
    interval->open = 1U;
    return 0;
}

int shadowspill_stream_interval_close(
    ShadowSpillRuntime *runtime,
    ShadowSpillStreamInterval *interval,
    ShadowSpillBackendStream stream
) {
    if (!interval->open) {
        return -1;
    }
    ShadowSpillBackend *backend = &runtime->backend;
    if (backend->record_event(backend->state, interval->end->event, stream) != 0) {
        shadowspill_stream_interval_discard(runtime, interval);
        return -1;
    }
    return 0;
}

int shadowspill_stream_interval_read(
    ShadowSpillRuntime *runtime,
    const ShadowSpillStreamInterval *interval,
    ShadowSpillBackendEvent origin,
    uint64_t *start_ns,
    uint64_t *end_ns
) {
    if (!interval->open) {
        return -1;
    }
    ShadowSpillBackend *backend = &runtime->backend;
    uint64_t start = 0U;
    uint64_t end = 0U;
    int status = backend->elapsed_nanoseconds(
        backend->state, origin, interval->start->event, &start
    );
    if (status == 0) {
        status = backend->elapsed_nanoseconds(
            backend->state, origin, interval->end->event, &end
        );
    }
    if (status != 0) {
        return status;
    }
    *start_ns = start;
    *end_ns = end;
    return 0;
}

void shadowspill_stream_interval_discard(
    ShadowSpillRuntime *runtime,
    ShadowSpillStreamInterval *interval
) {
    if (!interval->open) {
        return;
    }
    (void)shadowspill_event_lease_release(runtime, interval->start);
    (void)shadowspill_event_lease_release(runtime, interval->end);
    interval->open = 0U;
}
