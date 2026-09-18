/* The lane between a pinned-host pool and a device pool, either direction. */
#include "../internal.h"

#include <stdlib.h>

/*
 * A thin table over the backend. Everything it does, the runtime did inline
 * before there was a lane contract: a wait is `wait_event` on this lane's
 * stream, a copy is whichever backend copy entry the direction calls for, and
 * a signal is `record_event`.
 *
 * One instance per route. Direction is not configured: it is `from_kind` and
 * `to_kind` in the common struct, which the runtime fills from the description
 * this lane was found through. A route has one direction for its whole life,
 * and now only one place says which.
 */

/*
 * How many transfers this lane can have measured at once.
 *
 * A record exists only while a trace runs, from the copy that opens it to the
 * runtime's one query after that transfer completes -- the same lifetime the
 * runtime used to give the interval it kept on the action. So the ring only
 * has to outlast the transfers in flight on one route, and this is well past
 * that. A slot that is reused before it is read reports its transfer untimed
 * rather than reporting another transfer's numbers, which is what the handle
 * stored beside each interval is for.
 */
#define MEASURED_TRANSFERS 256U

typedef struct MeasuredTransfer {
    uint64_t handle;
    uint64_t bytes;
    ShadowSpillStreamInterval interval;
} MeasuredTransfer;

typedef struct PinnedHostDeviceLane {
    ShadowSpillLane base;

    /*
     * Handles are numbered from 1 over the lane's life, and 1 is the first
     * because 0 is how `copy` says it kept nothing. The counter is atomic
     * because calibration calls a lane off the worker thread; it costs one
     * relaxed add, and only on the traced path.
     */
    _Atomic uint64_t next_handle;
    MeasuredTransfer measured[MEASURED_TRANSFERS];
} PinnedHostDeviceLane;

static MeasuredTransfer *slot_of(PinnedHostDeviceLane *self, uint64_t handle) {
    return &self->measured[(handle - 1U) % MEASURED_TRANSFERS];
}

static int pinned_host_device_create(
    const ShadowSpillLane *base, void *configuration, ShadowSpillLane **lane
) {
    (void)configuration;
    if (base == NULL || lane == NULL) {
        return -1;
    }
    PinnedHostDeviceLane *created = calloc(1U, sizeof(*created));
    if (created == NULL) {
        return -1;
    }
    created->base = *base;
    atomic_store_explicit(&created->next_handle, 1U, memory_order_relaxed);
    *lane = &created->base;
    return 0;
}

static int pinned_host_device_wait(ShadowSpillLane *lane, ShadowSpillBackendEvent event) {
    shadowspill_lane_counted(&lane->waits, 1U);
    /* A device-side wait always enqueues, so this never asks for a retry. */
    return lane->backend->wait_event(lane->backend->state, lane->stream, event) == 0
        ? 0
        : -1;
}

static int pinned_host_device_copy(
    ShadowSpillLane *lane,
    void *destination,
    const void *source,
    uint64_t bytes,
    uint64_t *handle
) {
    PinnedHostDeviceLane *self = (PinnedHostDeviceLane *)lane;
    const ShadowSpillBackend *backend = lane->backend;
    *handle = 0U;

    /*
     * A traced transfer is bracketed on this lane's stream: the interval opens
     * just before the copy and closes just after it, ahead of whatever event
     * `signal` records, so observing the completion guarantees the interval is
     * readable. An untraced transfer pays the one question and nothing else,
     * and an interval that cannot be opened is a gap in the trace rather than
     * a failed transfer -- so nothing below branches on it.
     */
    MeasuredTransfer *record = NULL;
    if (shadowspill_lane_trace_active(lane->runtime)) {
        const uint64_t claimed = atomic_fetch_add_explicit(
            &self->next_handle, 1U, memory_order_relaxed
        );
        record = slot_of(self, claimed);
        /* Whatever was here was never queried. Its leases go back now. */
        shadowspill_stream_interval_discard(lane->runtime, &record->interval);
        record->handle = claimed;
        record->bytes = bytes;
        if (shadowspill_stream_interval_open(
                lane->runtime, &record->interval, lane->stream
            ) == 0) {
            *handle = claimed;
        } else {
            record->handle = 0U;
            record = NULL;
        }
    }

    const int failed = lane->to_kind == SHADOWSPILL_POOL_DEVICE
        ? backend->copy_host_to_device(
              backend->state, destination, source, bytes, lane->stream
          )
        : backend->copy_device_to_host(
              backend->state, destination, source, bytes, lane->stream
          );
    if (failed != 0) {
        if (record != NULL) {
            shadowspill_stream_interval_discard(lane->runtime, &record->interval);
            record->handle = 0U;
            *handle = 0U;
        }
        shadowspill_lane_counted(&lane->failures, 1U);
        return failed;
    }
    if (record != NULL) {
        (void)shadowspill_stream_interval_close(
            lane->runtime, &record->interval, lane->stream
        );
    }
    /* One chunk per copy: this lane hands the whole transfer to the backend
       and never splits it, so `chunks` equals `copies` by construction. */
    shadowspill_lane_counted(&lane->copies, 1U);
    shadowspill_lane_counted(&lane->chunks, 1U);
    shadowspill_lane_counted(&lane->bytes, bytes);
    return 0;
}

static int pinned_host_device_signal(
    ShadowSpillLane *lane, uint64_t handle, ShadowSpillBackendEvent event
) {
    /* The stream orders the event behind the copies already on it, so this
       lane needs nothing from the handle. */
    (void)handle;
    shadowspill_lane_counted(&lane->signals, 1U);
    return lane->backend->record_event(lane->backend->state, event, lane->stream);
}

static int pinned_host_device_synchronize(ShadowSpillLane *lane) {
    return lane->backend->synchronize_stream(lane->backend->state, lane->stream);
}

static int pinned_host_device_transfer(
    ShadowSpillLane *lane, uint64_t handle, ShadowSpillLaneTransfer *transfer
) {
    PinnedHostDeviceLane *self = (PinnedHostDeviceLane *)lane;
    if (handle == 0U) {
        return -1;
    }
    MeasuredTransfer *record = slot_of(self, handle);
    if (record->handle != handle) {
        /* Overwritten before it was read. Nothing to report, and nothing to
           release: whoever took the slot released these leases. */
        return -1;
    }
    uint64_t started = SHADOWSPILL_LANE_NO_TIME;
    uint64_t finished = SHADOWSPILL_LANE_NO_TIME;
    const int read = shadowspill_stream_interval_read(
        lane->runtime, &record->interval, lane->runtime->trace_origin_event,
        &started, &finished
    );
    *transfer = (ShadowSpillLaneTransfer){
        .started_at_nanoseconds = read == 0 ? started : SHADOWSPILL_LANE_NO_TIME,
        .finished_at_nanoseconds = read == 0 ? finished : SHADOWSPILL_LANE_NO_TIME,
        .bytes = record->bytes,
        .chunks = 1U,
    };
    /* The query retires the handle. */
    shadowspill_stream_interval_discard(lane->runtime, &record->interval);
    record->handle = 0U;
    return 0;
}

/*
 * No `timing`. This lane hands the backend a copy and returns, so the moment a
 * transfer *completes* is not something it observes -- an interval around the
 * copy is, and that is what `transfer` reports per transfer. Reporting a zero
 * duration here would be a lie, and a NULL entry is how the contract says so.
 */

/* The stream is the runtime's to destroy, along with the route that owns it. */
static void pinned_host_device_destroy(ShadowSpillLane *lane) {
    PinnedHostDeviceLane *self = (PinnedHostDeviceLane *)lane;
    /* Any record never queried still holds two timing leases. */
    for (uint32_t index = 0U; index < MEASURED_TRANSFERS; ++index) {
        shadowspill_stream_interval_discard(
            lane->runtime, &self->measured[index].interval
        );
    }
    free(self);
}

static const ShadowSpillLaneOperations pinned_host_device_operations = {
    .wait = pinned_host_device_wait,
    .copy = pinned_host_device_copy,
    .signal = pinned_host_device_signal,
    .synchronize = pinned_host_device_synchronize,
    .transfer = pinned_host_device_transfer,
    .destroy = pinned_host_device_destroy,
    .timing = NULL,
};

/*
 * Two entries, one per direction. The runtime calls this while building its
 * lane table, so the storage lives as long as the runtime and the descriptions
 * point into it. There is no configuration: direction is the pair of kinds
 * named right here, and `create` receives the runtime, so there is nothing
 * left to smuggle through.
 */
void shadowspill_pinned_host_device_lanes_describe(
    ShadowSpillRuntime *runtime, ShadowSpillLaneDescription descriptions[2]
) {
    (void)runtime;
    descriptions[0] = (ShadowSpillLaneDescription){
        .from_kind = SHADOWSPILL_POOL_PINNED_HOST,
        .to_kind = SHADOWSPILL_POOL_DEVICE,
        .operations = &pinned_host_device_operations,
        .create = pinned_host_device_create,
        .configuration = NULL,
    };
    descriptions[1] = (ShadowSpillLaneDescription){
        .from_kind = SHADOWSPILL_POOL_DEVICE,
        .to_kind = SHADOWSPILL_POOL_PINNED_HOST,
        .operations = &pinned_host_device_operations,
        .create = pinned_host_device_create,
        .configuration = NULL,
    };
}
