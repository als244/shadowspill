/* The lane between a pinned-host pool and a device pool, either direction. */
#include "../internal.h"

#include <stdlib.h>

/*
 * A thin table over the backend. Everything it does, the runtime did inline
 * before there was a lane contract: a wait is `wait_event` on this lane's
 * stream, a copy is whichever backend copy entry the direction calls for, a
 * signal is `record_event`, and an interval is the timing pool's.
 *
 * One instance per route. Direction is fixed at create rather than decided per
 * copy, because a route has one direction for its whole life: the built-in
 * registers two descriptions, one per kind pair, and each names the
 * configuration that says which way its copies go.
 */
typedef struct PinnedHostDeviceLane {
    ShadowSpillRuntime *runtime;
    const ShadowSpillBackend *backend;
    ShadowSpillBackendStream stream;
    uint8_t to_device;

    /*
     * What this lane has moved. Counted here rather than by the worker because
     * the count belongs to the lane the transfer went through, and the worker
     * does not know which that is once it dispatches against a contract.
     *
     * No timing: this lane hands the backend a copy and returns, so the moment
     * a transfer *completes* is not something it observes -- an interval is,
     * and that is what `interval_open`/`interval_close` are for. Reporting a
     * zero duration would be a lie, so `timed` stays 0 and the fields stay
     * zero, which is the difference the flag exists to record.
     */
    _Atomic uint64_t copies;
    _Atomic uint64_t bytes;
    _Atomic uint64_t signals;
    _Atomic uint64_t waits;
    _Atomic uint64_t failures;
} PinnedHostDeviceLane;

static int pinned_host_device_create(
    ShadowSpillRuntime *runtime,
    const ShadowSpillBackend *backend,
    ShadowSpillBackendStream stream,
    void *configuration,
    ShadowSpillLane **lane
) {
    if (backend == NULL || lane == NULL || configuration == NULL) {
        return -1;
    }
    PinnedHostDeviceLane *created = calloc(1U, sizeof(*created));
    if (created == NULL) {
        return -1;
    }
    created->backend = backend;
    created->stream = stream;
    created->runtime = runtime;
    created->to_device = *(const uint8_t *)configuration;
    *lane = (ShadowSpillLane *)created;
    return 0;
}

static int pinned_host_device_wait(ShadowSpillLane *lane, ShadowSpillBackendEvent event) {
    PinnedHostDeviceLane *self = (PinnedHostDeviceLane *)lane;
    (void)atomic_fetch_add_explicit(&self->waits, 1U, memory_order_relaxed);
    /* A device-side wait always enqueues, so this never asks for a retry. */
    return self->backend->wait_event(self->backend->state, self->stream, event) == 0
        ? 0
        : -1;
}

static int pinned_host_device_copy(
    ShadowSpillLane *lane, void *destination, const void *source, uint64_t bytes
) {
    PinnedHostDeviceLane *self = (PinnedHostDeviceLane *)lane;
    const ShadowSpillBackend *backend = self->backend;
    const int failed = self->to_device
        ? backend->copy_host_to_device(
              backend->state, destination, source, bytes, self->stream
          )
        : backend->copy_device_to_host(
              backend->state, destination, source, bytes, self->stream
          );
    if (failed != 0) {
        (void)atomic_fetch_add_explicit(&self->failures, 1U, memory_order_relaxed);
        return failed;
    }
    /* One chunk per copy: this lane hands the whole transfer to the backend
       and never splits it, so `chunks` equals `copies` by construction. */
    (void)atomic_fetch_add_explicit(&self->copies, 1U, memory_order_relaxed);
    (void)atomic_fetch_add_explicit(&self->bytes, bytes, memory_order_relaxed);
    return 0;
}

static int pinned_host_device_signal(ShadowSpillLane *lane, ShadowSpillBackendEvent event) {
    PinnedHostDeviceLane *self = (PinnedHostDeviceLane *)lane;
    (void)atomic_fetch_add_explicit(&self->signals, 1U, memory_order_relaxed);
    return self->backend->record_event(self->backend->state, event, self->stream);
}

static int pinned_host_device_synchronize(ShadowSpillLane *lane) {
    PinnedHostDeviceLane *self = (PinnedHostDeviceLane *)lane;
    return self->backend->synchronize_stream(self->backend->state, self->stream);
}

static int pinned_host_device_interval_open(
    ShadowSpillLane *lane, ShadowSpillStreamInterval *interval
) {
    PinnedHostDeviceLane *self = (PinnedHostDeviceLane *)lane;
    return shadowspill_stream_interval_open(self->runtime, interval, self->stream);
}

static int pinned_host_device_interval_close(
    ShadowSpillLane *lane, ShadowSpillStreamInterval *interval
) {
    PinnedHostDeviceLane *self = (PinnedHostDeviceLane *)lane;
    return shadowspill_stream_interval_close(self->runtime, interval, self->stream);
}

static int pinned_host_device_statistics(
    const ShadowSpillLane *lane, ShadowSpillLaneStatistics *statistics
) {
    if (lane == NULL || statistics == NULL) {
        return -1;
    }
    PinnedHostDeviceLane *self =
        (PinnedHostDeviceLane *)(uintptr_t)(const void *)lane;
    const uint64_t copies =
        atomic_load_explicit(&self->copies, memory_order_relaxed);
    *statistics = (ShadowSpillLaneStatistics){
        .copies = copies,
        .chunks = copies,
        .bytes = atomic_load_explicit(&self->bytes, memory_order_relaxed),
        .signals = atomic_load_explicit(&self->signals, memory_order_relaxed),
        .waits = atomic_load_explicit(&self->waits, memory_order_relaxed),
        .retries = 0U,
        .failures = atomic_load_explicit(&self->failures, memory_order_relaxed),
        .timed = 0U,
    };
    return 0;
}

/* The stream is the runtime's to destroy, along with the route that owns it. */
static void pinned_host_device_destroy(ShadowSpillLane *lane) {
    free(lane);
}

static const ShadowSpillLaneOperations pinned_host_device_operations = {
    .wait = pinned_host_device_wait,
    .copy = pinned_host_device_copy,
    .signal = pinned_host_device_signal,
    .synchronize = pinned_host_device_synchronize,
    .interval_open = pinned_host_device_interval_open,
    .interval_close = pinned_host_device_interval_close,
    .destroy = pinned_host_device_destroy,
    .statistics = pinned_host_device_statistics,
};

/*
 * Two entries, one per direction. The runtime calls this while building its
 * lane table, so the storage lives as long as the runtime and the descriptions
 * point into it. `configuration` is only the direction: `create` receives the
 * runtime, so there is nothing to smuggle through it.
 */
void shadowspill_pinned_host_device_lanes_describe(
    ShadowSpillRuntime *runtime,
    ShadowSpillPinnedHostDeviceConfiguration storage[2],
    ShadowSpillLaneDescription descriptions[2]
) {
    (void)runtime;
    storage[0] = (ShadowSpillPinnedHostDeviceConfiguration){.to_device = 1U};
    storage[1] = (ShadowSpillPinnedHostDeviceConfiguration){.to_device = 0U};
    descriptions[0] = (ShadowSpillLaneDescription){
        .from_kind = SHADOWSPILL_POOL_PINNED_HOST,
        .to_kind = SHADOWSPILL_POOL_DEVICE,
        .operations = &pinned_host_device_operations,
        .create = pinned_host_device_create,
        .configuration = &storage[0],
    };
    descriptions[1] = (ShadowSpillLaneDescription){
        .from_kind = SHADOWSPILL_POOL_DEVICE,
        .to_kind = SHADOWSPILL_POOL_PINNED_HOST,
        .operations = &pinned_host_device_operations,
        .create = pinned_host_device_create,
        .configuration = &storage[1],
    };
}
