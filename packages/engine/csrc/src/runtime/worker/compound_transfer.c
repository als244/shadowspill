/* A representation conversion uses the selected lane for its physical copy.
 * The I/O join and every collective share one final completion boundary. */
#include "internal.h"
#include "../../common/platform.h"

typedef struct SpillCopy {
    ShadowSpillQueuedAction *action;
    uint8_t called;
} SpillCopy;

static int copy_spill(
    void *context, void *destination, const void *source, uint64_t bytes
) {
    SpillCopy *copy = context;
    ShadowSpillQueuedAction *action = copy->action;
    if (copy->called || bytes != action->object->transfer.spill_size_bytes) {
        return -1;
    }
    copy->called = 1U;
    const ShadowSpillRouteState *route = action->route;
    if (route->operations->copy(
            route->lane, destination, source, bytes, &action->lane_handle
        ) != 0) return -1;
    return route->operations->join(route->lane);
}

int shadowspill_compound_transfer(
    ShadowSpillRuntime *runtime, ShadowSpillQueuedAction *action,
    void *destination, const void *source
) {
    const ShadowSpillRouteState *route = action->route;
    ShadowSpillLane *lane = route->lane;
    if (route->operations->join == NULL) return -1;
    if (shadowspill_lane_trace_active(runtime)) {
        action->compound_issued_ns = shadowspill_monotonic_ns();
        (void)shadowspill_stream_interval_open(
            runtime, &action->compound_interval, lane->stream
        );
    }
    const ShadowSpillObjectTransfer *transfer = &action->object->transfer;
    SpillCopy copy = {.action = action};
    const int status = transfer->enqueue(
        transfer->context, lane->backend, lane->stream,
        action->kind == SHADOWSPILL_RUNTIME_FETCH, destination, source,
        copy_spill, &copy
    );
    (void)shadowspill_stream_interval_close(
        runtime, &action->compound_interval, lane->stream
    );
    return status;
}
