#ifndef SHADOWSPILL_RUNTIME_TRANSFERS_INTERNAL_H
#define SHADOWSPILL_RUNTIME_TRANSFERS_INTERNAL_H

/*
 * Transfer lanes and routes.
 *
 * A route pairs a source and destination pool with the backend stream that
 * moves bytes between them; its lane orders the actions issued on it.
 */

#include <pthread.h>
#include <stdint.h>

#include <shadowspill/runtime.h>

typedef struct ShadowSpillQueuedAction ShadowSpillQueuedAction;
typedef struct ShadowSpillRouteState ShadowSpillRouteState;

typedef struct ShadowSpillTransferLane {
    pthread_mutex_t lock;
    ShadowSpillQueuedAction *pending_head;
    ShadowSpillQueuedAction *pending_tail;
    ShadowSpillQueuedAction *inflight_head;
    ShadowSpillQueuedAction *inflight_tail;
    uint8_t lock_initialized;
} ShadowSpillTransferLane;

enum {
    SHADOWSPILL_FETCH_ROUTE_ID = 0U,
    SHADOWSPILL_EVICT_ROUTE_ID = 1U,
    SHADOWSPILL_TRANSFER_FETCH = 0U,
    SHADOWSPILL_TRANSFER_EVICT = 1U,
};

struct ShadowSpillRouteState {
    uint32_t source_pool_id;
    uint32_t destination_pool_id;
    /* Copy direction, derived from the two pools' kinds at create. */
    uint8_t to_device;
    ShadowSpillTransferLane transfers;
    ShadowSpillBackendStream lane;
    uint8_t lane_created;
};

/* One asynchronous copy along a route, on the backend's copy for its direction. */
int shadowspill_route_copy_async(
    ShadowSpillRuntime *runtime,
    const ShadowSpillRouteState *route,
    void *destination,
    const void *source,
    uint64_t bytes,
    ShadowSpillBackendStream stream
);

int shadowspill_transfer_profiles_initialize(ShadowSpillRuntime *runtime);

void shadowspill_transfer_profiles_destroy(ShadowSpillRuntime *runtime);

int shadowspill_transfer_lane_initialize(ShadowSpillTransferLane *lane);

void shadowspill_transfer_lane_destroy(ShadowSpillTransferLane *lane);

ShadowSpillTransferLane *shadowspill_transfer_lane_for_action(
    ShadowSpillRuntime *runtime,
    const ShadowSpillQueuedAction *action
);

void shadowspill_transfer_lane_enqueue(
    ShadowSpillTransferLane *lane,
    ShadowSpillQueuedAction *action
);

int shadowspill_transfer_lane_claim(
    ShadowSpillTransferLane *lane,
    ShadowSpillQueuedAction *action
);

void shadowspill_transfer_lane_publish_inflight(
    ShadowSpillTransferLane *lane,
    ShadowSpillQueuedAction *action
);

int shadowspill_transfer_lane_is_inflight_head(
    ShadowSpillTransferLane *lane,
    const ShadowSpillQueuedAction *action
);

int shadowspill_transfer_lane_complete(
    ShadowSpillTransferLane *lane,
    ShadowSpillQueuedAction *action
);

#endif
