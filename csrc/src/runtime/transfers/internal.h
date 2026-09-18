#ifndef SHADOWSPILL_RUNTIME_TRANSFERS_INTERNAL_H
#define SHADOWSPILL_RUNTIME_TRANSFERS_INTERNAL_H

/*
 * Transfer queues and routes.
 *
 * A route pairs a source and destination pool with the backend stream that
 * moves bytes between them; its queue orders the actions issued on it.
 *
 * "Queue" and not "queue": the queue is the thing that moves the bytes, and a
 * route names exactly one. What this directory owns is the ordering in front
 * of it.
 */

#include <pthread.h>
#include <stdint.h>

#include <shadowspill/runtime.h>

typedef struct ShadowSpillQueuedAction ShadowSpillQueuedAction;
typedef struct ShadowSpillRouteState ShadowSpillRouteState;

/*
 * One queue serves two orders. The plan's transfers are dispatched in the
 * order their boundaries triggered them; background transfers (those the
 * plan did not schedule: an opening restore, a reconciliation) are dispatched
 * in their own order and only while the queue holds fewer than
 * `background_window_bytes` of them in flight, so a plan transfer never
 * waits behind more than the window. In flight, it is one FIFO in
 * dispatch order, which is the stream's order and what completion follows.
 */
/* Where an action sits in its queue. On the action as `queue_state`, declared
   here because the queue owns the vocabulary and its canary asserts on it. */
enum {
    SHADOWSPILL_QUEUE_NONE = 0,
    SHADOWSPILL_QUEUE_PENDING = 1,
    SHADOWSPILL_QUEUE_INFLIGHT = 2,
};

typedef struct ShadowSpillTransferQueue {
    pthread_mutex_t lock;
    ShadowSpillQueuedAction *pending_head;
    ShadowSpillQueuedAction *pending_tail;
    ShadowSpillQueuedAction *background_head;
    ShadowSpillQueuedAction *background_tail;
    ShadowSpillQueuedAction *inflight_head;
    ShadowSpillQueuedAction *inflight_tail;
    uint64_t background_window_bytes;
    uint64_t background_inflight_bytes;
    /* How many actions are in flight, read without the lock so the worker can
       skip polling a lane that has nothing to report. The list above is the
       truth; this is the same fact in a form a hot loop can afford. */
    _Atomic uint32_t inflight_count;
    uint8_t lock_initialized;
} ShadowSpillTransferQueue;

enum {
    SHADOWSPILL_FETCH_ROUTE_ID = 0U,
    SHADOWSPILL_EVICT_ROUTE_ID = 1U,
    SHADOWSPILL_TRANSFER_FETCH = 0U,
    SHADOWSPILL_TRANSFER_EVICT = 1U,
};

struct ShadowSpillRouteState {
    uint32_t source_pool_id;
    uint32_t destination_pool_id;
    ShadowSpillTransferQueue queue;
    ShadowSpillBackendStream stream;
    /* What moves this route's bytes, resolved from its two pools' kinds at
       create. The built-in lane copies on the stream above; a lane that works
       on its own stream gets one of its own and leaves this one to the
       runtime. */
    ShadowSpillLane *lane;
    const ShadowSpillLaneOperations *operations;
    uint8_t stream_created;
};

/* The built-in entries carry no configuration: a lane reads its direction from
   the pair of kinds in its common struct, which the runtime fills. */
void shadowspill_pinned_host_device_lanes_describe(
    ShadowSpillRuntime *runtime, ShadowSpillLaneDescription descriptions[2]
);

/*
 * Every lane the runtime can resolve: the built-ins first, then whatever the
 * config registered. One lookup serves both, which is the point -- there is no
 * branch asking whether a route is local.
 */
typedef struct ShadowSpillLaneTable {
    ShadowSpillLaneDescription *entries;
    uint32_t count;
} ShadowSpillLaneTable;

int shadowspill_lane_table_initialize(
    ShadowSpillLaneTable *table,
    ShadowSpillRuntime *runtime,
    const ShadowSpillLaneDescription *registered,
    uint32_t registered_count
);
void shadowspill_lane_table_destroy(ShadowSpillLaneTable *table);

/* The lane for a directional pool-kind pair, or NULL if none serves it. */
const ShadowSpillLaneDescription *shadowspill_lane_for_kinds(
    const ShadowSpillLaneTable *table, uint8_t from_kind, uint8_t to_kind
);

int shadowspill_transfer_profiles_initialize(ShadowSpillRuntime *runtime);

void shadowspill_transfer_profiles_destroy(ShadowSpillRuntime *runtime);

int shadowspill_transfer_queue_initialize(ShadowSpillTransferQueue *queue);

void shadowspill_transfer_queue_destroy(ShadowSpillTransferQueue *queue);

ShadowSpillTransferQueue *shadowspill_transfer_queue_for_action(
    ShadowSpillRuntime *runtime,
    const ShadowSpillQueuedAction *action
);

void shadowspill_transfer_queue_enqueue(
    ShadowSpillTransferQueue *queue,
    ShadowSpillQueuedAction *action
);

int shadowspill_transfer_queue_claim(
    ShadowSpillTransferQueue *queue,
    ShadowSpillQueuedAction *action
);

/* Put a claimed action back at its head, still pending, for a lane that asked
   to be retried. */
void shadowspill_transfer_queue_return(
    ShadowSpillTransferQueue *queue,
    ShadowSpillQueuedAction *action
);

void shadowspill_transfer_queue_publish_inflight(
    ShadowSpillTransferQueue *queue,
    ShadowSpillQueuedAction *action
);

int shadowspill_transfer_queue_is_inflight_head(
    ShadowSpillTransferQueue *queue,
    const ShadowSpillQueuedAction *action
);

int shadowspill_transfer_queue_complete(
    ShadowSpillTransferQueue *queue,
    ShadowSpillQueuedAction *action
);

#endif
