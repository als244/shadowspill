#ifndef SHADOWSPILL_RUNTIME_SYNC_INTERNAL_H
#define SHADOWSPILL_RUNTIME_SYNC_INTERNAL_H

/*
 * Events, completion tracking and the quiescence notification.
 *
 * An event lease is a reference-counted handle on one backend event. Nothing
 * here decides when memory becomes reusable: a completed event says only that
 * the backend finished, and the owning pool commits the matching lease
 * transition under its own lock.
 */

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>

#include <shadowspill/runtime.h>

typedef struct ShadowSpillEventLease ShadowSpillEventLease;
typedef struct ShadowSpillEventPool ShadowSpillEventPool;

struct ShadowSpillEventLease {
    ShadowSpillBackendEvent event;
    uint64_t generation;
    uint64_t completion_object_id;
    uint64_t completion_allocation_id;
    _Atomic uint32_t references;
    /*
     * The backend has reported that the event completed. This is deliberately
     * not a memory-availability flag: a range becomes reusable only when its
     * owning MemoryPool commits the matching lease transition under pool.lock.
     */
    _Atomic uint8_t backend_complete;
    struct ShadowSpillEventLease *completion_next;
    struct ShadowSpillEventLease *free_next;
    uint8_t pool_owned;
    uint8_t completion_linked;
};

typedef struct ShadowSpillEventPoolBlock {
    ShadowSpillEventLease *leases;
    uint64_t count;
    struct ShadowSpillEventPoolBlock *next;
} ShadowSpillEventPoolBlock;

/*
 * Owns reusable neutral event records. Backend event handles are still leased
 * through the synchronization backend, whose implementation may maintain its
 * own driver-event pool. Explicit cold-path reserve calls may grow this owner;
 * once reserved, a hot-path shortage fails instead of allocating from libc.
 */
struct ShadowSpillEventPool {
    pthread_mutex_t lock;
    ShadowSpillEventPoolBlock *blocks;
    ShadowSpillEventLease *free_head;
    uint64_t capacity;
    uint64_t available;
    uint64_t in_use;
    uint64_t peak_in_use;
    uint64_t growth_rejections;
    uint8_t initialized;
    uint8_t sealed;
};

typedef struct ShadowSpillCompletionStream {
    ShadowSpillBackendStream stream;
    ShadowSpillEventLease *head;
    ShadowSpillEventLease *tail;
    uint64_t next_poll_timestamp_ns;
    struct ShadowSpillCompletionStream *next;
} ShadowSpillCompletionStream;

typedef struct ShadowSpillCompletionTracker {
    pthread_mutex_t lock;
    ShadowSpillCompletionStream *streams;
    uint64_t pending;
} ShadowSpillCompletionTracker;

/*
 * Owns only the quiescence notification consumed by runtime_wait_idle. The
 * final action and retirement transitions advance this epoch after their
 * counters reach zero. It deliberately does not drive the worker thread or
 * alter transfer/retirement dispatch cadence.
 */
typedef struct ShadowSpillIdleWakeup {
    pthread_mutex_t lock;
    pthread_cond_t condition;
    uint64_t epoch;
    uint8_t initialized;
} ShadowSpillIdleWakeup;

int shadowspill_idle_wakeup_initialize(
    ShadowSpillIdleWakeup *wakeup
);

void shadowspill_idle_wakeup_destroy(ShadowSpillIdleWakeup *wakeup);

void shadowspill_idle_notify(ShadowSpillRuntime *runtime);

int shadowspill_event_pool_initialize(ShadowSpillEventPool *pool);
void shadowspill_event_pool_destroy(
    ShadowSpillRuntime *runtime,
    ShadowSpillEventPool *pool
);
ShadowSpillRuntimeStatus shadowspill_event_pool_reserve(
    ShadowSpillEventPool *pool,
    uint64_t minimum_free_leases
);

ShadowSpillRuntimeStatus shadowspill_event_lease_create_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillEventLease **lease
);
void shadowspill_event_lease_retain(ShadowSpillEventLease *lease);
int shadowspill_event_lease_is_complete(const ShadowSpillEventLease *lease);
int shadowspill_event_lease_release(
    ShadowSpillRuntime *runtime,
    ShadowSpillEventLease *lease
);
int shadowspill_event_lease_query(
    ShadowSpillRuntime *runtime,
    ShadowSpillEventLease *lease,
    int *complete
);

int shadowspill_completion_tracker_initialize(
    ShadowSpillCompletionTracker *tracker
);
void shadowspill_completion_tracker_destroy(
    ShadowSpillRuntime *runtime,
    ShadowSpillCompletionTracker *tracker
);
ShadowSpillRuntimeStatus shadowspill_completion_submit(
    ShadowSpillRuntime *runtime,
    ShadowSpillBackendStream stream,
    ShadowSpillEventLease *event,
    uint64_t object_id,
    uint64_t allocation_id
);
int shadowspill_completion_poll(
    ShadowSpillRuntime *runtime,
    uint64_t *next_poll_nanoseconds,
    uint64_t *failure_object_id,
    uint64_t *failure_allocation_id
);

#endif
