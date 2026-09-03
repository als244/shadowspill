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
    /* The backend event is created once and kept across leases. */
    uint8_t has_event;
    ShadowSpillEventPool *pool;
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
    /* Backend events created for this pool's leases. */
    uint64_t driver_creates;
    uint8_t initialized;
    uint8_t sealed;
    /* Whether this pool's events carry device timestamps. */
    uint8_t timing;
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

int shadowspill_event_pool_initialize(ShadowSpillEventPool *pool, uint8_t timing);
void shadowspill_event_pool_destroy(
    ShadowSpillRuntime *runtime,
    ShadowSpillEventPool *pool
);
ShadowSpillStatus shadowspill_event_pool_reserve(
    ShadowSpillRuntime *runtime,
    ShadowSpillEventPool *pool,
    uint64_t minimum_free_leases
);

ShadowSpillStatus shadowspill_event_lease_create_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillEventLease **lease
);

/* A lease from any pool of the runtime: the dependency pool by default, the
   timing pool for stream intervals. */
ShadowSpillStatus shadowspill_event_lease_acquire(
    ShadowSpillRuntime *runtime,
    ShadowSpillEventPool *pool,
    ShadowSpillEventLease **output
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
ShadowSpillStatus shadowspill_completion_submit(
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


/*
 * One interval of work on a backend stream, bracketed by two timing events
 * and read as nanoseconds from an origin event on the same device timeline.
 *
 * Open records the start on the stream, close records the end, read measures
 * both from the origin once the work has completed, and discard returns the
 * events to the backend. An interval that fails to open stays empty, and
 * every operation on an empty interval is a no-op, so a caller that traces
 * conditionally never branches on failure. The events come from the
 * synchronization backend's timing pool and are never used for dependencies.
 */
typedef struct ShadowSpillStreamInterval {
    ShadowSpillEventLease *start;
    ShadowSpillEventLease *end;
    uint8_t open;
} ShadowSpillStreamInterval;

int shadowspill_stream_interval_open(
    ShadowSpillRuntime *runtime,
    ShadowSpillStreamInterval *interval,
    ShadowSpillBackendStream stream
);
int shadowspill_stream_interval_close(
    ShadowSpillRuntime *runtime,
    ShadowSpillStreamInterval *interval,
    ShadowSpillBackendStream stream
);
/* 0 with both times on success; 1 while the work is still running; -1 when
 * the interval is empty or the backend cannot measure it. */
int shadowspill_stream_interval_read(
    ShadowSpillRuntime *runtime,
    const ShadowSpillStreamInterval *interval,
    ShadowSpillBackendEvent origin,
    uint64_t *start_ns,
    uint64_t *end_ns
);
void shadowspill_stream_interval_discard(
    ShadowSpillRuntime *runtime,
    ShadowSpillStreamInterval *interval
);

#endif
