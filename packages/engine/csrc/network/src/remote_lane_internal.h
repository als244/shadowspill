/* The remote lane's private shape, shared by its three files. */

#ifndef SHADOWSPILL_NETWORK_REMOTE_LANE_INTERNAL_H
#define SHADOWSPILL_NETWORK_REMOTE_LANE_INTERNAL_H

#include "../internal.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <time.h>

/*
 * Three files, one lane.
 *
 *   remote_lane.c           the lane: a transfer's life from `copy` to the
 *                           event, posted straight from the pool's memory --
 *                           the direct path, which is the norm
 *   remote_lane_staging.c   the fallback when the NIC cannot address the
 *                           pool: a host ring, and the device copies that
 *                           move each piece through it
 *   remote_lane_timeline.c  the measuring instrument behind
 *                           SHADOWSPILL_NETWORK_MEASURE, which records every
 *                           instant of a transfer that had the lane to itself
 *
 * Everything the three share is here, and nothing else includes this.
 */

/*
 * Three words the lane, the device and the route stream pass counts through.
 * Every one is written by exactly one side and read by the other, and every
 * one rises monotonically over the lane's whole life, which is why three
 * words serve any number of transfers.
 *
 *   SIGNAL_NIC     stored by the lane's thread as pieces land -- a fetch's in
 *                  the pool or the ring, an evict's at the peer. The route
 *                  stream waits on it before a staged copy and before the
 *                  event `signal` records.
 *   SIGNAL_DEVICE  staging only: the route stream stores it as the device
 *                  finishes with a ring slot; the thread reads it.
 *   SIGNAL_GATE    the direct path only: the route stream stores a transfer's
 *                  number once it has passed the waits the runtime enqueued
 *                  for the transfer; the thread posts nothing before it.
 *
 * THE THREAD MAKES NO DEVICE CALL, and this is a rule rather than a habit. A
 * launch that finds its stream's queue full blocks holding the driver's lock,
 * and it stays blocked until the device makes progress -- which, when the
 * device is waiting on one of these words, is the thread's doing. A thread
 * that then needed the driver to store the word would be waiting for the
 * lock the launch holds: a deadlock. So the
 * device's side of a transfer is enqueued at dispatch by the runtime's
 * worker, behind value waits on these words, and the thread's whole part is
 * memory: the completion queue, the posts, and these stores.
 */
#define SIGNAL_NIC 0U
#define SIGNAL_DEVICE 1U
#define SIGNAL_GATE 2U
#define SIGNAL_WORDS 3U

/*
 * How many transfers the lane can hold, and how many pieces it can have posted
 * at once. Both far larger than a route has in flight; `copy` waits rather
 * than fails if the first ever fills, and the second bounds the queue pair's
 * depth the endpoint was built with.
 */
#define WORK_SLOTS 256U
#define PIECES_IN_FLIGHT 128U

/* The instants one transfer passes through; see remote_lane_timeline.c. */
enum {
    /* The worker's chain, in its order. */
    TIMELINE_ENTERED,        /* `copy` called                                */
    TIMELINE_PUBLISHED,      /* the slot is visible to the thread            */
    TIMELINE_ENQUEUED,       /* the device's side is on the route stream     */
    TIMELINE_SYNC_ENTERED,   /* `synchronize` called                         */
    TIMELINE_SYNC_DRAINED,   /* it observed the thread's retirement          */
    TIMELINE_DEVICE_SEEN,    /* it observed the device's last store          */
    TIMELINE_SYNC_RETURNED,  /* the stream is synchronized                   */
    /* The thread's chain, in its order. */
    TIMELINE_SEEN,           /* the thread first looked at it                */
    TIMELINE_READY,          /* its first piece may be posted                */
    TIMELINE_POSTED,         /* `ibv_post_send` returned for its first piece */
    TIMELINE_COMPLETED,      /* its last completion was polled off the queue */
    TIMELINE_REPORTED,       /* the NIC's word stored for that piece         */
    TIMELINE_RETIRED,        /* retired under the lock, waiters woken        */
    TIMELINE_POINTS
};

#define TIMELINE_ROWS 64U

typedef struct TimelineRow {
    double at[TIMELINE_POINTS];
    uint64_t bytes;
    uint64_t first_chunk;
} TimelineRow;

/*
 * One transfer, as the lane's thread sees it.
 *
 * Its pieces are numbered over the lane's whole life -- this transfer owns
 * `[first_chunk, first_chunk + chunks)` -- so both signal words count on one
 * scale and a reader never has to ask which transfer a number belongs to.
 */
typedef struct Work {
    /* The end that is not remote: memory in the pool this lane connects to
       the peer's, which the NIC posts directly or the ring stages. */
    void *local;
    uint64_t remote_offset;
    uint64_t bytes;
    uint8_t to_remote;
    uint32_t chunks;
    uint64_t first_chunk;

    /*
     * Which transfer owns this slot, counted from 1 over the lane's life, and
     * 0 while nothing does. It is the handle `copy` hands back, and what lets
     * a reader tell "the slot still holds my transfer" from "the slot has
     * been taken by a later one" without a lock.
     */
    _Atomic uint64_t handle;

    /* Instants on the host clock for a trace that asked; see `transfer`.
       Filled only when `traced`. */
    uint8_t traced;
    uint64_t issued_host_ns;
    uint64_t started_host_ns;
    uint64_t finished_host_ns;

    /* Which timeline row this transfer stamps, or -1 when it is not captured.
       Set before the slot is published, so both threads read the same row. */
    int32_t timeline_row;
} Work;

typedef struct RemoteLane {
    /* Runtime, backend, the route's stream, this lane's two kinds, where its
       two pools live, and the seven counters every lane keeps. Filled by the
       runtime and copied in by `create`. */
    ShadowSpillLane base;
    ShadowSpillNetworkTuning tuning;

    /*
     * The one remote pool this route reaches, the queue pair this lane took
     * from it, and the pieces the port will carry in one work request.
     * A route joins a device or host pool to a remote one, so a lane serves
     * exactly one region and never has to ask which.
     */
    const ShadowSpillRemoteRegion *region;
    int queue_pair;
    uint64_t piece_bytes;

    /*
     * The NIC's key for the local pool, when the NIC can address it -- the
     * direct path -- and NULL when it cannot, in which case every byte passes
     * through the ring below and `stages` is set. Decided once, at create.
     */
    struct ibv_mr *pool_registration;
    uint8_t stages;

    /* The three words; see the top of this header. */
    ShadowSpillBackendSignals signals;
    uint64_t *signal_host;

    /* Pieces planned over this lane's life. Written and read under the lane's
       lock by whoever calls the lane, never by its thread. */
    uint64_t chunks_planned;

    /*
     * THE PIECE PIPELINE, touched only by the lane's thread. `chunk_posted`
     * and `chunk_retired` keep rising across transfers; retirement is in
     * order; `chunk_landed` records completions that arrived ahead of their
     * turn.
     */
    uint64_t chunk_posted;
    uint64_t chunk_retired;
    uint8_t chunk_landed[PIECES_IN_FLIGHT];
    /* When each in-flight piece was posted, for `timing`. Measuring only. */
    double chunk_posted_at[PIECES_IN_FLIGHT];

    /* Staging's ring: host memory registered with the backend and the NIC,
       `ring_slots` pieces of `chunk_bytes`. Unused on the direct path. */
    void *ring;
    uint64_t ring_bytes;
    struct ibv_mr *ring_registration;

    /*
     * Whether to read the clock on the transfer path: for the per-piece
     * interval `timing` reports, the completion-queue counts, and the
     * timeline. `SHADOWSPILL_NETWORK_MEASURE` sets it.
     */
    uint8_t measuring;
    _Atomic uint64_t stat_completion_micros;
    _Atomic uint64_t stat_longest_micros;
    _Atomic uint64_t stat_timed;
    uint64_t polls;
    uint64_t polls_empty;

    /* The captured transfers and what decides whether the next is captured;
       see remote_lane_timeline.c. Touched only by the calling thread. */
    TimelineRow timeline[TIMELINE_ROWS];
    uint32_t timeline_rows;
    uint64_t synchronized_through;
    uint64_t device_watch_timeouts;

    /* Non-zero while work is queued, readable without the lock so the thread
       can watch for it while spinning. */
    atomic_ullong queued;
    /*
     * The ring of transfers, and three counts that never wrap: accepted, taken
     * by the thread, and retired by it. Slot k of a count is `k % WORK_SLOTS`.
     * `synchronize` waits on `retired`: the thread takes a transfer before
     * doing it, so `started == accepted` says only that nothing is waiting.
     */
    Work work_ring[WORK_SLOTS];
    uint64_t accepted;
    uint64_t started;
    uint64_t retired;

    pthread_t thread;
    pthread_mutex_t lock;
    pthread_cond_t work_ready;
    pthread_cond_t drained;
    atomic_uint stopping;
    atomic_uint failed;
    uint8_t thread_started;
} RemoteLane;

/* ----------------------------------------------------------------- clocks */

static inline double remote_lane_seconds_now(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
}

/* The clock a transfer's instants are read on, and the one the runtime stamps
   its trace events with -- so `shadowspill_lane_origin_instant` can place them
   on the origin's axis. */
static inline uint64_t remote_lane_monotonic_ns(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (uint64_t)now.tv_sec * 1000000000U + (uint64_t)now.tv_nsec;
}

/* A read of one of the words, never a device call. */
static inline uint64_t remote_lane_word(const RemoteLane *lane, uint32_t index) {
    const _Atomic uint64_t *const word =
        (const _Atomic uint64_t *)&lane->signal_host[index];
    return atomic_load_explicit(word, memory_order_acquire);
}

/* Has the device finished with pieces below `reached`? */
static inline int remote_lane_device_reached(
    const RemoteLane *lane, uint64_t reached
) {
    return remote_lane_word(lane, SIGNAL_DEVICE) >= reached ? 1 : 0;
}

/* ------------------------------------------------------------ the lane's */

/* The transfer an outstanding piece belongs to, scanned from the oldest
   unretired transfer; NULL for a piece nobody holds. remote_lane.c. */
Work *remote_lane_owner_of_chunk(RemoteLane *lane, uint64_t chunk);

/* ---------------------------------------------------------------- staging */

/* Take the ring, and register it with the backend and with the region's
   protection domain. Called once at create, when the NIC could not address
   the pool. */
int remote_lane_stage_create(RemoteLane *lane);
void remote_lane_stage_destroy(RemoteLane *lane);

/* Where piece `chunk` lives in the ring. */
void *remote_lane_stage_slot(const RemoteLane *lane, uint64_t chunk);

/* The device's side of one transfer, all of it, on the route stream: called
   by `copy` on the runtime's worker, after the slot is published and from a
   copy of the plan. -1 on a device call failing. */
int remote_lane_stage_enqueue(RemoteLane *lane, const Work *plan);

/* Whether the device has done its part for piece `chunk` of `work`, so the
   thread may post it: an evict's slot filled, a fetch's slot drained of its
   previous occupant. A read of the device's word, never a device call. */
int remote_lane_stage_ready(const RemoteLane *lane, const Work *work, uint64_t chunk);

/* --------------------------------------------------------------- timeline */

/* Stamp one instant of one transfer. A transfer that is not captured costs a
   comparison. The first stamp of a point stands. */
void remote_lane_stamp(RemoteLane *lane, const Work *work, unsigned point);

/* Give `work` a row if it has the lane to itself; under the lane's lock,
   before the slot is published. `entered` is when `copy` was called. */
void remote_lane_capture_timeline(RemoteLane *lane, Work *work, double entered);

/* The transfer a `synchronize` stamps: the one accepted since the last, when
   exactly one was and it was captured. Under the lock. */
const Work *remote_lane_captured_transfer(const RemoteLane *lane);

/* Release the row a batch's first transfer claimed and cannot complete. Under
   the lock, after the drain. */
void remote_lane_release_batch_row(RemoteLane *lane, uint64_t accepted);

/* Watch, from the host, for the device's last store for `work` -- a staged
   fetch's copy out of the ring -- and stamp it. Bounded. Stamps at once where
   there is no such store. */
void remote_lane_watch_device(RemoteLane *lane, const Work *work);

/* What measuring found, to stderr, at destroy. */
void remote_lane_report(const RemoteLane *lane);

#endif /* SHADOWSPILL_NETWORK_REMOTE_LANE_INTERNAL_H */
