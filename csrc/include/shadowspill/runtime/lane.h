/* What moves bytes between two pools, and how the runtime finds one. */

#ifndef SHADOWSPILL_RUNTIME_LANE_H
#define SHADOWSPILL_RUNTIME_LANE_H

#include <stdint.h>

#include <shadowspill/shadowspill.h>
#include <shadowspill/backend.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------------
 * Lanes
 *
 * A lane moves bytes between two pools and says when they have landed.
 *
 * A route names a source and a destination pool; their kinds are a directional
 * pair, and that pair selects the lane. There is no branch anywhere asking what
 * kind of transport a route has, and a lane a loaded library registers is found
 * exactly the way a built-in one is.
 *
 * A lane is named for the pool kinds it connects, never for an argument it
 * takes. The host-device lane takes a stream; so will a peer lane. The stream
 * distinguishes nothing.
 *
 * Two neighbours it is not. The *queue* in front of it orders the actions
 * issued on a route and never touches a backend. The route's *stream* carries
 * the events that order a transfer against compute, whether or not the bytes
 * moved on it, and the runtime is its only writer -- the completion tracker,
 * retirement and readiness publication all depend on that. A lane gets a stream
 * of its own to work on and leaves the route's alone.
 */

/* One lane, made per route at create. What it holds is its own business. */
typedef struct ShadowSpillLane ShadowSpillLane;
typedef struct ShadowSpillRuntime ShadowSpillRuntime;

/* Runtime-internal. A lane passes one through to the interval entries below
   and never looks inside, which is why those two entries are the only ones a
   lane outside this library cannot implement. */
typedef struct ShadowSpillStreamInterval ShadowSpillStreamInterval;

/*
 * An event here is a ShadowSpillBackendEvent -- one opaque word -- and never
 * the runtime's event lease, which is refcounting and pool links a lane has no
 * business seeing.
 *
 * THE OBLIGATION. A lane makes the event it was given complete when the bytes
 * have landed, and **the runtime does not drive it**. How it arranges that is
 * its own business: a lane whose copies run on a stream lets the device order
 * the event behind them, and a lane that completes on its own schedule watches
 * for that however it likes -- a thread of its own, blocking on whatever its
 * transport offers -- and then releases the event. Nothing downstream can tell
 * the difference, because everything downstream reads an event.
 *
 * That is why there is no entry here for the runtime to poke a lane with. There
 * was one, and it cost 1.7 % of the shortest step in the qualification matrix
 * while doing nothing: the worker's loop gates every transfer, so work added
 * there is paid at whatever rate the loop happens to turn.
 *
 * The two intervals may be NULL; everything else is required. The rule is the
 * one the backend table already follows for its profiler entries: required when
 * the runtime cannot proceed without it, optional when its absence costs
 * nothing the runtime needs. A lane with no intervals moves bytes exactly as
 * well as one with them.
 */
typedef struct ShadowSpillLaneOperations {
    /*
     * Order this lane's work behind `event`. Returns 0 when the dependency is
     * enqueued or already satisfied, 1 when it cannot be enqueued yet and the
     * caller should retry at the next poll, and -1 on failure.
     *
     * This orders the lane's own stream, not the route's: a transfer whose
     * copy runs on the lane's stream still has to wait for whatever protects
     * its source or destination. The retry return exists for a lane that
     * cannot enqueue a device dependency at all and has to observe the event
     * completing instead; the action stays at its queue's pending head and the
     * existing cadence (`worker_poll_nanoseconds`) revisits it. A lane whose
     * waits are device-side never returns 1.
     */
    int (*wait)(ShadowSpillLane *lane, ShadowSpillBackendEvent event);

    /* Move `bytes` from `source` to `destination`, both addresses in the pools
       this lane connects. Nothing but the lane dereferences either. */
    int (*copy)(
        ShadowSpillLane *lane,
        void *destination,
        const void *source,
        uint64_t bytes
    );

    /*
     * Make `event` complete once everything issued on this lane so far has
     * landed. For a lane whose bytes move on a stream that is a recorded
     * event; for one that completes on its own schedule it is whatever holds a
     * stream until the lane says the bytes are there. Downstream sees an
     * ordinary backend event either way, which is what keeps the completion
     * tracker and retirement in one form.
     */
    int (*signal)(ShadowSpillLane *lane, ShadowSpillBackendEvent event);

    /* Block until everything issued on this lane has landed. */
    int (*synchronize)(ShadowSpillLane *lane);

    /* Optional, as a pair. Bracket the next copy so it can be timed. NULL when
       this lane cannot place an instant on the trace's clock, and its transfers
       are then recorded untimed rather than timed wrongly. */
    int (*interval_open)(
        ShadowSpillLane *lane, ShadowSpillStreamInterval *interval
    );
    int (*interval_close)(
        ShadowSpillLane *lane, ShadowSpillStreamInterval *interval
    );

    void (*destroy)(ShadowSpillLane *lane);
} ShadowSpillLaneOperations;

/*
 * One entry in the runtime config's `lanes`: the directional pool-kind pair
 * this lane serves, and how to make one. Two entries claiming the same pair
 * fails create -- order must never decide it silently.
 */
typedef struct ShadowSpillLaneDescription {
    /* ShadowSpillPoolKind values, source then destination. */
    uint8_t from_kind;
    uint8_t to_kind;
    const ShadowSpillLaneOperations *operations;
    /*
     * Made once per route at create, released in reverse at close.
     *
     * `runtime` is the lane's way back in. A lane that discovers a failure on a
     * thread of its own has no return value to fail through -- the runtime is
     * not calling it -- so it latches the failure itself. The latch is built for
     * concurrent callers and first writer wins, so the original cause survives,
     * and the worker already reads the latched status twice a turn: no new
     * mechanism and nothing added to the loop.
     *
     * `stream` is created by the runtime for this lane and is not the route's
     * stream. A lane may use `backend` through its own stream -- to copy, to
     * record, to query -- but never on the route's, which has one writer.
     *
     * This is also where a lane probes. Nothing a loaded library holds runs at
     * load, so everything that depends on what the hardware can actually do
     * happens here, where there is a failure path and an unwind.
     */
    int (*create)(
        ShadowSpillRuntime *runtime,
        const ShadowSpillBackend *backend,
        ShadowSpillBackendStream stream,
        void *configuration,
        ShadowSpillLane **lane
    );
    /* Passed to `create` untouched. Whoever registers the entry decides what
       it points at. */
    void *configuration;
} ShadowSpillLaneDescription;

#ifdef __cplusplus
}
#endif

#endif /* SHADOWSPILL_RUNTIME_LANE_H */
