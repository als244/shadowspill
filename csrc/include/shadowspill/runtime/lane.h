/* What moves bytes between two pools, and how the runtime finds one. */

#ifndef SHADOWSPILL_RUNTIME_LANE_H
#define SHADOWSPILL_RUNTIME_LANE_H

#include <stdint.h>

#include <shadowspill/shadowspill.h>
#include <shadowspill/backend.h>
/* For the status and reason a lane latches with. */
#include <shadowspill/runtime/vocabulary.h>

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

/*
 * How a lane reports a failure it finds on a thread of its own.
 *
 * The runtime is not calling it, so there is no return value to fail through.
 * This is the way back in, and the reason `create` receives a runtime at all.
 *
 * Safe from any thread and first-writer-wins, so the original cause survives
 * whatever it goes on to cause. It records no task or pool: a lane's thread is
 * inside neither, and attributing its failure to whatever the dispatching
 * thread happened to be doing would be worse than leaving it blank.
 *
 * The worker already reads the latched status twice a turn, so this costs
 * nothing in the loop and arrives by the path every other failure takes.
 */
/* Whether a trace is running, for a lane deciding whether to record the detail
   a trace wants. A lane is outside the runtime and cannot read its state; this
   is the one question it needs answered, and asking is cheaper than recording
   what nothing will read. */
SHADOWSPILL_API int shadowspill_lane_trace_active(ShadowSpillRuntime *runtime);

SHADOWSPILL_API void shadowspill_lane_latch_failure(
    ShadowSpillRuntime *runtime,
    ShadowSpillStatus status,
    ShadowSpillFailureReason reason
);

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
 * THE SECOND OBLIGATION, and the one that is easy to breach without noticing:
 *
 *     A lane's completion path must not depend on anything the lane has made
 *     wait.
 *
 * Stated about cycles rather than threads, because a lane need not have one.
 * The built-in satisfies it without trying: its completion path is the stream,
 * the driver advances it, and it makes nothing wait.
 *
 * A lane that completes on its own schedule has to be deliberate. Its value
 * wait is satisfied only by that completion path, so any device call on the
 * path can be blocked by the very wait it exists to satisfy -- and an
 * outstanding value wait blocks calls on *other* streams too, so a second
 * stream is not an escape. The practical form: whatever watches for
 * completions does that and nothing else, and any device work the transfer
 * needs is issued by the thread that called into the lane, before the watcher
 * ever sees it.
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
/*
 * What a lane has moved, and what it cost.
 *
 * One lane is one directional pair of pool kinds, so these are per route and
 * per direction. Bytes are what the runtime asked to move: a lane that splits
 * a transfer into chunks counts the transfer once in `copies` and its pieces
 * in `chunks`, which is the difference between "how many transfers" and "how
 * many times the hardware was asked".
 *
 * `posted_to_completion_seconds` is summed rather than averaged so the mean is
 * a division the reader does, and no sample is thrown away deciding what to
 * keep. It measures the interval a lane can actually see -- from handing the
 * hardware a chunk to observing its completion -- which is not the same as
 * time on the wire, and separating the two is the point.
 */
typedef struct ShadowSpillLaneStatistics {
    uint64_t copies;      /* transfers accepted */
    uint64_t chunks;      /* pieces the hardware was handed */
    uint64_t bytes;       /* bytes accepted, summed over copies */
    uint64_t signals;     /* completion signals issued */
    uint64_t waits;       /* dependency waits enqueued */
    uint64_t retries;     /* waits that asked to be retried */
    uint64_t failures;    /* transfers that did not land */

    /* Zero unless `timed` is 1. */
    uint8_t timed;
    double posted_to_completion_seconds;
    double longest_completion_seconds;
} ShadowSpillLaneStatistics;

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

    /*
     * What this lane has moved. **Optional**: NULL means the lane keeps no
     * count, and the runtime reports nothing for it rather than reporting
     * zeroes, which would be indistinguishable from a lane that moved nothing.
     *
     * Counters are maintained unconditionally, because a lane that only counts
     * when asked cannot explain the run that went wrong. The timing fields are
     * the exception and may be left zero by a lane for which reading a clock
     * on the transfer path is not free; `timed` says which it is, so a reader
     * never mistakes "not measured" for "instant".
     *
     * Called from the thread collecting diagnostics, never from the worker,
     * and must be safe against a lane actively transferring.
     */
    int (*statistics)(
        const ShadowSpillLane *lane,
        ShadowSpillLaneStatistics *statistics
    );
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
     * `stream` is **the route's stream** (`runtime.c` passes `route->stream`
     * here), and a lane may use `backend` through it -- to copy, to record, to
     * wait, to query.
     *
     * The rule it has to keep is single-writer ordering: one thread's worth of
     * work, in one order. That is satisfied for free by every entry in this
     * table, because each is called on the thread that called into the lane.
     * It is *not* satisfied by a thread the lane runs itself, which is
     * concurrent with the next call in -- so a lane's own thread writes no
     * stream at all. See the second obligation above for why that is a
     * deadlock and not merely a race.
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
