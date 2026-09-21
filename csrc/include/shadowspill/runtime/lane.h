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

typedef struct ShadowSpillRuntime ShadowSpillRuntime;

/*
 * One lane, made per route at create.
 *
 * Opaque here. Every lane embeds a common struct as its first member, holding
 * the five things they all hold and the seven they all count, so a transport
 * writes neither -- but that layout is for code implementing a lane, and lives
 * in <shadowspill/runtime/lane_base.h>. A caller that declares a runtime needs
 * only the pointer.
 */
typedef struct ShadowSpillLane ShadowSpillLane;

/*
 * What one transfer did, for a trace that asked.
 *
 * `copy` hands back a handle naming the transfer and the runtime gives it back
 * here once the transfer has completed, which is the only way a lane's own
 * view reaches the trace: a lane is outside the runtime and cannot see one.
 *
 * All three instants are **nanoseconds from the trace's origin**, the same axis
 * the rest of a step is placed on, and `SHADOWSPILL_LANE_NO_TIME` where a lane
 * has nothing to report -- which `bytes` and `chunks` are still worth reporting
 * beside, and are what a lane always knows.
 *
 * `issued_at` is when the runtime handed the transfer over, before any
 * dependency the lane was given had cleared. `started_at` is when its bytes
 * began moving. **The gap between them is the wait**, which is why they are
 * separate: folded together, a transfer held behind an event reads as a slow
 * one.
 *
 * Two clocks can reach this axis. A lane whose bytes move on a stream reads
 * instants off timing events it recorded around the copy, already on the
 * origin's axis. A lane whose bytes move elsewhere reads a host clock and
 * converts through the anchor the runtime records beside the origin event --
 * see `shadowspill_lane_origin_instant`. A lane may use both, and the
 * pinned-host lane does: its `issued_at` is a converted host instant and its
 * other two come off the stream. That costs it nothing per transfer, where a
 * third timing event would have.
 *
 * Converting needs a trace with an origin. Without one there is no axis to be
 * on, and every converted instant is `SHADOWSPILL_LANE_NO_TIME`.
 */
#define SHADOWSPILL_LANE_NO_TIME UINT64_MAX

typedef struct ShadowSpillLaneTransfer {
    uint64_t issued_at_nanoseconds;
    uint64_t started_at_nanoseconds;
    uint64_t finished_at_nanoseconds;
    uint64_t bytes;
    uint64_t chunks;
} ShadowSpillLaneTransfer;

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

/* Places a host instant on the trace origin's axis, for a lane whose bytes do
   not move on a stream and which therefore has only its own clock.

   `monotonic_nanoseconds` is read from CLOCK_MONOTONIC, the same clock the
   runtime stamps its own trace events with.
   Returns `SHADOWSPILL_LANE_NO_TIME` when no trace is running, when the trace
   was begun with no origin, or when the instant falls before the origin --
   which a transfer issued before the trace began legitimately does.

   The anchor is sampled once, where the origin event is recorded, and
   `shadowspill_trace_begin` says what a caller owes for it to be worth
   anything. */
SHADOWSPILL_API uint64_t shadowspill_lane_origin_instant(
    ShadowSpillRuntime *runtime,
    uint64_t monotonic_nanoseconds
);

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
 * the difference, because everything downstream reads an event -- or asks the
 * lane, for one that answers `landed` and `order` below, which reach
 * downstream by the same two paths the event does.
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
 * `transfer`, `timing`, `landed` and `order` may be NULL; everything else is
 * required. The first two follow the rule the backend table already follows
 * for its profiler entries: required when the runtime cannot proceed without
 * it, optional when its absence costs nothing the runtime needs. A lane that
 * reports neither moves bytes exactly as well as one that reports both, and a
 * reader sees a transfer with no numbers rather than one with wrong ones. The
 * last two are optional for a different reason: the event `signal` was given
 * answers for them, and only a lane whose bytes do not move on a stream has
 * anything better to say.
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
 * **The runtime fills the seven counters itself**, from the ones every lane
 * keeps in its common struct. No transport copies them out, so a transport
 * cannot report a count that disagrees with the one it kept -- which is what
 * two hand-written copies of the same seven loads were free to do. The timing
 * pair is all a transport is asked for, through `timing` below.
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

/*
 * What a lane's own transfers cost it, where it can say.
 *
 * `posted_to_completion_seconds` is summed rather than averaged so the mean is
 * a division the reader does, and no sample is thrown away deciding what to
 * keep. It measures the interval a lane can actually see -- from handing the
 * hardware a chunk to observing its completion -- which is not the same as
 * time on the wire, and separating the two is the point.
 */
typedef struct ShadowSpillLaneTiming {
    double posted_to_completion_seconds;
    double longest_completion_seconds;
} ShadowSpillLaneTiming;

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

    /*
     * Move `bytes` from `source` to `destination`, both addresses in the pools
     * this lane connects. Nothing but the lane dereferences either.
     *
     * `handle` names the transfer, for the `signal`, `transfer`, `landed` and
     * `order` entries below. **Zero means the lane is keeping nothing about
     * it**, which is what a lane answers when no trace is running --
     * `shadowspill_lane_trace_active` is the question to ask -- so nothing is
     * recorded that nothing will read. A lane that provides `landed` or
     * `order` keeps a record of every transfer, and its handle is nonzero
     * whether or not a trace runs.
     */
    int (*copy)(
        ShadowSpillLane *lane,
        void *destination,
        const void *source,
        uint64_t bytes,
        uint64_t *handle
    );

    /*
     * Make `event` complete once everything issued on this lane so far has
     * landed. For a lane whose bytes move on a stream that is a recorded
     * event; for one that completes on its own schedule it is whatever holds a
     * stream until the lane says the bytes are there. Downstream sees an
     * ordinary backend event either way, which is what keeps the completion
     * tracker and retirement in one form.
     */
    int (*signal)(
        ShadowSpillLane *lane,
        uint64_t handle,
        ShadowSpillBackendEvent event
    );

    /* Block until everything issued on this lane has landed. */
    int (*synchronize)(ShadowSpillLane *lane);

    /*
     * What one transfer did. **Optional**: NULL means this lane reports
     * nothing per transfer, and the trace records those transfers with no
     * numbers rather than with wrong ones.
     *
     * Asked once, after the transfer's event has completed, and asked only for
     * a handle `copy` returned nonzero. The query **retires the handle**: a
     * lane may free whatever it kept the moment it answers, and the runtime
     * will not ask again. That is the whole lifetime rule, and it is why there
     * is no release entry beside it.
     *
     * This replaced a pair that bracketed the copy with timing events on a
     * stream, which only a lane whose bytes move on one could implement. A
     * lane that moves bytes some other way now reports what it actually
     * observed instead of reporting nothing.
     */
    int (*transfer)(
        ShadowSpillLane *lane,
        uint64_t handle,
        ShadowSpillLaneTransfer *transfer
    );

    void (*destroy)(ShadowSpillLane *lane);

    /*
     * What this lane's transfers cost it. **Optional**: NULL means the lane
     * reads no clock on the transfer path, and a reader sees `timed` 0 rather
     * than a zero duration -- otherwise "not measured" and "instant" are the
     * same number.
     *
     * The counters are not asked for here. They live in the lane's common
     * struct, are maintained unconditionally -- a lane that counts only when
     * asked cannot explain the run that went wrong -- and the runtime reads
     * them straight out.
     *
     * Called from the thread collecting diagnostics, never from the worker,
     * and must be safe against a lane actively transferring.
     */
    int (*timing)(
        const ShadowSpillLane *lane,
        ShadowSpillLaneTiming *timing
    );

    /*
     * The two questions the runtime asks about a transfer's completion, and
     * the event is the default answer to both.
     *
     * From the host: *has it landed?* -- what publishes residency, releases
     * leases, retires the transfer and records the trace. From a stream: *make
     * this stream wait for it* -- what lets a consumer be issued before the
     * transfer is done. The runtime answers both through the event `signal`
     * was given: it queries that event, and enqueues waits on it. That is exact
     * for a lane whose bytes move on a stream, because the event, recorded
     * behind the copy, *is* the completion.
     *
     * A lane whose bytes land some other way -- a thread reaping a NIC's
     * completions, a storage engine, a fabric with its own queue -- can make
     * such an event true early only by making a stream wait on something it
     * will store later, and a stream that waits is resumed on the device's own
     * schedule. So it may answer the two questions itself, and record the
     * event when the bytes have landed rather than before:
     *
     *     landed   answers the host's question for one transfer;
     *     order    makes `stream` wait until that transfer has landed.
     *
     * **Both optional, singly or as a pair; NULL means the event is
     * authoritative**, and the runtime's behaviour for that lane does not
     * change by a single call. A lane that provides either keeps a record of
     * every transfer -- its `copy` hands back a nonzero handle whether or not a
     * trace runs -- and answers -1 for a handle it no longer keeps, on which
     * the runtime uses the event, authoritative by then.
     *
     * The rule such a lane keeps: **it answers `landed` yes only after it has
     * recorded the event**, so that from that moment the event says the same
     * thing. Retirement of the handle is unchanged: the `transfer` query, or,
     * for a lane with no `transfer`, the first `landed` that answered yes.
     *
     * `order` writes `stream`, which is the caller's -- a consumer's stream,
     * or another route's -- on the thread that called; a lane's own stream is
     * never handed to another lane. Both are called from the worker, and
     * `order` also from a frontend thread acquiring an object, so both must be
     * safe against the lane's own thread.
     */
    int (*landed)(ShadowSpillLane *lane, uint64_t handle, int *landed);
    int (*order)(
        ShadowSpillLane *lane,
        uint64_t handle,
        ShadowSpillBackendStream stream
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
     * `base` is the common struct the runtime has already filled -- runtime,
     * backend, the route's stream, this entry's two kinds, counters at zero.
     * A transport allocates its own struct and copies it into the first
     * member:
     *
     *     created->base = *base;
     *
     * and everything the base holds is then available for the rest of create,
     * which is why it arrives this way rather than being filled afterwards.
     * The runtime built it, so no transport can fill it wrong, and adding a
     * field to the base changes no transport at all.
     *
     * `runtime` is the lane's way back in. A lane that discovers a failure on a
     * thread of its own has no return value to fail through -- the runtime is
     * not calling it -- so it latches the failure itself. The latch is built for
     * concurrent callers and first writer wins, so the original cause survives,
     * and the worker already reads the latched status twice a turn: no new
     * mechanism and nothing added to the loop.
     *
     * `stream` is **the route's stream**, and a lane may use `backend` through
     * it -- to copy, to record, to wait, to query.
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
        const ShadowSpillLane *base,
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
