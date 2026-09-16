/* Tracing, waiting, recovery, and what a reader inspects afterwards. */

#ifndef SHADOWSPILL_RUNTIME_TELEMETRY_H
#define SHADOWSPILL_RUNTIME_TELEMETRY_H

#include <stdint.h>

#include <shadowspill/shadowspill.h>
#include <shadowspill/backend.h>
#include <shadowspill/runtime/timing.h>
#include <shadowspill/runtime/vocabulary.h>
#include <shadowspill/runtime/descriptions.h>
#include <shadowspill/runtime/diagnostics.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------------
 * Telemetry and tracing
 *
 * Two bounded rings, both off by default and both diagnostic: a step
 * never depends on either, and a full ring stops recording rather
 * than stopping the runtime.
 */

/*
 * Starts one bounded allocation-lifetime capture. Storage is allocated before
 * capture begins, so allocator callbacks only append fixed-size records. A
 * full buffer stops recording and counts what it could not keep, so the gap
 * is visible in the read-back rather than silently lost; the step itself is
 * never affected.
 */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_allocation_telemetry_start(
    ShadowSpillRuntime *runtime,
    uint64_t capacity
);

/* Stops capture. Events already recorded stay readable until the next start. */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_allocation_telemetry_stop(ShadowSpillRuntime *runtime);

/*
 * Copies the complete ordered event stream. Pass events=NULL and capacity=0
 * to query count. Caller owns the destination and no runtime pointer escapes.
 */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_allocation_telemetry_read(
    ShadowSpillRuntime *runtime,
    ShadowSpillAllocationEvent *events,
    uint64_t capacity,
    uint64_t *count
);

/*
 * Planning-only allocation of reusable trace buffers. Calling this does not
 * enable tracing. Growth is rejected while a trace or allocation-profile
 * session is active; no trace buffer grows from a runtime hot path.
 */
SHADOWSPILL_API ShadowSpillStatus shadowspill_trace_prepare(
    ShadowSpillRuntime *runtime,
    const ShadowSpillTraceConfig *config
);

/*
 * Begins one prepared trace and its allocation-lifetime capture.
 *
 * ``origin`` is a caller-held marker already recorded on the caller's compute
 * stream (see runtime/timing.h); transfer intervals in the trace are measured
 * from it, so they share the caller's timeline. It must outlive the trace and
 * is never released here. A null marker records no stream intervals.
 */
SHADOWSPILL_API ShadowSpillStatus shadowspill_trace_begin(
    ShadowSpillRuntime *runtime,
    uint64_t step_id,
    const ShadowSpillTimingMarker *origin
);

/*
 * Stops appending without synchronizing a stream or worker. Callers establish
 * their required completion boundary before ending the trace.
 */
SHADOWSPILL_API ShadowSpillStatus shadowspill_trace_end(
    ShadowSpillRuntime *runtime
);

/*
 * Copies one stopped trace into caller-owned arrays. NULL arrays with zero
 * capacities query the required counts through summary. No runtime pointer or
 * backend handle escapes.
 */
SHADOWSPILL_API ShadowSpillStatus shadowspill_trace_read(
    ShadowSpillRuntime *runtime,
    ShadowSpillTraceSummary *summary,
    ShadowSpillTraceEvent *events,
    uint64_t event_capacity,
    ShadowSpillAllocationEvent *allocation_events,
    uint64_t allocation_event_capacity
);

/* ------------------------------------------------------------------------
 * Waiting, recovery and inspection
 *
 * Draining outstanding work, recovering from a no-progress stall,
 * growing a pool, and asking what happened.
 */

/* Blocks until no action is queued and no retirement is pending, then returns
   the first latched failure. */
SHADOWSPILL_API ShadowSpillStatus shadowspill_runtime_wait_idle(
    ShadowSpillRuntime *runtime
);

/*
 * Clears a latched NO_PROGRESS allocation failure after every external
 * producer stream has been synchronized and the failed allocator caller has
 * returned. This exists only for deterministic fault teardown: it allows the
 * worker to drain already-owned actions so objects and pool leases can be
 * reclaimed. Every other failure remains permanently latched.
 */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_runtime_recover_no_progress(ShadowSpillRuntime *runtime);

/*
 * Planning-only growth of the pool `pool_id` names. The runtime must be idle
 * and hold no in-flight actions or pending retirements; existing object
 * offsets and payloads are preserved, and shrinkage is rejected. The old and
 * new arenas are both held while the payload is copied across, so a caller
 * must include that transient in its own budget.
 */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_memory_pool_grow(
    ShadowSpillRuntime *runtime,
    uint32_t pool_id,
    uint64_t capacity_bytes
);

/* Copies a lock-consistent telemetry snapshot into caller-owned storage. */
SHADOWSPILL_API ShadowSpillStatus shadowspill_runtime_statistics(
    ShadowSpillRuntime *runtime,
    ShadowSpillRuntimeStatistics *statistics
);

/* What the pool `pool_id` names holds. Unknown pool ids are INVALID_ARGUMENT. */
SHADOWSPILL_API ShadowSpillStatus shadowspill_memory_pool_statistics(
    ShadowSpillRuntime *runtime,
    uint32_t pool_id,
    ShadowSpillMemoryPoolStatistics *statistics
);

/*
 * Copies one entry per live allocation of the pool `pool_id` names into
 * caller-owned storage, in no particular order, and always writes the total
 * live count to `count` so a caller can size its buffer and call again.
 *
 * Statistics answer how many allocations are live; this answers which. A
 * fixed layout refused for want of a contiguous range is not explained by a
 * count, because a few bytes in the wrong place cost the largest free range
 * while leaving the free total almost untouched -- so the offsets are the
 * diagnosis.
 *
 * Writes min(capacity, live) entries and returns OK even when the buffer was
 * too small; `*count` greater than `capacity` is how truncation is reported.
 * `out` may be NULL when `capacity` is zero, which asks only for the count.
 */
SHADOWSPILL_API ShadowSpillStatus shadowspill_memory_pool_live_allocations(
    ShadowSpillRuntime *runtime,
    uint32_t pool_id,
    ShadowSpillLiveAllocation *out,
    uint64_t capacity,
    uint64_t *count
);

/* Copies the immutable first-failure snapshot; status is OK before failure. */
SHADOWSPILL_API ShadowSpillStatus shadowspill_runtime_failure(
    ShadowSpillRuntime *runtime,
    ShadowSpillRuntimeFailure *failure
);

/* Copies one object's current logical state into caller-owned storage. */
SHADOWSPILL_API ShadowSpillStatus shadowspill_object_snapshot(
    ShadowSpillRuntime *runtime,
    uint64_t object_id,
    ShadowSpillObjectSnapshot *snapshot
);

/* Copies one object's current location in an explicitly selected pool. */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_object_location_snapshot(
    ShadowSpillRuntime *runtime,
    uint64_t object_id,
    uint32_t pool_id,
    ShadowSpillObjectLocationSnapshot *snapshot
);

/* One sentence for any reason; see shadowspill_status_string() for the status. */
SHADOWSPILL_API const char *shadowspill_failure_reason_string(
    ShadowSpillFailureReason reason
);

#ifdef __cplusplus
}
#endif

#endif
