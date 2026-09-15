/* Opening a runtime, opening a plan on it, and closing both. */

#ifndef SHADOWSPILL_RUNTIME_LIFECYCLE_H
#define SHADOWSPILL_RUNTIME_LIFECYCLE_H

#include <stdint.h>

#include <shadowspill/shadowspill.h>
#include <shadowspill/backend.h>
#include <shadowspill/runtime/vocabulary.h>
#include <shadowspill/runtime/descriptions.h>
#include <shadowspill/runtime/diagnostics.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------------
 * Runtime and plan lifecycle
 *
 * Create a runtime, reserve the records it must never allocate on a
 * hot path, open and close plans, and tear it all down.
 */

/* Whether a backend table carries the contract version and every required
   entry (the profiler entries may be NULL). */
SHADOWSPILL_API int shadowspill_backend_is_valid(const ShadowSpillBackend *backend);

/*
 * Creates one runtime from explicit pool and directed-route registries over
 * one backend. Registry entries and the backend table are copied; the provider
 * state the table names is borrowed and must outlive the runtime. Pool and
 * route IDs must equal their contiguous registry indices. On failure, output
 * is set to NULL and whatever was created is reclaimed.
 */
SHADOWSPILL_API ShadowSpillStatus shadowspill_runtime_create(
    const ShadowSpillRuntimeConfig *config,
    ShadowSpillRuntime **runtime
);

/*
 * Cold-path capacity reservation for neutral event records. Repeated calls
 * grow the pool for additional admitted plans, each waiting for an idle
 * boundary first. After the first call, steady execution never falls back to
 * process allocation when the pool is full.
 */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_runtime_reserve_event_leases(
    ShadowSpillRuntime *runtime,
    uint64_t minimum_free_leases
);

/*
 * Cold-path capacity reservation for immutable retirement queue records.
 * Once reserved, queue publication fails closed instead of allocating process
 * memory when the inventory is exhausted.
 */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_runtime_reserve_retirement_records(
    ShadowSpillRuntime *runtime,
    uint64_t minimum_free_records
);

/*
 * Cold-path capacity reservation for one pool's reusable MemoryLease records.
 * The first call seals hot acquisition: later exhaustion fails closed instead
 * of allocating process-heap metadata from an allocator callback.
 */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_runtime_reserve_memory_lease_records(
    ShadowSpillRuntime *runtime,
    uint32_t pool_id,
    uint64_t minimum_free_records
);

/*
 * What became of one plan id. A lease can outlive the plan that made it, so the
 * id on a lease is answerable after the plan itself is gone -- CLOSED is how a
 * range left behind by earlier work is told from one the current plan made.
 */
typedef enum ShadowSpillPlanState {
    /* No plan was ever created with this id. */
    SHADOWSPILL_PLAN_STATE_UNKNOWN = 0,
    /* Created and open: it may still admit work. */
    SHADOWSPILL_PLAN_STATE_LIVE = 1,
    /* Closed, record still present: it admits no further work. */
    SHADOWSPILL_PLAN_STATE_CLOSED = 2,
    /* Closed and the record freed. The id stays claimed, because a lease may
     * still carry it. */
    SHADOWSPILL_PLAN_STATE_DESTROYED = 3
} ShadowSpillPlanState;

SHADOWSPILL_API ShadowSpillStatus shadowspill_runtime_plan_state(
    ShadowSpillRuntime *runtime,
    uint64_t plan_id,
    ShadowSpillPlanState *state
);

/*
 * The plan one id names, or NULL when no record holds it -- the id was never
 * created with, or its plan has been destroyed. Valid only while the caller
 * knows the plan is live, since a concurrent destroy would leave it dangling;
 * ask `shadowspill_runtime_plan_state` to learn what became of an id without
 * touching the record.
 */
SHADOWSPILL_API ShadowSpillPlan *shadowspill_runtime_plan(
    ShadowSpillRuntime *runtime,
    uint64_t plan_id
);

/*
 * Take the next plan id for this runtime. This only ever counts up, so an id is
 * unique for the life of the runtime and is not reissued when the plan holding
 * it is destroyed. A lease is therefore never readable as belonging to a later
 * plan that happens to have been given the same number.
 *
 * Taken before `shadowspill_plan_create` rather than returned by it, so the
 * caller can name the id on the allocation scopes it opens first.
 *
 * An id may be used for one plan only. Plan creation enforces that outright:
 * an id this did not issue is refused, and so is one that has already been
 * created with, whether that plan is still live or has since been closed.
 */
SHADOWSPILL_API ShadowSpillStatus shadowspill_runtime_next_plan_id(
    ShadowSpillRuntime *runtime,
    uint64_t *plan_id
);

/*
 * `description->plan_id` must be an id from `shadowspill_runtime_next_plan_id`
 * that no live plan on this runtime already holds; anything else is
 * INVALID_ARGUMENT.
 */
SHADOWSPILL_API ShadowSpillStatus shadowspill_plan_create(
    ShadowSpillRuntime *runtime,
    const ShadowSpillPlanDescription *description,
    ShadowSpillPlan **plan
);

/*
 * Take back every range this plan's own scopes allocated, whatever still points
 * at it, and write how many were reclaimed.
 *
 * A plan owes the pool every range its tasks and its profiling probes made, and
 * this is how the runtime collects when nothing else has. It works in leases and
 * bytes, which is what the runtime owns; it does not consult the framework, and
 * cannot, so a caller that may still read an object backed by one of these ranges
 * must drop it first. That is what makes this the forcing path rather than the
 * ordinary one, where a lease goes when the framework's own reference count
 * reaches zero.
 *
 * Ranges carrying no plan -- a provider taking its own workspace between tasks --
 * belong to no plan and are left alone.
 *
 * A lease the framework has not freed yet keeps its pointer and id indexed, so the
 * free that eventually arrives still resolves instead of faulting.
 */
SHADOWSPILL_API ShadowSpillStatus shadowspill_plan_reclaim_scoped_leases(
    ShadowSpillPlan *plan,
    uint64_t *reclaimed
);

SHADOWSPILL_API ShadowSpillStatus shadowspill_plan_close(
    ShadowSpillPlan *plan
);

SHADOWSPILL_API void shadowspill_plan_destroy(ShadowSpillPlan *plan);

/*
 * Rejects new work, drains queued work, synchronizes every route's lane, joins
 * the worker, and releases owned resources. This call is explicitly
 * synchronizing and idempotent. It returns the first latched failure.
 */
SHADOWSPILL_API ShadowSpillStatus shadowspill_runtime_close(
    ShadowSpillRuntime *runtime
);

/*
 * Closes without waiting for anything: no drain, no stream synchronization.
 * For a process that is already going away, where waiting is both pointless
 * and unbounded. Outstanding work cannot complete once the process is
 * exiting, and everything the drain protects -- device allocations, pinned
 * registrations, the context itself -- is reclaimed at process exit anyway,
 * so this stops the worker, releases what it owns, and returns.
 *
 * The counts a caller passes are filled in before anything is stopped, so a
 * caller can report what was still outstanding. Either may be NULL.
 */
SHADOWSPILL_API ShadowSpillStatus shadowspill_runtime_abandon(
    ShadowSpillRuntime *runtime,
    uint64_t *outstanding_actions,
    uint64_t *outstanding_retirements
);

/*
 * Recalibrates every configured directed route when ``routes`` is NULL and
 * ``route_count`` is zero, or only the supplied route keys otherwise. The
 * runtime must be locally idle. This function deliberately performs no
 * inter-process coordination, allowing callers to invoke it concurrently in
 * independent processes after establishing their own barriers. A successful
 * call atomically publishes one new matrix generation.
 */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_runtime_calibrate_transfer_capabilities(
    ShadowSpillRuntime *runtime,
    const ShadowSpillTransferCalibrationConfig *config,
    const ShadowSpillTransferRouteKey *routes,
    uint32_t route_count
);

/*
 * Copies the complete row-major N-by-N profile matrix. ``capacity`` must be at
 * least N*N. The caller receives a lock-consistent generation and count.
 */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_runtime_transfer_profiles(
    ShadowSpillRuntime *runtime,
    ShadowSpillTransferProfile *profiles,
    uint32_t capacity,
    uint32_t *count,
    uint64_t *generation
);

/* Calls close if needed and releases the runtime record. NULL is accepted. */
SHADOWSPILL_API void shadowspill_runtime_destroy(
    ShadowSpillRuntime *runtime
);

#ifdef __cplusplus
}
#endif

#endif
