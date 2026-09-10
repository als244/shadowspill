#ifndef SHADOWSPILL_PLANNER_CANDIDATES_INTERNAL_H
#define SHADOWSPILL_PLANNER_CANDIDATES_INTERNAL_H

#include "residency_internal.h"

#include <stddef.h>
#include <stdint.h>

#include <shadowspill/planner.h>
#include "../../../internal.h"
#include <shadowspill/pressurefit/pressurefit.h>

typedef struct ShadowSpillPressureFitProblem {
    uint32_t abi_version;
    /* What any code judging this problem's schedules needs. */
    ShadowSpillScheduleContext context;
    const ShadowSpillPressureFitResidencyProblem *residency;
    const uint8_t *seed_resident;
    const uint8_t *seed_breaks;
    /* The plan to beat: a plan for this resolved program already in hand --
       found at a smaller capacity, say -- or NULL. The search measures it at
       this capacity before any candidate runs and answers with it unless a
       candidate does strictly better, so a search given one never answers
       worse than it. Its aliases and tasks index this problem. */
    const ShadowSpillIndexedSchedule *incumbent;
} ShadowSpillPressureFitProblem;
/*
 * Evaluate several already-resolved problems together, on one set of worker
 * threads. This is what shadowspill_pressurefit_search() runs once it has
 * resolved its input; a caller reaches it through that.
 */
ShadowSpillStatus
shadowspill_pressurefit_evaluate_resolved(
    const ShadowSpillPressureFitProblem *problems,
    uint32_t problem_count,
    const ShadowSpillPressureFitOptions *options,
    ShadowSpillPressureFitResult *results
);
#include <shadowspill/simulator.h>

typedef struct ShadowSpillScheduleStorage {
    ShadowSpillIndexedSchedule value;
    uint32_t action_capacity;
    uint32_t initial_capacity;
    uint32_t final_capacity;
} ShadowSpillScheduleStorage;

typedef struct ShadowSpillScheduleFacts {
    const ShadowSpillPressureFitProblem *problem;
    /* What the plan being emitted has given back, per [device][boundary], or
     * NULL for none. The emitter places fetches and evictions against the
     * capacity it believes it has, so a plan built at a smaller capacity has
     * to be emitted against that one too -- the same array the reducer adds
     * to its own occupancy. */
    const uint64_t *extra_pressure;
    uint32_t alias_count;
    uint32_t task_count;
    uint32_t boundary_count;
    uint32_t device_count;
    uint32_t *earliest_access_task;
    uint8_t *write_events;
    /* The same write events, indexed for the passes that ask when an object
     * was written last rather than whether it was written here: the
     * boundaries alias `a` is written at are
     * `write_boundaries[write_offsets[a] .. write_offsets[a + 1])`, ascending. */
    uint32_t *write_offsets;
    uint32_t *write_boundaries;
} ShadowSpillScheduleFacts;

/*
 * A physical-admission repair constrains one logical fetch interval, identified
 * by the object and its next consumer.  The bounds survive residency
 * re-emission; addresses remain entirely dynamic.
 */
typedef struct ShadowSpillFetchTriggerConstraint {
    uint32_t alias;
    uint32_t consumer_task;
    uint32_t minimum_trigger;
    uint32_t maximum_trigger;
} ShadowSpillFetchTriggerConstraint;

/* Keep one logical object absent at one residency boundary. */
int shadowspill_schedule_facts_create(
    const ShadowSpillPressureFitProblem *problem,
    ShadowSpillScheduleFacts *facts
);

void shadowspill_schedule_facts_destroy(ShadowSpillScheduleFacts *facts);

int shadowspill_schedule_storage_create(
    uint32_t alias_count,
    ShadowSpillScheduleStorage *storage
);

void shadowspill_schedule_storage_clear(ShadowSpillScheduleStorage *storage);

void shadowspill_schedule_storage_destroy(ShadowSpillScheduleStorage *storage);

int shadowspill_schedule_storage_copy(
    ShadowSpillScheduleStorage *destination,
    const ShadowSpillScheduleStorage *source
);

/* Hold a copy of a schedule that arrived from outside the search. */
int shadowspill_schedule_storage_assign(
    ShadowSpillScheduleStorage *destination,
    const ShadowSpillIndexedSchedule *source
);

/* The best placed makespan so far, or zero when nothing has been placed.
 * Lock-free: the default search mode consults it for every plan that
 * simulated. */
uint64_t shadowspill_best_placed_bound(const ShadowSpillPressureFitBestPlaced *best);

/* Records `record` and keeps its own copy of `plan` if it beats what is
 * held, returning non-zero if it did. Internal because the plan it keeps is
 * an internal storage type; the rest of the gate is public. */
int shadowspill_best_placed_offer(
    ShadowSpillPressureFitBestPlaced *best,
    const ShadowSpillPressureFitBestPlacedRecord *record,
    const ShadowSpillScheduleStorage *plan
);

int shadowspill_extend_interval_entries(
    const ShadowSpillScheduleFacts *facts,
    uint8_t *resident,
    uint8_t *breaks
);

int shadowspill_emit_indexed_schedule(
    const ShadowSpillScheduleFacts *facts,
    const uint8_t *resident,
    const uint8_t *breaks,
    uint8_t fetch_rule,
    int coalesced,
    int fetch_headroom,
    ShadowSpillScheduleStorage *storage
);

int shadowspill_delay_indexed_fetch(
    const ShadowSpillScheduleFacts *facts,
    const ShadowSpillSimulationResult *failure,
    ShadowSpillScheduleStorage *storage,
    ShadowSpillFetchTriggerConstraint *constraint
);

/*
 * Split the evictions that held something up: a `WRITE_BACK` at the boundary
 * where the object was last written, and a `RELEASE` where the eviction was.
 *
 * An eviction exists to free device memory, and the memory is only free once
 * its copy has landed, so an eviction costs time exactly when something was
 * waiting for room while it ran. `simulation` -- a simulation of this
 * schedule as it stands -- says which ones those were. The rest are left
 * alone: moving a copy nothing waited on spends lane time and holds spill
 * capacity longer to buy nothing.
 *
 * Where the copy actually runs is not decided here. The write-back is
 * triggered at the last write because that is the earliest boundary at which
 * the copy is correct; the simulator owns the lane and prices the queue that
 * forms when several copies move at once. Residency is untouched, so the
 * device copy lives exactly as long as it did.
 *
 * The caller simulates again and keeps the result only if the plan got
 * faster. Returns how many evictions were split, or -1.
 */
int shadowspill_split_blocking_evictions(
    const ShadowSpillScheduleFacts *facts,
    const ShadowSpillSimulationResult *simulation,
    ShadowSpillScheduleStorage *storage
);

int shadowspill_advance_indexed_fetch_to_release(
    const ShadowSpillScheduleFacts *facts,
    uint32_t action_index,
    ShadowSpillScheduleStorage *storage,
    ShadowSpillFetchTriggerConstraint *constraint
);

int shadowspill_apply_fetch_trigger_constraints(
    const ShadowSpillScheduleFacts *facts,
    const ShadowSpillFetchTriggerConstraint *constraints,
    uint32_t constraint_count,
    ShadowSpillScheduleStorage *storage
);

#endif
