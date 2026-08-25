#ifndef SHADOWSPILL_PLANNER_CANDIDATES_INTERNAL_H
#define SHADOWSPILL_PLANNER_CANDIDATES_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include <shadowspill/planner.h>

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
} ShadowSpillScheduleFacts;

/*
 * A physical-admission repair constrains one logical fetch interval, identified
 * by the object and its next consumer.  The bounds survive residency
 * re-emission; addresses remain entirely dynamic.
 */
typedef struct ShadowSpillPrefetchTriggerConstraint {
    uint32_t alias;
    uint32_t consumer_task;
    uint32_t minimum_trigger;
    uint32_t maximum_trigger;
} ShadowSpillPrefetchTriggerConstraint;

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

/* Records `record` and keeps its own copy of `plan` if it beats what is
 * held, returning non-zero if it did. Internal because the plan it keeps is
 * an internal storage type; the rest of the gate is public. */
int shadowspill_best_placed_offer(
    ShadowSpillBestPlaced *best,
    const ShadowSpillBestPlacedRecord *record,
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
    uint8_t prefetch_rule,
    int coalesced,
    int prefetch_headroom,
    ShadowSpillScheduleStorage *storage
);

int shadowspill_delay_indexed_prefetch(
    const ShadowSpillScheduleFacts *facts,
    const ShadowSpillSimulationResult *failure,
    ShadowSpillScheduleStorage *storage,
    ShadowSpillPrefetchTriggerConstraint *constraint
);

int shadowspill_advance_indexed_prefetch_to_release(
    const ShadowSpillScheduleFacts *facts,
    uint32_t action_index,
    ShadowSpillScheduleStorage *storage,
    ShadowSpillPrefetchTriggerConstraint *constraint
);

int shadowspill_apply_prefetch_trigger_constraints(
    const ShadowSpillScheduleFacts *facts,
    const ShadowSpillPrefetchTriggerConstraint *constraints,
    uint32_t constraint_count,
    ShadowSpillScheduleStorage *storage
);

void shadowspill_bind_indexed_schedule(
    const ShadowSpillSimulationProgram *topology,
    const ShadowSpillIndexedSchedule *schedule,
    ShadowSpillSimulationProgram *program
);

void shadowspill_schedule_digest(
    const ShadowSpillPressureFitProblem *problem,
    const ShadowSpillIndexedSchedule *schedule,
    uint8_t digest[SHADOWSPILL_PLANNER_DIGEST_BYTES]
);

#endif
