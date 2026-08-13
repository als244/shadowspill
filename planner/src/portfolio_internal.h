#ifndef SHADOWSPILL_PLANNER_PORTFOLIO_INTERNAL_H
#define SHADOWSPILL_PLANNER_PORTFOLIO_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include <shadowspill/planner.h>

typedef struct ShadowSpillScheduleStorage {
    ShadowSpillDenseSchedule value;
    uint32_t action_capacity;
    uint32_t initial_capacity;
    uint32_t final_capacity;
} ShadowSpillScheduleStorage;

typedef struct ShadowSpillScheduleFacts {
    const ShadowSpillPressureFitContext *context;
    uint32_t alias_count;
    uint32_t task_count;
    uint32_t boundary_count;
    uint32_t device_count;
    uint32_t *earliest_access_task;
    uint8_t *write_events;
} ShadowSpillScheduleFacts;

int shadowspill_schedule_facts_create(
    const ShadowSpillPressureFitContext *context,
    ShadowSpillScheduleFacts *facts
);

void shadowspill_schedule_facts_destroy(ShadowSpillScheduleFacts *facts);

int shadowspill_schedule_storage_create(
    uint32_t alias_count,
    uint32_t task_count,
    ShadowSpillScheduleStorage *storage
);

void shadowspill_schedule_storage_clear(ShadowSpillScheduleStorage *storage);

void shadowspill_schedule_storage_destroy(ShadowSpillScheduleStorage *storage);

int shadowspill_schedule_storage_copy(
    ShadowSpillScheduleStorage *destination,
    const ShadowSpillScheduleStorage *source
);

int shadowspill_extend_interval_entries(
    const ShadowSpillScheduleFacts *facts,
    uint8_t *resident,
    uint8_t *breaks
);

int shadowspill_emit_dense_schedule(
    const ShadowSpillScheduleFacts *facts,
    const uint8_t *resident,
    const uint8_t *breaks,
    uint8_t prefetch_rule,
    int coalesced,
    int prefetch_headroom,
    ShadowSpillScheduleStorage *storage
);

int shadowspill_delay_dense_prefetch(
    const ShadowSpillScheduleFacts *facts,
    const ShadowSpillSimulationResult *failure,
    ShadowSpillScheduleStorage *storage
);

void shadowspill_bind_dense_schedule(
    const ShadowSpillSimulationProgram *topology,
    const ShadowSpillDenseSchedule *schedule,
    ShadowSpillSimulationProgram *program
);

void shadowspill_schedule_digest(
    const ShadowSpillPressureFitContext *context,
    const ShadowSpillDenseSchedule *schedule,
    uint8_t digest[SHADOWSPILL_PLANNER_DIGEST_BYTES]
);

#endif
