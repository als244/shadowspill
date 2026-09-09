#ifndef SHADOWSPILL_PLANNER_INTERNAL_H
#define SHADOWSPILL_PLANNER_INTERNAL_H

#include <shadowspill/planner.h>

/* Generic schedule helpers, in `search/toolkit/`. Neither knows which
 * search placed the schedule it is handed. */

/* Overlay a schedule onto its topology, giving the program to simulate. */
void shadowspill_bind_indexed_schedule(
    const ShadowSpillSimulationProgram *topology,
    const ShadowSpillIndexedSchedule *schedule,
    ShadowSpillSimulationProgram *program
);

void shadowspill_schedule_digest(
    const ShadowSpillScheduleContext *context,
    const ShadowSpillIndexedSchedule *schedule,
    uint8_t digest[SHADOWSPILL_PLANNER_DIGEST_BYTES]
);

#endif
