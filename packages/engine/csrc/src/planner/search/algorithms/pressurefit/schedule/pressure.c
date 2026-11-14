/* What one span holds resident, and what that costs. */
#include "internal.h"

uint32_t shadowspill_schedule_collect_spans(
    const ShadowSpillResidency *residency,
    uint32_t alias,
    Span *spans
) {
    const ShadowSpillSpanList *list = &residency->aliases[alias];
    for (uint32_t index = 0U; index < list->count; ++index) {
        spans[index] = (Span){
            .start = list->bounds[2U * index], .end = list->bounds[2U * index + 1U]
        };
    }
    return list->count;
}

int shadowspill_schedule_build_pressure(
    const ShadowSpillScheduleFacts *facts,
    const ShadowSpillResidency *residency,
    int fetch_headroom,
    uint64_t *pressure
) {
    size_t pressure_cells = 0U;
    if (shadowspill_schedule_checked_cells(facts->device_count, facts->boundary_count, &pressure_cells) !=
        0) {
        return -1;
    }
    if (facts->extra_pressure != NULL) {
        /* Start from what the plan gave back, so every capacity test below
         * measures against the capacity this plan kept. */
        memcpy(
            pressure,
            facts->extra_pressure,
            pressure_cells * sizeof(*pressure)
        );
    } else {
        memset(pressure, 0, pressure_cells * sizeof(*pressure));
    }
    /* The resident slice holds the aliases the reducer may not cut, and the
     * capacity measured against already excludes it. */
    return shadowspill_residency_add_charges(
        facts->problem->residency, fetch_headroom, residency, pressure
    );
}
