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

int shadowspill_extend_interval_entries(
    const ShadowSpillScheduleFacts *facts,
    ShadowSpillResidency *residency
) {
    if (facts == NULL || residency == NULL) {
        return -1;
    }
    size_t pressure_cells = 0U;
    if (shadowspill_schedule_checked_cells(facts->device_count, facts->boundary_count, &pressure_cells) !=
        0) {
        return -1;
    }
    uint64_t *pressure = calloc(
        pressure_cells == 0U ? 1U : pressure_cells,
        sizeof(*pressure)
    );
    Span *spans = calloc(facts->boundary_count, sizeof(*spans));
    if (pressure == NULL || spans == NULL ||
        shadowspill_schedule_build_pressure(facts, residency, 0, pressure) != 0) {
        free(pressure);
        free(spans);
        return -1;
    }
    const ShadowSpillPressureFitResidencyProblem *problem = facts->problem->residency;
    for (uint32_t alias = 0U; alias < facts->alias_count; ++alias) {
        if (!shadowspill_alias_may_cut(problem, alias)) {
            continue;
        }
        uint32_t span_count = shadowspill_schedule_collect_spans(residency, alias, spans);
        uint32_t device = problem->alias_device[alias];
        ShadowSpillSpanList *list = &residency->aliases[alias];
        for (uint32_t span_index = 1U; span_index < span_count; ++span_index) {
            Span *current = &spans[span_index];
            const Span *previous = &spans[span_index - 1U];

            /*
             * Residency canonicalization clears breaks on every absent cell.
             * Extending a later span by one absent cell therefore only moves
             * its start left; it cannot change any other span.  Keep that
             * local start directly instead of rediscovering every span after
             * each successful extension.
             */
            while (current->start > previous->end + 1U) {
                uint32_t candidate_cell = current->start - 1U;
                uint64_t position =
                    (uint64_t)device * facts->boundary_count + candidate_cell;
                uint64_t added = shadowspill_reserved_at(problem, alias, candidate_cell)
                    ? 0U
                    : problem->alias_size_bytes[alias];
                if (pressure[position] + added >
                    shadowspill_boundary_capacity(
                        problem,
                        device,
                        candidate_cell
                    )) {
                    break;
                }
                list->bounds[2U * span_index] = candidate_cell;
                pressure[position] += added;
                current->start = candidate_cell;
            }
        }
    }
    free(pressure);
    free(spans);
    return 0;
}
