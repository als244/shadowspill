/* What one span holds resident, and what that costs. */
#include "internal.h"

uint32_t shadowspill_schedule_collect_spans(
    const uint8_t *resident,
    const uint8_t *breaks,
    uint32_t alias,
    uint32_t boundary_count,
    Span *spans
) {
    uint32_t count = 0U;
    uint32_t index = 0U;
    while (index < boundary_count) {
        if (!shadowspill_cell_get(resident, shadowspill_schedule_cell(alias, boundary_count, index))) {
            ++index;
            continue;
        }
        uint32_t start = index;
        while (index + 1U < boundary_count &&
               shadowspill_cell_get(resident, shadowspill_schedule_cell(alias, boundary_count, index + 1U)) &&
               !shadowspill_cell_get(breaks, shadowspill_schedule_cell(alias, boundary_count, index))) {
            ++index;
        }
        spans[count++] = (Span){.start = start, .end = index};
        ++index;
    }
    return count;
}

static int has_future_access(
    const ShadowSpillScheduleFacts *facts,
    uint32_t alias,
    const Span *span
) {
    return shadowspill_span_accessed_after(
        facts->problem->residency, alias, span->start, span->end, (int32_t)span->end - 1
    );
}

static void alias_contribution(
    const ShadowSpillScheduleFacts *facts,
    const uint8_t *resident,
    const uint8_t *breaks,
    uint32_t alias,
    int fetch_headroom,
    uint8_t *contribution,
    Span *spans
) {
    const ShadowSpillPressureFitResidencyProblem *problem = facts->problem->residency;
    memset(contribution, 0, facts->boundary_count);
    uint32_t span_count = shadowspill_schedule_collect_spans(
        resident,
        breaks,
        alias,
        facts->boundary_count,
        spans
    );
    for (uint32_t index = 0U; index < span_count; ++index) {
        uint32_t start = spans[index].start;
        if (fetch_headroom != 0 && start > 0U &&
            problem->productions[shadowspill_schedule_cell(alias, facts->boundary_count, start)] ==
                0U) {
            --start;
        }
        int32_t end = (int32_t)spans[index].end;
        if (spans[index].end > 0U && problem->final_location[alias] != 0 &&
            !has_future_access(facts, alias, &spans[index])) {
            --end;
        }
        if (end >= (int32_t)start) {
            for (uint32_t boundary = start; boundary <= (uint32_t)end;
                 ++boundary) {
                contribution[boundary] = 1U;
            }
        }
    }
    for (uint32_t boundary = 0U; boundary < facts->boundary_count; ++boundary) {
        if (problem->output_reservations[shadowspill_schedule_cell(
                alias,
                facts->boundary_count,
                boundary
            )] != 0U) {
            contribution[boundary] = 1U;
        }
    }
}

int shadowspill_schedule_build_pressure(
    const ShadowSpillScheduleFacts *facts,
    const uint8_t *resident,
    const uint8_t *breaks,
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
    uint8_t *contribution = calloc(
        facts->boundary_count,
        sizeof(*contribution)
    );
    Span *spans = calloc(facts->boundary_count, sizeof(*spans));
    if (contribution == NULL || spans == NULL) {
        free(contribution);
        free(spans);
        return -1;
    }
    const ShadowSpillPressureFitResidencyProblem *problem = facts->problem->residency;
    for (uint32_t alias = 0U; alias < facts->alias_count; ++alias) {
        /* The resident slice holds the aliases the reducer may not cut, and
         * the capacity measured against already excludes it. */
        if (!shadowspill_alias_may_cut(problem, alias)) {
            continue;
        }
        alias_contribution(
            facts,
            resident,
            breaks,
            alias,
            fetch_headroom,
            contribution,
            spans
        );
        uint32_t device = problem->alias_device[alias];
        for (uint32_t boundary = 0U; boundary < facts->boundary_count;
             ++boundary) {
            if (contribution[boundary] != 0U) {
                pressure[(uint64_t)device * facts->boundary_count + boundary] +=
                    problem->alias_size_bytes[alias];
            }
        }
    }
    free(contribution);
    free(spans);
    return 0;
}

int shadowspill_extend_interval_entries(
    const ShadowSpillScheduleFacts *facts,
    uint8_t *resident,
    uint8_t *breaks
) {
    if (facts == NULL || resident == NULL || breaks == NULL) {
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
        shadowspill_schedule_build_pressure(facts, resident, breaks, 0, pressure) != 0) {
        free(pressure);
        free(spans);
        return -1;
    }
    const ShadowSpillPressureFitResidencyProblem *problem = facts->problem->residency;
    for (uint32_t alias = 0U; alias < facts->alias_count; ++alias) {
        if (!shadowspill_alias_may_cut(problem, alias)) {
            continue;
        }
        uint32_t span_count = shadowspill_schedule_collect_spans(
            resident,
            breaks,
            alias,
            facts->boundary_count,
            spans
        );
        uint32_t device = problem->alias_device[alias];
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
                uint64_t added = problem->output_reservations[shadowspill_schedule_cell(
                    alias,
                    facts->boundary_count,
                    candidate_cell
                )] != 0U
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
                shadowspill_cell_set(resident, shadowspill_schedule_cell(alias, facts->boundary_count, candidate_cell), 1);
                pressure[position] += added;
                current->start = candidate_cell;
            }
        }
    }
    free(pressure);
    free(spans);
    return 0;
}
