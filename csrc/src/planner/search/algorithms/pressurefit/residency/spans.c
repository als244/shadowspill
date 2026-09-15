/* What residency costs at a boundary, and over a span. */
#include "internal.h"

uint64_t shadowspill_residency_cell(uint32_t alias, uint32_t boundary_count, uint32_t index) {
    return (uint64_t)alias * boundary_count + index;
}

int shadowspill_residency_next_span(
    const uint8_t *resident,
    const uint8_t *breaks,
    uint32_t alias,
    uint32_t boundary_count,
    uint32_t *cursor,
    uint32_t *start,
    uint32_t *end
) {
    const uint64_t row = (uint64_t)alias * boundary_count;
    uint64_t bit = row + *cursor;
    const uint64_t limit = row + boundary_count;
    while (bit < limit) {
        if ((bit & 7U) == 0U && resident[bit >> 3U] == 0U) {
            bit += 8U;
            continue;
        }
        if (shadowspill_cell_get(resident, bit)) {
            break;
        }
        ++bit;
    }
    if (bit >= limit) {
        *cursor = boundary_count;
        return 0;
    }
    uint32_t left = (uint32_t)(bit - row);
    /* The span ends before the first later cell that is not resident or
     * that follows a break: scan those conditions a word at a time. */
    const size_t packed_bytes = shadowspill_packed_cells(
        (uint64_t)(alias + 1U) * boundary_count
    );
    uint32_t right = left;
    uint64_t probe = row + left + 1U;
    while (probe < limit) {
        const unsigned width = (unsigned)((limit - probe) < 64U ? limit - probe : 64U);
        const uint64_t present = shadowspill_cells_load(resident, packed_bytes, probe, width);
        const uint64_t broken = shadowspill_cells_load(breaks, packed_bytes, probe - 1U, width);
        uint64_t stop = ~present | broken;
        if (width < 64U) {
            stop &= (UINT64_C(1) << width) - 1U;
        }
        if (stop != 0U) {
            right = (uint32_t)(probe - row) + (uint32_t)__builtin_ctzll(stop) - 1U;
            break;
        }
        probe += width;
        right = (uint32_t)(probe - row) - 1U;
    }
    *start = left;
    *end = right;
    *cursor = right + 1U;
    return 1;
}

/* The charged interval of one span, or an empty one (end < start). */
void shadowspill_residency_span_charge(
    const ShadowSpillPressureFitResidencyProblem *problem,
    const ShadowSpillPressureFitResidencyOptions *options,
    uint32_t alias,
    uint32_t start,
    uint32_t end,
    int32_t *charged_start,
    int32_t *charged_end
) {
    const uint32_t count = problem->boundary_count;
    *charged_start = (int32_t)start;
    if (options->fetch_headroom != 0U && start > 0U &&
        problem->productions[shadowspill_residency_cell(alias, count, start)] == 0U) {
        --*charged_start;
    }
    *charged_end = (int32_t)end;
    const int32_t end_boundary = (int32_t)end - 1;
    if (end_boundary >= 0 && problem->final_location[alias] != 0 &&
        !shadowspill_span_accessed_after(problem, alias, start, end, end_boundary)) {
        --*charged_end;
    }
}

/* The span of `alias` that holds cell `inside`: the maximal run of resident
 * cells around it with no break between neighbours, found a word at a time
 * in both directions. */
void shadowspill_residency_span_around(
    const uint8_t *resident,
    const uint8_t *breaks,
    uint32_t alias,
    uint32_t boundary_count,
    uint32_t inside,
    uint32_t *start,
    uint32_t *end
) {
    const uint64_t row = (uint64_t)alias * boundary_count;
    const size_t packed_bytes = shadowspill_packed_cells(
        (uint64_t)(alias + 1U) * boundary_count
    );
    /* Rightwards: the span ends before the first later cell that is not
     * resident or that follows a break (next_span's rule). */
    uint32_t right = inside;
    uint64_t probe = row + inside + 1U;
    const uint64_t limit = row + boundary_count;
    while (probe < limit) {
        const unsigned width = (unsigned)((limit - probe) < 64U ? limit - probe : 64U);
        const uint64_t present = shadowspill_cells_load(resident, packed_bytes, probe, width);
        const uint64_t broken = shadowspill_cells_load(breaks, packed_bytes, probe - 1U, width);
        uint64_t stop = ~present | broken;
        if (width < 64U) {
            stop &= (UINT64_C(1) << width) - 1U;
        }
        if (stop != 0U) {
            right = (uint32_t)(probe - row) + (uint32_t)__builtin_ctzll(stop) - 1U;
            break;
        }
        probe += width;
        right = (uint32_t)(probe - row) - 1U;
    }
    /* Leftwards: the span starts after the last earlier cell that is not
     * resident or that carries a break. */
    uint32_t left = inside;
    while (left > 0U) {
        const unsigned width = (unsigned)(left < 64U ? left : 64U);
        const uint64_t offset = row + left - width;
        const uint64_t present = shadowspill_cells_load(resident, packed_bytes, offset, width);
        const uint64_t broken = shadowspill_cells_load(breaks, packed_bytes, offset, width);
        uint64_t stop = ~present | broken;
        if (width < 64U) {
            stop &= (UINT64_C(1) << width) - 1U;
        }
        if (stop != 0U) {
            const unsigned highest = 63U - (unsigned)__builtin_clzll(stop);
            left = (uint32_t)(offset - row) + highest + 1U;
            break;
        }
        left -= width;
    }
    *start = left;
    *end = right;
}

/* Mark the boundaries one span [start, end] charges: the span itself, one
 * boundary of fetch headroom before it when the span does not begin
 * with a production, and not its final boundary when nothing accesses the
 * alias later and it may leave. */
static void span_contribution(
    const ShadowSpillPressureFitResidencyProblem *problem,
    const ShadowSpillPressureFitResidencyOptions *options,
    uint32_t alias,
    uint32_t start,
    uint32_t end,
    uint8_t *contribution
) {
    int32_t charged_start;
    int32_t charged_end;
    shadowspill_residency_span_charge(problem, options, alias, start, end, &charged_start, &charged_end);
    for (int32_t boundary = charged_start; boundary <= charged_end; ++boundary) {
        contribution[boundary] = 1U;
    }
}

/* Mark every boundary the spans of one alias within [first, last] charge. */
static void spans_contribution(
    const ShadowSpillPressureFitResidencyProblem *problem,
    const ShadowSpillPressureFitResidencyOptions *options,
    const uint8_t *resident,
    const uint8_t *breaks,
    uint32_t alias,
    uint32_t first,
    uint32_t last,
    uint8_t *contribution
) {
    uint32_t cursor = first;
    while (cursor <= last) {
        uint32_t start = 0U;
        uint32_t end = 0U;
        if (!shadowspill_residency_next_span(
                resident,
                breaks,
                alias,
                problem->boundary_count,
                &cursor,
                &start,
                &end
            ) ||
            start > last) {
            break;
        }
        span_contribution(problem, options, alias, start, end, contribution);
    }
}

void shadowspill_residency_alias_contribution(
    const ShadowSpillPressureFitResidencyProblem *problem,
    const ShadowSpillPressureFitResidencyOptions *options,
    const uint8_t *resident,
    const uint8_t *breaks,
    uint32_t alias,
    uint8_t *contribution
) {
    uint32_t count = problem->boundary_count;
    memset(contribution, 0, count);
    spans_contribution(
        problem, options, resident, breaks, alias, 0U, count - 1U, contribution
    );
    for (uint32_t boundary = 0U; boundary < count; ++boundary) {
        if (problem->output_reservations[shadowspill_residency_cell(alias, count, boundary)] != 0U) {
            contribution[boundary] = 1U;
        }
    }
}

int shadowspill_residency_pressure_at(
    const ShadowSpillPressureFitResidencyProblem *problem,
    const ShadowSpillPressureFitResidencyOptions *options,
    const uint8_t *resident,
    const uint8_t *breaks,
    uint32_t device,
    uint32_t boundary,
    ShadowSpillPressureFitResidencyWorkspace *workspace,
    uint64_t *pressure_bytes
) {
    if (problem == NULL || options == NULL || resident == NULL ||
        breaks == NULL || workspace == NULL || pressure_bytes == NULL ||
        device >= problem->device_count || boundary >= problem->boundary_count ||
        workspace->alias_count != problem->alias_count ||
        workspace->boundary_count != problem->boundary_count ||
        workspace->device_count != problem->device_count) {
        return -1;
    }
    uint64_t pressure = 0U;
    for (uint32_t alias = 0U; alias < problem->alias_count; ++alias) {
        if (problem->alias_device[alias] != device ||
            !shadowspill_alias_may_cut(problem, alias)) {
            continue;
        }
        shadowspill_residency_alias_contribution(
            problem,
            options,
            resident,
            breaks,
            alias,
            workspace->before
        );
        if (workspace->before[boundary] == 0U) {
            continue;
        }
        if (pressure > UINT64_MAX - problem->alias_size_bytes[alias]) {
            return -1;
        }
        pressure += problem->alias_size_bytes[alias];
    }
    *pressure_bytes = pressure;
    return 0;
}
