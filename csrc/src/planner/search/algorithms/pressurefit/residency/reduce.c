/* The loop: cut, repressure, and stop when the budget is met. */
#include "internal.h"

static int apply_cut_and_repressure(
    const ShadowSpillPressureFitResidencyProblem *problem,
    const ShadowSpillPressureFitResidencyOptions *options,
    ShadowSpillPressureFitResidencyResult *result,
    ShadowSpillPressureFitResidencyWorkspace *workspace,
    const ResidencyCut *chosen
) {
    const uint32_t alias = chosen->alias;
    const uint32_t count = problem->boundary_count;
    const uint64_t row = (uint64_t)alias * count;
    uint8_t *resident = result->resident;
    uint8_t *breaks = result->breaks;
    if (workspace->touched_aliases[alias] == 0U) {
        workspace->touched_aliases[alias] = 1U;
        workspace->touched_list[workspace->touched_count++] = alias;
    }
    /* A cut changes one span. Its coverage can only move inside the cut
     * plus a boundary of charge on either side, and at the span's last two
     * boundaries, where the future-access rule can flip. Spans touching it
     * across a break matter only at the shared boundary, as a mask. */
    const uint32_t low = (uint32_t)(chosen->start > chosen->end
        ? chosen->end + 1
        : chosen->start + 1);
    const uint32_t high = (uint32_t)(chosen->end + 1);
    const int inside = shadowspill_cell_get(resident, row + low) != 0;
    uint32_t span_start = low;
    uint32_t span_end = high;
    int32_t before_start = 0;
    int32_t before_end = -1;
    if (inside) {
        shadowspill_residency_span_around(resident, breaks, alias, count, low, &span_start, &span_end);
        shadowspill_residency_span_charge(problem, options, alias, span_start, span_end, &before_start, &before_end);
    }
    /* What this reduction gave up, when anyone asked to be told. */
    if (result->cut_aliases != NULL && result->cut_count < result->cut_capacity) {
        result->cut_aliases[result->cut_count++] = alias;
    }
    shadowspill_residency_apply_cut(problem, resident, breaks, chosen);
    shadowspill_residency_refresh_alias_candidates(
        problem,
        resident,
        breaks,
        workspace->first_required,
        workspace->run_offsets,
        workspace->run_bounds,
        &workspace->cut_index,
        workspace->cut_active,
        alias
    );
    if (!inside) {
        return 0;
    }
    int32_t after_start[4];
    int32_t after_end[4];
    uint32_t pieces = 0U;
    uint32_t cursor = span_start;
    while (cursor <= span_end && pieces < 4U) {
        uint32_t start = 0U;
        uint32_t end = 0U;
        if (!shadowspill_residency_next_span(resident, breaks, alias, count, &cursor, &start, &end) ||
            start > span_end) {
            break;
        }
        shadowspill_residency_span_charge(
            problem, options, alias, start, end, &after_start[pieces], &after_end[pieces]
        );
        ++pieces;
    }
    /* The neighbours' charge at the two boundaries they can share. */
    int mask_left = 0;
    if (span_start > 0U && shadowspill_cell_get(resident, row + span_start - 1U)) {
        uint32_t neighbour_start = 0U;
        uint32_t neighbour_end = 0U;
        shadowspill_residency_span_around(resident, breaks, alias, count, span_start - 1U, &neighbour_start, &neighbour_end);
        int32_t charge_start = 0;
        int32_t charge_end = -1;
        shadowspill_residency_span_charge(problem, options, alias, neighbour_start, neighbour_end, &charge_start, &charge_end);
        mask_left = charge_end == (int32_t)span_start - 1;
    }
    int mask_right = 0;
    if (span_end + 1U < count && shadowspill_cell_get(resident, row + span_end + 1U)) {
        mask_right = options->fetch_headroom != 0U &&
            problem->productions[shadowspill_residency_cell(alias, count, span_end + 1U)] == 0U;
    }
    /* Boundaries whose charge changed: those the span covered that no piece
     * covers (a decrement), and the rare boundary a piece covers that the
     * span did not (an increment; only the span's last boundary can be one).
     * A reserved boundary or one a neighbour charges is unchanged either way. */
    const uint32_t device = problem->alias_device[alias];
    const uint64_t size = problem->alias_size_bytes[alias];
    const uint32_t reserved_first = problem->reserved_offsets[alias];
    const uint32_t reserved_last = problem->reserved_offsets[alias + 1U];
    int32_t lost_start[4];
    int32_t lost_end[4];
    uint32_t lost = 0U;
    int32_t cursor_start = before_start;
    for (uint32_t piece = 0U; piece < pieces; ++piece) {
        if (after_end[piece] < after_start[piece]) {
            continue;
        }
        if (after_start[piece] > cursor_start) {
            lost_start[lost] = cursor_start;
            lost_end[lost] = (after_start[piece] - 1 < before_end)
                ? after_start[piece] - 1
                : before_end;
            if (lost_end[lost] >= lost_start[lost]) {
                ++lost;
            }
        }
        if (after_end[piece] + 1 > cursor_start) {
            cursor_start = after_end[piece] + 1;
        }
    }
    if (cursor_start <= before_end) {
        lost_start[lost] = cursor_start;
        lost_end[lost] = before_end;
        ++lost;
    }
    for (uint32_t index = 0U; index < lost; ++index) {
        int32_t from = lost_start[index];
        const int32_t to = lost_end[index];
        uint32_t reserved = shadowspill_anchor_lower_bound(
            problem->reserved_positions, reserved_first, reserved_last, (uint32_t)from
        );
        while (from <= to) {
            int32_t stop = to;
            if (reserved < reserved_last &&
                (int32_t)problem->reserved_positions[reserved] <= to) {
                stop = (int32_t)problem->reserved_positions[reserved] - 1;
            }
            const int32_t masked = (mask_left && from <= (int32_t)span_start - 1 &&
                                    (int32_t)span_start - 1 <= stop)
                ? (int32_t)span_start - 1
                : (mask_right && from <= (int32_t)span_end && (int32_t)span_end <= stop)
                    ? (int32_t)span_end
                    : -1;
            if (masked >= 0) {
                if (masked > from) {
                    shadowspill_residency_pressure_add(workspace, device, (uint32_t)from, (uint32_t)masked - 1U, 0U - size);
                }
                from = masked + 1;
                continue;
            }
            if (stop >= from) {
                shadowspill_residency_pressure_add(workspace, device, (uint32_t)from, (uint32_t)stop, 0U - size);
            }
            if (stop < to) {
                from = stop + 2;
                ++reserved;
            } else {
                from = to + 1;
            }
        }
    }
    for (uint32_t piece = 0U; piece < pieces; ++piece) {
        /* A piece can gain charge only outside the span's old charged range:
         * before it or after it, each a short interval. */
        int32_t gained_start[2];
        int32_t gained_end[2];
        uint32_t gained = 0U;
        if (after_start[piece] < before_start) {
            gained_start[gained] = after_start[piece];
            gained_end[gained] = after_end[piece] < before_start - 1
                ? after_end[piece]
                : before_start - 1;
            ++gained;
        }
        if (after_end[piece] > before_end) {
            gained_start[gained] = after_start[piece] > before_end + 1
                ? after_start[piece]
                : before_end + 1;
            gained_end[gained] = after_end[piece];
            ++gained;
        }
        for (uint32_t index = 0U; index < gained; ++index)
        for (int32_t boundary = gained_start[index]; boundary <= gained_end[index]; ++boundary) {
            const int mask =
                problem->output_reservations[shadowspill_residency_cell(alias, count, (uint32_t)boundary)] != 0U ||
                (boundary == (int32_t)span_start - 1 && mask_left) ||
                (boundary == (int32_t)span_end && mask_right);
            if (mask) {
                continue;
            }
            shadowspill_residency_pressure_add(workspace, device, (uint32_t)boundary, (uint32_t)boundary, size);
            const uint64_t position =
                (uint64_t)device * problem->boundary_count + (uint32_t)boundary;
            const uint64_t used =
                shadowspill_residency_tree_pressure_at(workspace, device, (uint32_t)boundary) +
                options->extra_pressure_bytes[position];
            const uint64_t capacity =
                shadowspill_boundary_capacity(problem, device, (uint32_t)boundary);
            if (used <= capacity) {
                continue;
            }
            const ExcessEntry entry = {
                used - capacity,
                (uint32_t)boundary,
                problem->device_priority[device],
                device,
            };
            if (shadowspill_residency_excess_heap_push(workspace, entry) != 0) {
                return -1;
            }
        }
    }
    return 0;
}

/* No legal cut relieves this boundary, so no residency this strategy can
 * reach fits it. That is a fact about the problem rather than a failure. */
static ShadowSpillStatus report_analytic_infeasible(
    const ShadowSpillPressureFitResidencyProblem *problem,
    ShadowSpillPressureFitResidencyResult *result,
    uint32_t device,
    uint32_t boundary,
    uint64_t used_bytes
) {
    result->status = SHADOWSPILL_STATUS_ANALYTIC_INFEASIBLE;
    result->error_device = device;
    result->error_boundary = (int32_t)boundary - 1;
    result->required_bytes = used_bytes;
    result->capacity_bytes =
        shadowspill_boundary_capacity(problem, device, boundary);
    return SHADOWSPILL_STATUS_ANALYTIC_INFEASIBLE;
}

/*
 * Remove objects until every boundary fits.
 *
 * One step: take the boundary that is furthest over capacity, choose the cut
 * that relieves it best under the strategy's score, apply it, and let the
 * pressure map absorb the change. Repeat until nothing is over capacity, or
 * until a boundary has no legal cut left.
 */
ShadowSpillStatus shadowspill_pressurefit_reduce_residency_reusing(
    const ShadowSpillPressureFitResidencyProblem *problem,
    const ShadowSpillPressureFitResidencyOptions *options,
    ShadowSpillPressureFitResidencyResult *result,
    ShadowSpillPressureFitResidencyWorkspace *workspace
) {
    if (!shadowspill_residency_valid_problem(problem, options, result) || workspace == NULL ||
        workspace->alias_count != problem->alias_count ||
        workspace->boundary_count != problem->boundary_count ||
        workspace->device_count != problem->device_count) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    shadowspill_residency_reset_residency_result(result);
    shadowspill_residency_seed_residency(problem, options, result);
    if (shadowspill_residency_prepare_seed_geometry(problem, options, workspace) != 0 ||
        shadowspill_residency_prepare_base_pressure(problem, options, workspace) != 0) {
        return SHADOWSPILL_STATUS_PLANNER_INTERNAL_ERROR;
    }
    workspace->touched_count = 0U;
    if (shadowspill_residency_reset_cut_candidates(workspace) != 0) {
        return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    }
    shadowspill_residency_reset_working_pressure(problem, options, workspace);
    if (shadowspill_residency_seed_excess_heap(problem, options, workspace) != 0) {
        return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    }

    while (1) {
        uint32_t device = UINT32_MAX;
        uint32_t boundary = UINT32_MAX;
        uint64_t used_bytes = 0U;
        const int over_capacity = shadowspill_residency_pop_worst_boundary(
            problem, options, workspace, &device, &boundary, &used_bytes
        );
        if (over_capacity < 0) {
            shadowspill_residency_clear_touched(workspace);
            return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
        }
        if (over_capacity == 0) {
            for (uint32_t index = 0U; index < workspace->touched_count; ++index) {
                const uint32_t touched = workspace->touched_list[index];
                shadowspill_residency_canonicalize_row(
                    result->breaks,
                    result->resident,
                    touched,
                    problem->boundary_count
                );
                workspace->touched_aliases[touched] = 0U;
            }
            workspace->touched_count = 0U;
            result->status = SHADOWSPILL_STATUS_OK;
            return SHADOWSPILL_STATUS_OK;
        }
        ResidencyCut chosen;
        if (!shadowspill_residency_select_cut(
                problem,
                device,
                (int32_t)boundary - 1,
                options->minimize_transfer != 0U,
                &workspace->cut_index,
                workspace->cut_active,
                workspace->cut_cursors,
                &chosen
            )) {
            shadowspill_residency_clear_touched(workspace);
            return report_analytic_infeasible(
                problem, result, device, boundary, used_bytes
            );
        }
        if (apply_cut_and_repressure(problem, options, result, workspace, &chosen) != 0) {
            shadowspill_residency_clear_touched(workspace);
            return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
        }
    }
}
