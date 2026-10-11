/* One cut: what it would take out, and what that is worth. */
#include "internal.h"

int shadowspill_residency_compare_score(
    const CutScore *left,
    const CutScore *right,
    int minimize_transfer
) {
    if (!minimize_transfer && left->exposed_ns != right->exposed_ns) {
        return left->exposed_ns < right->exposed_ns ? -1 : 1;
    }
    for (uint32_t index = 0U; index < 7U; ++index) {
        if (left->values[index] == right->values[index]) {
            continue;
        }
        return left->values[index] < right->values[index] ? -1 : 1;
    }
    return 0;
}

CutScore shadowspill_residency_score_cut(
    const ShadowSpillPressureFitResidencyProblem *problem,
    const ResidencyCut *cut,
    int minimize_transfer
) {
    uint32_t alias = cut->alias;
    int32_t departure = cut->start - 1;
    int32_t entry = cut->end + 1;
    int writeback = problem->alias_retain_spill_copy[alias] == 0U;
    if (!writeback && departure >= -1) {
        writeback = shadowspill_written_by(problem, alias, (uint32_t)(departure + 1));
    }
    uint64_t fetch_ns = problem->fetch_runtime_ns[alias];
    uint64_t evict_ns = writeback ? problem->evict_runtime_ns[alias] : 0U;
    uint64_t departure_ns =
        departure >= 0 ? problem->task_ideal_end_ns[departure] : 0U;
    int32_t last_task = (int32_t)problem->boundary_count - 2;
    int32_t deadline_task = entry + 1;
    if (deadline_task > last_task) {
        deadline_task = last_task;
    }
    uint64_t deadline_ns = deadline_task > 0
        ? problem->task_ideal_end_ns[deadline_task - 1]
        : 0U;
    uint64_t finish_ns = departure_ns + evict_ns + fetch_ns;
    uint64_t exposed_ns = finish_ns > deadline_ns ? finish_ns - deadline_ns : 0U;
    int64_t length = cut->end >= cut->start
        ? (int64_t)cut->end - cut->start + 1
        : 0;
    CutScore score = {
        .exposed_ns = minimize_transfer ? 0U : exposed_ns,
        .values = {
            writeback,
            cut->start <= -1 ? -1 : 0,
            -(int64_t)problem->first_input_task[alias],
            -(int64_t)problem->alias_size_bytes[alias],
            -length,
            (int64_t)alias,
            (int64_t)cut->start,
        },
    };
    return score;
}

int shadowspill_residency_candidate_cut(
    const ShadowSpillPressureFitResidencyProblem *problem,
    const ShadowSpillResidency *residency,
    const uint32_t *first_required,
    const uint32_t *run_offsets,
    const int32_t *run_bounds,
    uint32_t alias,
    int32_t boundary,
    ResidencyCut *cut
) {
    uint32_t boundary_index = (uint32_t)(boundary + 1);
    const uint32_t span = shadowspill_residency_span_index(residency, alias, boundary_index);
    if (span == UINT32_MAX) {
        return 0;
    }
    if (shadowspill_residency_resident_at(residency, alias, 0U) &&
        problem->initial_location[alias] == 1 &&
        shadowspill_anchor_at(problem, alias, 0U) == UINT32_MAX &&
        first_required[alias] != UINT32_MAX &&
        boundary_index < first_required[alias]) {
        *cut = (ResidencyCut){
            .alias = alias,
            .start = -1,
            .end = (int32_t)first_required[alias] - 2,
        };
        return 1;
    }

    int32_t start = boundary;
    int32_t end = boundary;
    const uint32_t anchor = shadowspill_anchor_at(problem, alias, boundary_index);
    if (anchor == UINT32_MAX) {
        uint32_t low = run_offsets[alias];
        uint32_t high = run_offsets[alias + 1U];
        int inside = 0;
        while (low < high) {
            uint32_t middle = low + (high - low) / 2U;
            int32_t run_start = run_bounds[2U * middle];
            int32_t run_end = run_bounds[2U * middle + 1U];
            if ((int32_t)boundary_index < run_start) {
                high = middle;
            } else if ((int32_t)boundary_index > run_end) {
                low = middle + 1U;
            } else {
                start = run_start - 1;
                end = run_end - 1;
                inside = 1;
                break;
            }
        }
        if (inside == 0) {
            return 0;
        }
    } else {
        uint32_t latest = problem->anchor_tasks[anchor];
        /* Connected after: the next cell is resident and in this same span,
         * which is what no break between them means. */
        int connected_after =
            residency->aliases[alias].bounds[2U * span + 1U] > boundary_index;
        int can_split = connected_after &&
            (latest == UINT32_MAX || (int32_t)latest <= boundary);
        if (!can_split) {
            return 0;
        }
        start = boundary + 1;
        end = boundary;
    }
    if (start <= -1) {
        return 0;
    }
    *cut = (ResidencyCut){
        .alias = alias,
        .start = start,
        .end = end,
    };
    return 1;
}

int shadowspill_residency_same_cut(const ResidencyCut *left, const ResidencyCut *right) {
    return left->alias == right->alias && left->start == right->start &&
        left->end == right->end;
}

/* The best cut still available at one boundary.
 *
 * The cuts relieving the boundary sit on the nodes from its leaf to the
 * root, each node's in score order behind a cursor that has passed the
 * cuts gone inactive. The answer is the best of the nodes' first available
 * cuts, ties to the lower cut id, which is the order one merged list would
 * have had; the cursors move only past inactive cuts, so a cut is skipped
 * at a node exactly when it would have been skipped in that list. */
int shadowspill_residency_select_cut(
    const ShadowSpillPressureFitResidencyProblem *problem,
    uint32_t device,
    int32_t boundary,
    int minimize_transfer,
    const CutIndex *index,
    const uint8_t *active,
    uint32_t *cursors,
    ResidencyCut *selected,
    uint32_t *selected_id
) {
    const uint32_t boundary_index = (uint32_t)(boundary + 1);
    const uint32_t *refs = index->refs[minimize_transfer != 0];
    const uint64_t device_base = (uint64_t)device * index->node_count;
    uint32_t best = UINT32_MAX;
    for (uint32_t node = index->leaf_base + boundary_index; node >= 1U; node >>= 1U) {
        const uint64_t slot = device_base + node;
        const uint64_t begin = index->node_offsets[slot];
        const uint64_t end = index->node_offsets[slot + 1U];
        uint64_t ref = begin + cursors[slot];
        while (ref < end && active[refs[ref]] == 0U) {
            ++ref;
        }
        cursors[slot] = (uint32_t)(ref - begin);
        if (ref == end) {
            continue;
        }
        const uint32_t cut_id = refs[ref];
        if (best == UINT32_MAX) {
            best = cut_id;
            continue;
        }
        const int comparison = shadowspill_residency_compare_score(
            &index->cuts[cut_id].score, &index->cuts[best].score, minimize_transfer
        );
        if (comparison < 0 || (comparison == 0 && cut_id < best)) {
            best = cut_id;
        }
    }
    (void)problem;
    if (best == UINT32_MAX) {
        return 0;
    }
    *selected = index->cuts[best].cut;
    *selected_id = best;
    return 1;
}

void shadowspill_residency_refresh_alias_candidates(
    const ShadowSpillPressureFitResidencyProblem *problem,
    const ShadowSpillResidency *residency,
    const uint32_t *first_required,
    const uint32_t *run_offsets,
    const int32_t *run_bounds,
    const CutIndex *index,
    uint8_t *active,
    uint32_t alias
) {
    uint32_t begin = index->alias_offsets[alias];
    uint32_t end = index->alias_offsets[alias + 1U];
    for (uint32_t cut_id = begin; cut_id < end; ++cut_id) {
        const IndexedCut *indexed = &index->cuts[cut_id];
        ResidencyCut current;
        int valid = shadowspill_residency_candidate_cut(
            problem,
            residency,
            first_required,
            run_offsets,
            run_bounds,
            alias,
            (int32_t)indexed->first_boundary - 1,
            &current
        );
        active[cut_id] = valid != 0 && shadowspill_residency_same_cut(&current, &indexed->cut);
    }
}

int shadowspill_residency_apply_cut(
    ShadowSpillResidency *residency,
    const ResidencyCut *cut
) {
    if (cut->start > cut->end) {
        return shadowspill_residency_split_after(
            residency, cut->alias, (uint32_t)(cut->end + 1)
        );
    }
    return shadowspill_residency_clear_cells(
        residency, cut->alias, (uint32_t)(cut->start + 1), (uint32_t)(cut->end + 1)
    );
}
