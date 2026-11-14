/* One cut: what it would take out, and what that is worth. */
#include "internal.h"

int shadowspill_residency_compare_score(
    const CutScore *left, const CutScore *right, int minimize_transfer
) {
    if (!minimize_transfer && left->exposed_ns != right->exposed_ns) {
        return left->exposed_ns < right->exposed_ns ? -1 : 1;
    }
    for (uint32_t index = 0U; index < 4U; ++index) {
        if (index == 1U) {
            /* Maximize signed slack after writeback, without narrowing it. */
            if (left->slack_high != right->slack_high) {
                return left->slack_high > right->slack_high ? -1 : 1;
            }
            if (left->slack_low != right->slack_low) {
                return left->slack_low > right->slack_low ? -1 : 1;
            }
        }
        if (left->values[index] != right->values[index]) {
            return left->values[index] < right->values[index] ? -1 : 1;
        }
    }
    return 0;
}

CutScore shadowspill_residency_score_cut(
    const ShadowSpillPressureFitResidencyProblem *problem, const ResidencyCut *cut
) {
    const uint32_t alias = cut->alias;
    /* A prefix cut leaves an already-spilled object in spill. Writeback is
     * conservative: earlier cuts do not refresh the retained-copy history. */
    const int writeback = cut->start >= 0 &&
        (problem->alias_retain_spill_copy[alias] == 0U ||
         shadowspill_written_by(problem, alias, (uint32_t)cut->start));
    return (CutScore){
        /* Exposure and slack are evaluated at the selected boundary. */
        .values = {writeback, -(int64_t)problem->alias_size_bytes[alias],
                   (int64_t)alias, (int64_t)cut->start},
    };
}

CutTiming shadowspill_residency_cut_timing(
    const ShadowSpillPressureFitResidencyProblem *problem, const ResidencyCut *cut
) {
    const uint32_t alias = cut->alias;
    const uint32_t begin = problem->anchor_offsets[alias];
    const uint32_t end = problem->anchor_offsets[alias + 1U];
    const uint32_t tasks = problem->boundary_count - 1U;
    CutTiming timing = {
        .next_use_begin_ns = tasks == 0U ? 0U : problem->task_ideal_end_ns[tasks - 1U],
        .fetch_ns = problem->fetch_runtime_ns[alias],
    };
    /* Input anchors sit before their task; produced-output anchors sit after
     * it. Read the task identity, rather than subtracting one from a cell. */
    if (cut->start >= 0) {
        uint32_t anchor = shadowspill_anchor_lower_bound(
            problem->anchor_positions, begin, end, (uint32_t)cut->start + 1U);
        while (anchor > begin) {
            const uint32_t task = problem->anchor_tasks[--anchor];
            if (task != UINT32_MAX) {
                timing.prior_use_end_ns = problem->task_ideal_end_ns[task];
                break;
            }
        }
        if (problem->alias_retain_spill_copy[alias] == 0U ||
            shadowspill_written_by(problem, alias, (uint32_t)cut->start)) {
            timing.evict_ns = problem->evict_runtime_ns[alias];
        }
    }
    uint32_t anchor = shadowspill_anchor_lower_bound(
        problem->anchor_positions, begin, end, (uint32_t)(cut->end + 2));
    for (; anchor < end; ++anchor) {
        const uint32_t task = problem->anchor_earliest[anchor];
        if (task != UINT32_MAX) {
            timing.next_use_begin_ns = task == 0U ? 0U : problem->task_ideal_end_ns[task - 1U];
            break;
        }
    }
    return timing;
}

CutScore shadowspill_residency_score_at(
    const ShadowSpillPressureFitResidencyProblem *problem,
    const IndexedCut *cut, uint32_t boundary, int minimize_transfer
) {
    CutScore score = cut->score;
    const uint32_t tasks = problem->boundary_count - 1U;
    const uint64_t begin = boundary == 0U ? 0U : problem->task_ideal_end_ns[boundary - 1U];
    const uint64_t end = boundary < tasks ? problem->task_ideal_end_ns[boundary] : begin;
    const CutTiming *timing = &cut->timing;
    const uint64_t evict_slack = begin > timing->prior_use_end_ns
        ? begin - timing->prior_use_end_ns : 0U;
    const uint64_t fetch_slack = timing->next_use_begin_ns > end
        ? timing->next_use_begin_ns - end : 0U;
    score.evict_exposed_ns = timing->evict_ns > evict_slack
        ? timing->evict_ns - evict_slack : 0U;
    score.fetch_exposed_ns = timing->fetch_ns > fetch_slack
        ? timing->fetch_ns - fetch_slack : 0U;
    score.exposed_ns = UINT64_MAX - score.evict_exposed_ns < score.fetch_exposed_ns
        ? UINT64_MAX : score.evict_exposed_ns + score.fetch_exposed_ns;
    const uint64_t windows = evict_slack + fetch_slack;
    const uint64_t transfers = timing->evict_ns + timing->fetch_ns;
    score.slack_low = windows - transfers;
    score.slack_high = (int64_t)(windows < evict_slack) -
        (int64_t)(transfers < timing->evict_ns) - (int64_t)(windows < transfers);
    if (minimize_transfer) { score.exposed_ns = 0U; }
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

/* Exposure and slack both depend on the selected boundary. Examine every
 * active cut on its covering nodes; zero exposure alone cannot settle ties.
 * The sparse index retains O(cuts log boundaries) storage. */
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
    CutScore best_score = {0};
    for (uint32_t node = index->leaf_base + boundary_index; node >= 1U; node >>= 1U) {
        const uint64_t slot = device_base + node;
        const uint64_t begin = index->node_offsets[slot];
        const uint64_t end = index->node_offsets[slot + 1U];
        uint64_t ref = begin + cursors[slot];
        while (ref < end && active[refs[ref]] == 0U) {
            ++ref;
        }
        cursors[slot] = (uint32_t)(ref - begin);
        for (; ref < end; ++ref) {
            const uint32_t cut_id = refs[ref];
            if (active[cut_id] == 0U) { continue; }
            const CutScore score = shadowspill_residency_score_at(
                problem, &index->cuts[cut_id], boundary_index, minimize_transfer);
            const int comparison = best == UINT32_MAX ? -1 :
                shadowspill_residency_compare_score(
                    &score, &best_score, minimize_transfer);
            if (comparison < 0 || (comparison == 0 && cut_id < best)) {
                best = cut_id;
                best_score = score;
            }

        }
    }
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
