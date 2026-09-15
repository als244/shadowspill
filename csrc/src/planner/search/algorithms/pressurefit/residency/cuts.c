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
        uint32_t index = (uint32_t)(departure + 1);
        writeback = problem->write_prefix[shadowspill_residency_cell(
            alias,
            problem->boundary_count,
            index
        )] != 0U;
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
    const uint8_t *resident,
    const uint8_t *breaks,
    const uint32_t *first_required,
    const uint32_t *run_offsets,
    const int32_t *run_bounds,
    uint32_t alias,
    int32_t boundary,
    ResidencyCut *cut
) {
    uint32_t boundary_index = (uint32_t)(boundary + 1);
    uint64_t position = shadowspill_residency_cell(alias, problem->boundary_count, boundary_index);
    if (!shadowspill_cell_get(resident, position)) {
        return 0;
    }
    if (shadowspill_cell_get(resident, shadowspill_residency_cell(alias, problem->boundary_count, 0U)) &&
        problem->initial_location[alias] == 1 &&
        problem->anchors[shadowspill_residency_cell(alias, problem->boundary_count, 0U)] == 0U &&
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
    if (problem->anchors[position] == 0U) {
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
        uint32_t latest = problem->latest_access_task[position];
        int connected_after = boundary_index + 1U < problem->boundary_count &&
            shadowspill_cell_get(
                resident,
                shadowspill_residency_cell(alias, problem->boundary_count, boundary_index + 1U)
            ) &&
            !shadowspill_cell_get(breaks, position);
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

int shadowspill_residency_select_cut(
    const ShadowSpillPressureFitResidencyProblem *problem,
    uint32_t device,
    int32_t boundary,
    int minimize_transfer,
    const CutIndex *index,
    const uint8_t *active,
    uint64_t *cursors,
    ResidencyCut *selected
) {
    uint32_t boundary_index = (uint32_t)(boundary + 1);
    uint64_t position = (uint64_t)device * problem->boundary_count + boundary_index;
    uint64_t begin = index->offsets[position];
    uint64_t end = index->offsets[position + 1U];
    const uint32_t *refs = index->refs[minimize_transfer != 0];
    uint64_t ref = begin + cursors[position];
    for (; ref < end; ++ref) {
        uint32_t cut_id = refs[ref];
        if (active[cut_id] == 0U) {
            continue;
        }
        cursors[position] = ref - begin + 1U;
        *selected = index->cuts[cut_id].cut;
        return 1;
    }
    cursors[position] = end - begin;
    return 0;
}

void shadowspill_residency_refresh_alias_candidates(
    const ShadowSpillPressureFitResidencyProblem *problem,
    const uint8_t *resident,
    const uint8_t *breaks,
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
            resident,
            breaks,
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

void shadowspill_residency_apply_cut(
    const ShadowSpillPressureFitResidencyProblem *problem,
    uint8_t *resident,
    uint8_t *breaks,
    const ResidencyCut *cut
) {
    if (cut->start > cut->end) {
        uint32_t after = (uint32_t)(cut->end + 1);
        shadowspill_cell_set(breaks, shadowspill_residency_cell(cut->alias, problem->boundary_count, after), 1);
        return;
    }
    for (int32_t boundary = cut->start; boundary <= cut->end; ++boundary) {
        uint32_t index = (uint32_t)(boundary + 1);
        shadowspill_cell_set(resident, shadowspill_residency_cell(cut->alias, problem->boundary_count, index), 0);
    }
}
