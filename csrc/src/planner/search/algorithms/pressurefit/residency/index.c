/* Every candidate cut, ordered so the best is cheap to find. */
#include "internal.h"

void shadowspill_residency_destroy_cut_index(CutIndex *index) {
    free(index->cuts);
    free(index->alias_offsets);
    free(index->offsets);
    free(index->refs[0]);
    free(index->refs[1]);
    memset(index, 0, sizeof(*index));
}

static int append_indexed_cut(
    const ShadowSpillPressureFitResidencyProblem *problem,
    CutIndex *index,
    ResidencyCut cut,
    uint32_t first_boundary,
    uint32_t last_boundary
) {
    if (index->cut_count == index->cut_capacity) {
        uint32_t next = index->cut_capacity == 0U
            ? 256U
            : index->cut_capacity * 2U;
        if (next < index->cut_capacity) {
            return -1;
        }
        void *storage = realloc(index->cuts, (size_t)next * sizeof(*index->cuts));
        if (storage == NULL) {
            return -1;
        }
        index->cuts = storage;
        index->cut_capacity = next;
    }
    index->cuts[index->cut_count++] = (IndexedCut){
        .cut = cut,
        .score = shadowspill_residency_score_cut(problem, &cut, 0),
        .first_boundary = first_boundary,
        .last_boundary = last_boundary,
    };
    return 0;
}

static _Thread_local const IndexedCut *sort_cuts;

static _Thread_local int sort_minimize_transfer;

static int cut_ref_compare(const void *left_value, const void *right_value) {
    uint32_t left = *(const uint32_t *)left_value;
    uint32_t right = *(const uint32_t *)right_value;
    int comparison = shadowspill_residency_compare_score(
        &sort_cuts[left].score,
        &sort_cuts[right].score,
        sort_minimize_transfer
    );
    if (comparison != 0) {
        return comparison;
    }
    return left < right ? -1 : left != right;
}

static int build_cut_index(
    const ShadowSpillPressureFitResidencyProblem *problem,
    ShadowSpillPressureFitResidencyWorkspace *workspace
) {
    CutIndex *index = &workspace->cut_index;
    shadowspill_residency_destroy_cut_index(index);

    const uint8_t *resident = workspace->seed_resident;
    const uint8_t *breaks = workspace->seed_breaks;
    uint32_t boundaries = problem->boundary_count;
    index->alias_offsets = malloc(
        ((size_t)problem->alias_count + 1U) * sizeof(*index->alias_offsets)
    );
    if (index->alias_offsets == NULL) {
        shadowspill_residency_destroy_cut_index(index);
        return -1;
    }
    for (uint32_t alias = 0U; alias < problem->alias_count; ++alias) {
        index->alias_offsets[alias] = index->cut_count;
        if (!shadowspill_alias_may_cut(problem, alias)) {
            index->alias_offsets[alias + 1U] = index->cut_count;
            continue;
        }
        int active = 0;
        ResidencyCut current = {0};
        uint32_t first = 0U;
        uint32_t last = 0U;
        for (uint32_t boundary = 0U; boundary < boundaries; ++boundary) {
            ResidencyCut candidate;
            int valid = shadowspill_residency_candidate_cut(
                problem,
                resident,
                breaks,
                workspace->first_required,
                workspace->run_offsets,
                workspace->run_bounds,
                alias,
                (int32_t)boundary - 1,
                &candidate
            );
            if (valid != 0 && active != 0 && shadowspill_residency_same_cut(&candidate, &current) &&
                boundary == last + 1U) {
                last = boundary;
                continue;
            }
            if (active != 0 && append_indexed_cut(
                    problem,
                    index,
                    current,
                    first,
                    last
                ) != 0) {
                shadowspill_residency_destroy_cut_index(index);
                return -1;
            }
            active = valid;
            if (valid != 0) {
                current = candidate;
                first = boundary;
                last = boundary;
            }
        }
        if (active != 0 && append_indexed_cut(
                problem,
                index,
                current,
                first,
                last
            ) != 0) {
            shadowspill_residency_destroy_cut_index(index);
            return -1;
        }
        index->alias_offsets[alias + 1U] = index->cut_count;
    }

    uint64_t index_cells =
        (uint64_t)problem->device_count * problem->boundary_count;
    if (index_cells > SIZE_MAX / sizeof(*index->offsets)) {
        shadowspill_residency_destroy_cut_index(index);
        return -1;
    }
    index->offsets = calloc(
        (size_t)index_cells + 1U,
        sizeof(*index->offsets)
    );
    if (index->offsets == NULL) {
        shadowspill_residency_destroy_cut_index(index);
        return -1;
    }
    for (uint32_t cut_id = 0U; cut_id < index->cut_count; ++cut_id) {
        const IndexedCut *item = &index->cuts[cut_id];
        uint32_t device = problem->alias_device[item->cut.alias];
        for (uint32_t boundary = item->first_boundary;
             boundary <= item->last_boundary;
             ++boundary) {
            uint64_t position = (uint64_t)device * boundaries + boundary;
            ++index->offsets[position + 1U];
        }
    }
    for (uint64_t position = 0U; position < index_cells; ++position) {
        index->offsets[position + 1U] += index->offsets[position];
    }
    index->ref_count = index->offsets[index_cells];
    if (index->ref_count > SIZE_MAX / sizeof(*index->refs[0])) {
        shadowspill_residency_destroy_cut_index(index);
        return -1;
    }
    index->refs[0] = malloc(
        (index->ref_count == 0U ? 1U : (size_t)index->ref_count) *
        sizeof(*index->refs[0])
    );
    index->refs[1] = malloc(
        (index->ref_count == 0U ? 1U : (size_t)index->ref_count) *
        sizeof(*index->refs[1])
    );
    uint64_t *cursor = malloc(
        (index_cells == 0U ? 1U : (size_t)index_cells) * sizeof(*cursor)
    );
    uint32_t *ranked = malloc(
        (index->cut_count == 0U ? 1U : (size_t)index->cut_count) *
        sizeof(*ranked)
    );
    if (index->refs[0] == NULL || index->refs[1] == NULL || cursor == NULL ||
        ranked == NULL) {
        free(cursor);
        free(ranked);
        shadowspill_residency_destroy_cut_index(index);
        return -1;
    }
    for (uint32_t mode = 0U; mode < 2U; ++mode) {
        for (uint32_t cut_id = 0U; cut_id < index->cut_count; ++cut_id) {
            ranked[cut_id] = cut_id;
        }
        sort_cuts = index->cuts;
        sort_minimize_transfer = mode != 0U;
        qsort(ranked, index->cut_count, sizeof(*ranked), cut_ref_compare);
        sort_cuts = NULL;
        memcpy(cursor, index->offsets, (size_t)index_cells * sizeof(*cursor));
        for (uint32_t rank = 0U; rank < index->cut_count; ++rank) {
            uint32_t cut_id = ranked[rank];
            const IndexedCut *item = &index->cuts[cut_id];
            uint32_t device = problem->alias_device[item->cut.alias];
            for (uint32_t boundary = item->first_boundary;
                 boundary <= item->last_boundary;
                 ++boundary) {
                uint64_t position = (uint64_t)device * boundaries + boundary;
                index->refs[mode][cursor[position]++] = cut_id;
            }
        }
    }
    free(cursor);
    free(ranked);
    return 0;
}

static int append_run(
    ShadowSpillPressureFitResidencyWorkspace *workspace, uint32_t start, uint32_t end
) {
    if (workspace->run_count == workspace->run_capacity) {
        uint64_t capacity = workspace->run_capacity == 0U
            ? 1024U
            : workspace->run_capacity * 2U;
        int32_t *grown = realloc(
            workspace->run_bounds, (size_t)capacity * 2U * sizeof(*grown)
        );
        if (grown == NULL) {
            return -1;
        }
        workspace->run_bounds = grown;
        workspace->run_capacity = capacity;
    }
    workspace->run_bounds[2U * workspace->run_count] = (int32_t)start;
    workspace->run_bounds[2U * workspace->run_count + 1U] = (int32_t)end;
    ++workspace->run_count;
    return 0;
}

/* Record, per alias, the first anchored boundary and the removable runs of
 * the residency: the maximal stretches of a span that no anchor touches. An
 * alias that may not be cut has neither. */
static int build_cut_geometry(
    const ShadowSpillPressureFitResidencyProblem *problem,
    const uint8_t *resident,
    const uint8_t *breaks,
    ShadowSpillPressureFitResidencyWorkspace *workspace
) {
    uint32_t count = problem->boundary_count;
    workspace->run_count = 0U;
    for (uint32_t alias = 0U; alias < problem->alias_count; ++alias) {
        workspace->run_offsets[alias] = (uint32_t)workspace->run_count;
        workspace->first_required[alias] = UINT32_MAX;
        if (!shadowspill_alias_may_cut(problem, alias)) {
            continue;
        }
        for (uint32_t index = 1U; index < count; ++index) {
            if (problem->anchors[shadowspill_residency_cell(alias, count, index)] != 0U) {
                workspace->first_required[alias] = index;
                break;
            }
        }
        uint32_t span_cursor = 0U;
        while (span_cursor < count) {
            uint32_t span_start = 0U;
            uint32_t span_end = 0U;
            if (!shadowspill_residency_next_span(
                    resident,
                    breaks,
                    alias,
                    count,
                    &span_cursor,
                    &span_start,
                    &span_end
                )) {
                break;
            }
            uint32_t cursor = span_start;
            while (cursor <= span_end) {
                if (problem->anchors[shadowspill_residency_cell(alias, count, cursor)] != 0U) {
                    ++cursor;
                    continue;
                }
                uint32_t run_start = cursor;
                while (cursor < span_end &&
                       problem->anchors[shadowspill_residency_cell(alias, count, cursor + 1U)] == 0U) {
                    ++cursor;
                }
                if (append_run(workspace, run_start, cursor) != 0) {
                    return -1;
                }
                ++cursor;
            }
        }
    }
    workspace->run_offsets[problem->alias_count] = (uint32_t)workspace->run_count;
    return 0;
}

int shadowspill_residency_prepare_seed_geometry(
    const ShadowSpillPressureFitResidencyProblem *problem,
    const ShadowSpillPressureFitResidencyOptions *options,
    ShadowSpillPressureFitResidencyWorkspace *workspace
) {
    if (workspace->geometry_problem == problem) {
        return 0;
    }
    workspace->geometry_problem = NULL;
    workspace->seed_resident = options->seed_resident;
    workspace->seed_breaks = options->seed_breaks;
    if (build_cut_geometry(
            problem,
            workspace->seed_resident,
            workspace->seed_breaks,
            workspace
        ) != 0 ||
        build_cut_index(problem, workspace) != 0) {
        return -1;
    }
    workspace->geometry_problem = problem;
    workspace->pressure_valid[0] = 0U;
    workspace->pressure_valid[1] = 0U;
    return 0;
}

int shadowspill_residency_prepare_base_pressure(
    const ShadowSpillPressureFitResidencyProblem *problem,
    const ShadowSpillPressureFitResidencyOptions *options,
    ShadowSpillPressureFitResidencyWorkspace *workspace
) {
    uint32_t variant = options->fetch_headroom != 0U ? 1U : 0U;
    if (workspace->pressure_valid[variant] != 0U) {
        return 0;
    }
    size_t pressure_cells =
        (size_t)problem->device_count * problem->boundary_count;
    uint64_t *pressure = workspace->base_pressure[variant];
    memset(pressure, 0, pressure_cells * sizeof(*pressure));
    /* An alias the reducer may not cut lives in the resident slice, which
     * the capacity here already excludes, so it adds no pressure. */
    for (uint32_t alias = 0U; alias < problem->alias_count; ++alias) {
        if (!shadowspill_alias_may_cut(problem, alias)) {
            continue;
        }
        shadowspill_residency_alias_contribution(
            problem,
            options,
            workspace->seed_resident,
            workspace->seed_breaks,
            alias,
            workspace->before
        );
        uint32_t device = problem->alias_device[alias];
        for (uint32_t boundary = 0U; boundary < problem->boundary_count;
             ++boundary) {
            if (workspace->before[boundary] != 0U) {
                pressure[(uint64_t)device * problem->boundary_count + boundary] +=
                    problem->alias_size_bytes[alias];
            }
        }
    }
    workspace->pressure_valid[variant] = 1U;
    return 0;
}
