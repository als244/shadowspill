#include "internal.h"
#include "residency_internal.h"

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct ResidencyCut {
    uint32_t alias;
    int32_t start;
    int32_t end;
} ResidencyCut;

typedef struct CutScore {
    uint64_t exposed_ns;
    int64_t values[7];
} CutScore;

typedef struct IndexedCut {
    ResidencyCut cut;
    CutScore score;
    uint32_t first_boundary;
    uint32_t last_boundary;
} IndexedCut;

typedef struct CutIndex {
    IndexedCut *cuts;
    uint32_t cut_count;
    uint32_t cut_capacity;
    uint32_t *alias_offsets;
    uint64_t *offsets;
    uint32_t *refs[2];
    uint64_t ref_count;
} CutIndex;

struct ShadowSpillResidencyWorkspace {
    uint32_t alias_count;
    uint32_t boundary_count;
    uint32_t device_count;
    uint64_t *pressure;
    uint8_t *before;
    uint8_t *after;
    uint32_t *first_required;
    int32_t *gap_start;
    int32_t *gap_end;
    uint8_t *seed_resident;
    uint8_t *seed_breaks;
    uint64_t *base_pressure[2];
    uint64_t *cut_cursors;
    uint8_t *cut_active;
    uint32_t cut_active_capacity;
    CutIndex cut_index;
    uint8_t geometry_valid;
    uint8_t pressure_valid[2];
};

static uint64_t cell(uint32_t alias, uint32_t boundary_count, uint32_t index) {
    return (uint64_t)alias * boundary_count + index;
}

static int next_span(
    const uint8_t *resident,
    const uint8_t *breaks,
    uint32_t alias,
    uint32_t boundary_count,
    uint32_t *cursor,
    uint32_t *start,
    uint32_t *end
) {
    while (*cursor < boundary_count &&
           resident[cell(alias, boundary_count, *cursor)] == 0U) {
        ++*cursor;
    }
    if (*cursor == boundary_count) {
        return 0;
    }
    uint32_t left = *cursor;
    uint32_t right = *cursor;
    while (right + 1U < boundary_count) {
        uint64_t current = cell(alias, boundary_count, right);
        uint64_t next = cell(alias, boundary_count, right + 1U);
        if (resident[next] == 0U || breaks[current] != 0U) {
            break;
        }
        ++right;
    }
    *start = left;
    *end = right;
    *cursor = right + 1U;
    return 1;
}

static void alias_contribution(
    const ShadowSpillResidencyProblem *problem,
    const ShadowSpillResidencyOptions *options,
    const uint8_t *resident,
    const uint8_t *breaks,
    uint32_t alias,
    uint8_t *contribution
) {
    uint32_t count = problem->boundary_count;
    memset(contribution, 0, count);
    uint32_t cursor = 0U;
    while (cursor < count) {
        uint32_t start = 0U;
        uint32_t end = 0U;
        if (!next_span(
                resident,
                breaks,
                alias,
                count,
                &cursor,
                &start,
                &end
            )) {
            break;
        }
        uint32_t charged_start = start;
        if (options->prefetch_headroom != 0U && start > 0U &&
            problem->productions[cell(alias, count, start)] == 0U) {
            --charged_start;
        }
        int32_t charged_end = (int32_t)end;
        int32_t end_boundary = (int32_t)end - 1;
        if (end_boundary >= 0 && problem->final_location[alias] != 0) {
            int future_access = 0;
            for (uint32_t anchor = start; anchor <= end; ++anchor) {
                uint32_t task =
                    problem->latest_access_task[cell(alias, count, anchor)];
                if (task != UINT32_MAX && (int32_t)task > end_boundary) {
                    future_access = 1;
                    break;
                }
            }
            if (!future_access) {
                --charged_end;
            }
        }
        if (charged_end >= (int32_t)charged_start) {
            for (uint32_t boundary = charged_start;
                 boundary <= (uint32_t)charged_end;
                 ++boundary) {
                contribution[boundary] = 1U;
            }
        }
    }
    for (uint32_t boundary = 0U; boundary < count; ++boundary) {
        if (problem->output_reservations[cell(alias, count, boundary)] != 0U) {
            contribution[boundary] = 1U;
        }
    }
}

int shadowspill_residency_pressure_at(
    const ShadowSpillResidencyProblem *problem,
    const ShadowSpillResidencyOptions *options,
    const uint8_t *resident,
    const uint8_t *breaks,
    uint32_t device,
    uint32_t boundary,
    ShadowSpillResidencyWorkspace *workspace,
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
        if (problem->alias_device[alias] != device) {
            continue;
        }
        alias_contribution(
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

static int compare_score(
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

static CutScore score_cut(
    const ShadowSpillResidencyProblem *problem,
    const ResidencyCut *cut,
    int minimize_transfer
) {
    uint32_t alias = cut->alias;
    int32_t departure = cut->start - 1;
    int32_t entry = cut->end + 1;
    int writeback = problem->alias_retain_spill_copy[alias] == 0U;
    if (!writeback && departure >= -1) {
        uint32_t index = (uint32_t)(departure + 1);
        writeback = problem->write_prefix[cell(
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

static int candidate_cut(
    const ShadowSpillResidencyProblem *problem,
    const uint8_t *resident,
    const uint8_t *breaks,
    const uint32_t *first_required,
    const int32_t *gap_start,
    const int32_t *gap_end,
    uint32_t alias,
    int32_t boundary,
    ResidencyCut *cut
) {
    uint32_t boundary_index = (uint32_t)(boundary + 1);
    uint64_t position = cell(alias, problem->boundary_count, boundary_index);
    if (resident[position] == 0U) {
        return 0;
    }
    if (resident[cell(alias, problem->boundary_count, 0U)] != 0U &&
        problem->initial_location[alias] == 1 &&
        problem->anchors[cell(alias, problem->boundary_count, 0U)] == 0U &&
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
        start = gap_start[position];
        end = gap_end[position];
        if (start == INT32_MIN) {
            return 0;
        }
    } else {
        uint32_t latest = problem->latest_access_task[position];
        int connected_after = boundary_index + 1U < problem->boundary_count &&
            resident[cell(
                alias,
                problem->boundary_count,
                boundary_index + 1U
            )] != 0U && breaks[position] == 0U;
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

static int same_cut(const ResidencyCut *left, const ResidencyCut *right) {
    return left->alias == right->alias && left->start == right->start &&
        left->end == right->end;
}

static void destroy_cut_index(CutIndex *index) {
    free(index->cuts);
    free(index->alias_offsets);
    free(index->offsets);
    free(index->refs[0]);
    free(index->refs[1]);
    memset(index, 0, sizeof(*index));
}

static int append_indexed_cut(
    const ShadowSpillResidencyProblem *problem,
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
        .score = score_cut(problem, &cut, 0),
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
    int comparison = compare_score(
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
    const ShadowSpillResidencyProblem *problem,
    ShadowSpillResidencyWorkspace *workspace
) {
    CutIndex *index = &workspace->cut_index;
    destroy_cut_index(index);

    const uint8_t *resident = workspace->seed_resident;
    const uint8_t *breaks = workspace->seed_breaks;
    uint32_t boundaries = problem->boundary_count;
    index->alias_offsets = malloc(
        ((size_t)problem->alias_count + 1U) * sizeof(*index->alias_offsets)
    );
    if (index->alias_offsets == NULL) {
        destroy_cut_index(index);
        return -1;
    }
    for (uint32_t alias = 0U; alias < problem->alias_count; ++alias) {
        index->alias_offsets[alias] = index->cut_count;
        int active = 0;
        ResidencyCut current = {0};
        uint32_t first = 0U;
        uint32_t last = 0U;
        for (uint32_t boundary = 0U; boundary < boundaries; ++boundary) {
            ResidencyCut candidate;
            int valid = candidate_cut(
                problem,
                resident,
                breaks,
                workspace->first_required,
                workspace->gap_start,
                workspace->gap_end,
                alias,
                (int32_t)boundary - 1,
                &candidate
            );
            if (valid != 0 && active != 0 && same_cut(&candidate, &current) &&
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
                destroy_cut_index(index);
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
            destroy_cut_index(index);
            return -1;
        }
        index->alias_offsets[alias + 1U] = index->cut_count;
    }

    uint64_t index_cells =
        (uint64_t)problem->device_count * problem->boundary_count;
    if (index_cells > SIZE_MAX / sizeof(*index->offsets)) {
        destroy_cut_index(index);
        return -1;
    }
    index->offsets = calloc(
        (size_t)index_cells + 1U,
        sizeof(*index->offsets)
    );
    if (index->offsets == NULL) {
        destroy_cut_index(index);
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
        destroy_cut_index(index);
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
        destroy_cut_index(index);
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

static int select_cut(
    const ShadowSpillResidencyProblem *problem,
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

static void refresh_alias_candidates(
    const ShadowSpillResidencyProblem *problem,
    const uint8_t *resident,
    const uint8_t *breaks,
    const uint32_t *first_required,
    const int32_t *gap_start,
    const int32_t *gap_end,
    const CutIndex *index,
    uint8_t *active,
    uint32_t alias
) {
    uint32_t begin = index->alias_offsets[alias];
    uint32_t end = index->alias_offsets[alias + 1U];
    for (uint32_t cut_id = begin; cut_id < end; ++cut_id) {
        const IndexedCut *indexed = &index->cuts[cut_id];
        ResidencyCut current;
        int valid = candidate_cut(
            problem,
            resident,
            breaks,
            first_required,
            gap_start,
            gap_end,
            alias,
            (int32_t)indexed->first_boundary - 1,
            &current
        );
        active[cut_id] = valid != 0 && same_cut(&current, &indexed->cut);
    }
}

static void build_cut_geometry(
    const ShadowSpillResidencyProblem *problem,
    const uint8_t *resident,
    const uint8_t *breaks,
    uint32_t *first_required,
    int32_t *gap_start,
    int32_t *gap_end
) {
    uint32_t count = problem->boundary_count;
    uint64_t cells = (uint64_t)problem->alias_count * count;
    for (uint64_t position = 0U; position < cells; ++position) {
        gap_start[position] = INT32_MIN;
        gap_end[position] = INT32_MIN;
    }
    for (uint32_t alias = 0U; alias < problem->alias_count; ++alias) {
        first_required[alias] = UINT32_MAX;
        for (uint32_t index = 1U; index < count; ++index) {
            if (problem->anchors[cell(alias, count, index)] != 0U) {
                first_required[alias] = index;
                break;
            }
        }

        uint32_t span_cursor = 0U;
        while (span_cursor < count) {
            uint32_t span_start = 0U;
            uint32_t span_end = 0U;
            if (!next_span(
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
                if (problem->anchors[cell(alias, count, cursor)] != 0U) {
                    ++cursor;
                    continue;
                }
                uint32_t run_start = cursor;
                while (cursor < span_end &&
                       problem->anchors[cell(alias, count, cursor + 1U)] == 0U) {
                    ++cursor;
                }
                uint32_t run_end = cursor;
                for (uint32_t value = run_start; value <= run_end; ++value) {
                    uint64_t position = cell(alias, count, value);
                    gap_start[position] = (int32_t)run_start - 1;
                    gap_end[position] = (int32_t)run_end - 1;
                }
                ++cursor;
            }
        }
    }
}

static int prepare_seed_geometry(
    const ShadowSpillResidencyProblem *problem,
    const ShadowSpillResidencyOptions *options,
    ShadowSpillResidencyWorkspace *workspace
) {
    size_t cells = (size_t)problem->alias_count * problem->boundary_count;
    if (workspace->geometry_valid != 0U &&
        memcmp(workspace->seed_resident, options->seed_resident, cells) == 0 &&
        memcmp(workspace->seed_breaks, options->seed_breaks, cells) == 0) {
        return 0;
    }
    memcpy(workspace->seed_resident, options->seed_resident, cells);
    memcpy(workspace->seed_breaks, options->seed_breaks, cells);
    build_cut_geometry(
        problem,
        workspace->seed_resident,
        workspace->seed_breaks,
        workspace->first_required,
        workspace->gap_start,
        workspace->gap_end
    );
    if (build_cut_index(problem, workspace) != 0) {
        workspace->geometry_valid = 0U;
        return -1;
    }
    workspace->geometry_valid = 1U;
    workspace->pressure_valid[0] = 0U;
    workspace->pressure_valid[1] = 0U;
    return 0;
}

static int prepare_base_pressure(
    const ShadowSpillResidencyProblem *problem,
    const ShadowSpillResidencyOptions *options,
    ShadowSpillResidencyWorkspace *workspace
) {
    uint32_t variant = options->prefetch_headroom != 0U ? 1U : 0U;
    if (workspace->pressure_valid[variant] != 0U) {
        return 0;
    }
    size_t pressure_cells =
        (size_t)problem->device_count * problem->boundary_count;
    uint64_t *pressure = workspace->base_pressure[variant];
    memset(pressure, 0, pressure_cells * sizeof(*pressure));
    for (uint32_t alias = 0U; alias < problem->alias_count; ++alias) {
        alias_contribution(
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

static void apply_cut(
    const ShadowSpillResidencyProblem *problem,
    uint8_t *resident,
    uint8_t *breaks,
    const ResidencyCut *cut
) {
    if (cut->start > cut->end) {
        uint32_t after = (uint32_t)(cut->end + 1);
        breaks[cell(cut->alias, problem->boundary_count, after)] = 1U;
        return;
    }
    for (int32_t boundary = cut->start; boundary <= cut->end; ++boundary) {
        uint32_t index = (uint32_t)(boundary + 1);
        resident[cell(cut->alias, problem->boundary_count, index)] = 0U;
    }
}

static void canonicalize_breaks(
    uint8_t *breaks,
    const uint8_t *resident,
    uint32_t alias_count,
    uint32_t boundary_count
) {
    for (uint32_t alias = 0U; alias < alias_count; ++alias) {
        uint64_t row = (uint64_t)alias * boundary_count;
        int has_later_residency = 0;
        for (uint32_t cursor = boundary_count; cursor > 0U; --cursor) {
            uint32_t index = cursor - 1U;
            if (resident[row + index] == 0U) {
                breaks[row + index] = 0U;
                continue;
            }
            int next_is_resident = index + 1U < boundary_count &&
                resident[row + index + 1U] != 0U;
            if (!next_is_resident) {
                breaks[row + index] = has_later_residency ? 1U : 0U;
            }
            has_later_residency = 1;
        }
    }
}

int shadowspill_residency_force_absent(
    const ShadowSpillResidencyProblem *problem,
    uint8_t *resident,
    uint8_t *breaks,
    uint32_t alias,
    uint32_t boundary,
    ShadowSpillResidencyWorkspace *workspace
) {
    if (problem == NULL || resident == NULL || breaks == NULL ||
        workspace == NULL || alias >= problem->alias_count ||
        boundary >= problem->boundary_count ||
        workspace->alias_count != problem->alias_count ||
        workspace->boundary_count != problem->boundary_count ||
        workspace->device_count != problem->device_count) {
        return -1;
    }
    const uint64_t position = cell(alias, problem->boundary_count, boundary);
    if (resident[position] == 0U) {
        return 2;
    }
    if (problem->anchors[position] != 0U ||
        problem->output_reservations[position] != 0U) {
        return 0;
    }
    build_cut_geometry(
        problem,
        resident,
        breaks,
        workspace->first_required,
        workspace->gap_start,
        workspace->gap_end
    );
    ResidencyCut cut;
    if (candidate_cut(
            problem,
            resident,
            breaks,
            workspace->first_required,
            workspace->gap_start,
            workspace->gap_end,
            alias,
            (int32_t)boundary - 1,
            &cut
        ) == 0 || cut.start > cut.end) {
        return 0;
    }
    apply_cut(problem, resident, breaks, &cut);
    canonicalize_breaks(
        breaks, resident, problem->alias_count, problem->boundary_count
    );
    return resident[position] == 0U ? 1 : -1;
}

int shadowspill_residency_mark_removable(
    const ShadowSpillResidencyProblem *problem,
    const uint8_t *resident,
    const uint8_t *breaks,
    uint32_t boundary,
    ShadowSpillResidencyWorkspace *workspace,
    uint8_t *removable,
    uint32_t removable_capacity
) {
    if (problem == NULL || resident == NULL || breaks == NULL ||
        workspace == NULL || removable == NULL ||
        boundary >= problem->boundary_count ||
        removable_capacity < problem->alias_count ||
        workspace->alias_count != problem->alias_count ||
        workspace->boundary_count != problem->boundary_count ||
        workspace->device_count != problem->device_count) {
        return -1;
    }
    memset(removable, 0, problem->alias_count);
    build_cut_geometry(
        problem,
        resident,
        breaks,
        workspace->first_required,
        workspace->gap_start,
        workspace->gap_end
    );
    for (uint32_t alias = 0U; alias < problem->alias_count; ++alias) {
        const uint64_t position = cell(
            alias, problem->boundary_count, boundary
        );
        if (resident[position] == 0U || problem->anchors[position] != 0U ||
            problem->output_reservations[position] != 0U) {
            continue;
        }
        ResidencyCut cut;
        if (candidate_cut(
                problem,
                resident,
                breaks,
                workspace->first_required,
                workspace->gap_start,
                workspace->gap_end,
                alias,
                (int32_t)boundary - 1,
                &cut
            ) != 0 && cut.start <= cut.end) {
            removable[alias] = 1U;
        }
    }
    return 0;
}

static int valid_problem(
    const ShadowSpillResidencyProblem *problem,
    const ShadowSpillResidencyOptions *options,
    const ShadowSpillResidencyResult *result
) {
    if (problem == NULL || options == NULL || result == NULL ||
        problem->abi_version != SHADOWSPILL_PLANNER_ABI_VERSION ||
        problem->alias_count == 0U || problem->boundary_count == 0U ||
        problem->device_count == 0U) {
        return 0;
    }
    uint64_t cells = (uint64_t)problem->alias_count * problem->boundary_count;
    return problem->alias_size_bytes != NULL && problem->alias_device != NULL &&
        problem->alias_retain_spill_copy != NULL && problem->initial_location != NULL &&
        problem->final_location != NULL && problem->anchors != NULL &&
        problem->productions != NULL && problem->latest_access_task != NULL &&
        problem->output_reservations != NULL && problem->write_prefix != NULL &&
        problem->first_input_task != NULL && problem->fetch_runtime_ns != NULL &&
        problem->evict_runtime_ns != NULL && problem->task_ideal_end_ns != NULL &&
        problem->device_capacity_bytes != NULL &&
        problem->boundary_capacity_bytes != NULL &&
        problem->device_priority != NULL && options->seed_resident != NULL &&
        options->seed_breaks != NULL && options->extra_pressure_bytes != NULL &&
        result->resident != NULL && result->breaks != NULL &&
        result->resident_capacity >= cells && result->break_capacity >= cells;
}

int shadowspill_residency_workspace_create(
    const ShadowSpillResidencyProblem *problem,
    ShadowSpillResidencyWorkspace **workspace_output
) {
    if (problem == NULL || workspace_output == NULL ||
        problem->alias_count == 0U || problem->boundary_count == 0U ||
        problem->device_count == 0U) {
        return -1;
    }
    *workspace_output = NULL;
    uint64_t cells =
        (uint64_t)problem->alias_count * problem->boundary_count;
    uint64_t pressure_cells =
        (uint64_t)problem->device_count * problem->boundary_count;
    if (cells > SIZE_MAX / sizeof(int32_t) ||
        pressure_cells > SIZE_MAX / sizeof(uint64_t)) {
        return -1;
    }
    ShadowSpillResidencyWorkspace *workspace =
        calloc(1U, sizeof(*workspace));
    if (workspace == NULL) {
        return -1;
    }
    workspace->alias_count = problem->alias_count;
    workspace->boundary_count = problem->boundary_count;
    workspace->device_count = problem->device_count;
    workspace->pressure = malloc(
        (size_t)pressure_cells * sizeof(*workspace->pressure)
    );
    workspace->before = malloc(
        (size_t)problem->boundary_count * sizeof(*workspace->before)
    );
    workspace->after = malloc(
        (size_t)problem->boundary_count * sizeof(*workspace->after)
    );
    workspace->first_required = malloc(
        (size_t)problem->alias_count * sizeof(*workspace->first_required)
    );
    workspace->gap_start = malloc((size_t)cells * sizeof(*workspace->gap_start));
    workspace->gap_end = malloc((size_t)cells * sizeof(*workspace->gap_end));
    workspace->seed_resident = malloc((size_t)cells);
    workspace->seed_breaks = malloc((size_t)cells);
    workspace->base_pressure[0] = malloc(
        (size_t)pressure_cells * sizeof(*workspace->base_pressure[0])
    );
    workspace->base_pressure[1] = malloc(
        (size_t)pressure_cells * sizeof(*workspace->base_pressure[1])
    );
    workspace->cut_cursors = malloc(
        (size_t)pressure_cells * sizeof(*workspace->cut_cursors)
    );
    if (workspace->pressure == NULL || workspace->before == NULL ||
        workspace->after == NULL ||
        workspace->first_required == NULL || workspace->gap_start == NULL ||
        workspace->gap_end == NULL || workspace->seed_resident == NULL ||
        workspace->seed_breaks == NULL || workspace->base_pressure[0] == NULL ||
        workspace->base_pressure[1] == NULL || workspace->cut_cursors == NULL) {
        shadowspill_residency_workspace_destroy(workspace);
        return -1;
    }
    *workspace_output = workspace;
    return 0;
}

void shadowspill_residency_workspace_destroy(
    ShadowSpillResidencyWorkspace *workspace
) {
    if (workspace == NULL) {
        return;
    }
    free(workspace->pressure);
    free(workspace->before);
    free(workspace->after);
    free(workspace->first_required);
    free(workspace->gap_start);
    free(workspace->gap_end);
    free(workspace->seed_resident);
    free(workspace->seed_breaks);
    free(workspace->base_pressure[0]);
    free(workspace->base_pressure[1]);
    free(workspace->cut_cursors);
    free(workspace->cut_active);
    destroy_cut_index(&workspace->cut_index);
    free(workspace);
}

ShadowSpillPlannerStatus shadowspill_reduce_residency_reusing(
    const ShadowSpillResidencyProblem *problem,
    const ShadowSpillResidencyOptions *options,
    ShadowSpillResidencyResult *result,
    ShadowSpillResidencyWorkspace *workspace
) {
    if (!valid_problem(problem, options, result) || workspace == NULL ||
        workspace->alias_count != problem->alias_count ||
        workspace->boundary_count != problem->boundary_count ||
        workspace->device_count != problem->device_count) {
        return SHADOWSPILL_PLANNER_INVALID_ARGUMENT;
    }
    uint8_t *resident_output = result->resident;
    uint64_t resident_capacity = result->resident_capacity;
    uint8_t *break_output = result->breaks;
    uint64_t break_capacity = result->break_capacity;
    memset(result, 0, sizeof(*result));
    result->resident = resident_output;
    result->resident_capacity = resident_capacity;
    result->breaks = break_output;
    result->break_capacity = break_capacity;
    result->error_device = UINT32_MAX;
    result->error_boundary = INT32_MIN;

    uint64_t cells = (uint64_t)problem->alias_count * problem->boundary_count;
    if (cells != 0U) {
        memcpy(result->resident, options->seed_resident, cells);
        memcpy(result->breaks, options->seed_breaks, cells);
    }

    uint64_t pressure_cells =
        (uint64_t)problem->device_count * problem->boundary_count;
    uint64_t *pressure = workspace->pressure;
    uint8_t *before = workspace->before;
    uint8_t *after = workspace->after;
    uint32_t *first_required = workspace->first_required;
    int32_t *gap_start = workspace->gap_start;
    int32_t *gap_end = workspace->gap_end;
    if (prepare_seed_geometry(problem, options, workspace) != 0 ||
        prepare_base_pressure(problem, options, workspace) != 0) {
        return SHADOWSPILL_PLANNER_INTERNAL_ERROR;
    }
    if (workspace->cut_active_capacity < workspace->cut_index.cut_count) {
        uint8_t *active = realloc(
            workspace->cut_active,
            workspace->cut_index.cut_count == 0U
                ? 1U
                : (size_t)workspace->cut_index.cut_count
        );
        if (active == NULL) {
            return SHADOWSPILL_PLANNER_ALLOCATION_FAILURE;
        }
        workspace->cut_active = active;
        workspace->cut_active_capacity = workspace->cut_index.cut_count;
    }
    if (workspace->cut_index.cut_count != 0U) {
        memset(
            workspace->cut_active,
            1,
            (size_t)workspace->cut_index.cut_count
        );
    }
    uint32_t pressure_variant = options->prefetch_headroom != 0U ? 1U : 0U;
    if (pressure_cells != 0U) {
        memcpy(
            pressure,
            workspace->base_pressure[pressure_variant],
            (size_t)pressure_cells * sizeof(*pressure)
        );
        memset(
            workspace->cut_cursors,
            0,
            (size_t)pressure_cells * sizeof(*workspace->cut_cursors)
        );
    }

    while (1) {
        uint32_t selected_device = UINT32_MAX;
        uint32_t selected_boundary = UINT32_MAX;
        uint64_t selected_excess = 0U;
        uint64_t selected_used = 0U;
        for (uint32_t boundary = 0U; boundary < problem->boundary_count;
             ++boundary) {
            for (uint32_t device = 0U; device < problem->device_count; ++device) {
                uint64_t position =
                    (uint64_t)device * problem->boundary_count + boundary;
                uint64_t used = pressure[position] +
                    options->extra_pressure_bytes[position];
                uint64_t capacity = shadowspill_boundary_capacity(
                    problem,
                    device,
                    boundary
                );
                if (used <= capacity) {
                    continue;
                }
                uint64_t excess = used - capacity;
                int better = selected_device == UINT32_MAX ||
                    excess > selected_excess ||
                    (excess == selected_excess && boundary < selected_boundary) ||
                    (excess == selected_excess && boundary == selected_boundary &&
                     problem->device_priority[device] <
                         problem->device_priority[selected_device]);
                if (better) {
                    selected_device = device;
                    selected_boundary = boundary;
                    selected_excess = excess;
                    selected_used = used;
                }
            }
        }
        if (selected_device == UINT32_MAX) {
            canonicalize_breaks(
                result->breaks,
                result->resident,
                problem->alias_count,
                problem->boundary_count
            );
            result->status = SHADOWSPILL_PLANNER_OK;
            return SHADOWSPILL_PLANNER_OK;
        }

        int32_t boundary_value = (int32_t)selected_boundary - 1;
        ResidencyCut chosen;
        if (!select_cut(
            problem,
            selected_device,
            boundary_value,
            options->minimize_transfer != 0U,
            &workspace->cut_index,
            workspace->cut_active,
            workspace->cut_cursors,
            &chosen
        )) {
            result->status = SHADOWSPILL_PLANNER_ANALYTIC_INFEASIBLE;
            result->error_device = selected_device;
            result->error_boundary = boundary_value;
            result->required_bytes = selected_used;
            result->capacity_bytes =
                shadowspill_boundary_capacity(
                    problem,
                    selected_device,
                    selected_boundary
                );
            return SHADOWSPILL_PLANNER_ANALYTIC_INFEASIBLE;
        }
        uint32_t alias = chosen.alias;
        alias_contribution(
            problem,
            options,
            result->resident,
            result->breaks,
            alias,
            before
        );
        apply_cut(problem, result->resident, result->breaks, &chosen);
        refresh_alias_candidates(
            problem,
            result->resident,
            result->breaks,
            first_required,
            gap_start,
            gap_end,
            &workspace->cut_index,
            workspace->cut_active,
            alias
        );
        alias_contribution(
            problem,
            options,
            result->resident,
            result->breaks,
            alias,
            after
        );
        uint32_t device = problem->alias_device[alias];
        for (uint32_t boundary = 0U; boundary < problem->boundary_count;
             ++boundary) {
            int64_t delta = (int64_t)after[boundary] - before[boundary];
            if (delta < 0) {
                pressure[(uint64_t)device * problem->boundary_count + boundary] -=
                    problem->alias_size_bytes[alias];
            } else if (delta > 0) {
                pressure[(uint64_t)device * problem->boundary_count + boundary] +=
                    problem->alias_size_bytes[alias];
            }
        }
    }
}

ShadowSpillPlannerStatus shadowspill_reduce_residency(
    const ShadowSpillResidencyProblem *problem,
    const ShadowSpillResidencyOptions *options,
    ShadowSpillResidencyResult *result
) {
    if (!valid_problem(problem, options, result)) {
        return SHADOWSPILL_PLANNER_INVALID_ARGUMENT;
    }
    ShadowSpillResidencyWorkspace *workspace = NULL;
    if (shadowspill_residency_workspace_create(problem, &workspace) != 0) {
        return SHADOWSPILL_PLANNER_ALLOCATION_FAILURE;
    }
    ShadowSpillPlannerStatus status = shadowspill_reduce_residency_reusing(
        problem,
        options,
        result,
        workspace
    );
    shadowspill_residency_workspace_destroy(workspace);
    return status;
}
