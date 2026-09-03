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

/*
 * One lazily validated max-excess candidate. Entries are ordered by the
 * exact selection total order of the reducer: larger excess first, then
 * smaller boundary, then smaller device priority, then smaller device
 * index. Stale entries (whose recorded excess no longer matches the
 * current pressure) are corrected or discarded at pop time, so the heap
 * yields the same selection sequence as a full scan.
 */
typedef struct {
    uint64_t excess;
    uint32_t boundary;
    uint32_t priority;
    uint32_t device;
} ExcessEntry;

struct ShadowSpillResidencyWorkspace {
    uint32_t alias_count;
    uint32_t boundary_count;
    uint32_t device_count;
    uint64_t *pressure;
    uint8_t *before;
    uint8_t *after;
    uint32_t *first_required;
    /* Removable runs of the residency the geometry was built from, per
     * alias: run_offsets[alias] .. run_offsets[alias + 1] index run_bounds,
     * each run a (start, end) boundary pair, ascending within an alias. */
    uint32_t *run_offsets;
    int32_t *run_bounds;
    uint64_t run_count;
    uint64_t run_capacity;
    /* The seed the geometry was built from: the problem's own arrays. */
    const uint8_t *seed_resident;
    const uint8_t *seed_breaks;
    uint64_t *base_pressure[2];
    uint64_t *cut_cursors;
    uint8_t *cut_active;
    uint32_t cut_active_capacity;
    ExcessEntry *excess_entries;
    uint64_t excess_count;
    uint64_t excess_capacity;
    CutIndex cut_index;
    const ShadowSpillResidencyProblem *geometry_problem;
    /* Aliases a cut touched during the current reduction; only their rows
     * need canonical breaks at the end, the rest still equal the seed. */
    uint8_t *touched_aliases;
    uint32_t *touched_list;
    uint32_t touched_count;
    uint8_t pressure_valid[2];
};

static int excess_entry_before(const ExcessEntry *a, const ExcessEntry *b) {
    if (a->excess != b->excess) {
        return a->excess > b->excess;
    }
    if (a->boundary != b->boundary) {
        return a->boundary < b->boundary;
    }
    if (a->priority != b->priority) {
        return a->priority < b->priority;
    }
    return a->device < b->device;
}

static int excess_heap_push(
    ShadowSpillResidencyWorkspace *workspace,
    ExcessEntry entry
) {
    if (workspace->excess_count == workspace->excess_capacity) {
        uint64_t grown = workspace->excess_capacity == 0U
            ? 256U
            : workspace->excess_capacity * 2U;
        ExcessEntry *entries = realloc(
            workspace->excess_entries,
            (size_t)grown * sizeof(*entries)
        );
        if (entries == NULL) {
            return -1;
        }
        workspace->excess_entries = entries;
        workspace->excess_capacity = grown;
    }
    ExcessEntry *entries = workspace->excess_entries;
    uint64_t child = workspace->excess_count++;
    while (child != 0U) {
        uint64_t parent = (child - 1U) / 2U;
        if (!excess_entry_before(&entry, &entries[parent])) {
            break;
        }
        entries[child] = entries[parent];
        child = parent;
    }
    entries[child] = entry;
    return 0;
}

static void excess_heap_pop(ShadowSpillResidencyWorkspace *workspace) {
    ExcessEntry *entries = workspace->excess_entries;
    uint64_t count = --workspace->excess_count;
    if (count == 0U) {
        return;
    }
    ExcessEntry moved = entries[count];
    uint64_t parent = 0U;
    while (1) {
        uint64_t left = parent * 2U + 1U;
        if (left >= count) {
            break;
        }
        uint64_t right = left + 1U;
        uint64_t best = left;
        if (right < count &&
            excess_entry_before(&entries[right], &entries[left])) {
            best = right;
        }
        if (!excess_entry_before(&entries[best], &moved)) {
            break;
        }
        entries[parent] = entries[best];
        parent = best;
    }
    entries[parent] = moved;
}

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
static void span_charge(
    const ShadowSpillResidencyProblem *problem,
    const ShadowSpillResidencyOptions *options,
    uint32_t alias,
    uint32_t start,
    uint32_t end,
    int32_t *charged_start,
    int32_t *charged_end
) {
    const uint32_t count = problem->boundary_count;
    *charged_start = (int32_t)start;
    if (options->fetch_headroom != 0U && start > 0U &&
        problem->productions[cell(alias, count, start)] == 0U) {
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
static void span_around(
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
    const ShadowSpillResidencyProblem *problem,
    const ShadowSpillResidencyOptions *options,
    uint32_t alias,
    uint32_t start,
    uint32_t end,
    uint8_t *contribution
) {
    int32_t charged_start;
    int32_t charged_end;
    span_charge(problem, options, alias, start, end, &charged_start, &charged_end);
    for (int32_t boundary = charged_start; boundary <= charged_end; ++boundary) {
        contribution[boundary] = 1U;
    }
}

/* Mark every boundary the spans of one alias within [first, last] charge. */
static void spans_contribution(
    const ShadowSpillResidencyProblem *problem,
    const ShadowSpillResidencyOptions *options,
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
        if (!next_span(
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
    spans_contribution(
        problem, options, resident, breaks, alias, 0U, count - 1U, contribution
    );
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
        if (problem->alias_device[alias] != device ||
            !shadowspill_alias_may_cut(problem, alias)) {
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
    const uint32_t *run_offsets,
    const int32_t *run_bounds,
    uint32_t alias,
    int32_t boundary,
    ResidencyCut *cut
) {
    uint32_t boundary_index = (uint32_t)(boundary + 1);
    uint64_t position = cell(alias, problem->boundary_count, boundary_index);
    if (!shadowspill_cell_get(resident, position)) {
        return 0;
    }
    if (shadowspill_cell_get(resident, cell(alias, problem->boundary_count, 0U)) &&
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
                cell(alias, problem->boundary_count, boundary_index + 1U)
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
            int valid = candidate_cut(
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
        int valid = candidate_cut(
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
        active[cut_id] = valid != 0 && same_cut(&current, &indexed->cut);
    }
}

static int append_run(
    ShadowSpillResidencyWorkspace *workspace, uint32_t start, uint32_t end
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
    const ShadowSpillResidencyProblem *problem,
    const uint8_t *resident,
    const uint8_t *breaks,
    ShadowSpillResidencyWorkspace *workspace
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
            if (problem->anchors[cell(alias, count, index)] != 0U) {
                workspace->first_required[alias] = index;
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

static int prepare_seed_geometry(
    const ShadowSpillResidencyProblem *problem,
    const ShadowSpillResidencyOptions *options,
    ShadowSpillResidencyWorkspace *workspace
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

static int prepare_base_pressure(
    const ShadowSpillResidencyProblem *problem,
    const ShadowSpillResidencyOptions *options,
    ShadowSpillResidencyWorkspace *workspace
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
        shadowspill_cell_set(breaks, cell(cut->alias, problem->boundary_count, after), 1);
        return;
    }
    for (int32_t boundary = cut->start; boundary <= cut->end; ++boundary) {
        uint32_t index = (uint32_t)(boundary + 1);
        shadowspill_cell_set(resident, cell(cut->alias, problem->boundary_count, index), 0);
    }
}

/* One row's canonical breaks: none on a cell that is not resident, and at
 * a span's last cell exactly when the alias is resident again later. This
 * never changes the span structure, which is why it can run once on the
 * seed and afterwards only on rows a cut touched. */
static void canonicalize_row(
    uint8_t *breaks,
    const uint8_t *resident,
    uint32_t alias,
    uint32_t boundary_count
) {
    const uint64_t row = (uint64_t)alias * boundary_count;
    const size_t packed_bytes = shadowspill_packed_cells(
        (uint64_t)(alias + 1U) * boundary_count
    );
    /* From the top of the row down: a cell keeps its break only while
     * resident; a span's last cell (resident, next not resident) takes a
     * break exactly when the alias is resident again later. */
    int later = 0;
    int next_resident = 0;
    uint64_t offset = row + boundary_count;
    while (offset > row) {
        const unsigned width = (unsigned)((offset - row) < 64U ? offset - row : 64U);
        offset -= width;
        const uint64_t present = shadowspill_cells_load(resident, packed_bytes, offset, width);
        const uint64_t broken = shadowspill_cells_load(breaks, packed_bytes, offset, width);
        uint64_t following = present >> 1U;
        if (next_resident) {
            following |= UINT64_C(1) << (width - 1U);
        }
        const uint64_t ends = present & ~following;
        uint64_t above = present >> 1U;
        above |= above >> 1U;
        above |= above >> 2U;
        above |= above >> 4U;
        above |= above >> 8U;
        above |= above >> 16U;
        above |= above >> 32U;
        if (later) {
            above = ~UINT64_C(0);
        }
        uint64_t updated = (broken & present & ~ends) | (ends & above);
        if (width < 64U) {
            updated &= (UINT64_C(1) << width) - 1U;
        }
        if (updated != broken) {
            shadowspill_cells_store(breaks, offset, width, updated);
        }
        later = later || present != 0U;
        next_resident = (int)(present & 1U);
    }
}

void shadowspill_canonicalize_breaks(
    uint8_t *breaks,
    const uint8_t *resident,
    uint32_t alias_count,
    uint32_t boundary_count
) {
    for (uint32_t alias = 0U; alias < alias_count; ++alias) {
        canonicalize_row(breaks, resident, alias, boundary_count);
    }
}

static int valid_problem(
    const ShadowSpillResidencyProblem *problem,
    const ShadowSpillResidencyOptions *options,
    const ShadowSpillResidencyResult *result
) {
    if (problem == NULL || options == NULL || result == NULL ||
        problem->abi_version != SHADOWSPILL_ABI_VERSION ||
        problem->boundary_count == 0U || problem->device_count == 0U) {
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
        problem->boundary_count == 0U || problem->device_count == 0U) {
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
    size_t aliases = problem->alias_count == 0U ? 1U : problem->alias_count;
    workspace->alias_count = problem->alias_count;
    workspace->boundary_count = problem->boundary_count;
    workspace->device_count = problem->device_count;
    workspace->pressure = malloc(
        ((size_t)pressure_cells + problem->device_count) * sizeof(*workspace->pressure)
    );
    workspace->before = malloc(
        (size_t)problem->boundary_count * sizeof(*workspace->before)
    );
    workspace->after = malloc(
        (size_t)problem->boundary_count * sizeof(*workspace->after)
    );
    workspace->first_required = malloc(
        aliases * sizeof(*workspace->first_required)
    );
    workspace->run_offsets = malloc(
        (aliases + 1U) * sizeof(*workspace->run_offsets)
    );
    workspace->touched_aliases = calloc(aliases, 1U);
    workspace->touched_list = malloc(aliases * sizeof(*workspace->touched_list));
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
        workspace->first_required == NULL || workspace->run_offsets == NULL ||
        workspace->touched_aliases == NULL || workspace->touched_list == NULL ||
        workspace->base_pressure[0] == NULL ||
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
    free(workspace->run_offsets);
    free(workspace->run_bounds);
    free(workspace->touched_aliases);
    free(workspace->touched_list);
    free(workspace->base_pressure[0]);
    free(workspace->base_pressure[1]);
    free(workspace->cut_cursors);
    free(workspace->cut_active);
    free(workspace->excess_entries);
    destroy_cut_index(&workspace->cut_index);
    free(workspace);
}

/* The caller owns every buffer the result points at, so those survive the
 * reset; everything the reduction is about to decide does not. */
static void reset_residency_result(ShadowSpillResidencyResult *result) {
    const ShadowSpillResidencyResult borrowed = {
        .resident = result->resident,
        .resident_capacity = result->resident_capacity,
        .breaks = result->breaks,
        .break_capacity = result->break_capacity,
        .cut_aliases = result->cut_aliases,
        .cut_capacity = result->cut_capacity,
    };
    *result = borrowed;
    result->error_device = UINT32_MAX;
    result->error_boundary = INT32_MIN;
}

/* Start from the residency the caller seeded rather than from nothing: a
 * repair reduces again from the same base its candidate began with. */
static void seed_residency(
    const ShadowSpillResidencyProblem *problem,
    const ShadowSpillResidencyOptions *options,
    ShadowSpillResidencyResult *result
) {
    const uint64_t cells =
        (uint64_t)problem->alias_count * problem->boundary_count;
    if (cells == 0U) {
        return;
    }
    const size_t packed = shadowspill_packed_cells(cells);
    memcpy(result->resident, options->seed_resident, packed);
    memcpy(result->breaks, options->seed_breaks, packed);
}

/* Every cut starts available again, and the per-cell cursors that remember
 * how far each boundary has searched start over. */
static int reset_cut_candidates(ShadowSpillResidencyWorkspace *workspace) {
    const uint32_t cut_count = workspace->cut_index.cut_count;
    if (workspace->cut_active_capacity < cut_count) {
        uint8_t *active =
            realloc(workspace->cut_active, cut_count == 0U ? 1U : (size_t)cut_count);
        if (active == NULL) {
            return -1;
        }
        workspace->cut_active = active;
        workspace->cut_active_capacity = cut_count;
    }
    if (cut_count != 0U) {
        memset(workspace->cut_active, 1, (size_t)cut_count);
    }
    return 0;
}

/* Pressure per boundary lives in one Fenwick tree per device, indexed
 * 1..boundary_count, over the differences between neighbouring boundaries:
 * a cut's decrement over a range is two updates, and a boundary's pressure
 * is a prefix sum. Arithmetic is modulo 2^64 on the way and exact at the
 * end, because every prefix is a true, non-negative sum. */
static uint64_t *pressure_tree(
    ShadowSpillResidencyWorkspace *workspace, uint32_t device
) {
    return workspace->pressure + (size_t)device * (workspace->boundary_count + 1U);
}

static void pressure_build(
    ShadowSpillResidencyWorkspace *workspace,
    const uint64_t *base,
    uint32_t device_count
) {
    const uint32_t count = workspace->boundary_count;
    for (uint32_t device = 0U; device < device_count; ++device) {
        uint64_t *tree = pressure_tree(workspace, device);
        const uint64_t *values = base + (size_t)device * count;
        tree[0] = 0U;
        for (uint32_t index = 1U; index <= count; ++index) {
            tree[index] = values[index - 1U] - (index > 1U ? values[index - 2U] : 0U);
        }
        for (uint32_t index = 1U; index <= count; ++index) {
            const uint32_t parent = index + (index & (0U - index));
            if (parent <= count) {
                tree[parent] += tree[index];
            }
        }
    }
}

static uint64_t pressure_at(
    ShadowSpillResidencyWorkspace *workspace, uint32_t device, uint32_t boundary
) {
    const uint64_t *tree = pressure_tree(workspace, device);
    uint64_t sum = 0U;
    for (uint32_t index = boundary + 1U; index != 0U; index -= index & (0U - index)) {
        sum += tree[index];
    }
    return sum;
}

static void pressure_add_from(
    ShadowSpillResidencyWorkspace *workspace,
    uint32_t device,
    uint32_t boundary,
    uint64_t delta
) {
    uint64_t *tree = pressure_tree(workspace, device);
    const uint32_t count = workspace->boundary_count;
    for (uint32_t index = boundary + 1U; index <= count; index += index & (0U - index)) {
        tree[index] += delta;
    }
}

/* Add `delta` (modular) to every boundary in [first, last]. */
static void pressure_add(
    ShadowSpillResidencyWorkspace *workspace,
    uint32_t device,
    uint32_t first,
    uint32_t last,
    uint64_t delta
) {
    pressure_add_from(workspace, device, first, delta);
    if (last + 1U < workspace->boundary_count) {
        pressure_add_from(workspace, device, last + 1U, 0U - delta);
    }
}

/* The pressure this reduction works on is a copy of the base map, so the
 * base survives for the next candidate built on the same strategy. */
static void reset_working_pressure(
    const ShadowSpillResidencyProblem *problem,
    const ShadowSpillResidencyOptions *options,
    ShadowSpillResidencyWorkspace *workspace
) {
    const uint64_t pressure_cells =
        (uint64_t)problem->device_count * problem->boundary_count;
    if (pressure_cells == 0U) {
        return;
    }
    const uint32_t variant = options->fetch_headroom != 0U ? 1U : 0U;
    pressure_build(workspace, workspace->base_pressure[variant], problem->device_count);
    memset(
        workspace->cut_cursors,
        0,
        (size_t)pressure_cells * sizeof(*workspace->cut_cursors)
    );
}

/*
 * Seed the max-excess heap once from the initial pressure map. The per-cut
 * delta loop pushes a corrected entry whenever a cell's pressure rises, so
 * the full boundary-by-device scan never repeats. Excess is a pure function
 * of the current pressure -- capacity and extra pressure are constant within
 * one reduction -- so stale entries are validated and corrected at pop time.
 */
static int seed_excess_heap(
    const ShadowSpillResidencyProblem *problem,
    const ShadowSpillResidencyOptions *options,
    ShadowSpillResidencyWorkspace *workspace
) {
    workspace->excess_count = 0U;
    for (uint32_t device = 0U; device < problem->device_count; ++device) {
        for (uint32_t boundary = 0U; boundary < problem->boundary_count;
             ++boundary) {
            const uint64_t position =
                (uint64_t)device * problem->boundary_count + boundary;
            const uint64_t used =
                pressure_at(workspace, device, boundary) + options->extra_pressure_bytes[position];
            const uint64_t capacity =
                shadowspill_boundary_capacity(problem, device, boundary);
            if (used <= capacity) {
                continue;
            }
            const ExcessEntry entry = {
                used - capacity,
                boundary,
                problem->device_priority[device],
                device,
            };
            if (excess_heap_push(workspace, entry) != 0) {
                return -1;
            }
        }
    }
    return 0;
}

/*
 * The boundary with the largest excess that is still genuinely over capacity.
 *
 * Entries go stale as pressure moves under them, so the top of the heap is
 * validated before it is trusted: one whose cell now fits is dropped, and one
 * whose excess has changed is re-pushed with the current value. Returns 1 with
 * the boundary, 0 when nothing is over capacity, and -1 on failure.
 */
static int pop_worst_boundary(
    const ShadowSpillResidencyProblem *problem,
    const ShadowSpillResidencyOptions *options,
    ShadowSpillResidencyWorkspace *workspace,
    uint32_t *device,
    uint32_t *boundary,
    uint64_t *used_bytes
) {
    while (workspace->excess_count != 0U) {
        ExcessEntry top = workspace->excess_entries[0];
        const uint64_t position =
            (uint64_t)top.device * problem->boundary_count + top.boundary;
        const uint64_t used =
            pressure_at(workspace, top.device, top.boundary) + options->extra_pressure_bytes[position];
        const uint64_t capacity =
            shadowspill_boundary_capacity(problem, top.device, top.boundary);
        if (used <= capacity) {
            excess_heap_pop(workspace);
            continue;
        }
        const uint64_t excess = used - capacity;
        if (excess != top.excess) {
            excess_heap_pop(workspace);
            top.excess = excess;
            if (excess_heap_push(workspace, top) != 0) {
                return -1;
            }
            continue;
        }
        *device = top.device;
        *boundary = top.boundary;
        *used_bytes = used;
        return 1;
    }
    return 0;
}

/*
 * Give up one object, and carry the change through the pressure map.
 *
 * Cutting an alias changes what it contributes at every boundary, not just
 * the one that was over capacity: it stops occupying the boundaries it is no
 * longer resident at, and starts occupying any it newly spans. A cell that
 * rose may now be over capacity itself, so it joins the heap.
 */
static void clear_touched(ShadowSpillResidencyWorkspace *workspace) {
    for (uint32_t index = 0U; index < workspace->touched_count; ++index) {
        workspace->touched_aliases[workspace->touched_list[index]] = 0U;
    }
    workspace->touched_count = 0U;
}

static int apply_cut_and_repressure(
    const ShadowSpillResidencyProblem *problem,
    const ShadowSpillResidencyOptions *options,
    ShadowSpillResidencyResult *result,
    ShadowSpillResidencyWorkspace *workspace,
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
        span_around(resident, breaks, alias, count, low, &span_start, &span_end);
        span_charge(problem, options, alias, span_start, span_end, &before_start, &before_end);
    }
    /* What this reduction gave up, when anyone asked to be told. */
    if (result->cut_aliases != NULL && result->cut_count < result->cut_capacity) {
        result->cut_aliases[result->cut_count++] = alias;
    }
    apply_cut(problem, resident, breaks, chosen);
    refresh_alias_candidates(
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
        if (!next_span(resident, breaks, alias, count, &cursor, &start, &end) ||
            start > span_end) {
            break;
        }
        span_charge(
            problem, options, alias, start, end, &after_start[pieces], &after_end[pieces]
        );
        ++pieces;
    }
    /* The neighbours' charge at the two boundaries they can share. */
    int mask_left = 0;
    if (span_start > 0U && shadowspill_cell_get(resident, row + span_start - 1U)) {
        uint32_t neighbour_start = 0U;
        uint32_t neighbour_end = 0U;
        span_around(resident, breaks, alias, count, span_start - 1U, &neighbour_start, &neighbour_end);
        int32_t charge_start = 0;
        int32_t charge_end = -1;
        span_charge(problem, options, alias, neighbour_start, neighbour_end, &charge_start, &charge_end);
        mask_left = charge_end == (int32_t)span_start - 1;
    }
    int mask_right = 0;
    if (span_end + 1U < count && shadowspill_cell_get(resident, row + span_end + 1U)) {
        mask_right = options->fetch_headroom != 0U &&
            problem->productions[cell(alias, count, span_end + 1U)] == 0U;
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
                    pressure_add(workspace, device, (uint32_t)from, (uint32_t)masked - 1U, 0U - size);
                }
                from = masked + 1;
                continue;
            }
            if (stop >= from) {
                pressure_add(workspace, device, (uint32_t)from, (uint32_t)stop, 0U - size);
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
                problem->output_reservations[cell(alias, count, (uint32_t)boundary)] != 0U ||
                (boundary == (int32_t)span_start - 1 && mask_left) ||
                (boundary == (int32_t)span_end && mask_right);
            if (mask) {
                continue;
            }
            pressure_add(workspace, device, (uint32_t)boundary, (uint32_t)boundary, size);
            const uint64_t position =
                (uint64_t)device * problem->boundary_count + (uint32_t)boundary;
            const uint64_t used =
                pressure_at(workspace, device, (uint32_t)boundary) +
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
            if (excess_heap_push(workspace, entry) != 0) {
                return -1;
            }
        }
    }
    return 0;
}

/* No legal cut relieves this boundary, so no residency this strategy can
 * reach fits it. That is a fact about the problem rather than a failure. */
static ShadowSpillStatus report_analytic_infeasible(
    const ShadowSpillResidencyProblem *problem,
    ShadowSpillResidencyResult *result,
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
ShadowSpillStatus shadowspill_reduce_residency_reusing(
    const ShadowSpillResidencyProblem *problem,
    const ShadowSpillResidencyOptions *options,
    ShadowSpillResidencyResult *result,
    ShadowSpillResidencyWorkspace *workspace
) {
    if (!valid_problem(problem, options, result) || workspace == NULL ||
        workspace->alias_count != problem->alias_count ||
        workspace->boundary_count != problem->boundary_count ||
        workspace->device_count != problem->device_count) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    reset_residency_result(result);
    seed_residency(problem, options, result);
    if (prepare_seed_geometry(problem, options, workspace) != 0 ||
        prepare_base_pressure(problem, options, workspace) != 0) {
        return SHADOWSPILL_STATUS_PLANNER_INTERNAL_ERROR;
    }
    workspace->touched_count = 0U;
    if (reset_cut_candidates(workspace) != 0) {
        return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    }
    reset_working_pressure(problem, options, workspace);
    if (seed_excess_heap(problem, options, workspace) != 0) {
        return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    }

    while (1) {
        uint32_t device = UINT32_MAX;
        uint32_t boundary = UINT32_MAX;
        uint64_t used_bytes = 0U;
        const int over_capacity = pop_worst_boundary(
            problem, options, workspace, &device, &boundary, &used_bytes
        );
        if (over_capacity < 0) {
            clear_touched(workspace);
            return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
        }
        if (over_capacity == 0) {
            for (uint32_t index = 0U; index < workspace->touched_count; ++index) {
                const uint32_t touched = workspace->touched_list[index];
                canonicalize_row(
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
        if (!select_cut(
                problem,
                device,
                (int32_t)boundary - 1,
                options->minimize_transfer != 0U,
                &workspace->cut_index,
                workspace->cut_active,
                workspace->cut_cursors,
                &chosen
            )) {
            clear_touched(workspace);
            return report_analytic_infeasible(
                problem, result, device, boundary, used_bytes
            );
        }
        if (apply_cut_and_repressure(problem, options, result, workspace, &chosen) != 0) {
            clear_touched(workspace);
            return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
        }
    }
}

/* Per alias: its anchors (boundary order, with the latest task at each) and
 * its reserved boundaries, derived from the dense cell arrays. */
int shadowspill_residency_sparse_lists_build(
    const uint8_t *anchors,
    const uint32_t *latest_access_task,
    const uint8_t *output_reservations,
    uint32_t alias_count,
    uint32_t boundary_count,
    ShadowSpillResidencySparseLists *lists
) {
    const uint64_t cells = (uint64_t)alias_count * boundary_count;
    uint64_t anchor_total = 0U;
    uint64_t reserved_total = 0U;
    for (uint64_t cell = 0U; cell < cells; ++cell) {
        anchor_total += anchors[cell] != 0U;
        reserved_total += output_reservations[cell] != 0U;
    }
    lists->anchor_offsets = malloc(((size_t)alias_count + 1U) * sizeof(uint32_t));
    lists->anchor_positions = malloc((anchor_total == 0U ? 1U : (size_t)anchor_total) * sizeof(uint32_t));
    lists->anchor_tasks = malloc((anchor_total == 0U ? 1U : (size_t)anchor_total) * sizeof(uint32_t));
    lists->reserved_offsets = malloc(((size_t)alias_count + 1U) * sizeof(uint32_t));
    lists->reserved_positions = malloc((reserved_total == 0U ? 1U : (size_t)reserved_total) * sizeof(uint32_t));
    if (lists->anchor_offsets == NULL || lists->anchor_positions == NULL ||
        lists->anchor_tasks == NULL || lists->reserved_offsets == NULL ||
        lists->reserved_positions == NULL) {
        shadowspill_residency_sparse_lists_destroy(lists);
        return -1;
    }
    uint32_t anchor_written = 0U;
    uint32_t reserved_written = 0U;
    for (uint32_t alias = 0U; alias < alias_count; ++alias) {
        lists->anchor_offsets[alias] = anchor_written;
        lists->reserved_offsets[alias] = reserved_written;
        const uint64_t row = (uint64_t)alias * boundary_count;
        for (uint32_t boundary = 0U; boundary < boundary_count; ++boundary) {
            if (anchors[row + boundary] != 0U) {
                lists->anchor_positions[anchor_written] = boundary;
                lists->anchor_tasks[anchor_written] = latest_access_task[row + boundary];
                ++anchor_written;
            }
            if (output_reservations[row + boundary] != 0U) {
                lists->reserved_positions[reserved_written++] = boundary;
            }
        }
    }
    lists->anchor_offsets[alias_count] = anchor_written;
    lists->reserved_offsets[alias_count] = reserved_written;
    return 0;
}

void shadowspill_residency_sparse_lists_destroy(ShadowSpillResidencySparseLists *lists) {
    free(lists->anchor_offsets);
    free(lists->anchor_positions);
    free(lists->anchor_tasks);
    free(lists->reserved_offsets);
    free(lists->reserved_positions);
    memset(lists, 0, sizeof(*lists));
}

/* The public entry keeps byte-per-cell arrays at the boundary: it packs the
 * caller's seeds, reduces into packed scratch, and unpacks the answer. */
ShadowSpillStatus shadowspill_reduce_residency(
    const ShadowSpillResidencyProblem *problem,
    const ShadowSpillResidencyOptions *options,
    ShadowSpillResidencyResult *result
) {
    if (!valid_problem(problem, options, result)) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    ShadowSpillResidencySparseLists lists = {0};
    if (shadowspill_residency_sparse_lists_build(
            problem->anchors,
            problem->latest_access_task,
            problem->output_reservations,
            problem->alias_count,
            problem->boundary_count,
            &lists
        ) != 0) {
        return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    }
    ShadowSpillResidencyProblem derived = *problem;
    derived.anchor_offsets = lists.anchor_offsets;
    derived.anchor_positions = lists.anchor_positions;
    derived.anchor_tasks = lists.anchor_tasks;
    derived.reserved_offsets = lists.reserved_offsets;
    derived.reserved_positions = lists.reserved_positions;
    problem = &derived;
    const uint64_t cells = (uint64_t)problem->alias_count * problem->boundary_count;
    const size_t packed = shadowspill_packed_cells(cells);
    uint8_t *buffers = calloc(packed == 0U ? 4U : packed * 4U, 1U);
    ShadowSpillResidencyWorkspace *workspace = NULL;
    if (buffers == NULL ||
        shadowspill_residency_workspace_create(problem, &workspace) != 0) {
        free(buffers);
        shadowspill_residency_sparse_lists_destroy(&lists);
        return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    }
    uint8_t *seed_resident = buffers;
    uint8_t *seed_breaks = buffers + packed;
    for (uint64_t index = 0U; index < cells; ++index) {
        shadowspill_cell_set(seed_resident, index, options->seed_resident[index] != 0U);
        shadowspill_cell_set(seed_breaks, index, options->seed_breaks[index] != 0U);
    }
    shadowspill_canonicalize_breaks(
        seed_breaks, seed_resident, problem->alias_count, problem->boundary_count
    );
    ShadowSpillResidencyOptions packed_options = *options;
    packed_options.seed_resident = seed_resident;
    packed_options.seed_breaks = seed_breaks;
    ShadowSpillResidencyResult packed_result = *result;
    packed_result.resident = buffers + 2U * packed;
    packed_result.breaks = buffers + 3U * packed;
    ShadowSpillStatus status = shadowspill_reduce_residency_reusing(
        problem,
        &packed_options,
        &packed_result,
        workspace
    );
    for (uint64_t index = 0U; index < cells; ++index) {
        result->resident[index] =
            (uint8_t)shadowspill_cell_get(packed_result.resident, index);
        result->breaks[index] =
            (uint8_t)shadowspill_cell_get(packed_result.breaks, index);
    }
    packed_result.resident = result->resident;
    packed_result.breaks = result->breaks;
    *result = packed_result;
    shadowspill_residency_workspace_destroy(workspace);
    free(buffers);
    shadowspill_residency_sparse_lists_destroy(&lists);
    return status;
}
