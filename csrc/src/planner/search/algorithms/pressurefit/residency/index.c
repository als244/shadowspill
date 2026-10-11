/* Every candidate cut, ordered so the best is cheap to find. */
#include "internal.h"

void shadowspill_residency_destroy_cut_index(CutIndex *index) {
    free(index->cuts);
    free(index->alias_offsets);
    free(index->node_offsets);
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

/* The stretch of boundaries being gathered into one indexed cut: the
 * boundaries an alias yields the same cut at, taken in order. A stretch that
 * continues the open one with the same cut extends it; anything else closes
 * the open one and starts afresh. This is the merge a walk over every
 * boundary would do, fed stretches instead of single boundaries. */
typedef struct CutStretch {
    int active;
    ResidencyCut current;
    uint32_t first;
    uint32_t last;
} CutStretch;

static int offer_cut_stretch(
    const ShadowSpillPressureFitResidencyProblem *problem,
    CutIndex *index,
    CutStretch *stretch,
    uint32_t first,
    uint32_t last,
    int valid,
    const ResidencyCut *candidate
) {
    if (valid != 0 && stretch->active != 0 &&
        shadowspill_residency_same_cut(candidate, &stretch->current) &&
        first == stretch->last + 1U) {
        stretch->last = last;
        return 0;
    }
    if (stretch->active != 0 &&
        append_indexed_cut(
            problem, index, stretch->current, stretch->first, stretch->last
        ) != 0) {
        return -1;
    }
    stretch->active = valid;
    if (valid != 0) {
        stretch->current = *candidate;
        stretch->first = first;
        stretch->last = last;
    }
    return 0;
}

/* Offer the cut every boundary of [first, last] yields, evaluated at
 * `first`. The boundaries of one anchor-free run inside a span all yield
 * the run's cut, and an anchor yields its own, so one evaluation per
 * stretch is the whole walk. */
static int offer_evaluated_stretch(
    const ShadowSpillPressureFitResidencyProblem *problem,
    ShadowSpillPressureFitResidencyWorkspace *workspace,
    CutIndex *index,
    CutStretch *stretch,
    uint32_t alias,
    uint32_t first,
    uint32_t last
) {
    ResidencyCut candidate;
    const int valid = shadowspill_residency_candidate_cut(
        problem,
        workspace->seed,
        workspace->first_required,
        workspace->run_offsets,
        workspace->run_bounds,
        alias,
        (int32_t)first - 1,
        &candidate
    );
    return offer_cut_stretch(problem, index, stretch, first, last, valid, &candidate);
}

/* Every cut one alias's seed residency admits, in boundary order: its
 * spans, and within each span the anchors and the anchor-free runs
 * between them. */
static int index_alias_cuts(
    const ShadowSpillPressureFitResidencyProblem *problem,
    ShadowSpillPressureFitResidencyWorkspace *workspace,
    CutIndex *index,
    uint32_t alias
) {
    const ShadowSpillResidency *seed = workspace->seed;
    const ShadowSpillSpanList *spans = &seed->aliases[alias];
    const uint32_t begin = problem->anchor_offsets[alias];
    const uint32_t end = problem->anchor_offsets[alias + 1U];
    /* An alias that starts the step in spill yet is resident from the first
     * boundary, with no anchor there, yields one cut at every boundary
     * before its first required anchor, whatever else is at them. */
    const uint32_t required = workspace->first_required[alias];
    const uint32_t prefix =
        shadowspill_residency_resident_at(seed, alias, 0U) &&
            problem->initial_location[alias] == 1 &&
            shadowspill_anchor_at(problem, alias, 0U) == UINT32_MAX &&
            required != UINT32_MAX
        ? required
        : 0U;
    CutStretch stretch = {0};
    for (uint32_t span = 0U; span < spans->count; ++span) {
        const uint32_t span_start = spans->bounds[2U * span];
        const uint32_t span_end = spans->bounds[2U * span + 1U];
        uint32_t anchor = shadowspill_anchor_lower_bound(
            problem->anchor_positions, begin, end, span_start
        );
        uint32_t boundary = span_start;
        while (boundary <= span_end) {
            const uint32_t next_anchor =
                anchor < end && problem->anchor_positions[anchor] <= span_end
                ? problem->anchor_positions[anchor]
                : span_end + 1U;
            if (next_anchor > boundary) {
                const uint32_t run_end = next_anchor - 1U;
                if (boundary < prefix && prefix <= run_end) {
                    if (offer_evaluated_stretch(
                            problem, workspace, index, &stretch, alias, boundary, prefix - 1U
                        ) != 0) {
                        return -1;
                    }
                    boundary = prefix;
                }
                if (offer_evaluated_stretch(
                        problem, workspace, index, &stretch, alias, boundary, run_end
                    ) != 0) {
                    return -1;
                }
                boundary = run_end + 1U;
            }
            if (next_anchor <= span_end) {
                if (offer_evaluated_stretch(
                        problem, workspace, index, &stretch, alias, next_anchor, next_anchor
                    ) != 0) {
                    return -1;
                }
                boundary = next_anchor + 1U;
                ++anchor;
            }
        }
    }
    const ResidencyCut none = {0};
    return offer_cut_stretch(problem, index, &stretch, 0U, 0U, 0, &none);
}

/* The nodes of one device's tree that together cover [first, last] exactly,
 * passed to `visit` with its argument; the standard bottom-up walk from the
 * two leaves inward. */
static void cover_nodes(
    const CutIndex *index,
    uint32_t first,
    uint32_t last,
    void (*visit)(uint32_t node, void *argument),
    void *argument
) {
    uint32_t left = index->leaf_base + first;
    uint32_t right = index->leaf_base + last + 1U;
    while (left < right) {
        if ((left & 1U) != 0U) {
            visit(left, argument);
            ++left;
        }
        if ((right & 1U) != 0U) {
            --right;
            visit(right, argument);
        }
        left >>= 1U;
        right >>= 1U;
    }
}

typedef struct NodeCount {
    uint64_t *counts;
    uint64_t base;
} NodeCount;

static void count_node(uint32_t node, void *argument) {
    NodeCount *count = argument;
    ++count->counts[count->base + node + 1U];
}

typedef struct NodeFill {
    uint32_t *refs;
    uint64_t *cursor;
    uint64_t base;
    uint32_t cut_id;
} NodeFill;

static void fill_node(uint32_t node, void *argument) {
    NodeFill *fill = argument;
    fill->refs[fill->cursor[fill->base + node]++] = fill->cut_id;
}

static int build_cut_index(
    const ShadowSpillPressureFitResidencyProblem *problem,
    ShadowSpillPressureFitResidencyWorkspace *workspace
) {
    CutIndex *index = &workspace->cut_index;
    shadowspill_residency_destroy_cut_index(index);

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
        if (index_alias_cuts(problem, workspace, index, alias) != 0) {
            shadowspill_residency_destroy_cut_index(index);
            return -1;
        }
        index->alias_offsets[alias + 1U] = index->cut_count;
    }

    /* One tree per device over the boundaries: leaves from the first power
     * of two at or above the boundary count. */
    index->leaf_base = 1U;
    while (index->leaf_base < problem->boundary_count) {
        index->leaf_base *= 2U;
    }
    index->node_count = 2U * index->leaf_base;
    const uint64_t slots = (uint64_t)problem->device_count * index->node_count;
    if (slots > SIZE_MAX / sizeof(*index->node_offsets) - 1U) {
        shadowspill_residency_destroy_cut_index(index);
        return -1;
    }
    index->node_offsets = calloc((size_t)slots + 1U, sizeof(*index->node_offsets));
    if (index->node_offsets == NULL) {
        shadowspill_residency_destroy_cut_index(index);
        return -1;
    }
    for (uint32_t cut_id = 0U; cut_id < index->cut_count; ++cut_id) {
        const IndexedCut *item = &index->cuts[cut_id];
        NodeCount count = {
            .counts = index->node_offsets,
            .base = (uint64_t)problem->alias_device[item->cut.alias] * index->node_count,
        };
        cover_nodes(index, item->first_boundary, item->last_boundary, count_node, &count);
    }
    for (uint64_t slot = 0U; slot < slots; ++slot) {
        index->node_offsets[slot + 1U] += index->node_offsets[slot];
    }
    index->ref_count = index->node_offsets[slots];
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
        (slots == 0U ? 1U : (size_t)slots) * sizeof(*cursor)
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
    /* Writing the cuts into their nodes in rank order leaves every node's
     * list in that order, which is what selection reads. */
    for (uint32_t mode = 0U; mode < 2U; ++mode) {
        for (uint32_t cut_id = 0U; cut_id < index->cut_count; ++cut_id) {
            ranked[cut_id] = cut_id;
        }
        sort_cuts = index->cuts;
        sort_minimize_transfer = mode != 0U;
        qsort(ranked, index->cut_count, sizeof(*ranked), cut_ref_compare);
        sort_cuts = NULL;
        memcpy(cursor, index->node_offsets, (size_t)slots * sizeof(*cursor));
        for (uint32_t rank = 0U; rank < index->cut_count; ++rank) {
            const uint32_t cut_id = ranked[rank];
            const IndexedCut *item = &index->cuts[cut_id];
            NodeFill fill = {
                .refs = index->refs[mode],
                .cursor = cursor,
                .base = (uint64_t)problem->alias_device[item->cut.alias] * index->node_count,
                .cut_id = cut_id,
            };
            cover_nodes(index, item->first_boundary, item->last_boundary, fill_node, &fill);
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
    ShadowSpillPressureFitResidencyWorkspace *workspace
) {
    const ShadowSpillResidency *seed = workspace->seed;
    workspace->run_count = 0U;
    for (uint32_t alias = 0U; alias < problem->alias_count; ++alias) {
        workspace->run_offsets[alias] = (uint32_t)workspace->run_count;
        workspace->first_required[alias] = UINT32_MAX;
        if (!shadowspill_alias_may_cut(problem, alias)) {
            continue;
        }
        const uint32_t begin = problem->anchor_offsets[alias];
        const uint32_t end = problem->anchor_offsets[alias + 1U];
        const uint32_t required = shadowspill_anchor_lower_bound(
            problem->anchor_positions, begin, end, 1U
        );
        if (required < end) {
            workspace->first_required[alias] = problem->anchor_positions[required];
        }
        const ShadowSpillSpanList *spans = &seed->aliases[alias];
        for (uint32_t span = 0U; span < spans->count; ++span) {
            const uint32_t span_start = spans->bounds[2U * span];
            const uint32_t span_end = spans->bounds[2U * span + 1U];
            /* The runs are what lies between the span's anchors. */
            uint32_t anchor = shadowspill_anchor_lower_bound(
                problem->anchor_positions, begin, end, span_start
            );
            uint32_t cursor = span_start;
            while (cursor <= span_end) {
                const uint32_t next_anchor =
                    anchor < end && problem->anchor_positions[anchor] <= span_end
                    ? problem->anchor_positions[anchor]
                    : span_end + 1U;
                if (next_anchor > cursor &&
                    append_run(workspace, cursor, next_anchor - 1U) != 0) {
                    return -1;
                }
                cursor = next_anchor + 1U;
                if (next_anchor <= span_end) {
                    ++anchor;
                }
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
    workspace->seed = options->seed;
    if (build_cut_geometry(problem, workspace) != 0 ||
        build_cut_index(problem, workspace) != 0) {
        return -1;
    }
    /* One cursor per node of the index, started over by each reduction. */
    const uint64_t slots = (uint64_t)problem->device_count * workspace->cut_index.node_count;
    if (slots > workspace->node_cursor_count) {
        uint32_t *cursors = realloc(
            workspace->node_cursors, (size_t)slots * sizeof(*cursors)
        );
        if (cursors == NULL) {
            return -1;
        }
        workspace->node_cursors = cursors;
        workspace->node_cursor_count = (uint32_t)slots;
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
    if (shadowspill_residency_add_charges(
            problem, options->fetch_headroom != 0U, workspace->seed, pressure
        ) != 0) {
        return -1;
    }
    workspace->pressure_valid[variant] = 1U;
    return 0;
}
