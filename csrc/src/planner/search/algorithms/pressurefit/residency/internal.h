#ifndef SHADOWSPILL_RESIDENCY_INTERNAL_H
#define SHADOWSPILL_RESIDENCY_INTERNAL_H

/*
 * Reducing what a step holds resident, one cut at a time.
 *
 * A cut takes an alias out of residency over a range of boundaries. The
 * files here are the steps of choosing one: spans price what residency
 * costs where, cuts propose and score the candidates, the index keeps them
 * ordered, the tree and the heap find the boundary in most excess, and
 * reduce drives the loop until the budget is met or nothing is left to cut.
 */

#include "../../../../internal.h"
#include "../residency_internal.h"

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

void shadowspill_residency_reset_residency_result(ShadowSpillPressureFitResidencyResult *result);

void shadowspill_residency_seed_residency(
    const ShadowSpillPressureFitResidencyProblem *problem,
    const ShadowSpillPressureFitResidencyOptions *options,
    ShadowSpillPressureFitResidencyResult *result
);

int shadowspill_residency_reset_cut_candidates(ShadowSpillPressureFitResidencyWorkspace *workspace);

uint64_t shadowspill_residency_cell(uint32_t alias, uint32_t boundary_count, uint32_t index);

int shadowspill_residency_next_span(
    const uint8_t *resident,
    const uint8_t *breaks,
    uint32_t alias,
    uint32_t boundary_count,
    uint32_t *cursor,
    uint32_t *start,
    uint32_t *end
);

void shadowspill_residency_span_charge(
    const ShadowSpillPressureFitResidencyProblem *problem,
    const ShadowSpillPressureFitResidencyOptions *options,
    uint32_t alias,
    uint32_t start,
    uint32_t end,
    int32_t *charged_start,
    int32_t *charged_end
);

void shadowspill_residency_span_around(
    const uint8_t *resident,
    const uint8_t *breaks,
    uint32_t alias,
    uint32_t boundary_count,
    uint32_t inside,
    uint32_t *start,
    uint32_t *end
);

void shadowspill_residency_alias_contribution(
    const ShadowSpillPressureFitResidencyProblem *problem,
    const ShadowSpillPressureFitResidencyOptions *options,
    const uint8_t *resident,
    const uint8_t *breaks,
    uint32_t alias,
    uint8_t *contribution
);

int shadowspill_residency_compare_score(
    const CutScore *left,
    const CutScore *right,
    int minimize_transfer
);

CutScore shadowspill_residency_score_cut(
    const ShadowSpillPressureFitResidencyProblem *problem,
    const ResidencyCut *cut,
    int minimize_transfer
);

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
);

int shadowspill_residency_same_cut(const ResidencyCut *left, const ResidencyCut *right);

int shadowspill_residency_select_cut(
    const ShadowSpillPressureFitResidencyProblem *problem,
    uint32_t device,
    int32_t boundary,
    int minimize_transfer,
    const CutIndex *index,
    const uint8_t *active,
    uint64_t *cursors,
    ResidencyCut *selected
);

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
);

void shadowspill_residency_apply_cut(
    const ShadowSpillPressureFitResidencyProblem *problem,
    uint8_t *resident,
    uint8_t *breaks,
    const ResidencyCut *cut
);

void shadowspill_residency_destroy_cut_index(CutIndex *index);

int shadowspill_residency_prepare_seed_geometry(
    const ShadowSpillPressureFitResidencyProblem *problem,
    const ShadowSpillPressureFitResidencyOptions *options,
    ShadowSpillPressureFitResidencyWorkspace *workspace
);

int shadowspill_residency_prepare_base_pressure(
    const ShadowSpillPressureFitResidencyProblem *problem,
    const ShadowSpillPressureFitResidencyOptions *options,
    ShadowSpillPressureFitResidencyWorkspace *workspace
);

void shadowspill_residency_canonicalize_row(
    uint8_t *breaks,
    const uint8_t *resident,
    uint32_t alias,
    uint32_t boundary_count
);

int shadowspill_residency_valid_problem(
    const ShadowSpillPressureFitResidencyProblem *problem,
    const ShadowSpillPressureFitResidencyOptions *options,
    const ShadowSpillPressureFitResidencyResult *result
);

uint64_t shadowspill_residency_tree_pressure_at(
    ShadowSpillPressureFitResidencyWorkspace *workspace, uint32_t device, uint32_t boundary
);

void shadowspill_residency_pressure_add(
    ShadowSpillPressureFitResidencyWorkspace *workspace,
    uint32_t device,
    uint32_t first,
    uint32_t last,
    uint64_t delta
);

void shadowspill_residency_reset_working_pressure(
    const ShadowSpillPressureFitResidencyProblem *problem,
    const ShadowSpillPressureFitResidencyOptions *options,
    ShadowSpillPressureFitResidencyWorkspace *workspace
);

/*
 * One lazily validated max-excess candidate. Entries are ordered by the
 * exact selection total order of the reducer: larger excess first, then
 * smaller boundary, then smaller device priority, then smaller device
 * index. Stale entries -- those whose recorded excess does not match the
 * current pressure -- are corrected or discarded at pop time, so the heap
 * yields the same selection sequence as a full scan.
 */
typedef struct {
    uint64_t excess;
    uint32_t boundary;
    uint32_t priority;
    uint32_t device;
} ExcessEntry;

struct ShadowSpillPressureFitResidencyWorkspace {
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
    const ShadowSpillPressureFitResidencyProblem *geometry_problem;
    /* Aliases a cut touched during the current reduction; only their rows
     * need canonical breaks at the end, the rest still equal the seed. */
    uint8_t *touched_aliases;
    uint32_t *touched_list;
    uint32_t touched_count;
    uint8_t pressure_valid[2];
};

int shadowspill_residency_excess_heap_push(
    ShadowSpillPressureFitResidencyWorkspace *workspace,
    ExcessEntry entry
);

int shadowspill_residency_seed_excess_heap(
    const ShadowSpillPressureFitResidencyProblem *problem,
    const ShadowSpillPressureFitResidencyOptions *options,
    ShadowSpillPressureFitResidencyWorkspace *workspace
);

int shadowspill_residency_pop_worst_boundary(
    const ShadowSpillPressureFitResidencyProblem *problem,
    const ShadowSpillPressureFitResidencyOptions *options,
    ShadowSpillPressureFitResidencyWorkspace *workspace,
    uint32_t *device,
    uint32_t *boundary,
    uint64_t *used_bytes
);

void shadowspill_residency_clear_touched(ShadowSpillPressureFitResidencyWorkspace *workspace);

#endif  /* SHADOWSPILL_RESIDENCY_INTERNAL_H */
