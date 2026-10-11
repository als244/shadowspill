#ifndef SHADOWSPILL_PLANNER_RESIDENCY_INTERNAL_H
#define SHADOWSPILL_PLANNER_RESIDENCY_INTERNAL_H

#include <shadowspill/planner.h>
#include <shadowspill/pressurefit/pressurefit.h>

#include <stddef.h>
#include <stdint.h>

/* What the program does to an alias at one of its anchors. */
enum ShadowSpillPressureFitAnchorFlags {
    /* The task at the boundary reads the alias. */
    SHADOWSPILL_ANCHOR_INPUT = 1U << 0U,
    /* The task before the boundary produces the alias fresh. */
    SHADOWSPILL_ANCHOR_PRODUCED = 1U << 1U,
};

/* PressureFit's residency reducer works on this form of the problem: the
 * per-boundary capacities and, per alias, the boundaries the program
 * touches it at. Everything about an alias is a sorted list over its own
 * boundaries, never an array over every boundary: a step of tens of
 * thousands of tasks and aliases touches each alias at a handful of them,
 * and anything sized alias by boundary would be billions of cells. It is
 * derived from a ShadowSpillIndexedProblem inside the search and never
 * crosses the library boundary. */
/* PressureFit's own input, never seen by a caller: the residency problem
 * its reducer works on, the seeds it starts from, and the resolved problem
 * one candidate is placed against. A caller supplies a
 * ShadowSpillIndexedProblem; these are derived from it inside the search.
 */
typedef struct ShadowSpillPressureFitResidencyProblem {
    uint32_t abi_version;
    uint32_t alias_count;
    uint32_t boundary_count;
    uint32_t device_count;

    const uint64_t *alias_size_bytes;
    const uint32_t *alias_device;
    const uint8_t *alias_retain_spill_copy;
    const int8_t *initial_location;
    const int8_t *final_location;
    const uint32_t *first_input_task;
    const uint64_t *fetch_runtime_ns;
    const uint64_t *evict_runtime_ns;
    const uint64_t *task_ideal_end_ns;
    const uint64_t *device_capacity_bytes;
    /* Maximum task-object pressure at each [device][boundary] cell. */
    const uint64_t *boundary_capacity_bytes;
    const uint32_t *device_priority;
    /* The anchors of each alias -- every boundary the program accesses it
       at or declares it resident at -- as a sorted list: anchor_offsets[alias]
       .. anchor_offsets[alias + 1] index anchor_positions (the boundary),
       anchor_tasks (the latest task accessing the alias at that boundary),
       anchor_earliest (the earliest such task) and anchor_flags. A boundary
       an alias only starts or ends the step resident at, or that a mutation
       reaches after its task, has UINT32_MAX for both tasks. A zero-size
       alias has no anchors. */
    const uint32_t *anchor_offsets;
    const uint32_t *anchor_positions;
    const uint32_t *anchor_tasks;
    const uint32_t *anchor_earliest;
    const uint8_t *anchor_flags;
    /* The boundaries each alias reserves for a produced output -- the
       producing task's own boundary, when that task neither reads nor
       mutates the alias -- as a sorted list: reserved_offsets[alias] ..
       reserved_offsets[alias + 1] index reserved_positions. */
    const uint32_t *reserved_offsets;
    const uint32_t *reserved_positions;
    /* The boundaries each alias is written at -- after a task that mutates
       or produces it -- as a sorted list: write_offsets[alias] ..
       write_offsets[alias + 1] index write_boundaries. */
    const uint32_t *write_offsets;
    const uint32_t *write_boundaries;
    /* Per alias, whether the reducer may cut its residency; NULL means every
       alias may be cut. An alias that may not be cut stays resident from its
       first to its last access and is charged in the required floor; the
       emitter still produces its opening fetch, its release, and its
       terminal writeback. */
    const uint8_t *alias_evict_eligible;
    /* Per alias the reducer may not cut and that starts the step in spill:
       the task after which it is fetched, chosen once at preparation so the
       resident slice is sized for it. UINT32_MAX elsewhere. */
    const uint32_t *fixed_fetch_trigger;
} ShadowSpillPressureFitResidencyProblem;

/* One alias's residency: its spans, ascending and disjoint, as inclusive
 * [start, end] boundary pairs at bounds[2i] and bounds[2i + 1]. Two spans
 * that touch are two spans; that is what a break between them is. */
typedef struct ShadowSpillSpanList {
    uint32_t *bounds;
    uint32_t count;
    uint32_t capacity;
} ShadowSpillSpanList;

/* A residency: which boundaries each alias is held resident at, as one span
 * list per alias. Everything the search holds about residency -- the seed,
 * a strategy's base, the plan being reduced -- is one of these. Its size is
 * the number of spans, which is about the number of aliases plus the cuts
 * made, never the number of alias-boundary cells. There is no separate
 * record of breaks and nothing to canonicalize: the list of spans is the
 * one spelling of a residency. */
typedef struct ShadowSpillResidency {
    uint32_t alias_count;
    uint32_t boundary_count;
    ShadowSpillSpanList *aliases;
} ShadowSpillResidency;

int shadowspill_residency_create(
    ShadowSpillResidency *residency, uint32_t alias_count, uint32_t boundary_count
);
void shadowspill_residency_destroy(ShadowSpillResidency *residency);
/* `destination` takes `source`'s spans; both must have the same geometry. */
int shadowspill_residency_copy(
    ShadowSpillResidency *destination, const ShadowSpillResidency *source
);
/* Append [start, end] after `alias`'s last span, which it must follow. */
int shadowspill_residency_append_span(
    ShadowSpillResidency *residency, uint32_t alias, uint32_t start, uint32_t end
);
/* Take the cells [first, last] out of `alias`'s residency: spans are
 * trimmed, split or dropped as the range covers them. */
int shadowspill_residency_clear_cells(
    ShadowSpillResidency *residency, uint32_t alias, uint32_t first, uint32_t last
);
/* End the span holding `boundary` there, so the cell after it starts a new
 * span; nothing happens when the cell is not resident or already last. */
int shadowspill_residency_split_after(
    ShadowSpillResidency *residency, uint32_t alias, uint32_t boundary
);
/* Two independent 64-bit hashes of the residency's spans. */
uint64_t shadowspill_residency_hash(
    const ShadowSpillResidency *residency, uint64_t hash, uint64_t prime
);

/* The index of `alias`'s span holding `boundary`, or UINT32_MAX. */
static inline uint32_t shadowspill_residency_span_index(
    const ShadowSpillResidency *residency, uint32_t alias, uint32_t boundary
) {
    const ShadowSpillSpanList *list = &residency->aliases[alias];
    uint32_t low = 0U;
    uint32_t high = list->count;
    while (low < high) {
        const uint32_t middle = low + (high - low) / 2U;
        if (list->bounds[2U * middle + 1U] < boundary) {
            low = middle + 1U;
        } else {
            high = middle;
        }
    }
    return low < list->count && list->bounds[2U * low] <= boundary
        ? low
        : UINT32_MAX;
}

static inline int shadowspill_residency_resident_at(
    const ShadowSpillResidency *residency, uint32_t alias, uint32_t boundary
) {
    return shadowspill_residency_span_index(residency, alias, boundary) != UINT32_MAX;
}

typedef struct ShadowSpillPressureFitResidencyOptions {
    uint8_t minimize_transfer;
    uint8_t fetch_headroom;
    /* The residency every reduction starts from. */
    const ShadowSpillResidency *seed;
    const uint64_t *extra_pressure_bytes;
} ShadowSpillPressureFitResidencyOptions;

typedef struct ShadowSpillPressureFitResidencyResult {
    uint32_t status;
    uint32_t error_device;
    int32_t error_boundary;
    uint64_t required_bytes;
    uint64_t capacity_bytes;
    /* Caller-owned, with the problem's geometry; the reduction writes the
       residency it reached into it. */
    ShadowSpillResidency *residency;
    /* Optional: the aliases this reduction cut, in the order it cut them.
       NULL asks for no record, which is what planning normally wants. */
    uint32_t *cut_aliases;
    uint64_t cut_capacity;
    uint64_t cut_count;
} ShadowSpillPressureFitResidencyResult;

static inline uint64_t shadowspill_boundary_capacity(
    const ShadowSpillPressureFitResidencyProblem *problem,
    uint32_t device,
    uint32_t boundary
) {
    return problem->boundary_capacity_bytes[
        (uint64_t)device * problem->boundary_count + boundary
    ];
}

/* Residency bitmaps are packed: one bit per (alias, boundary) cell, cell
 * index = alias * boundary_count + boundary, least significant bit first
 * within a byte. Every bitmap the planner passes between its stages --
 * workspace residencies, breaks, seeds -- uses this layout. */
static inline size_t shadowspill_packed_cells(uint64_t cells) {
    return (size_t)(cells / 8U + (cells % 8U != 0U));
}

/* Whether the reducer may cut an alias; NULL eligibility means every alias. */
static inline int shadowspill_alias_may_cut(
    const ShadowSpillPressureFitResidencyProblem *problem, uint32_t alias
) {
    return problem->alias_evict_eligible == NULL ||
        problem->alias_evict_eligible[alias] != 0U;
}

/* The latest task in [earliest, latest] whose ideal end is at or before
 * target_ns, or earliest when none is: the trigger that starts a fetch as
 * late as the ideal timeline allows. */
static inline uint32_t shadowspill_latest_safe_trigger(
    const uint64_t *task_ideal_end_ns,
    uint32_t earliest,
    uint32_t latest,
    uint64_t target_ns
) {
    uint32_t insertion = earliest;
    while (insertion <= latest && task_ideal_end_ns[insertion] <= target_ns) {
        ++insertion;
    }
    return insertion == earliest ? earliest : insertion - 1U;
}

/* The static home a resident lease takes: the first offset at or past the
 * cursor that satisfies the alignment, which then moves past the lease.
 * Nonzero when the arithmetic overflows. */
static inline int shadowspill_resident_home(
    uint64_t *cursor, uint64_t bytes, uint64_t alignment, uint64_t *offset
) {
    const uint64_t step = alignment == 0U ? 1U : alignment;
    if (*cursor > UINT64_MAX - (step - 1U)) {
        return -1;
    }
    const uint64_t start = (*cursor + step - 1U) / step * step;
    if (bytes > UINT64_MAX - start) {
        return -1;
    }
    *offset = start;
    *cursor = start + bytes;
    return 0;
}

/* The first index in [begin, end) whose anchor position is >= boundary. */
static inline uint32_t shadowspill_anchor_lower_bound(
    const uint32_t *positions, uint32_t begin, uint32_t end, uint32_t boundary
) {
    while (begin < end) {
        const uint32_t middle = begin + (end - begin) / 2U;
        if (positions[middle] < boundary) {
            begin = middle + 1U;
        } else {
            end = middle;
        }
    }
    return begin;
}

/* The index of the anchor `alias` has at `boundary`, or UINT32_MAX. */
static inline uint32_t shadowspill_anchor_at(
    const ShadowSpillPressureFitResidencyProblem *problem,
    uint32_t alias,
    uint32_t boundary
) {
    const uint32_t last = problem->anchor_offsets[alias + 1U];
    const uint32_t index = shadowspill_anchor_lower_bound(
        problem->anchor_positions, problem->anchor_offsets[alias], last, boundary
    );
    return index < last && problem->anchor_positions[index] == boundary
        ? index
        : UINT32_MAX;
}

/* Whether a produced output of `alias` enters at `boundary`. */
static inline int shadowspill_produced_at(
    const ShadowSpillPressureFitResidencyProblem *problem,
    uint32_t alias,
    uint32_t boundary
) {
    const uint32_t anchor = shadowspill_anchor_at(problem, alias, boundary);
    return anchor != UINT32_MAX &&
        (problem->anchor_flags[anchor] & SHADOWSPILL_ANCHOR_PRODUCED) != 0U;
}

/* Whether `alias` reserves `boundary` for an output. */
static inline int shadowspill_reserved_at(
    const ShadowSpillPressureFitResidencyProblem *problem,
    uint32_t alias,
    uint32_t boundary
) {
    const uint32_t last = problem->reserved_offsets[alias + 1U];
    const uint32_t index = shadowspill_anchor_lower_bound(
        problem->reserved_positions, problem->reserved_offsets[alias], last, boundary
    );
    return index < last && problem->reserved_positions[index] == boundary;
}

/* Whether `alias` has been written at or before `boundary`. */
static inline int shadowspill_written_by(
    const ShadowSpillPressureFitResidencyProblem *problem,
    uint32_t alias,
    uint32_t boundary
) {
    const uint32_t first = problem->write_offsets[alias];
    return first < problem->write_offsets[alias + 1U] &&
        problem->write_boundaries[first] <= boundary;
}

/* Whether some anchor of `alias` within [start, end] has a task later
 * than `after`, the same question as scanning the latest task at each of
 * the span's cells. */
static inline int shadowspill_span_accessed_after(
    const ShadowSpillPressureFitResidencyProblem *problem,
    uint32_t alias,
    uint32_t start,
    uint32_t end,
    int32_t after
) {
    const uint32_t last = problem->anchor_offsets[alias + 1U];
    uint32_t index = shadowspill_anchor_lower_bound(
        problem->anchor_positions, problem->anchor_offsets[alias], last, start
    );
    for (; index < last && problem->anchor_positions[index] <= end; ++index) {
        const uint32_t task = problem->anchor_tasks[index];
        if (task != UINT32_MAX && (int32_t)task > after) {
            return 1;
        }
    }
    return 0;
}

typedef struct ShadowSpillPressureFitResidencyWorkspace ShadowSpillPressureFitResidencyWorkspace;

/* The per-alias lists of the residency problem, owned by whoever prepared
 * it: the anchors, the output reservations and the writes, each a sorted
 * list per alias behind an offset array. */
typedef struct ShadowSpillPressureFitResidencySparseLists {
    uint32_t *anchor_offsets;
    uint32_t *anchor_positions;
    uint32_t *anchor_tasks;
    uint32_t *anchor_earliest;
    uint8_t *anchor_flags;
    uint32_t *reserved_offsets;
    uint32_t *reserved_positions;
    uint32_t *write_offsets;
    uint32_t *write_boundaries;
} ShadowSpillPressureFitResidencySparseLists;

void shadowspill_residency_sparse_lists_destroy(ShadowSpillPressureFitResidencySparseLists *lists);

/* The next span of `alias` at or after `*cursor`: from its first resident
 * cell at or after the cursor to the span's end. Leaves `*cursor` past the
 * span, or at the end when there is none, which is also when it returns 0. */
int shadowspill_residency_next_span(
    const ShadowSpillResidency *residency,
    uint32_t alias,
    uint32_t *cursor,
    uint32_t *start,
    uint32_t *end
);

/* The boundaries one alias's residency charges: the charged interval of
 * every span, merged with its output reservations, written to `intervals`
 * as ascending, disjoint (start, end) pairs. The buffer must hold two
 * entries per span plus per reservation, which `2 * boundary_count` pairs
 * always does. Returns how many pairs were written. */
uint32_t shadowspill_residency_charged_intervals(
    const ShadowSpillPressureFitResidencyProblem *problem,
    int fetch_headroom,
    const ShadowSpillResidency *residency,
    uint32_t alias,
    uint32_t *intervals
);

/* Add the residency charge of every alias the reducer may cut to
 * `pressure`, one entry per [device][boundary], on top of what it holds. */
int shadowspill_residency_add_charges(
    const ShadowSpillPressureFitResidencyProblem *problem,
    int fetch_headroom,
    const ShadowSpillResidency *residency,
    uint64_t *pressure
);

int shadowspill_residency_workspace_create(
    const ShadowSpillPressureFitResidencyProblem *problem,
    ShadowSpillPressureFitResidencyWorkspace **workspace
);

void shadowspill_residency_workspace_destroy(
    ShadowSpillPressureFitResidencyWorkspace *workspace
);

int shadowspill_residency_pressure_at(
    const ShadowSpillPressureFitResidencyProblem *problem,
    const ShadowSpillPressureFitResidencyOptions *options,
    const ShadowSpillResidency *residency,
    uint32_t device,
    uint32_t boundary,
    uint64_t *pressure_bytes
);

ShadowSpillStatus shadowspill_pressurefit_reduce_residency_reusing(
    const ShadowSpillPressureFitResidencyProblem *problem,
    const ShadowSpillPressureFitResidencyOptions *options,
    ShadowSpillPressureFitResidencyResult *result,
    ShadowSpillPressureFitResidencyWorkspace *workspace
);

#endif
