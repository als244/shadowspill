#ifndef SHADOWSPILL_PLANNER_RESIDENCY_INTERNAL_H
#define SHADOWSPILL_PLANNER_RESIDENCY_INTERNAL_H

#include <shadowspill/planner.h>

#include <stddef.h>
#include <stdint.h>

/* Residency bitmaps are packed: one bit per (alias, boundary) cell, cell
 * index = alias * boundary_count + boundary, least significant bit first
 * within a byte. Every bitmap the planner passes between its stages --
 * workspace residencies, breaks, seeds -- uses this layout. */
static inline size_t shadowspill_packed_cells(uint64_t cells) {
    return (size_t)(cells / 8U + (cells % 8U != 0U));
}

/* Whether the reducer may cut an alias; NULL eligibility means every alias. */
static inline int shadowspill_alias_may_cut(
    const ShadowSpillResidencyProblem *problem, uint32_t alias
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

/* Whether some anchor of `alias` within [start, end] has a task later
 * than `after`, the same question as scanning latest_access_task over the
 * span's cells. */
static inline int shadowspill_span_accessed_after(
    const ShadowSpillResidencyProblem *problem,
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

static inline int shadowspill_cell_get(const uint8_t *bits, uint64_t cell) {
    return (bits[cell >> 3U] >> (cell & 7U)) & 1U;
}

static inline void shadowspill_cell_set(uint8_t *bits, uint64_t cell, int value) {
    if (value != 0) {
        bits[cell >> 3U] |= (uint8_t)(1U << (cell & 7U));
    } else {
        bits[cell >> 3U] &= (uint8_t)~(1U << (cell & 7U));
    }
}

/* A window of up to 64 cells starting at any bit offset, least significant
 * bit first, from a packed array of `packed_bytes` bytes. Bits past the
 * requested `width` read as zero. */
static inline uint64_t shadowspill_cells_load(
    const uint8_t *bits, size_t packed_bytes, uint64_t offset, unsigned width
) {
    const size_t first = (size_t)(offset >> 3U);
    const unsigned shift = (unsigned)(offset & 7U);
    uint64_t low = 0U;
    uint64_t high = 0U;
    for (unsigned index = 0U; index < 8U && first + index < packed_bytes; ++index) {
        low |= (uint64_t)bits[first + index] << (8U * index);
    }
    if (shift != 0U && first + 8U < packed_bytes) {
        high = bits[first + 8U];
    }
    uint64_t window = (low >> shift) | (high << (64U - shift));
    if (shift == 0U) {
        window = low;
    }
    return width >= 64U ? window : window & ((UINT64_C(1) << width) - 1U);
}

/* Write the low `width` bits of `value` at any bit offset. */
static inline void shadowspill_cells_store(
    uint8_t *bits, uint64_t offset, unsigned width, uint64_t value
) {
    unsigned index = 0U;
    while (index < width && ((offset + index) & 7U) != 0U) {
        shadowspill_cell_set(bits, offset + index, (int)((value >> index) & 1U));
        ++index;
    }
    for (; index + 8U <= width; index += 8U) {
        bits[(offset + index) >> 3U] = (uint8_t)(value >> index);
    }
    for (; index < width; ++index) {
        shadowspill_cell_set(bits, offset + index, (int)((value >> index) & 1U));
    }
}


typedef struct ShadowSpillResidencyWorkspace ShadowSpillResidencyWorkspace;

/* The sparse companions of the dense anchor, access, and reservation arrays. */
typedef struct ShadowSpillResidencySparseLists {
    uint32_t *anchor_offsets;
    uint32_t *anchor_positions;
    uint32_t *anchor_tasks;
    uint32_t *reserved_offsets;
    uint32_t *reserved_positions;
} ShadowSpillResidencySparseLists;

int shadowspill_residency_sparse_lists_build(
    const uint8_t *anchors,
    const uint32_t *latest_access_task,
    const uint8_t *output_reservations,
    uint32_t alias_count,
    uint32_t boundary_count,
    ShadowSpillResidencySparseLists *lists
);

void shadowspill_residency_sparse_lists_destroy(ShadowSpillResidencySparseLists *lists);

int shadowspill_residency_workspace_create(
    const ShadowSpillResidencyProblem *problem,
    ShadowSpillResidencyWorkspace **workspace
);

void shadowspill_residency_workspace_destroy(
    ShadowSpillResidencyWorkspace *workspace
);

int shadowspill_residency_pressure_at(
    const ShadowSpillResidencyProblem *problem,
    const ShadowSpillResidencyOptions *options,
    const uint8_t *resident,
    const uint8_t *breaks,
    uint32_t device,
    uint32_t boundary,
    ShadowSpillResidencyWorkspace *workspace,
    uint64_t *pressure_bytes
);

/*
 * Remove one alias from a boundary using the same legal-cut rules as the
 * residency reducer. Returns 1 when changed, 2 when already absent, 0 when
 * the boundary is semantically required, and -1 for invalid input.
 */
/* Canonical breaks for every row: none on non-resident cells, and at a span's
 * last cell exactly when the alias is resident again later. Seeds handed to
 * reductions must be canonical; reductions keep them so. */
void shadowspill_canonicalize_breaks(
    uint8_t *breaks,
    const uint8_t *resident,
    uint32_t alias_count,
    uint32_t boundary_count
);

ShadowSpillStatus shadowspill_reduce_residency_reusing(
    const ShadowSpillResidencyProblem *problem,
    const ShadowSpillResidencyOptions *options,
    ShadowSpillResidencyResult *result,
    ShadowSpillResidencyWorkspace *workspace
);

#endif
