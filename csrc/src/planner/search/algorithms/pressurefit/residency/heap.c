/* The boundaries in excess, worst first. */
#include "internal.h"

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

int shadowspill_residency_excess_heap_push(
    ShadowSpillPressureFitResidencyWorkspace *workspace,
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

static void excess_heap_pop(ShadowSpillPressureFitResidencyWorkspace *workspace) {
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

/*
 * Seed the max-excess heap once from the initial pressure map. The per-cut
 * delta loop pushes a corrected entry whenever a cell's pressure rises, so
 * the full boundary-by-device scan never repeats. Excess is a pure function
 * of the current pressure -- capacity and extra pressure are constant within
 * one reduction -- so stale entries are validated and corrected at pop time.
 */
int shadowspill_residency_seed_excess_heap(
    const ShadowSpillPressureFitResidencyProblem *problem,
    const ShadowSpillPressureFitResidencyOptions *options,
    ShadowSpillPressureFitResidencyWorkspace *workspace
) {
    workspace->excess_count = 0U;
    for (uint32_t device = 0U; device < problem->device_count; ++device) {
        for (uint32_t boundary = 0U; boundary < problem->boundary_count;
             ++boundary) {
            const uint64_t position =
                (uint64_t)device * problem->boundary_count + boundary;
            const uint64_t used =
                shadowspill_residency_tree_pressure_at(workspace, device, boundary) + options->extra_pressure_bytes[position];
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
            if (shadowspill_residency_excess_heap_push(workspace, entry) != 0) {
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
int shadowspill_residency_pop_worst_boundary(
    const ShadowSpillPressureFitResidencyProblem *problem,
    const ShadowSpillPressureFitResidencyOptions *options,
    ShadowSpillPressureFitResidencyWorkspace *workspace,
    uint32_t *device,
    uint32_t *boundary,
    uint64_t *used_bytes
) {
    while (workspace->excess_count != 0U) {
        ExcessEntry top = workspace->excess_entries[0];
        const uint64_t position =
            (uint64_t)top.device * problem->boundary_count + top.boundary;
        const uint64_t used =
            shadowspill_residency_tree_pressure_at(workspace, top.device, top.boundary) + options->extra_pressure_bytes[position];
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
            if (shadowspill_residency_excess_heap_push(workspace, top) != 0) {
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
void shadowspill_residency_clear_touched(ShadowSpillPressureFitResidencyWorkspace *workspace) {
    for (uint32_t index = 0U; index < workspace->touched_count; ++index) {
        workspace->touched_aliases[workspace->touched_list[index]] = 0U;
    }
    workspace->touched_count = 0U;
}
