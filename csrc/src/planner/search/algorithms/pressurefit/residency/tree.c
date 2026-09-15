/* Pressure as a Fenwick tree: add over a range, read at a point. */
#include "internal.h"

/* Pressure per boundary lives in one Fenwick tree per device, indexed
 * 1..boundary_count, over the differences between neighbouring boundaries:
 * a cut's decrement over a range is two updates, and a boundary's pressure
 * is a prefix sum. Arithmetic is modulo 2^64 on the way and exact at the
 * end, because every prefix is a true, non-negative sum. */
static uint64_t *pressure_tree(
    ShadowSpillPressureFitResidencyWorkspace *workspace, uint32_t device
) {
    return workspace->pressure + (size_t)device * (workspace->boundary_count + 1U);
}

static void pressure_build(
    ShadowSpillPressureFitResidencyWorkspace *workspace,
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

uint64_t shadowspill_residency_tree_pressure_at(
    ShadowSpillPressureFitResidencyWorkspace *workspace, uint32_t device, uint32_t boundary
) {
    const uint64_t *tree = pressure_tree(workspace, device);
    uint64_t sum = 0U;
    for (uint32_t index = boundary + 1U; index != 0U; index -= index & (0U - index)) {
        sum += tree[index];
    }
    return sum;
}

static void pressure_add_from(
    ShadowSpillPressureFitResidencyWorkspace *workspace,
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
void shadowspill_residency_pressure_add(
    ShadowSpillPressureFitResidencyWorkspace *workspace,
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
void shadowspill_residency_reset_working_pressure(
    const ShadowSpillPressureFitResidencyProblem *problem,
    const ShadowSpillPressureFitResidencyOptions *options,
    ShadowSpillPressureFitResidencyWorkspace *workspace
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
