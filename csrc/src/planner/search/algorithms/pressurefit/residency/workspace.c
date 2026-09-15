/* The arrays a reduction reuses across its candidates. */
#include "internal.h"

int shadowspill_residency_workspace_create(
    const ShadowSpillPressureFitResidencyProblem *problem,
    ShadowSpillPressureFitResidencyWorkspace **workspace_output
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
    ShadowSpillPressureFitResidencyWorkspace *workspace =
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
    ShadowSpillPressureFitResidencyWorkspace *workspace
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
    shadowspill_residency_destroy_cut_index(&workspace->cut_index);
    free(workspace);
}

/* The caller owns every buffer the result points at, so those survive the
 * reset; everything the reduction is about to decide does not. */
void shadowspill_residency_reset_residency_result(ShadowSpillPressureFitResidencyResult *result) {
    const ShadowSpillPressureFitResidencyResult borrowed = {
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
void shadowspill_residency_seed_residency(
    const ShadowSpillPressureFitResidencyProblem *problem,
    const ShadowSpillPressureFitResidencyOptions *options,
    ShadowSpillPressureFitResidencyResult *result
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
int shadowspill_residency_reset_cut_candidates(ShadowSpillPressureFitResidencyWorkspace *workspace) {
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
