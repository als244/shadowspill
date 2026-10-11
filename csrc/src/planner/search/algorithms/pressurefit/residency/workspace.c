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
    uint64_t pressure_cells =
        (uint64_t)problem->device_count * problem->boundary_count;
    if (pressure_cells > SIZE_MAX / sizeof(uint64_t)) {
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
    workspace->first_required = malloc(
        aliases * sizeof(*workspace->first_required)
    );
    workspace->run_offsets = malloc(
        (aliases + 1U) * sizeof(*workspace->run_offsets)
    );
    workspace->base_pressure[0] = malloc(
        (size_t)pressure_cells * sizeof(*workspace->base_pressure[0])
    );
    workspace->base_pressure[1] = malloc(
        (size_t)pressure_cells * sizeof(*workspace->base_pressure[1])
    );
    if (workspace->pressure == NULL ||
        workspace->first_required == NULL || workspace->run_offsets == NULL ||
        workspace->base_pressure[0] == NULL ||
        workspace->base_pressure[1] == NULL) {
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
    free(workspace->first_required);
    free(workspace->run_offsets);
    free(workspace->run_bounds);
    free(workspace->base_pressure[0]);
    free(workspace->base_pressure[1]);
    free(workspace->node_cursors);
    free(workspace->cut_active);
    free(workspace->excess_entries);
    shadowspill_residency_destroy_cut_index(&workspace->cut_index);
    free(workspace);
}

/* The caller owns every buffer the result points at, so those survive the
 * reset; everything the reduction is about to decide does not. */
void shadowspill_residency_reset_residency_result(ShadowSpillPressureFitResidencyResult *result) {
    const ShadowSpillPressureFitResidencyResult borrowed = {
        .residency = result->residency,
        .cut_aliases = result->cut_aliases,
        .cut_capacity = result->cut_capacity,
    };
    *result = borrowed;
    result->error_device = UINT32_MAX;
    result->error_boundary = INT32_MIN;
}

/* Start from the residency the caller seeded rather than from nothing: a
 * repair reduces again from the same base its candidate began with. */
int shadowspill_residency_seed_residency(
    const ShadowSpillPressureFitResidencyOptions *options,
    ShadowSpillPressureFitResidencyResult *result
) {
    return shadowspill_residency_copy(result->residency, options->seed);
}

int shadowspill_residency_valid_problem(
    const ShadowSpillPressureFitResidencyProblem *problem,
    const ShadowSpillPressureFitResidencyOptions *options,
    const ShadowSpillPressureFitResidencyResult *result
) {
    if (problem == NULL || options == NULL || result == NULL ||
        problem->abi_version != SHADOWSPILL_ABI_VERSION ||
        problem->boundary_count == 0U || problem->device_count == 0U) {
        return 0;
    }
    return problem->alias_size_bytes != NULL && problem->alias_device != NULL &&
        problem->alias_retain_spill_copy != NULL && problem->initial_location != NULL &&
        problem->final_location != NULL && problem->anchor_offsets != NULL &&
        problem->anchor_positions != NULL && problem->anchor_tasks != NULL &&
        problem->anchor_earliest != NULL && problem->anchor_flags != NULL &&
        problem->reserved_offsets != NULL && problem->reserved_positions != NULL &&
        problem->write_offsets != NULL && problem->write_boundaries != NULL &&
        problem->first_input_task != NULL && problem->fetch_runtime_ns != NULL &&
        problem->evict_runtime_ns != NULL && problem->task_ideal_end_ns != NULL &&
        problem->device_capacity_bytes != NULL &&
        problem->boundary_capacity_bytes != NULL &&
        problem->device_priority != NULL && options->seed != NULL &&
        options->seed->alias_count == problem->alias_count &&
        options->seed->boundary_count == problem->boundary_count &&
        options->extra_pressure_bytes != NULL &&
        result->residency != NULL &&
        result->residency->alias_count == problem->alias_count &&
        result->residency->boundary_count == problem->boundary_count;
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
