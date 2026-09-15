/* The buffers one candidate reuses across its plans. */
#include "internal.h"

static int simulation_workspace_create(
    const ShadowSpillPressureFitProblem *problem,
    SimulationWorkspace *workspace
) {
    memset(workspace, 0, sizeof(*workspace));
    workspace->task_capacity = problem->context.simulation->task_count;
    workspace->device_capacity = problem->context.simulation->device_count;
    workspace->tasks = calloc(
        workspace->task_capacity == 0U ? 1U : workspace->task_capacity,
        sizeof(*workspace->tasks)
    );
    workspace->transfers = calloc(
        1U,
        sizeof(*workspace->transfers)
    );
    workspace->peaks = calloc(
        workspace->device_capacity == 0U ? 1U : workspace->device_capacity,
        sizeof(*workspace->peaks)
    );
    if (workspace->tasks == NULL || workspace->transfers == NULL ||
        workspace->peaks == NULL) {
        free(workspace->tasks);
        free(workspace->transfers);
        free(workspace->peaks);
        memset(workspace, 0, sizeof(*workspace));
        return -1;
    }
    return 0;
}

int shadowspill_candidate_simulation_workspace_reserve_transfers(
    SimulationWorkspace *workspace,
    uint32_t capacity
) {
    if (capacity <= workspace->transfer_capacity) {
        return 0;
    }
    uint32_t selected = workspace->transfer_capacity == 0U
        ? 64U
        : workspace->transfer_capacity;
    while (selected < capacity) {
        if (selected > UINT32_MAX / 2U) {
            selected = capacity;
            break;
        }
        selected *= 2U;
    }
    ShadowSpillTransferInterval *replacement = realloc(
        workspace->transfers,
        (size_t)selected * sizeof(*replacement)
    );
    if (replacement == NULL) {
        return -1;
    }
    workspace->transfers = replacement;
    workspace->transfer_capacity = selected;
    return 0;
}

static void simulation_workspace_destroy(SimulationWorkspace *workspace) {
    free(workspace->tasks);
    free(workspace->transfers);
    free(workspace->peaks);
    memset(workspace, 0, sizeof(*workspace));
}

int shadowspill_candidate_placement_reserve(
    PlacementWorkspace *workspace,
    uint64_t operations,
    uint64_t leases,
    uint32_t aliases,
    uint32_t allocation_slots
) {
    /* A program can legitimately have none of a kind -- no allocation steps,
     * no aliases. Reserving one anyway keeps every buffer a valid pointer,
     * which is what the builders below validate. */
    operations = operations ? operations : 1U;
    leases = leases ? leases : 1U;
    aliases = aliases ? aliases : 1U;
    allocation_slots = allocation_slots ? allocation_slots : 1U;
    if (operations > workspace->operation_capacity) {
        uint64_t want = operations;
        free(workspace->lease_ids);
        free(workspace->dependency_ids);
        free(workspace->bytes);
        free(workspace->alignments);
        free(workspace->kinds);
        free(workspace->purposes);
        free(workspace->boundaries);
        free(workspace->indices);
        free(workspace->allocation_offsets);
        workspace->lease_ids = malloc(want * sizeof(*workspace->lease_ids));
        workspace->dependency_ids =
            malloc(want * sizeof(*workspace->dependency_ids));
        workspace->bytes = malloc(want * sizeof(*workspace->bytes));
        workspace->alignments = malloc(want * sizeof(*workspace->alignments));
        workspace->kinds = malloc(want * sizeof(*workspace->kinds));
        workspace->purposes = malloc(want * sizeof(*workspace->purposes));
        workspace->boundaries = malloc(want * sizeof(*workspace->boundaries));
        workspace->indices = malloc(want * sizeof(*workspace->indices));
        workspace->allocation_offsets =
            malloc(want * sizeof(*workspace->allocation_offsets));
        workspace->operation_capacity = want;
        if (workspace->lease_ids == NULL || workspace->dependency_ids == NULL ||
            workspace->bytes == NULL || workspace->alignments == NULL ||
            workspace->kinds == NULL || workspace->purposes == NULL ||
            workspace->boundaries == NULL || workspace->indices == NULL ||
            workspace->allocation_offsets == NULL) {
            return -1;
        }
    }
    if (leases > workspace->lease_capacity) {
        free(workspace->lifetimes);
        free(workspace->identities);
        free(workspace->offsets);
        free(workspace->excluded);
        free(workspace->lease_aliases);
        free(workspace->lease_starts);
        free(workspace->lease_retires);
        workspace->excluded = malloc(leases * sizeof(*workspace->excluded));
        workspace->lease_aliases =
            malloc(leases * sizeof(*workspace->lease_aliases));
        workspace->lease_starts =
            malloc(leases * sizeof(*workspace->lease_starts));
        workspace->lease_retires =
            malloc(leases * sizeof(*workspace->lease_retires));
        workspace->lifetimes = malloc(leases * sizeof(*workspace->lifetimes));
        workspace->identities = malloc(leases * sizeof(*workspace->identities));
        workspace->offsets = malloc(leases * sizeof(*workspace->offsets));
        workspace->lease_capacity = leases;
        if (workspace->lifetimes == NULL || workspace->identities == NULL ||
            workspace->offsets == NULL || workspace->excluded == NULL ||
            workspace->lease_aliases == NULL ||
            workspace->lease_starts == NULL ||
            workspace->lease_retires == NULL) {
            return -1;
        }
    }
    if (allocation_slots > workspace->allocation_slot_capacity) {
        /* One entry per flattened allocation step, which is neither a lease
         * nor an alias count. */
        free(workspace->allocation_step_leases);
        workspace->allocation_step_leases = malloc(
            (size_t)allocation_slots * sizeof(*workspace->allocation_step_leases)
        );
        workspace->allocation_slot_capacity = allocation_slots;
        if (workspace->allocation_step_leases == NULL) {
            return -1;
        }
    }
    if (aliases > workspace->alias_capacity) {
        free(workspace->alias_leases);
        free(workspace->dynamic_aliases);
        workspace->alias_leases =
            malloc((size_t)aliases * sizeof(*workspace->alias_leases));
        workspace->dynamic_aliases =
            malloc((size_t)aliases * sizeof(*workspace->dynamic_aliases));
        workspace->alias_capacity = aliases;
        if (workspace->alias_leases == NULL ||
            workspace->dynamic_aliases == NULL) {
            return -1;
        }
    }
    return 0;
}

static void placement_workspace_destroy(PlacementWorkspace *workspace) {
    free(workspace->lease_ids);
    free(workspace->dependency_ids);
    free(workspace->bytes);
    free(workspace->alignments);
    free(workspace->kinds);
    free(workspace->purposes);
    free(workspace->boundaries);
    free(workspace->indices);
    free(workspace->allocation_offsets);
    free(workspace->lease_aliases);
    free(workspace->lease_starts);
    free(workspace->lease_retires);
    free(workspace->excluded);
    free(workspace->lifetimes);
    free(workspace->identities);
    free(workspace->allocation_step_leases);
    free(workspace->alias_leases);
    free(workspace->offsets);
    free(workspace->dynamic_aliases);
    memset(workspace, 0, sizeof(*workspace));
}

/*
 * Can this plan be placed in the execution pool, and if not, by how much did
 * it overrun?
 *
 * The simulator answers whether a plan fits by bytes; this answers whether it
 * fits by *placement*, which is a stricter question -- leases need contiguous
 * ranges and a pool with room in total can still have nowhere to put one.
 * A plan the pool cannot place cannot run, so the overage is what the plan
 * has to give back before it is worth anything.
 *
 * Returns 0 on success, writing `required_bytes`; -1 if the measurement could
 * not be taken at all, which is a different thing from a plan that does not
 * fit.
 */
int shadowspill_candidate_candidate_workspace_create(
    const ShadowSpillPressureFitProblem *problem,
    CandidateWorkspace *workspace
) {
    memset(workspace, 0, sizeof(*workspace));
    uint64_t cell_count = (uint64_t)problem->residency->alias_count *
        problem->residency->boundary_count;
    uint64_t pressure_count = (uint64_t)problem->residency->device_count *
        problem->residency->boundary_count;
    if (cell_count > SIZE_MAX || pressure_count > SIZE_MAX) {
        return -1;
    }
    size_t cells = (size_t)cell_count;
    workspace->cell_count = cells;
    workspace->packed_cell_count = shadowspill_packed_cells(cells);
    const size_t packed = workspace->packed_cell_count == 0U
        ? 1U
        : workspace->packed_cell_count;
    workspace->resident = calloc(packed, 1U);
    workspace->breaks = calloc(packed, 1U);
    workspace->base_resident = calloc(packed, 1U);
    workspace->base_breaks = calloc(packed, 1U);
    workspace->repair_resident = calloc(packed, 1U);
    workspace->repair_breaks = calloc(packed, 1U);
    workspace->packed_seed_resident = calloc(packed, 1U);
    workspace->packed_seed_breaks = calloc(packed, 1U);
    if (workspace->packed_seed_resident != NULL &&
        workspace->packed_seed_breaks != NULL) {
        for (uint64_t index = 0U; index < cells; ++index) {
            shadowspill_cell_set(
                workspace->packed_seed_resident,
                index,
                problem->seed_resident[index] != 0U
            );
            shadowspill_cell_set(
                workspace->packed_seed_breaks,
                index,
                problem->seed_breaks[index] != 0U
            );
        }
        shadowspill_canonicalize_breaks(
            workspace->packed_seed_breaks,
            workspace->packed_seed_resident,
            problem->residency->alias_count,
            problem->residency->boundary_count
        );
    }
    workspace->cut_scratch_capacity = problem->residency->alias_count;
    workspace->cut_scratch = calloc(
        workspace->cut_scratch_capacity == 0U
            ? 1U
            : (size_t)workspace->cut_scratch_capacity,
        sizeof(*workspace->cut_scratch)
    );
    workspace->removable_aliases = calloc(
        problem->residency->alias_count == 0U
            ? 1U
            : problem->residency->alias_count,
        1U
    );
    workspace->extra_pressure = calloc(
        pressure_count == 0U ? 1U : (size_t)pressure_count,
        sizeof(*workspace->extra_pressure)
    );
    if (workspace->resident == NULL || workspace->breaks == NULL ||
        workspace->base_resident == NULL || workspace->base_breaks == NULL ||
        workspace->repair_resident == NULL ||
        workspace->repair_breaks == NULL ||
        workspace->packed_seed_resident == NULL ||
        workspace->packed_seed_breaks == NULL ||
        workspace->cut_scratch == NULL ||
        workspace->removable_aliases == NULL ||
        workspace->extra_pressure == NULL ||
        shadowspill_schedule_storage_create(
            problem->residency->alias_count,
            &workspace->schedule
        ) != 0 ||
        shadowspill_schedule_storage_create(
            problem->residency->alias_count,
            &workspace->selected
        ) != 0 ||
        shadowspill_schedule_storage_create(
            problem->residency->alias_count,
            &workspace->best
        ) != 0 ||
        simulation_workspace_create(
            problem,
            &workspace->simulation
        ) != 0 ||
        (problem->context.admission != NULL &&
         shadowspill_candidate_admission_workspace_create(
             &problem->context, &workspace->admission
         ) != 0) ||
        shadowspill_residency_workspace_create(
            problem->residency,
            &workspace->residency_workspace
        ) != 0) {
        return -1;
    }
    return 0;
}


void shadowspill_candidate_candidate_workspace_destroy(CandidateWorkspace *workspace) {
    if (workspace == NULL) {
        return;
    }
    free(workspace->resident);
    free(workspace->breaks);
    free(workspace->base_resident);
    free(workspace->base_breaks);
    free(workspace->repair_resident);
    free(workspace->repair_breaks);
    free(workspace->removable_aliases);
    free(workspace->extra_pressure);
    free(workspace->cut_scratch);
    free(workspace->fetch_constraints);
    shadowspill_schedule_storage_destroy(&workspace->schedule);
    shadowspill_schedule_storage_destroy(&workspace->selected);
    shadowspill_schedule_storage_destroy(&workspace->best);
    simulation_workspace_destroy(&workspace->simulation);
    placement_workspace_destroy(&workspace->placement);
    shadowspill_candidate_admission_workspace_destroy(&workspace->admission);
    shadowspill_residency_workspace_destroy(workspace->residency_workspace);
    free(workspace->packed_seed_resident);
    free(workspace->packed_seed_breaks);
    for (uint32_t index = 0U; index < workspace->schedule_memo.count; ++index) {
        shadowspill_candidate_free_indexed_schedule(&workspace->schedule_memo.entries[index].schedule);
    }
    free(workspace->simulation_memo.entries);
    free(workspace->simulation_memo.index.slots);
    memset(workspace, 0, sizeof(*workspace));
}

