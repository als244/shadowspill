/* Validating an admission topology and sizing the buffers it requires.
 *
 * Every other file here assumes a topology that has already passed
 * `shadowspill_admission_facts_valid`, and a workspace big enough for the
 * operations the schedule will produce. Both are established once, here.
 */

#include "internal.h"

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define TASK_ALLOCATION_ALLOCATE 0U
#define TASK_ALLOCATION_RELEASE 1U

static int checked_add(uint64_t left, uint64_t right, uint64_t *result) {
    if (right > UINT64_MAX - left) {
        return -1;
    }
    *result = left + right;
    return 0;
}

int shadowspill_admission_facts_valid(
    const ShadowSpillPressureFitProblem *problem
) {
    if (problem == NULL || problem->admission == NULL ||
        problem->simulation == NULL) {
        return 0;
    }
    const ShadowSpillAdmissionFacts *topology = problem->admission;
    const ShadowSpillSimulationProgram *program = problem->simulation;
    if (topology->abi_version != SHADOWSPILL_ABI_VERSION ||
        program->device_count != 1U || topology->task_count != program->task_count ||
        topology->alias_count != program->alias_count ||
        topology->pool_capacity_bytes == 0U ||
        topology->object_capacity_bytes == 0U ||
        topology->object_capacity_bytes > topology->pool_capacity_bytes ||
        topology->minimum_alignment == 0U ||
        topology->task_workspace_offsets == NULL ||
        topology->fresh_output_offsets == NULL ||
        topology->replacement_offsets == NULL ||
        topology->handoff_offsets == NULL ||
        topology->task_allocation_offsets == NULL) {
        return 0;
    }
    const uint32_t workspace_count =
        topology->task_workspace_offsets[topology->task_count];
    const uint32_t fresh_count =
        topology->fresh_output_offsets[topology->task_count];
    const uint32_t replacement_count =
        topology->replacement_offsets[topology->task_count];
    const uint32_t handoff_count =
        topology->handoff_offsets[topology->task_count];
    const uint32_t allocation_count =
        topology->task_allocation_offsets[topology->task_count];
    if ((workspace_count != 0U &&
         topology->task_workspace_extent_bytes == NULL) ||
        (fresh_count != 0U && topology->fresh_output_aliases == NULL) ||
        (replacement_count != 0U && topology->replacement_aliases == NULL) ||
        (handoff_count != 0U &&
         (topology->handoff_source_aliases == NULL ||
          topology->handoff_destination_aliases == NULL)) ||
        (allocation_count != 0U &&
         (topology->task_allocation_slots == NULL ||
          topology->task_allocation_bytes == NULL ||
          topology->task_allocation_aliases == NULL ||
          topology->task_allocation_kinds == NULL))) {
        return 0;
    }
    for (uint32_t task = 0U; task < topology->task_count; ++task) {
        if (topology->task_workspace_offsets[task] >
                topology->task_workspace_offsets[task + 1U] ||
            topology->fresh_output_offsets[task] >
                topology->fresh_output_offsets[task + 1U] ||
            topology->replacement_offsets[task] >
                topology->replacement_offsets[task + 1U] ||
            topology->handoff_offsets[task] >
                topology->handoff_offsets[task + 1U] ||
            topology->task_allocation_offsets[task] >
                topology->task_allocation_offsets[task + 1U]) {
            return 0;
        }
    }
    for (uint32_t index = 0U; index < workspace_count; ++index) {
        if (topology->task_workspace_extent_bytes[index] == 0U) {
            return 0;
        }
    }
    for (uint32_t index = 0U; index < fresh_count; ++index) {
        if (topology->fresh_output_aliases[index] >= topology->alias_count) {
            return 0;
        }
    }
    for (uint32_t index = 0U; index < replacement_count; ++index) {
        if (topology->replacement_aliases[index] >= topology->alias_count) {
            return 0;
        }
    }
    for (uint32_t index = 0U; index < handoff_count; ++index) {
        if (topology->handoff_source_aliases[index] >= topology->alias_count ||
            topology->handoff_destination_aliases[index] >=
                topology->alias_count ||
            topology->handoff_source_aliases[index] ==
                topology->handoff_destination_aliases[index]) {
            return 0;
        }
    }
    for (uint32_t index = 0U; index < allocation_count; ++index) {
        const uint8_t kind = topology->task_allocation_kinds[index];
        const uint32_t alias = topology->task_allocation_aliases[index];
        if (topology->task_allocation_slots[index] >=
                topology->allocation_slot_count ||
            kind > TASK_ALLOCATION_RELEASE ||
            (kind == TASK_ALLOCATION_ALLOCATE &&
             topology->task_allocation_bytes[index] == 0U) ||
            (kind == TASK_ALLOCATION_RELEASE &&
             (topology->task_allocation_bytes[index] != 0U ||
              alias != SHADOWSPILL_SIMULATOR_NO_INDEX)) ||
            (alias != SHADOWSPILL_SIMULATOR_NO_INDEX &&
             alias >= topology->alias_count)) {
            return 0;
        }
    }
    return 1;
}

static uint64_t invariant_lease_count(
    const ShadowSpillPressureFitProblem *problem
) {
    const ShadowSpillAdmissionFacts *topology = problem->admission;
    const ShadowSpillSimulationProgram *program = problem->simulation;
    uint64_t count = program->alias_count;
    count += topology->allocation_slot_count;
    return count;
}

static uint64_t invariant_operation_count(
    const ShadowSpillPressureFitProblem *problem
) {
    const ShadowSpillAdmissionFacts *topology = problem->admission;
    const ShadowSpillSimulationProgram *program = problem->simulation;
    uint64_t count = (uint64_t)program->alias_count * 2U;
    const uint32_t allocation_count =
        topology->task_allocation_offsets[topology->task_count];
    for (uint32_t index = 0U; index < allocation_count; ++index) {
        count += topology->task_allocation_kinds[index] ==
            TASK_ALLOCATION_ALLOCATE ? 1U : 2U;
    }
    count += (uint64_t)topology->replacement_offsets[topology->task_count] * 2U;
    return count;
}

static void free_action_buffers(
    ShadowSpillCandidateAdmissionWorkspace *workspace
) {
    free(workspace->action_trigger_deltas);
    free(workspace->action_completion_deltas);
    free(workspace->reuse_predecessor_actions);
    free(workspace->reuse_successor_tasks);
    free(workspace->reuse_successor_actions);
    workspace->action_trigger_deltas = NULL;
    workspace->action_completion_deltas = NULL;
    workspace->reuse_predecessor_actions = NULL;
    workspace->reuse_successor_tasks = NULL;
    workspace->reuse_successor_actions = NULL;
    workspace->action_capacity = 0U;
}

/* How many leases and operations a schedule can produce. Pure arithmetic:
 * callers that only want the sizes must not pay for the buffers. */
static int admission_counts(
    const ShadowSpillPressureFitProblem *problem,
    const ShadowSpillIndexedSchedule *schedule,
    uint64_t *lease_count,
    uint64_t *operation_count,
    uint64_t *dependency_count
) {
    *lease_count = invariant_lease_count(problem);
    *operation_count = invariant_operation_count(problem);
    *dependency_count = problem->simulation->task_count;
    for (uint32_t action = 0U; action < schedule->action_count; ++action) {
        const uint8_t kind = schedule->action_kinds[action];
        if (kind == SHADOWSPILL_MEMORY_PREFETCH) {
            if (checked_add(*lease_count, 1U, lease_count) != 0 ||
                checked_add(*operation_count, 2U, operation_count) != 0) {
                return -1;
            }
        } else if (kind == SHADOWSPILL_MEMORY_OFFLOAD) {
            if (checked_add(*dependency_count, 1U, dependency_count) != 0 ||
                checked_add(*operation_count, 2U, operation_count) != 0) {
                return -1;
            }
        } else if (kind == SHADOWSPILL_MEMORY_RELEASE) {
            if (checked_add(*operation_count, 2U, operation_count) != 0) {
                return -1;
            }
        } else {
            return -1;
        }
    }
    return 0;
}

int shadowspill_admission_counts(
    const ShadowSpillPressureFitProblem *problem,
    const ShadowSpillIndexedSchedule *schedule,
    uint64_t *lease_count,
    uint64_t *operation_count
) {
    uint64_t dependency_count = 0U;
    return admission_counts(
        problem, schedule, lease_count, operation_count, &dependency_count
    );
}

/*
 * Growing the admission workspace.
 *
 * The workspace keeps one scratch buffer per thing admission counts, and a
 * candidate reuses it across its whole search. Reserving means growing every
 * buffer that a bigger schedule outgrew -- all of them or none, because a
 * workspace half-grown by a failed allocation would be read with the old
 * capacities and the new pointers.
 *
 * Each buffer is described once here rather than written out at every step,
 * so what the workspace holds and how each one is sized can be read in one
 * place.
 */
typedef struct ReserveSpec {
    /* The workspace field, as void ** so one loop can fill every buffer.
     * Every object pointer has the same representation, which is what makes
     * that safe. */
    void **destination;
    size_t element_size;
    uint64_t count;
    /* Whether the buffer is read before it is written. */
    int zeroed;
} ReserveSpec;

/* Allocate every buffer or none, then swap them in and release what they
 * replace. */
static int reserve_buffers(const ReserveSpec *specs, size_t spec_count) {
    void **staged = calloc(spec_count, sizeof(*staged));
    if (staged == NULL) {
        return -1;
    }
    int failed = 0;
    for (size_t index = 0U; index < spec_count; ++index) {
        const size_t slots =
            specs[index].count == 0U ? 1U : (size_t)specs[index].count;
        staged[index] = specs[index].zeroed
            ? calloc(slots, specs[index].element_size)
            : malloc(slots * specs[index].element_size);
        failed |= staged[index] == NULL;
    }
    if (failed) {
        for (size_t index = 0U; index < spec_count; ++index) {
            free(staged[index]);
        }
        free(staged);
        return -1;
    }
    for (size_t index = 0U; index < spec_count; ++index) {
        free(*specs[index].destination);
        *specs[index].destination = staged[index];
    }
    free(staged);
    return 0;
}

/* Every buffer sized by the lease and operation counts. */
static int reserve_lease_buffers(
    ShadowSpillCandidateAdmissionWorkspace *workspace,
    uint64_t operation_count,
    uint64_t lease_count,
    uint64_t dependency_count,
    uint64_t reuse_dependency_count
) {
    const ReserveSpec specs[] = {
        {(void **)&workspace->operations, sizeof(*workspace->operations),
         operation_count, 1},
        {(void **)&workspace->decisions, sizeof(*workspace->decisions),
         operation_count, 1},
        {(void **)&workspace->annotations, sizeof(*workspace->annotations),
         operation_count, 1},
        {(void **)&workspace->purposes, sizeof(*workspace->purposes),
         operation_count, 1},
        {(void **)&workspace->allocation_offsets,
         sizeof(*workspace->allocation_offsets), operation_count, 0},
        {(void **)&workspace->dependencies, sizeof(*workspace->dependencies),
         reuse_dependency_count, 1},
        {(void **)&workspace->live_leases, sizeof(*workspace->live_leases),
         lease_count, 1},
        {(void **)&workspace->lease_aliases, sizeof(*workspace->lease_aliases),
         lease_count, 0},
        {(void **)&workspace->lease_start_operations,
         sizeof(*workspace->lease_start_operations), lease_count, 0},
        {(void **)&workspace->lease_retire_operations,
         sizeof(*workspace->lease_retire_operations), lease_count, 0},
        /* Two entries per lease plus both ends. */
        {(void **)&workspace->repair_candidate_starts,
         sizeof(*workspace->repair_candidate_starts), lease_count * 2U + 2U, 0},
        /* Prefix sums, so one longer than what they sum over. */
        {(void **)&workspace->repair_blocked_prefix,
         sizeof(*workspace->repair_blocked_prefix), lease_count + 1U, 0},
        {(void **)&workspace->repair_unremovable_prefix,
         sizeof(*workspace->repair_unremovable_prefix), lease_count + 1U, 0},
        {(void **)&workspace->pending_retirements,
         sizeof(*workspace->pending_retirements), lease_count, 1},
        {(void **)&workspace->predecessor_actions,
         sizeof(*workspace->predecessor_actions), lease_count, 0},
        {(void **)&workspace->predecessor_tasks,
         sizeof(*workspace->predecessor_tasks), lease_count, 0},
    };
    /* The replay workspace owns its own allocations, so it is built beside
     * the table rather than in it, and discarded if the table fails. */
    ShadowSpillAdmissionReplayWorkspace *replay = NULL;
    if (shadowspill_admission_replay_workspace_create(
            lease_count, dependency_count, &replay
        ) != SHADOWSPILL_STATUS_OK) {
        return -1;
    }
    if (reserve_buffers(specs, sizeof(specs) / sizeof(*specs)) != 0) {
        shadowspill_admission_replay_workspace_destroy(replay);
        return -1;
    }
    shadowspill_admission_replay_workspace_destroy(workspace->replay);
    workspace->replay = replay;
    workspace->operation_capacity = operation_count;
    workspace->lease_capacity = lease_count;
    workspace->dependency_capacity = dependency_count;
    workspace->reuse_dependency_capacity = reuse_dependency_count;
    return 0;
}

/* Every buffer sized by how many actions the schedule has. */
static int reserve_action_buffers(
    ShadowSpillCandidateAdmissionWorkspace *workspace, uint32_t action_count
) {
    const ReserveSpec specs[] = {
        {(void **)&workspace->action_trigger_deltas,
         sizeof(*workspace->action_trigger_deltas), action_count, 1},
        {(void **)&workspace->action_completion_deltas,
         sizeof(*workspace->action_completion_deltas), action_count, 1},
        {(void **)&workspace->reuse_predecessor_actions,
         sizeof(*workspace->reuse_predecessor_actions), action_count, 1},
        {(void **)&workspace->reuse_successor_tasks,
         sizeof(*workspace->reuse_successor_tasks), action_count, 1},
        {(void **)&workspace->reuse_successor_actions,
         sizeof(*workspace->reuse_successor_actions), action_count, 1},
    };
    if (reserve_buffers(specs, sizeof(specs) / sizeof(*specs)) != 0) {
        return -1;
    }
    workspace->action_capacity = action_count;
    return 0;
}

/* Whether any dimension outgrew what the workspace holds. */
static int lease_buffers_outgrown(
    const ShadowSpillCandidateAdmissionWorkspace *workspace,
    uint64_t operation_count,
    uint64_t lease_count,
    uint64_t dependency_count,
    uint64_t reuse_dependency_count
) {
    return operation_count > workspace->operation_capacity ||
        lease_count > workspace->lease_capacity ||
        dependency_count > workspace->dependency_capacity ||
        reuse_dependency_count > workspace->reuse_dependency_capacity;
}

static uint64_t at_least(uint64_t value, uint64_t floor_value) {
    return value < floor_value ? floor_value : value;
}

int shadowspill_admission_reserve_buffers(
    const ShadowSpillPressureFitProblem *problem,
    const ShadowSpillIndexedSchedule *schedule,
    ShadowSpillCandidateAdmissionWorkspace *workspace
) {
    uint64_t lease_count = 0U;
    uint64_t operation_count = 0U;
    uint64_t dependency_count = 0U;
    if (admission_counts(
            problem, schedule, &lease_count, &operation_count, &dependency_count
        ) != 0) {
        return -1;
    }
    if (lease_count > SIZE_MAX || operation_count > SIZE_MAX ||
        dependency_count > SIZE_MAX || lease_count > (SIZE_MAX - 2U) / 2U) {
        return -1;
    }
    uint64_t reuse_dependency_count = lease_count;
    if (lease_buffers_outgrown(
            workspace, operation_count, lease_count, dependency_count,
            reuse_dependency_count
        )) {
        /* Keep every scratch dimension at its prior high-water mark: growing
         * one must not shrink another the same buffers are shared with. */
        if (reserve_lease_buffers(
                workspace,
                at_least(operation_count, workspace->operation_capacity),
                at_least(lease_count, workspace->lease_capacity),
                at_least(dependency_count, workspace->dependency_capacity),
                at_least(
                    reuse_dependency_count, workspace->reuse_dependency_capacity
                )
            ) != 0) {
            return -1;
        }
    }
    if (schedule->action_count > workspace->action_capacity &&
        reserve_action_buffers(workspace, schedule->action_count) != 0) {
        return -1;
    }
    return 0;
}

int shadowspill_candidate_admission_workspace_create(
    const ShadowSpillPressureFitProblem *problem,
    ShadowSpillCandidateAdmissionWorkspace *workspace
) {
    if (workspace == NULL || !shadowspill_admission_facts_valid(problem)) {
        return -1;
    }
    memset(workspace, 0, sizeof(*workspace));
    const uint32_t aliases = problem->simulation->alias_count;
    const uint32_t tasks = problem->simulation->task_count;
    workspace->active_alias_leases = malloc(
        (aliases == 0U ? 1U : aliases) * sizeof(*workspace->active_alias_leases)
    );
    workspace->new_alias_leases = malloc(
        (aliases == 0U ? 1U : aliases) * sizeof(*workspace->new_alias_leases)
    );
    const uint32_t allocation_slots = problem->admission->allocation_slot_count;
    workspace->task_allocation_leases = malloc(
        (allocation_slots == 0U ? 1U : allocation_slots) *
            sizeof(*workspace->task_allocation_leases)
    );
    workspace->task_allocation_live = calloc(
        allocation_slots == 0U ? 1U : allocation_slots,
        sizeof(*workspace->task_allocation_live)
    );
    workspace->handoff_sources = calloc(
        aliases == 0U ? 1U : aliases,
        sizeof(*workspace->handoff_sources)
    );
    workspace->task_start_deltas = calloc(
        tasks == 0U ? 1U : tasks,
        sizeof(*workspace->task_start_deltas)
    );
    workspace->task_completion_deltas = calloc(
        tasks == 0U ? 1U : tasks,
        sizeof(*workspace->task_completion_deltas)
    );
    if (workspace->active_alias_leases == NULL ||
        workspace->new_alias_leases == NULL ||
        workspace->task_allocation_leases == NULL ||
        workspace->task_allocation_live == NULL ||
        workspace->handoff_sources == NULL ||
        workspace->task_start_deltas == NULL ||
        workspace->task_completion_deltas == NULL) {
        shadowspill_candidate_admission_workspace_destroy(workspace);
        return -1;
    }
    return 0;
}

void shadowspill_candidate_admission_workspace_destroy(
    ShadowSpillCandidateAdmissionWorkspace *workspace
) {
    if (workspace == NULL) {
        return;
    }
    free(workspace->operations);
    free(workspace->decisions);
    free(workspace->dependencies);
    free(workspace->live_leases);
    free(workspace->annotations);
    free(workspace->purposes);
    free(workspace->allocation_offsets);
    free(workspace->active_alias_leases);
    free(workspace->new_alias_leases);
    free(workspace->task_allocation_leases);
    free(workspace->task_allocation_live);
    free(workspace->lease_aliases);
    free(workspace->lease_start_operations);
    free(workspace->lease_retire_operations);
    free(workspace->repair_candidate_starts);
    free(workspace->repair_blocked_prefix);
    free(workspace->repair_unremovable_prefix);
    free(workspace->predecessor_actions);
    free(workspace->predecessor_tasks);
    free(workspace->handoff_sources);
    free(workspace->pending_retirements);
    free(workspace->task_start_deltas);
    free(workspace->task_completion_deltas);
    free_action_buffers(workspace);
    shadowspill_admission_replay_workspace_destroy(workspace->replay);
    memset(workspace, 0, sizeof(*workspace));
}
