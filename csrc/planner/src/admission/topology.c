/* Validating an admission topology and sizing the buffers it requires.
 *
 * Every other file here assumes a topology that has already passed
 * `shadowspill_admission_topology_valid`, and a workspace big enough for the
 * operations the schedule will produce. Both are established once, here.
 */

#include "admission_internal.h"

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

int shadowspill_admission_topology_valid(
    const ShadowSpillPressureFitContext *context
) {
    if (context == NULL || context->admission == NULL ||
        context->simulation == NULL) {
        return 0;
    }
    const ShadowSpillAdmissionTopology *topology = context->admission;
    const ShadowSpillSimulationProgram *program = context->simulation;
    if (topology->abi_version != SHADOWSPILL_PLANNER_ABI_VERSION ||
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
    const ShadowSpillPressureFitContext *context
) {
    const ShadowSpillAdmissionTopology *topology = context->admission;
    const ShadowSpillSimulationProgram *program = context->simulation;
    uint64_t count = program->alias_count;
    count += topology->allocation_slot_count;
    return count;
}

static uint64_t invariant_operation_count(
    const ShadowSpillPressureFitContext *context
) {
    const ShadowSpillAdmissionTopology *topology = context->admission;
    const ShadowSpillSimulationProgram *program = context->simulation;
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

int shadowspill_admission_reserve_buffers(
    const ShadowSpillPressureFitContext *context,
    const ShadowSpillIndexedSchedule *schedule,
    ShadowSpillCandidateAdmissionWorkspace *workspace
) {
    uint64_t lease_count = invariant_lease_count(context);
    uint64_t operation_count = invariant_operation_count(context);
    uint64_t dependency_count = context->simulation->task_count;
    for (uint32_t action = 0U; action < schedule->action_count; ++action) {
        const uint8_t kind = schedule->action_kinds[action];
        if (kind == SHADOWSPILL_MEMORY_PREFETCH) {
            if (checked_add(lease_count, 1U, &lease_count) != 0 ||
                checked_add(operation_count, 2U, &operation_count) != 0) {
                return -1;
            }
        } else if (kind == SHADOWSPILL_MEMORY_OFFLOAD) {
            if (checked_add(dependency_count, 1U, &dependency_count) != 0 ||
                checked_add(operation_count, 2U, &operation_count) != 0) {
                return -1;
            }
        } else if (kind == SHADOWSPILL_MEMORY_RELEASE) {
            if (checked_add(operation_count, 2U, &operation_count) != 0) {
                return -1;
            }
        } else {
            return -1;
        }
    }
    if (lease_count > SIZE_MAX || operation_count > SIZE_MAX ||
        dependency_count > SIZE_MAX || lease_count > (SIZE_MAX - 2U) / 2U) {
        return -1;
    }
    uint64_t reuse_dependency_count = lease_count;
    if (operation_count > workspace->operation_capacity ||
        lease_count > workspace->lease_capacity ||
        dependency_count > workspace->dependency_capacity ||
        reuse_dependency_count > workspace->reuse_dependency_capacity) {
        /* Keep every scratch dimension at its prior high-water mark. */
        if (operation_count < workspace->operation_capacity) {
            operation_count = workspace->operation_capacity;
        }
        if (lease_count < workspace->lease_capacity) {
            lease_count = workspace->lease_capacity;
        }
        if (dependency_count < workspace->dependency_capacity) {
            dependency_count = workspace->dependency_capacity;
        }
        if (reuse_dependency_count < workspace->reuse_dependency_capacity) {
            reuse_dependency_count = workspace->reuse_dependency_capacity;
        }
        ShadowSpillAdmissionReplayOperation *operations = calloc(
            operation_count == 0U ? 1U : (size_t)operation_count,
            sizeof(*operations)
        );
        ShadowSpillAdmissionReplayDecision *decisions = calloc(
            operation_count == 0U ? 1U : (size_t)operation_count,
            sizeof(*decisions)
        );
        ShadowSpillAdmissionAnnotation *annotations = calloc(
            operation_count == 0U ? 1U : (size_t)operation_count,
            sizeof(*annotations)
        );
        ShadowSpillAdmissionReuseDependency *dependencies = calloc(
            reuse_dependency_count == 0U
                ? 1U : (size_t)reuse_dependency_count,
            sizeof(*dependencies)
        );
        ShadowSpillAdmissionReplayLiveLease *live_leases = calloc(
            lease_count == 0U ? 1U : (size_t)lease_count,
            sizeof(*live_leases)
        );
        uint32_t *lease_aliases = malloc(
            (lease_count == 0U ? 1U : (size_t)lease_count) *
                sizeof(*lease_aliases)
        );
        uint64_t *repair_candidate_starts = malloc(
            ((size_t)lease_count * 2U + 2U) *
                sizeof(*repair_candidate_starts)
        );
        uint64_t *repair_blocked_prefix = malloc(
            ((size_t)lease_count + 1U) * sizeof(*repair_blocked_prefix)
        );
        uint32_t *repair_unremovable_prefix = malloc(
            ((size_t)lease_count + 1U) * sizeof(*repair_unremovable_prefix)
        );
        ShadowSpillPendingRetirement *pending = calloc(
            lease_count == 0U ? 1U : (size_t)lease_count,
            sizeof(*pending)
        );
        uint32_t *predecessors = malloc(
            (lease_count == 0U ? 1U : (size_t)lease_count) *
                sizeof(*predecessors)
        );
        uint32_t *predecessor_tasks = malloc(
            (lease_count == 0U ? 1U : (size_t)lease_count) *
                sizeof(*predecessor_tasks)
        );
        ShadowSpillAdmissionReplayWorkspace *replay = NULL;
        if (operations == NULL || decisions == NULL || annotations == NULL ||
            dependencies == NULL || live_leases == NULL ||
            lease_aliases == NULL || repair_candidate_starts == NULL ||
            repair_blocked_prefix == NULL ||
            repair_unremovable_prefix == NULL || pending == NULL ||
            predecessors == NULL || predecessor_tasks == NULL ||
            shadowspill_admission_replay_workspace_create(
                lease_count, dependency_count, &replay
            ) != SHADOWSPILL_ADMISSION_REPLAY_OK) {
            free(operations);
            free(decisions);
            free(annotations);
            free(dependencies);
            free(live_leases);
            free(lease_aliases);
            free(repair_candidate_starts);
            free(repair_blocked_prefix);
            free(repair_unremovable_prefix);
            free(pending);
            free(predecessors);
            free(predecessor_tasks);
            shadowspill_admission_replay_workspace_destroy(replay);
            return -1;
        }
        free(workspace->operations);
        free(workspace->decisions);
        free(workspace->annotations);
        free(workspace->dependencies);
        free(workspace->live_leases);
        free(workspace->lease_aliases);
        free(workspace->repair_candidate_starts);
        free(workspace->repair_blocked_prefix);
        free(workspace->repair_unremovable_prefix);
        free(workspace->pending_retirements);
        free(workspace->predecessor_actions);
        free(workspace->predecessor_tasks);
        shadowspill_admission_replay_workspace_destroy(workspace->replay);
        workspace->operations = operations;
        workspace->decisions = decisions;
        workspace->annotations = annotations;
        workspace->dependencies = dependencies;
        workspace->live_leases = live_leases;
        workspace->lease_aliases = lease_aliases;
        workspace->repair_candidate_starts = repair_candidate_starts;
        workspace->repair_blocked_prefix = repair_blocked_prefix;
        workspace->repair_unremovable_prefix = repair_unremovable_prefix;
        workspace->pending_retirements = pending;
        workspace->predecessor_actions = predecessors;
        workspace->predecessor_tasks = predecessor_tasks;
        workspace->replay = replay;
        workspace->operation_capacity = operation_count;
        workspace->lease_capacity = lease_count;
        workspace->dependency_capacity = dependency_count;
        workspace->reuse_dependency_capacity = reuse_dependency_count;
    }
    if (schedule->action_count > workspace->action_capacity) {
        const size_t count = schedule->action_count == 0U
            ? 1U : (size_t)schedule->action_count;
        int64_t *trigger = calloc(count, sizeof(*trigger));
        int64_t *completion = calloc(count, sizeof(*completion));
        uint32_t *predecessors = calloc(count, sizeof(*predecessors));
        uint32_t *successor_tasks = calloc(count, sizeof(*successor_tasks));
        uint32_t *successor_actions = calloc(count, sizeof(*successor_actions));
        if (trigger == NULL || completion == NULL || predecessors == NULL ||
            successor_tasks == NULL || successor_actions == NULL) {
            free(trigger);
            free(completion);
            free(predecessors);
            free(successor_tasks);
            free(successor_actions);
            return -1;
        }
        free_action_buffers(workspace);
        workspace->action_trigger_deltas = trigger;
        workspace->action_completion_deltas = completion;
        workspace->reuse_predecessor_actions = predecessors;
        workspace->reuse_successor_tasks = successor_tasks;
        workspace->reuse_successor_actions = successor_actions;
        workspace->action_capacity = schedule->action_count;
    }
    return 0;
}

int shadowspill_candidate_admission_workspace_create(
    const ShadowSpillPressureFitContext *context,
    ShadowSpillCandidateAdmissionWorkspace *workspace
) {
    if (workspace == NULL || !shadowspill_admission_topology_valid(context)) {
        return -1;
    }
    memset(workspace, 0, sizeof(*workspace));
    const uint32_t aliases = context->simulation->alias_count;
    const uint32_t tasks = context->simulation->task_count;
    workspace->active_alias_leases = malloc(
        (aliases == 0U ? 1U : aliases) * sizeof(*workspace->active_alias_leases)
    );
    workspace->new_alias_leases = malloc(
        (aliases == 0U ? 1U : aliases) * sizeof(*workspace->new_alias_leases)
    );
    const uint32_t allocation_slots = context->admission->allocation_slot_count;
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
    free(workspace->active_alias_leases);
    free(workspace->new_alias_leases);
    free(workspace->task_allocation_leases);
    free(workspace->task_allocation_live);
    free(workspace->lease_aliases);
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
