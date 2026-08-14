#define _POSIX_C_SOURCE 200809L

#include "admission_internal.h"
#include "portfolio_internal.h"

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NO_LEASE UINT64_MAX

static uint64_t monotonic_time_ns(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        return 0U;
    }
    return (uint64_t)value.tv_sec * UINT64_C(1000000000) +
        (uint64_t)value.tv_nsec;
}

static int checked_add(uint64_t left, uint64_t right, uint64_t *result) {
    if (right > UINT64_MAX - left) {
        return -1;
    }
    *result = left + right;
    return 0;
}

static int topology_valid(const ShadowSpillPressureFitContext *context) {
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
        topology->task_workspace_bytes == NULL ||
        topology->fresh_output_offsets == NULL ||
        topology->replacement_offsets == NULL ||
        topology->handoff_offsets == NULL) {
        return 0;
    }
    const uint32_t fresh_count =
        topology->fresh_output_offsets[topology->task_count];
    const uint32_t replacement_count =
        topology->replacement_offsets[topology->task_count];
    const uint32_t handoff_count =
        topology->handoff_offsets[topology->task_count];
    if ((fresh_count != 0U && topology->fresh_output_aliases == NULL) ||
        (replacement_count != 0U && topology->replacement_aliases == NULL) ||
        (handoff_count != 0U &&
         (topology->handoff_source_aliases == NULL ||
          topology->handoff_destination_aliases == NULL))) {
        return 0;
    }
    for (uint32_t task = 0U; task < topology->task_count; ++task) {
        if (topology->fresh_output_offsets[task] >
                topology->fresh_output_offsets[task + 1U] ||
            topology->replacement_offsets[task] >
                topology->replacement_offsets[task + 1U] ||
            topology->handoff_offsets[task] >
                topology->handoff_offsets[task + 1U]) {
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
    return 1;
}

static uint64_t invariant_lease_count(
    const ShadowSpillPressureFitContext *context
) {
    const ShadowSpillAdmissionTopology *topology = context->admission;
    const ShadowSpillSimulationProgram *program = context->simulation;
    uint64_t count = program->alias_count;
    for (uint32_t task = 0U; task < topology->task_count; ++task) {
        count += topology->task_workspace_bytes[task] != 0U ? 1U : 0U;
    }
    count += topology->fresh_output_offsets[topology->task_count];
    count += topology->replacement_offsets[topology->task_count];
    return count;
}

static uint64_t invariant_operation_count(
    const ShadowSpillPressureFitContext *context
) {
    const ShadowSpillAdmissionTopology *topology = context->admission;
    const ShadowSpillSimulationProgram *program = context->simulation;
    uint64_t count = (uint64_t)program->alias_count * 2U;
    for (uint32_t task = 0U; task < topology->task_count; ++task) {
        count += topology->task_workspace_bytes[task] != 0U ? 3U : 0U;
    }
    count += (uint64_t)topology->fresh_output_offsets[topology->task_count] * 2U;
    count +=
        (uint64_t)topology->replacement_offsets[topology->task_count] * 3U;
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

static int reserve_candidate_buffers(
    const ShadowSpillPressureFitContext *context,
    const ShadowSpillDenseSchedule *schedule,
    ShadowSpillCandidateAdmissionWorkspace *workspace
) {
    uint64_t lease_count = invariant_lease_count(context);
    uint64_t operation_count = invariant_operation_count(context);
    uint64_t dependency_count = 0U;
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
            if (checked_add(operation_count, 1U, &operation_count) != 0) {
                return -1;
            }
        } else {
            return -1;
        }
    }
    if (lease_count > SIZE_MAX || operation_count > SIZE_MAX ||
        dependency_count > SIZE_MAX) {
        return -1;
    }
    if (operation_count > workspace->operation_capacity ||
        lease_count > workspace->lease_capacity ||
        dependency_count > workspace->dependency_capacity) {
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
            dependency_count == 0U ? 1U : (size_t)dependency_count,
            sizeof(*dependencies)
        );
        ShadowSpillPendingEviction *pending = calloc(
            dependency_count == 0U ? 1U : (size_t)dependency_count,
            sizeof(*pending)
        );
        uint32_t *predecessors = malloc(
            (lease_count == 0U ? 1U : (size_t)lease_count) *
                sizeof(*predecessors)
        );
        ShadowSpillAdmissionReplayWorkspace *replay = NULL;
        if (operations == NULL || decisions == NULL || annotations == NULL ||
            dependencies == NULL || pending == NULL || predecessors == NULL ||
            shadowspill_admission_replay_workspace_create(
                lease_count, dependency_count, &replay
            ) != SHADOWSPILL_ADMISSION_REPLAY_OK) {
            free(operations);
            free(decisions);
            free(annotations);
            free(dependencies);
            free(pending);
            free(predecessors);
            shadowspill_admission_replay_workspace_destroy(replay);
            return -1;
        }
        free(workspace->operations);
        free(workspace->decisions);
        free(workspace->annotations);
        free(workspace->dependencies);
        free(workspace->pending_evictions);
        free(workspace->predecessor_actions);
        shadowspill_admission_replay_workspace_destroy(workspace->replay);
        workspace->operations = operations;
        workspace->decisions = decisions;
        workspace->annotations = annotations;
        workspace->dependencies = dependencies;
        workspace->pending_evictions = pending;
        workspace->predecessor_actions = predecessors;
        workspace->replay = replay;
        workspace->operation_capacity = operation_count;
        workspace->lease_capacity = lease_count;
        workspace->dependency_capacity = dependency_count;
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
    if (workspace == NULL || !topology_valid(context)) {
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
    free(workspace->annotations);
    free(workspace->active_alias_leases);
    free(workspace->new_alias_leases);
    free(workspace->predecessor_actions);
    free(workspace->handoff_sources);
    free(workspace->pending_evictions);
    free(workspace->task_start_deltas);
    free(workspace->task_completion_deltas);
    free_action_buffers(workspace);
    shadowspill_admission_replay_workspace_destroy(workspace->replay);
    memset(workspace, 0, sizeof(*workspace));
}

typedef struct ScriptState {
    ShadowSpillCandidateAdmissionWorkspace *workspace;
    uint64_t operation_count;
    uint64_t lease_count;
    uint64_t dependency_count;
    uint64_t pending_count;
} ScriptState;

static int append_operation(
    ScriptState *state,
    uint64_t lease_id,
    uint8_t kind,
    uint64_t bytes,
    uint64_t alignment,
    uint64_t dependency_id,
    uint8_t dependency_expected,
    uint8_t boundary,
    uint32_t index
) {
    if (state->operation_count >= state->workspace->operation_capacity) {
        return -1;
    }
    const uint64_t operation = state->operation_count++;
    state->workspace->operations[operation] =
        (ShadowSpillAdmissionReplayOperation){
            .sequence = operation,
            .lease_id = lease_id,
            .dependency_id = dependency_id,
            .bytes = bytes,
            .alignment = alignment,
            .kind = kind,
            .dependency_expected = dependency_expected,
        };
    state->workspace->annotations[operation] =
        (ShadowSpillAdmissionAnnotation){.index = index, .boundary = boundary};
    return 0;
}

static int acquire_lease(
    ScriptState *state,
    uint64_t bytes,
    uint64_t alignment,
    uint8_t boundary,
    uint32_t index,
    uint64_t *lease_id
) {
    if (bytes == 0U || state->lease_count >= state->workspace->lease_capacity) {
        return -1;
    }
    const uint64_t lease = state->lease_count++;
    if (append_operation(
            state, lease, SHADOWSPILL_ADMISSION_REPLAY_RESERVE,
            bytes, alignment, SHADOWSPILL_ADMISSION_REPLAY_NO_ID, 0U,
            boundary, index
        ) != 0 ||
        append_operation(
            state, lease, SHADOWSPILL_ADMISSION_REPLAY_ACQUIRE_RESERVED,
            0U, 0U, SHADOWSPILL_ADMISSION_REPLAY_NO_ID, 0U,
            boundary, index
        ) != 0) {
        return -1;
    }
    *lease_id = lease;
    return 0;
}

static int require_alias(
    const ShadowSpillSimulationProgram *program,
    const ShadowSpillCandidateAdmissionWorkspace *workspace,
    uint32_t alias
) {
    return alias < program->alias_count &&
        (program->alias_size_bytes[alias] == 0U ||
         workspace->active_alias_leases[alias] != NO_LEASE);
}

static int build_script(
    const ShadowSpillPressureFitContext *context,
    const ShadowSpillDenseSchedule *schedule,
    ShadowSpillCandidateAdmissionWorkspace *workspace,
    ScriptState *state
) {
    const ShadowSpillSimulationProgram *program = context->simulation;
    const ShadowSpillAdmissionTopology *topology = context->admission;
    memset(state, 0, sizeof(*state));
    state->workspace = workspace;
    for (uint32_t alias = 0U; alias < program->alias_count; ++alias) {
        workspace->active_alias_leases[alias] = NO_LEASE;
        workspace->new_alias_leases[alias] = NO_LEASE;
        workspace->handoff_sources[alias] = 0U;
    }

    for (uint32_t index = 0U; index < schedule->initial_count; ++index) {
        const uint32_t alias = schedule->initial_aliases[index];
        if (alias >= program->alias_count ||
            schedule->initial_locations[index] > SHADOWSPILL_MEMORY_HOST) {
            return -1;
        }
        if (schedule->initial_locations[index] != SHADOWSPILL_MEMORY_DEVICE ||
            program->alias_size_bytes[alias] == 0U) {
            continue;
        }
        if (workspace->active_alias_leases[alias] != NO_LEASE ||
            acquire_lease(
                state, program->alias_size_bytes[alias],
                topology->minimum_alignment,
                SHADOWSPILL_ADMISSION_BOUNDARY_INITIAL, 0U,
                &workspace->active_alias_leases[alias]
            ) != 0) {
            return -1;
        }
    }

    uint32_t action_cursor = 0U;
    for (uint32_t task = 0U; task < program->task_count; ++task) {
        for (uint32_t offset = program->input_offsets[task];
             offset < program->input_offsets[task + 1U]; ++offset) {
            if (!require_alias(program, workspace, program->input_aliases[offset])) {
                return -1;
            }
        }
        for (uint32_t offset = program->mutation_offsets[task];
             offset < program->mutation_offsets[task + 1U]; ++offset) {
            if (!require_alias(
                    program, workspace, program->mutation_aliases[offset]
                )) {
                return -1;
            }
        }

        uint64_t workspace_lease = NO_LEASE;
        if (topology->task_workspace_bytes[task] != 0U &&
            acquire_lease(
                state, topology->task_workspace_bytes[task],
                topology->minimum_alignment,
                SHADOWSPILL_ADMISSION_BOUNDARY_TASK_START, task,
                &workspace_lease
            ) != 0) {
            return -1;
        }
        for (uint32_t offset = topology->fresh_output_offsets[task];
             offset < topology->fresh_output_offsets[task + 1U]; ++offset) {
            const uint32_t alias = topology->fresh_output_aliases[offset];
            if (program->alias_size_bytes[alias] == 0U) {
                continue;
            }
            if (workspace->active_alias_leases[alias] != NO_LEASE ||
                workspace->new_alias_leases[alias] != NO_LEASE ||
                acquire_lease(
                    state, program->alias_size_bytes[alias],
                    topology->minimum_alignment,
                    SHADOWSPILL_ADMISSION_BOUNDARY_TASK_START, task,
                    &workspace->new_alias_leases[alias]
                ) != 0) {
                return -1;
            }
        }
        for (uint32_t offset = topology->replacement_offsets[task];
             offset < topology->replacement_offsets[task + 1U]; ++offset) {
            const uint32_t alias = topology->replacement_aliases[offset];
            if (program->alias_size_bytes[alias] == 0U) {
                continue;
            }
            if (workspace->active_alias_leases[alias] == NO_LEASE ||
                workspace->new_alias_leases[alias] != NO_LEASE ||
                acquire_lease(
                    state, program->alias_size_bytes[alias],
                    topology->minimum_alignment,
                    SHADOWSPILL_ADMISSION_BOUNDARY_TASK_START, task,
                    &workspace->new_alias_leases[alias]
                ) != 0) {
                return -1;
            }
        }

        if (workspace_lease != NO_LEASE && append_operation(
                state, workspace_lease, SHADOWSPILL_ADMISSION_REPLAY_RELEASE,
                0U, 0U, SHADOWSPILL_ADMISSION_REPLAY_NO_ID, 0U,
                SHADOWSPILL_ADMISSION_BOUNDARY_TASK_COMPLETION, task
            ) != 0) {
            return -1;
        }
        for (uint32_t offset = topology->handoff_offsets[task];
             offset < topology->handoff_offsets[task + 1U]; ++offset) {
            const uint32_t source = topology->handoff_source_aliases[offset];
            const uint32_t destination =
                topology->handoff_destination_aliases[offset];
            if (program->alias_size_bytes[destination] == 0U) {
                continue;
            }
            if (workspace->active_alias_leases[source] == NO_LEASE ||
                workspace->active_alias_leases[destination] != NO_LEASE) {
                return -1;
            }
            workspace->active_alias_leases[destination] =
                workspace->active_alias_leases[source];
            workspace->active_alias_leases[source] = NO_LEASE;
            workspace->handoff_sources[source] = 1U;
        }
        for (uint32_t offset = topology->replacement_offsets[task];
             offset < topology->replacement_offsets[task + 1U]; ++offset) {
            const uint32_t alias = topology->replacement_aliases[offset];
            if (program->alias_size_bytes[alias] == 0U) {
                continue;
            }
            if (append_operation(
                    state, workspace->active_alias_leases[alias],
                    SHADOWSPILL_ADMISSION_REPLAY_RELEASE,
                    0U, 0U, SHADOWSPILL_ADMISSION_REPLAY_NO_ID, 0U,
                    SHADOWSPILL_ADMISSION_BOUNDARY_TASK_COMPLETION, task
                ) != 0) {
                return -1;
            }
            workspace->active_alias_leases[alias] =
                workspace->new_alias_leases[alias];
            workspace->new_alias_leases[alias] = NO_LEASE;
        }
        for (uint32_t offset = topology->fresh_output_offsets[task];
             offset < topology->fresh_output_offsets[task + 1U]; ++offset) {
            const uint32_t alias = topology->fresh_output_aliases[offset];
            if (program->alias_size_bytes[alias] == 0U) {
                continue;
            }
            workspace->active_alias_leases[alias] =
                workspace->new_alias_leases[alias];
            workspace->new_alias_leases[alias] = NO_LEASE;
        }

        while (action_cursor < schedule->action_count &&
               schedule->action_trigger_tasks[action_cursor] == task) {
            const uint32_t action = action_cursor++;
            const uint32_t alias = schedule->action_aliases[action];
            if (alias >= program->alias_count) {
                return -1;
            }
            const uint8_t kind = schedule->action_kinds[action];
            if (kind == SHADOWSPILL_MEMORY_RELEASE) {
                if (workspace->handoff_sources[alias] != 0U) {
                    continue;
                }
                const uint64_t lease = workspace->active_alias_leases[alias];
                if ((program->alias_size_bytes[alias] != 0U && lease == NO_LEASE) ||
                    (lease != NO_LEASE && append_operation(
                        state, lease, SHADOWSPILL_ADMISSION_REPLAY_RELEASE,
                        0U, 0U, SHADOWSPILL_ADMISSION_REPLAY_NO_ID, 0U,
                        SHADOWSPILL_ADMISSION_BOUNDARY_ACTION_TRIGGER, action
                    ) != 0)) {
                    return -1;
                }
                workspace->active_alias_leases[alias] = NO_LEASE;
            } else if (kind == SHADOWSPILL_MEMORY_OFFLOAD) {
                const uint64_t lease = workspace->active_alias_leases[alias];
                if (program->alias_size_bytes[alias] == 0U) {
                    continue;
                }
                if (lease == NO_LEASE ||
                    state->dependency_count >= workspace->dependency_capacity ||
                    state->pending_count >= workspace->dependency_capacity) {
                    return -1;
                }
                const uint64_t dependency = state->dependency_count++;
                if (append_operation(
                        state, lease,
                        SHADOWSPILL_ADMISSION_REPLAY_BEGIN_RETIREMENT,
                        0U, 0U, dependency, 1U,
                        SHADOWSPILL_ADMISSION_BOUNDARY_ACTION_TRIGGER, action
                    ) != 0) {
                    return -1;
                }
                workspace->pending_evictions[state->pending_count++] =
                    (ShadowSpillPendingEviction){lease, dependency, action};
                workspace->predecessor_actions[lease] = action;
                workspace->active_alias_leases[alias] = NO_LEASE;
            } else if (kind == SHADOWSPILL_MEMORY_PREFETCH) {
                if (program->alias_size_bytes[alias] == 0U) {
                    continue;
                }
                if (workspace->active_alias_leases[alias] != NO_LEASE ||
                    acquire_lease(
                        state, program->alias_size_bytes[alias],
                        topology->minimum_alignment,
                        SHADOWSPILL_ADMISSION_BOUNDARY_ACTION_TRIGGER, action,
                        &workspace->active_alias_leases[alias]
                    ) != 0) {
                    return -1;
                }
            } else {
                return -1;
            }
        }
        for (uint32_t offset = topology->handoff_offsets[task];
             offset < topology->handoff_offsets[task + 1U]; ++offset) {
            workspace->handoff_sources[
                topology->handoff_source_aliases[offset]
            ] = 0U;
        }
    }
    if (action_cursor != schedule->action_count) {
        return -1;
    }
    for (uint64_t index = 0U; index < state->pending_count; ++index) {
        const ShadowSpillPendingEviction pending =
            workspace->pending_evictions[index];
        if (append_operation(
                state, pending.lease_id,
                SHADOWSPILL_ADMISSION_REPLAY_COMPLETE_RETIREMENT,
                0U, 0U, pending.dependency_id, 0U,
                SHADOWSPILL_ADMISSION_BOUNDARY_ACTION_COMPLETION,
                pending.action_index
            ) != 0) {
            return -1;
        }
    }
    for (uint32_t index = 0U; index < schedule->final_count; ++index) {
        const uint32_t alias = schedule->final_aliases[index];
        if (alias >= program->alias_count ||
            (schedule->final_locations[index] == SHADOWSPILL_MEMORY_DEVICE &&
             program->alias_size_bytes[alias] != 0U &&
             workspace->active_alias_leases[alias] == NO_LEASE)) {
            return -1;
        }
    }
    return 0;
}

static int add_delta(int64_t *target, int64_t delta) {
    if ((delta > 0 && *target > INT64_MAX - delta) ||
        (delta < 0 && *target < INT64_MIN - delta)) {
        return -1;
    }
    *target += delta;
    return 0;
}

static int project_result(
    const ShadowSpillPressureFitContext *context,
    const ShadowSpillDenseSchedule *schedule,
    ShadowSpillCandidateAdmissionWorkspace *workspace,
    const ScriptState *state,
    const ShadowSpillAdmissionReplayResult *result
) {
    const uint32_t tasks = context->simulation->task_count;
    memset(workspace->task_start_deltas, 0, tasks * sizeof(int64_t));
    memset(workspace->task_completion_deltas, 0, tasks * sizeof(int64_t));
    memset(
        workspace->action_trigger_deltas, 0,
        schedule->action_count * sizeof(int64_t)
    );
    memset(
        workspace->action_completion_deltas, 0,
        schedule->action_count * sizeof(int64_t)
    );
    workspace->initial_physical_bytes = 0U;
    for (uint64_t operation = 0U; operation < state->operation_count; ++operation) {
        const ShadowSpillAdmissionAnnotation annotation =
            workspace->annotations[operation];
        const int64_t delta = workspace->decisions[operation].physical_bytes_delta;
        switch ((ShadowSpillAdmissionBoundaryKind)annotation.boundary) {
            case SHADOWSPILL_ADMISSION_BOUNDARY_INITIAL:
                if (delta < 0 || (uint64_t)delta >
                        UINT64_MAX - workspace->initial_physical_bytes) {
                    return -1;
                }
                workspace->initial_physical_bytes += (uint64_t)delta;
                break;
            case SHADOWSPILL_ADMISSION_BOUNDARY_TASK_START:
                if (annotation.index >= tasks || add_delta(
                        &workspace->task_start_deltas[annotation.index], delta
                    ) != 0) {
                    return -1;
                }
                break;
            case SHADOWSPILL_ADMISSION_BOUNDARY_TASK_COMPLETION:
                if (annotation.index >= tasks || add_delta(
                        &workspace->task_completion_deltas[annotation.index], delta
                    ) != 0) {
                    return -1;
                }
                break;
            case SHADOWSPILL_ADMISSION_BOUNDARY_ACTION_TRIGGER:
                if (annotation.index >= schedule->action_count || add_delta(
                        &workspace->action_trigger_deltas[annotation.index], delta
                    ) != 0) {
                    return -1;
                }
                break;
            case SHADOWSPILL_ADMISSION_BOUNDARY_ACTION_COMPLETION:
                if (annotation.index >= schedule->action_count || add_delta(
                        &workspace->action_completion_deltas[annotation.index], delta
                    ) != 0) {
                    return -1;
                }
                break;
            default:
                return -1;
        }
    }
    for (uint64_t index = 0U; index < result->dependency_result_count; ++index) {
        const ShadowSpillAdmissionReuseDependency dependency =
            workspace->dependencies[index];
        if (dependency.predecessor_lease_id >= state->lease_count ||
            dependency.consumer_operation_index >= state->operation_count) {
            return -1;
        }
        const uint32_t predecessor =
            workspace->predecessor_actions[dependency.predecessor_lease_id];
        const ShadowSpillAdmissionAnnotation successor =
            workspace->annotations[dependency.consumer_operation_index];
        if (predecessor >= schedule->action_count ||
            schedule->action_kinds[predecessor] != SHADOWSPILL_MEMORY_OFFLOAD) {
            return -1;
        }
        workspace->reuse_predecessor_actions[index] = predecessor;
        if (successor.boundary == SHADOWSPILL_ADMISSION_BOUNDARY_TASK_START) {
            workspace->reuse_successor_tasks[index] = successor.index;
            workspace->reuse_successor_actions[index] =
                SHADOWSPILL_SIMULATOR_NO_INDEX;
        } else if (
            successor.boundary == SHADOWSPILL_ADMISSION_BOUNDARY_ACTION_TRIGGER
        ) {
            workspace->reuse_successor_tasks[index] =
                SHADOWSPILL_SIMULATOR_NO_INDEX;
            workspace->reuse_successor_actions[index] = successor.index;
        } else {
            return -1;
        }
    }
    return 0;
}

ShadowSpillAdmissionReplayStatus shadowspill_admit_dense_schedule(
    const ShadowSpillPressureFitContext *context,
    const ShadowSpillDenseSchedule *schedule,
    ShadowSpillCandidateAdmissionWorkspace *workspace,
    ShadowSpillSimulationProgram *program,
    ShadowSpillAdmissionReplayResult *replay_result
) {
    if (!topology_valid(context) || schedule == NULL || workspace == NULL ||
        program == NULL || replay_result == NULL ||
        reserve_candidate_buffers(context, schedule, workspace) != 0) {
        return SHADOWSPILL_ADMISSION_REPLAY_ALLOCATION_FAILURE;
    }
    const uint64_t started = monotonic_time_ns();
    ScriptState script = {0};
    if (build_script(context, schedule, workspace, &script) != 0) {
        return SHADOWSPILL_ADMISSION_REPLAY_INVALID_SCRIPT;
    }
    for (uint64_t index = 0U; index < script.lease_count; ++index) {
        workspace->predecessor_actions[index] = UINT32_MAX;
    }
    for (uint64_t index = 0U; index < script.pending_count; ++index) {
        const ShadowSpillPendingEviction pending =
            workspace->pending_evictions[index];
        workspace->predecessor_actions[pending.lease_id] = pending.action_index;
    }
    const ShadowSpillAdmissionReplayProgram replay_program = {
        .abi_version = SHADOWSPILL_ADMISSION_REPLAY_ABI_VERSION,
        .capacity_bytes = context->admission->pool_capacity_bytes,
        .minimum_alignment = context->admission->minimum_alignment,
        .lease_count = script.lease_count,
        .dependency_count = script.dependency_count,
        .operations = workspace->operations,
        .operation_count = script.operation_count,
    };
    *replay_result = (ShadowSpillAdmissionReplayResult){
        .decisions = workspace->decisions,
        .decision_capacity = workspace->operation_capacity,
        .dependencies = workspace->dependencies,
        .dependency_capacity = workspace->dependency_capacity,
    };
    const ShadowSpillAdmissionReplayStatus status =
        shadowspill_admission_replay_run_reusing(
            &replay_program, replay_result, workspace->replay
        );
    ++workspace->calls;
    workspace->time_ns += monotonic_time_ns() - started;
    if (status != SHADOWSPILL_ADMISSION_REPLAY_OK) {
        return status;
    }
    if (project_result(
            context, schedule, workspace, &script, replay_result
        ) != 0) {
        return SHADOWSPILL_ADMISSION_REPLAY_INVALID_SCRIPT;
    }
    shadowspill_bind_dense_schedule(context->simulation, schedule, program);
    workspace->physical_device = context->simulation->devices[0];
    workspace->physical_device.capacity_bytes =
        context->admission->pool_capacity_bytes;
    program->devices = &workspace->physical_device;
    program->use_admission_accounting = 1U;
    program->initial_physical_bytes = &workspace->initial_physical_bytes;
    program->task_start_physical_deltas = workspace->task_start_deltas;
    program->task_completion_physical_deltas = workspace->task_completion_deltas;
    program->action_trigger_physical_deltas = workspace->action_trigger_deltas;
    program->action_completion_physical_deltas =
        workspace->action_completion_deltas;
    program->reuse_dependency_count =
        (uint32_t)replay_result->dependency_result_count;
    program->reuse_predecessor_actions = workspace->reuse_predecessor_actions;
    program->reuse_successor_tasks = workspace->reuse_successor_tasks;
    program->reuse_successor_actions = workspace->reuse_successor_actions;
    workspace->decision_digest = replay_result->decision_digest;
    workspace->peak_allocated_bytes = replay_result->peak_allocated_bytes;
    workspace->peak_reserved_bytes = replay_result->peak_reserved_bytes;
    workspace->peak_fragmentation_bytes = replay_result->peak_fragmentation_bytes;
    return status;
}

ShadowSpillPlannerStatus shadowspill_evaluate_schedule_admission(
    const ShadowSpillSimulationProgram *simulation,
    const ShadowSpillAdmissionTopology *admission,
    const ShadowSpillDenseSchedule *schedule,
    ShadowSpillScheduleAdmissionResult *result
) {
    if (result == NULL) {
        return SHADOWSPILL_PLANNER_INVALID_ARGUMENT;
    }
    int64_t *task_start = result->task_start_deltas;
    int64_t *task_completion = result->task_completion_deltas;
    const uint32_t task_capacity = result->task_capacity;
    int64_t *action_trigger = result->action_trigger_deltas;
    int64_t *action_completion = result->action_completion_deltas;
    const uint32_t action_capacity = result->action_capacity;
    uint32_t *reuse_predecessors = result->reuse_predecessor_actions;
    uint32_t *reuse_tasks = result->reuse_successor_tasks;
    uint32_t *reuse_actions = result->reuse_successor_actions;
    const uint32_t reuse_capacity = result->reuse_capacity;
    *result = (ShadowSpillScheduleAdmissionResult){
        .status = SHADOWSPILL_ADMISSION_REPLAY_INVALID_ARGUMENT,
        .task_start_deltas = task_start,
        .task_completion_deltas = task_completion,
        .task_capacity = task_capacity,
        .action_trigger_deltas = action_trigger,
        .action_completion_deltas = action_completion,
        .action_capacity = action_capacity,
        .reuse_predecessor_actions = reuse_predecessors,
        .reuse_successor_tasks = reuse_tasks,
        .reuse_successor_actions = reuse_actions,
        .reuse_capacity = reuse_capacity,
    };
    const ShadowSpillPressureFitContext context = {
        .abi_version = SHADOWSPILL_PLANNER_ABI_VERSION,
        .simulation = simulation,
        .admission = admission,
    };
    if (!topology_valid(&context) || schedule == NULL ||
        task_capacity < simulation->task_count ||
        action_capacity < schedule->action_count ||
        (simulation->task_count != 0U &&
         (task_start == NULL || task_completion == NULL)) ||
        (schedule->action_count != 0U &&
         (action_trigger == NULL || action_completion == NULL))) {
        return SHADOWSPILL_PLANNER_INVALID_ARGUMENT;
    }
    ShadowSpillCandidateAdmissionWorkspace workspace = {0};
    if (shadowspill_candidate_admission_workspace_create(
            &context, &workspace
        ) != 0) {
        return SHADOWSPILL_PLANNER_ALLOCATION_FAILURE;
    }
    ShadowSpillSimulationProgram admitted_program = {0};
    ShadowSpillAdmissionReplayResult replay = {0};
    const ShadowSpillAdmissionReplayStatus status =
        shadowspill_admit_dense_schedule(
            &context,
            schedule,
            &workspace,
            &admitted_program,
            &replay
        );
    result->status = (uint32_t)status;
    result->decision_digest = replay.decision_digest;
    result->peak_allocated_bytes = replay.peak_allocated_bytes;
    result->peak_reserved_bytes = replay.peak_reserved_bytes;
    result->peak_fragmentation_bytes = replay.peak_fragmentation_bytes;
    result->error_operation_index = replay.error_operation_index;
    result->error_requested_bytes = replay.error_requested_bytes;
    result->error_free_bytes = replay.error_free_bytes;
    result->error_largest_free_range_bytes =
        replay.error_largest_free_range_bytes;
    if (status == SHADOWSPILL_ADMISSION_REPLAY_OK) {
        if (replay.dependency_result_count > reuse_capacity) {
            shadowspill_candidate_admission_workspace_destroy(&workspace);
            return SHADOWSPILL_PLANNER_INVALID_ARGUMENT;
        }
        result->initial_physical_bytes = workspace.initial_physical_bytes;
        result->reuse_count = (uint32_t)replay.dependency_result_count;
        if (simulation->task_count != 0U) {
            memcpy(
                task_start,
                workspace.task_start_deltas,
                (size_t)simulation->task_count * sizeof(*task_start)
            );
            memcpy(
                task_completion,
                workspace.task_completion_deltas,
                (size_t)simulation->task_count * sizeof(*task_completion)
            );
        }
        if (schedule->action_count != 0U) {
            memcpy(
                action_trigger,
                workspace.action_trigger_deltas,
                (size_t)schedule->action_count * sizeof(*action_trigger)
            );
            memcpy(
                action_completion,
                workspace.action_completion_deltas,
                (size_t)schedule->action_count * sizeof(*action_completion)
            );
        }
        if (result->reuse_count != 0U) {
            memcpy(
                reuse_predecessors,
                workspace.reuse_predecessor_actions,
                (size_t)result->reuse_count * sizeof(*reuse_predecessors)
            );
            memcpy(
                reuse_tasks,
                workspace.reuse_successor_tasks,
                (size_t)result->reuse_count * sizeof(*reuse_tasks)
            );
            memcpy(
                reuse_actions,
                workspace.reuse_successor_actions,
                (size_t)result->reuse_count * sizeof(*reuse_actions)
            );
        }
    }
    shadowspill_candidate_admission_workspace_destroy(&workspace);
    if (status == SHADOWSPILL_ADMISSION_REPLAY_OK) {
        return SHADOWSPILL_PLANNER_OK;
    }
    if (status == SHADOWSPILL_ADMISSION_REPLAY_INFEASIBLE) {
        return SHADOWSPILL_PLANNER_NO_FEASIBLE_CANDIDATE;
    }
    return status == SHADOWSPILL_ADMISSION_REPLAY_ALLOCATION_FAILURE
        ? SHADOWSPILL_PLANNER_ALLOCATION_FAILURE
        : SHADOWSPILL_PLANNER_INVALID_ARGUMENT;
}
