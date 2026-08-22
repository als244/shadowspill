/* The pool operations a schedule implies.
 *
 * Executing a schedule means acquiring a lease per object generation, retiring
 * it when the object is released, evicted or replaced, and publishing the
 * dependency that makes a later reuse of its address safe. This file derives
 * that operation sequence. It runs nothing: replaying the sequence through the
 * pool is `candidate.c`, and placing the leases at fixed addresses is
 * `placement.c`.
 */

#include "admission_internal.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define NO_LEASE UINT64_MAX
#define TASK_ALLOCATION_ALLOCATE 0U
#define TASK_ALLOCATION_RELEASE 1U

static int append_operation(
    ScriptState *state,
    uint64_t lease_id,
    uint8_t kind,
    uint64_t bytes,
    uint64_t alignment,
    uint64_t dependency_id,
    uint8_t dependency_expected,
    uint8_t boundary,
    uint32_t index,
    uint8_t purpose,
    uint32_t allocation_offset
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
    state->workspace->purposes[operation] = purpose;
    state->workspace->allocation_offsets[operation] = allocation_offset;
    return 0;
}

static int acquire_lease(
    ScriptState *state,
    uint64_t bytes,
    uint64_t alignment,
    uint8_t boundary,
    uint32_t index,
    uint32_t owner_alias,
    uint8_t purpose,
    uint64_t *lease_id
) {
    if (bytes == 0U || state->lease_count >= state->workspace->lease_capacity) {
        return -1;
    }
    const uint64_t lease = state->lease_count++;
    state->workspace->lease_aliases[lease] = owner_alias;
    if (append_operation(
            state, lease, SHADOWSPILL_ADMISSION_REPLAY_RESERVE,
            bytes, alignment, SHADOWSPILL_ADMISSION_REPLAY_NO_ID, 0U,
            boundary, index, purpose, SHADOWSPILL_PLANNER_NO_INDEX
        ) != 0 ||
        append_operation(
            state, lease, SHADOWSPILL_ADMISSION_REPLAY_ACQUIRE_RESERVED,
            0U, 0U, SHADOWSPILL_ADMISSION_REPLAY_NO_ID, 0U,
            boundary, index, purpose, SHADOWSPILL_PLANNER_NO_INDEX
        ) != 0) {
        return -1;
    }
    *lease_id = lease;
    return 0;
}

static int acquire_task_lease(
    ScriptState *state,
    uint64_t bytes,
    uint64_t alignment,
    uint32_t task,
    uint32_t owner_alias,
    uint8_t purpose,
    uint32_t allocation_offset,
    uint64_t *lease_id
) {
    if (bytes == 0U || state->lease_count >= state->workspace->lease_capacity) {
        return -1;
    }
    const uint64_t lease = state->lease_count++;
    state->workspace->lease_aliases[lease] = owner_alias;
    if (append_operation(
            state, lease, SHADOWSPILL_ADMISSION_REPLAY_ACQUIRE,
            bytes, alignment, SHADOWSPILL_ADMISSION_REPLAY_NO_ID, 0U,
            SHADOWSPILL_ADMISSION_BOUNDARY_TASK_START, task, purpose,
            allocation_offset
        ) != 0) {
        return -1;
    }
    *lease_id = lease;
    return 0;
}

static uint64_t task_completion_dependency(
    ScriptState *state,
    uint64_t *dependency
) {
    if (*dependency == SHADOWSPILL_ADMISSION_REPLAY_NO_ID) {
        if (state->dependency_count >= state->workspace->dependency_capacity) {
            return SHADOWSPILL_ADMISSION_REPLAY_NO_ID;
        }
        *dependency = state->dependency_count++;
    }
    return *dependency;
}

static int begin_retirement(
    ScriptState *state,
    uint64_t lease_id,
    uint64_t dependency_id,
    uint8_t begin_boundary,
    uint32_t begin_index,
    uint8_t completion_boundary,
    uint32_t completion_index,
    uint32_t predecessor_task,
    uint32_t predecessor_action,
    uint8_t purpose
) {
    if (state->pending_count >= state->workspace->lease_capacity ||
        append_operation(
            state, lease_id,
            SHADOWSPILL_ADMISSION_REPLAY_BEGIN_RETIREMENT,
            0U, 0U, dependency_id, 0U, begin_boundary, begin_index, purpose,
            SHADOWSPILL_PLANNER_NO_INDEX
        ) != 0) {
        return -1;
    }
    state->workspace->pending_retirements[state->pending_count++] =
        (ShadowSpillPendingRetirement){
            .lease_id = lease_id,
            .dependency_id = dependency_id,
            .completion_index = completion_index,
            .completion_boundary = completion_boundary,
            /* An eviction's begin only starts the copy; by its completion
             * the copy has landed and the lease is simply gone. */
            .completion_purpose =
                purpose == SHADOWSPILL_ADMISSION_PURPOSE_EVICTION
                    ? SHADOWSPILL_ADMISSION_PURPOSE_TERMINAL_COMPLETION
                    : purpose,
        };
    state->workspace->predecessor_tasks[lease_id] = predecessor_task;
    state->workspace->predecessor_actions[lease_id] = predecessor_action;
    return 0;
}

static int task_slot_reused_after(
    const ShadowSpillAdmissionTopology *topology,
    uint32_t task,
    uint32_t operation,
    uint32_t slot
) {
    for (uint32_t offset = operation + 1U;
         offset < topology->task_allocation_offsets[task + 1U]; ++offset) {
        if (topology->task_allocation_kinds[offset] ==
                TASK_ALLOCATION_ALLOCATE &&
            topology->task_allocation_slots[offset] == slot) {
            return 1;
        }
    }
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

static int task_replaces_alias(
    const ShadowSpillAdmissionTopology *topology,
    uint32_t task,
    uint32_t alias
) {
    for (uint32_t offset = topology->replacement_offsets[task];
         offset < topology->replacement_offsets[task + 1U]; ++offset) {
        if (topology->replacement_aliases[offset] == alias) {
            return 1;
        }
    }
    return 0;
}

int shadowspill_admission_build_operations(
    const ShadowSpillPressureFitContext *context,
    const ShadowSpillIndexedSchedule *schedule,
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
    for (uint32_t slot = 0U; slot < topology->allocation_slot_count; ++slot) {
        workspace->task_allocation_leases[slot] = NO_LEASE;
        workspace->task_allocation_live[slot] = 0U;
    }
    for (uint64_t lease = 0U; lease < workspace->lease_capacity; ++lease) {
        workspace->predecessor_actions[lease] = UINT32_MAX;
        workspace->predecessor_tasks[lease] = UINT32_MAX;
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
                SHADOWSPILL_ADMISSION_BOUNDARY_INITIAL, 0U, alias,
                SHADOWSPILL_ADMISSION_PURPOSE_INITIAL_OBJECT,
                &workspace->active_alias_leases[alias]
            ) != 0) {
            return -1;
        }
    }

    uint32_t action_cursor = 0U;
    for (uint32_t task = 0U; task < program->task_count; ++task) {
        uint64_t task_dependency = SHADOWSPILL_ADMISSION_REPLAY_NO_ID;
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

        for (uint32_t offset = topology->task_allocation_offsets[task];
             offset < topology->task_allocation_offsets[task + 1U]; ++offset) {
            const uint32_t slot = topology->task_allocation_slots[offset];
            const uint8_t kind = topology->task_allocation_kinds[offset];
            const uint32_t alias = topology->task_allocation_aliases[offset];
            if (kind == TASK_ALLOCATION_ALLOCATE) {
                if (workspace->task_allocation_live[slot] != 0U ||
                    (alias != SHADOWSPILL_SIMULATOR_NO_INDEX &&
                     (workspace->new_alias_leases[alias] != NO_LEASE ||
                      ((workspace->active_alias_leases[alias] != NO_LEASE) !=
                       task_replaces_alias(topology, task, alias))))) {
                    return -1;
                }
                if (workspace->task_allocation_leases[slot] == NO_LEASE) {
                    if (acquire_task_lease(
                            state,
                            topology->task_allocation_bytes[offset],
                            topology->minimum_alignment,
                            task,
                            alias,
                            alias == SHADOWSPILL_SIMULATOR_NO_INDEX
                                ? SHADOWSPILL_ADMISSION_PURPOSE_TASK_WORKSPACE
                                : task_replaces_alias(topology, task, alias)
                                ? SHADOWSPILL_ADMISSION_PURPOSE_MUTATION_REPLACEMENT
                                : SHADOWSPILL_ADMISSION_PURPOSE_TASK_OUTPUT,
                            offset,
                            &workspace->task_allocation_leases[slot]
                        ) != 0) {
                        return -1;
                    }
                } else {
                    workspace->lease_aliases[
                        workspace->task_allocation_leases[slot]
                    ] = alias;
                }
                workspace->task_allocation_live[slot] = 1U;
                if (alias != SHADOWSPILL_SIMULATOR_NO_INDEX) {
                    workspace->new_alias_leases[alias] =
                        workspace->task_allocation_leases[slot];
                }
            } else {
                const uint64_t lease = workspace->task_allocation_leases[slot];
                if (lease == NO_LEASE ||
                    workspace->task_allocation_live[slot] == 0U) {
                    return -1;
                }
                workspace->task_allocation_live[slot] = 0U;
            }
        }
        for (uint32_t offset = topology->task_allocation_offsets[task];
             offset < topology->task_allocation_offsets[task + 1U]; ++offset) {
            const uint32_t slot = topology->task_allocation_slots[offset];
            if (topology->task_allocation_kinds[offset] !=
                    TASK_ALLOCATION_RELEASE ||
                task_slot_reused_after(topology, task, offset, slot)) {
                continue;
            }
            const uint64_t dependency = task_completion_dependency(
                state, &task_dependency
            );
            const uint64_t lease = workspace->task_allocation_leases[slot];
            if (lease == NO_LEASE ||
                dependency == SHADOWSPILL_ADMISSION_REPLAY_NO_ID ||
                begin_retirement(
                    state,
                    lease,
                    dependency,
                    SHADOWSPILL_ADMISSION_BOUNDARY_TASK_COMPLETION,
                    task,
                    SHADOWSPILL_ADMISSION_BOUNDARY_TASK_COMPLETION,
                    task,
                    task,
                    UINT32_MAX,
                    SHADOWSPILL_ADMISSION_PURPOSE_TASK_WORKSPACE
                ) != 0) {
                return -1;
            }
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
            workspace->lease_aliases[
                workspace->active_alias_leases[destination]
            ] = destination;
            workspace->active_alias_leases[source] = NO_LEASE;
            workspace->handoff_sources[source] = 1U;
        }
        for (uint32_t offset = topology->replacement_offsets[task];
             offset < topology->replacement_offsets[task + 1U]; ++offset) {
            const uint32_t alias = topology->replacement_aliases[offset];
            if (program->alias_size_bytes[alias] == 0U) {
                continue;
            }
            const uint64_t dependency = task_completion_dependency(
                state, &task_dependency
            );
            if (workspace->active_alias_leases[alias] == NO_LEASE ||
                workspace->new_alias_leases[alias] == NO_LEASE ||
                dependency == SHADOWSPILL_ADMISSION_REPLAY_NO_ID ||
                begin_retirement(
                    state,
                    workspace->active_alias_leases[alias],
                    dependency,
                    SHADOWSPILL_ADMISSION_BOUNDARY_TASK_COMPLETION,
                    task,
                    SHADOWSPILL_ADMISSION_BOUNDARY_TASK_COMPLETION,
                    task,
                    task,
                    UINT32_MAX,
                    SHADOWSPILL_ADMISSION_PURPOSE_MUTATION_REPLACEMENT
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
            if (workspace->active_alias_leases[alias] != NO_LEASE ||
                workspace->new_alias_leases[alias] == NO_LEASE) {
                return -1;
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
                if (program->alias_size_bytes[alias] == 0U) {
                    continue;
                }
                if (workspace->handoff_sources[alias] != 0U) {
                    continue;
                }
                const uint64_t lease = workspace->active_alias_leases[alias];
                const uint64_t dependency = task_completion_dependency(
                    state, &task_dependency
                );
                if (lease == NO_LEASE ||
                    dependency == SHADOWSPILL_ADMISSION_REPLAY_NO_ID ||
                    begin_retirement(
                        state,
                        lease,
                        dependency,
                        SHADOWSPILL_ADMISSION_BOUNDARY_ACTION_TRIGGER,
                        action,
                        SHADOWSPILL_ADMISSION_BOUNDARY_ACTION_COMPLETION,
                        action,
                        task,
                        UINT32_MAX,
                        SHADOWSPILL_ADMISSION_PURPOSE_RELEASE
                    ) != 0) {
                    return -1;
                }
                workspace->active_alias_leases[alias] = NO_LEASE;
            } else if (kind == SHADOWSPILL_MEMORY_OFFLOAD) {
                const uint64_t lease = workspace->active_alias_leases[alias];
                if (program->alias_size_bytes[alias] == 0U) {
                    continue;
                }
                if (lease == NO_LEASE ||
                    state->dependency_count >= workspace->dependency_capacity) {
                    return -1;
                }
                const uint64_t dependency = state->dependency_count++;
                if (begin_retirement(
                        state,
                        lease,
                        dependency,
                        SHADOWSPILL_ADMISSION_BOUNDARY_ACTION_TRIGGER,
                        action,
                        SHADOWSPILL_ADMISSION_BOUNDARY_ACTION_COMPLETION,
                        action,
                        UINT32_MAX,
                        action,
                        SHADOWSPILL_ADMISSION_PURPOSE_EVICTION
                    ) != 0) {
                    return -1;
                }
                state->evict_bytes += program->alias_size_bytes[alias];
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
                        alias,
                        SHADOWSPILL_ADMISSION_PURPOSE_FETCH_DESTINATION,
                        &workspace->active_alias_leases[alias]
                    ) != 0) {
                    return -1;
                }
                state->fetch_bytes += program->alias_size_bytes[alias];
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
        const ShadowSpillPendingRetirement pending =
            workspace->pending_retirements[index];
        if (append_operation(
                state, pending.lease_id,
                SHADOWSPILL_ADMISSION_REPLAY_COMPLETE_RETIREMENT,
                0U, 0U, pending.dependency_id, 0U,
                pending.completion_boundary,
                pending.completion_index,
                pending.completion_purpose,
                SHADOWSPILL_PLANNER_NO_INDEX
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

/* ---------------------------------------------------------------- public */

/* A context carrying only what operation building reads: the resolved task
 * set and the physical ownership facts. Residency and seed state belong to
 * candidate search, not here. */
static ShadowSpillPressureFitContext operations_context(
    const ShadowSpillSimulationProgram *simulation,
    const ShadowSpillAdmissionTopology *admission
) {
    return (ShadowSpillPressureFitContext){
        .abi_version = SHADOWSPILL_PLANNER_ABI_VERSION,
        .simulation = simulation,
        .admission = admission,
    };
}

ShadowSpillPlannerStatus shadowspill_admission_operation_bounds(
    const ShadowSpillSimulationProgram *simulation,
    const ShadowSpillAdmissionTopology *admission,
    const ShadowSpillIndexedSchedule *schedule,
    uint64_t *operation_capacity,
    uint64_t *lease_capacity
) {
    if (schedule == NULL || operation_capacity == NULL ||
        lease_capacity == NULL) {
        return SHADOWSPILL_PLANNER_INVALID_ARGUMENT;
    }
    const ShadowSpillPressureFitContext context =
        operations_context(simulation, admission);
    if (!shadowspill_admission_topology_valid(&context)) {
        return SHADOWSPILL_PLANNER_INVALID_ARGUMENT;
    }
    ShadowSpillCandidateAdmissionWorkspace workspace = {0};
    if (shadowspill_admission_reserve_buffers(&context, schedule, &workspace)
            != 0) {
        shadowspill_candidate_admission_workspace_destroy(&workspace);
        return SHADOWSPILL_PLANNER_ALLOCATION_FAILURE;
    }
    *operation_capacity = workspace.operation_capacity;
    *lease_capacity = workspace.lease_capacity;
    shadowspill_candidate_admission_workspace_destroy(&workspace);
    return SHADOWSPILL_PLANNER_OK;
}

ShadowSpillPlannerStatus shadowspill_build_admission_operations(
    const ShadowSpillSimulationProgram *simulation,
    const ShadowSpillAdmissionTopology *admission,
    const ShadowSpillIndexedSchedule *schedule,
    ShadowSpillAdmissionOperations *result
) {
    if (schedule == NULL || result == NULL || result->lease_ids == NULL ||
        result->dependency_ids == NULL || result->bytes == NULL || result->alignments == NULL ||
        result->kinds == NULL || result->purposes == NULL ||
        result->boundaries == NULL || result->indices == NULL ||
        result->allocation_offsets == NULL || result->lease_aliases == NULL) {
        return SHADOWSPILL_PLANNER_INVALID_ARGUMENT;
    }
    const ShadowSpillPressureFitContext context =
        operations_context(simulation, admission);
    if (!shadowspill_admission_topology_valid(&context)) {
        return SHADOWSPILL_PLANNER_INVALID_ARGUMENT;
    }

    ShadowSpillCandidateAdmissionWorkspace workspace = {0};
    if (shadowspill_candidate_admission_workspace_create(&context, &workspace)
            != 0 ||
        shadowspill_admission_reserve_buffers(&context, schedule, &workspace)
            != 0) {
        shadowspill_candidate_admission_workspace_destroy(&workspace);
        return SHADOWSPILL_PLANNER_ALLOCATION_FAILURE;
    }
    if (workspace.operation_capacity > result->operation_capacity ||
        workspace.lease_capacity > result->lease_capacity) {
        shadowspill_candidate_admission_workspace_destroy(&workspace);
        return SHADOWSPILL_PLANNER_INVALID_ARGUMENT;
    }

    ScriptState state;
    if (shadowspill_admission_build_operations(
            &context, schedule, &workspace, &state) != 0) {
        shadowspill_candidate_admission_workspace_destroy(&workspace);
        return SHADOWSPILL_PLANNER_INTERNAL_ERROR;
    }

    for (uint64_t index = 0U; index < state.operation_count; ++index) {
        const ShadowSpillAdmissionReplayOperation operation =
            workspace.operations[index];
        result->lease_ids[index] = operation.lease_id;
        result->dependency_ids[index] = operation.dependency_id;
        result->bytes[index] = operation.bytes;
        result->alignments[index] = operation.alignment;
        result->kinds[index] = operation.kind;
        result->purposes[index] = workspace.purposes[index];
        result->boundaries[index] = workspace.annotations[index].boundary;
        result->indices[index] = workspace.annotations[index].index;
        result->allocation_offsets[index] = workspace.allocation_offsets[index];
    }
    for (uint64_t lease = 0U; lease < state.lease_count; ++lease) {
        result->lease_aliases[lease] = workspace.lease_aliases[lease];
    }
    result->operation_count = state.operation_count;
    result->lease_count = state.lease_count;
    result->dependency_count = state.dependency_count;
    result->fetch_bytes = state.fetch_bytes;
    result->evict_bytes = state.evict_bytes;
    shadowspill_candidate_admission_workspace_destroy(&workspace);
    return SHADOWSPILL_PLANNER_OK;
}
