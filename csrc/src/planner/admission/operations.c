/* The pool operations a schedule implies.
 *
 * Executing a schedule means acquiring a lease per object generation, retiring
 * it when the object is released, evicted or replaced, and publishing the
 * dependency that makes a later reuse of its address safe. This file derives
 * that operation sequence. It runs nothing: replaying the sequence through the
 * pool is `candidate.c`, and placing the leases at fixed addresses is
 * `placement.c`.
 */

#include "internal.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define NO_LEASE SHADOWSPILL_ADMISSION_NO_LEASE
#define TASK_ALLOCATION_ALLOCATE 0U
#define TASK_ALLOCATION_RELEASE 1U

static int append_operation(
    OperationTally *tally,
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
    if (tally->operation_count >= tally->workspace->operation_capacity) {
        return -1;
    }
    const uint64_t operation = tally->operation_count++;
    tally->workspace->operations[operation] =
        (ShadowSpillAdmissionReplayOperation){
            .sequence = operation,
            .lease_id = lease_id,
            .dependency_id = dependency_id,
            .bytes = bytes,
            .alignment = alignment,
            .kind = kind,
            .dependency_expected = dependency_expected,
        };
    tally->workspace->annotations[operation] =
        (ShadowSpillAdmissionAnnotation){.index = index, .boundary = boundary};
    tally->workspace->purposes[operation] = purpose;
    tally->workspace->allocation_offsets[operation] = allocation_offset;
    return 0;
}

static int acquire_lease(
    OperationTally *tally,
    uint64_t bytes,
    uint64_t alignment,
    uint8_t boundary,
    uint32_t index,
    uint32_t owner_alias,
    uint8_t purpose,
    uint64_t *lease_id
) {
    if (bytes == 0U || tally->lease_count >= tally->workspace->lease_capacity) {
        return -1;
    }
    const uint64_t lease = tally->lease_count++;
    tally->workspace->lease_aliases[lease] = owner_alias;
    tally->workspace->lease_start_operations[lease] = tally->operation_count;
    tally->workspace->lease_retire_operations[lease] =
        SHADOWSPILL_ADMISSION_NO_OPERATION;
    if (append_operation(
            tally, lease, SHADOWSPILL_ADMISSION_REPLAY_RESERVE,
            bytes, alignment, SHADOWSPILL_ADMISSION_REPLAY_NO_ID, 0U,
            boundary, index, purpose, SHADOWSPILL_PLANNER_NO_INDEX
        ) != 0 ||
        append_operation(
            tally, lease, SHADOWSPILL_ADMISSION_REPLAY_ACQUIRE_RESERVED,
            0U, 0U, SHADOWSPILL_ADMISSION_REPLAY_NO_ID, 0U,
            boundary, index, purpose, SHADOWSPILL_PLANNER_NO_INDEX
        ) != 0) {
        return -1;
    }
    *lease_id = lease;
    return 0;
}

static int acquire_task_lease(
    OperationTally *tally,
    uint64_t bytes,
    uint64_t alignment,
    uint32_t task,
    uint32_t owner_alias,
    uint8_t purpose,
    uint32_t allocation_offset,
    uint64_t *lease_id
) {
    if (bytes == 0U || tally->lease_count >= tally->workspace->lease_capacity) {
        return -1;
    }
    const uint64_t lease = tally->lease_count++;
    tally->workspace->lease_aliases[lease] = owner_alias;
    tally->workspace->lease_start_operations[lease] = tally->operation_count;
    tally->workspace->lease_retire_operations[lease] =
        SHADOWSPILL_ADMISSION_NO_OPERATION;
    if (append_operation(
            tally, lease, SHADOWSPILL_ADMISSION_REPLAY_ACQUIRE,
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
    OperationTally *tally,
    uint64_t *dependency
) {
    if (*dependency == SHADOWSPILL_ADMISSION_REPLAY_NO_ID) {
        if (tally->dependency_count >= tally->workspace->dependency_capacity) {
            return SHADOWSPILL_ADMISSION_REPLAY_NO_ID;
        }
        *dependency = tally->dependency_count++;
    }
    return *dependency;
}

static int begin_retirement(
    OperationTally *tally,
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
    if (tally->workspace->lease_retire_operations[lease_id] ==
        SHADOWSPILL_ADMISSION_NO_OPERATION) {
        tally->workspace->lease_retire_operations[lease_id] =
            tally->operation_count;
    }
    if (tally->pending_count >= tally->workspace->lease_capacity ||
        append_operation(
            tally, lease_id,
            SHADOWSPILL_ADMISSION_REPLAY_BEGIN_RETIREMENT,
            0U, 0U, dependency_id, 0U, begin_boundary, begin_index, purpose,
            SHADOWSPILL_PLANNER_NO_INDEX
        ) != 0) {
        return -1;
    }
    tally->workspace->pending_retirements[tally->pending_count++] =
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
    tally->workspace->predecessor_tasks[lease_id] = predecessor_task;
    tally->workspace->predecessor_actions[lease_id] = predecessor_action;
    return 0;
}

static int task_slot_reused_after(
    const ShadowSpillAdmissionFacts *topology,
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
    const ShadowSpillAdmissionFacts *topology,
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

/*
 * Deriving pool operations from a schedule.
 *
 * Walking the step in order, every task does the same sequence of things to
 * memory: it requires its inputs to be there, opens the allocations it was
 * given, retires the ones nothing reuses, takes over aliases handed to it,
 * replaces what it mutated, adopts what it produced, and then the schedule's
 * own moves at that boundary happen. `build_task` is that sequence, and
 * `OperationBuild` is what the steps share.
 */
typedef struct OperationBuild {
    const ShadowSpillSimulationProgram *program;
    const ShadowSpillAdmissionFacts *topology;
    const ShadowSpillIndexedSchedule *schedule;
    ShadowSpillCandidateAdmissionWorkspace *workspace;
    OperationTally *tally;
    /* How far the schedule's actions have been consumed. Actions are ordered
     * by the task they trigger on, so one cursor walks them all. */
    uint32_t action_cursor;
} OperationBuild;

/* Nothing is live before the step starts. */
static void reset_alias_tracking(OperationBuild *build) {
    ShadowSpillCandidateAdmissionWorkspace *workspace = build->workspace;
    for (uint32_t alias = 0U; alias < build->program->alias_count; ++alias) {
        workspace->active_alias_leases[alias] = NO_LEASE;
        workspace->new_alias_leases[alias] = NO_LEASE;
        workspace->handoff_sources[alias] = 0U;
    }
    for (uint32_t slot = 0U; slot < build->topology->allocation_slot_count;
         ++slot) {
        workspace->task_allocation_leases[slot] = NO_LEASE;
        workspace->task_allocation_live[slot] = 0U;
    }
    for (uint64_t lease = 0U; lease < workspace->lease_capacity; ++lease) {
        workspace->predecessor_actions[lease] = UINT32_MAX;
        workspace->predecessor_tasks[lease] = UINT32_MAX;
    }
}

/* Objects the step starts holding on the device already occupy the pool. */
static int admit_initial_residency(OperationBuild *build) {
    const ShadowSpillIndexedSchedule *schedule = build->schedule;
    for (uint32_t index = 0U; index < schedule->initial_count; ++index) {
        const uint32_t alias = schedule->initial_aliases[index];
        if (alias >= build->program->alias_count ||
            schedule->initial_locations[index] > SHADOWSPILL_MEMORY_SPILL) {
            return -1;
        }
        if (schedule->initial_locations[index] != SHADOWSPILL_MEMORY_DEVICE ||
            build->program->alias_size_bytes[alias] == 0U) {
            continue;
        }
        if (build->workspace->active_alias_leases[alias] != NO_LEASE ||
            acquire_lease(
                build->tally,
                build->program->alias_size_bytes[alias],
                build->topology->minimum_alignment,
                SHADOWSPILL_ADMISSION_BOUNDARY_INITIAL,
                0U,
                alias,
                SHADOWSPILL_ADMISSION_PURPOSE_INITIAL_OBJECT,
                &build->workspace->active_alias_leases[alias]
            ) != 0) {
            return -1;
        }
    }
    return 0;
}

/* Everything a task reads has to be resident when it runs. */
static int require_task_aliases(OperationBuild *build, uint32_t task) {
    const ShadowSpillSimulationProgram *program = build->program;
    for (uint32_t offset = program->input_offsets[task];
         offset < program->input_offsets[task + 1U]; ++offset) {
        if (!require_alias(
                program, build->workspace, program->input_aliases[offset]
            )) {
            return -1;
        }
    }
    for (uint32_t offset = program->mutation_offsets[task];
         offset < program->mutation_offsets[task + 1U]; ++offset) {
        if (!require_alias(
                program, build->workspace, program->mutation_aliases[offset]
            )) {
            return -1;
        }
    }
    return 0;
}

/* The allocations the task was given: a slot becomes live on allocate and
 * stops being live on release, and a slot reused across tasks keeps its
 * lease rather than taking a new one. */
static int open_task_allocations(OperationBuild *build, uint32_t task) {
    const ShadowSpillAdmissionFacts *topology = build->topology;
    ShadowSpillCandidateAdmissionWorkspace *workspace = build->workspace;
    for (uint32_t offset = topology->task_allocation_offsets[task];
         offset < topology->task_allocation_offsets[task + 1U]; ++offset) {
        const uint32_t slot = topology->task_allocation_slots[offset];
        const uint8_t kind = topology->task_allocation_kinds[offset];
        const uint32_t alias = topology->task_allocation_aliases[offset];
        if (kind != TASK_ALLOCATION_ALLOCATE) {
            if (workspace->task_allocation_leases[slot] == NO_LEASE ||
                workspace->task_allocation_live[slot] == 0U) {
                return -1;
            }
            workspace->task_allocation_live[slot] = 0U;
            continue;
        }
        if (workspace->task_allocation_live[slot] != 0U ||
            (alias != SHADOWSPILL_SIMULATOR_NO_INDEX &&
             (workspace->new_alias_leases[alias] != NO_LEASE ||
              ((workspace->active_alias_leases[alias] != NO_LEASE) !=
               task_replaces_alias(topology, task, alias))))) {
            return -1;
        }
        if (workspace->task_allocation_leases[slot] == NO_LEASE) {
            if (acquire_task_lease(
                    build->tally,
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
            workspace->lease_aliases[workspace->task_allocation_leases[slot]] =
                alias;
        }
        workspace->task_allocation_live[slot] = 1U;
        if (alias != SHADOWSPILL_SIMULATOR_NO_INDEX) {
            workspace->new_alias_leases[alias] =
                workspace->task_allocation_leases[slot];
        }
    }
    return 0;
}

/* A slot released here and not reused later is finished with, so its lease
 * retires once the task completes. */
static int retire_task_allocations(
    OperationBuild *build, uint32_t task, uint64_t *task_dependency
) {
    const ShadowSpillAdmissionFacts *topology = build->topology;
    for (uint32_t offset = topology->task_allocation_offsets[task];
         offset < topology->task_allocation_offsets[task + 1U]; ++offset) {
        const uint32_t slot = topology->task_allocation_slots[offset];
        if (topology->task_allocation_kinds[offset] != TASK_ALLOCATION_RELEASE ||
            task_slot_reused_after(topology, task, offset, slot)) {
            continue;
        }
        const uint64_t dependency =
            task_completion_dependency(build->tally, task_dependency);
        const uint64_t lease = build->workspace->task_allocation_leases[slot];
        if (lease == NO_LEASE ||
            dependency == SHADOWSPILL_ADMISSION_REPLAY_NO_ID ||
            begin_retirement(
                build->tally,
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
    return 0;
}

/* A handoff moves an object's identity to another alias without moving the
 * memory: the destination takes over the source's lease. */
static int adopt_handoffs(OperationBuild *build, uint32_t task) {
    const ShadowSpillAdmissionFacts *topology = build->topology;
    ShadowSpillCandidateAdmissionWorkspace *workspace = build->workspace;
    for (uint32_t offset = topology->handoff_offsets[task];
         offset < topology->handoff_offsets[task + 1U]; ++offset) {
        const uint32_t source = topology->handoff_source_aliases[offset];
        const uint32_t destination = topology->handoff_destination_aliases[offset];
        if (build->program->alias_size_bytes[destination] == 0U) {
            continue;
        }
        if (workspace->active_alias_leases[source] == NO_LEASE ||
            workspace->active_alias_leases[destination] != NO_LEASE) {
            return -1;
        }
        workspace->active_alias_leases[destination] =
            workspace->active_alias_leases[source];
        workspace->lease_aliases[workspace->active_alias_leases[destination]] =
            destination;
        workspace->active_alias_leases[source] = NO_LEASE;
        workspace->handoff_sources[source] = 1U;
    }
    return 0;
}

/* A replaced object's old memory retires when the task completes, and the
 * alias starts naming the new allocation instead. */
static int apply_replacements(
    OperationBuild *build, uint32_t task, uint64_t *task_dependency
) {
    const ShadowSpillAdmissionFacts *topology = build->topology;
    ShadowSpillCandidateAdmissionWorkspace *workspace = build->workspace;
    for (uint32_t offset = topology->replacement_offsets[task];
         offset < topology->replacement_offsets[task + 1U]; ++offset) {
        const uint32_t alias = topology->replacement_aliases[offset];
        if (build->program->alias_size_bytes[alias] == 0U) {
            continue;
        }
        const uint64_t dependency =
            task_completion_dependency(build->tally, task_dependency);
        if (workspace->active_alias_leases[alias] == NO_LEASE ||
            workspace->new_alias_leases[alias] == NO_LEASE ||
            dependency == SHADOWSPILL_ADMISSION_REPLAY_NO_ID ||
            begin_retirement(
                build->tally,
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
    return 0;
}

/* An object the task produced fresh becomes live under the allocation it was
 * written into. */
static int adopt_fresh_outputs(OperationBuild *build, uint32_t task) {
    const ShadowSpillAdmissionFacts *topology = build->topology;
    ShadowSpillCandidateAdmissionWorkspace *workspace = build->workspace;
    for (uint32_t offset = topology->fresh_output_offsets[task];
         offset < topology->fresh_output_offsets[task + 1U]; ++offset) {
        const uint32_t alias = topology->fresh_output_aliases[offset];
        if (build->program->alias_size_bytes[alias] == 0U) {
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
    return 0;
}

/* A release gives the memory back once the task that was using it finishes.
 * An alias handed off earlier in this task no longer owns anything to
 * release. */
static int apply_release_action(
    OperationBuild *build,
    uint32_t task,
    uint32_t action,
    uint32_t alias,
    uint64_t *task_dependency
) {
    ShadowSpillCandidateAdmissionWorkspace *workspace = build->workspace;
    if (build->program->alias_size_bytes[alias] == 0U ||
        workspace->handoff_sources[alias] != 0U) {
        return 0;
    }
    const uint64_t lease = workspace->active_alias_leases[alias];
    const uint64_t dependency =
        task_completion_dependency(build->tally, task_dependency);
    if (lease == NO_LEASE ||
        dependency == SHADOWSPILL_ADMISSION_REPLAY_NO_ID ||
        begin_retirement(
            build->tally,
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
    return 0;
}

/* An eviction has to finish copying before the memory can be reused, so it
 * takes a dependency of its own rather than the task's. */
static int apply_offload_action(
    OperationBuild *build, uint32_t action, uint32_t alias
) {
    ShadowSpillCandidateAdmissionWorkspace *workspace = build->workspace;
    if (build->program->alias_size_bytes[alias] == 0U) {
        return 0;
    }
    const uint64_t lease = workspace->active_alias_leases[alias];
    if (lease == NO_LEASE ||
        build->tally->dependency_count >= workspace->dependency_capacity) {
        return -1;
    }
    const uint64_t dependency = build->tally->dependency_count++;
    if (begin_retirement(
            build->tally,
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
    build->tally->evict_bytes += build->program->alias_size_bytes[alias];
    workspace->active_alias_leases[alias] = NO_LEASE;
    return 0;
}

/* A fetch needs somewhere to land before the copy starts. */
static int apply_prefetch_action(
    OperationBuild *build, uint32_t action, uint32_t alias
) {
    if (build->program->alias_size_bytes[alias] == 0U) {
        return 0;
    }
    if (build->workspace->active_alias_leases[alias] != NO_LEASE ||
        acquire_lease(
            build->tally,
            build->program->alias_size_bytes[alias],
            build->topology->minimum_alignment,
            SHADOWSPILL_ADMISSION_BOUNDARY_ACTION_TRIGGER,
            action,
            alias,
            SHADOWSPILL_ADMISSION_PURPOSE_FETCH_DESTINATION,
            &build->workspace->active_alias_leases[alias]
        ) != 0) {
        return -1;
    }
    build->tally->fetch_bytes += build->program->alias_size_bytes[alias];
    return 0;
}

/* Every schedule action triggering at this task. */
static int apply_task_actions(
    OperationBuild *build, uint32_t task, uint64_t *task_dependency
) {
    const ShadowSpillIndexedSchedule *schedule = build->schedule;
    while (build->action_cursor < schedule->action_count &&
           schedule->action_trigger_tasks[build->action_cursor] == task) {
        const uint32_t action = build->action_cursor++;
        const uint32_t alias = schedule->action_aliases[action];
        if (alias >= build->program->alias_count) {
            return -1;
        }
        int applied = -1;
        switch (schedule->action_kinds[action]) {
        case SHADOWSPILL_MEMORY_RELEASE:
            applied =
                apply_release_action(build, task, action, alias, task_dependency);
            break;
        case SHADOWSPILL_MEMORY_OFFLOAD:
            applied = apply_offload_action(build, action, alias);
            break;
        case SHADOWSPILL_MEMORY_PREFETCH:
            applied = apply_prefetch_action(build, action, alias);
            break;
        default:
            return -1;
        }
        if (applied != 0) {
            return -1;
        }
    }
    return 0;
}

/* Handoff marks only suppress releases within the task that made them. */
static void clear_handoff_marks(OperationBuild *build, uint32_t task) {
    const ShadowSpillAdmissionFacts *topology = build->topology;
    for (uint32_t offset = topology->handoff_offsets[task];
         offset < topology->handoff_offsets[task + 1U]; ++offset) {
        build->workspace
            ->handoff_sources[topology->handoff_source_aliases[offset]] = 0U;
    }
}

/* Everything one task does to memory, in the order it does it. */
static int build_task(OperationBuild *build, uint32_t task) {
    uint64_t task_dependency = SHADOWSPILL_ADMISSION_REPLAY_NO_ID;
    if (require_task_aliases(build, task) != 0 ||
        open_task_allocations(build, task) != 0 ||
        retire_task_allocations(build, task, &task_dependency) != 0 ||
        adopt_handoffs(build, task) != 0 ||
        apply_replacements(build, task, &task_dependency) != 0 ||
        adopt_fresh_outputs(build, task) != 0 ||
        apply_task_actions(build, task, &task_dependency) != 0) {
        return -1;
    }
    clear_handoff_marks(build, task);
    return 0;
}

/* A retirement that began during the step completes at the end of it. */
static int flush_pending_retirements(OperationBuild *build) {
    for (uint64_t index = 0U; index < build->tally->pending_count; ++index) {
        const ShadowSpillPendingRetirement pending =
            build->workspace->pending_retirements[index];
        if (append_operation(
                build->tally,
                pending.lease_id,
                SHADOWSPILL_ADMISSION_REPLAY_COMPLETE_RETIREMENT,
                0U,
                0U,
                pending.dependency_id,
                0U,
                pending.completion_boundary,
                pending.completion_index,
                pending.completion_purpose,
                SHADOWSPILL_PLANNER_NO_INDEX
            ) != 0) {
            return -1;
        }
    }
    return 0;
}

/* Whatever the step promised to end holding has to still be held. */
static int check_final_residency(OperationBuild *build) {
    const ShadowSpillIndexedSchedule *schedule = build->schedule;
    for (uint32_t index = 0U; index < schedule->final_count; ++index) {
        const uint32_t alias = schedule->final_aliases[index];
        if (alias >= build->program->alias_count ||
            (schedule->final_locations[index] == SHADOWSPILL_MEMORY_DEVICE &&
             build->program->alias_size_bytes[alias] != 0U &&
             build->workspace->active_alias_leases[alias] == NO_LEASE)) {
            return -1;
        }
    }
    return 0;
}

int shadowspill_admission_build_operations(
    const ShadowSpillPressureFitProblem *problem,
    const ShadowSpillIndexedSchedule *schedule,
    ShadowSpillCandidateAdmissionWorkspace *workspace,
    OperationTally *tally
) {
    memset(tally, 0, sizeof(*tally));
    tally->workspace = workspace;
    OperationBuild build = {
        .program = problem->simulation,
        .topology = problem->admission,
        .schedule = schedule,
        .workspace = workspace,
        .tally = tally,
    };

    reset_alias_tracking(&build);
    if (admit_initial_residency(&build) != 0) {
        return -1;
    }
    for (uint32_t task = 0U; task < build.program->task_count; ++task) {
        if (build_task(&build, task) != 0) {
            return -1;
        }
    }
    /* Every action must have been claimed by the task it triggers on. */
    if (build.action_cursor != schedule->action_count ||
        flush_pending_retirements(&build) != 0 ||
        check_final_residency(&build) != 0) {
        return -1;
    }
    return 0;
}

/* ---------------------------------------------------------------- public */

/* A problem carrying only what operation building reads: the resolved task
 * set and the physical ownership facts. Residency and seed tally belong to
 * candidate search, not here. */
static ShadowSpillPressureFitProblem operations_problem(
    const ShadowSpillSimulationProgram *simulation,
    const ShadowSpillAdmissionFacts *admission
) {
    return (ShadowSpillPressureFitProblem){
        .abi_version = SHADOWSPILL_ABI_VERSION,
        .simulation = simulation,
        .admission = admission,
    };
}

ShadowSpillStatus shadowspill_admission_operation_bounds(
    const ShadowSpillSimulationProgram *simulation,
    const ShadowSpillAdmissionFacts *admission,
    const ShadowSpillIndexedSchedule *schedule,
    uint64_t *operation_capacity,
    uint64_t *lease_capacity
) {
    if (schedule == NULL || operation_capacity == NULL ||
        lease_capacity == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    const ShadowSpillPressureFitProblem problem =
        operations_problem(simulation, admission);
    if (!shadowspill_admission_facts_valid(&problem)) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    if (shadowspill_admission_counts(
            &problem, schedule, lease_capacity, operation_capacity) != 0) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    return SHADOWSPILL_STATUS_OK;
}

ShadowSpillStatus shadowspill_build_admission_operations(
    const ShadowSpillSimulationProgram *simulation,
    const ShadowSpillAdmissionFacts *admission,
    const ShadowSpillIndexedSchedule *schedule,
    ShadowSpillAdmissionOperations *result
) {
    if (schedule == NULL || result == NULL || result->lease_ids == NULL ||
        result->dependency_ids == NULL || result->bytes == NULL || result->alignments == NULL ||
        result->kinds == NULL || result->purposes == NULL ||
        result->boundaries == NULL || result->indices == NULL ||
        result->allocation_offsets == NULL || result->lease_aliases == NULL ||
        result->lease_starts == NULL || result->lease_retires == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    const ShadowSpillPressureFitProblem problem =
        operations_problem(simulation, admission);
    if (!shadowspill_admission_facts_valid(&problem)) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }

    ShadowSpillCandidateAdmissionWorkspace workspace = {0};
    if (shadowspill_candidate_admission_workspace_create(&problem, &workspace)
            != 0 ||
        shadowspill_admission_reserve_buffers(&problem, schedule, &workspace)
            != 0) {
        shadowspill_candidate_admission_workspace_destroy(&workspace);
        return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    }
    if (workspace.operation_capacity > result->operation_capacity ||
        workspace.lease_capacity > result->lease_capacity) {
        shadowspill_candidate_admission_workspace_destroy(&workspace);
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }

    OperationTally tally;
    if (shadowspill_admission_build_operations(
            &problem, schedule, &workspace, &tally) != 0) {
        shadowspill_candidate_admission_workspace_destroy(&workspace);
        return SHADOWSPILL_STATUS_PLANNER_INTERNAL_ERROR;
    }

    for (uint64_t index = 0U; index < tally.operation_count; ++index) {
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
    for (uint64_t lease = 0U; lease < tally.lease_count; ++lease) {
        result->lease_aliases[lease] = workspace.lease_aliases[lease];
        result->lease_starts[lease] = workspace.lease_start_operations[lease];
        result->lease_retires[lease] = workspace.lease_retire_operations[lease];
    }
    result->operation_count = tally.operation_count;
    result->lease_count = tally.lease_count;
    result->dependency_count = tally.dependency_count;
    result->fetch_bytes = tally.fetch_bytes;
    result->evict_bytes = tally.evict_bytes;
    shadowspill_candidate_admission_workspace_destroy(&workspace);
    return SHADOWSPILL_STATUS_OK;
}
