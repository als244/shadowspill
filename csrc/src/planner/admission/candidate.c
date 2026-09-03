/* Replaying a schedule's operations through the production pool policy.
 *
 * Candidate admission asks whether a schedule survives the real allocator, and
 * projects the per-task and per-action physical deltas the simulator needs to
 * price it. The operation sequence it replays comes from `operations.c`.
 */


#include "internal.h"
#include "../../common/platform.h"
#include "../candidates_internal.h"

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>


static int add_delta(int64_t *target, int64_t delta) {
    if ((delta > 0 && *target > INT64_MAX - delta) ||
        (delta < 0 && *target < INT64_MIN - delta)) {
        return -1;
    }
    *target += delta;
    return 0;
}

static int project_result(
    const ShadowSpillPressureFitProblem *problem,
    const ShadowSpillIndexedSchedule *schedule,
    ShadowSpillCandidateAdmissionWorkspace *workspace,
    const OperationTally *tally,
    const ShadowSpillAdmissionReplayResult *result
) {
    const uint32_t tasks = problem->simulation->task_count;
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
    workspace->projected_reuse_count = 0U;
    for (uint64_t operation = 0U; operation < tally->operation_count; ++operation) {
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
            case SHADOWSPILL_ADMISSION_BOUNDARY_TASK_COMPLETION:
                if (annotation.index >= tasks || add_delta(
                        &workspace->task_start_deltas[annotation.index], delta
                    ) != 0) {
                    return -1;
                }
                if (workspace->task_start_deltas[annotation.index] >
                    workspace->task_completion_deltas[annotation.index]) {
                    workspace->task_completion_deltas[annotation.index] =
                        workspace->task_start_deltas[annotation.index];
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
    for (uint32_t task = 0U; task < tasks; ++task) {
        const int64_t total = workspace->task_start_deltas[task];
        const int64_t peak = workspace->task_completion_deltas[task];
        workspace->task_start_deltas[task] = peak;
        workspace->task_completion_deltas[task] = total - peak;
    }
    for (uint64_t index = 0U; index < result->dependency_result_count; ++index) {
        const ShadowSpillAdmissionReuseDependency dependency =
            workspace->dependencies[index];
        if (dependency.predecessor_lease_id >= tally->lease_count ||
            dependency.consumer_operation_index >= tally->operation_count) {
            return -1;
        }
        const uint32_t predecessor_action =
            workspace->predecessor_actions[dependency.predecessor_lease_id];
        const uint32_t predecessor_task =
            workspace->predecessor_tasks[dependency.predecessor_lease_id];
        const ShadowSpillAdmissionAnnotation successor =
            workspace->annotations[dependency.consumer_operation_index];
        if ((predecessor_action == UINT32_MAX) ==
            (predecessor_task == UINT32_MAX)) {
            return -1;
        }
        uint32_t successor_task = UINT32_MAX;
        uint32_t successor_action = UINT32_MAX;
        if (successor.boundary == SHADOWSPILL_ADMISSION_BOUNDARY_TASK_START ||
            successor.boundary ==
                SHADOWSPILL_ADMISSION_BOUNDARY_TASK_COMPLETION) {
            successor_task = successor.index;
        } else if (
            successor.boundary == SHADOWSPILL_ADMISSION_BOUNDARY_ACTION_TRIGGER ||
            successor.boundary ==
                SHADOWSPILL_ADMISSION_BOUNDARY_ACTION_COMPLETION
        ) {
            if (successor.index >= schedule->action_count) {
                return -1;
            }
            successor_action = successor.index;
            successor_task = schedule->action_trigger_tasks[successor.index];
        } else {
            return -1;
        }
        if (predecessor_task != UINT32_MAX) {
            if (predecessor_task >= tasks || successor_task < predecessor_task) {
                return -1;
            }
            continue;
        }
        if (predecessor_action >= schedule->action_count ||
            schedule->action_kinds[predecessor_action] !=
                SHADOWSPILL_MEMORY_EVICT ||
            workspace->projected_reuse_count >= workspace->action_capacity) {
            return -1;
        }
        const uint32_t projected = workspace->projected_reuse_count++;
        workspace->reuse_predecessor_actions[projected] = predecessor_action;
        workspace->reuse_successor_tasks[projected] =
            successor_action == UINT32_MAX
                ? successor_task : SHADOWSPILL_SIMULATOR_NO_INDEX;
        workspace->reuse_successor_actions[projected] = successor_action;
    }
    return 0;
}

ShadowSpillStatus shadowspill_admit_indexed_schedule(
    const ShadowSpillPressureFitProblem *problem,
    const ShadowSpillIndexedSchedule *schedule,
    ShadowSpillCandidateAdmissionWorkspace *workspace,
    ShadowSpillSimulationProgram *program,
    ShadowSpillAdmissionReplayResult *replay_result
) {
    if (!shadowspill_admission_facts_valid(problem) || schedule == NULL || workspace == NULL ||
        program == NULL || replay_result == NULL ||
        shadowspill_admission_reserve_buffers(problem, schedule, workspace) != 0) {
        return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    }
    const uint64_t started = shadowspill_monotonic_ns();
    OperationTally tally = {0};
    if (shadowspill_admission_build_operations(problem, schedule, workspace, &tally) != 0) {
        return SHADOWSPILL_STATUS_INVALID_OPERATIONS;
    }
    const ShadowSpillAdmissionReplayProgram replay_program = {
        .abi_version = SHADOWSPILL_ABI_VERSION,
        .capacity_bytes = problem->admission->pool_capacity_bytes,
        .minimum_alignment = problem->admission->minimum_alignment,
        .large_request_threshold_bytes = 0U,
        .lease_count = tally.lease_count,
        .dependency_count = tally.dependency_count,
        .operations = workspace->operations,
        .operation_count = tally.operation_count,
    };
    *replay_result = (ShadowSpillAdmissionReplayResult){
        .decisions = workspace->decisions,
        .decision_capacity = workspace->operation_capacity,
        .dependencies = workspace->dependencies,
        .dependency_capacity = workspace->reuse_dependency_capacity,
        .live_leases = workspace->live_leases,
        .live_lease_capacity = workspace->lease_capacity,
    };
    const ShadowSpillStatus status =
        shadowspill_admission_replay_run_reusing(
            &replay_program, replay_result, workspace->replay
        );
    ++workspace->calls;
    workspace->time_ns += shadowspill_monotonic_ns() - started;
    if (status != SHADOWSPILL_STATUS_OK) {
        return status;
    }
    if (project_result(
            problem, schedule, workspace, &tally, replay_result
        ) != 0) {
        return SHADOWSPILL_STATUS_INVALID_OPERATIONS;
    }
    shadowspill_bind_indexed_schedule(problem->simulation, schedule, program);
    workspace->physical_device = problem->simulation->devices[0];
    workspace->physical_device.capacity_bytes =
        problem->admission->pool_capacity_bytes;
    program->devices = &workspace->physical_device;
    program->use_admission_accounting = 1U;
    program->initial_physical_bytes = &workspace->initial_physical_bytes;
    program->task_start_physical_deltas = workspace->task_start_deltas;
    program->task_completion_physical_deltas = workspace->task_completion_deltas;
    program->action_trigger_physical_deltas = workspace->action_trigger_deltas;
    program->action_completion_physical_deltas =
        workspace->action_completion_deltas;
    program->reuse_dependency_count = workspace->projected_reuse_count;
    program->reuse_predecessor_actions = workspace->reuse_predecessor_actions;
    program->reuse_successor_tasks = workspace->reuse_successor_tasks;
    program->reuse_successor_actions = workspace->reuse_successor_actions;
    workspace->decision_digest = replay_result->decision_digest;
    workspace->peak_allocated_bytes = replay_result->peak_allocated_bytes;
    workspace->peak_reserved_bytes = replay_result->peak_reserved_bytes;
    workspace->peak_fragmentation_bytes = replay_result->peak_fragmentation_bytes;
    return status;
}

ShadowSpillStatus shadowspill_evaluate_schedule_admission(
    const ShadowSpillSimulationProgram *simulation,
    const ShadowSpillAdmissionFacts *admission,
    const ShadowSpillIndexedSchedule *schedule,
    ShadowSpillScheduleAdmissionResult *result
) {
    if (result == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
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
        .status = SHADOWSPILL_STATUS_INVALID_ARGUMENT,
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
    const ShadowSpillPressureFitProblem problem = {
        .abi_version = SHADOWSPILL_ABI_VERSION,
        .simulation = simulation,
        .admission = admission,
    };
    if (!shadowspill_admission_facts_valid(&problem) || schedule == NULL ||
        task_capacity < simulation->task_count ||
        action_capacity < schedule->action_count ||
        (simulation->task_count != 0U &&
         (task_start == NULL || task_completion == NULL)) ||
        (schedule->action_count != 0U &&
         (action_trigger == NULL || action_completion == NULL))) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    ShadowSpillCandidateAdmissionWorkspace workspace = {0};
    if (shadowspill_candidate_admission_workspace_create(
            &problem, &workspace
        ) != 0) {
        return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    }
    ShadowSpillSimulationProgram admitted_program = {0};
    ShadowSpillAdmissionReplayResult replay = {0};
    const ShadowSpillStatus status =
        shadowspill_admit_indexed_schedule(
            &problem,
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
    if (status == SHADOWSPILL_STATUS_OK) {
        if (workspace.projected_reuse_count > reuse_capacity) {
            shadowspill_candidate_admission_workspace_destroy(&workspace);
            return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
        }
        result->initial_physical_bytes = workspace.initial_physical_bytes;
        result->reuse_count = workspace.projected_reuse_count;
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
    if (status == SHADOWSPILL_STATUS_OK) {
        return SHADOWSPILL_STATUS_OK;
    }
    if (status == SHADOWSPILL_STATUS_REPLAY_INFEASIBLE) {
        return SHADOWSPILL_STATUS_NO_FEASIBLE_CANDIDATE;
    }
    return status == SHADOWSPILL_STATUS_INTERNAL_FAILURE
        ? SHADOWSPILL_STATUS_INTERNAL_FAILURE
        : SHADOWSPILL_STATUS_INVALID_ARGUMENT;
}
