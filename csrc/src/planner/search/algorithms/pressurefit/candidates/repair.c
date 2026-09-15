/* What a refused plan asks for: analytic pressure where the simulation
 * stalled, and fetch moves where the admission could not place it. */
#include "internal.h"

void shadowspill_candidate_copy_simulation_error(
    ShadowSpillPressureFitCandidateDiagnostic *diagnostic,
    const ShadowSpillSimulationResult *simulation
) {
    diagnostic->simulation_status = simulation->status;
    diagnostic->error_task = simulation->error_task;
    diagnostic->error_alias = simulation->error_alias;
    diagnostic->error_device = simulation->error_device;
    diagnostic->error_location = simulation->error_location;
    diagnostic->error_time_ns = simulation->error_time_ns;
    diagnostic->error_capacity_bytes = simulation->error_capacity_bytes;
    diagnostic->error_used_bytes = simulation->error_used_bytes;
    diagnostic->error_requested_bytes = simulation->error_requested_bytes;
}

void shadowspill_candidate_copy_analytic_error(
    ShadowSpillPressureFitCandidateDiagnostic *diagnostic,
    const ShadowSpillPressureFitResidencyResult *residency
) {
    diagnostic->status = SHADOWSPILL_PRESSUREFIT_CANDIDATE_ANALYTIC_INFEASIBLE;
    diagnostic->error_device = residency->error_device;
    diagnostic->error_boundary = residency->error_boundary;
    diagnostic->error_required_bytes = residency->required_bytes;
    diagnostic->error_capacity_bytes = residency->capacity_bytes;
}

/*
 * Make room at the boundary where the plan came up short.
 *
 * The ask is the shortfall the simulator measured. A failure that repeats at
 * the same task and the same moment after that ask was met says the analytic
 * room did not become simulated room -- copies still in flight hold it, or
 * the emitter packed the freed bytes again -- so a repeat asks for twice
 * what the last round asked, up to the task's whole request: the same few
 * bytes again would be the same plan again, and a candidate can repeat that
 * round without end. `escalation` is how many times in a row the failure has
 * repeated; `asked_beyond_shortfall`
 * reports what the ask added over the shortfall, so a reduction that cannot
 * meet the larger ask can take exactly that back.
 */
int shadowspill_candidate_add_repair_pressure(
    const ShadowSpillPressureFitProblem *problem,
    CandidateWorkspace *workspace,
    const ShadowSpillSimulationResult *failure,
    uint32_t escalation,
    uint64_t *asked_beyond_shortfall
) {
    *asked_beyond_shortfall = 0U;
    if (failure->status != SHADOWSPILL_STATUS_INITIAL_DEVICE_CAPACITY &&
        failure->status != SHADOWSPILL_STATUS_FETCH_DEVICE_CAPACITY &&
        failure->status != SHADOWSPILL_STATUS_TASK_DEVICE_CAPACITY) {
        return 0;
    }
    if (failure->error_device == SHADOWSPILL_SIMULATOR_NO_INDEX ||
        failure->error_device >= problem->residency->device_count) {
        return 0;
    }
    int32_t boundary = -1;
    if (failure->status == SHADOWSPILL_STATUS_TASK_DEVICE_CAPACITY) {
        if (failure->error_task == SHADOWSPILL_SIMULATOR_NO_INDEX) {
            return 0;
        }
        boundary = (int32_t)failure->error_task - 1;
    } else if (failure->status ==
               SHADOWSPILL_STATUS_FETCH_DEVICE_CAPACITY) {
        if (failure->error_task == SHADOWSPILL_SIMULATOR_NO_INDEX) {
            return 0;
        }
        boundary = (int32_t)failure->error_task;
    }
    uint64_t capacity = failure->error_capacity_bytes != 0U
        ? failure->error_capacity_bytes
        : shadowspill_boundary_capacity(
            problem->residency,
            failure->error_device,
            (uint32_t)(boundary + 1)
        );
    uint64_t total = failure->error_used_bytes;
    if (failure->error_requested_bytes > UINT64_MAX - total) {
        total = UINT64_MAX;
    } else {
        total += failure->error_requested_bytes;
    }
    uint64_t excess = total > capacity ? total - capacity : 1U;
    const uint64_t shortfall = excess;
    const uint64_t ceiling = failure->error_requested_bytes != 0U &&
            failure->error_requested_bytes < capacity
        ? failure->error_requested_bytes
        : capacity;
    for (uint32_t step = 0U; step < escalation && excess < ceiling; ++step) {
        excess = excess > UINT64_MAX / 2U ? UINT64_MAX : excess * 2U;
    }
    if (excess > ceiling && ceiling > shortfall) {
        excess = ceiling;
    }
    *asked_beyond_shortfall = excess > shortfall ? excess - shortfall : 0U;
    uint32_t index = (uint32_t)(boundary + 1);
    uint64_t position =
        (uint64_t)failure->error_device * problem->residency->boundary_count +
        index;
    if (workspace->extra_pressure[position] > UINT64_MAX - excess) {
        workspace->extra_pressure[position] = UINT64_MAX;
    } else {
        workspace->extra_pressure[position] += excess;
    }
    workspace->last_pressure_position = position;
    return 1;
}

int shadowspill_candidate_simulation_failure_may_be_repairable(
    ShadowSpillStatus status
) {
    return status == SHADOWSPILL_STATUS_INITIAL_DEVICE_CAPACITY ||
        status == SHADOWSPILL_STATUS_FETCH_DEVICE_CAPACITY ||
        status == SHADOWSPILL_STATUS_TASK_DEVICE_CAPACITY;
}

static int admission_failure_boundary(
    const ShadowSpillPressureFitProblem *problem,
    const ShadowSpillIndexedSchedule *schedule,
    ShadowSpillAdmissionAnnotation annotation,
    uint32_t *task,
    uint32_t *pressure_index,
    uint32_t *alias
) {
    const uint32_t no_index = SHADOWSPILL_SIMULATOR_NO_INDEX;
    *task = no_index;
    *pressure_index = no_index;
    *alias = no_index;
    switch ((ShadowSpillAdmissionBoundaryKind)annotation.boundary) {
        case SHADOWSPILL_ADMISSION_BOUNDARY_INITIAL:
            *pressure_index = 0U;
            return 1;
        case SHADOWSPILL_ADMISSION_BOUNDARY_TASK_START:
            if (annotation.index >= problem->context.simulation->task_count) {
                return -1;
            }
            *task = annotation.index;
            *pressure_index = annotation.index;
            return 1;
        case SHADOWSPILL_ADMISSION_BOUNDARY_TASK_COMPLETION:
            if (annotation.index >= problem->context.simulation->task_count) {
                return -1;
            }
            *task = annotation.index;
            *pressure_index = annotation.index + 1U;
            return 1;
        case SHADOWSPILL_ADMISSION_BOUNDARY_ACTION_TRIGGER:
        case SHADOWSPILL_ADMISSION_BOUNDARY_ACTION_COMPLETION:
            if (annotation.index >= schedule->action_count) {
                return -1;
            }
            *task = schedule->action_trigger_tasks[annotation.index];
            *alias = schedule->action_aliases[annotation.index];
            if (*task >= problem->context.simulation->task_count) {
                return -1;
            }
            *pressure_index = *task + 1U;
            return 1;
        default:
            return -1;
    }
}

int shadowspill_candidate_delay_admission_fetch(
    const ShadowSpillScheduleFacts *facts,
    const ShadowSpillAdmissionReplayResult *failure,
    ShadowSpillAdmissionAnnotation annotation,
    ShadowSpillScheduleStorage *schedule,
    ShadowSpillFetchTriggerConstraint *constraint
) {
    ShadowSpillSimulationResult projected = {
        .error_alias = SHADOWSPILL_SIMULATOR_NO_INDEX,
        .error_device = 0U,
        .error_capacity_bytes = facts->problem->context.admission->pool_capacity_bytes,
        .error_used_bytes =
            facts->problem->context.admission->pool_capacity_bytes -
            failure->error_free_bytes,
        .error_requested_bytes = failure->error_requested_bytes,
    };
    if (annotation.boundary == SHADOWSPILL_ADMISSION_BOUNDARY_TASK_START) {
        if (annotation.index >= facts->task_count) {
            return -1;
        }
        projected.status = SHADOWSPILL_STATUS_TASK_DEVICE_CAPACITY;
        projected.error_task = annotation.index;
    } else if (
        annotation.boundary == SHADOWSPILL_ADMISSION_BOUNDARY_ACTION_TRIGGER &&
        annotation.index < schedule->value.action_count &&
        schedule->value.action_kinds[annotation.index] ==
            SHADOWSPILL_MEMORY_FETCH
    ) {
        projected.status = SHADOWSPILL_STATUS_FETCH_DEVICE_CAPACITY;
        projected.error_task =
            schedule->value.action_trigger_tasks[annotation.index];
        projected.error_alias =
            schedule->value.action_aliases[annotation.index];
    } else {
        return 0;
    }
    return shadowspill_delay_indexed_fetch(
        facts, &projected, schedule, constraint
    );
}

int shadowspill_candidate_advance_admission_fetch(
    const ShadowSpillScheduleFacts *facts,
    ShadowSpillAdmissionAnnotation annotation,
    ShadowSpillScheduleStorage *schedule,
    ShadowSpillFetchTriggerConstraint *constraint
) {
    if (annotation.boundary !=
            SHADOWSPILL_ADMISSION_BOUNDARY_ACTION_TRIGGER ||
        annotation.index >= schedule->value.action_count ||
        schedule->value.action_kinds[annotation.index] !=
            SHADOWSPILL_MEMORY_FETCH) {
        return 0;
    }
    return shadowspill_advance_indexed_fetch_to_release(
        facts, annotation.index, schedule, constraint
    );
}

int shadowspill_candidate_add_admission_repair_pressure(
    const ShadowSpillPressureFitProblem *problem,
    CandidateWorkspace *workspace,
    const ShadowSpillPressureFitResidencyOptions *options,
    const ShadowSpillAdmissionReplayResult *failure,
    ShadowSpillAdmissionAnnotation annotation,
    const ShadowSpillIndexedSchedule *schedule
) {
    uint32_t task = SHADOWSPILL_SIMULATOR_NO_INDEX;
    uint32_t pressure_index = SHADOWSPILL_SIMULATOR_NO_INDEX;
    uint32_t alias = SHADOWSPILL_SIMULATOR_NO_INDEX;
    const int boundary = admission_failure_boundary(
        problem,
        schedule,
        annotation,
        &task,
        &pressure_index,
        &alias
    );
    (void)task;
    (void)alias;
    if (boundary <= 0 ||
        pressure_index >= problem->residency->boundary_count) {
        return boundary;
    }
    uint64_t required_reduction = failure->error_requested_bytes >
            failure->error_largest_free_range_bytes
        ? failure->error_requested_bytes -
            failure->error_largest_free_range_bytes
        : 1U;
    const uint64_t position = pressure_index;
    uint64_t resident_pressure = 0U;
    if (shadowspill_residency_pressure_at(
            problem->residency,
            options,
            workspace->resident,
            workspace->breaks,
            0U,
            pressure_index,
            workspace->residency_workspace,
            &resident_pressure
        ) != 0) {
        return -1;
    }
    uint64_t current_pressure = resident_pressure;
    if (current_pressure >
        UINT64_MAX - workspace->extra_pressure[position]) {
        current_pressure = UINT64_MAX;
    } else {
        current_pressure += workspace->extra_pressure[position];
    }
    const uint64_t capacity = shadowspill_boundary_capacity(
        problem->residency,
        0U,
        pressure_index
    );
    const uint64_t unused_capacity = current_pressure < capacity
        ? capacity - current_pressure
        : 0U;
    uint64_t pressure_increment = required_reduction;
    if (pressure_increment > UINT64_MAX - unused_capacity) {
        pressure_increment = UINT64_MAX;
    } else {
        pressure_increment += unused_capacity;
    }
    if (workspace->extra_pressure[position] >
        UINT64_MAX - pressure_increment) {
        workspace->extra_pressure[position] = UINT64_MAX;
    } else {
        workspace->extra_pressure[position] += pressure_increment;
    }
    return 1;
}


void shadowspill_candidate_copy_admission_error(
    const ShadowSpillPressureFitProblem *problem,
    const ShadowSpillIndexedSchedule *schedule,
    const ShadowSpillAdmissionReplayResult *failure,
    ShadowSpillAdmissionAnnotation annotation,
    ShadowSpillPressureFitCandidateDiagnostic *diagnostic
) {
    uint32_t task = SHADOWSPILL_SIMULATOR_NO_INDEX;
    uint32_t pressure_index = SHADOWSPILL_SIMULATOR_NO_INDEX;
    uint32_t alias = SHADOWSPILL_SIMULATOR_NO_INDEX;
    (void)admission_failure_boundary(
        problem,
        schedule,
        annotation,
        &task,
        &pressure_index,
        &alias
    );
    diagnostic->status = SHADOWSPILL_PRESSUREFIT_CANDIDATE_ADMISSION_INFEASIBLE;
    diagnostic->error_task = task;
    diagnostic->error_alias = alias;
    diagnostic->error_device = 0U;
    diagnostic->error_boundary = pressure_index ==
            SHADOWSPILL_SIMULATOR_NO_INDEX
        ? INT32_MIN
        : (int32_t)pressure_index - 1;
    diagnostic->error_capacity_bytes = problem->context.admission->pool_capacity_bytes;
    diagnostic->error_used_bytes =
        problem->context.admission->pool_capacity_bytes - failure->error_free_bytes;
    diagnostic->error_requested_bytes = failure->error_requested_bytes;
    diagnostic->error_required_bytes = failure->error_requested_bytes >
            failure->error_largest_free_range_bytes
        ? failure->error_requested_bytes -
            failure->error_largest_free_range_bytes
        : 0U;
}

int shadowspill_candidate_reduce_repaired_candidate(
    const ShadowSpillPressureFitProblem *problem,
    CandidateWorkspace *workspace,
    const ShadowSpillPressureFitResidencyOptions *options,
    uint8_t strategy,
    ShadowSpillPressureFitCandidateDiagnostic *diagnostic
) {
    ShadowSpillPressureFitResidencyResult residency;
    const ShadowSpillStatus status = shadowspill_candidate_reduce_residency(
        problem,
        workspace,
        options,
        strategy,
        workspace->resident,
        workspace->breaks,
        &residency
    );
    if (status == SHADOWSPILL_STATUS_ANALYTIC_INFEASIBLE) {
        shadowspill_candidate_copy_analytic_error(diagnostic, &residency);
        return 0;
    }
    return status == SHADOWSPILL_STATUS_OK ? 1 : -1;
}

void shadowspill_candidate_initialize_diagnostic(
    ShadowSpillPressureFitCandidateDiagnostic *diagnostic,
    uint8_t strategy,
    uint8_t rule,
    uint8_t coalesced
) {
    memset(diagnostic, 0, sizeof(*diagnostic));
    diagnostic->repairs_at_best = UINT32_MAX;
    diagnostic->status = SHADOWSPILL_PRESSUREFIT_CANDIDATE_INTERNAL_ERROR;
    diagnostic->residency_strategy = strategy;
    diagnostic->fetch_rule = rule;
    diagnostic->coalesced = coalesced;
    diagnostic->simulation_status = SHADOWSPILL_STATUS_OK;
    diagnostic->error_task = SHADOWSPILL_SIMULATOR_NO_INDEX;
    diagnostic->error_alias = SHADOWSPILL_SIMULATOR_NO_INDEX;
    diagnostic->error_device = SHADOWSPILL_SIMULATOR_NO_INDEX;
    diagnostic->error_boundary = INT32_MIN;
}

/*
 * Whether this candidate gets another reduction.
 *
 * Effort is the only thing that stops it. A plan that does not fit waits
 * rather than failing, so a reduction relieves the waiting as often as it
 * adds to it: a candidate behind the plan in hand is exactly the one with
 * something to gain, and cutting it off there would abandon the plans most
 * worth finding.
 *
 * A candidate that runs out reports `SHADOWSPILL_PRESSUREFIT_CANDIDATE_REPAIR_EXHAUSTED`,
 * which says the effort ran out, never that there is no plan.
 */
int shadowspill_candidate_may_repair_again(
    const ShadowSpillPressureFitOptions *candidate_options,
    const ShadowSpillPressureFitCandidateDiagnostic *diagnostic
) {
    return shadowspill_candidate_repair_total(&diagnostic->repairs) <
        candidate_options->max_repair_attempts;
}

/*
 * One candidate's search.
 *
 * A candidate starts from its strategy's base residency and repeats a fixed
 * cycle: emit a schedule, simulate it, name it, measure whether its layout
 * fits, then decide whether to keep looking. It leaves the cycle with an
 * answer, or without one when it runs out of ways to improve.
 *
 * `evaluate_candidate` is that cycle and nothing else. Every step of it is a
 * stage below, the state they share is `CandidateSearch`, and the timing
 * sections are opened and closed around the stage calls so that what a
 * section covers is exactly what its name says.
 */

/* What a stage tells the cycle to do next. */
