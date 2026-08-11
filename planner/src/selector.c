#include "internal.h"

#include <stdint.h>
#include <stdlib.h>

typedef struct SimulationBuffers {
    ShadowSpillTaskInterval *tasks;
    ShadowSpillTransferInterval *transfers;
    ShadowSpillDevicePeak *peaks;
} SimulationBuffers;

static int allocate_buffers(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationResult *simulation,
    SimulationBuffers *buffers
) {
    uint32_t task_count = program->task_count == 0U ? 1U : program->task_count;
    uint32_t transfer_count =
        program->action_count == 0U ? 1U : program->action_count;
    uint32_t device_count =
        program->device_count == 0U ? 1U : program->device_count;
    buffers->tasks = calloc(task_count, sizeof(*buffers->tasks));
    buffers->transfers = calloc(transfer_count, sizeof(*buffers->transfers));
    buffers->peaks = calloc(device_count, sizeof(*buffers->peaks));
    if (buffers->tasks == NULL || buffers->transfers == NULL ||
        buffers->peaks == NULL) {
        return -1;
    }
    simulation->task_intervals = buffers->tasks;
    simulation->task_interval_capacity = task_count;
    simulation->transfer_intervals = buffers->transfers;
    simulation->transfer_interval_capacity = transfer_count;
    simulation->device_peaks = buffers->peaks;
    simulation->device_peak_capacity = device_count;
    return 0;
}

static void free_buffers(SimulationBuffers *buffers) {
    free(buffers->tasks);
    free(buffers->transfers);
    free(buffers->peaks);
}

ShadowSpillPlannerStatus shadowspill_select_plan(
    const ShadowSpillPlanCandidate *candidates,
    uint32_t candidate_count,
    ShadowSpillPlanSelectionResult *result
) {
    if (result == NULL) {
        return SHADOWSPILL_PLANNER_INVALID_ARGUMENT;
    }
    shadowspill_planner_reset_result(result);
    if (candidates == NULL || candidate_count == 0U ||
        (result->candidate_results != NULL &&
         result->candidate_result_capacity < candidate_count)) {
        return SHADOWSPILL_PLANNER_INVALID_ARGUMENT;
    }

    for (uint32_t index = 0; index < candidate_count; ++index) {
        const ShadowSpillSimulationProgram *program = candidates[index].program;
        if (program == NULL) {
            result->status = SHADOWSPILL_PLANNER_INVALID_ARGUMENT;
            return SHADOWSPILL_PLANNER_INVALID_ARGUMENT;
        }
        ShadowSpillSimulationResult simulation = {0};
        SimulationBuffers buffers = {0};
        if (allocate_buffers(program, &simulation, &buffers) != 0) {
            free_buffers(&buffers);
            result->status = SHADOWSPILL_PLANNER_ALLOCATION_FAILURE;
            return SHADOWSPILL_PLANNER_ALLOCATION_FAILURE;
        }
        ShadowSpillSimulationStatus simulation_status =
            shadowspill_simulate(program, &simulation);
        free_buffers(&buffers);

        if (result->candidate_results != NULL) {
            result->candidate_results[index] = (ShadowSpillCandidateResult){
                .simulation_status = (uint32_t)simulation_status,
                .valid = (uint8_t)(simulation_status == SHADOWSPILL_SIMULATION_OK),
                .makespan_ns = simulation.makespan_ns,
            };
        }
        result->candidate_result_count = index + 1U;
        if (simulation_status != SHADOWSPILL_SIMULATION_OK) {
            if (result->first_failure_index == SHADOWSPILL_PLANNER_NO_INDEX) {
                result->first_failure_index = index;
                result->first_failure_status = (uint32_t)simulation_status;
            }
            continue;
        }
        ++result->valid_candidate_count;
        if (result->selected_index == SHADOWSPILL_PLANNER_NO_INDEX ||
            simulation.makespan_ns < result->selected_makespan_ns) {
            result->selected_index = index;
            result->selected_candidate_id = candidates[index].candidate_id;
            result->selected_selection_id = candidates[index].selection_id;
            result->selected_makespan_ns = simulation.makespan_ns;
        }
    }

    if (result->selected_index == SHADOWSPILL_PLANNER_NO_INDEX) {
        result->status = SHADOWSPILL_PLANNER_NO_FEASIBLE_CANDIDATE;
        return SHADOWSPILL_PLANNER_NO_FEASIBLE_CANDIDATE;
    }
    result->status = SHADOWSPILL_PLANNER_OK;
    return SHADOWSPILL_PLANNER_OK;
}
