#include <stdint.h>

#include "internal.h"

static int validate_result_buffers(
    const ShadowSpillSimulationProgram *program,
    const ShadowSpillSimulationResult *result
) {
    return result->task_interval_capacity >= program->task_count &&
        result->transfer_interval_capacity >= program->action_count &&
        result->device_peak_capacity >= program->device_count &&
        (program->task_count == 0U || result->task_intervals != NULL) &&
        (program->action_count == 0U || result->transfer_intervals != NULL) &&
        result->device_peaks != NULL;
}

static ShadowSpillSimulationStatus finish_failure(
    ShadowSpillSimulationWork *work,
    const ShadowSpillSimulationResult *result
) {
    ShadowSpillSimulationStatus status =
        (ShadowSpillSimulationStatus)result->status;
    shadowspill_free_work(work);
    return status;
}

ShadowSpillSimulationStatus shadowspill_simulate(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationResult *result
) {
    if (result == NULL) {
        return SHADOWSPILL_SIMULATION_INVALID_ARGUMENT;
    }
    shadowspill_initialize_result(result);
    if (!shadowspill_validate_program(program) ||
        !validate_result_buffers(program, result)) {
        result->status = SHADOWSPILL_SIMULATION_INVALID_ARGUMENT;
        return SHADOWSPILL_SIMULATION_INVALID_ARGUMENT;
    }
    ShadowSpillSimulationWork work = {0};
    if (!shadowspill_allocate_work(program, &work)) {
        shadowspill_free_work(&work);
        result->status = SHADOWSPILL_SIMULATION_ALLOCATION_FAILURE;
        return SHADOWSPILL_SIMULATION_ALLOCATION_FAILURE;
    }
    if (!shadowspill_initialize_memory(program, &work, result)) {
        return finish_failure(&work, result);
    }
    while (shadowspill_has_pending_work(program, &work)) {
        int changed = 1;
        while (changed != 0) {
            changed = shadowspill_try_start_transfers(program, &work);
            changed |= shadowspill_try_launch_tasks(program, &work);
        }
        uint64_t next = shadowspill_next_event_time(program, &work);
        if (next == UINT64_MAX) {
            shadowspill_report_deadlock(program, &work, result);
            return finish_failure(&work, result);
        }
        work.now_ns = next;
        if (!shadowspill_complete_events(program, &work, result)) {
            return finish_failure(&work, result);
        }
    }
    if (!shadowspill_check_final_residency(program, &work, result)) {
        return finish_failure(&work, result);
    }
    result->makespan_ns = work.now_ns;
    result->host_peak_bytes = work.host_peak_bytes;
    for (uint32_t device = 0; device < program->device_count; ++device) {
        result->device_peaks[device] = (ShadowSpillDevicePeak){
            .object_bytes = work.device_object_peaks[device],
            .workspace_bytes = work.device_workspace_peaks[device],
            .total_bytes = work.device_total_peaks[device],
        };
    }
    result->status = SHADOWSPILL_SIMULATION_OK;
    shadowspill_free_work(&work);
    return SHADOWSPILL_SIMULATION_OK;
}
