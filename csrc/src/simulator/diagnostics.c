#include <stdint.h>
#include <string.h>

#include "internal.h"

void shadowspill_initialize_result(ShadowSpillSimulationResult *result) {
    ShadowSpillTaskInterval *task_intervals = result->task_intervals;
    uint32_t task_capacity = result->task_interval_capacity;
    ShadowSpillTransferInterval *transfer_intervals = result->transfer_intervals;
    uint32_t transfer_capacity = result->transfer_interval_capacity;
    ShadowSpillDevicePeak *device_peaks = result->device_peaks;
    uint32_t device_capacity = result->device_peak_capacity;
    ShadowSpillCapacityViolation *violations = result->capacity_violations;
    uint32_t violation_capacity = result->capacity_violation_capacity;
    memset(result, 0, sizeof(*result));
    result->task_intervals = task_intervals;
    result->task_interval_capacity = task_capacity;
    result->transfer_intervals = transfer_intervals;
    result->transfer_interval_capacity = transfer_capacity;
    result->device_peaks = device_peaks;
    result->device_peak_capacity = device_capacity;
    result->capacity_violations = violations;
    result->capacity_violation_capacity = violation_capacity;
    result->error_task = SHADOWSPILL_SIMULATOR_NO_INDEX;
    result->error_alias = SHADOWSPILL_SIMULATOR_NO_INDEX;
    result->error_device = SHADOWSPILL_SIMULATOR_NO_INDEX;
}

void shadowspill_record_capacity_violation(
    ShadowSpillSimulationResult *result,
    const ShadowSpillSimulationWork *work,
    uint8_t reason,
    uint32_t task,
    uint32_t alias,
    uint32_t device,
    uint8_t location,
    uint64_t capacity,
    uint64_t used,
    uint64_t requested
) {
    /* Counted even when it cannot be stored, so a caller can distinguish a
     * complete list from a truncated one. */
    uint32_t index = result->capacity_violation_count;
    if (result->capacity_violation_count != UINT32_MAX) {
        ++result->capacity_violation_count;
    }
    if (result->capacity_violations == NULL ||
        index >= result->capacity_violation_capacity) {
        return;
    }
    result->capacity_violations[index] = (ShadowSpillCapacityViolation){
        .time_ns = work == NULL ? 0U : work->now_ns,
        .capacity_bytes = capacity,
        .used_bytes = used,
        .requested_bytes = requested,
        .task = task,
        .alias = alias,
        .device = device,
        .location = location,
        .reason = reason,
    };
}

void shadowspill_set_error(
    ShadowSpillSimulationResult *result,
    ShadowSpillStatus status,
    const ShadowSpillSimulationWork *work,
    uint32_t task,
    uint32_t alias,
    uint32_t device
) {
    result->status = (uint32_t)status;
    result->error_time_ns = work == NULL ? 0U : work->now_ns;
    result->error_task = task;
    result->error_alias = alias;
    result->error_device = device;
}

void shadowspill_set_capacity_error(
    ShadowSpillSimulationResult *result,
    ShadowSpillStatus status,
    const ShadowSpillSimulationWork *work,
    uint32_t task,
    uint32_t alias,
    uint32_t device,
    uint8_t location,
    uint64_t capacity,
    uint64_t used,
    uint64_t requested
) {
    shadowspill_set_error(result, status, work, task, alias, device);
    result->error_location = location;
    result->error_capacity_bytes = capacity;
    result->error_used_bytes = used;
    result->error_requested_bytes = requested;
}

int shadowspill_report_deadlock(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work,
    ShadowSpillSimulationResult *result
) {
    for (uint32_t index = 0; index < program->action_count; ++index) {
        ShadowSpillTransferState *transfer = &work->transfers[index];
        if (transfer->state != SHADOWSPILL_TRANSFER_QUEUED) {
            continue;
        }
        uint32_t alias = transfer->alias;
        uint32_t device = transfer->device;
        ShadowSpillAliasState *state = &work->aliases[alias];
        if (transfer->direction == SHADOWSPILL_TRANSFER_FETCH &&
            state->device_allocated == 0U) {
            uint64_t used = shadowspill_device_used_bytes(
                program, work, device
            );
            uint64_t total = 0U;
            if (shadowspill_add_overflow_u64(
                    used, program->alias_size_bytes[alias], &total
                ) || total > program->devices[device].capacity_bytes) {
                shadowspill_set_capacity_error(
                    result,
                    SHADOWSPILL_STATUS_PREFETCH_DEVICE_CAPACITY,
                    work,
                    transfer->trigger_task,
                    alias,
                    device,
                    SHADOWSPILL_MEMORY_DEVICE,
                    program->devices[device].capacity_bytes,
                    used,
                    program->alias_size_bytes[alias]
                );
                return 0;
            }
        }
        if (transfer->direction == SHADOWSPILL_TRANSFER_EVICT &&
            state->spill_allocated == 0U) {
            uint64_t total = 0U;
            if (shadowspill_add_overflow_u64(
                    work->spill_bytes,
                    program->alias_size_bytes[alias],
                    &total
                ) || total > program->spill_capacity_bytes) {
                shadowspill_set_capacity_error(
                    result,
                    SHADOWSPILL_STATUS_OFFLOAD_SPILL_CAPACITY,
                    work,
                    transfer->trigger_task,
                    alias,
                    device,
                    SHADOWSPILL_MEMORY_SPILL,
                    program->spill_capacity_bytes,
                    work->spill_bytes,
                    program->alias_size_bytes[alias]
                );
                return 0;
            }
        }
    }
    for (uint32_t task = 0; task < program->task_count; ++task) {
        ShadowSpillTaskState *state = &work->tasks[task];
        if (state->state != SHADOWSPILL_TASK_UNLAUNCHED) {
            continue;
        }
        uint64_t dependency_ready = 0U;
        if (!shadowspill_task_dependencies_complete(
                program, work, task, &dependency_ready
            )) {
            continue;
        }
        uint32_t input_begin = program->input_offsets[task];
        uint32_t input_end = program->input_offsets[task + 1U];
        for (uint32_t index = input_begin; index < input_end; ++index) {
            uint32_t alias = program->input_aliases[index];
            if (work->aliases[alias].device_ready == 0U ||
                work->aliases[alias].fetch_pending != 0U ||
                work->aliases[alias].evict_pending != 0U) {
                shadowspill_set_error(
                    result,
                    SHADOWSPILL_STATUS_TASK_INPUT_DEADLOCK,
                    work,
                    task,
                    alias,
                    program->task_device[task]
                );
                return 0;
            }
        }
        uint32_t device = program->task_device[task];
        uint64_t logical_requested = program->task_workspace_bytes[task];
        uint32_t output_begin = program->output_offsets[task];
        uint32_t output_end = program->output_offsets[task + 1U];
        for (uint32_t index = output_begin; index < output_end; ++index) {
            uint32_t alias = program->output_aliases[index];
            if (work->aliases[alias].device_allocated == 0U) {
                if (shadowspill_add_overflow_u64(
                        logical_requested,
                        program->alias_size_bytes[alias],
                        &logical_requested
                    )) {
                    logical_requested = UINT64_MAX;
                    break;
                }
            }
        }
        int64_t physical_delta = 0;
        if (logical_requested > (uint64_t)INT64_MAX ||
            !shadowspill_resolve_physical_delta(
                program,
                program->task_start_physical_deltas,
                task,
                (int64_t)logical_requested,
                &physical_delta
            )) {
            shadowspill_set_error(
                result,
                SHADOWSPILL_STATUS_SIMULATION_INTERNAL_ERROR,
                work,
                task,
                SHADOWSPILL_SIMULATOR_NO_INDEX,
                device
            );
            return 0;
        }
        uint64_t requested = physical_delta > 0
            ? (uint64_t)physical_delta : 0U;
        uint64_t used = shadowspill_device_used_bytes(program, work, device);
        uint64_t capacity = program->devices[device].capacity_bytes;
        uint64_t admitted_used = used > capacity ? capacity : used;
        if (requested > capacity - admitted_used) {
            shadowspill_set_capacity_error(
                result,
                SHADOWSPILL_STATUS_TASK_DEVICE_CAPACITY,
                work,
                task,
                output_begin < output_end
                    ? program->output_aliases[output_begin]
                    : SHADOWSPILL_SIMULATOR_NO_INDEX,
                device,
                SHADOWSPILL_MEMORY_DEVICE,
                capacity,
                used,
                requested
            );
            return 0;
        }
    }
    shadowspill_set_error(
        result,
        SHADOWSPILL_STATUS_TRANSFER_DEADLOCK,
        work,
        SHADOWSPILL_SIMULATOR_NO_INDEX,
        SHADOWSPILL_SIMULATOR_NO_INDEX,
        SHADOWSPILL_SIMULATOR_NO_INDEX
    );
    return 0;
}

int shadowspill_check_final_residency(
    const ShadowSpillSimulationProgram *program,
    const ShadowSpillSimulationWork *work,
    ShadowSpillSimulationResult *result
) {
    for (uint32_t index = 0; index < program->final_count; ++index) {
        uint32_t alias = program->final_aliases[index];
        const ShadowSpillAliasState *state = &work->aliases[alias];
        int ready = program->final_locations[index] == SHADOWSPILL_MEMORY_DEVICE
            ? state->device_ready != 0U
            : state->spill_ready != 0U;
        if (!ready) {
            shadowspill_set_error(
                result,
                SHADOWSPILL_STATUS_FINAL_RESIDENCY,
                work,
                SHADOWSPILL_SIMULATOR_NO_INDEX,
                alias,
                program->alias_device[alias]
            );
            result->error_location = program->final_locations[index];
            return 0;
        }
    }
    return 1;
}
