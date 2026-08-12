#include <stdint.h>

#include "internal.h"

static uint64_t multiply_divide_ceil_bounded(
    uint64_t multiplicand,
    uint32_t multiplier,
    uint64_t divisor
) {
    uint64_t quotient = 0U;
    uint64_t remainder = 0U;
    uint32_t mask = 1U << 31U;
    while (mask != 0U) {
        quotient *= 2U;
        if (remainder >= divisor - remainder) {
            remainder -= divisor - remainder;
            quotient += 1U;
        } else {
            remainder *= 2U;
        }
        if ((multiplier & mask) != 0U) {
            if (remainder >= divisor - multiplicand) {
                remainder -= divisor - multiplicand;
                quotient += 1U;
            } else {
                remainder += multiplicand;
            }
        }
        mask >>= 1U;
    }
    return quotient + (remainder != 0U ? 1U : 0U);
}

static uint64_t transfer_runtime_ns(
    const ShadowSpillSimulationProgram *program,
    uint32_t alias,
    uint8_t direction
) {
    uint32_t device = program->alias_device[alias];
    const ShadowSpillSimulationDevice *config = &program->devices[device];
    uint64_t bandwidth = direction == SHADOWSPILL_TRANSFER_FETCH
        ? config->fetch_bandwidth_bytes_per_second
        : config->evict_bandwidth_bytes_per_second;
    uint64_t latency = direction == SHADOWSPILL_TRANSFER_FETCH
        ? config->fetch_latency_ns
        : config->evict_latency_ns;
    uint64_t size = program->alias_size_bytes[alias];
    uint64_t quotient = size / bandwidth;
    uint64_t remainder = size % bandwidth;
    uint64_t seconds_ns = quotient > UINT64_MAX / 1000000000U
        ? UINT64_MAX
        : quotient * 1000000000U;
    uint64_t partial = multiply_divide_ceil_bounded(
        remainder, 1000000000U, bandwidth
    );
    uint64_t runtime = 0U;
    if (seconds_ns == UINT64_MAX || shadowspill_add_overflow_u64(
            seconds_ns, partial, &runtime
        ) || shadowspill_add_overflow_u64(runtime, latency, &runtime)) {
        return UINT64_MAX;
    }
    return runtime;
}

static int earlier_queued_transfer(
    const ShadowSpillSimulationProgram *program,
    const ShadowSpillSimulationWork *work,
    uint32_t transfer,
    uint32_t device,
    uint8_t direction
) {
    for (uint32_t index = 0; index < transfer; ++index) {
        const ShadowSpillTransferState *candidate = &work->transfers[index];
        if (candidate->state == SHADOWSPILL_TRANSFER_QUEUED &&
            candidate->device == device &&
            candidate->direction == direction) {
            return 1;
        }
    }
    (void)program;
    return 0;
}

static int try_start_direction(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work,
    uint32_t device,
    uint8_t direction
) {
    int32_t *active = direction == SHADOWSPILL_TRANSFER_FETCH
        ? &work->active_fetch[device]
        : &work->active_evict[device];
    if (*active >= 0) {
        return 0;
    }
    for (uint32_t index = 0; index < program->action_count; ++index) {
        ShadowSpillTransferState *transfer = &work->transfers[index];
        if (transfer->state != SHADOWSPILL_TRANSFER_QUEUED ||
            transfer->device != device || transfer->direction != direction ||
            earlier_queued_transfer(program, work, index, device, direction)) {
            continue;
        }
        uint32_t alias = transfer->alias;
        ShadowSpillAliasState *state = &work->aliases[alias];
        if (direction == SHADOWSPILL_TRANSFER_FETCH) {
            if (state->evict_pending != 0U || state->host_ready == 0U) {
                transfer->stall_mask |= SHADOWSPILL_STALL_SOURCE_READINESS;
                return 0;
            }
            if (state->device_allocated == 0U) {
                transfer->stall_mask |= SHADOWSPILL_STALL_DEVICE_CAPACITY;
                return 0;
            }
        } else {
            if (state->device_ready == 0U) {
                transfer->stall_mask |= SHADOWSPILL_STALL_SOURCE_READINESS;
                return 0;
            }
            if (state->host_allocated == 0U) {
                return 0;
            }
        }
        transfer->state = SHADOWSPILL_TRANSFER_ACTIVE;
        transfer->start_ns = work->now_ns;
        uint64_t runtime = transfer_runtime_ns(program, alias, direction);
        if (shadowspill_add_overflow_u64(
                work->now_ns, runtime, &transfer->end_ns
            )) {
            transfer->end_ns = UINT64_MAX;
        }
        *active = (int32_t)index;
        shadowspill_update_peaks(program, work);
        return 1;
    }
    return 0;
}

int shadowspill_try_start_transfers(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work
) {
    int changed = 0;
    for (uint32_t device = 0; device < program->device_count; ++device) {
        changed |= try_start_direction(
            program, work, device, SHADOWSPILL_TRANSFER_FETCH
        );
        changed |= try_start_direction(
            program, work, device, SHADOWSPILL_TRANSFER_EVICT
        );
    }
    return changed;
}

static int submit_action(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work,
    ShadowSpillSimulationResult *result,
    uint32_t action
) {
    uint32_t task = program->action_trigger_tasks[action];
    uint32_t alias = program->action_aliases[action];
    uint32_t device = program->alias_device[alias];
    uint8_t kind = program->action_kinds[action];
    ShadowSpillAliasState *state = &work->aliases[alias];
    if (kind == SHADOWSPILL_MEMORY_RELEASE) {
        if (state->device_allocated == 0U || state->device_ready == 0U) {
            shadowspill_set_error(
                result,
                SHADOWSPILL_SIMULATION_INVALID_RELEASE,
                work,
                task,
                alias,
                device
            );
            return 0;
        }
        if (state->fetch_pending != 0U || state->evict_pending != 0U) {
            shadowspill_set_error(
                result,
                SHADOWSPILL_SIMULATION_RELEASE_TRANSFER_CONFLICT,
                work,
                task,
                alias,
                device
            );
            return 0;
        }
        state->device_allocated = 0U;
        state->device_ready = 0U;
        work->device_object_bytes[device] -= program->alias_size_bytes[alias];
        if (state->host_allocated != 0U &&
            program->alias_retain_spill_copy[alias] == 0U) {
            state->host_allocated = 0U;
            state->host_ready = 0U;
            work->host_bytes -= program->alias_size_bytes[alias];
        }
        shadowspill_update_peaks(program, work);
        return 1;
    }
    ShadowSpillTransferState *transfer = &work->transfers[action];
    transfer->alias = alias;
    transfer->trigger_task = task;
    transfer->device = device;
    transfer->ready_ns = work->now_ns;
    if (kind == SHADOWSPILL_MEMORY_OFFLOAD) {
        if (state->device_allocated == 0U || state->device_ready == 0U) {
            shadowspill_set_error(
                result,
                SHADOWSPILL_SIMULATION_INVALID_OFFLOAD,
                work,
                task,
                alias,
                device
            );
            return 0;
        }
        transfer->direction = SHADOWSPILL_TRANSFER_EVICT;
        transfer->sequence = work->evict_sequence[device]++;
        if (state->host_allocated == 0U) {
            uint64_t total = 0U;
            if (shadowspill_add_overflow_u64(
                    work->host_bytes,
                    program->alias_size_bytes[alias],
                    &total
                ) || total > program->host_capacity_bytes) {
                shadowspill_set_capacity_error(
                    result,
                    SHADOWSPILL_SIMULATION_OFFLOAD_HOST_CAPACITY,
                    work,
                    task,
                    alias,
                    device,
                    SHADOWSPILL_MEMORY_HOST,
                    program->host_capacity_bytes,
                    work->host_bytes,
                    program->alias_size_bytes[alias]
                );
                return 0;
            }
            state->host_allocated = 1U;
            state->host_ready = 0U;
            work->host_bytes = total;
        }
        state->evict_pending = 1U;
    } else {
        if ((state->device_allocated != 0U && state->evict_pending == 0U) ||
            (state->host_ready == 0U && state->evict_pending == 0U)) {
            shadowspill_set_error(
                result,
                SHADOWSPILL_SIMULATION_INVALID_PREFETCH,
                work,
                task,
                alias,
                device
            );
            return 0;
        }
        transfer->direction = SHADOWSPILL_TRANSFER_FETCH;
        transfer->sequence = work->fetch_sequence[device]++;
        if (state->device_allocated == 0U) {
            uint64_t used = work->device_object_bytes[device] +
                work->device_workspace_bytes[device];
            uint64_t total = 0U;
            if (shadowspill_add_overflow_u64(
                    used, program->alias_size_bytes[alias], &total
                ) || total > program->devices[device].capacity_bytes) {
                shadowspill_set_capacity_error(
                    result,
                    SHADOWSPILL_SIMULATION_PREFETCH_DEVICE_CAPACITY,
                    work,
                    task,
                    alias,
                    device,
                    SHADOWSPILL_MEMORY_DEVICE,
                    program->devices[device].capacity_bytes,
                    used,
                    program->alias_size_bytes[alias]
                );
                return 0;
            }
            state->device_allocated = 1U;
            state->device_ready = 0U;
            work->device_object_bytes[device] = total;
        }
        state->fetch_pending = 1U;
    }
    transfer->state = SHADOWSPILL_TRANSFER_QUEUED;
    shadowspill_update_peaks(program, work);
    return 1;
}

int shadowspill_submit_ready_actions(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work,
    ShadowSpillSimulationResult *result
) {
    while (work->submitted_actions < program->action_count) {
        uint32_t action = work->submitted_actions;
        uint32_t trigger = program->action_trigger_tasks[action];
        if (work->tasks[trigger].state != SHADOWSPILL_TASK_COMPLETE) {
            break;
        }
        if (!submit_action(program, work, result, action)) {
            return 0;
        }
        work->submitted_actions += 1U;
    }
    return 1;
}

static int append_transfer_interval(
    const ShadowSpillSimulationProgram *program,
    const ShadowSpillSimulationWork *work,
    ShadowSpillSimulationResult *result,
    uint32_t transfer_index
) {
    if (result->transfer_interval_count >= result->transfer_interval_capacity) {
        return 0;
    }
    const ShadowSpillTransferState *transfer = &work->transfers[transfer_index];
    result->transfer_intervals[result->transfer_interval_count++] =
        (ShadowSpillTransferInterval){
            .alias = transfer->alias,
            .trigger_task = transfer->trigger_task,
            .device = transfer->device,
            .direction = transfer->direction,
            .sequence = transfer->sequence,
            .ready_ns = transfer->ready_ns,
            .start_ns = transfer->start_ns,
            .end_ns = transfer->end_ns,
            .bytes = program->alias_size_bytes[transfer->alias],
            .stall_mask = transfer->stall_mask,
        };
    return 1;
}

int shadowspill_complete_transfer(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work,
    ShadowSpillSimulationResult *result,
    uint32_t device,
    uint8_t direction
) {
    int32_t *active = direction == SHADOWSPILL_TRANSFER_FETCH
        ? &work->active_fetch[device]
        : &work->active_evict[device];
    uint32_t index = (uint32_t)*active;
    ShadowSpillTransferState *transfer = &work->transfers[index];
    ShadowSpillAliasState *state = &work->aliases[transfer->alias];
    if (direction == SHADOWSPILL_TRANSFER_FETCH) {
        state->device_ready = 1U;
        state->device_version = state->host_version;
        state->fetch_pending = 0U;
        if (program->alias_retain_spill_copy[transfer->alias] == 0U) {
            state->host_allocated = 0U;
            state->host_ready = 0U;
            work->host_bytes -= program->alias_size_bytes[transfer->alias];
        }
    } else {
        state->host_ready = 1U;
        state->host_version = state->device_version;
        state->evict_pending = 0U;
        state->device_ready = 0U;
        if (state->fetch_pending == 0U) {
            state->device_allocated = 0U;
            work->device_object_bytes[device] -=
                program->alias_size_bytes[transfer->alias];
        }
    }
    transfer->state = SHADOWSPILL_TRANSFER_COMPLETE;
    *active = -1;
    if (!append_transfer_interval(program, work, result, index)) {
        shadowspill_set_error(
            result,
            SHADOWSPILL_SIMULATION_INTERNAL_ERROR,
            work,
            transfer->trigger_task,
            transfer->alias,
            device
        );
        return 0;
    }
    shadowspill_update_peaks(program, work);
    return 1;
}
