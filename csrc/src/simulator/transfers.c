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

static int try_start_direction(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work,
    uint32_t device,
    uint8_t direction
) {
    int32_t *active = direction == SHADOWSPILL_TRANSFER_FETCH
        ? &work->active_fetch[device]
        : &work->active_evict[device];
    uint32_t *cursor = direction == SHADOWSPILL_TRANSFER_FETCH
        ? &work->fetch_cursor[device]
        : &work->evict_cursor[device];
    if (*active >= 0) {
        return 0;
    }
    for (uint32_t index = *cursor; index < work->submitted_actions; ++index) {
        ShadowSpillTransferState *transfer = &work->transfers[index];
        if (transfer->state != SHADOWSPILL_TRANSFER_QUEUED ||
            transfer->device != device || transfer->direction != direction) {
            *cursor = index + 1U;
            continue;
        }
        *cursor = index;
        if (!shadowspill_action_reuse_dependencies_complete(
                program, work, index
            )) {
            transfer->stall_mask |= SHADOWSPILL_STALL_MEMORY_REUSE;
            return 0;
        }
        uint32_t alias = transfer->alias;
        ShadowSpillAliasState *state = &work->aliases[alias];
        if (direction == SHADOWSPILL_TRANSFER_FETCH) {
            if (state->evict_pending != 0U || state->spill_ready == 0U) {
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
            if (state->spill_allocated == 0U) {
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
        *cursor = index + 1U;
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

/*
 * Bring one scheduled action into the simulation.
 *
 * `deferred` reports the runtime's own answer to a prefetch that has nowhere
 * to land: wait and try again, rather than fail. The action stays unsubmitted
 * and no state is touched, so the next time anything frees memory the caller
 * retries it. A plan that can never make room deadlocks instead, which the
 * main loop reports with the stall reasons that caused it.
 */
static int submit_action(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work,
    ShadowSpillSimulationResult *result,
    uint32_t action,
    int *deferred
) {
    *deferred = 0;
    uint32_t task = program->action_trigger_tasks[action];
    uint32_t alias = program->action_aliases[action];
    uint32_t device = program->alias_device[alias];
    uint8_t kind = program->action_kinds[action];
    ShadowSpillAliasState *state = &work->aliases[alias];
    uint64_t size = program->alias_size_bytes[alias];
    int64_t physical_delta = 0;
    if (program->use_admission_accounting != 0U) {
        if (size > (uint64_t)INT64_MAX) {
            shadowspill_set_error(
                result,
                SHADOWSPILL_STATUS_SIMULATION_INTERNAL_ERROR,
                work,
                task,
                alias,
                device
            );
            return 0;
        }
        int64_t default_physical_delta = 0;
        if (kind == SHADOWSPILL_MEMORY_RELEASE) {
            default_physical_delta = -(int64_t)size;
        } else if (kind == SHADOWSPILL_MEMORY_PREFETCH &&
            state->device_allocated == 0U) {
            default_physical_delta = (int64_t)size;
        }
        if (!shadowspill_resolve_physical_delta(
                program,
                program->action_trigger_physical_deltas,
                action,
                default_physical_delta,
                &physical_delta
            )) {
            shadowspill_set_error(
                result,
                SHADOWSPILL_STATUS_SIMULATION_INTERNAL_ERROR,
                work,
                task,
                alias,
                device
            );
            return 0;
        }
        if (!shadowspill_physical_delta_fits(
                program, work, device, physical_delta
            )) {
            if (program->relax_capacity == 0U) {
                /* Nothing has been mutated yet, so waiting is free. */
                *deferred = 1;
                return 1;
            }
            shadowspill_record_capacity_violation(
                result,
                work,
                SHADOWSPILL_CAPACITY_PREFETCH_DEVICE,
                task,
                alias,
                device,
                SHADOWSPILL_MEMORY_DEVICE,
                program->devices[device].capacity_bytes,
                shadowspill_device_used_bytes(program, work, device),
                physical_delta > 0 ? (uint64_t)physical_delta : 0U
            );
        }
    }
    if (kind == SHADOWSPILL_MEMORY_RELEASE) {
        if (state->device_allocated == 0U || state->device_ready == 0U) {
            shadowspill_set_error(
                result,
                SHADOWSPILL_STATUS_INVALID_RELEASE,
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
                SHADOWSPILL_STATUS_RELEASE_TRANSFER_CONFLICT,
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
        if (state->spill_allocated != 0U &&
            program->alias_retain_spill_copy[alias] == 0U) {
            state->spill_allocated = 0U;
            state->spill_ready = 0U;
            work->spill_bytes -= program->alias_size_bytes[alias];
        }
        if (program->use_admission_accounting != 0U &&
            !shadowspill_apply_physical_delta(
                program, work, device, physical_delta
            )) {
            shadowspill_set_error(
                result,
                SHADOWSPILL_STATUS_SIMULATION_INTERNAL_ERROR,
                work,
                task,
                alias,
                device
            );
            return 0;
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
                SHADOWSPILL_STATUS_INVALID_OFFLOAD,
                work,
                task,
                alias,
                device
            );
            return 0;
        }
        transfer->direction = SHADOWSPILL_TRANSFER_EVICT;
        transfer->sequence = work->evict_sequence[device]++;
        if (state->spill_allocated == 0U) {
            uint64_t total = 0U;
            if (shadowspill_add_overflow_u64(
                    work->spill_bytes,
                    program->alias_size_bytes[alias],
                    &total
                )) {
                shadowspill_set_capacity_error(
                    result,
                    SHADOWSPILL_STATUS_OFFLOAD_SPILL_CAPACITY,
                    work,
                    task,
                    alias,
                    device,
                    SHADOWSPILL_MEMORY_SPILL,
                    program->spill_capacity_bytes,
                    work->spill_bytes,
                    program->alias_size_bytes[alias]
                );
                return 0;
            }
            if (total > program->spill_capacity_bytes) {
                if (program->relax_capacity == 0U) {
                    shadowspill_set_capacity_error(
                        result,
                        SHADOWSPILL_STATUS_OFFLOAD_SPILL_CAPACITY,
                        work,
                        task,
                        alias,
                        device,
                        SHADOWSPILL_MEMORY_SPILL,
                        program->spill_capacity_bytes,
                        work->spill_bytes,
                        program->alias_size_bytes[alias]
                    );
                    return 0;
                }
                shadowspill_record_capacity_violation(
                    result,
                    work,
                    SHADOWSPILL_CAPACITY_OFFLOAD_SPILL,
                    task,
                    alias,
                    device,
                    SHADOWSPILL_MEMORY_SPILL,
                    program->spill_capacity_bytes,
                    work->spill_bytes,
                    program->alias_size_bytes[alias]
                );
            }
            state->spill_allocated = 1U;
            state->spill_ready = 0U;
            work->spill_bytes = total;
        }
        state->evict_pending = 1U;
    } else {
        if ((state->device_allocated != 0U && state->evict_pending == 0U) ||
            (state->spill_ready == 0U && state->evict_pending == 0U)) {
            shadowspill_set_error(
                result,
                SHADOWSPILL_STATUS_INVALID_PREFETCH,
                work,
                task,
                alias,
                device
            );
            return 0;
        }
        /* Tested before anything is mutated, so a deferral leaves no trace
         * and the retry sees exactly the state this call found. */
        if (state->device_allocated == 0U &&
            program->use_admission_accounting == 0U) {
            uint64_t used = shadowspill_device_used_bytes(
                program, work, device
            );
            if (size > program->devices[device].capacity_bytes ||
                used > program->devices[device].capacity_bytes - size) {
                if (program->relax_capacity == 0U) {
                    *deferred = 1;
                    return 1;
                }
                shadowspill_record_capacity_violation(
                    result,
                    work,
                    SHADOWSPILL_CAPACITY_PREFETCH_DEVICE,
                    task,
                    alias,
                    device,
                    SHADOWSPILL_MEMORY_DEVICE,
                    program->devices[device].capacity_bytes,
                    used,
                    size
                );
            }
        }
        transfer->direction = SHADOWSPILL_TRANSFER_FETCH;
        transfer->sequence = work->fetch_sequence[device]++;
        if (state->device_allocated == 0U) {
            state->device_allocated = 1U;
            state->device_ready = 0U;
            work->device_object_bytes[device] +=
                program->alias_size_bytes[alias];
        }
        state->fetch_pending = 1U;
    }
    if (program->use_admission_accounting != 0U &&
        !shadowspill_apply_physical_delta(
            program, work, device, physical_delta
        )) {
        shadowspill_set_error(
            result,
            SHADOWSPILL_STATUS_SIMULATION_INTERNAL_ERROR,
            work,
            task,
            alias,
            device
        );
        return 0;
    }
    transfer->state = SHADOWSPILL_TRANSFER_QUEUED;
    work->pending_transfers += 1U;
    shadowspill_update_peaks(program, work);
    return 1;
}

int shadowspill_submit_ready_actions(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work,
    ShadowSpillSimulationResult *result,
    int *submitted
) {
    if (submitted != NULL) {
        *submitted = 0;
    }
    while (work->submitted_actions < program->action_count) {
        uint32_t action = work->submitted_actions;
        uint32_t trigger = program->action_trigger_tasks[action];
        if (work->tasks[trigger].state != SHADOWSPILL_TASK_COMPLETE) {
            break;
        }
        int deferred = 0;
        if (!submit_action(program, work, result, action, &deferred)) {
            return 0;
        }
        if (deferred != 0) {
            /* Actions are submitted in order, so a waiting prefetch holds the
             * ones behind it. Whatever frees memory wakes the whole queue. */
            break;
        }
        work->submitted_actions += 1U;
        if (submitted != NULL) {
            *submitted = 1;
        }
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
    uint64_t size = program->alias_size_bytes[transfer->alias];
    int64_t physical_delta = 0;
    if (program->use_admission_accounting != 0U) {
        if (size > (uint64_t)INT64_MAX) {
            shadowspill_set_error(
                result,
                SHADOWSPILL_STATUS_SIMULATION_INTERNAL_ERROR,
                work,
                transfer->trigger_task,
                transfer->alias,
                device
            );
            return 0;
        }
        int64_t default_physical_delta = 0;
        if (direction == SHADOWSPILL_TRANSFER_EVICT &&
            state->fetch_pending == 0U) {
            default_physical_delta = -(int64_t)size;
        }
        if (!shadowspill_resolve_physical_delta(
                program,
                program->action_completion_physical_deltas,
                index,
                default_physical_delta,
                &physical_delta
            )) {
            shadowspill_set_error(
                result,
                SHADOWSPILL_STATUS_SIMULATION_INTERNAL_ERROR,
                work,
                transfer->trigger_task,
                transfer->alias,
                device
            );
            return 0;
        }
    }
    if (direction == SHADOWSPILL_TRANSFER_FETCH) {
        state->device_ready = 1U;
        state->device_version = state->spill_version;
        state->fetch_pending = 0U;
        if (program->alias_retain_spill_copy[transfer->alias] == 0U) {
            state->spill_allocated = 0U;
            state->spill_ready = 0U;
            work->spill_bytes -= program->alias_size_bytes[transfer->alias];
        }
    } else {
        state->spill_ready = 1U;
        state->spill_version = state->device_version;
        state->evict_pending = 0U;
        state->device_ready = 0U;
        if (state->fetch_pending == 0U) {
            state->device_allocated = 0U;
            work->device_object_bytes[device] -=
                program->alias_size_bytes[transfer->alias];
        }
    }
    if (program->use_admission_accounting != 0U &&
        !shadowspill_apply_physical_delta(
            program, work, device, physical_delta
        )) {
        shadowspill_set_error(
            result,
            SHADOWSPILL_STATUS_SIMULATION_INTERNAL_ERROR,
            work,
            transfer->trigger_task,
            transfer->alias,
            device
        );
        return 0;
    }
    transfer->state = SHADOWSPILL_TRANSFER_COMPLETE;
    work->pending_transfers -= 1U;
    *active = -1;
    if (!append_transfer_interval(program, work, result, index)) {
        shadowspill_set_error(
            result,
            SHADOWSPILL_STATUS_SIMULATION_INTERNAL_ERROR,
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
