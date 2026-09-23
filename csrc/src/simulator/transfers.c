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
    for (uint32_t index = *cursor; index < work->submitted_actions; ++index) {
        ShadowSpillTransferState *transfer = &work->transfers[index];
        if (transfer->state != SHADOWSPILL_TRANSFER_QUEUED ||
            transfer->device != device || transfer->direction != direction) {
            *cursor = index + 1U;
            continue;
        }
        *cursor = index;
        /* The head of this lane's queue, found before the lane's state is
         * consulted, so a copy that is eligible and waiting for a lane
         * carrying another says so. The cursor parks here and only moves
         * forward, so the walk costs nothing once it has reached the head.
         */
        if (*active >= 0) {
            transfer->stall_mask |= SHADOWSPILL_STALL_LANE_BUSY;
            return 0;
        }
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
 * Note that this action wanted to go and could not, so the wait is visible
 * as memory pressure rather than disappearing into a later ready time.
 *
 * `ready_ns` is stamped at the first refusal, because that is the moment the
 * action became eligible; stamping it at the eventual submission instead
 * would report the whole wait as "not ready yet" and hide the pressure that
 * caused it.
 */
static void defer_action(
    ShadowSpillSimulationWork *work,
    ShadowSpillSimulationResult *result,
    uint32_t action,
    uint8_t reason,
    uint8_t location,
    uint32_t stall,
    uint32_t task,
    uint32_t alias,
    uint32_t device,
    uint64_t capacity,
    uint64_t used,
    uint64_t requested
) {
    work->submission_deferred = 1U;
    ShadowSpillTransferState *pending = &work->transfers[action];
    if ((pending->stall_mask & stall) == 0U) {
        pending->ready_ns = work->now_ns;
        /* Recorded once, at the first refusal, so the list counts places the
         * plan came up short rather than ticks it spent waiting. */
        shadowspill_record_capacity_violation(
            result,
            work,
            reason,
            task,
            alias,
            device,
            location,
            capacity,
            used,
            requested
        );
    }
    pending->stall_mask |= stall;
}

/*
 * Hold one action until a copy of its object lands. Nothing is mutated and
 * no shortfall is recorded: the plan is not short of memory, it is early.
 */
static void wait_for_transfer(
    ShadowSpillSimulationWork *work,
    uint32_t action,
    uint32_t stall
) {
    work->submission_deferred = 1U;
    ShadowSpillTransferState *pending = &work->transfers[action];
    if ((pending->stall_mask & stall) == 0U) {
        pending->ready_ns = work->now_ns;
    }
    pending->stall_mask |= stall;
}

/*
 * Bring one scheduled action into the simulation.
 *
 * `deferred` reports the runtime's own answer to a fetch that has nowhere
 * to land: wait and try again, rather than fail. The action stays unsubmitted
 * and no state is touched, so the next time anything frees memory the caller
 * retries it. A plan that can never make room deadlocks instead, which the
 * main loop reports with the stall reasons that caused it.
 */
/*
 * What one action names: the task that triggers it, the alias it moves, the
 * device that alias is on, and the state that alias is in. Read once, so the
 * steps below take one pointer rather than six values each.
 */
typedef struct ActionContext {
    uint32_t task;
    uint32_t alias;
    uint32_t device;
    uint8_t kind;
    uint64_t size;
    ShadowSpillAliasState *state;
} ActionContext;

/*
 * What this action does to the device's physical total, and whether the
 * device has room for it. Answers 0 on an error it has already set, and 1
 * otherwise, having deferred the action when capacity is what it waits on.
 */
static int action_physical_delta(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work,
    ShadowSpillSimulationResult *result,
    uint32_t action,
    const ActionContext *of,
    int64_t *physical_delta,
    int *deferred
) {
    if (program->use_admission_accounting != 0U) {
        if (of->size > (uint64_t)INT64_MAX) {
            shadowspill_set_error(
                result,
                SHADOWSPILL_STATUS_SIMULATION_INTERNAL_ERROR,
                work,
                of->task,
                of->alias,
                of->device
            );
            return 0;
        }
        int64_t default_physical_delta = 0;
        if (of->kind == SHADOWSPILL_MEMORY_RELEASE) {
            default_physical_delta = -(int64_t)of->size;
        } else if (of->kind == SHADOWSPILL_MEMORY_FETCH &&
            of->state->device_allocated == 0U) {
            default_physical_delta = (int64_t)of->size;
        }
        if (!shadowspill_resolve_physical_delta(
                program,
                program->action_trigger_physical_deltas,
                action,
                default_physical_delta,
                physical_delta
            )) {
            shadowspill_set_error(
                result,
                SHADOWSPILL_STATUS_SIMULATION_INTERNAL_ERROR,
                work,
                of->task,
                of->alias,
                of->device
            );
            return 0;
        }
        if (!shadowspill_physical_delta_fits(
                program, work, of->device, *physical_delta
            )) {
            /* Nothing has been mutated yet, so waiting is free. */
            defer_action(
                work,
                result,
                action,
                SHADOWSPILL_CAPACITY_FETCH_DEVICE,
                SHADOWSPILL_MEMORY_DEVICE,
                SHADOWSPILL_STALL_DEVICE_CAPACITY,
                of->task,
                of->alias,
                of->device,
                program->devices[of->device].capacity_bytes,
                shadowspill_device_used_bytes(program, work, of->device),
                *physical_delta > 0 ? (uint64_t)*physical_delta : 0U
            );
            *deferred = 1;
            return 1;
        }
    }
    return 1;
}

/* Dropping the device copy: what must be true of it, and what the drop frees. */
static int submit_release(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work,
    ShadowSpillSimulationResult *result,
    uint32_t action,
    const ActionContext *of,
    int64_t physical_delta,
    int *deferred
) {
    if (of->state->device_allocated == 0U || of->state->device_ready == 0U) {
        shadowspill_set_error(
            result,
            SHADOWSPILL_STATUS_INVALID_RELEASE,
            work,
            of->task,
            of->alias,
            of->device
        );
        return 0;
    }
    if (of->state->fetch_pending != 0U || of->state->evict_pending != 0U) {
        shadowspill_set_error(
            result,
            SHADOWSPILL_STATUS_RELEASE_TRANSFER_CONFLICT,
            work,
            of->task,
            of->alias,
            of->device
        );
        return 0;
    }
    if (of->state->write_back_pending != 0U) {
        /* The release drops the copy the write-back is still reading,
         * so it waits for the copy to land, as the runtime's does. */
        wait_for_transfer(
            work, action, SHADOWSPILL_STALL_SOURCE_READINESS
        );
        *deferred = 1;
        return 1;
    }
    uint32_t last_reader = work->alias_last_reader[of->alias];
    if (of->state->spill_ready == 0U &&
        (work->alias_final_required[of->alias] != 0U ||
         (last_reader != SHADOWSPILL_SIMULATOR_NO_INDEX &&
          work->tasks[last_reader].state != SHADOWSPILL_TASK_COMPLETE))) {
        /* Dropping the only current copy of a value still needed is a
         * loss, reported here rather than at the fetch, task or final
         * residency that would miss it. */
        shadowspill_set_error(
            result,
            SHADOWSPILL_STATUS_INVALID_RELEASE,
            work,
            of->task,
            of->alias,
            of->device
        );
        return 0;
    }
    of->state->device_allocated = 0U;
    of->state->device_ready = 0U;
    work->device_object_bytes[of->device] -= program->alias_size_bytes[of->alias];
    if (of->state->spill_allocated != 0U &&
        program->alias_retain_spill_copy[of->alias] == 0U) {
        of->state->spill_allocated = 0U;
        of->state->spill_ready = 0U;
        work->spill_bytes -= program->alias_size_bytes[of->alias];
    }
    if (program->use_admission_accounting != 0U &&
        !shadowspill_apply_physical_delta(
            program, work, of->device, physical_delta
        )) {
        shadowspill_set_error(
            result,
            SHADOWSPILL_STATUS_SIMULATION_INTERNAL_ERROR,
            work,
            of->task,
            of->alias,
            of->device
        );
        return 0;
    }
    shadowspill_update_peaks(program, work);
    return 1;

}

/*
 * A departure: an evict, which gives the device copy up, or a write-back,
 * which keeps it. One copy of an object at a time, so a second departure
 * while the first is still on the lane is refused rather than queued.
 *
 * Answers 0 on an error it has already set, 1 with the copy on the lane, and
 * 2 when there was nothing to copy and the action is already finished.
 */
static int submit_departure(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work,
    ShadowSpillSimulationResult *result,
    uint32_t action,
    const ActionContext *of,
    ShadowSpillTransferState *transfer,
    int64_t physical_delta,
    int *deferred
) {
    /* One copy of an object at a time: a second departure while the
     * first is still on the lane has no single version to carry. */
    if (of->state->device_allocated == 0U || of->state->device_ready == 0U ||
        of->state->evict_pending != 0U || of->state->write_back_pending != 0U) {
        shadowspill_set_error(
            result,
            of->kind == SHADOWSPILL_MEMORY_EVICT
                ? SHADOWSPILL_STATUS_INVALID_EVICT
                : SHADOWSPILL_STATUS_INVALID_WRITE_BACK,
            work,
            of->task,
            of->alias,
            of->device
        );
        return 0;
    }
    if (of->kind == SHADOWSPILL_MEMORY_WRITE_BACK &&
        of->state->spill_ready != 0U) {
        /* Nothing to copy: the spill copy already holds this version,
         * so the action completes at its trigger and takes no lane. */
        if (program->use_admission_accounting != 0U &&
            !shadowspill_apply_physical_delta(
                program, work, of->device, physical_delta
            )) {
            shadowspill_set_error(
                result,
                SHADOWSPILL_STATUS_SIMULATION_INTERNAL_ERROR,
                work,
                of->task,
                of->alias,
                of->device
            );
            return 0;
        }
        shadowspill_update_peaks(program, work);
        return 2;
    }
    /* Tested before anything is mutated, so a deferral leaves no trace
     * and the retry sees exactly the state this call found. */
    uint64_t total = 0U;
    if (of->state->spill_allocated == 0U) {
        if (shadowspill_add_overflow_u64(
                work->spill_bytes,
                program->alias_size_bytes[of->alias],
                &total
            )) {
            shadowspill_set_capacity_error(
                result,
                SHADOWSPILL_STATUS_EVICT_SPILL_CAPACITY,
                work,
                of->task,
                of->alias,
                of->device,
                SHADOWSPILL_MEMORY_SPILL,
                program->spill_capacity_bytes,
                work->spill_bytes,
                program->alias_size_bytes[of->alias]
            );
            return 0;
        }
        if (total > program->spill_capacity_bytes) {
            /* The spill pool is the same question as the device pool,
             * one level down: an eviction with nowhere to land waits for
             * room, which a release of a copy the plan does not retain
             * eventually provides. */
            defer_action(
                work,
                result,
                action,
                SHADOWSPILL_CAPACITY_EVICT_SPILL,
                SHADOWSPILL_MEMORY_SPILL,
                SHADOWSPILL_STALL_SPILL_CAPACITY,
                of->task,
                of->alias,
                of->device,
                program->spill_capacity_bytes,
                work->spill_bytes,
                program->alias_size_bytes[of->alias]
            );
            *deferred = 1;
            return 1;
        }
    }
    transfer->direction = SHADOWSPILL_TRANSFER_EVICT;
    transfer->sequence = work->evict_sequence[of->device]++;
    transfer->version = of->state->device_version;
    if (of->state->spill_allocated == 0U) {
        of->state->spill_allocated = 1U;
        of->state->spill_ready = 0U;
        work->spill_bytes = total;
    }
    if (of->kind == SHADOWSPILL_MEMORY_EVICT) {
        of->state->evict_pending = 1U;
    } else {
        of->state->write_back_pending = 1U;
    }
    return 1;
}

/* A fetch: the spill copy comes back to the device, if there is room for it. */
static int submit_fetch(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work,
    ShadowSpillSimulationResult *result,
    uint32_t action,
    const ActionContext *of,
    ShadowSpillTransferState *transfer,
    int *deferred
) {
    if ((of->state->device_allocated != 0U && of->state->evict_pending == 0U) ||
        (of->state->spill_ready == 0U && of->state->evict_pending == 0U)) {
        shadowspill_set_error(
            result,
            SHADOWSPILL_STATUS_INVALID_FETCH,
            work,
            of->task,
            of->alias,
            of->device
        );
        return 0;
    }
    /* Tested before anything is mutated, so a deferral leaves no trace
     * and the retry sees exactly the state this call found. */
    if (of->state->device_allocated == 0U &&
        program->use_admission_accounting == 0U) {
        uint64_t used = shadowspill_device_used_bytes(
            program, work, of->device
        );
        if (of->size > program->devices[of->device].capacity_bytes ||
            used > program->devices[of->device].capacity_bytes - of->size) {
            defer_action(
                work,
                result,
                action,
                SHADOWSPILL_CAPACITY_FETCH_DEVICE,
                SHADOWSPILL_MEMORY_DEVICE,
                SHADOWSPILL_STALL_DEVICE_CAPACITY,
                of->task,
                of->alias,
                of->device,
                program->devices[of->device].capacity_bytes,
                used,
                of->size
            );
            *deferred = 1;
            return 1;
        }
    }
    transfer->direction = SHADOWSPILL_TRANSFER_FETCH;
    transfer->sequence = work->fetch_sequence[of->device]++;
    if (of->state->device_allocated == 0U) {
        of->state->device_allocated = 1U;
        of->state->device_ready = 0U;
        work->device_object_bytes[of->device] +=
            program->alias_size_bytes[of->alias];
    }
    of->state->fetch_pending = 1U;
    return 1;
}

/* Putting a copy on a lane, and charging what that does to the device. */
static int submit_transfer(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work,
    ShadowSpillSimulationResult *result,
    uint32_t action,
    const ActionContext *of,
    int64_t physical_delta,
    int *deferred
) {
    ShadowSpillTransferState *transfer = &work->transfers[action];
    transfer->alias = of->alias;
    transfer->trigger_task = of->task;
    transfer->device = of->device;
    if ((transfer->stall_mask & SHADOWSPILL_STALL_DEVICE_CAPACITY) == 0U) {
        transfer->ready_ns = work->now_ns;
    }
    const int submitted = of->kind == SHADOWSPILL_MEMORY_FETCH
        ? submit_fetch(program, work, result, action, of, transfer, deferred)
        : submit_departure(
              program, work, result, action, of, transfer, physical_delta, deferred
          );
    if (submitted != 1) {
        /* An error, or an action that finished without taking a lane. */
        return submitted == 0 ? 0 : 1;
    }
    if (*deferred != 0) {
        return 1;
    }
    if (program->use_admission_accounting != 0U &&
        !shadowspill_apply_physical_delta(
            program, work, of->device, physical_delta
        )) {
        shadowspill_set_error(
            result,
            SHADOWSPILL_STATUS_SIMULATION_INTERNAL_ERROR,
            work,
            of->task,
            of->alias,
            of->device
        );
        return 0;
    }
    transfer->state = SHADOWSPILL_TRANSFER_QUEUED;
    work->pending_transfers += 1U;
    shadowspill_update_peaks(program, work);
    return 1;
}

static int submit_action(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work,
    ShadowSpillSimulationResult *result,
    uint32_t action,
    int *deferred
) {
    *deferred = 0;
    const uint32_t alias = program->action_aliases[action];
    const ActionContext of = {
        .task = program->action_trigger_tasks[action],
        .alias = alias,
        .device = program->alias_device[alias],
        .kind = program->action_kinds[action],
        .size = program->alias_size_bytes[alias],
        .state = &work->aliases[alias],
    };
    int64_t physical_delta = 0;
    if (!action_physical_delta(
            program, work, result, action, &of, &physical_delta, deferred
        )) {
        return 0;
    }
    if (*deferred != 0) {
        return 1;
    }
    if (of.kind == SHADOWSPILL_MEMORY_RELEASE) {
        return submit_release(
            program, work, result, action, &of, physical_delta, deferred
        );
    }
    return submit_transfer(
        program, work, result, action, &of, physical_delta, deferred
    );
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
            /* Actions are submitted in order, so a waiting fetch holds the
             * ones behind it. Whatever frees memory wakes the whole queue. */
            break;
        }
        work->submitted_actions += 1U;
        work->submission_deferred = 0U;
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
            .kind = program->action_kinds[transfer_index],
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
    const uint8_t kind = program->action_kinds[index];
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
        if (kind == SHADOWSPILL_MEMORY_EVICT && state->fetch_pending == 0U) {
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
        /* The copy carries the version it started from. A write that
         * landed meanwhile leaves the spill copy stale, never current by
         * fiat, so a later fetch or the final residency reports the loss. */
        state->spill_version = transfer->version;
        state->spill_ready =
            state->device_version == transfer->version ? 1U : 0U;
        if (kind == SHADOWSPILL_MEMORY_WRITE_BACK) {
            /* The device copy stays, allocated and authoritative. */
            state->write_back_pending = 0U;
        } else {
            state->evict_pending = 0U;
            state->device_ready = 0U;
            if (state->fetch_pending == 0U) {
                state->device_allocated = 0U;
                work->device_object_bytes[device] -=
                    program->alias_size_bytes[transfer->alias];
            }
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
