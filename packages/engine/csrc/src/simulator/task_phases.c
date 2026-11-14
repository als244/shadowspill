/* Sequential task resource phases share transfer submission order. */
#include <stdlib.h>

#include "internal.h"

static int has_phases(const ShadowSpillSimulationProgram *program, uint32_t task) {
    return program->task_phase_offsets != NULL &&
        program->task_phase_offsets[task] < program->task_phase_offsets[task + 1U];
}

int shadowspill_computing_task(
    const ShadowSpillSimulationProgram *program,
    const ShadowSpillTaskState *state,
    uint32_t task
) {
    return !has_phases(program, task) || (!state->task_phase_waiting &&
        program->task_phases[state->task_phase_index].resource == UINT32_MAX);
}

int shadowspill_phase_task_free(
    const ShadowSpillSimulationProgram *program,
    const ShadowSpillSimulationWork *work,
    uint32_t resource
) {
    if (program->task_phase_offsets == NULL) {
        return 1;
    }
    for (uint32_t word = 0; word < work->task_word_count; ++word) {
        uint64_t active = work->active_tasks[word];
        while (active != 0U) {
            uint32_t task = word * 64U + (uint32_t)__builtin_ctzll(active);
            active &= active - 1U;
            const ShadowSpillTaskState *state = &work->tasks[task];
            if (has_phases(program, task) && !state->task_phase_waiting &&
                program->task_phases[state->task_phase_index].resource == resource) {
                return 0;
            }
        }
    }
    return 1;
}

int shadowspill_order_resource_phases(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work
) {
    if (program->transfer_resource_count == 0U && program->task_phase_count == 0U) {
        return 1;
    }
    work->task_phase_orders = calloc(
        program->task_phase_count ? program->task_phase_count : 1U,
        sizeof(*work->task_phase_orders)
    );
    if (work->task_phase_orders == NULL) {
        return 0;
    }
    for (uint32_t resource = 0; resource < program->transfer_resource_count; ++resource) {
        work->transfer_resource_completed[resource] = 0;
    }
    uint32_t action_begin = 0;
    for (uint32_t task = 0; task < program->task_count; ++task) {
        uint32_t phase_begin = has_phases(program, task)
            ? program->task_phase_offsets[task] : 0U;
        uint32_t phase_end = has_phases(program, task)
            ? program->task_phase_offsets[task + 1U] : 0U;
        for (uint32_t phase = phase_begin; phase < phase_end; ++phase) {
            uint32_t resource = program->task_phases[phase].resource;
            if (resource != UINT32_MAX) {
                work->task_phase_orders[phase] =
                    ++work->transfer_resource_completed[resource];
            }
        }
        uint32_t action_end = action_begin;
        while (action_end < program->action_count &&
               program->action_trigger_tasks[action_end] == task) {
            ++action_end;
        }
        for (int direction = SHADOWSPILL_TRANSFER_EVICT;
             direction >= SHADOWSPILL_TRANSFER_FETCH; --direction) {
            for (uint32_t action = action_begin; action < action_end; ++action) {
                uint8_t kind = program->action_kinds[action];
                if ((direction == SHADOWSPILL_TRANSFER_FETCH &&
                     kind != SHADOWSPILL_MEMORY_FETCH) ||
                    (direction == SHADOWSPILL_TRANSFER_EVICT &&
                     kind != SHADOWSPILL_MEMORY_EVICT)) {
                    continue;
                }
                uint32_t row = program->action_aliases[action] * 2U +
                    (uint32_t)direction;
                if (program->alias_transfer_offsets == NULL) {
                    continue;
                }
                for (uint32_t phase = program->alias_transfer_offsets[row];
                     phase < program->alias_transfer_offsets[row + 1U]; ++phase) {
                    const ShadowSpillTransferPhase *item = &program->transfer_phases[phase];
                    if (item->submission_ordered) {
                        work->transfers[action].resource_order =
                            ++work->transfer_resource_completed[item->resource];
                    }
                }
            }
        }
        action_begin = action_end;
    }
    for (uint32_t resource = 0; resource < program->transfer_resource_count; ++resource) {
        work->transfer_resource_completed[resource] = 0;
    }
    return 1;
}

static int resource_free(
    const ShadowSpillSimulationProgram *program,
    const ShadowSpillSimulationWork *work,
    uint32_t resource
) {
    if (!shadowspill_phase_task_free(program, work, resource)) {
        return 0;
    }
    for (uint32_t device = 0; device < program->device_count; ++device) {
        int32_t peers[2] = {work->active_fetch[device], work->active_evict[device]};
        for (uint32_t lane = 0; lane < 2; ++lane) {
            if (peers[lane] < 0) {
                continue;
            }
            const ShadowSpillTransferState *state = &work->transfers[peers[lane]];
            if (state->timed_phase && !state->awaiting_resource_order &&
                program->transfer_phases[state->phase_index].resource == resource) {
                return 0;
            }
        }
    }
    return 1;
}

static void begin_phase(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work,
    uint32_t task
) {
    ShadowSpillTaskState *state = &work->tasks[task];
    const ShadowSpillTaskResourcePhase *phase =
        &program->task_phases[state->task_phase_index];
    state->remaining_work_ns = phase->duration_ns;
    state->progress_ns = work->now_ns;
    state->work_rate = 1;
    state->task_phase_waiting = 1;
    if (phase->resource != UINT32_MAX &&
        (work->task_phase_orders[state->task_phase_index] !=
            work->transfer_resource_completed[phase->resource] + 1U ||
         !resource_free(program, work, phase->resource))) {
        state->end_ns = UINT64_MAX;
        state->stall_mask |= SHADOWSPILL_STALL_TRANSFER_RESOURCE;
        return;
    }
    state->task_phase_waiting = 0;
    state->end_ns = phase->duration_ns > UINT64_MAX - work->now_ns
        ? UINT64_MAX : work->now_ns + phase->duration_ns;
}

void shadowspill_start_task_phases(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work,
    uint32_t task
) {
    if (!has_phases(program, task)) {
        return;
    }
    work->tasks[task].task_phase_index = program->task_phase_offsets[task];
    begin_phase(program, work, task);
}

int shadowspill_advance_task_phase(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work,
    uint32_t task
) {
    if (!has_phases(program, task)) {
        return 0;
    }
    ShadowSpillTaskState *state = &work->tasks[task];
    const ShadowSpillTaskResourcePhase *phase =
        &program->task_phases[state->task_phase_index];
    if (phase->resource != UINT32_MAX) {
        work->transfer_resource_completed[phase->resource] =
            work->task_phase_orders[state->task_phase_index];
    }
    if (state->task_phase_index + 1U == program->task_phase_offsets[task + 1U]) {
        return 0;
    }
    ++state->task_phase_index;
    begin_phase(program, work, task);
    return 1;
}

void shadowspill_refresh_task_phases(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work
) {
    if (program->task_phase_offsets == NULL) {
        return;
    }
    for (uint32_t word = 0; word < work->task_word_count; ++word) {
          uint64_t active = work->active_tasks[word];
          while (active != 0U) {
            uint32_t task = word * 64U + (uint32_t)__builtin_ctzll(active);
            active &= active - 1U;
            if (work->tasks[task].task_phase_waiting) {
                begin_phase(program, work, task);
            }
        }
    }
}
