#include <stdint.h>

#include "internal.h"

int shadowspill_task_dependencies_complete(
    const ShadowSpillSimulationProgram *program,
    const ShadowSpillSimulationWork *work,
    uint32_t task,
    uint64_t *ready_ns
) {
    uint64_t ready = 0U;
    uint32_t begin = program->dependency_offsets[task];
    uint32_t end = program->dependency_offsets[task + 1U];
    for (uint32_t index = begin; index < end; ++index) {
        uint32_t dependency = program->dependencies[index];
        if (work->tasks[dependency].state != SHADOWSPILL_TASK_COMPLETE) {
            return 0;
        }
        if (work->tasks[dependency].end_ns > ready) {
            ready = work->tasks[dependency].end_ns;
        }
    }
    *ready_ns = ready;
    return 1;
}

static int lane_available(
    const ShadowSpillSimulationProgram *program,
    const ShadowSpillSimulationWork *work,
    uint32_t task
) {
    for (uint32_t other = 0; other < program->task_count; ++other) {
        if (other == task) {
            continue;
        }
        if (work->tasks[other].state == SHADOWSPILL_TASK_ACTIVE &&
            program->task_device[other] == program->task_device[task] &&
            program->task_resource_kind[other] ==
                program->task_resource_kind[task] &&
            program->task_resource_lane[other] ==
                program->task_resource_lane[task]) {
            return 0;
        }
        if (other < task &&
            work->tasks[other].state == SHADOWSPILL_TASK_UNLAUNCHED &&
            program->task_device[other] == program->task_device[task] &&
            program->task_resource_kind[other] ==
                program->task_resource_kind[task] &&
            program->task_resource_lane[other] ==
                program->task_resource_lane[task]) {
            return 0;
        }
    }
    return 1;
}

int shadowspill_inputs_ready(
    const ShadowSpillSimulationProgram *program,
    const ShadowSpillSimulationWork *work,
    uint32_t task
) {
    uint32_t begin = program->input_offsets[task];
    uint32_t end = program->input_offsets[task + 1U];
    for (uint32_t index = begin; index < end; ++index) {
        const ShadowSpillAliasState *state =
            &work->aliases[program->input_aliases[index]];
        if (state->device_ready == 0U || state->h2d_pending != 0U ||
            state->d2h_pending != 0U) {
            return 0;
        }
    }
    return 1;
}

int shadowspill_try_launch_tasks(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work
) {
    int changed = 0;
    for (uint32_t task = 0; task < program->task_count; ++task) {
        ShadowSpillTaskState *state = &work->tasks[task];
        if (state->state != SHADOWSPILL_TASK_UNLAUNCHED ||
            !lane_available(program, work, task)) {
            continue;
        }
        uint64_t dependency_ready = 0U;
        if (!shadowspill_task_dependencies_complete(
                program, work, task, &dependency_ready
            )) {
            continue;
        }
        if (state->ready_set == 0U) {
            state->ready_set = 1U;
            state->ready_ns = work->now_ns > dependency_ready
                ? work->now_ns
                : dependency_ready;
        }
        if (!shadowspill_inputs_ready(program, work, task)) {
            state->stall_mask |= SHADOWSPILL_STALL_INPUT_RESIDENCY;
            continue;
        }
        uint32_t device = program->task_device[task];
        uint64_t output_bytes = 0U;
        uint32_t output_begin = program->output_offsets[task];
        uint32_t output_end = program->output_offsets[task + 1U];
        int output_overflow = 0;
        for (uint32_t index = output_begin; index < output_end; ++index) {
            uint32_t alias = program->output_aliases[index];
            if (work->aliases[alias].device_allocated == 0U &&
                shadowspill_add_overflow_u64(
                    output_bytes,
                    program->alias_size_bytes[alias],
                    &output_bytes
                )) {
                output_overflow = 1;
                break;
            }
        }
        if (output_overflow != 0) {
            state->stall_mask |= SHADOWSPILL_STALL_DEVICE_CAPACITY;
            continue;
        }
        uint64_t used = work->device_object_bytes[device] +
            work->device_workspace_bytes[device];
        uint64_t requested = output_bytes + program->task_workspace_bytes[task];
        uint64_t total = 0U;
        if (shadowspill_add_overflow_u64(used, requested, &total) ||
            total > program->devices[device].capacity_bytes) {
            state->stall_mask |= SHADOWSPILL_STALL_DEVICE_CAPACITY;
            continue;
        }
        for (uint32_t index = output_begin; index < output_end; ++index) {
            uint32_t alias = program->output_aliases[index];
            ShadowSpillAliasState *alias_state = &work->aliases[alias];
            if (alias_state->device_allocated == 0U) {
                alias_state->device_allocated = 1U;
                work->device_object_bytes[device] +=
                    program->alias_size_bytes[alias];
            }
            alias_state->device_ready = 0U;
            alias_state->h2d_pending = 0U;
            alias_state->d2h_pending = 0U;
            alias_state->host_ready = 0U;
        }
        work->device_workspace_bytes[device] +=
            program->task_workspace_bytes[task];
        state->state = SHADOWSPILL_TASK_ACTIVE;
        state->start_ns = work->now_ns;
        if (shadowspill_add_overflow_u64(
                work->now_ns,
                program->task_runtime_ns[task],
                &state->end_ns
            )) {
            state->end_ns = UINT64_MAX;
        }
        shadowspill_update_peaks(program, work);
        changed = 1;
    }
    return changed;
}

static int append_task_interval(
    const ShadowSpillSimulationProgram *program,
    const ShadowSpillSimulationWork *work,
    ShadowSpillSimulationResult *result,
    uint32_t task
) {
    if (result->task_interval_count >= result->task_interval_capacity) {
        return 0;
    }
    const ShadowSpillTaskState *state = &work->tasks[task];
    result->task_intervals[result->task_interval_count++] =
        (ShadowSpillTaskInterval){
            .task = task,
            .ready_ns = state->ready_ns,
            .start_ns = state->start_ns,
            .end_ns = state->end_ns,
            .workspace_bytes = program->task_workspace_bytes[task],
            .stall_mask = state->stall_mask,
        };
    return 1;
}

int shadowspill_complete_task(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work,
    ShadowSpillSimulationResult *result,
    uint32_t task
) {
    ShadowSpillTaskState *task_state = &work->tasks[task];
    uint32_t device = program->task_device[task];
    work->device_workspace_bytes[device] -= program->task_workspace_bytes[task];
    uint32_t output_begin = program->output_offsets[task];
    uint32_t output_end = program->output_offsets[task + 1U];
    for (uint32_t index = output_begin; index < output_end; ++index) {
        uint32_t alias = program->output_aliases[index];
        work->aliases[alias].device_ready = 1U;
        work->aliases[alias].device_version += 1U;
    }
    uint32_t mutation_begin = program->mutation_offsets[task];
    uint32_t mutation_end = program->mutation_offsets[task + 1U];
    for (uint32_t index = mutation_begin; index < mutation_end; ++index) {
        uint32_t alias = program->mutation_aliases[index];
        work->aliases[alias].device_version +=
            program->mutation_version_deltas[index];
        work->aliases[alias].host_ready = 0U;
    }
    task_state->state = SHADOWSPILL_TASK_COMPLETE;
    work->completed_tasks += 1U;
    if (!append_task_interval(program, work, result, task)) {
        shadowspill_set_error(
            result,
            SHADOWSPILL_SIMULATION_INTERNAL_ERROR,
            work,
            task,
            SHADOWSPILL_SIMULATOR_NO_INDEX,
            device
        );
        return 0;
    }
    shadowspill_update_peaks(program, work);
    return shadowspill_submit_ready_actions(program, work, result);
}
