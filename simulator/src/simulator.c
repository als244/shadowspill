#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "internal.h"

enum {
    TASK_UNLAUNCHED = 0,
    TASK_ACTIVE = 1,
    TASK_COMPLETE = 2,
    TRANSFER_UNUSED = 0,
    TRANSFER_QUEUED = 1,
    TRANSFER_ACTIVE = 2,
    TRANSFER_COMPLETE = 3,
};

static int add_overflow_u64(uint64_t left, uint64_t right, uint64_t *result) {
    if (right > UINT64_MAX - left) {
        return 1;
    }
    *result = left + right;
    return 0;
}

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

static int require_pointer(const void *pointer, uint32_t count) {
    return count == 0U || pointer != NULL;
}

static int validate_offsets(
    const uint32_t *offsets,
    uint32_t row_count,
    uint32_t value_count
) {
    if (!require_pointer(offsets, row_count + 1U) || offsets[0] != 0U) {
        return 0;
    }
    for (uint32_t index = 0; index < row_count; ++index) {
        if (offsets[index] > offsets[index + 1U]) {
            return 0;
        }
    }
    return offsets[row_count] == value_count;
}

static int validate_program(const ShadowSpillSimulationProgram *program) {
    if (program == NULL ||
        program->abi_version != SHADOWSPILL_SIMULATOR_ABI_VERSION ||
        program->device_count == 0U) {
        return 0;
    }
    if (!require_pointer(program->devices, program->device_count) ||
        !require_pointer(program->alias_device, program->alias_count) ||
        !require_pointer(program->alias_size_bytes, program->alias_count) ||
        !require_pointer(program->alias_initial_version, program->alias_count) ||
        !require_pointer(program->alias_retain_host_backing, program->alias_count) ||
        !require_pointer(program->task_device, program->task_count) ||
        !require_pointer(program->task_resource_kind, program->task_count) ||
        !require_pointer(program->task_resource_lane, program->task_count) ||
        !require_pointer(program->task_runtime_ns, program->task_count) ||
        !require_pointer(program->task_workspace_bytes, program->task_count) ||
        !require_pointer(program->dependencies, program->dependency_count) ||
        !require_pointer(program->input_aliases, program->input_count) ||
        !require_pointer(program->output_aliases, program->output_count) ||
        !require_pointer(program->mutation_aliases, program->mutation_count) ||
        !require_pointer(
            program->mutation_version_deltas, program->mutation_count
        ) ||
        !require_pointer(program->action_trigger_tasks, program->action_count) ||
        !require_pointer(program->action_aliases, program->action_count) ||
        !require_pointer(program->action_kinds, program->action_count) ||
        !require_pointer(program->initial_aliases, program->initial_count) ||
        !require_pointer(program->initial_locations, program->initial_count) ||
        !require_pointer(program->final_aliases, program->final_count) ||
        !require_pointer(program->final_locations, program->final_count)) {
        return 0;
    }
    if (!validate_offsets(
            program->dependency_offsets,
            program->task_count,
            program->dependency_count
        ) ||
        !validate_offsets(
            program->input_offsets, program->task_count, program->input_count
        ) ||
        !validate_offsets(
            program->output_offsets, program->task_count, program->output_count
        ) ||
        !validate_offsets(
            program->mutation_offsets,
            program->task_count,
            program->mutation_count
        )) {
        return 0;
    }
    for (uint32_t index = 0; index < program->device_count; ++index) {
        if (program->devices[index].h2d_bandwidth_bytes_per_second == 0U ||
            program->devices[index].d2h_bandwidth_bytes_per_second == 0U) {
            return 0;
        }
    }
    for (uint32_t index = 0; index < program->alias_count; ++index) {
        if (program->alias_device[index] >= program->device_count) {
            return 0;
        }
    }
    for (uint32_t index = 0; index < program->task_count; ++index) {
        if (program->task_device[index] >= program->device_count) {
            return 0;
        }
    }
    for (uint32_t index = 0; index < program->dependency_count; ++index) {
        if (program->dependencies[index] >= program->task_count) {
            return 0;
        }
    }
    for (uint32_t index = 0; index < program->input_count; ++index) {
        if (program->input_aliases[index] >= program->alias_count) {
            return 0;
        }
    }
    for (uint32_t index = 0; index < program->output_count; ++index) {
        if (program->output_aliases[index] >= program->alias_count) {
            return 0;
        }
    }
    for (uint32_t index = 0; index < program->mutation_count; ++index) {
        if (program->mutation_aliases[index] >= program->alias_count) {
            return 0;
        }
    }
    for (uint32_t index = 0; index < program->action_count; ++index) {
        if (program->action_trigger_tasks[index] >= program->task_count ||
            program->action_aliases[index] >= program->alias_count ||
            program->action_kinds[index] > SHADOWSPILL_MEMORY_PREFETCH) {
            return 0;
        }
    }
    for (uint32_t index = 0; index < program->initial_count; ++index) {
        if (program->initial_aliases[index] >= program->alias_count ||
            program->initial_locations[index] > SHADOWSPILL_MEMORY_HOST) {
            return 0;
        }
    }
    for (uint32_t index = 0; index < program->final_count; ++index) {
        if (program->final_aliases[index] >= program->alias_count ||
            program->final_locations[index] > SHADOWSPILL_MEMORY_HOST) {
            return 0;
        }
    }
    return 1;
}

static void initialize_result(ShadowSpillSimulationResult *result) {
    ShadowSpillTaskInterval *task_intervals = result->task_intervals;
    uint32_t task_capacity = result->task_interval_capacity;
    ShadowSpillTransferInterval *transfer_intervals = result->transfer_intervals;
    uint32_t transfer_capacity = result->transfer_interval_capacity;
    ShadowSpillDevicePeak *device_peaks = result->device_peaks;
    uint32_t device_capacity = result->device_peak_capacity;
    memset(result, 0, sizeof(*result));
    result->task_intervals = task_intervals;
    result->task_interval_capacity = task_capacity;
    result->transfer_intervals = transfer_intervals;
    result->transfer_interval_capacity = transfer_capacity;
    result->device_peaks = device_peaks;
    result->device_peak_capacity = device_capacity;
    result->error_task = SHADOWSPILL_SIMULATOR_NO_INDEX;
    result->error_alias = SHADOWSPILL_SIMULATOR_NO_INDEX;
    result->error_device = SHADOWSPILL_SIMULATOR_NO_INDEX;
}

static void set_error(
    ShadowSpillSimulationResult *result,
    ShadowSpillSimulationStatus status,
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

static void set_capacity_error(
    ShadowSpillSimulationResult *result,
    ShadowSpillSimulationStatus status,
    const ShadowSpillSimulationWork *work,
    uint32_t task,
    uint32_t alias,
    uint32_t device,
    uint8_t location,
    uint64_t capacity,
    uint64_t used,
    uint64_t requested
) {
    set_error(result, status, work, task, alias, device);
    result->error_location = location;
    result->error_capacity_bytes = capacity;
    result->error_used_bytes = used;
    result->error_requested_bytes = requested;
}

static int allocate_work(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work
) {
    work->aliases = calloc(
        program->alias_count == 0U ? 1U : program->alias_count,
        sizeof(*work->aliases)
    );
    work->tasks = calloc(
        program->task_count == 0U ? 1U : program->task_count,
        sizeof(*work->tasks)
    );
    work->transfers = calloc(
        program->action_count == 0U ? 1U : program->action_count,
        sizeof(*work->transfers)
    );
    work->active_h2d = malloc(program->device_count * sizeof(*work->active_h2d));
    work->active_d2h = malloc(program->device_count * sizeof(*work->active_d2h));
    work->h2d_sequence = calloc(program->device_count, sizeof(*work->h2d_sequence));
    work->d2h_sequence = calloc(program->device_count, sizeof(*work->d2h_sequence));
    work->device_object_bytes = calloc(
        program->device_count, sizeof(*work->device_object_bytes)
    );
    work->device_workspace_bytes = calloc(
        program->device_count, sizeof(*work->device_workspace_bytes)
    );
    work->device_object_peaks = calloc(
        program->device_count, sizeof(*work->device_object_peaks)
    );
    work->device_workspace_peaks = calloc(
        program->device_count, sizeof(*work->device_workspace_peaks)
    );
    work->device_total_peaks = calloc(
        program->device_count, sizeof(*work->device_total_peaks)
    );
    if (work->aliases == NULL || work->tasks == NULL ||
        work->transfers == NULL || work->active_h2d == NULL ||
        work->active_d2h == NULL || work->h2d_sequence == NULL ||
        work->d2h_sequence == NULL || work->device_object_bytes == NULL ||
        work->device_workspace_bytes == NULL ||
        work->device_object_peaks == NULL ||
        work->device_workspace_peaks == NULL ||
        work->device_total_peaks == NULL) {
        return 0;
    }
    for (uint32_t index = 0; index < program->device_count; ++index) {
        work->active_h2d[index] = -1;
        work->active_d2h[index] = -1;
    }
    return 1;
}

static void free_work(ShadowSpillSimulationWork *work) {
    free(work->aliases);
    free(work->tasks);
    free(work->transfers);
    free(work->active_h2d);
    free(work->active_d2h);
    free(work->h2d_sequence);
    free(work->d2h_sequence);
    free(work->device_object_bytes);
    free(work->device_workspace_bytes);
    free(work->device_object_peaks);
    free(work->device_workspace_peaks);
    free(work->device_total_peaks);
}

static void update_peaks(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work
) {
    for (uint32_t device = 0; device < program->device_count; ++device) {
        uint64_t total = work->device_object_bytes[device] +
            work->device_workspace_bytes[device];
        if (work->device_object_bytes[device] >
            work->device_object_peaks[device]) {
            work->device_object_peaks[device] =
                work->device_object_bytes[device];
        }
        if (work->device_workspace_bytes[device] >
            work->device_workspace_peaks[device]) {
            work->device_workspace_peaks[device] =
                work->device_workspace_bytes[device];
        }
        if (total > work->device_total_peaks[device]) {
            work->device_total_peaks[device] = total;
        }
    }
    if (work->host_bytes > work->host_peak_bytes) {
        work->host_peak_bytes = work->host_bytes;
    }
}

static int initialize_memory(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work,
    ShadowSpillSimulationResult *result
) {
    for (uint32_t alias = 0; alias < program->alias_count; ++alias) {
        ShadowSpillAliasState *state = &work->aliases[alias];
        state->device_version = program->alias_initial_version[alias];
        state->host_version = program->alias_initial_version[alias];
        if (program->alias_retain_host_backing[alias] != 0U) {
            state->host_allocated = 1U;
            state->host_ready = 1U;
            if (add_overflow_u64(
                    work->host_bytes,
                    program->alias_size_bytes[alias],
                    &work->host_bytes
                )) {
                set_error(
                    result,
                    SHADOWSPILL_SIMULATION_INVALID_ARGUMENT,
                    work,
                    SHADOWSPILL_SIMULATOR_NO_INDEX,
                    alias,
                    SHADOWSPILL_SIMULATOR_NO_INDEX
                );
                return 0;
            }
        }
    }
    for (uint32_t index = 0; index < program->initial_count; ++index) {
        uint32_t alias = program->initial_aliases[index];
        uint32_t device = program->alias_device[alias];
        ShadowSpillAliasState *state = &work->aliases[alias];
        if (program->initial_locations[index] == SHADOWSPILL_MEMORY_DEVICE) {
            state->device_allocated = 1U;
            state->device_ready = 1U;
            if (add_overflow_u64(
                    work->device_object_bytes[device],
                    program->alias_size_bytes[alias],
                    &work->device_object_bytes[device]
                )) {
                set_error(
                    result,
                    SHADOWSPILL_SIMULATION_INVALID_ARGUMENT,
                    work,
                    SHADOWSPILL_SIMULATOR_NO_INDEX,
                    alias,
                    device
                );
                return 0;
            }
        } else {
            if (state->host_allocated == 0U) {
                state->host_allocated = 1U;
                if (add_overflow_u64(
                        work->host_bytes,
                        program->alias_size_bytes[alias],
                        &work->host_bytes
                    )) {
                    set_error(
                        result,
                        SHADOWSPILL_SIMULATION_INVALID_ARGUMENT,
                        work,
                        SHADOWSPILL_SIMULATOR_NO_INDEX,
                        alias,
                        SHADOWSPILL_SIMULATOR_NO_INDEX
                    );
                    return 0;
                }
            }
            state->host_ready = 1U;
        }
    }
    update_peaks(program, work);
    for (uint32_t device = 0; device < program->device_count; ++device) {
        if (work->device_object_bytes[device] >
            program->devices[device].capacity_bytes) {
            set_capacity_error(
                result,
                SHADOWSPILL_SIMULATION_INITIAL_DEVICE_CAPACITY,
                work,
                SHADOWSPILL_SIMULATOR_NO_INDEX,
                SHADOWSPILL_SIMULATOR_NO_INDEX,
                device,
                SHADOWSPILL_MEMORY_DEVICE,
                program->devices[device].capacity_bytes,
                work->device_object_bytes[device],
                0U
            );
            return 0;
        }
    }
    if (work->host_bytes > program->host_capacity_bytes) {
        set_capacity_error(
            result,
            SHADOWSPILL_SIMULATION_INITIAL_HOST_CAPACITY,
            work,
            SHADOWSPILL_SIMULATOR_NO_INDEX,
            SHADOWSPILL_SIMULATOR_NO_INDEX,
            SHADOWSPILL_SIMULATOR_NO_INDEX,
            SHADOWSPILL_MEMORY_HOST,
            program->host_capacity_bytes,
            work->host_bytes,
            0U
        );
        return 0;
    }
    return 1;
}

static int task_dependencies_complete(
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
        if (work->tasks[dependency].state != TASK_COMPLETE) {
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
        if (work->tasks[other].state == TASK_ACTIVE &&
            program->task_device[other] == program->task_device[task] &&
            program->task_resource_kind[other] ==
                program->task_resource_kind[task] &&
            program->task_resource_lane[other] ==
                program->task_resource_lane[task]) {
            return 0;
        }
        if (other < task && work->tasks[other].state == TASK_UNLAUNCHED &&
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

static int inputs_ready(
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

static int try_launch_tasks(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work
) {
    int changed = 0;
    for (uint32_t task = 0; task < program->task_count; ++task) {
        ShadowSpillTaskState *state = &work->tasks[task];
        if (state->state != TASK_UNLAUNCHED ||
            !lane_available(program, work, task)) {
            continue;
        }
        uint64_t dependency_ready = 0U;
        if (!task_dependencies_complete(
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
        if (!inputs_ready(program, work, task)) {
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
                add_overflow_u64(
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
        if (add_overflow_u64(used, requested, &total) ||
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
        state->state = TASK_ACTIVE;
        state->start_ns = work->now_ns;
        if (add_overflow_u64(
                work->now_ns,
                program->task_runtime_ns[task],
                &state->end_ns
            )) {
            state->end_ns = UINT64_MAX;
        }
        update_peaks(program, work);
        changed = 1;
    }
    return changed;
}

static uint64_t transfer_runtime_ns(
    const ShadowSpillSimulationProgram *program,
    uint32_t alias,
    uint8_t direction
) {
    uint32_t device = program->alias_device[alias];
    const ShadowSpillSimulationDevice *config = &program->devices[device];
    uint64_t bandwidth = direction == SHADOWSPILL_TRANSFER_HOST_TO_DEVICE
        ? config->h2d_bandwidth_bytes_per_second
        : config->d2h_bandwidth_bytes_per_second;
    uint64_t latency = direction == SHADOWSPILL_TRANSFER_HOST_TO_DEVICE
        ? config->h2d_latency_ns
        : config->d2h_latency_ns;
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
    if (seconds_ns == UINT64_MAX || add_overflow_u64(
            seconds_ns, partial, &runtime
        ) || add_overflow_u64(runtime, latency, &runtime)) {
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
        if (candidate->state == TRANSFER_QUEUED &&
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
    int32_t *active = direction == SHADOWSPILL_TRANSFER_HOST_TO_DEVICE
        ? &work->active_h2d[device]
        : &work->active_d2h[device];
    if (*active >= 0) {
        return 0;
    }
    for (uint32_t index = 0; index < program->action_count; ++index) {
        ShadowSpillTransferState *transfer = &work->transfers[index];
        if (transfer->state != TRANSFER_QUEUED ||
            transfer->device != device || transfer->direction != direction ||
            earlier_queued_transfer(program, work, index, device, direction)) {
            continue;
        }
        uint32_t alias = transfer->alias;
        ShadowSpillAliasState *state = &work->aliases[alias];
        if (direction == SHADOWSPILL_TRANSFER_HOST_TO_DEVICE) {
            if (state->d2h_pending != 0U || state->host_ready == 0U) {
                transfer->stall_mask |= SHADOWSPILL_STALL_SOURCE_READINESS;
                return 0;
            }
            if (state->device_allocated == 0U) {
                uint64_t used = work->device_object_bytes[device] +
                    work->device_workspace_bytes[device];
                uint64_t total = 0U;
                if (add_overflow_u64(
                        used, program->alias_size_bytes[alias], &total
                    ) || total > program->devices[device].capacity_bytes) {
                    transfer->stall_mask |= SHADOWSPILL_STALL_DEVICE_CAPACITY;
                    return 0;
                }
                state->device_allocated = 1U;
                state->device_ready = 0U;
                work->device_object_bytes[device] +=
                    program->alias_size_bytes[alias];
            }
        } else {
            if (state->device_ready == 0U) {
                transfer->stall_mask |= SHADOWSPILL_STALL_SOURCE_READINESS;
                return 0;
            }
            if (state->host_allocated == 0U) {
                uint64_t total = 0U;
                if (add_overflow_u64(
                        work->host_bytes,
                        program->alias_size_bytes[alias],
                        &total
                    ) || total > program->host_capacity_bytes) {
                    transfer->stall_mask |= SHADOWSPILL_STALL_HOST_CAPACITY;
                    return 0;
                }
                state->host_allocated = 1U;
                state->host_ready = 0U;
                work->host_bytes = total;
            }
        }
        transfer->state = TRANSFER_ACTIVE;
        transfer->start_ns = work->now_ns;
        uint64_t runtime = transfer_runtime_ns(program, alias, direction);
        if (add_overflow_u64(work->now_ns, runtime, &transfer->end_ns)) {
            transfer->end_ns = UINT64_MAX;
        }
        *active = (int32_t)index;
        update_peaks(program, work);
        return 1;
    }
    return 0;
}

static int try_start_transfers(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work
) {
    int changed = 0;
    for (uint32_t device = 0; device < program->device_count; ++device) {
        changed |= try_start_direction(
            program, work, device, SHADOWSPILL_TRANSFER_HOST_TO_DEVICE
        );
        changed |= try_start_direction(
            program, work, device, SHADOWSPILL_TRANSFER_DEVICE_TO_HOST
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
            set_error(
                result,
                SHADOWSPILL_SIMULATION_INVALID_RELEASE,
                work,
                task,
                alias,
                device
            );
            return 0;
        }
        if (state->h2d_pending != 0U || state->d2h_pending != 0U) {
            set_error(
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
            program->alias_retain_host_backing[alias] == 0U) {
            state->host_allocated = 0U;
            state->host_ready = 0U;
            work->host_bytes -= program->alias_size_bytes[alias];
        }
        update_peaks(program, work);
        return 1;
    }
    ShadowSpillTransferState *transfer = &work->transfers[action];
    transfer->alias = alias;
    transfer->trigger_task = task;
    transfer->device = device;
    transfer->ready_ns = work->now_ns;
    if (kind == SHADOWSPILL_MEMORY_OFFLOAD) {
        if (state->device_allocated == 0U || state->device_ready == 0U) {
            set_error(
                result,
                SHADOWSPILL_SIMULATION_INVALID_OFFLOAD,
                work,
                task,
                alias,
                device
            );
            return 0;
        }
        transfer->direction = SHADOWSPILL_TRANSFER_DEVICE_TO_HOST;
        transfer->sequence = work->d2h_sequence[device]++;
        state->d2h_pending = 1U;
    } else {
        if ((state->device_allocated != 0U && state->d2h_pending == 0U) ||
            (state->host_ready == 0U && state->d2h_pending == 0U)) {
            set_error(
                result,
                SHADOWSPILL_SIMULATION_INVALID_PREFETCH,
                work,
                task,
                alias,
                device
            );
            return 0;
        }
        transfer->direction = SHADOWSPILL_TRANSFER_HOST_TO_DEVICE;
        transfer->sequence = work->h2d_sequence[device]++;
        state->h2d_pending = 1U;
    }
    transfer->state = TRANSFER_QUEUED;
    return 1;
}

static int submit_ready_actions(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work,
    ShadowSpillSimulationResult *result
) {
    while (work->submitted_actions < program->action_count) {
        uint32_t action = work->submitted_actions;
        uint32_t trigger = program->action_trigger_tasks[action];
        if (work->tasks[trigger].state != TASK_COMPLETE) {
            break;
        }
        if (!submit_action(program, work, result, action)) {
            return 0;
        }
        work->submitted_actions += 1U;
    }
    return 1;
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

static int complete_task(
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
    task_state->state = TASK_COMPLETE;
    work->completed_tasks += 1U;
    if (!append_task_interval(program, work, result, task)) {
        set_error(
            result,
            SHADOWSPILL_SIMULATION_INTERNAL_ERROR,
            work,
            task,
            SHADOWSPILL_SIMULATOR_NO_INDEX,
            device
        );
        return 0;
    }
    update_peaks(program, work);
    return submit_ready_actions(program, work, result);
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

static int complete_transfer(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work,
    ShadowSpillSimulationResult *result,
    uint32_t device,
    uint8_t direction
) {
    int32_t *active = direction == SHADOWSPILL_TRANSFER_HOST_TO_DEVICE
        ? &work->active_h2d[device]
        : &work->active_d2h[device];
    uint32_t index = (uint32_t)*active;
    ShadowSpillTransferState *transfer = &work->transfers[index];
    ShadowSpillAliasState *state = &work->aliases[transfer->alias];
    if (direction == SHADOWSPILL_TRANSFER_HOST_TO_DEVICE) {
        state->device_ready = 1U;
        state->device_version = state->host_version;
        state->h2d_pending = 0U;
        if (program->alias_retain_host_backing[transfer->alias] == 0U) {
            state->host_allocated = 0U;
            state->host_ready = 0U;
            work->host_bytes -= program->alias_size_bytes[transfer->alias];
        }
    } else {
        state->host_ready = 1U;
        state->host_version = state->device_version;
        state->d2h_pending = 0U;
        state->device_allocated = 0U;
        state->device_ready = 0U;
        work->device_object_bytes[device] -=
            program->alias_size_bytes[transfer->alias];
    }
    transfer->state = TRANSFER_COMPLETE;
    *active = -1;
    if (!append_transfer_interval(program, work, result, index)) {
        set_error(
            result,
            SHADOWSPILL_SIMULATION_INTERNAL_ERROR,
            work,
            transfer->trigger_task,
            transfer->alias,
            device
        );
        return 0;
    }
    update_peaks(program, work);
    return 1;
}

static uint64_t next_event_time(
    const ShadowSpillSimulationProgram *program,
    const ShadowSpillSimulationWork *work
) {
    uint64_t next = UINT64_MAX;
    for (uint32_t task = 0; task < program->task_count; ++task) {
        if (work->tasks[task].state == TASK_ACTIVE &&
            work->tasks[task].end_ns < next) {
            next = work->tasks[task].end_ns;
        }
    }
    for (uint32_t device = 0; device < program->device_count; ++device) {
        if (work->active_h2d[device] >= 0) {
            uint32_t index = (uint32_t)work->active_h2d[device];
            if (work->transfers[index].end_ns < next) {
                next = work->transfers[index].end_ns;
            }
        }
        if (work->active_d2h[device] >= 0) {
            uint32_t index = (uint32_t)work->active_d2h[device];
            if (work->transfers[index].end_ns < next) {
                next = work->transfers[index].end_ns;
            }
        }
    }
    return next;
}

static int complete_events(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work,
    ShadowSpillSimulationResult *result
) {
    for (uint32_t device = 0; device < program->device_count; ++device) {
        if (work->active_h2d[device] >= 0) {
            uint32_t index = (uint32_t)work->active_h2d[device];
            if (work->transfers[index].end_ns == work->now_ns &&
                !complete_transfer(
                    program,
                    work,
                    result,
                    device,
                    SHADOWSPILL_TRANSFER_HOST_TO_DEVICE
                )) {
                return 0;
            }
        }
    }
    for (uint32_t device = 0; device < program->device_count; ++device) {
        if (work->active_d2h[device] >= 0) {
            uint32_t index = (uint32_t)work->active_d2h[device];
            if (work->transfers[index].end_ns == work->now_ns &&
                !complete_transfer(
                    program,
                    work,
                    result,
                    device,
                    SHADOWSPILL_TRANSFER_DEVICE_TO_HOST
                )) {
                return 0;
            }
        }
    }
    for (uint32_t task = 0; task < program->task_count; ++task) {
        if (work->tasks[task].state == TASK_ACTIVE &&
            work->tasks[task].end_ns == work->now_ns &&
            !complete_task(program, work, result, task)) {
            return 0;
        }
    }
    return 1;
}

static int has_pending_work(
    const ShadowSpillSimulationProgram *program,
    const ShadowSpillSimulationWork *work
) {
    if (work->completed_tasks < program->task_count ||
        work->submitted_actions < program->action_count) {
        return 1;
    }
    for (uint32_t index = 0; index < program->action_count; ++index) {
        if (work->transfers[index].state == TRANSFER_QUEUED ||
            work->transfers[index].state == TRANSFER_ACTIVE) {
            return 1;
        }
    }
    return 0;
}

static int report_deadlock(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work,
    ShadowSpillSimulationResult *result
) {
    for (uint32_t index = 0; index < program->action_count; ++index) {
        ShadowSpillTransferState *transfer = &work->transfers[index];
        if (transfer->state != TRANSFER_QUEUED) {
            continue;
        }
        uint32_t alias = transfer->alias;
        uint32_t device = transfer->device;
        ShadowSpillAliasState *state = &work->aliases[alias];
        if (transfer->direction == SHADOWSPILL_TRANSFER_HOST_TO_DEVICE &&
            state->device_allocated == 0U) {
            uint64_t used = work->device_object_bytes[device] +
                work->device_workspace_bytes[device];
            uint64_t total = 0U;
            if (add_overflow_u64(
                    used, program->alias_size_bytes[alias], &total
                ) || total > program->devices[device].capacity_bytes) {
                set_capacity_error(
                    result,
                    SHADOWSPILL_SIMULATION_PREFETCH_DEVICE_CAPACITY,
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
        if (transfer->direction == SHADOWSPILL_TRANSFER_DEVICE_TO_HOST &&
            state->host_allocated == 0U) {
            uint64_t total = 0U;
            if (add_overflow_u64(
                    work->host_bytes,
                    program->alias_size_bytes[alias],
                    &total
                ) || total > program->host_capacity_bytes) {
                set_capacity_error(
                    result,
                    SHADOWSPILL_SIMULATION_OFFLOAD_HOST_CAPACITY,
                    work,
                    transfer->trigger_task,
                    alias,
                    device,
                    SHADOWSPILL_MEMORY_HOST,
                    program->host_capacity_bytes,
                    work->host_bytes,
                    program->alias_size_bytes[alias]
                );
                return 0;
            }
        }
    }
    for (uint32_t task = 0; task < program->task_count; ++task) {
        ShadowSpillTaskState *state = &work->tasks[task];
        if (state->state != TASK_UNLAUNCHED) {
            continue;
        }
        uint64_t dependency_ready = 0U;
        if (!task_dependencies_complete(
                program, work, task, &dependency_ready
            )) {
            continue;
        }
        uint32_t input_begin = program->input_offsets[task];
        uint32_t input_end = program->input_offsets[task + 1U];
        for (uint32_t index = input_begin; index < input_end; ++index) {
            uint32_t alias = program->input_aliases[index];
            if (work->aliases[alias].device_ready == 0U ||
                work->aliases[alias].h2d_pending != 0U ||
                work->aliases[alias].d2h_pending != 0U) {
                set_error(
                    result,
                    SHADOWSPILL_SIMULATION_TASK_INPUT_DEADLOCK,
                    work,
                    task,
                    alias,
                    program->task_device[task]
                );
                return 0;
            }
        }
        uint32_t device = program->task_device[task];
        uint64_t requested = program->task_workspace_bytes[task];
        uint32_t output_begin = program->output_offsets[task];
        uint32_t output_end = program->output_offsets[task + 1U];
        for (uint32_t index = output_begin; index < output_end; ++index) {
            uint32_t alias = program->output_aliases[index];
            if (work->aliases[alias].device_allocated == 0U) {
                requested += program->alias_size_bytes[alias];
            }
        }
        uint64_t used = work->device_object_bytes[device] +
            work->device_workspace_bytes[device];
        if (requested > program->devices[device].capacity_bytes -
            (used > program->devices[device].capacity_bytes
                ? program->devices[device].capacity_bytes
                : used)) {
            set_capacity_error(
                result,
                SHADOWSPILL_SIMULATION_TASK_DEVICE_CAPACITY,
                work,
                task,
                output_begin < output_end
                    ? program->output_aliases[output_begin]
                    : SHADOWSPILL_SIMULATOR_NO_INDEX,
                device,
                SHADOWSPILL_MEMORY_DEVICE,
                program->devices[device].capacity_bytes,
                used,
                requested
            );
            return 0;
        }
    }
    set_error(
        result,
        SHADOWSPILL_SIMULATION_TRANSFER_DEADLOCK,
        work,
        SHADOWSPILL_SIMULATOR_NO_INDEX,
        SHADOWSPILL_SIMULATOR_NO_INDEX,
        SHADOWSPILL_SIMULATOR_NO_INDEX
    );
    return 0;
}

static int check_final_residency(
    const ShadowSpillSimulationProgram *program,
    const ShadowSpillSimulationWork *work,
    ShadowSpillSimulationResult *result
) {
    for (uint32_t index = 0; index < program->final_count; ++index) {
        uint32_t alias = program->final_aliases[index];
        const ShadowSpillAliasState *state = &work->aliases[alias];
        int ready = program->final_locations[index] == SHADOWSPILL_MEMORY_DEVICE
            ? state->device_ready != 0U
            : state->host_ready != 0U;
        if (!ready) {
            set_error(
                result,
                SHADOWSPILL_SIMULATION_FINAL_RESIDENCY,
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

ShadowSpillSimulationStatus shadowspill_simulate(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationResult *result
) {
    if (result == NULL) {
        return SHADOWSPILL_SIMULATION_INVALID_ARGUMENT;
    }
    initialize_result(result);
    if (!validate_program(program) ||
        result->task_interval_capacity < program->task_count ||
        result->transfer_interval_capacity < program->action_count ||
        result->device_peak_capacity < program->device_count ||
        !require_pointer(result->task_intervals, program->task_count) ||
        !require_pointer(result->transfer_intervals, program->action_count) ||
        !require_pointer(result->device_peaks, program->device_count)) {
        result->status = SHADOWSPILL_SIMULATION_INVALID_ARGUMENT;
        return SHADOWSPILL_SIMULATION_INVALID_ARGUMENT;
    }
    ShadowSpillSimulationWork work = {0};
    if (!allocate_work(program, &work)) {
        free_work(&work);
        result->status = SHADOWSPILL_SIMULATION_ALLOCATION_FAILURE;
        return SHADOWSPILL_SIMULATION_ALLOCATION_FAILURE;
    }
    if (!initialize_memory(program, &work, result)) {
        free_work(&work);
        return (ShadowSpillSimulationStatus)result->status;
    }
    while (has_pending_work(program, &work)) {
        int changed = 1;
        while (changed != 0) {
            changed = try_start_transfers(program, &work);
            changed |= try_launch_tasks(program, &work);
        }
        uint64_t next = next_event_time(program, &work);
        if (next == UINT64_MAX) {
            report_deadlock(program, &work, result);
            free_work(&work);
            return (ShadowSpillSimulationStatus)result->status;
        }
        work.now_ns = next;
        if (!complete_events(program, &work, result)) {
            free_work(&work);
            return (ShadowSpillSimulationStatus)result->status;
        }
    }
    if (!check_final_residency(program, &work, result)) {
        free_work(&work);
        return (ShadowSpillSimulationStatus)result->status;
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
    free_work(&work);
    return SHADOWSPILL_SIMULATION_OK;
}
