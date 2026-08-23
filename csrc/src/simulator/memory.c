#include <stdint.h>
#include <stdlib.h>

#include "internal.h"

int shadowspill_allocate_work(
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
    work->lane_successors = malloc(
        (program->task_count == 0U ? 1U : program->task_count) *
        sizeof(*work->lane_successors)
    );
    work->task_word_count =
        program->task_count / 64U + (program->task_count % 64U != 0U);
    work->lane_heads = calloc(
        work->task_word_count == 0U ? 1U : work->task_word_count,
        sizeof(*work->lane_heads)
    );
    work->active_tasks = calloc(
        work->task_word_count == 0U ? 1U : work->task_word_count,
        sizeof(*work->active_tasks)
    );
    work->transfers = calloc(
        program->action_count == 0U ? 1U : program->action_count,
        sizeof(*work->transfers)
    );
    work->active_fetch = malloc(program->device_count * sizeof(*work->active_fetch));
    work->active_evict = malloc(program->device_count * sizeof(*work->active_evict));
    work->fetch_cursor = calloc(program->device_count, sizeof(*work->fetch_cursor));
    work->evict_cursor = calloc(program->device_count, sizeof(*work->evict_cursor));
    work->fetch_sequence = calloc(program->device_count, sizeof(*work->fetch_sequence));
    work->evict_sequence = calloc(program->device_count, sizeof(*work->evict_sequence));
    work->device_object_bytes = calloc(
        program->device_count, sizeof(*work->device_object_bytes)
    );
    work->device_workspace_bytes = calloc(
        program->device_count, sizeof(*work->device_workspace_bytes)
    );
    work->device_physical_bytes = calloc(
        program->device_count, sizeof(*work->device_physical_bytes)
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
        work->lane_successors == NULL || work->lane_heads == NULL ||
        work->active_tasks == NULL ||
        work->transfers == NULL || work->active_fetch == NULL ||
        work->active_evict == NULL || work->fetch_cursor == NULL ||
        work->evict_cursor == NULL || work->fetch_sequence == NULL ||
        work->evict_sequence == NULL || work->device_object_bytes == NULL ||
        work->device_workspace_bytes == NULL ||
        work->device_physical_bytes == NULL ||
        work->device_object_peaks == NULL ||
        work->device_workspace_peaks == NULL ||
        work->device_total_peaks == NULL) {
        return 0;
    }
    for (uint32_t index = 0; index < program->device_count; ++index) {
        work->active_fetch[index] = -1;
        work->active_evict[index] = -1;
    }
    for (uint32_t task = 0; task < program->task_count; ++task) {
        work->lane_successors[task] = SHADOWSPILL_SIMULATOR_NO_INDEX;
        uint32_t predecessor = SHADOWSPILL_SIMULATOR_NO_INDEX;
        for (uint32_t previous = task; previous > 0U; --previous) {
            uint32_t candidate = previous - 1U;
            if (program->task_device[candidate] == program->task_device[task] &&
                program->task_resource_kind[candidate] ==
                    program->task_resource_kind[task] &&
                program->task_resource_lane[candidate] ==
                    program->task_resource_lane[task]) {
                predecessor = candidate;
                break;
            }
        }
        if (predecessor == SHADOWSPILL_SIMULATOR_NO_INDEX) {
            work->lane_heads[task >> 6U] |= UINT64_C(1) << (task & 63U);
        } else {
            work->lane_successors[predecessor] = task;
        }
    }
    return 1;
}

void shadowspill_free_work(ShadowSpillSimulationWork *work) {
    free(work->aliases);
    free(work->tasks);
    free(work->lane_successors);
    free(work->lane_heads);
    free(work->active_tasks);
    free(work->transfers);
    free(work->active_fetch);
    free(work->active_evict);
    free(work->fetch_cursor);
    free(work->evict_cursor);
    free(work->fetch_sequence);
    free(work->evict_sequence);
    free(work->device_object_bytes);
    free(work->device_workspace_bytes);
    free(work->device_physical_bytes);
    free(work->device_object_peaks);
    free(work->device_workspace_peaks);
    free(work->device_total_peaks);
}

void shadowspill_update_peaks(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work
) {
    for (uint32_t device = 0; device < program->device_count; ++device) {
        uint64_t total = shadowspill_device_used_bytes(program, work, device);
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
    if (work->spill_bytes > work->spill_peak_bytes) {
        work->spill_peak_bytes = work->spill_bytes;
    }
}

int shadowspill_initialize_memory(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work,
    ShadowSpillSimulationResult *result
) {
    for (uint32_t alias = 0; alias < program->alias_count; ++alias) {
        ShadowSpillAliasState *state = &work->aliases[alias];
        state->device_version = program->alias_initial_version[alias];
        state->spill_version = program->alias_initial_version[alias];
        if (program->alias_size_bytes[alias] == 0U) {
            /* Zero-length values carry dependencies but require no residency. */
            state->device_allocated = 1U;
            state->device_ready = 1U;
            continue;
        }
        if (program->alias_retain_spill_copy[alias] != 0U) {
            state->spill_allocated = 1U;
            state->spill_ready = 1U;
            if (shadowspill_add_overflow_u64(
                    work->spill_bytes,
                    program->alias_size_bytes[alias],
                    &work->spill_bytes
                )) {
                shadowspill_set_error(
                    result,
                    SHADOWSPILL_STATUS_INVALID_ARGUMENT,
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
            if (shadowspill_add_overflow_u64(
                    work->device_object_bytes[device],
                    program->alias_size_bytes[alias],
                    &work->device_object_bytes[device]
                )) {
                shadowspill_set_error(
                    result,
                    SHADOWSPILL_STATUS_INVALID_ARGUMENT,
                    work,
                    SHADOWSPILL_SIMULATOR_NO_INDEX,
                    alias,
                    device
                );
                return 0;
            }
        } else {
            if (state->spill_allocated == 0U) {
                state->spill_allocated = 1U;
                if (shadowspill_add_overflow_u64(
                        work->spill_bytes,
                        program->alias_size_bytes[alias],
                        &work->spill_bytes
                    )) {
                    shadowspill_set_error(
                        result,
                        SHADOWSPILL_STATUS_INVALID_ARGUMENT,
                        work,
                        SHADOWSPILL_SIMULATOR_NO_INDEX,
                        alias,
                        SHADOWSPILL_SIMULATOR_NO_INDEX
                    );
                    return 0;
                }
            }
            state->spill_ready = 1U;
        }
    }
    if (program->use_admission_accounting != 0U) {
        for (uint32_t device = 0; device < program->device_count; ++device) {
            work->device_physical_bytes[device] =
                program->initial_physical_bytes[device];
        }
    } else {
        for (uint32_t device = 0; device < program->device_count; ++device) {
            work->device_physical_bytes[device] =
                work->device_object_bytes[device];
        }
    }
    shadowspill_update_peaks(program, work);
    for (uint32_t device = 0; device < program->device_count; ++device) {
        uint64_t used = shadowspill_device_used_bytes(program, work, device);
        if (used >
            program->devices[device].capacity_bytes) {
            shadowspill_set_capacity_error(
                result,
                SHADOWSPILL_STATUS_INITIAL_DEVICE_CAPACITY,
                work,
                SHADOWSPILL_SIMULATOR_NO_INDEX,
                SHADOWSPILL_SIMULATOR_NO_INDEX,
                device,
                SHADOWSPILL_MEMORY_DEVICE,
                program->devices[device].capacity_bytes,
                used,
                0U
            );
            return 0;
        }
    }
    if (work->spill_bytes > program->spill_capacity_bytes) {
        shadowspill_set_capacity_error(
            result,
            SHADOWSPILL_STATUS_INITIAL_SPILL_CAPACITY,
            work,
            SHADOWSPILL_SIMULATOR_NO_INDEX,
            SHADOWSPILL_SIMULATOR_NO_INDEX,
            SHADOWSPILL_SIMULATOR_NO_INDEX,
            SHADOWSPILL_MEMORY_SPILL,
            program->spill_capacity_bytes,
            work->spill_bytes,
            0U
        );
        return 0;
    }
    return 1;
}
