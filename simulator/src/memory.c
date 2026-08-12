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
    work->transfers = calloc(
        program->action_count == 0U ? 1U : program->action_count,
        sizeof(*work->transfers)
    );
    work->active_fetch = malloc(program->device_count * sizeof(*work->active_fetch));
    work->active_evict = malloc(program->device_count * sizeof(*work->active_evict));
    work->fetch_sequence = calloc(program->device_count, sizeof(*work->fetch_sequence));
    work->evict_sequence = calloc(program->device_count, sizeof(*work->evict_sequence));
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
        work->transfers == NULL || work->active_fetch == NULL ||
        work->active_evict == NULL || work->fetch_sequence == NULL ||
        work->evict_sequence == NULL || work->device_object_bytes == NULL ||
        work->device_workspace_bytes == NULL ||
        work->device_object_peaks == NULL ||
        work->device_workspace_peaks == NULL ||
        work->device_total_peaks == NULL) {
        return 0;
    }
    for (uint32_t index = 0; index < program->device_count; ++index) {
        work->active_fetch[index] = -1;
        work->active_evict[index] = -1;
    }
    return 1;
}

void shadowspill_free_work(ShadowSpillSimulationWork *work) {
    free(work->aliases);
    free(work->tasks);
    free(work->transfers);
    free(work->active_fetch);
    free(work->active_evict);
    free(work->fetch_sequence);
    free(work->evict_sequence);
    free(work->device_object_bytes);
    free(work->device_workspace_bytes);
    free(work->device_object_peaks);
    free(work->device_workspace_peaks);
    free(work->device_total_peaks);
}

void shadowspill_update_peaks(
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

int shadowspill_initialize_memory(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work,
    ShadowSpillSimulationResult *result
) {
    for (uint32_t alias = 0; alias < program->alias_count; ++alias) {
        ShadowSpillAliasState *state = &work->aliases[alias];
        state->device_version = program->alias_initial_version[alias];
        state->host_version = program->alias_initial_version[alias];
        if (program->alias_size_bytes[alias] == 0U) {
            /* Zero-length values carry dependencies but require no residency. */
            state->device_allocated = 1U;
            state->device_ready = 1U;
            continue;
        }
        if (program->alias_retain_spill_copy[alias] != 0U) {
            state->host_allocated = 1U;
            state->host_ready = 1U;
            if (shadowspill_add_overflow_u64(
                    work->host_bytes,
                    program->alias_size_bytes[alias],
                    &work->host_bytes
                )) {
                shadowspill_set_error(
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
            if (shadowspill_add_overflow_u64(
                    work->device_object_bytes[device],
                    program->alias_size_bytes[alias],
                    &work->device_object_bytes[device]
                )) {
                shadowspill_set_error(
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
                if (shadowspill_add_overflow_u64(
                        work->host_bytes,
                        program->alias_size_bytes[alias],
                        &work->host_bytes
                    )) {
                    shadowspill_set_error(
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
    shadowspill_update_peaks(program, work);
    for (uint32_t device = 0; device < program->device_count; ++device) {
        if (work->device_object_bytes[device] >
            program->devices[device].capacity_bytes) {
            shadowspill_set_capacity_error(
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
        shadowspill_set_capacity_error(
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
