#include <stdint.h>
#include <math.h>

#include "internal.h"

int shadowspill_add_overflow_u64(
    uint64_t left,
    uint64_t right,
    uint64_t *result
) {
    if (right > UINT64_MAX - left) {
        return 1;
    }
    *result = left + right;
    return 0;
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

static int validate_timing(const ShadowSpillSimulationProgram *program) {
    if ((program->task_phase_count != 0U &&
         (program->task_phases == NULL || program->task_phase_offsets == NULL)) ||
        (program->task_phase_offsets != NULL && !validate_offsets(
            program->task_phase_offsets, program->task_count, program->task_phase_count))) {
        return 0;
    }
    if (program->task_phase_offsets != NULL) {
        for (uint32_t task = 0; task < program->task_count; ++task) {
            uint64_t duration = 0U;
            uint32_t begin = program->task_phase_offsets[task];
            uint32_t end = program->task_phase_offsets[task + 1U];
            for (uint32_t index = begin; index < end; ++index) {
                const ShadowSpillTaskResourcePhase *phase = &program->task_phases[index];
                if ((phase->resource != SHADOWSPILL_SIMULATOR_NO_INDEX &&
                     phase->resource >= program->transfer_resource_count) ||
                    shadowspill_add_overflow_u64(duration, phase->duration_ns, &duration)) return 0;
            }
            if (begin != end && duration != program->task_runtime_ns[task]) return 0;
        }
    }
    if (program->overlap_category_count == 0U) return 1;
    if (!require_pointer(program->task_overlap_profiles, program->task_count) ||
        !require_pointer(program->transfer_overlap_categories, program->transfer_phase_count) ||
        !require_pointer(program->task_overlap_rates, program->overlap_profile_count) ||
        !require_pointer(program->transfer_overlap_rates, program->overlap_profile_count)) return 0;
    for (uint32_t index = 0; index < program->transfer_phase_count; ++index) {
        uint32_t category = program->transfer_overlap_categories[index];
        if (category != SHADOWSPILL_SIMULATOR_NO_INDEX &&
            (category >= program->overlap_category_count ||
             program->transfer_phases[index].kind != SHADOWSPILL_TRANSFER_PHASE_TIMED)) return 0;
    }
    for (uint32_t task = 0; task < program->task_count; ++task) {
        if (program->task_overlap_profiles[task] >= program->overlap_profile_count) return 0;
    }
    uint64_t count = (uint64_t)program->overlap_profile_count * program->overlap_category_count;
    if (count > SIZE_MAX / sizeof(double)) return 0;
    for (uint64_t index = 0; index < count; ++index) {
        double task = program->task_overlap_rates[index];
        double transfer = program->transfer_overlap_rates[index];
        if (!isfinite(task) || task <= 0.0 || task > 1.0 ||
            !isfinite(transfer) || transfer <= 0.0 || transfer > 1.0) return 0;
    }
    return 1;
}

int shadowspill_validate_program(
    const ShadowSpillSimulationProgram *program
) {
    if (program == NULL ||
        program->abi_version != SHADOWSPILL_ABI_VERSION ||
        program->device_count == 0U ||
        program->use_admission_accounting > 1U) {
        return 0;
    }
    if (program->alias_count > (UINT32_MAX - 1U) / 2U ||
        (program->transfer_phase_count != 0U &&
         (program->transfer_phases == NULL || program->alias_transfer_offsets == NULL)) ||
        (program->alias_transfer_offsets != NULL && !validate_offsets(
            program->alias_transfer_offsets, program->alias_count * 2U,
            program->transfer_phase_count
        ))) {
        return 0;
    }
    for (uint32_t index = 0U; index < program->transfer_phase_count; ++index) {
        const ShadowSpillTransferPhase *phase = &program->transfer_phases[index];
        if (phase->kind > SHADOWSPILL_TRANSFER_PHASE_TIMED ||
            phase->submission_ordered > 1U ||
            (phase->kind == SHADOWSPILL_TRANSFER_PHASE_COPY &&
             (phase->duration_ns != 0U || phase->submission_ordered != 0U)) ||
            (phase->kind == SHADOWSPILL_TRANSFER_PHASE_TIMED &&
             phase->resource >= program->transfer_resource_count)) {
            return 0;
        }
    }
    if (program->alias_transfer_offsets != NULL) {
        for (uint32_t row = 0U; row < program->alias_count * 2U; ++row) {
            uint32_t ordered = 0U;
            for (uint32_t phase = program->alias_transfer_offsets[row];
                 phase < program->alias_transfer_offsets[row + 1U]; ++phase) {
                ordered += program->transfer_phases[phase].submission_ordered;
            }
            if (ordered > 1U) {
                return 0;
            }
        }
    }
    if (!require_pointer(program->devices, program->device_count) ||
        !require_pointer(program->alias_device, program->alias_count) ||
        !require_pointer(program->alias_size_bytes, program->alias_count) ||
        !require_pointer(program->alias_initial_version, program->alias_count) ||
        !require_pointer(program->alias_retain_spill_copy, program->alias_count) ||
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
        !require_pointer(program->final_locations, program->final_count) ||
        (program->use_admission_accounting != 0U &&
            (!require_pointer(
                program->task_start_physical_deltas, program->task_count
            ) || !require_pointer(
                program->task_completion_physical_deltas,
                program->task_count
            ) || !require_pointer(
                program->action_trigger_physical_deltas,
                program->action_count
            ) || !require_pointer(
                program->action_completion_physical_deltas,
                program->action_count
            ) || !require_pointer(
                program->initial_physical_bytes, program->device_count
            ) || !require_pointer(
                program->reuse_predecessor_actions,
                program->reuse_dependency_count
            ) || !require_pointer(
                program->reuse_successor_tasks,
                program->reuse_dependency_count
            ) || !require_pointer(
                program->reuse_successor_actions,
                program->reuse_dependency_count
            )))) {
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
        if (program->devices[index].fetch_solo_bandwidth_bytes_per_second == 0U ||
            program->devices[index].evict_solo_bandwidth_bytes_per_second == 0U ||
            program->devices[index].fetch_concurrent_bandwidth_bytes_per_second == 0U ||
            program->devices[index].evict_concurrent_bandwidth_bytes_per_second == 0U) {
            return 0;
        }
    }
    for (uint32_t index = 0; index < program->alias_count; ++index) {
        if (program->alias_device[index] >= program->device_count) {
            return 0;
        }
    }
    if (!validate_timing(program)) return 0;
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
            program->action_kinds[index] > SHADOWSPILL_MEMORY_WRITE_BACK) {
            return 0;
        }
        if (program->action_kinds[index] == SHADOWSPILL_MEMORY_WRITE_BACK &&
            program->alias_transfer_offsets != NULL) {
            const uint32_t row = program->action_aliases[index] * 2U + SHADOWSPILL_TRANSFER_EVICT;
            for (uint32_t phase = program->alias_transfer_offsets[row];
                 phase < program->alias_transfer_offsets[row + 1U]; ++phase) {
                /* Ordered eviction conversions can overwrite their source.
                   They require EVICT; WRITE_BACK retains the execution copy.
                   Refuse before resource ordering rather than leaving an
                   unnumbered phase waiting forever. */
                if (program->transfer_phases[phase].submission_ordered) {
                    return 0;
                }
            }
        }
    }
    for (uint32_t index = 0; index < program->initial_count; ++index) {
        if (program->initial_aliases[index] >= program->alias_count ||
            program->initial_locations[index] > SHADOWSPILL_MEMORY_SPILL) {
            return 0;
        }
    }
    for (uint32_t index = 0; index < program->final_count; ++index) {
        if (program->final_aliases[index] >= program->alias_count ||
            program->final_locations[index] > SHADOWSPILL_MEMORY_SPILL) {
            return 0;
        }
    }
    for (uint32_t index = 0; index < program->reuse_dependency_count; ++index) {
        uint32_t predecessor = program->reuse_predecessor_actions[index];
        uint32_t predecessor_task = program->reuse_predecessor_tasks == NULL
            ? SHADOWSPILL_SIMULATOR_NO_INDEX : program->reuse_predecessor_tasks[index];
        uint32_t successor_task = program->reuse_successor_tasks[index];
        uint32_t successor_action = program->reuse_successor_actions[index];
        if (((predecessor == SHADOWSPILL_SIMULATOR_NO_INDEX) ==
             (predecessor_task == SHADOWSPILL_SIMULATOR_NO_INDEX)) ||
            (predecessor != SHADOWSPILL_SIMULATOR_NO_INDEX &&
             (predecessor >= program->action_count ||
              program->action_kinds[predecessor] != SHADOWSPILL_MEMORY_EVICT)) ||
            (predecessor_task != SHADOWSPILL_SIMULATOR_NO_INDEX &&
             (predecessor_task >= program->task_count || predecessor_task == successor_task)) ||
            ((successor_task == SHADOWSPILL_SIMULATOR_NO_INDEX) ==
                (successor_action == SHADOWSPILL_SIMULATOR_NO_INDEX)) ||
            (successor_task != SHADOWSPILL_SIMULATOR_NO_INDEX &&
                successor_task >= program->task_count) ||
            (successor_action != SHADOWSPILL_SIMULATOR_NO_INDEX &&
                successor_action >= program->action_count)) {
            return 0;
        }
    }
    return 1;
}
