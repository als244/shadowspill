#include <stdint.h>

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

int shadowspill_validate_program(
    const ShadowSpillSimulationProgram *program
) {
    if (program == NULL ||
        program->abi_version != SHADOWSPILL_SIMULATOR_ABI_VERSION ||
        program->device_count == 0U ||
        program->use_admission_accounting > 1U) {
        return 0;
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
        if (program->devices[index].fetch_bandwidth_bytes_per_second == 0U ||
            program->devices[index].evict_bandwidth_bytes_per_second == 0U) {
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
    for (uint32_t index = 0; index < program->reuse_dependency_count; ++index) {
        uint32_t predecessor = program->reuse_predecessor_actions[index];
        uint32_t successor_task = program->reuse_successor_tasks[index];
        uint32_t successor_action = program->reuse_successor_actions[index];
        if (predecessor >= program->action_count ||
            program->action_kinds[predecessor] !=
                SHADOWSPILL_MEMORY_OFFLOAD ||
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
