/* What the program says about its tasks, aliases and boundaries. */
#include "internal.h"

int shadowspill_problem_add_u64(uint64_t left, uint64_t right, uint64_t *result) {
    if (right > UINT64_MAX - left) {
        return -1;
    }
    *result = left + right;
    return 0;
}

/* Return ceil(numerator * 1e9 / denominator) without overflowing uint64_t. */
static int transfer_duration_ns(
    uint64_t numerator,
    uint64_t denominator,
    uint64_t latency_ns,
    uint64_t *result
) {
    const uint64_t scale = UINT64_C(1000000000);
    if (denominator == 0U) {
        return -1;
    }
    uint64_t whole = numerator / denominator;
    uint64_t remainder = numerator % denominator;
    if (whole > UINT64_MAX / scale) {
        return -1;
    }
    uint64_t quotient = 0U;
    uint64_t fractional_remainder = 0U;
    uint64_t highest_bit = UINT64_C(1) << 29U;
    for (uint64_t bit = highest_bit; bit != 0U; bit >>= 1U) {
        if (quotient > UINT64_MAX / 2U) {
            return -1;
        }
        quotient *= 2U;
        if (fractional_remainder >= denominator - fractional_remainder) {
            fractional_remainder -= denominator - fractional_remainder;
            ++quotient;
        } else {
            fractional_remainder *= 2U;
        }
        if ((scale & bit) == 0U) {
            continue;
        }
        if (fractional_remainder >= denominator - remainder) {
            fractional_remainder -= denominator - remainder;
            ++quotient;
        } else {
            fractional_remainder += remainder;
        }
    }
    if (fractional_remainder != 0U) {
        ++quotient;
    }
    uint64_t duration = whole * scale;
    if (shadowspill_problem_add_u64(duration, quotient, &duration) != 0 ||
        shadowspill_problem_add_u64(duration, latency_ns, result) != 0) {
        return -1;
    }
    return 0;
}

int shadowspill_problem_compare_u32(uint32_t left, uint32_t right) {
    return left < right ? -1 : left > right ? 1 : 0;
}

int shadowspill_problem_compare_u64(uint64_t left, uint64_t right) {
    return left < right ? -1 : left > right ? 1 : 0;
}

int shadowspill_problem_program_problem_valid(
    const ShadowSpillIndexedProblem *problem,
    const ShadowSpillPressureFitOptions *options
) {
    if (problem == NULL || options == NULL || problem->context.simulation == NULL ||
        problem->device_priority == NULL || problem->context.alias_json_names == NULL ||
        problem->context.task_json_names == NULL ||
        problem->abi_version != SHADOWSPILL_ABI_VERSION ||
        problem->context.simulation->abi_version != SHADOWSPILL_ABI_VERSION ||
        problem->context.simulation->device_count == 0U ||
        problem->context.simulation->task_count == 0U ||
        options->initial_placement > SHADOWSPILL_PRESSUREFIT_INITIAL_PLACEMENT_GREEDY) {
        return 0;
    }
    const ShadowSpillSimulationProgram *program = problem->context.simulation;
    if (program->devices == NULL || program->alias_device == NULL ||
        program->alias_size_bytes == NULL ||
        program->alias_retain_spill_copy == NULL ||
        program->task_device == NULL || program->task_runtime_ns == NULL ||
        program->task_workspace_bytes == NULL ||
        program->input_offsets == NULL || program->output_offsets == NULL ||
        program->mutation_offsets == NULL ||
        (program->input_count != 0U && program->input_aliases == NULL) ||
        (program->output_count != 0U && program->output_aliases == NULL) ||
        (program->mutation_count != 0U && program->mutation_aliases == NULL) ||
        (program->initial_count != 0U &&
         (program->initial_aliases == NULL || program->initial_locations == NULL)) ||
        (program->final_count != 0U &&
         (program->final_aliases == NULL || program->final_locations == NULL))) {
        return 0;
    }
    for (uint32_t alias = 0U; alias < program->alias_count; ++alias) {
        if (problem->context.alias_json_names[alias] == NULL) {
            return 0;
        }
    }
    for (uint32_t task = 0U; task < program->task_count; ++task) {
        if (problem->context.task_json_names[task] == NULL) {
            return 0;
        }
    }
    return 1;
}

int shadowspill_problem_bind_residency(
    uint32_t alias_count,
    uint32_t value_count,
    const uint32_t *aliases,
    const uint8_t *locations,
    int8_t *destination
) {
    for (uint32_t index = 0U; index < value_count; ++index) {
        uint32_t alias = aliases[index];
        uint8_t location = locations[index];
        if (alias >= alias_count || location > SHADOWSPILL_MEMORY_SPILL ||
            destination[alias] >= 0) {
            return -1;
        }
        destination[alias] = (int8_t)location;
    }
    return 0;
}

static int contains_alias(
    const uint32_t *values,
    uint32_t start,
    uint32_t end,
    uint32_t alias
) {
    for (uint32_t index = start; index < end; ++index) {
        if (values[index] == alias) {
            return 1;
        }
    }
    return 0;
}

static int validate_offsets(
    const uint32_t *offsets,
    uint32_t task_count,
    uint32_t value_count
) {
    if (offsets[0] != 0U || offsets[task_count] != value_count) {
        return -1;
    }
    for (uint32_t task = 0U; task < task_count; ++task) {
        if (offsets[task] > offsets[task + 1U]) {
            return -1;
        }
    }
    return 0;
}

static void record_access(
    PreparedProblem *prepared,
    uint32_t alias,
    uint32_t position,
    uint32_t task,
    uint32_t boundary_count
) {
    size_t cell = (size_t)alias * boundary_count + position;
    prepared->anchors[cell] = 1U;
    if (prepared->latest_access_task[cell] == UINT32_MAX ||
        task > prepared->latest_access_task[cell]) {
        prepared->latest_access_task[cell] = task;
    }
    if (prepared->first_access_task[alias] == UINT32_MAX ||
        task < prepared->first_access_task[alias]) {
        prepared->first_access_task[alias] = task;
    }
}

ShadowSpillStatus shadowspill_problem_build_sparse_lists(
    const ShadowSpillSimulationProgram *program,
    PreparedProblem *prepared
) {
    return shadowspill_residency_sparse_lists_build(
               prepared->anchors,
               prepared->latest_access_task,
               prepared->output_reservations,
               program->alias_count,
               program->task_count + 1U,
               &prepared->sparse
           ) == 0
        ? SHADOWSPILL_STATUS_OK
        : SHADOWSPILL_STATUS_INTERNAL_FAILURE;
}

ShadowSpillStatus shadowspill_problem_derive_task_facts(
    const ShadowSpillSimulationProgram *program,
    PreparedProblem *prepared
) {
    uint32_t boundary_count = program->task_count + 1U;
    if (validate_offsets(program->input_offsets, program->task_count,
                         program->input_count) != 0 ||
        validate_offsets(program->output_offsets, program->task_count,
                         program->output_count) != 0 ||
        validate_offsets(program->mutation_offsets, program->task_count,
                         program->mutation_count) != 0) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }

    for (uint32_t device = 0U; device < program->device_count; ++device) {
        prepared->device_capacity_bytes[device] =
            program->devices[device].capacity_bytes;
    }
    uint64_t ideal_end = 0U;
    ShadowSpillStatus status = SHADOWSPILL_STATUS_OK;
    for (uint32_t task = 0U; task < program->task_count; ++task) {
        uint32_t device = program->task_device[task];
        if (device >= program->device_count ||
            shadowspill_problem_add_u64(ideal_end, program->task_runtime_ns[task], &ideal_end) != 0) {
            status = SHADOWSPILL_STATUS_INVALID_ARGUMENT;
            break;
        }
        prepared->task_ideal_end_ns[task] = ideal_end;
        prepared->boundary_capacity_bytes[
            (uint64_t)device * boundary_count + task
        ] = program->task_workspace_bytes[task];

        uint32_t input_start = program->input_offsets[task];
        uint32_t input_end = program->input_offsets[task + 1U];
        uint32_t output_start = program->output_offsets[task];
        uint32_t output_end = program->output_offsets[task + 1U];
        uint32_t mutation_start = program->mutation_offsets[task];
        uint32_t mutation_end = program->mutation_offsets[task + 1U];
        for (uint32_t index = input_start; index < input_end; ++index) {
            uint32_t alias = program->input_aliases[index];
            if (alias >= program->alias_count) {
                status = SHADOWSPILL_STATUS_INVALID_ARGUMENT;
                break;
            }
            if (program->alias_size_bytes[alias] != 0U) {
                prepared->seen_input[alias] = 1U;
                if (prepared->first_input_task[alias] == UINT32_MAX) {
                    prepared->first_input_task[alias] = task;
                }
                record_access(prepared, alias, task, task, boundary_count);
            }
        }
        if (status != SHADOWSPILL_STATUS_OK) {
            break;
        }
        for (uint32_t index = mutation_start; index < mutation_end; ++index) {
            uint32_t alias = program->mutation_aliases[index];
            if (alias >= program->alias_count) {
                status = SHADOWSPILL_STATUS_INVALID_ARGUMENT;
                break;
            }
            if (program->alias_size_bytes[alias] != 0U) {
                record_access(prepared, alias, task, task, boundary_count);
                prepared->anchors[
                    (size_t)alias * boundary_count + task + 1U
                ] = 1U;
                prepared->write_prefix[
                    (size_t)alias * boundary_count + task + 1U
                ] = 1U;
            }
        }
        if (status != SHADOWSPILL_STATUS_OK) {
            break;
        }
        for (uint32_t index = output_start; index < output_end; ++index) {
            uint32_t alias = program->output_aliases[index];
            if (alias >= program->alias_count) {
                status = SHADOWSPILL_STATUS_INVALID_ARGUMENT;
                break;
            }
            if (program->alias_size_bytes[alias] == 0U) {
                continue;
            }
            size_t output_cell =
                (size_t)alias * boundary_count + task + 1U;
            record_access(prepared, alias, task + 1U, task, boundary_count);
            prepared->productions[output_cell] = 1U;
            prepared->write_prefix[output_cell] = 1U;
            prepared->produced[alias] = 1U;
            if (!contains_alias(program->input_aliases, input_start, input_end,
                                alias) &&
                !contains_alias(program->mutation_aliases, mutation_start,
                                mutation_end, alias)) {
                prepared->output_reservations[
                    (size_t)alias * boundary_count + task
                ] = 1U;
            }
        }
        if (status != SHADOWSPILL_STATUS_OK) {
            break;
        }
    }

    return status;
}

ShadowSpillStatus shadowspill_problem_finalize_boundary_capacities(
    const ShadowSpillSimulationProgram *program,
    PreparedProblem *prepared
) {
    const uint32_t boundary_count = program->task_count + 1U;
    for (uint32_t device = 0U; device < program->device_count; ++device) {
        const uint64_t capacity = prepared->device_capacity_bytes[device];
        for (uint32_t boundary = 0U; boundary < boundary_count; ++boundary) {
            const uint64_t position =
                (uint64_t)device * boundary_count + boundary;
            const uint64_t workspace =
                prepared->boundary_capacity_bytes[position];
            if (workspace > capacity) {
                prepared->failure_kind =
                    SHADOWSPILL_PRESSUREFIT_PREFLIGHT_WORKSPACE_CAPACITY;
                prepared->error_device = device;
                prepared->error_boundary = (int32_t)boundary;
                prepared->failure_required_bytes = workspace;
                prepared->failure_capacity_bytes = capacity;
                return SHADOWSPILL_STATUS_ANALYTIC_INFEASIBLE;
            }
            prepared->boundary_capacity_bytes[position] = capacity - workspace;
        }
    }
    return SHADOWSPILL_STATUS_OK;
}

ShadowSpillStatus shadowspill_problem_finalize_alias_facts(
    const ShadowSpillSimulationProgram *program,
    PreparedProblem *prepared
) {
    uint32_t boundary_count = program->task_count + 1U;
    for (uint32_t alias = 0U; alias < program->alias_count; ++alias) {
        uint32_t device = program->alias_device[alias];
        if (device >= program->device_count ||
            transfer_duration_ns(
                program->alias_size_bytes[alias],
                program->devices[device].fetch_bandwidth_bytes_per_second,
                program->devices[device].fetch_latency_ns,
                &prepared->fetch_runtime_ns[alias]
            ) != 0 ||
            transfer_duration_ns(
                program->alias_size_bytes[alias],
                program->devices[device].evict_bandwidth_bytes_per_second,
                program->devices[device].evict_latency_ns,
                &prepared->evict_runtime_ns[alias]
            ) != 0) {
            return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
        }
        if (program->alias_size_bytes[alias] != 0U &&
            prepared->initial_location[alias] == SHADOWSPILL_MEMORY_DEVICE) {
            prepared->anchors[(size_t)alias * boundary_count] = 1U;
        }
        if (program->alias_size_bytes[alias] != 0U &&
            prepared->final_location[alias] == SHADOWSPILL_MEMORY_DEVICE) {
            prepared->anchors[
                (size_t)alias * boundary_count + program->task_count
            ] = 1U;
        }
        if (prepared->seen_input[alias] != 0U &&
            prepared->first_input_task[alias] == 0U &&
            prepared->produced[alias] == 0U &&
            prepared->initial_location[alias] < 0) {
            prepared->failure_kind =
                SHADOWSPILL_PRESSUREFIT_PREFLIGHT_MISSING_INITIAL_RESIDENCY;
            prepared->error_alias = alias;
            prepared->error_boundary = 0;
            return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
        }
        uint8_t seen_write = 0U;
        for (uint32_t position = 0U; position < boundary_count; ++position) {
            size_t cell = (size_t)alias * boundary_count + position;
            if (prepared->write_prefix[cell] != 0U) {
                seen_write = 1U;
            }
            prepared->write_prefix[cell] = seen_write;
        }
    }
    return SHADOWSPILL_STATUS_OK;
}
