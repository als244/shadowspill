#include <shadowspill/planner.h>

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct PreparedContext {
    ShadowSpillResidencyProblem residency;
    ShadowSpillPressureFitContext context;

    int8_t *initial_location;
    int8_t *final_location;
    uint8_t *anchors;
    uint8_t *productions;
    uint32_t *latest_access_task;
    uint8_t *output_reservations;
    uint8_t *write_prefix;
    uint32_t *first_input_task;
    uint64_t *fetch_runtime_ns;
    uint64_t *evict_runtime_ns;
    uint64_t *task_ideal_end_ns;
    uint64_t *device_capacity_bytes;
    uint8_t *seed_resident;
    uint8_t *seed_breaks;

    uint32_t *first_access_task;
    uint8_t *produced;
    uint8_t *seen_input;
    uint8_t *charged_anchors;
    uint64_t *required_bytes;
} PreparedContext;

typedef struct ColdAlias {
    uint32_t alias;
    uint32_t device;
    uint32_t first_use;
    uint64_t deadline;
    uint64_t transfer_ns;
    uint64_t miss_ns;
    uint64_t slack_ns;
    uint64_t size_bytes;
} ColdAlias;

static int checked_cells(
    uint32_t rows,
    uint32_t columns,
    size_t *result
) {
    uint64_t value = (uint64_t)rows * columns;
    if (value > SIZE_MAX) {
        return -1;
    }
    *result = (size_t)value;
    return 0;
}

static int add_u64(uint64_t left, uint64_t right, uint64_t *result) {
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
    if (add_u64(duration, quotient, &duration) != 0 ||
        add_u64(duration, latency_ns, result) != 0) {
        return -1;
    }
    return 0;
}

static int compare_u32(uint32_t left, uint32_t right) {
    return left < right ? -1 : left > right ? 1 : 0;
}

static int compare_u64(uint64_t left, uint64_t right) {
    return left < right ? -1 : left > right ? 1 : 0;
}

static int compare_cold_deadline(const void *left_value, const void *right_value) {
    const ColdAlias *left = left_value;
    const ColdAlias *right = right_value;
    int result = compare_u64(left->deadline, right->deadline);
    if (result == 0) {
        result = compare_u32(left->first_use, right->first_use);
    }
    return result == 0 ? compare_u32(left->alias, right->alias) : result;
}

static int compare_cold_placement(
    const void *left_value,
    const void *right_value
) {
    const ColdAlias *left = left_value;
    const ColdAlias *right = right_value;
    int result = compare_u32(left->first_use, right->first_use);
    if (result == 0) {
        result = compare_u64(left->slack_ns, right->slack_ns);
    }
    if (result == 0) {
        result = compare_u64(right->miss_ns, left->miss_ns);
    }
    if (result == 0) {
        result = compare_u64(right->size_bytes, left->size_bytes);
    }
    return result == 0 ? compare_u32(left->alias, right->alias) : result;
}

static void prepared_context_destroy(PreparedContext *prepared) {
    free(prepared->initial_location);
    free(prepared->final_location);
    free(prepared->anchors);
    free(prepared->productions);
    free(prepared->latest_access_task);
    free(prepared->output_reservations);
    free(prepared->write_prefix);
    free(prepared->first_input_task);
    free(prepared->fetch_runtime_ns);
    free(prepared->evict_runtime_ns);
    free(prepared->task_ideal_end_ns);
    free(prepared->device_capacity_bytes);
    free(prepared->seed_resident);
    free(prepared->seed_breaks);
    free(prepared->first_access_task);
    free(prepared->produced);
    free(prepared->seen_input);
    free(prepared->charged_anchors);
    free(prepared->required_bytes);
    memset(prepared, 0, sizeof(*prepared));
}

static int program_context_valid(
    const ShadowSpillPressureFitProgramContext *context,
    const ShadowSpillPressureFitContextOptions *options
) {
    if (context == NULL || options == NULL || context->simulation == NULL ||
        context->device_priority == NULL || context->alias_json_names == NULL ||
        context->task_json_names == NULL ||
        context->abi_version != SHADOWSPILL_PLANNER_ABI_VERSION ||
        context->simulation->abi_version != SHADOWSPILL_SIMULATOR_ABI_VERSION ||
        context->simulation->device_count == 0U ||
        context->simulation->alias_count == 0U ||
        context->simulation->task_count == 0U ||
        options->initial_placement > SHADOWSPILL_INITIAL_PLACEMENT_GREEDY) {
        return 0;
    }
    const ShadowSpillSimulationProgram *program = context->simulation;
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
        if (context->alias_json_names[alias] == NULL) {
            return 0;
        }
    }
    for (uint32_t task = 0U; task < program->task_count; ++task) {
        if (context->task_json_names[task] == NULL) {
            return 0;
        }
    }
    return 1;
}

static int allocate_prepared_buffers(
    const ShadowSpillSimulationProgram *program,
    PreparedContext *prepared
) {
    size_t cells = 0U;
    size_t pressure_cells = 0U;
    uint32_t boundary_count = program->task_count + 1U;
    if (boundary_count == 0U ||
        checked_cells(program->alias_count, boundary_count, &cells) != 0 ||
        checked_cells(program->device_count, boundary_count, &pressure_cells) != 0) {
        return -1;
    }
    prepared->initial_location = malloc(
        (size_t)program->alias_count * sizeof(*prepared->initial_location)
    );
    prepared->final_location = malloc(
        (size_t)program->alias_count * sizeof(*prepared->final_location)
    );
    prepared->anchors = calloc(cells, sizeof(*prepared->anchors));
    prepared->productions = calloc(cells, sizeof(*prepared->productions));
    prepared->latest_access_task = malloc(
        cells * sizeof(*prepared->latest_access_task)
    );
    prepared->output_reservations = calloc(
        cells,
        sizeof(*prepared->output_reservations)
    );
    prepared->write_prefix = calloc(cells, sizeof(*prepared->write_prefix));
    prepared->first_input_task = malloc(
        (size_t)program->alias_count * sizeof(*prepared->first_input_task)
    );
    prepared->fetch_runtime_ns = calloc(
        program->alias_count,
        sizeof(*prepared->fetch_runtime_ns)
    );
    prepared->evict_runtime_ns = calloc(
        program->alias_count,
        sizeof(*prepared->evict_runtime_ns)
    );
    prepared->task_ideal_end_ns = calloc(
        program->task_count,
        sizeof(*prepared->task_ideal_end_ns)
    );
    prepared->device_capacity_bytes = calloc(
        program->device_count,
        sizeof(*prepared->device_capacity_bytes)
    );
    prepared->seed_resident = calloc(cells, sizeof(*prepared->seed_resident));
    prepared->seed_breaks = calloc(cells, sizeof(*prepared->seed_breaks));
    prepared->first_access_task = malloc(
        (size_t)program->alias_count * sizeof(*prepared->first_access_task)
    );
    prepared->produced = calloc(program->alias_count, sizeof(*prepared->produced));
    prepared->seen_input = calloc(
        program->alias_count,
        sizeof(*prepared->seen_input)
    );
    prepared->charged_anchors = calloc(
        cells,
        sizeof(*prepared->charged_anchors)
    );
    prepared->required_bytes = calloc(
        pressure_cells,
        sizeof(*prepared->required_bytes)
    );
    if (prepared->initial_location == NULL ||
        prepared->final_location == NULL || prepared->anchors == NULL ||
        prepared->productions == NULL ||
        prepared->latest_access_task == NULL ||
        prepared->output_reservations == NULL ||
        prepared->write_prefix == NULL ||
        prepared->first_input_task == NULL ||
        prepared->fetch_runtime_ns == NULL ||
        prepared->evict_runtime_ns == NULL ||
        prepared->task_ideal_end_ns == NULL ||
        prepared->device_capacity_bytes == NULL ||
        prepared->seed_resident == NULL || prepared->seed_breaks == NULL ||
        prepared->first_access_task == NULL || prepared->produced == NULL ||
        prepared->seen_input == NULL || prepared->charged_anchors == NULL ||
        prepared->required_bytes == NULL) {
        return -1;
    }
    memset(prepared->initial_location, -1, program->alias_count);
    memset(prepared->final_location, -1, program->alias_count);
    memset(prepared->latest_access_task, 0xff,
           cells * sizeof(*prepared->latest_access_task));
    memset(prepared->first_input_task, 0xff,
           (size_t)program->alias_count * sizeof(*prepared->first_input_task));
    memset(prepared->first_access_task, 0xff,
           (size_t)program->alias_count * sizeof(*prepared->first_access_task));
    return 0;
}

static int bind_residency(
    uint32_t alias_count,
    uint32_t value_count,
    const uint32_t *aliases,
    const uint8_t *locations,
    int8_t *destination
) {
    for (uint32_t index = 0U; index < value_count; ++index) {
        uint32_t alias = aliases[index];
        uint8_t location = locations[index];
        if (alias >= alias_count || location > SHADOWSPILL_MEMORY_HOST ||
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
    PreparedContext *prepared,
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

static ShadowSpillPlannerStatus derive_task_facts(
    const ShadowSpillSimulationProgram *program,
    PreparedContext *prepared
) {
    uint32_t boundary_count = program->task_count + 1U;
    if (validate_offsets(program->input_offsets, program->task_count,
                         program->input_count) != 0 ||
        validate_offsets(program->output_offsets, program->task_count,
                         program->output_count) != 0 ||
        validate_offsets(program->mutation_offsets, program->task_count,
                         program->mutation_count) != 0) {
        return SHADOWSPILL_PLANNER_INVALID_ARGUMENT;
    }

    uint64_t *max_workspace = calloc(
        program->device_count,
        sizeof(*max_workspace)
    );
    if (max_workspace == NULL) {
        return SHADOWSPILL_PLANNER_ALLOCATION_FAILURE;
    }
    uint64_t ideal_end = 0U;
    ShadowSpillPlannerStatus status = SHADOWSPILL_PLANNER_OK;
    for (uint32_t task = 0U; task < program->task_count; ++task) {
        uint32_t device = program->task_device[task];
        if (device >= program->device_count ||
            add_u64(ideal_end, program->task_runtime_ns[task], &ideal_end) != 0) {
            status = SHADOWSPILL_PLANNER_INVALID_ARGUMENT;
            break;
        }
        prepared->task_ideal_end_ns[task] = ideal_end;
        if (program->task_workspace_bytes[task] > max_workspace[device]) {
            max_workspace[device] = program->task_workspace_bytes[task];
        }

        uint32_t input_start = program->input_offsets[task];
        uint32_t input_end = program->input_offsets[task + 1U];
        uint32_t output_start = program->output_offsets[task];
        uint32_t output_end = program->output_offsets[task + 1U];
        uint32_t mutation_start = program->mutation_offsets[task];
        uint32_t mutation_end = program->mutation_offsets[task + 1U];
        for (uint32_t index = input_start; index < input_end; ++index) {
            uint32_t alias = program->input_aliases[index];
            if (alias >= program->alias_count) {
                status = SHADOWSPILL_PLANNER_INVALID_ARGUMENT;
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
        if (status != SHADOWSPILL_PLANNER_OK) {
            break;
        }
        for (uint32_t index = mutation_start; index < mutation_end; ++index) {
            uint32_t alias = program->mutation_aliases[index];
            if (alias >= program->alias_count) {
                status = SHADOWSPILL_PLANNER_INVALID_ARGUMENT;
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
        if (status != SHADOWSPILL_PLANNER_OK) {
            break;
        }
        for (uint32_t index = output_start; index < output_end; ++index) {
            uint32_t alias = program->output_aliases[index];
            if (alias >= program->alias_count) {
                status = SHADOWSPILL_PLANNER_INVALID_ARGUMENT;
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
        if (status != SHADOWSPILL_PLANNER_OK) {
            break;
        }
    }

    if (status == SHADOWSPILL_PLANNER_OK) {
        for (uint32_t device = 0U; device < program->device_count; ++device) {
            if (max_workspace[device] > program->devices[device].capacity_bytes) {
                status = SHADOWSPILL_PLANNER_ANALYTIC_INFEASIBLE;
                break;
            }
            prepared->device_capacity_bytes[device] =
                program->devices[device].capacity_bytes - max_workspace[device];
        }
    }
    free(max_workspace);
    return status;
}

static ShadowSpillPlannerStatus finalize_alias_facts(
    const ShadowSpillSimulationProgram *program,
    PreparedContext *prepared
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
            return SHADOWSPILL_PLANNER_INVALID_ARGUMENT;
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
            return SHADOWSPILL_PLANNER_INVALID_ARGUMENT;
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
    return SHADOWSPILL_PLANNER_OK;
}

static ShadowSpillPlannerStatus validate_required_floor(
    const ShadowSpillSimulationProgram *program,
    PreparedContext *prepared
) {
    uint32_t boundary_count = program->task_count + 1U;
    for (uint32_t alias = 0U; alias < program->alias_count; ++alias) {
        uint32_t device = program->alias_device[alias];
        uint64_t size = program->alias_size_bytes[alias];
        for (uint32_t position = 0U; position < boundary_count; ++position) {
            size_t cell = (size_t)alias * boundary_count + position;
            if (prepared->anchors[cell] == 0U) {
                continue;
            }
            int contributes = position == 0U ||
                prepared->final_location[alias] == SHADOWSPILL_MEMORY_DEVICE ||
                (prepared->latest_access_task[cell] != UINT32_MAX &&
                 prepared->latest_access_task[cell] > position - 1U);
            if (!contributes) {
                continue;
            }
            prepared->charged_anchors[cell] = 1U;
            size_t pressure = (size_t)device * boundary_count + position;
            if (add_u64(prepared->required_bytes[pressure], size,
                        &prepared->required_bytes[pressure]) != 0) {
                return SHADOWSPILL_PLANNER_INVALID_ARGUMENT;
            }
        }
    }
    for (uint32_t alias = 0U; alias < program->alias_count; ++alias) {
        uint32_t device = program->alias_device[alias];
        for (uint32_t task = 0U; task < program->task_count; ++task) {
            size_t cell = (size_t)alias * boundary_count + task;
            if (prepared->output_reservations[cell] == 0U ||
                prepared->charged_anchors[cell] != 0U) {
                continue;
            }
            size_t pressure = (size_t)device * boundary_count + task;
            if (add_u64(
                    prepared->required_bytes[pressure],
                    program->alias_size_bytes[alias],
                    &prepared->required_bytes[pressure]
                ) != 0) {
                return SHADOWSPILL_PLANNER_INVALID_ARGUMENT;
            }
        }
    }
    for (uint32_t device = 0U; device < program->device_count; ++device) {
        for (uint32_t position = 0U; position < boundary_count; ++position) {
            uint64_t required = prepared->required_bytes[
                (size_t)device * boundary_count + position
            ];
            if (required > prepared->device_capacity_bytes[device]) {
                return SHADOWSPILL_PLANNER_ANALYTIC_INFEASIBLE;
            }
        }
    }
    return SHADOWSPILL_PLANNER_OK;
}

static void build_anchor_seed(
    const ShadowSpillSimulationProgram *program,
    PreparedContext *prepared
) {
    uint32_t boundary_count = program->task_count + 1U;
    for (uint32_t alias = 0U; alias < program->alias_count; ++alias) {
        uint32_t first = UINT32_MAX;
        uint32_t last = 0U;
        for (uint32_t position = 0U; position < boundary_count; ++position) {
            if (prepared->anchors[(size_t)alias * boundary_count + position] == 0U) {
                continue;
            }
            if (first == UINT32_MAX) {
                first = position;
            }
            last = position;
        }
        if (first == UINT32_MAX) {
            continue;
        }
        for (uint32_t position = first; position <= last; ++position) {
            prepared->seed_resident[
                (size_t)alias * boundary_count + position
            ] = 1U;
        }
    }
}

static ShadowSpillPlannerStatus greedily_place_initial_aliases(
    const ShadowSpillSimulationProgram *program,
    PreparedContext *prepared
) {
    uint32_t cold_count = 0U;
    for (uint32_t alias = 0U; alias < program->alias_count; ++alias) {
        if (prepared->initial_location[alias] == SHADOWSPILL_MEMORY_HOST &&
            prepared->first_access_task[alias] != UINT32_MAX &&
            prepared->first_access_task[alias] > 0U) {
            ++cold_count;
        }
    }
    ColdAlias *cold = calloc(cold_count == 0U ? 1U : cold_count, sizeof(*cold));
    uint64_t *cursor = calloc(program->device_count, sizeof(*cursor));
    uint64_t *initial_bytes = calloc(
        program->device_count,
        sizeof(*initial_bytes)
    );
    if (cold == NULL || cursor == NULL || initial_bytes == NULL) {
        free(cold);
        free(cursor);
        free(initial_bytes);
        return SHADOWSPILL_PLANNER_ALLOCATION_FAILURE;
    }
    uint32_t boundary_count = program->task_count + 1U;
    uint32_t next = 0U;
    for (uint32_t alias = 0U; alias < program->alias_count; ++alias) {
        uint32_t first_use = prepared->first_access_task[alias];
        if (prepared->initial_location[alias] != SHADOWSPILL_MEMORY_HOST ||
            first_use == UINT32_MAX || first_use == 0U) {
            continue;
        }
        cold[next++] = (ColdAlias){
            .alias = alias,
            .device = program->alias_device[alias],
            .first_use = first_use,
            .deadline = prepared->task_ideal_end_ns[first_use - 1U],
            .transfer_ns = prepared->fetch_runtime_ns[alias],
            .size_bytes = program->alias_size_bytes[alias],
        };
    }
    qsort(cold, cold_count, sizeof(*cold), compare_cold_deadline);
    uint64_t first_task_end = prepared->task_ideal_end_ns[0];
    for (uint32_t device = 0U; device < program->device_count; ++device) {
        cursor[device] = first_task_end;
    }
    for (uint32_t index = 0U; index < cold_count; ++index) {
        ColdAlias *value = &cold[index];
        uint64_t finish = 0U;
        if (add_u64(cursor[value->device], value->transfer_ns, &finish) != 0) {
            free(cold);
            free(cursor);
            free(initial_bytes);
            return SHADOWSPILL_PLANNER_INVALID_ARGUMENT;
        }
        value->miss_ns = finish > value->deadline
            ? finish - value->deadline
            : 0U;
        cursor[value->device] = finish;
        uint64_t unavailable = first_task_end;
        if (add_u64(unavailable, value->transfer_ns, &unavailable) != 0) {
            free(cold);
            free(cursor);
            free(initial_bytes);
            return SHADOWSPILL_PLANNER_INVALID_ARGUMENT;
        }
        value->slack_ns = value->deadline > unavailable
            ? value->deadline - unavailable
            : 0U;
    }

    for (uint32_t alias = 0U; alias < program->alias_count; ++alias) {
        uint32_t device = program->alias_device[alias];
        size_t initial_cell = (size_t)alias * boundary_count;
        int charged = prepared->seed_resident[initial_cell] != 0U ||
            prepared->output_reservations[initial_cell] != 0U;
        if (charged &&
            add_u64(initial_bytes[device], program->alias_size_bytes[alias],
                    &initial_bytes[device]) != 0) {
            free(cold);
            free(cursor);
            free(initial_bytes);
            return SHADOWSPILL_PLANNER_INVALID_ARGUMENT;
        }
    }
    qsort(cold, cold_count, sizeof(*cold), compare_cold_placement);
    for (uint32_t index = 0U; index < cold_count; ++index) {
        ColdAlias *value = &cold[index];
        uint64_t proposed = 0U;
        if (add_u64(initial_bytes[value->device], value->size_bytes, &proposed) != 0) {
            free(cold);
            free(cursor);
            free(initial_bytes);
            return SHADOWSPILL_PLANNER_INVALID_ARGUMENT;
        }
        if (proposed > prepared->device_capacity_bytes[value->device]) {
            continue;
        }
        size_t row = (size_t)value->alias * boundary_count;
        uint32_t last = 0U;
        for (uint32_t position = 0U; position < boundary_count; ++position) {
            if (prepared->anchors[row + position] != 0U) {
                last = position;
            }
        }
        for (uint32_t position = 0U; position <= last; ++position) {
            prepared->seed_resident[row + position] = 1U;
        }
        initial_bytes[value->device] = proposed;
    }
    free(cold);
    free(cursor);
    free(initial_bytes);
    return SHADOWSPILL_PLANNER_OK;
}

static ShadowSpillPlannerStatus prepare_context(
    const ShadowSpillPressureFitProgramContext *source,
    const ShadowSpillPressureFitContextOptions *options,
    PreparedContext *prepared
) {
    memset(prepared, 0, sizeof(*prepared));
    const ShadowSpillSimulationProgram *program = source->simulation;
    if (allocate_prepared_buffers(program, prepared) != 0) {
        return SHADOWSPILL_PLANNER_ALLOCATION_FAILURE;
    }
    if (bind_residency(program->alias_count, program->initial_count,
                       program->initial_aliases, program->initial_locations,
                       prepared->initial_location) != 0 ||
        bind_residency(program->alias_count, program->final_count,
                       program->final_aliases, program->final_locations,
                       prepared->final_location) != 0) {
        return SHADOWSPILL_PLANNER_INVALID_ARGUMENT;
    }
    ShadowSpillPlannerStatus status = derive_task_facts(program, prepared);
    if (status != SHADOWSPILL_PLANNER_OK) {
        return status;
    }
    status = finalize_alias_facts(program, prepared);
    if (status != SHADOWSPILL_PLANNER_OK) {
        return status;
    }
    status = validate_required_floor(program, prepared);
    if (status != SHADOWSPILL_PLANNER_OK) {
        return status;
    }
    build_anchor_seed(program, prepared);
    if (options->initial_placement == SHADOWSPILL_INITIAL_PLACEMENT_GREEDY) {
        status = greedily_place_initial_aliases(program, prepared);
        if (status != SHADOWSPILL_PLANNER_OK) {
            return status;
        }
    }

    prepared->residency = (ShadowSpillResidencyProblem){
        .abi_version = SHADOWSPILL_PLANNER_ABI_VERSION,
        .alias_count = program->alias_count,
        .boundary_count = program->task_count + 1U,
        .device_count = program->device_count,
        .alias_size_bytes = program->alias_size_bytes,
        .alias_device = program->alias_device,
        .alias_retain_spill_copy = program->alias_retain_spill_copy,
        .initial_location = prepared->initial_location,
        .final_location = prepared->final_location,
        .anchors = prepared->anchors,
        .productions = prepared->productions,
        .latest_access_task = prepared->latest_access_task,
        .output_reservations = prepared->output_reservations,
        .write_prefix = prepared->write_prefix,
        .first_input_task = prepared->first_input_task,
        .fetch_runtime_ns = prepared->fetch_runtime_ns,
        .evict_runtime_ns = prepared->evict_runtime_ns,
        .task_ideal_end_ns = prepared->task_ideal_end_ns,
        .device_capacity_bytes = prepared->device_capacity_bytes,
        .device_priority = source->device_priority,
    };
    prepared->context = (ShadowSpillPressureFitContext){
        .abi_version = SHADOWSPILL_PLANNER_ABI_VERSION,
        .residency = &prepared->residency,
        .simulation = program,
        .seed_resident = prepared->seed_resident,
        .seed_breaks = prepared->seed_breaks,
        .alias_json_names = source->alias_json_names,
        .task_json_names = source->task_json_names,
    };
    return SHADOWSPILL_PLANNER_OK;
}

ShadowSpillPlannerStatus shadowspill_evaluate_pressurefit_program_context(
    const ShadowSpillPressureFitProgramContext *context,
    const ShadowSpillPressureFitContextOptions *options,
    ShadowSpillPressureFitContextResult *result
) {
    if (result == NULL) {
        return SHADOWSPILL_PLANNER_INVALID_ARGUMENT;
    }
    memset(result, 0, sizeof(*result));
    result->selected_candidate_index = SHADOWSPILL_PLANNER_NO_INDEX;
    result->status = SHADOWSPILL_PLANNER_INVALID_ARGUMENT;
    if (!program_context_valid(context, options)) {
        return SHADOWSPILL_PLANNER_INVALID_ARGUMENT;
    }
    PreparedContext prepared = {0};
    ShadowSpillPlannerStatus status = prepare_context(context, options, &prepared);
    if (status == SHADOWSPILL_PLANNER_OK) {
        status = shadowspill_evaluate_pressurefit_context(
            &prepared.context,
            options,
            result
        );
    } else {
        result->status = status;
    }
    prepared_context_destroy(&prepared);
    return status;
}
