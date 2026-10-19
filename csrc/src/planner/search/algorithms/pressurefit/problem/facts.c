/* What the program says about its tasks, aliases and boundaries. */
#include "internal.h"

int shadowspill_problem_add_u64(uint64_t left, uint64_t right, uint64_t *result) {
    if (right > UINT64_MAX - left) {
        return -1;
    }
    *result = left + right;
    return 0;
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
        problem->context.simulation->task_count == 0U) {
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

/* Every alias an access names has to exist. Checked once up front, so the
 * two passes below can index without looking. */
static int aliases_in_range(
    const uint32_t *values, uint32_t count, uint32_t alias_count
) {
    for (uint32_t index = 0U; index < count; ++index) {
        if (values[index] >= alias_count) {
            return 0;
        }
    }
    return 1;
}

/*
 * The per-alias lists, being filled.
 *
 * The program is walked task by task, and within a task its inputs, then
 * its mutations, then its outputs. An input or mutation touches the
 * boundary before the task, a mutation or output the one after, so an
 * alias's entries arrive in boundary order, and an entry at the boundary of
 * the alias's last entry is the same anchor seen again: it merges into it,
 * the way a cell array would have been marked twice. Each list is written
 * at a per-alias cursor into room reserved from an upper bound, and packed
 * tight once the walk is done.
 */
typedef struct ListBuild {
    ShadowSpillPressureFitResidencySparseLists *lists;
    uint32_t *anchor_cursor;
    uint32_t *reserved_cursor;
    uint32_t *write_cursor;
} ListBuild;

static void record_anchor(
    ListBuild *build,
    uint32_t alias,
    uint32_t boundary,
    uint32_t task,
    uint8_t flags
) {
    ShadowSpillPressureFitResidencySparseLists *lists = build->lists;
    uint32_t *cursor = &build->anchor_cursor[alias];
    if (*cursor > lists->anchor_offsets[alias] &&
        lists->anchor_positions[*cursor - 1U] == boundary) {
        const uint32_t at = *cursor - 1U;
        if (task != UINT32_MAX) {
            if (lists->anchor_tasks[at] == UINT32_MAX ||
                task > lists->anchor_tasks[at]) {
                lists->anchor_tasks[at] = task;
            }
            if (lists->anchor_earliest[at] == UINT32_MAX ||
                task < lists->anchor_earliest[at]) {
                lists->anchor_earliest[at] = task;
            }
        }
        lists->anchor_flags[at] |= flags;
        return;
    }
    lists->anchor_positions[*cursor] = boundary;
    lists->anchor_tasks[*cursor] = task;
    lists->anchor_earliest[*cursor] = task;
    lists->anchor_flags[*cursor] = flags;
    ++*cursor;
}

static void record_reservation(ListBuild *build, uint32_t alias, uint32_t boundary) {
    ShadowSpillPressureFitResidencySparseLists *lists = build->lists;
    uint32_t *cursor = &build->reserved_cursor[alias];
    if (*cursor > lists->reserved_offsets[alias] &&
        lists->reserved_positions[*cursor - 1U] == boundary) {
        return;
    }
    lists->reserved_positions[(*cursor)++] = boundary;
}

static void record_write(ListBuild *build, uint32_t alias, uint32_t boundary) {
    ShadowSpillPressureFitResidencySparseLists *lists = build->lists;
    uint32_t *cursor = &build->write_cursor[alias];
    if (*cursor > lists->write_offsets[alias] &&
        lists->write_boundaries[*cursor - 1U] == boundary) {
        return;
    }
    lists->write_boundaries[(*cursor)++] = boundary;
}

/* Turn per-alias room counts into offsets, allocating the arrays behind
 * them. `offsets` holds the counts on entry and the exclusive prefix sums on
 * exit; the total is returned through `total`. */
static int reserve_list(uint32_t *offsets, uint32_t alias_count, uint64_t *total) {
    uint64_t running = 0U;
    for (uint32_t alias = 0U; alias < alias_count; ++alias) {
        const uint32_t count = offsets[alias];
        if (running > UINT32_MAX - count) {
            return -1;
        }
        offsets[alias] = (uint32_t)running;
        running += count;
    }
    offsets[alias_count] = (uint32_t)running;
    *total = running;
    return 0;
}

/* Close the gaps the upper bound left: each alias's entries move down to
 * follow the previous alias's, and the offsets become exact. */
static void pack_list(
    uint32_t *offsets,
    const uint32_t *cursor,
    uint32_t alias_count,
    void *const *arrays,
    const size_t *element_sizes,
    size_t array_count
) {
    uint32_t packed = 0U;
    for (uint32_t alias = 0U; alias < alias_count; ++alias) {
        const uint32_t start = offsets[alias];
        const uint32_t count = cursor[alias] - start;
        if (packed != start && count != 0U) {
            for (size_t which = 0U; which < array_count; ++which) {
                uint8_t *base = arrays[which];
                memmove(
                    base + (size_t)packed * element_sizes[which],
                    base + (size_t)start * element_sizes[which],
                    (size_t)count * element_sizes[which]
                );
            }
        }
        offsets[alias] = packed;
        packed += count;
    }
    offsets[alias_count] = packed;
}

static ShadowSpillStatus build_alias_lists(
    const ShadowSpillSimulationProgram *program,
    PreparedProblem *prepared
) {
    ShadowSpillPressureFitResidencySparseLists *lists = &prepared->sparse;
    const uint32_t aliases = program->alias_count;
    const uint32_t tasks = program->task_count;
    const size_t offset_slots = (size_t)aliases + 1U;
    lists->anchor_offsets = calloc(offset_slots, sizeof(*lists->anchor_offsets));
    lists->reserved_offsets = calloc(offset_slots, sizeof(*lists->reserved_offsets));
    lists->write_offsets = calloc(offset_slots, sizeof(*lists->write_offsets));
    if (lists->anchor_offsets == NULL || lists->reserved_offsets == NULL ||
        lists->write_offsets == NULL) {
        return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    }
    /* How much room each alias can need: one anchor per access, two per
     * mutation, one each for a declared start and end on the device. */
    for (uint32_t alias = 0U; alias < aliases; ++alias) {
        if (program->alias_size_bytes[alias] == 0U) {
            continue;
        }
        lists->anchor_offsets[alias] =
            (prepared->initial_location[alias] == SHADOWSPILL_MEMORY_DEVICE) +
            (prepared->final_location[alias] == SHADOWSPILL_MEMORY_DEVICE);
    }
    for (uint32_t index = 0U; index < program->input_count; ++index) {
        const uint32_t alias = program->input_aliases[index];
        lists->anchor_offsets[alias] += program->alias_size_bytes[alias] != 0U;
    }
    for (uint32_t index = 0U; index < program->mutation_count; ++index) {
        const uint32_t alias = program->mutation_aliases[index];
        if (program->alias_size_bytes[alias] != 0U) {
            lists->anchor_offsets[alias] += 2U;
            lists->write_offsets[alias] += 1U;
        }
    }
    for (uint32_t index = 0U; index < program->output_count; ++index) {
        const uint32_t alias = program->output_aliases[index];
        if (program->alias_size_bytes[alias] != 0U) {
            lists->anchor_offsets[alias] += 1U;
            lists->write_offsets[alias] += 1U;
            lists->reserved_offsets[alias] += 1U;
        }
    }
    uint64_t anchor_total = 0U;
    uint64_t reserved_total = 0U;
    uint64_t write_total = 0U;
    if (reserve_list(lists->anchor_offsets, aliases, &anchor_total) != 0 ||
        reserve_list(lists->reserved_offsets, aliases, &reserved_total) != 0 ||
        reserve_list(lists->write_offsets, aliases, &write_total) != 0) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    const size_t anchor_slots = anchor_total == 0U ? 1U : (size_t)anchor_total;
    const size_t reserved_slots = reserved_total == 0U ? 1U : (size_t)reserved_total;
    const size_t write_slots = write_total == 0U ? 1U : (size_t)write_total;
    lists->anchor_positions = malloc(anchor_slots * sizeof(*lists->anchor_positions));
    lists->anchor_tasks = malloc(anchor_slots * sizeof(*lists->anchor_tasks));
    lists->anchor_earliest = malloc(anchor_slots * sizeof(*lists->anchor_earliest));
    lists->anchor_flags = malloc(anchor_slots * sizeof(*lists->anchor_flags));
    lists->reserved_positions =
        malloc(reserved_slots * sizeof(*lists->reserved_positions));
    lists->write_boundaries = malloc(write_slots * sizeof(*lists->write_boundaries));
    ListBuild build = {
        .lists = lists,
        .anchor_cursor = malloc(offset_slots * sizeof(*build.anchor_cursor)),
        .reserved_cursor = malloc(offset_slots * sizeof(*build.reserved_cursor)),
        .write_cursor = malloc(offset_slots * sizeof(*build.write_cursor)),
    };
    ShadowSpillStatus status = SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    if (lists->anchor_positions == NULL || lists->anchor_tasks == NULL ||
        lists->anchor_earliest == NULL || lists->anchor_flags == NULL ||
        lists->reserved_positions == NULL || lists->write_boundaries == NULL ||
        build.anchor_cursor == NULL || build.reserved_cursor == NULL ||
        build.write_cursor == NULL) {
        goto done;
    }
    memcpy(build.anchor_cursor, lists->anchor_offsets, offset_slots * sizeof(uint32_t));
    memcpy(build.reserved_cursor, lists->reserved_offsets, offset_slots * sizeof(uint32_t));
    memcpy(build.write_cursor, lists->write_offsets, offset_slots * sizeof(uint32_t));

    /* An alias the step starts holding on the device is anchored at the
     * first boundary, before anything accesses it. */
    for (uint32_t alias = 0U; alias < aliases; ++alias) {
        if (program->alias_size_bytes[alias] != 0U &&
            prepared->initial_location[alias] == SHADOWSPILL_MEMORY_DEVICE) {
            record_anchor(&build, alias, 0U, UINT32_MAX, 0U);
        }
    }
    for (uint32_t task = 0U; task < tasks; ++task) {
        const uint32_t input_start = program->input_offsets[task];
        const uint32_t input_end = program->input_offsets[task + 1U];
        const uint32_t output_start = program->output_offsets[task];
        const uint32_t output_end = program->output_offsets[task + 1U];
        const uint32_t mutation_start = program->mutation_offsets[task];
        const uint32_t mutation_end = program->mutation_offsets[task + 1U];
        for (uint32_t index = input_start; index < input_end; ++index) {
            const uint32_t alias = program->input_aliases[index];
            if (program->alias_size_bytes[alias] == 0U) {
                continue;
            }
            prepared->seen_input[alias] = 1U;
            if (prepared->first_input_task[alias] == UINT32_MAX) {
                prepared->first_input_task[alias] = task;
            }
            record_anchor(&build, alias, task, task, SHADOWSPILL_ANCHOR_INPUT);
        }
        for (uint32_t index = mutation_start; index < mutation_end; ++index) {
            const uint32_t alias = program->mutation_aliases[index];
            if (program->alias_size_bytes[alias] == 0U) {
                continue;
            }
            record_anchor(&build, alias, task, task, 0U);
            record_anchor(&build, alias, task + 1U, UINT32_MAX, 0U);
            record_write(&build, alias, task + 1U);
        }
        for (uint32_t index = output_start; index < output_end; ++index) {
            const uint32_t alias = program->output_aliases[index];
            if (program->alias_size_bytes[alias] == 0U) {
                continue;
            }
            record_anchor(&build, alias, task + 1U, task, SHADOWSPILL_ANCHOR_PRODUCED);
            record_write(&build, alias, task + 1U);
            prepared->produced[alias] = 1U;
            if (!contains_alias(program->input_aliases, input_start, input_end,
                                alias) &&
                !contains_alias(program->mutation_aliases, mutation_start,
                                mutation_end, alias)) {
                record_reservation(&build, alias, task);
            }
        }
    }
    /* And one the step must end holding on the device is anchored at the
     * last boundary, after everything. */
    for (uint32_t alias = 0U; alias < aliases; ++alias) {
        if (program->alias_size_bytes[alias] != 0U &&
            prepared->final_location[alias] == SHADOWSPILL_MEMORY_DEVICE) {
            record_anchor(&build, alias, tasks, UINT32_MAX, 0U);
        }
    }
    {
        void *const anchor_arrays[] = {
            lists->anchor_positions, lists->anchor_tasks,
            lists->anchor_earliest, lists->anchor_flags,
        };
        const size_t anchor_sizes[] = {
            sizeof(*lists->anchor_positions), sizeof(*lists->anchor_tasks),
            sizeof(*lists->anchor_earliest), sizeof(*lists->anchor_flags),
        };
        pack_list(lists->anchor_offsets, build.anchor_cursor, aliases,
                  anchor_arrays, anchor_sizes, 4U);
        void *const reserved_arrays[] = {lists->reserved_positions};
        const size_t reserved_sizes[] = {sizeof(*lists->reserved_positions)};
        pack_list(lists->reserved_offsets, build.reserved_cursor, aliases,
                  reserved_arrays, reserved_sizes, 1U);
        void *const write_arrays[] = {lists->write_boundaries};
        const size_t write_sizes[] = {sizeof(*lists->write_boundaries)};
        pack_list(lists->write_offsets, build.write_cursor, aliases,
                  write_arrays, write_sizes, 1U);
    }
    status = SHADOWSPILL_STATUS_OK;
done:
    free(build.anchor_cursor);
    free(build.reserved_cursor);
    free(build.write_cursor);
    return status;
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
                         program->mutation_count) != 0 ||
        !aliases_in_range(program->input_aliases, program->input_count,
                          program->alias_count) ||
        !aliases_in_range(program->output_aliases, program->output_count,
                          program->alias_count) ||
        !aliases_in_range(program->mutation_aliases, program->mutation_count,
                          program->alias_count)) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }

    for (uint32_t device = 0U; device < program->device_count; ++device) {
        prepared->device_capacity_bytes[device] =
            program->devices[device].capacity_bytes;
    }
    uint64_t ideal_end = 0U;
    for (uint32_t task = 0U; task < program->task_count; ++task) {
        uint32_t device = program->task_device[task];
        if (device >= program->device_count ||
            shadowspill_problem_add_u64(ideal_end, program->task_runtime_ns[task], &ideal_end) != 0) {
            return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
        }
        prepared->task_ideal_end_ns[task] = ideal_end;
        prepared->boundary_capacity_bytes[
            (uint64_t)device * boundary_count + task
        ] = program->task_workspace_bytes[task];
    }
    return build_alias_lists(program, prepared);
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
    for (uint32_t alias = 0U; alias < program->alias_count; ++alias) {
        uint32_t device = program->alias_device[alias];
        if (device >= program->device_count ||
            shadowspill_transfer_duration_ns(
                program, alias, SHADOWSPILL_TRANSFER_FETCH,
                (program->devices[device].fetch_solo_bandwidth_bytes_per_second > program->devices[device].fetch_concurrent_bandwidth_bytes_per_second
                    ? program->devices[device].fetch_solo_bandwidth_bytes_per_second
                    : program->devices[device].fetch_concurrent_bandwidth_bytes_per_second),
                &prepared->fetch_runtime_ns[alias]
            ) != 0 ||
            shadowspill_transfer_duration_ns(
                program, alias, SHADOWSPILL_TRANSFER_EVICT,
                (program->devices[device].evict_solo_bandwidth_bytes_per_second > program->devices[device].evict_concurrent_bandwidth_bytes_per_second
                    ? program->devices[device].evict_solo_bandwidth_bytes_per_second
                    : program->devices[device].evict_concurrent_bandwidth_bytes_per_second),
                &prepared->evict_runtime_ns[alias]
            ) != 0) {
            return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
        }
        if (prepared->seen_input[alias] != 0U &&
            prepared->first_input_task[alias] == 0U &&
            prepared->produced[alias] == 0U &&
            program->alias_size_bytes[alias] != 0U &&
            prepared->initial_location[alias] != SHADOWSPILL_MEMORY_DEVICE) {
            prepared->failure_kind =
                SHADOWSPILL_PRESSUREFIT_PREFLIGHT_MISSING_INITIAL_RESIDENCY;
            prepared->error_alias = alias;
            prepared->error_boundary = 0;
            return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
        }
    }
    return SHADOWSPILL_STATUS_OK;
}
