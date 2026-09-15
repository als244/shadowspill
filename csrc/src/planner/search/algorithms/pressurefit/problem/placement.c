/* What starts resident: the coldest aliases, placed greedily. */
#include "internal.h"

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

static int compare_cold_deadline(const void *left_value, const void *right_value) {
    const ColdAlias *left = left_value;
    const ColdAlias *right = right_value;
    int result = shadowspill_problem_compare_u64(left->deadline, right->deadline);
    if (result == 0) {
        result = shadowspill_problem_compare_u32(left->first_use, right->first_use);
    }
    return result == 0 ? shadowspill_problem_compare_u32(left->alias, right->alias) : result;
}

static int compare_cold_placement(
    const void *left_value,
    const void *right_value
) {
    const ColdAlias *left = left_value;
    const ColdAlias *right = right_value;
    int result = shadowspill_problem_compare_u32(left->first_use, right->first_use);
    if (result == 0) {
        result = shadowspill_problem_compare_u64(left->slack_ns, right->slack_ns);
    }
    if (result == 0) {
        result = shadowspill_problem_compare_u64(right->miss_ns, left->miss_ns);
    }
    if (result == 0) {
        result = shadowspill_problem_compare_u64(right->size_bytes, left->size_bytes);
    }
    return result == 0 ? shadowspill_problem_compare_u32(left->alias, right->alias) : result;
}

void shadowspill_problem_build_anchor_seed(
    const ShadowSpillSimulationProgram *program,
    PreparedProblem *prepared
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

ShadowSpillStatus shadowspill_problem_greedily_place_initial_aliases(
    const ShadowSpillSimulationProgram *program,
    PreparedProblem *prepared
) {
    uint32_t cold_count = 0U;
    for (uint32_t alias = 0U; alias < program->alias_count; ++alias) {
        if (prepared->evict_eligible[alias] != 0U &&
            prepared->initial_location[alias] == SHADOWSPILL_MEMORY_SPILL &&
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
        return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    }
    uint32_t boundary_count = program->task_count + 1U;
    uint32_t next = 0U;
    for (uint32_t alias = 0U; alias < program->alias_count; ++alias) {
        uint32_t first_use = prepared->first_access_task[alias];
        if (prepared->evict_eligible[alias] == 0U ||
            prepared->initial_location[alias] != SHADOWSPILL_MEMORY_SPILL ||
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
        if (shadowspill_problem_add_u64(cursor[value->device], value->transfer_ns, &finish) != 0) {
            free(cold);
            free(cursor);
            free(initial_bytes);
            return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
        }
        value->miss_ns = finish > value->deadline
            ? finish - value->deadline
            : 0U;
        cursor[value->device] = finish;
        uint64_t unavailable = first_task_end;
        if (shadowspill_problem_add_u64(unavailable, value->transfer_ns, &unavailable) != 0) {
            free(cold);
            free(cursor);
            free(initial_bytes);
            return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
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
            shadowspill_problem_add_u64(initial_bytes[device], program->alias_size_bytes[alias],
                    &initial_bytes[device]) != 0) {
            free(cold);
            free(cursor);
            free(initial_bytes);
            return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
        }
    }
    qsort(cold, cold_count, sizeof(*cold), compare_cold_placement);
    for (uint32_t index = 0U; index < cold_count; ++index) {
        ColdAlias *value = &cold[index];
        uint64_t proposed = 0U;
        if (shadowspill_problem_add_u64(initial_bytes[value->device], value->size_bytes, &proposed) != 0) {
            free(cold);
            free(cursor);
            free(initial_bytes);
            return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
        }
        if (proposed > prepared->boundary_capacity_bytes[
                (uint64_t)value->device * (program->task_count + 1U)
            ]) {
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
    return SHADOWSPILL_STATUS_OK;
}
