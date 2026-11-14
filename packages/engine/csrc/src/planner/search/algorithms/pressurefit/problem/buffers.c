/* The arrays one prepared problem owns, and their lifetime. */
#include "internal.h"

int shadowspill_problem_allocate_prepared_buffers(
    const ShadowSpillSimulationProgram *program,
    PreparedProblem *prepared
) {
    size_t pressure_cells = 0U;
    uint32_t boundary_count = program->task_count + 1U;
    uint64_t pressure = (uint64_t)program->device_count * boundary_count;
    if (boundary_count == 0U || pressure > SIZE_MAX) {
        return -1;
    }
    pressure_cells = (size_t)pressure;
    size_t aliases = program->alias_count == 0U ? 1U : program->alias_count;
    prepared->initial_location = malloc(
        aliases * sizeof(*prepared->initial_location)
    );
    prepared->final_location = malloc(
        aliases * sizeof(*prepared->final_location)
    );
    prepared->first_input_task = malloc(
        aliases * sizeof(*prepared->first_input_task)
    );
    prepared->fetch_runtime_ns = calloc(
        aliases,
        sizeof(*prepared->fetch_runtime_ns)
    );
    prepared->evict_runtime_ns = calloc(
        aliases,
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
    prepared->boundary_capacity_bytes = calloc(
        pressure_cells,
        sizeof(*prepared->boundary_capacity_bytes)
    );
    prepared->produced = calloc(aliases, sizeof(*prepared->produced));
    prepared->seen_input = calloc(
        aliases,
        sizeof(*prepared->seen_input)
    );
    prepared->required_bytes = calloc(
        pressure_cells,
        sizeof(*prepared->required_bytes)
    );
    prepared->evict_eligible = malloc(aliases * sizeof(*prepared->evict_eligible));
    prepared->fixed_fetch_trigger = malloc(
        aliases * sizeof(*prepared->fixed_fetch_trigger)
    );
    prepared->resident_slice_bytes = calloc(
        program->device_count,
        sizeof(*prepared->resident_slice_bytes)
    );
    if (prepared->initial_location == NULL ||
        prepared->final_location == NULL ||
        prepared->first_input_task == NULL ||
        prepared->fetch_runtime_ns == NULL ||
        prepared->evict_runtime_ns == NULL ||
        prepared->task_ideal_end_ns == NULL ||
        prepared->device_capacity_bytes == NULL ||
        prepared->boundary_capacity_bytes == NULL ||
        prepared->produced == NULL ||
        prepared->seen_input == NULL ||
        prepared->required_bytes == NULL || prepared->evict_eligible == NULL ||
        prepared->fixed_fetch_trigger == NULL ||
        prepared->resident_slice_bytes == NULL) {
        return -1;
    }
    memset(prepared->evict_eligible, 1, program->alias_count);
    memset(prepared->fixed_fetch_trigger, 0xff,
           (size_t)program->alias_count * sizeof(*prepared->fixed_fetch_trigger));
    memset(prepared->initial_location, -1, program->alias_count);
    memset(prepared->final_location, -1, program->alias_count);
    memset(prepared->first_input_task, 0xff,
           (size_t)program->alias_count * sizeof(*prepared->first_input_task));
    return 0;
}

void shadowspill_problem_prepared_problem_destroy(PreparedProblem *prepared) {
    free(prepared->initial_location);
    free(prepared->final_location);
    shadowspill_residency_sparse_lists_destroy(&prepared->sparse);
    free(prepared->first_input_task);
    free(prepared->fetch_runtime_ns);
    free(prepared->evict_runtime_ns);
    free(prepared->task_ideal_end_ns);
    free(prepared->device_capacity_bytes);
    free(prepared->boundary_capacity_bytes);
    shadowspill_residency_destroy(&prepared->seed);
    free(prepared->produced);
    free(prepared->seen_input);
    free(prepared->required_bytes);
    free(prepared->evict_eligible);
    free(prepared->fixed_fetch_trigger);
    free(prepared->resident_slice_bytes);
    memset(prepared, 0, sizeof(*prepared));
}
