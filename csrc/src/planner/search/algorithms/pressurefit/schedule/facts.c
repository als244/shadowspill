/* The write events a program has, indexed once per problem. */
#include "internal.h"

uint64_t shadowspill_schedule_cell(uint32_t alias, uint32_t count, uint32_t index) {
    return (uint64_t)alias * count + index;
}

int shadowspill_schedule_checked_cells(uint32_t left, uint32_t right, size_t *result) {
    if (left != 0U && (size_t)right > SIZE_MAX / left) {
        return -1;
    }
    *result = (size_t)left * right;
    return 0;
}

static void record_earliest(
    ShadowSpillScheduleFacts *facts,
    uint32_t alias,
    uint32_t index,
    uint32_t task
) {
    uint64_t position = shadowspill_schedule_cell(alias, facts->boundary_count, index);
    if (facts->earliest_access_task[position] == UINT32_MAX ||
        task < facts->earliest_access_task[position]) {
        facts->earliest_access_task[position] = task;
    }
}

/* Index the write events per alias, so a pass can find the last write before
 * a boundary by search rather than by walking the row. */
static int index_write_events(ShadowSpillScheduleFacts *facts) {
    facts->write_offsets = calloc(
        (size_t)facts->alias_count + 1U, sizeof(*facts->write_offsets)
    );
    if (facts->write_offsets == NULL) {
        return -1;
    }
    size_t total = 0U;
    for (uint32_t alias = 0U; alias < facts->alias_count; ++alias) {
        facts->write_offsets[alias + 1U] = facts->write_offsets[alias];
        for (uint32_t boundary = 0U; boundary < facts->boundary_count;
             ++boundary) {
            if (facts->write_events[
                    shadowspill_schedule_cell(alias, facts->boundary_count, boundary)
                ] != 0U) {
                ++facts->write_offsets[alias + 1U];
                ++total;
            }
        }
    }
    facts->write_boundaries = malloc(
        (total == 0U ? 1U : total) * sizeof(*facts->write_boundaries)
    );
    if (facts->write_boundaries == NULL) {
        return -1;
    }
    size_t position = 0U;
    for (uint32_t alias = 0U; alias < facts->alias_count; ++alias) {
        for (uint32_t boundary = 0U; boundary < facts->boundary_count;
             ++boundary) {
            if (facts->write_events[
                    shadowspill_schedule_cell(alias, facts->boundary_count, boundary)
                ] != 0U) {
                facts->write_boundaries[position++] = boundary;
            }
        }
    }
    return 0;
}

int shadowspill_schedule_facts_create(
    const ShadowSpillPressureFitProblem *problem,
    ShadowSpillScheduleFacts *facts
) {
    if (problem == NULL || facts == NULL || problem->residency == NULL ||
        problem->context.simulation == NULL ||
        problem->abi_version != SHADOWSPILL_ABI_VERSION ||
        problem->residency->abi_version != SHADOWSPILL_ABI_VERSION ||
        problem->context.simulation->abi_version != SHADOWSPILL_ABI_VERSION ||
        problem->residency->alias_count != problem->context.simulation->alias_count ||
        problem->residency->device_count != problem->context.simulation->device_count ||
        problem->residency->boundary_count !=
            problem->context.simulation->task_count + 1U) {
        return -1;
    }
    memset(facts, 0, sizeof(*facts));
    facts->problem = problem;
    facts->alias_count = problem->residency->alias_count;
    facts->task_count = problem->context.simulation->task_count;
    facts->boundary_count = problem->residency->boundary_count;
    facts->device_count = problem->residency->device_count;

    size_t cells = 0U;
    if (shadowspill_schedule_checked_cells(facts->alias_count, facts->boundary_count, &cells) != 0) {
        return -1;
    }
    facts->earliest_access_task = malloc(
        (cells == 0U ? 1U : cells) * sizeof(*facts->earliest_access_task)
    );
    facts->write_events = calloc(
        cells == 0U ? 1U : cells,
        sizeof(*facts->write_events)
    );
    if (facts->earliest_access_task == NULL || facts->write_events == NULL) {
        shadowspill_schedule_facts_destroy(facts);
        return -1;
    }
    for (size_t position = 0U; position < cells; ++position) {
        facts->earliest_access_task[position] = UINT32_MAX;
    }

    const ShadowSpillSimulationProgram *program = problem->context.simulation;
    for (uint32_t task = 0U; task < facts->task_count; ++task) {
        for (uint32_t offset = program->input_offsets[task];
             offset < program->input_offsets[task + 1U];
             ++offset) {
            record_earliest(facts, program->input_aliases[offset], task, task);
        }
        for (uint32_t offset = program->mutation_offsets[task];
             offset < program->mutation_offsets[task + 1U];
             ++offset) {
            uint32_t alias = program->mutation_aliases[offset];
            record_earliest(facts, alias, task, task);
            facts->write_events[shadowspill_schedule_cell(alias, facts->boundary_count, task + 1U)] =
                1U;
        }
        for (uint32_t offset = program->output_offsets[task];
             offset < program->output_offsets[task + 1U];
             ++offset) {
            uint32_t alias = program->output_aliases[offset];
            record_earliest(facts, alias, task + 1U, task);
            facts->write_events[shadowspill_schedule_cell(alias, facts->boundary_count, task + 1U)] =
                1U;
        }
    }
    return index_write_events(facts);
}

void shadowspill_schedule_facts_destroy(ShadowSpillScheduleFacts *facts) {
    if (facts == NULL) {
        return;
    }
    free(facts->earliest_access_task);
    free(facts->write_events);
    free(facts->write_offsets);
    free(facts->write_boundaries);
    memset(facts, 0, sizeof(*facts));
}
