#include "portfolio_internal.h"

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct Span {
    uint32_t start;
    uint32_t end;
} Span;

typedef struct Reload {
    uint32_t alias;
    uint32_t earliest_trigger;
    uint32_t latest_trigger;
    uint32_t entry_boundary;
    uint32_t ordinal;
    uint32_t trigger;
} Reload;

typedef struct Departure {
    uint32_t alias;
    uint32_t trigger;
    uint8_t kind;
} Departure;

typedef struct Action {
    uint32_t trigger;
    uint32_t alias;
    uint8_t kind;
} Action;

static uint64_t cell(uint32_t alias, uint32_t count, uint32_t index) {
    return (uint64_t)alias * count + index;
}

static int checked_cells(uint32_t left, uint32_t right, size_t *result) {
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
    uint64_t position = cell(alias, facts->boundary_count, index);
    if (facts->earliest_access_task[position] == UINT32_MAX ||
        task < facts->earliest_access_task[position]) {
        facts->earliest_access_task[position] = task;
    }
}

int shadowspill_schedule_facts_create(
    const ShadowSpillPressureFitContext *context,
    ShadowSpillScheduleFacts *facts
) {
    if (context == NULL || facts == NULL || context->residency == NULL ||
        context->simulation == NULL ||
        context->abi_version != SHADOWSPILL_PLANNER_ABI_VERSION ||
        context->residency->abi_version != SHADOWSPILL_PLANNER_ABI_VERSION ||
        context->simulation->abi_version != SHADOWSPILL_SIMULATOR_ABI_VERSION ||
        context->residency->alias_count != context->simulation->alias_count ||
        context->residency->device_count != context->simulation->device_count ||
        context->residency->boundary_count !=
            context->simulation->task_count + 1U) {
        return -1;
    }
    memset(facts, 0, sizeof(*facts));
    facts->context = context;
    facts->alias_count = context->residency->alias_count;
    facts->task_count = context->simulation->task_count;
    facts->boundary_count = context->residency->boundary_count;
    facts->device_count = context->residency->device_count;

    size_t cells = 0U;
    if (checked_cells(facts->alias_count, facts->boundary_count, &cells) != 0) {
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

    const ShadowSpillSimulationProgram *program = context->simulation;
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
            facts->write_events[cell(alias, facts->boundary_count, task + 1U)] =
                1U;
        }
        for (uint32_t offset = program->output_offsets[task];
             offset < program->output_offsets[task + 1U];
             ++offset) {
            uint32_t alias = program->output_aliases[offset];
            record_earliest(facts, alias, task + 1U, task);
            facts->write_events[cell(alias, facts->boundary_count, task + 1U)] =
                1U;
        }
    }
    return 0;
}

void shadowspill_schedule_facts_destroy(ShadowSpillScheduleFacts *facts) {
    if (facts == NULL) {
        return;
    }
    free(facts->earliest_access_task);
    free(facts->write_events);
    memset(facts, 0, sizeof(*facts));
}

int shadowspill_schedule_storage_create(
    uint32_t alias_count,
    uint32_t task_count,
    ShadowSpillScheduleStorage *storage
) {
    if (storage == NULL) {
        return -1;
    }
    memset(storage, 0, sizeof(*storage));
    uint64_t action_capacity = (uint64_t)alias_count * (task_count + 1U) * 2U;
    if (action_capacity > UINT32_MAX) {
        return -1;
    }
    storage->action_capacity = (uint32_t)action_capacity;
    storage->initial_capacity = alias_count;
    storage->final_capacity = alias_count;
    uint32_t actions = storage->action_capacity == 0U
        ? 1U
        : storage->action_capacity;
    uint32_t aliases = alias_count == 0U ? 1U : alias_count;
    storage->value.action_trigger_tasks = calloc(
        actions,
        sizeof(*storage->value.action_trigger_tasks)
    );
    storage->value.action_aliases = calloc(
        actions,
        sizeof(*storage->value.action_aliases)
    );
    storage->value.action_kinds = calloc(
        actions,
        sizeof(*storage->value.action_kinds)
    );
    storage->value.initial_aliases = calloc(
        aliases,
        sizeof(*storage->value.initial_aliases)
    );
    storage->value.initial_locations = calloc(
        aliases,
        sizeof(*storage->value.initial_locations)
    );
    storage->value.final_aliases = calloc(
        aliases,
        sizeof(*storage->value.final_aliases)
    );
    storage->value.final_locations = calloc(
        aliases,
        sizeof(*storage->value.final_locations)
    );
    if (storage->value.action_trigger_tasks == NULL ||
        storage->value.action_aliases == NULL ||
        storage->value.action_kinds == NULL ||
        storage->value.initial_aliases == NULL ||
        storage->value.initial_locations == NULL ||
        storage->value.final_aliases == NULL ||
        storage->value.final_locations == NULL) {
        shadowspill_schedule_storage_destroy(storage);
        return -1;
    }
    return 0;
}

void shadowspill_schedule_storage_clear(ShadowSpillScheduleStorage *storage) {
    if (storage == NULL) {
        return;
    }
    storage->value.action_count = 0U;
    storage->value.initial_count = 0U;
    storage->value.final_count = 0U;
}

void shadowspill_schedule_storage_destroy(ShadowSpillScheduleStorage *storage) {
    if (storage == NULL) {
        return;
    }
    free(storage->value.action_trigger_tasks);
    free(storage->value.action_aliases);
    free(storage->value.action_kinds);
    free(storage->value.initial_aliases);
    free(storage->value.initial_locations);
    free(storage->value.final_aliases);
    free(storage->value.final_locations);
    memset(storage, 0, sizeof(*storage));
}

int shadowspill_schedule_storage_copy(
    ShadowSpillScheduleStorage *destination,
    const ShadowSpillScheduleStorage *source
) {
    if (destination == NULL || source == NULL ||
        destination->action_capacity < source->value.action_count ||
        destination->initial_capacity < source->value.initial_count ||
        destination->final_capacity < source->value.final_count) {
        return -1;
    }
    destination->value.action_count = source->value.action_count;
    destination->value.initial_count = source->value.initial_count;
    destination->value.final_count = source->value.final_count;
    memcpy(
        destination->value.action_trigger_tasks,
        source->value.action_trigger_tasks,
        (size_t)source->value.action_count *
            sizeof(*source->value.action_trigger_tasks)
    );
    memcpy(
        destination->value.action_aliases,
        source->value.action_aliases,
        (size_t)source->value.action_count * sizeof(*source->value.action_aliases)
    );
    memcpy(
        destination->value.action_kinds,
        source->value.action_kinds,
        (size_t)source->value.action_count * sizeof(*source->value.action_kinds)
    );
    memcpy(
        destination->value.initial_aliases,
        source->value.initial_aliases,
        (size_t)source->value.initial_count *
            sizeof(*source->value.initial_aliases)
    );
    memcpy(
        destination->value.initial_locations,
        source->value.initial_locations,
        (size_t)source->value.initial_count *
            sizeof(*source->value.initial_locations)
    );
    memcpy(
        destination->value.final_aliases,
        source->value.final_aliases,
        (size_t)source->value.final_count * sizeof(*source->value.final_aliases)
    );
    memcpy(
        destination->value.final_locations,
        source->value.final_locations,
        (size_t)source->value.final_count * sizeof(*source->value.final_locations)
    );
    return 0;
}

static uint32_t collect_spans(
    const uint8_t *resident,
    const uint8_t *breaks,
    uint32_t alias,
    uint32_t boundary_count,
    Span *spans
) {
    uint32_t count = 0U;
    uint32_t index = 0U;
    while (index < boundary_count) {
        if (resident[cell(alias, boundary_count, index)] == 0U) {
            ++index;
            continue;
        }
        uint32_t start = index;
        while (index + 1U < boundary_count &&
               resident[cell(alias, boundary_count, index + 1U)] != 0U &&
               breaks[cell(alias, boundary_count, index)] == 0U) {
            ++index;
        }
        spans[count++] = (Span){.start = start, .end = index};
        ++index;
    }
    return count;
}

static int has_future_access(
    const ShadowSpillScheduleFacts *facts,
    uint32_t alias,
    const Span *span
) {
    const ShadowSpillResidencyProblem *problem = facts->context->residency;
    int32_t end_boundary = (int32_t)span->end - 1;
    for (uint32_t index = span->start; index <= span->end; ++index) {
        uint32_t task = problem->latest_access_task[cell(
            alias,
            facts->boundary_count,
            index
        )];
        if (task != UINT32_MAX && (int32_t)task > end_boundary) {
            return 1;
        }
    }
    return 0;
}

static void alias_contribution(
    const ShadowSpillScheduleFacts *facts,
    const uint8_t *resident,
    const uint8_t *breaks,
    uint32_t alias,
    int prefetch_headroom,
    uint8_t *contribution,
    Span *spans
) {
    const ShadowSpillResidencyProblem *problem = facts->context->residency;
    memset(contribution, 0, facts->boundary_count);
    uint32_t span_count = collect_spans(
        resident,
        breaks,
        alias,
        facts->boundary_count,
        spans
    );
    for (uint32_t index = 0U; index < span_count; ++index) {
        uint32_t start = spans[index].start;
        if (prefetch_headroom != 0 && start > 0U &&
            problem->productions[cell(alias, facts->boundary_count, start)] ==
                0U) {
            --start;
        }
        int32_t end = (int32_t)spans[index].end;
        if (spans[index].end > 0U && problem->final_location[alias] != 0 &&
            !has_future_access(facts, alias, &spans[index])) {
            --end;
        }
        if (end >= (int32_t)start) {
            for (uint32_t boundary = start; boundary <= (uint32_t)end;
                 ++boundary) {
                contribution[boundary] = 1U;
            }
        }
    }
    for (uint32_t boundary = 0U; boundary < facts->boundary_count; ++boundary) {
        if (problem->output_reservations[cell(
                alias,
                facts->boundary_count,
                boundary
            )] != 0U) {
            contribution[boundary] = 1U;
        }
    }
}

static int build_pressure(
    const ShadowSpillScheduleFacts *facts,
    const uint8_t *resident,
    const uint8_t *breaks,
    int prefetch_headroom,
    uint64_t *pressure
) {
    size_t pressure_cells = 0U;
    if (checked_cells(facts->device_count, facts->boundary_count, &pressure_cells) !=
        0) {
        return -1;
    }
    memset(pressure, 0, pressure_cells * sizeof(*pressure));
    uint8_t *contribution = calloc(
        facts->boundary_count,
        sizeof(*contribution)
    );
    Span *spans = calloc(facts->boundary_count, sizeof(*spans));
    if (contribution == NULL || spans == NULL) {
        free(contribution);
        free(spans);
        return -1;
    }
    const ShadowSpillResidencyProblem *problem = facts->context->residency;
    for (uint32_t alias = 0U; alias < facts->alias_count; ++alias) {
        alias_contribution(
            facts,
            resident,
            breaks,
            alias,
            prefetch_headroom,
            contribution,
            spans
        );
        uint32_t device = problem->alias_device[alias];
        for (uint32_t boundary = 0U; boundary < facts->boundary_count;
             ++boundary) {
            if (contribution[boundary] != 0U) {
                pressure[(uint64_t)device * facts->boundary_count + boundary] +=
                    problem->alias_size_bytes[alias];
            }
        }
    }
    free(contribution);
    free(spans);
    return 0;
}

int shadowspill_extend_interval_entries(
    const ShadowSpillScheduleFacts *facts,
    uint8_t *resident,
    uint8_t *breaks
) {
    if (facts == NULL || resident == NULL || breaks == NULL) {
        return -1;
    }
    size_t pressure_cells = 0U;
    if (checked_cells(facts->device_count, facts->boundary_count, &pressure_cells) !=
        0) {
        return -1;
    }
    uint64_t *pressure = calloc(
        pressure_cells == 0U ? 1U : pressure_cells,
        sizeof(*pressure)
    );
    Span *spans = calloc(facts->boundary_count, sizeof(*spans));
    if (pressure == NULL || spans == NULL ||
        build_pressure(facts, resident, breaks, 0, pressure) != 0) {
        free(pressure);
        free(spans);
        return -1;
    }
    const ShadowSpillResidencyProblem *problem = facts->context->residency;
    for (uint32_t alias = 0U; alias < facts->alias_count; ++alias) {
        uint32_t span_index = 1U;
        while (1) {
            uint32_t span_count = collect_spans(
                resident,
                breaks,
                alias,
                facts->boundary_count,
                spans
            );
            if (span_index >= span_count) {
                break;
            }
            Span current = spans[span_index];
            Span previous = spans[span_index - 1U];
            if (current.start == 0U) {
                ++span_index;
                continue;
            }
            int32_t candidate_boundary = (int32_t)current.start - 2;
            int32_t previous_end = (int32_t)previous.end - 1;
            if (candidate_boundary <= previous_end) {
                ++span_index;
                continue;
            }
            uint32_t candidate_cell = current.start - 1U;
            uint32_t device = problem->alias_device[alias];
            uint64_t position =
                (uint64_t)device * facts->boundary_count + candidate_cell;
            uint64_t added = problem->output_reservations[cell(
                alias,
                facts->boundary_count,
                candidate_cell
            )] != 0U
                ? 0U
                : problem->alias_size_bytes[alias];
            if (pressure[position] + added <=
                problem->device_capacity_bytes[device]) {
                resident[cell(alias, facts->boundary_count, candidate_cell)] = 1U;
                pressure[position] += added;
                continue;
            }
            ++span_index;
        }
    }
    free(pressure);
    free(spans);
    return 0;
}

static uint32_t event_min_task(
    const ShadowSpillScheduleFacts *facts,
    uint32_t alias,
    const Span *span
) {
    uint32_t selected = UINT32_MAX;
    for (uint32_t index = span->start; index <= span->end; ++index) {
        uint32_t task = facts->earliest_access_task[cell(
            alias,
            facts->boundary_count,
            index
        )];
        if (task != UINT32_MAX && (selected == UINT32_MAX || task < selected)) {
            selected = task;
        }
    }
    return selected;
}

static uint32_t event_max_task(
    const ShadowSpillScheduleFacts *facts,
    uint32_t alias,
    const Span *span
) {
    const ShadowSpillResidencyProblem *problem = facts->context->residency;
    uint32_t selected = UINT32_MAX;
    for (uint32_t index = span->start; index <= span->end; ++index) {
        uint32_t task = problem->latest_access_task[cell(
            alias,
            facts->boundary_count,
            index
        )];
        if (task != UINT32_MAX && (selected == UINT32_MAX || task > selected)) {
            selected = task;
        }
    }
    return selected;
}

static int has_write_since(
    const ShadowSpillScheduleFacts *facts,
    uint32_t alias,
    int32_t refreshed_at,
    int32_t through
) {
    int32_t first = refreshed_at + 1;
    if (first < 0) {
        first = 0;
    }
    for (int32_t boundary = first; boundary <= through; ++boundary) {
        uint32_t index = (uint32_t)(boundary + 1);
        if (facts->write_events[cell(alias, facts->boundary_count, index)] != 0U) {
            return 1;
        }
    }
    return 0;
}

static int reload_compare_descending(const void *left_value, const void *right_value) {
    const Reload *left = left_value;
    const Reload *right = right_value;
    if (left->latest_trigger != right->latest_trigger) {
        return left->latest_trigger > right->latest_trigger ? -1 : 1;
    }
    if (left->alias != right->alias) {
        return left->alias > right->alias ? -1 : 1;
    }
    if (left->ordinal != right->ordinal) {
        return left->ordinal < right->ordinal ? -1 : 1;
    }
    return 0;
}

static uint64_t ideal_trigger_time(
    const ShadowSpillResidencyProblem *problem,
    uint32_t trigger
) {
    return problem->task_ideal_end_ns[trigger];
}

static uint32_t latest_trigger_at_or_before(
    const ShadowSpillResidencyProblem *problem,
    uint32_t earliest,
    uint32_t latest,
    uint64_t target_ns
) {
    uint32_t insertion = earliest;
    while (insertion <= latest &&
           problem->task_ideal_end_ns[insertion] <= target_ns) {
        ++insertion;
    }
    return insertion == earliest ? earliest : insertion - 1U;
}

static void choose_packed_triggers(
    const ShadowSpillScheduleFacts *facts,
    Reload *reloads,
    uint32_t reload_count
) {
    const ShadowSpillResidencyProblem *problem = facts->context->residency;
    qsort(reloads, reload_count, sizeof(*reloads), reload_compare_descending);
    uint64_t *packed_start = calloc(facts->device_count, sizeof(*packed_start));
    uint8_t *has_packed_start = calloc(
        facts->device_count,
        sizeof(*has_packed_start)
    );
    if (packed_start == NULL || has_packed_start == NULL) {
        free(packed_start);
        free(has_packed_start);
        for (uint32_t index = 0U; index < reload_count; ++index) {
            reloads[index].trigger = reloads[index].latest_trigger;
        }
        return;
    }
    for (uint32_t index = 0U; index < reload_count; ++index) {
        Reload *reload = &reloads[index];
        uint32_t device = problem->alias_device[reload->alias];
        uint64_t deadline = ideal_trigger_time(problem, reload->latest_trigger);
        uint64_t finish = has_packed_start[device] != 0U &&
                packed_start[device] < deadline
            ? packed_start[device]
            : deadline;
        uint64_t runtime = problem->fetch_runtime_ns[reload->alias];
        uint64_t desired = finish > runtime ? finish - runtime : 0U;
        reload->trigger = latest_trigger_at_or_before(
            problem,
            reload->earliest_trigger,
            reload->latest_trigger,
            desired
        );
        uint64_t trigger_time = ideal_trigger_time(problem, reload->trigger);
        packed_start[device] = trigger_time > desired ? trigger_time : desired;
        has_packed_start[device] = 1U;
    }
    free(packed_start);
    free(has_packed_start);
}

static int clamp_triggers_to_fit(
    const ShadowSpillScheduleFacts *facts,
    const uint8_t *resident,
    const uint8_t *breaks,
    Reload *reloads,
    uint32_t reload_count,
    int prefetch_headroom
) {
    const ShadowSpillResidencyProblem *problem = facts->context->residency;
    size_t pressure_cells = 0U;
    size_t active_cells = 0U;
    if (checked_cells(facts->device_count, facts->boundary_count, &pressure_cells) !=
            0 ||
        checked_cells(reload_count, facts->task_count, &active_cells) != 0) {
        return -1;
    }
    uint64_t *used = calloc(
        pressure_cells == 0U ? 1U : pressure_cells,
        sizeof(*used)
    );
    uint32_t *counts = calloc(
        (size_t)facts->alias_count * (facts->task_count == 0U ? 1U : facts->task_count),
        sizeof(*counts)
    );
    uint8_t *active = calloc(active_cells == 0U ? 1U : active_cells, 1U);
    if (used == NULL || counts == NULL || active == NULL ||
        build_pressure(
            facts,
            resident,
            breaks,
            prefetch_headroom,
            used
        ) != 0) {
        free(used);
        free(counts);
        free(active);
        return -1;
    }

    for (uint32_t reload_index = 0U; reload_index < reload_count; ++reload_index) {
        Reload *reload = &reloads[reload_index];
        for (uint32_t boundary = reload->trigger;
             boundary < reload->entry_boundary;
             ++boundary) {
            if (resident[cell(
                    reload->alias,
                    facts->boundary_count,
                    boundary + 1U
                )] != 0U) {
                continue;
            }
            active[(uint64_t)reload_index * facts->task_count + boundary] = 1U;
            uint64_t count_position =
                (uint64_t)reload->alias * facts->task_count + boundary;
            if (counts[count_position]++ == 0U) {
                uint32_t device = problem->alias_device[reload->alias];
                used[(uint64_t)device * facts->boundary_count + boundary + 1U] +=
                    problem->alias_size_bytes[reload->alias];
            }
        }
    }

    while (1) {
        uint32_t selected_reload = UINT32_MAX;
        uint32_t selected_boundary = UINT32_MAX;
        for (uint32_t device = 0U; device < facts->device_count; ++device) {
            for (uint32_t boundary = 0U; boundary < facts->task_count;
                 ++boundary) {
                if (used[(uint64_t)device * facts->boundary_count + boundary + 1U] <=
                    problem->device_capacity_bytes[device]) {
                    continue;
                }
                for (uint32_t index = 0U; index < reload_count; ++index) {
                    Reload *candidate = &reloads[index];
                    if (active[(uint64_t)index * facts->task_count + boundary] ==
                            0U ||
                        candidate->trigger >= candidate->latest_trigger) {
                        continue;
                    }
                    if (selected_reload == UINT32_MAX) {
                        selected_reload = index;
                        continue;
                    }
                    Reload *selected = &reloads[selected_reload];
                    uint64_t candidate_size =
                        problem->alias_size_bytes[candidate->alias];
                    uint64_t selected_size =
                        problem->alias_size_bytes[selected->alias];
                    if (candidate->entry_boundary > selected->entry_boundary ||
                        (candidate->entry_boundary == selected->entry_boundary &&
                         candidate_size > selected_size) ||
                        (candidate->entry_boundary == selected->entry_boundary &&
                         candidate_size == selected_size &&
                         candidate->alias > selected->alias)) {
                        selected_reload = index;
                    }
                }
                if (selected_reload != UINT32_MAX) {
                    selected_boundary = boundary;
                    break;
                }
            }
            if (selected_reload != UINT32_MAX) {
                break;
            }
        }
        if (selected_reload == UINT32_MAX) {
            break;
        }
        Reload *reload = &reloads[selected_reload];
        uint32_t old_trigger = reload->trigger;
        uint32_t new_trigger = selected_boundary + 1U;
        if (new_trigger > reload->latest_trigger) {
            new_trigger = reload->latest_trigger;
        }
        reload->trigger = new_trigger;
        for (uint32_t boundary = old_trigger; boundary < new_trigger; ++boundary) {
            uint64_t active_position =
                (uint64_t)selected_reload * facts->task_count + boundary;
            if (active[active_position] == 0U) {
                continue;
            }
            active[active_position] = 0U;
            uint64_t count_position =
                (uint64_t)reload->alias * facts->task_count + boundary;
            --counts[count_position];
            if (counts[count_position] == 0U) {
                uint32_t device = problem->alias_device[reload->alias];
                used[(uint64_t)device * facts->boundary_count + boundary + 1U] -=
                    problem->alias_size_bytes[reload->alias];
            }
        }
    }
    free(used);
    free(counts);
    free(active);
    return 0;
}

static int action_compare(const void *left_value, const void *right_value) {
    const Action *left = left_value;
    const Action *right = right_value;
    if (left->trigger != right->trigger) {
        return left->trigger < right->trigger ? -1 : 1;
    }
    if (left->kind != right->kind) {
        return left->kind < right->kind ? -1 : 1;
    }
    if (left->alias != right->alias) {
        return left->alias < right->alias ? -1 : 1;
    }
    return 0;
}

int shadowspill_delay_dense_prefetch(
    const ShadowSpillScheduleFacts *facts,
    const ShadowSpillSimulationResult *failure,
    ShadowSpillScheduleStorage *storage
) {
    if (facts == NULL || failure == NULL || storage == NULL ||
        (failure->status != SHADOWSPILL_SIMULATION_PREFETCH_DEVICE_CAPACITY &&
         failure->status != SHADOWSPILL_SIMULATION_TASK_DEVICE_CAPACITY)) {
        return 0;
    }
    const ShadowSpillSimulationProgram *program = facts->context->simulation;
    uint32_t selected = UINT32_MAX;
    uint32_t selected_target = UINT32_MAX;
    uint64_t selected_size = 0U;
    uint32_t selected_trigger = 0U;
    for (uint32_t index = 0U; index < storage->value.action_count; ++index) {
        if (storage->value.action_kinds[index] != SHADOWSPILL_MEMORY_PREFETCH) {
            continue;
        }
        uint32_t alias = storage->value.action_aliases[index];
        uint32_t trigger = storage->value.action_trigger_tasks[index];
        if (failure->error_alias != SHADOWSPILL_SIMULATOR_NO_INDEX &&
            alias != failure->error_alias) {
            continue;
        }
        if (failure->status == SHADOWSPILL_SIMULATION_PREFETCH_DEVICE_CAPACITY &&
            failure->error_task != SHADOWSPILL_SIMULATOR_NO_INDEX &&
            trigger != failure->error_task) {
            continue;
        }
        if (failure->status == SHADOWSPILL_SIMULATION_TASK_DEVICE_CAPACITY &&
            failure->error_task != SHADOWSPILL_SIMULATOR_NO_INDEX &&
            trigger >= failure->error_task) {
            continue;
        }
        uint32_t next_consumer = UINT32_MAX;
        for (uint32_t task = trigger + 1U; task < facts->task_count; ++task) {
            for (uint32_t offset = program->input_offsets[task];
                 offset < program->input_offsets[task + 1U];
                 ++offset) {
                if (program->input_aliases[offset] == alias) {
                    next_consumer = task;
                    break;
                }
            }
            if (next_consumer != UINT32_MAX) {
                break;
            }
        }
        uint32_t latest = next_consumer == UINT32_MAX
            ? facts->task_count - 1U
            : next_consumer - 1U;
        uint32_t target = trigger + 1U;
        if (failure->status == SHADOWSPILL_SIMULATION_TASK_DEVICE_CAPACITY &&
            failure->error_task != SHADOWSPILL_SIMULATOR_NO_INDEX &&
            target < failure->error_task) {
            target = failure->error_task;
        }
        if (target > latest) {
            continue;
        }
        uint64_t size = program->alias_size_bytes[alias];
        if (selected == UINT32_MAX || size > selected_size ||
            (size == selected_size && trigger > selected_trigger) ||
            (size == selected_size && trigger == selected_trigger &&
             index < selected)) {
            selected = index;
            selected_target = target;
            selected_size = size;
            selected_trigger = trigger;
        }
    }
    if (selected == UINT32_MAX) {
        return 0;
    }
    storage->value.action_trigger_tasks[selected] = selected_target;
    Action *actions = malloc(
        (storage->value.action_count == 0U ? 1U :
            (size_t)storage->value.action_count) * sizeof(*actions)
    );
    if (actions == NULL) {
        return -1;
    }
    for (uint32_t index = 0U; index < storage->value.action_count; ++index) {
        actions[index] = (Action){
            .trigger = storage->value.action_trigger_tasks[index],
            .alias = storage->value.action_aliases[index],
            .kind = storage->value.action_kinds[index],
        };
    }
    qsort(actions, storage->value.action_count, sizeof(*actions), action_compare);
    for (uint32_t index = 0U; index < storage->value.action_count; ++index) {
        storage->value.action_trigger_tasks[index] = actions[index].trigger;
        storage->value.action_aliases[index] = actions[index].alias;
        storage->value.action_kinds[index] = actions[index].kind;
    }
    free(actions);
    return 1;
}

static int append_action(
    Action *actions,
    uint32_t capacity,
    uint32_t *count,
    uint32_t trigger,
    uint32_t alias,
    uint8_t kind
) {
    if (*count >= capacity) {
        return -1;
    }
    actions[(*count)++] = (Action){
        .trigger = trigger,
        .alias = alias,
        .kind = kind,
    };
    return 0;
}

int shadowspill_emit_dense_schedule(
    const ShadowSpillScheduleFacts *facts,
    const uint8_t *resident,
    const uint8_t *breaks,
    uint8_t prefetch_rule,
    int coalesced,
    int prefetch_headroom,
    ShadowSpillScheduleStorage *storage
) {
    if (facts == NULL || resident == NULL || breaks == NULL || storage == NULL ||
        prefetch_rule > SHADOWSPILL_PREFETCH_LATEST_SAFE) {
        return -1;
    }
    shadowspill_schedule_storage_clear(storage);
    const ShadowSpillResidencyProblem *problem = facts->context->residency;
    uint32_t transition_capacity = storage->action_capacity;
    size_t transitions = transition_capacity == 0U
        ? 1U
        : (size_t)transition_capacity;
    Reload *reloads = malloc(transitions * sizeof(*reloads));
    Departure *departures = malloc(transitions * sizeof(*departures));
    Action *actions = malloc(transitions * sizeof(*actions));
    Span *spans = malloc((size_t)facts->boundary_count * sizeof(*spans));
    if (reloads == NULL || departures == NULL || actions == NULL || spans == NULL) {
        free(reloads);
        free(departures);
        free(actions);
        free(spans);
        return -1;
    }
    uint32_t reload_count = 0U;
    uint32_t departure_count = 0U;

    for (uint32_t alias = 0U; alias < facts->alias_count; ++alias) {
        if (problem->alias_size_bytes[alias] == 0U) {
            continue;
        }
        uint32_t span_count = collect_spans(
            resident,
            breaks,
            alias,
            facts->boundary_count,
            spans
        );
        if (span_count == 0U) {
            continue;
        }
        int32_t host_refreshed =
            problem->initial_location[alias] == 1 ||
                problem->alias_retain_spill_copy[alias] != 0U
            ? -1
            : -2;
        int has_previous_departure = 0;
        Departure previous_departure = {0};
        for (uint32_t span_index = 0U; span_index < span_count; ++span_index) {
            Span span = spans[span_index];
            int32_t start_boundary = (int32_t)span.start - 1;
            int32_t end_boundary = (int32_t)span.end - 1;
            int produced_at_entry = problem->productions[cell(
                alias,
                facts->boundary_count,
                span.start
            )] != 0U;
            if (start_boundary > -1 && !produced_at_entry) {
                uint32_t first_task = event_min_task(facts, alias, &span);
                uint32_t latest = first_task == UINT32_MAX
                    ? facts->task_count - 1U
                    : first_task - 1U;
                uint32_t earliest = 0U;
                if (has_previous_departure != 0) {
                    earliest = previous_departure.trigger + 1U;
                    if (coalesced != 0 &&
                        previous_departure.kind == SHADOWSPILL_MEMORY_RELEASE) {
                        earliest = previous_departure.trigger;
                    }
                }
                if (latest < earliest || reload_count >= transition_capacity) {
                    free(reloads);
                    free(departures);
                    free(actions);
                    free(spans);
                    return -2;
                }
                reloads[reload_count] = (Reload){
                    .alias = alias,
                    .earliest_trigger = earliest,
                    .latest_trigger = latest,
                    .entry_boundary = (uint32_t)start_boundary,
                    .ordinal = reload_count,
                    .trigger = latest,
                };
                ++reload_count;
            }

            uint32_t departure_task = event_max_task(facts, alias, &span);
            if (departure_task == UINT32_MAX) {
                int32_t clamped = end_boundary;
                if (clamped < 0) {
                    clamped = 0;
                }
                if (clamped >= (int32_t)facts->task_count) {
                    clamped = (int32_t)facts->task_count - 1;
                }
                departure_task = (uint32_t)clamped;
            }
            int has_later_span = span_index + 1U < span_count;
            int8_t final_location = problem->final_location[alias];
            int needs_departure = has_later_span || final_location != 0;
            if (!needs_departure) {
                continue;
            }
            uint8_t kind = SHADOWSPILL_MEMORY_RELEASE;
            if (has_later_span || final_location == 1) {
                if (problem->alias_retain_spill_copy[alias] != 0U &&
                    !has_write_since(
                        facts,
                        alias,
                        host_refreshed,
                        end_boundary
                    )) {
                    kind = SHADOWSPILL_MEMORY_RELEASE;
                } else {
                    kind = SHADOWSPILL_MEMORY_OFFLOAD;
                    host_refreshed = end_boundary;
                }
            }
            if (departure_count >= transition_capacity) {
                free(reloads);
                free(departures);
                free(actions);
                free(spans);
                return -1;
            }
            previous_departure = (Departure){
                .alias = alias,
                .trigger = departure_task,
                .kind = kind,
            };
            has_previous_departure = 1;
            departures[departure_count++] = previous_departure;
        }
    }

    if (prefetch_rule == SHADOWSPILL_PREFETCH_LATEST_SAFE) {
        for (uint32_t index = 0U; index < reload_count; ++index) {
            reloads[index].trigger = reloads[index].latest_trigger;
        }
    } else {
        choose_packed_triggers(facts, reloads, reload_count);
        if (prefetch_rule == SHADOWSPILL_PREFETCH_PACKED_FIT &&
            clamp_triggers_to_fit(
                facts,
                resident,
                breaks,
                reloads,
                reload_count,
                prefetch_headroom
            ) != 0) {
            free(reloads);
            free(departures);
            free(actions);
            free(spans);
            return -1;
        }
    }

    uint32_t action_count = 0U;
    for (uint32_t index = 0U; index < departure_count; ++index) {
        if (append_action(
                actions,
                transition_capacity,
                &action_count,
                departures[index].trigger,
                departures[index].alias,
                departures[index].kind
            ) != 0) {
            free(reloads);
            free(departures);
            free(actions);
            free(spans);
            return -1;
        }
    }
    for (uint32_t index = 0U; index < reload_count; ++index) {
        if (append_action(
                actions,
                transition_capacity,
                &action_count,
                reloads[index].trigger,
                reloads[index].alias,
                SHADOWSPILL_MEMORY_PREFETCH
            ) != 0) {
            free(reloads);
            free(departures);
            free(actions);
            free(spans);
            return -1;
        }
    }
    qsort(actions, action_count, sizeof(*actions), action_compare);

    uint32_t output_action = 0U;
    for (uint32_t index = 0U; index < action_count; ++index) {
        int remove = 0;
        if (coalesced != 0 &&
            (actions[index].kind == SHADOWSPILL_MEMORY_RELEASE ||
             actions[index].kind == SHADOWSPILL_MEMORY_PREFETCH)) {
            uint8_t counterpart = actions[index].kind == SHADOWSPILL_MEMORY_RELEASE
                ? SHADOWSPILL_MEMORY_PREFETCH
                : SHADOWSPILL_MEMORY_RELEASE;
            for (uint32_t other = 0U; other < action_count; ++other) {
                if (actions[other].trigger == actions[index].trigger &&
                    actions[other].alias == actions[index].alias &&
                    actions[other].kind == counterpart) {
                    remove = 1;
                    break;
                }
            }
        }
        if (remove != 0) {
            continue;
        }
        storage->value.action_trigger_tasks[output_action] = actions[index].trigger;
        storage->value.action_aliases[output_action] = actions[index].alias;
        storage->value.action_kinds[output_action] = actions[index].kind;
        ++output_action;
    }
    storage->value.action_count = output_action;

    for (uint32_t alias = 0U; alias < facts->alias_count; ++alias) {
        if (problem->alias_size_bytes[alias] == 0U) {
            continue;
        }
        if (problem->initial_location[alias] >= 0) {
            uint32_t output = storage->value.initial_count++;
            storage->value.initial_aliases[output] = alias;
            storage->value.initial_locations[output] =
                resident[cell(alias, facts->boundary_count, 0U)] != 0U
                ? SHADOWSPILL_MEMORY_DEVICE
                : SHADOWSPILL_MEMORY_HOST;
        }
        if (problem->final_location[alias] >= 0) {
            uint32_t output = storage->value.final_count++;
            storage->value.final_aliases[output] = alias;
            storage->value.final_locations[output] =
                (uint8_t)problem->final_location[alias];
        }
    }
    free(reloads);
    free(departures);
    free(actions);
    free(spans);
    return 0;
}

void shadowspill_bind_dense_schedule(
    const ShadowSpillSimulationProgram *topology,
    const ShadowSpillDenseSchedule *schedule,
    ShadowSpillSimulationProgram *program
) {
    *program = *topology;
    program->action_count = schedule->action_count;
    program->action_trigger_tasks = schedule->action_trigger_tasks;
    program->action_aliases = schedule->action_aliases;
    program->action_kinds = schedule->action_kinds;
    program->initial_count = schedule->initial_count;
    program->initial_aliases = schedule->initial_aliases;
    program->initial_locations = schedule->initial_locations;
    program->final_count = schedule->final_count;
    program->final_aliases = schedule->final_aliases;
    program->final_locations = schedule->final_locations;
}
