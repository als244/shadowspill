#include "internal.h"
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

typedef struct ReloadRank {
    uint32_t index;
    uint32_t entry_boundary;
    uint64_t size_bytes;
    uint32_t alias;
} ReloadRank;

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
    ShadowSpillScheduleStorage *storage
) {
    if (storage == NULL) {
        return -1;
    }
    memset(storage, 0, sizeof(*storage));
    storage->initial_capacity = alias_count;
    storage->final_capacity = alias_count;
    uint32_t aliases = alias_count == 0U ? 1U : alias_count;
    storage->value.action_trigger_tasks = calloc(
        1U,
        sizeof(*storage->value.action_trigger_tasks)
    );
    storage->value.action_aliases = calloc(
        1U,
        sizeof(*storage->value.action_aliases)
    );
    storage->value.action_kinds = calloc(
        1U,
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

static int reserve_schedule_actions(
    ShadowSpillScheduleStorage *storage,
    uint32_t capacity
) {
    if (capacity <= storage->action_capacity) {
        return 0;
    }
    uint32_t selected = storage->action_capacity == 0U
        ? 64U
        : storage->action_capacity;
    while (selected < capacity) {
        if (selected > UINT32_MAX / 2U) {
            selected = capacity;
            break;
        }
        selected *= 2U;
    }
    uint32_t *triggers = malloc(
        (size_t)selected * sizeof(*storage->value.action_trigger_tasks)
    );
    uint32_t *aliases = malloc(
        (size_t)selected * sizeof(*storage->value.action_aliases)
    );
    uint8_t *kinds = malloc(
        (size_t)selected * sizeof(*storage->value.action_kinds)
    );
    if (triggers == NULL || aliases == NULL || kinds == NULL) {
        free(triggers);
        free(aliases);
        free(kinds);
        return -1;
    }
    memcpy(
        triggers,
        storage->value.action_trigger_tasks,
        (size_t)storage->value.action_count * sizeof(*triggers)
    );
    memcpy(
        aliases,
        storage->value.action_aliases,
        (size_t)storage->value.action_count * sizeof(*aliases)
    );
    memcpy(
        kinds,
        storage->value.action_kinds,
        (size_t)storage->value.action_count * sizeof(*kinds)
    );
    free(storage->value.action_trigger_tasks);
    free(storage->value.action_aliases);
    free(storage->value.action_kinds);
    storage->value.action_trigger_tasks = triggers;
    storage->value.action_aliases = aliases;
    storage->value.action_kinds = kinds;
    storage->action_capacity = selected;
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
        destination->initial_capacity < source->value.initial_count ||
        destination->final_capacity < source->value.final_count ||
        reserve_schedule_actions(destination, source->value.action_count) != 0) {
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
        uint32_t span_count = collect_spans(
            resident,
            breaks,
            alias,
            facts->boundary_count,
            spans
        );
        uint32_t device = problem->alias_device[alias];
        for (uint32_t span_index = 1U; span_index < span_count; ++span_index) {
            Span *current = &spans[span_index];
            const Span *previous = &spans[span_index - 1U];

            /*
             * Residency canonicalization clears breaks on every absent cell.
             * Extending a later span by one absent cell therefore only moves
             * its start left; it cannot change any other span.  Keep that
             * local start directly instead of rediscovering every span after
             * each successful extension.
             */
            while (current->start > previous->end + 1U) {
                uint32_t candidate_cell = current->start - 1U;
                uint64_t position =
                    (uint64_t)device * facts->boundary_count + candidate_cell;
                uint64_t added = problem->output_reservations[cell(
                    alias,
                    facts->boundary_count,
                    candidate_cell
                )] != 0U
                    ? 0U
                    : problem->alias_size_bytes[alias];
                if (pressure[position] + added >
                    shadowspill_boundary_capacity(
                        problem,
                        device,
                        candidate_cell
                    )) {
                    break;
                }
                resident[cell(alias, facts->boundary_count, candidate_cell)] = 1U;
                pressure[position] += added;
                current->start = candidate_cell;
            }
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

static int reload_rank_compare(const void *left_value, const void *right_value) {
    const ReloadRank *left = left_value;
    const ReloadRank *right = right_value;
    if (left->entry_boundary != right->entry_boundary) {
        return left->entry_boundary > right->entry_boundary ? -1 : 1;
    }
    if (left->size_bytes != right->size_bytes) {
        return left->size_bytes > right->size_bytes ? -1 : 1;
    }
    if (left->alias != right->alias) {
        return left->alias > right->alias ? -1 : 1;
    }
    if (left->index != right->index) {
        return left->index < right->index ? -1 : 1;
    }
    return 0;
}

static void clear_active_reload(
    uint64_t *active,
    uint32_t word_count,
    uint32_t rank,
    uint32_t start,
    uint32_t end
) {
    uint64_t mask = ~(UINT64_C(1) << (rank & 63U));
    uint32_t word = rank >> 6U;
    for (uint32_t boundary = start; boundary < end; ++boundary) {
        active[(uint64_t)boundary * word_count + word] &= mask;
    }
}

static uint32_t first_active_reload(
    const uint64_t *active,
    uint32_t word_count,
    uint32_t boundary,
    const ReloadRank *ranked
) {
    uint64_t row = (uint64_t)boundary * word_count;
    for (uint32_t word = 0U; word < word_count; ++word) {
        uint64_t values = active[row + word];
        if (values != 0U) {
            uint32_t rank = word * 64U + (uint32_t)__builtin_ctzll(values);
            return ranked[rank].index;
        }
    }
    return UINT32_MAX;
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

static void choose_latest_safe_triggers(
    const ShadowSpillScheduleFacts *facts,
    Reload *reloads,
    uint32_t reload_count
) {
    const ShadowSpillResidencyProblem *problem = facts->context->residency;
    for (uint32_t index = 0U; index < reload_count; ++index) {
        Reload *reload = &reloads[index];
        const uint64_t deadline = ideal_trigger_time(
            problem,
            reload->latest_trigger
        );
        const uint64_t runtime = problem->fetch_runtime_ns[reload->alias];
        const uint64_t desired = deadline > runtime ? deadline - runtime : 0U;
        reload->trigger = latest_trigger_at_or_before(
            problem,
            reload->earliest_trigger,
            reload->latest_trigger,
            desired
        );
    }
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
    size_t active_words = 0U;
    uint32_t word_count = reload_count / 64U + (reload_count % 64U != 0U);
    if (checked_cells(facts->device_count, facts->boundary_count, &pressure_cells) !=
            0 ||
        checked_cells(word_count, facts->task_count, &active_words) != 0) {
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
    uint64_t *active = calloc(
        active_words == 0U ? 1U : active_words,
        sizeof(*active)
    );
    ReloadRank *ranked = malloc(
        (reload_count == 0U ? 1U : (size_t)reload_count) * sizeof(*ranked)
    );
    uint32_t *rank_by_reload = malloc(
        (reload_count == 0U ? 1U : (size_t)reload_count) *
        sizeof(*rank_by_reload)
    );
    if (used == NULL || counts == NULL || active == NULL || ranked == NULL ||
        rank_by_reload == NULL ||
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
        free(ranked);
        free(rank_by_reload);
        return -1;
    }

    for (uint32_t reload_index = 0U; reload_index < reload_count; ++reload_index) {
        Reload *reload = &reloads[reload_index];
        ranked[reload_index] = (ReloadRank){
            .index = reload_index,
            .entry_boundary = reload->entry_boundary,
            .size_bytes = problem->alias_size_bytes[reload->alias],
            .alias = reload->alias,
        };
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
            uint64_t count_position =
                (uint64_t)reload->alias * facts->task_count + boundary;
            if (counts[count_position]++ == 0U) {
                uint32_t device = problem->alias_device[reload->alias];
                used[(uint64_t)device * facts->boundary_count + boundary + 1U] +=
                    problem->alias_size_bytes[reload->alias];
            }
        }
    }
    qsort(ranked, reload_count, sizeof(*ranked), reload_rank_compare);
    for (uint32_t rank = 0U; rank < reload_count; ++rank) {
        rank_by_reload[ranked[rank].index] = rank;
    }
    for (uint32_t reload_index = 0U; reload_index < reload_count; ++reload_index) {
        Reload *reload = &reloads[reload_index];
        if (reload->trigger >= reload->latest_trigger) {
            continue;
        }
        uint32_t rank = rank_by_reload[reload_index];
        uint64_t mask = UINT64_C(1) << (rank & 63U);
        uint32_t word = rank >> 6U;
        for (uint32_t boundary = reload->trigger;
             boundary < reload->entry_boundary;
             ++boundary) {
            if (resident[cell(
                    reload->alias,
                    facts->boundary_count,
                    boundary + 1U
                )] == 0U) {
                active[(uint64_t)boundary * word_count + word] |= mask;
            }
        }
    }

    for (uint32_t device = 0U; device < facts->device_count; ++device) {
        for (uint32_t boundary = 0U; boundary < facts->task_count; ++boundary) {
            uint64_t used_position =
                (uint64_t)device * facts->boundary_count + boundary + 1U;
            while (used[used_position] > shadowspill_boundary_capacity(
                    problem,
                    device,
                    boundary + 1U
                )) {
                uint32_t selected_reload = first_active_reload(
                    active,
                    word_count,
                    boundary,
                    ranked
                );
                if (selected_reload == UINT32_MAX) {
                    break;
                }
                Reload *reload = &reloads[selected_reload];
                uint32_t old_trigger = reload->trigger;
                uint32_t new_trigger = boundary + 1U;
                if (new_trigger > reload->latest_trigger) {
                    new_trigger = reload->latest_trigger;
                }
                reload->trigger = new_trigger;
                uint32_t rank = rank_by_reload[selected_reload];
                clear_active_reload(
                    active,
                    word_count,
                    rank,
                    old_trigger,
                    new_trigger
                );
                if (new_trigger == reload->latest_trigger) {
                    clear_active_reload(
                        active,
                        word_count,
                        rank,
                        new_trigger,
                        facts->task_count
                    );
                }
                for (uint32_t retired = old_trigger; retired < new_trigger;
                     ++retired) {
                    if (resident[cell(
                            reload->alias,
                            facts->boundary_count,
                            retired + 1U
                        )] != 0U) {
                        continue;
                    }
                    uint64_t count_position =
                        (uint64_t)reload->alias * facts->task_count + retired;
                    --counts[count_position];
                    if (counts[count_position] == 0U) {
                        uint32_t reload_device =
                            problem->alias_device[reload->alias];
                        used[(uint64_t)reload_device * facts->boundary_count +
                             retired + 1U] -=
                            problem->alias_size_bytes[reload->alias];
                    }
                }
            }
        }
    }
    free(used);
    free(counts);
    free(active);
    free(ranked);
    free(rank_by_reload);
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

static int sort_storage_actions(ShadowSpillScheduleStorage *storage) {
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
    return 0;
}

static uint32_t next_input_consumer(
    const ShadowSpillScheduleFacts *facts,
    uint32_t alias,
    uint32_t trigger
) {
    const ShadowSpillSimulationProgram *program = facts->context->simulation;
    for (uint32_t task = trigger + 1U; task < facts->task_count; ++task) {
        for (uint32_t offset = program->input_offsets[task];
             offset < program->input_offsets[task + 1U];
             ++offset) {
            if (program->input_aliases[offset] == alias) {
                return task;
            }
        }
    }
    return UINT32_MAX;
}

static int copy_actions(
    const ShadowSpillScheduleFacts *facts,
    const Action *actions,
    uint32_t action_count,
    int coalesced,
    ShadowSpillScheduleStorage *storage
) {
    if (reserve_schedule_actions(storage, action_count) != 0) {
        return -1;
    }
    if (coalesced == 0) {
        for (uint32_t index = 0U; index < action_count; ++index) {
            storage->value.action_trigger_tasks[index] = actions[index].trigger;
            storage->value.action_aliases[index] = actions[index].alias;
            storage->value.action_kinds[index] = actions[index].kind;
        }
        storage->value.action_count = action_count;
        return 0;
    }

    uint8_t *masks = calloc(
        facts->alias_count == 0U ? 1U : facts->alias_count,
        sizeof(*masks)
    );
    uint32_t *touched = malloc(
        (facts->alias_count == 0U ? 1U : (size_t)facts->alias_count) *
        sizeof(*touched)
    );
    if (masks == NULL || touched == NULL) {
        free(masks);
        free(touched);
        return -1;
    }

    uint32_t output = 0U;
    uint32_t begin = 0U;
    while (begin < action_count) {
        uint32_t end = begin + 1U;
        while (end < action_count && actions[end].trigger == actions[begin].trigger) {
            ++end;
        }
        uint32_t touched_count = 0U;
        for (uint32_t index = begin; index < end; ++index) {
            uint8_t kind = actions[index].kind;
            if (kind != SHADOWSPILL_MEMORY_RELEASE &&
                kind != SHADOWSPILL_MEMORY_PREFETCH) {
                continue;
            }
            uint32_t alias = actions[index].alias;
            if (masks[alias] == 0U) {
                touched[touched_count++] = alias;
            }
            masks[alias] |= kind == SHADOWSPILL_MEMORY_RELEASE ? 1U : 2U;
        }
        for (uint32_t index = begin; index < end; ++index) {
            uint8_t kind = actions[index].kind;
            if ((kind == SHADOWSPILL_MEMORY_RELEASE ||
                 kind == SHADOWSPILL_MEMORY_PREFETCH) &&
                masks[actions[index].alias] == 3U) {
                continue;
            }
            storage->value.action_trigger_tasks[output] = actions[index].trigger;
            storage->value.action_aliases[output] = actions[index].alias;
            storage->value.action_kinds[output] = kind;
            ++output;
        }
        for (uint32_t index = 0U; index < touched_count; ++index) {
            masks[touched[index]] = 0U;
        }
        begin = end;
    }
    storage->value.action_count = output;
    free(masks);
    free(touched);
    return 0;
}

int shadowspill_delay_indexed_prefetch(
    const ShadowSpillScheduleFacts *facts,
    const ShadowSpillSimulationResult *failure,
    ShadowSpillScheduleStorage *storage,
    ShadowSpillPrefetchTriggerConstraint *constraint
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
        uint32_t next_consumer = next_input_consumer(facts, alias, trigger);
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
    const uint32_t alias = storage->value.action_aliases[selected];
    const uint32_t consumer = next_input_consumer(
        facts, alias, storage->value.action_trigger_tasks[selected]
    );
    if (consumer == UINT32_MAX) {
        return 0;
    }
    storage->value.action_trigger_tasks[selected] = selected_target;
    if (constraint != NULL) {
        *constraint = (ShadowSpillPrefetchTriggerConstraint){
            .alias = alias,
            .consumer_task = consumer,
            .minimum_trigger = selected_target,
            .maximum_trigger = UINT32_MAX,
        };
    }
    return sort_storage_actions(storage) == 0 ? 1 : -1;
}

int shadowspill_advance_indexed_prefetch_to_release(
    const ShadowSpillScheduleFacts *facts,
    uint32_t action_index,
    ShadowSpillScheduleStorage *storage,
    ShadowSpillPrefetchTriggerConstraint *constraint
) {
    if (facts == NULL || storage == NULL ||
        action_index >= storage->value.action_count ||
        storage->value.action_kinds[action_index] !=
            SHADOWSPILL_MEMORY_PREFETCH) {
        return 0;
    }
    const ShadowSpillSimulationProgram *program = facts->context->simulation;
    const uint32_t alias = storage->value.action_aliases[action_index];
    const uint32_t current_trigger =
        storage->value.action_trigger_tasks[action_index];
    if (alias >= facts->alias_count || current_trigger == 0U) {
        return 0;
    }

    const uint32_t consumer = next_input_consumer(
        facts, alias, current_trigger
    );
    if (consumer == UINT32_MAX) {
        return 0;
    }
    uint32_t minimum_trigger = 0U;
    int initial_spill_copy = 0;
    for (uint32_t index = 0U; index < storage->value.initial_count; ++index) {
        if (storage->value.initial_aliases[index] == alias &&
            storage->value.initial_locations[index] ==
                SHADOWSPILL_MEMORY_SPILL) {
            initial_spill_copy = 1;
            break;
        }
    }
    uint32_t latest_write = UINT32_MAX;
    for (uint32_t task = 0U; task < current_trigger; ++task) {
        if (facts->write_events[cell(
                alias, facts->boundary_count, task + 1U
            )] != 0U) {
            latest_write = task;
        }
    }
    int authoritative_spill_copy = initial_spill_copy;
    for (uint32_t index = 0U; index < storage->value.action_count; ++index) {
        if (storage->value.action_aliases[index] != alias ||
            storage->value.action_trigger_tasks[index] >= current_trigger) {
            continue;
        }
        const uint8_t kind = storage->value.action_kinds[index];
        if (kind != SHADOWSPILL_MEMORY_OFFLOAD &&
            kind != SHADOWSPILL_MEMORY_RELEASE) {
            continue;
        }
        const uint32_t trigger = storage->value.action_trigger_tasks[index];
        if (kind == SHADOWSPILL_MEMORY_OFFLOAD &&
            (latest_write == UINT32_MAX || trigger >= latest_write)) {
            authoritative_spill_copy = 1;
        }
        if (trigger > minimum_trigger) {
            minimum_trigger = trigger;
        }
    }
    if (!authoritative_spill_copy ||
        (latest_write != UINT32_MAX && minimum_trigger < latest_write)) {
        return 0;
    }

    uint32_t selected_trigger = UINT32_MAX;
    uint32_t selected_alias = UINT32_MAX;
    uint64_t selected_size = UINT64_MAX;
    const uint64_t required = program->alias_size_bytes[alias];
    for (uint32_t index = 0U; index < storage->value.action_count; ++index) {
        const uint8_t kind = storage->value.action_kinds[index];
        if (kind != SHADOWSPILL_MEMORY_RELEASE &&
            kind != SHADOWSPILL_MEMORY_OFFLOAD) {
            continue;
        }
        const uint32_t trigger = storage->value.action_trigger_tasks[index];
        const uint32_t candidate_alias = storage->value.action_aliases[index];
        if (trigger < minimum_trigger || trigger >= current_trigger ||
            candidate_alias == alias || candidate_alias >= facts->alias_count) {
            continue;
        }
        const uint64_t candidate_size =
            program->alias_size_bytes[candidate_alias];
        if (candidate_size < required) {
            continue;
        }
        if (selected_trigger == UINT32_MAX || trigger > selected_trigger ||
            (trigger == selected_trigger && candidate_size < selected_size) ||
            (trigger == selected_trigger && candidate_size == selected_size &&
             candidate_alias < selected_alias)) {
            selected_trigger = trigger;
            selected_alias = candidate_alias;
            selected_size = candidate_size;
        }
    }
    if (selected_trigger == UINT32_MAX) {
        return 0;
    }
    storage->value.action_trigger_tasks[action_index] = selected_trigger;
    if (constraint != NULL) {
        *constraint = (ShadowSpillPrefetchTriggerConstraint){
            .alias = alias,
            .consumer_task = consumer,
            .minimum_trigger = 0U,
            .maximum_trigger = selected_trigger,
        };
    }
    return sort_storage_actions(storage) == 0 ? 1 : -1;
}

int shadowspill_apply_prefetch_trigger_constraints(
    const ShadowSpillScheduleFacts *facts,
    const ShadowSpillPrefetchTriggerConstraint *constraints,
    uint32_t constraint_count,
    ShadowSpillScheduleStorage *storage
) {
    if (facts == NULL || storage == NULL ||
        (constraint_count != 0U && constraints == NULL)) {
        return -1;
    }
    if (constraint_count == 0U) {
        return 0;
    }
    int changed = 0;
    for (uint32_t action = 0U; action < storage->value.action_count; ++action) {
        if (storage->value.action_kinds[action] !=
            SHADOWSPILL_MEMORY_PREFETCH) {
            continue;
        }
        const uint32_t alias = storage->value.action_aliases[action];
        uint32_t trigger = storage->value.action_trigger_tasks[action];
        const uint32_t consumer = next_input_consumer(facts, alias, trigger);
        if (consumer == UINT32_MAX) {
            continue;
        }
        for (uint32_t index = 0U; index < constraint_count; ++index) {
            const ShadowSpillPrefetchTriggerConstraint *constraint =
                &constraints[index];
            if (constraint->alias != alias ||
                constraint->consumer_task != consumer) {
                continue;
            }
            if (constraint->minimum_trigger > constraint->maximum_trigger ||
                (constraint->maximum_trigger != UINT32_MAX &&
                 constraint->maximum_trigger >= consumer)) {
                return 1;
            }
            if (trigger < constraint->minimum_trigger) {
                trigger = constraint->minimum_trigger;
            }
            if (trigger > constraint->maximum_trigger) {
                trigger = constraint->maximum_trigger;
            }
            if (trigger >= consumer) {
                return 1;
            }
        }
        if (trigger != storage->value.action_trigger_tasks[action]) {
            storage->value.action_trigger_tasks[action] = trigger;
            changed = 1;
        }
    }
    if (changed != 0 && sort_storage_actions(storage) != 0) {
        return -1;
    }
    return 0;
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

static int reserve_reloads(
    Reload **values,
    uint32_t *capacity,
    uint32_t count
) {
    if (count < *capacity) {
        return 0;
    }
    uint32_t selected = *capacity == 0U ? 64U : *capacity * 2U;
    if (selected <= *capacity) {
        return -1;
    }
    Reload *replacement = realloc(
        *values,
        (size_t)selected * sizeof(*replacement)
    );
    if (replacement == NULL) {
        return -1;
    }
    *values = replacement;
    *capacity = selected;
    return 0;
}

static int reserve_departures(
    Departure **values,
    uint32_t *capacity,
    uint32_t count
) {
    if (count < *capacity) {
        return 0;
    }
    uint32_t selected = *capacity == 0U ? 64U : *capacity * 2U;
    if (selected <= *capacity) {
        return -1;
    }
    Departure *replacement = realloc(
        *values,
        (size_t)selected * sizeof(*replacement)
    );
    if (replacement == NULL) {
        return -1;
    }
    *values = replacement;
    *capacity = selected;
    return 0;
}

int shadowspill_emit_indexed_schedule(
    const ShadowSpillScheduleFacts *facts,
    const uint8_t *resident,
    const uint8_t *breaks,
    uint8_t prefetch_rule,
    int coalesced,
    int prefetch_headroom,
    ShadowSpillScheduleStorage *storage
) {
    if (facts == NULL || resident == NULL || breaks == NULL || storage == NULL ||
        prefetch_rule > SHADOWSPILL_PREFETCH_DEMAND) {
        return -1;
    }
    shadowspill_schedule_storage_clear(storage);
    const ShadowSpillResidencyProblem *problem = facts->context->residency;
    uint32_t reload_capacity = 0U;
    uint32_t departure_capacity = 0U;
    Reload *reloads = NULL;
    Departure *departures = NULL;
    Span *spans = malloc((size_t)facts->boundary_count * sizeof(*spans));
    if (spans == NULL) {
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
        int32_t spill_refreshed =
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
                if (latest < earliest ||
                    reserve_reloads(
                        &reloads,
                        &reload_capacity,
                        reload_count
                    ) != 0) {
                    free(reloads);
                    free(departures);
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
                        spill_refreshed,
                        end_boundary
                    )) {
                    kind = SHADOWSPILL_MEMORY_RELEASE;
                } else {
                    kind = SHADOWSPILL_MEMORY_OFFLOAD;
                    spill_refreshed = end_boundary;
                }
            }
            if (reserve_departures(
                    &departures,
                    &departure_capacity,
                    departure_count
                ) != 0) {
                free(reloads);
                free(departures);
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

    if (prefetch_rule == SHADOWSPILL_PREFETCH_DEMAND) {
        for (uint32_t index = 0U; index < reload_count; ++index) {
            reloads[index].trigger = reloads[index].latest_trigger;
        }
    } else if (prefetch_rule == SHADOWSPILL_PREFETCH_LATEST_SAFE) {
        choose_latest_safe_triggers(facts, reloads, reload_count);
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
            free(spans);
            return -1;
        }
    }

    if (reload_count > UINT32_MAX - departure_count) {
        free(reloads);
        free(departures);
        free(spans);
        return -1;
    }
    uint32_t transition_count = reload_count + departure_count;
    Action *actions = malloc(
        (transition_count == 0U ? 1U : (size_t)transition_count) *
        sizeof(*actions)
    );
    if (actions == NULL) {
        free(reloads);
        free(departures);
        free(spans);
        return -1;
    }
    uint32_t action_count = 0U;
    for (uint32_t index = 0U; index < departure_count; ++index) {
        if (append_action(
                actions,
                transition_count,
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
                transition_count,
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

    if (copy_actions(facts, actions, action_count, coalesced, storage) != 0) {
        free(reloads);
        free(departures);
        free(actions);
        free(spans);
        return -1;
    }

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
                : SHADOWSPILL_MEMORY_SPILL;
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

void shadowspill_bind_indexed_schedule(
    const ShadowSpillSimulationProgram *topology,
    const ShadowSpillIndexedSchedule *schedule,
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
