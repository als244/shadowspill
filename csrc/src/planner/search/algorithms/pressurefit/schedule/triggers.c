/* When each reload fires: as late as is safe, or packed. */
#include "internal.h"

uint32_t shadowspill_schedule_event_min_task(
    const ShadowSpillScheduleFacts *facts,
    uint32_t alias,
    const Span *span
) {
    uint32_t selected = UINT32_MAX;
    for (uint32_t index = span->start; index <= span->end; ++index) {
        uint32_t task = facts->earliest_access_task[shadowspill_schedule_cell(
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

uint32_t shadowspill_schedule_event_max_task(
    const ShadowSpillScheduleFacts *facts,
    uint32_t alias,
    const Span *span
) {
    const ShadowSpillPressureFitResidencyProblem *problem = facts->problem->residency;
    const uint32_t last = problem->anchor_offsets[alias + 1U];
    uint32_t selected = UINT32_MAX;
    for (uint32_t index = shadowspill_anchor_lower_bound(
             problem->anchor_positions, problem->anchor_offsets[alias], last, span->start
         );
         index < last && problem->anchor_positions[index] <= span->end;
         ++index) {
        const uint32_t task = problem->anchor_tasks[index];
        if (task != UINT32_MAX && (selected == UINT32_MAX || task > selected)) {
            selected = task;
        }
    }
    return selected;
}

int shadowspill_schedule_has_write_since(
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
        if (facts->write_events[shadowspill_schedule_cell(alias, facts->boundary_count, index)] != 0U) {
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

int shadowspill_schedule_reload_rank_compare(const void *left_value, const void *right_value) {
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

void shadowspill_schedule_clear_active_reload(
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

uint32_t shadowspill_schedule_first_active_reload(
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
    const ShadowSpillPressureFitResidencyProblem *problem,
    uint32_t trigger
) {
    return problem->task_ideal_end_ns[trigger];
}

void shadowspill_schedule_choose_latest_safe_triggers(
    const ShadowSpillScheduleFacts *facts,
    Reload *reloads,
    uint32_t reload_count
) {
    const ShadowSpillPressureFitResidencyProblem *problem = facts->problem->residency;
    for (uint32_t index = 0U; index < reload_count; ++index) {
        Reload *reload = &reloads[index];
        const uint64_t deadline = ideal_trigger_time(
            problem,
            reload->latest_trigger
        );
        const uint64_t runtime = problem->fetch_runtime_ns[reload->alias];
        const uint64_t desired = deadline > runtime ? deadline - runtime : 0U;
        reload->trigger = shadowspill_latest_safe_trigger(
            problem->task_ideal_end_ns,
            reload->earliest_trigger,
            reload->latest_trigger,
            desired
        );
    }
}

void shadowspill_schedule_choose_packed_triggers(
    const ShadowSpillScheduleFacts *facts,
    Reload *reloads,
    uint32_t reload_count
) {
    const ShadowSpillPressureFitResidencyProblem *problem = facts->problem->residency;
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
        reload->trigger = shadowspill_latest_safe_trigger(
            problem->task_ideal_end_ns,
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
