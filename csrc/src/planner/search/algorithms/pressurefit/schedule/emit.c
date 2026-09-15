/* The schedule itself: every reload and departure, in order. */
#include "internal.h"

typedef struct Departure {
    uint32_t alias;
    uint32_t trigger;
    uint8_t kind;
} Departure;

/*
 * Emitting one schedule.
 *
 * A residency plan says where every object is at every boundary. A schedule
 * says what to do about it: fetch an object back before the span that needs
 * it, and release or evict it at the end of a span something later wants.
 * The stages below are that translation, and `Emission` is the scratch they
 * share -- collected in one place so that every failure releases the same
 * set, rather than each exit spelling it out again.
 */
typedef struct Emission {
    /* One alias's residency spans, reused across aliases. */
    Span *spans;
    Reload *reloads;
    uint32_t reload_count;
    uint32_t reload_capacity;
    Departure *departures;
    uint32_t departure_count;
    uint32_t departure_capacity;
    Action *actions;
    uint32_t action_count;
} Emission;

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

static void emission_destroy(Emission *emission) {
    free(emission->spans);
    free(emission->reloads);
    free(emission->departures);
    free(emission->actions);
    memset(emission, 0, sizeof(*emission));
}

/*
 * A span whose object is not produced at its entry has to be fetched before
 * its first use. The window runs from just after the previous departure to
 * the boundary before that use; coalescing lets the fetch share the previous
 * release's boundary rather than starting after it.
 *
 * Returns -2 when no boundary in that window can carry the fetch, which is a
 * fact about the residency rather than a failure to allocate.
 */
static int plan_span_reload(
    const ShadowSpillScheduleFacts *facts,
    uint32_t alias,
    Span span,
    int coalesced,
    int has_previous_departure,
    Departure previous_departure,
    Emission *emission
) {
    const ShadowSpillPressureFitResidencyProblem *problem = facts->problem->residency;
    const int32_t start_boundary = (int32_t)span.start - 1;
    const int produced_at_entry =
        problem->productions[shadowspill_schedule_cell(alias, facts->boundary_count, span.start)] != 0U;
    if (start_boundary <= -1 || produced_at_entry) {
        return 0;
    }
    const uint32_t first_task = shadowspill_schedule_event_min_task(facts, alias, &span);
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
    /* An alias the reducer may not cut is fetched at the trigger its resident
     * slice was sized for, so no rule moves it. */
    if (problem->fixed_fetch_trigger != NULL &&
        problem->fixed_fetch_trigger[alias] != UINT32_MAX &&
        problem->fixed_fetch_trigger[alias] <= latest) {
        earliest = problem->fixed_fetch_trigger[alias];
        latest = earliest;
    }
    if (latest < earliest ||
        reserve_reloads(
            &emission->reloads, &emission->reload_capacity, emission->reload_count
        ) != 0) {
        return -2;
    }
    emission->reloads[emission->reload_count] = (Reload){
        .alias = alias,
        .earliest_trigger = earliest,
        .latest_trigger = latest,
        .entry_boundary = (uint32_t)start_boundary,
        .ordinal = emission->reload_count,
        .trigger = latest,
    };
    ++emission->reload_count;
    return 0;
}

/*
 * The end of a span needs a move only when something later wants the object:
 * another span, or the final residency. A value the spill copy does not match
 * has to be written back; anything else is released.
 */
static int plan_span_departure(
    const ShadowSpillScheduleFacts *facts,
    uint32_t alias,
    Span span,
    int has_later_span,
    int32_t *spill_refreshed,
    Departure *previous_departure,
    int *has_previous_departure,
    Emission *emission
) {
    const ShadowSpillPressureFitResidencyProblem *problem = facts->problem->residency;
    const int8_t final_location = problem->final_location[alias];
    if (!has_later_span && final_location == 0) {
        return 0;
    }
    const int32_t end_boundary = (int32_t)span.end - 1;
    uint32_t departure_task = shadowspill_schedule_event_max_task(facts, alias, &span);
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
    uint8_t kind = SHADOWSPILL_MEMORY_RELEASE;
    if (has_later_span || final_location == 1) {
        if (problem->alias_retain_spill_copy[alias] != 0U &&
            !shadowspill_schedule_has_write_since(facts, alias, *spill_refreshed, end_boundary)) {
            kind = SHADOWSPILL_MEMORY_RELEASE;
        } else {
            kind = SHADOWSPILL_MEMORY_EVICT;
            *spill_refreshed = end_boundary;
        }
    }
    if (reserve_departures(
            &emission->departures,
            &emission->departure_capacity,
            emission->departure_count
        ) != 0) {
        return -1;
    }
    *previous_departure = (Departure){
        .alias = alias,
        .trigger = departure_task,
        .kind = kind,
    };
    *has_previous_departure = 1;
    emission->departures[emission->departure_count++] = *previous_departure;
    return 0;
}

/* Every move one alias's residency implies, in span order: the departures
 * chain, because where a fetch may start depends on where the last release
 * or eviction landed. */
static int collect_alias_transitions(
    const ShadowSpillScheduleFacts *facts,
    const uint8_t *resident,
    const uint8_t *breaks,
    uint32_t alias,
    int coalesced,
    Emission *emission
) {
    const ShadowSpillPressureFitResidencyProblem *problem = facts->problem->residency;
    const uint32_t span_count = shadowspill_schedule_collect_spans(
        resident, breaks, alias, facts->boundary_count, emission->spans
    );
    if (span_count == 0U) {
        return 0;
    }
    /* Where the spill copy was last made to match. -1 means it already does;
     * -2 means there is no spill copy to keep current. */
    int32_t spill_refreshed = problem->initial_location[alias] == 1 ||
            problem->alias_retain_spill_copy[alias] != 0U
        ? -1
        : -2;
    int has_previous_departure = 0;
    Departure previous_departure = {0};
    for (uint32_t index = 0U; index < span_count; ++index) {
        const Span span = emission->spans[index];
        const int reloaded = plan_span_reload(
            facts,
            alias,
            span,
            coalesced,
            has_previous_departure,
            previous_departure,
            emission
        );
        if (reloaded != 0) {
            return reloaded;
        }
        if (plan_span_departure(
                facts,
                alias,
                span,
                index + 1U < span_count,
                &spill_refreshed,
                &previous_departure,
                &has_previous_departure,
                emission
            ) != 0) {
            return -1;
        }
    }
    return 0;
}

/* Where in its window each fetch actually goes. Every rule starts from the
 * same windows and differs only here. */
static int choose_triggers(
    const ShadowSpillScheduleFacts *facts,
    const uint8_t *resident,
    const uint8_t *breaks,
    uint8_t fetch_rule,
    int fetch_headroom,
    Emission *emission
) {
    if (fetch_rule == SHADOWSPILL_PRESSUREFIT_FETCH_DEMAND) {
        for (uint32_t index = 0U; index < emission->reload_count; ++index) {
            emission->reloads[index].trigger =
                emission->reloads[index].latest_trigger;
        }
        return 0;
    }
    if (fetch_rule == SHADOWSPILL_PRESSUREFIT_FETCH_LATEST_SAFE) {
        shadowspill_schedule_choose_latest_safe_triggers(facts, emission->reloads, emission->reload_count);
        return 0;
    }
    shadowspill_schedule_choose_packed_triggers(facts, emission->reloads, emission->reload_count);
    if (fetch_rule != SHADOWSPILL_PRESSUREFIT_FETCH_PACKED_FIT) {
        return 0;
    }
    return shadowspill_schedule_clamp_triggers_to_fit(
        facts,
        resident,
        breaks,
        emission->reloads,
        emission->reload_count,
        fetch_headroom
    );
}

/* Both move lists become one list, ordered by the boundary each fires on. */
static int build_actions(Emission *emission) {
    if (emission->reload_count > UINT32_MAX - emission->departure_count) {
        return -1;
    }
    const uint32_t transition_count =
        emission->reload_count + emission->departure_count;
    emission->actions = malloc(
        (transition_count == 0U ? 1U : (size_t)transition_count) *
        sizeof(*emission->actions)
    );
    if (emission->actions == NULL) {
        return -1;
    }
    for (uint32_t index = 0U; index < emission->departure_count; ++index) {
        if (append_action(
                emission->actions,
                transition_count,
                &emission->action_count,
                emission->departures[index].trigger,
                emission->departures[index].alias,
                emission->departures[index].kind
            ) != 0) {
            return -1;
        }
    }
    for (uint32_t index = 0U; index < emission->reload_count; ++index) {
        if (append_action(
                emission->actions,
                transition_count,
                &emission->action_count,
                emission->reloads[index].trigger,
                emission->reloads[index].alias,
                SHADOWSPILL_MEMORY_FETCH
            ) != 0) {
            return -1;
        }
    }
    qsort(
        emission->actions,
        emission->action_count,
        sizeof(*emission->actions),
        shadowspill_schedule_action_compare
    );
    return 0;
}

/* What the schedule starts and ends holding, for the objects that declare a
 * boundary state at all. */
static void emit_boundary_residency(
    const ShadowSpillScheduleFacts *facts,
    const uint8_t *resident,
    ShadowSpillScheduleStorage *storage
) {
    const ShadowSpillPressureFitResidencyProblem *problem = facts->problem->residency;
    for (uint32_t alias = 0U; alias < facts->alias_count; ++alias) {
        if (problem->alias_size_bytes[alias] == 0U) {
            continue;
        }
        if (problem->initial_location[alias] >= 0) {
            const uint32_t output = storage->value.initial_count++;
            storage->value.initial_aliases[output] = alias;
            storage->value.initial_locations[output] =
                shadowspill_cell_get(resident, shadowspill_schedule_cell(alias, facts->boundary_count, 0U))
                ? SHADOWSPILL_MEMORY_DEVICE
                : SHADOWSPILL_MEMORY_SPILL;
        }
        if (problem->final_location[alias] >= 0) {
            const uint32_t output = storage->value.final_count++;
            storage->value.final_aliases[output] = alias;
            storage->value.final_locations[output] =
                (uint8_t)problem->final_location[alias];
        }
    }
}

int shadowspill_emit_indexed_schedule(
    const ShadowSpillScheduleFacts *facts,
    const uint8_t *resident,
    const uint8_t *breaks,
    uint8_t fetch_rule,
    int coalesced,
    int fetch_headroom,
    ShadowSpillScheduleStorage *storage
) {
    if (facts == NULL || resident == NULL || breaks == NULL || storage == NULL ||
        fetch_rule > SHADOWSPILL_PRESSUREFIT_FETCH_DEMAND) {
        return -1;
    }
    shadowspill_schedule_storage_clear(storage);
    const ShadowSpillPressureFitResidencyProblem *problem = facts->problem->residency;
    Emission emission = {0};
    emission.spans = malloc((size_t)facts->boundary_count * sizeof(*emission.spans));
    if (emission.spans == NULL) {
        return -1;
    }

    for (uint32_t alias = 0U; alias < facts->alias_count; ++alias) {
        if (problem->alias_size_bytes[alias] == 0U) {
            continue;
        }
        const int collected = collect_alias_transitions(
            facts, resident, breaks, alias, coalesced, &emission
        );
        if (collected != 0) {
            emission_destroy(&emission);
            return collected;
        }
    }

    if (choose_triggers(
            facts, resident, breaks, fetch_rule, fetch_headroom, &emission
        ) != 0 ||
        build_actions(&emission) != 0 ||
        shadowspill_schedule_copy_actions(
            facts, emission.actions, emission.action_count, coalesced, storage
        ) != 0) {
        emission_destroy(&emission);
        return -1;
    }

    emit_boundary_residency(facts, resident, storage);
    emission_destroy(&emission);
    return 0;
}
