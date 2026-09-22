/* Moving one fetch within a schedule that already exists. */
#include "internal.h"

int shadowspill_schedule_action_compare(const void *left_value, const void *right_value) {
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
    qsort(actions, storage->value.action_count, sizeof(*actions), shadowspill_schedule_action_compare);
    for (uint32_t index = 0U; index < storage->value.action_count; ++index) {
        storage->value.action_trigger_tasks[index] = actions[index].trigger;
        storage->value.action_aliases[index] = actions[index].alias;
        storage->value.action_kinds[index] = actions[index].kind;
    }
    free(actions);
    return 0;
}

uint32_t shadowspill_schedule_next_input_consumer(
    const ShadowSpillScheduleFacts *facts,
    uint32_t alias,
    uint32_t trigger
) {
    const ShadowSpillSimulationProgram *program = facts->problem->context.simulation;
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

int shadowspill_schedule_copy_actions(
    const ShadowSpillScheduleFacts *facts,
    const Action *actions,
    uint32_t action_count,
    int coalesced,
    ShadowSpillScheduleStorage *storage
) {
    if (shadowspill_schedule_reserve_actions(storage, action_count) != 0) {
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
                kind != SHADOWSPILL_MEMORY_FETCH) {
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
                 kind == SHADOWSPILL_MEMORY_FETCH) &&
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

int shadowspill_delay_indexed_fetch(
    const ShadowSpillScheduleFacts *facts,
    const ShadowSpillSimulationResult *failure,
    ShadowSpillScheduleStorage *storage,
    ShadowSpillFetchTriggerConstraint *constraint
) {
    if (facts == NULL || failure == NULL || storage == NULL ||
        (failure->status != SHADOWSPILL_STATUS_FETCH_DEVICE_CAPACITY &&
         failure->status != SHADOWSPILL_STATUS_TASK_DEVICE_CAPACITY)) {
        return 0;
    }
    const ShadowSpillSimulationProgram *program = facts->problem->context.simulation;
    uint32_t selected = UINT32_MAX;
    uint32_t selected_target = UINT32_MAX;
    uint64_t selected_size = 0U;
    uint32_t selected_trigger = 0U;
    for (uint32_t index = 0U; index < storage->value.action_count; ++index) {
        if (storage->value.action_kinds[index] != SHADOWSPILL_MEMORY_FETCH) {
            continue;
        }
        uint32_t alias = storage->value.action_aliases[index];
        uint32_t trigger = storage->value.action_trigger_tasks[index];
        if (failure->error_alias != SHADOWSPILL_SIMULATOR_NO_INDEX &&
            alias != failure->error_alias) {
            continue;
        }
        if (failure->status == SHADOWSPILL_STATUS_FETCH_DEVICE_CAPACITY &&
            failure->error_task != SHADOWSPILL_SIMULATOR_NO_INDEX &&
            trigger != failure->error_task) {
            continue;
        }
        if (failure->status == SHADOWSPILL_STATUS_TASK_DEVICE_CAPACITY &&
            failure->error_task != SHADOWSPILL_SIMULATOR_NO_INDEX &&
            trigger >= failure->error_task) {
            continue;
        }
        uint32_t next_consumer = shadowspill_schedule_next_input_consumer(facts, alias, trigger);
        uint32_t latest = next_consumer == UINT32_MAX
            ? facts->task_count - 1U
            : next_consumer - 1U;
        uint32_t target = trigger + 1U;
        if (failure->status == SHADOWSPILL_STATUS_TASK_DEVICE_CAPACITY &&
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
    const uint32_t consumer = shadowspill_schedule_next_input_consumer(
        facts, alias, storage->value.action_trigger_tasks[selected]
    );
    if (consumer == UINT32_MAX) {
        return 0;
    }
    storage->value.action_trigger_tasks[selected] = selected_target;
    if (constraint != NULL) {
        *constraint = (ShadowSpillFetchTriggerConstraint){
            .alias = alias,
            .consumer_task = consumer,
            .minimum_trigger = selected_target,
            .maximum_trigger = UINT32_MAX,
        };
    }
    return sort_storage_actions(storage) == 0 ? 1 : -1;
}

int shadowspill_advance_indexed_fetch_to_release(
    const ShadowSpillScheduleFacts *facts,
    uint32_t action_index,
    ShadowSpillScheduleStorage *storage,
    ShadowSpillFetchTriggerConstraint *constraint
) {
    if (facts == NULL || storage == NULL ||
        action_index >= storage->value.action_count ||
        storage->value.action_kinds[action_index] !=
            SHADOWSPILL_MEMORY_FETCH) {
        return 0;
    }
    const ShadowSpillSimulationProgram *program = facts->problem->context.simulation;
    const uint32_t alias = storage->value.action_aliases[action_index];
    const uint32_t current_trigger =
        storage->value.action_trigger_tasks[action_index];
    if (alias >= facts->alias_count || current_trigger == 0U ||
        !shadowspill_alias_may_cut(facts->problem->residency, alias)) {
        return 0;
    }

    const uint32_t consumer = shadowspill_schedule_next_input_consumer(
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
        if (facts->write_events[shadowspill_schedule_cell(
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
        if (kind != SHADOWSPILL_MEMORY_EVICT &&
            kind != SHADOWSPILL_MEMORY_RELEASE) {
            continue;
        }
        const uint32_t trigger = storage->value.action_trigger_tasks[index];
        if (kind == SHADOWSPILL_MEMORY_EVICT &&
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
            kind != SHADOWSPILL_MEMORY_EVICT) {
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
        *constraint = (ShadowSpillFetchTriggerConstraint){
            .alias = alias,
            .consumer_task = consumer,
            .minimum_trigger = 0U,
            .maximum_trigger = selected_trigger,
        };
    }
    return sort_storage_actions(storage) == 0 ? 1 : -1;
}

int shadowspill_apply_fetch_trigger_constraints(
    const ShadowSpillScheduleFacts *facts,
    const ShadowSpillFetchTriggerConstraint *constraints,
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
            SHADOWSPILL_MEMORY_FETCH) {
            continue;
        }
        const uint32_t alias = storage->value.action_aliases[action];
        uint32_t trigger = storage->value.action_trigger_tasks[action];
        const uint32_t consumer = shadowspill_schedule_next_input_consumer(facts, alias, trigger);
        if (consumer == UINT32_MAX) {
            continue;
        }
        for (uint32_t index = 0U; index < constraint_count; ++index) {
            const ShadowSpillFetchTriggerConstraint *constraint =
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
