/* The actions a candidate schedule holds, and their storage. */
#include "internal.h"

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

int shadowspill_schedule_reserve_actions(
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

int shadowspill_schedule_storage_assign(
    ShadowSpillScheduleStorage *destination,
    const ShadowSpillIndexedSchedule *source
) {
    if (destination == NULL || source == NULL ||
        destination->initial_capacity < source->initial_count ||
        destination->final_capacity < source->final_count ||
        shadowspill_schedule_reserve_actions(destination, source->action_count) != 0) {
        return -1;
    }
    destination->value.action_count = source->action_count;
    destination->value.initial_count = source->initial_count;
    destination->value.final_count = source->final_count;
    memcpy(
        destination->value.action_trigger_tasks,
        source->action_trigger_tasks,
        (size_t)source->action_count * sizeof(*source->action_trigger_tasks)
    );
    memcpy(
        destination->value.action_aliases,
        source->action_aliases,
        (size_t)source->action_count * sizeof(*source->action_aliases)
    );
    memcpy(
        destination->value.action_kinds,
        source->action_kinds,
        (size_t)source->action_count * sizeof(*source->action_kinds)
    );
    memcpy(
        destination->value.initial_aliases,
        source->initial_aliases,
        (size_t)source->initial_count * sizeof(*source->initial_aliases)
    );
    memcpy(
        destination->value.initial_locations,
        source->initial_locations,
        (size_t)source->initial_count * sizeof(*source->initial_locations)
    );
    memcpy(
        destination->value.final_aliases,
        source->final_aliases,
        (size_t)source->final_count * sizeof(*source->final_aliases)
    );
    memcpy(
        destination->value.final_locations,
        source->final_locations,
        (size_t)source->final_count * sizeof(*source->final_locations)
    );
    return 0;
}

int shadowspill_schedule_storage_copy(
    ShadowSpillScheduleStorage *destination,
    const ShadowSpillScheduleStorage *source
) {
    if (destination == NULL || source == NULL) {
        return -1;
    }
    return shadowspill_schedule_storage_assign(destination, &source->value);
}
