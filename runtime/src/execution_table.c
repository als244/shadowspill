#include "internal.h"
#include "internal/task_boundaries.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

static uint64_t execution_bucket(
    const ShadowSpillExecutionTable *table,
    uint64_t task_id
) {
    task_id ^= task_id >> 33U;
    task_id *= UINT64_C(0xff51afd7ed558ccd);
    task_id ^= task_id >> 33U;
    return task_id % table->bucket_count;
}

static ShadowSpillExecutionRecord *find_unlocked(
    const ShadowSpillExecutionTable *table,
    uint64_t task_id
) {
    if (table->by_id == NULL || table->bucket_count == 0U) {
        return NULL;
    }
    const uint64_t bucket = execution_bucket(table, task_id);
    for (ShadowSpillExecutionRecord *record = table->by_id[bucket];
         record != NULL; record = record->hash_next) {
        if (record->task_id == task_id) {
            return record;
        }
    }
    return NULL;
}

static void destroy_record(ShadowSpillExecutionRecord *record) {
    if (record == NULL) {
        return;
    }
    for (uint32_t index = 0U; index < record->input_count; ++index) {
        shadowspill_object_release(record->inputs[index]);
    }
    for (uint32_t index = 0U; index < record->update_count; ++index) {
        shadowspill_object_release(record->updates[index].object);
    }
    for (uint32_t index = 0U; index < record->action_count; ++index) {
        shadowspill_object_release(record->actions[index].object);
    }
    free(record->inputs);
    free(record->input_object_ids);
    free(record->updates);
    free(record->legacy_updates);
    free(record->actions);
    free(record->legacy_actions);
    free(record);
}

int shadowspill_execution_table_initialize(
    ShadowSpillExecutionTable *table,
    uint64_t bucket_count
) {
    if (table == NULL || bucket_count == 0U || bucket_count > SIZE_MAX) {
        return -1;
    }
    if (pthread_rwlock_init(&table->lock, NULL) != 0) {
        return -1;
    }
    table->lock_initialized = 1U;
    table->by_id = calloc((size_t)bucket_count, sizeof(*table->by_id));
    if (table->by_id == NULL) {
        pthread_rwlock_destroy(&table->lock);
        table->lock_initialized = 0U;
        return -1;
    }
    table->bucket_count = bucket_count;
    return 0;
}

void shadowspill_execution_table_destroy(ShadowSpillExecutionTable *table) {
    if (table == NULL) {
        return;
    }
    ShadowSpillExecutionRecord *record = table->owned_head;
    while (record != NULL) {
        ShadowSpillExecutionRecord *next = record->ownership_next;
        destroy_record(record);
        record = next;
    }
    free(table->by_id);
    if (table->lock_initialized) {
        pthread_rwlock_destroy(&table->lock);
    }
    *table = (ShadowSpillExecutionTable){0};
}

ShadowSpillExecutionRecord *shadowspill_execution_table_acquire(
    ShadowSpillExecutionTable *table,
    uint64_t task_id
) {
    if (table == NULL || !table->lock_initialized) {
        return NULL;
    }
    pthread_rwlock_rdlock(&table->lock);
    ShadowSpillExecutionRecord *record = find_unlocked(table, task_id);
    pthread_rwlock_unlock(&table->lock);
    return record;
}

static int same_description(
    const ShadowSpillExecutionRecord *record,
    const ShadowSpillExecutionDescription *description
) {
    if (record->input_count != description->input_count ||
        record->update_count != description->update_count ||
        record->action_count != description->action_count) {
        return 0;
    }
    for (uint32_t index = 0U; index < record->input_count; ++index) {
        if (record->input_object_ids[index] !=
            description->input_object_ids[index]) {
            return 0;
        }
    }
    for (uint32_t index = 0U; index < record->update_count; ++index) {
        if (record->legacy_updates[index].object_id !=
                description->updates[index].object_id ||
            record->legacy_updates[index].version_delta !=
                description->updates[index].version_delta) {
            return 0;
        }
    }
    for (uint32_t index = 0U; index < record->action_count; ++index) {
        if (record->legacy_actions[index].object_id !=
                description->actions[index].object_id ||
            record->legacy_actions[index].kind !=
                description->actions[index].kind) {
            return 0;
        }
    }
    return 1;
}

static ShadowSpillExecutionRecord *create_record(
    ShadowSpillRuntime *runtime,
    const ShadowSpillExecutionDescription *description
) {
    ShadowSpillExecutionRecord *record = calloc(1U, sizeof(*record));
    if (record == NULL) {
        return NULL;
    }
    record->task_id = description->task_id;
    record->input_count = description->input_count;
    record->update_count = description->update_count;
    record->action_count = description->action_count;
    if (record->input_count != 0U) {
        record->inputs = calloc(record->input_count, sizeof(*record->inputs));
        record->input_object_ids = calloc(
            record->input_count, sizeof(*record->input_object_ids)
        );
    }
    if (record->update_count != 0U) {
        record->updates = calloc(record->update_count, sizeof(*record->updates));
        record->legacy_updates = calloc(
            record->update_count, sizeof(*record->legacy_updates)
        );
    }
    if (record->action_count != 0U) {
        record->actions = calloc(record->action_count, sizeof(*record->actions));
        record->legacy_actions = calloc(
            record->action_count, sizeof(*record->legacy_actions)
        );
    }
    if ((record->input_count != 0U &&
         (record->inputs == NULL || record->input_object_ids == NULL)) ||
        (record->update_count != 0U &&
         (record->updates == NULL || record->legacy_updates == NULL)) ||
        (record->action_count != 0U &&
         (record->actions == NULL || record->legacy_actions == NULL))) {
        destroy_record(record);
        return NULL;
    }
    for (uint32_t index = 0U; index < record->input_count; ++index) {
        ShadowSpillObjectRecord *object = shadowspill_object_table_acquire(
            &runtime->objects, description->input_object_ids[index]
        );
        if (object == NULL) {
            destroy_record(record);
            return NULL;
        }
        record->inputs[index] = object;
        record->input_object_ids[index] = object->object_id;
    }
    for (uint32_t index = 0U; index < record->update_count; ++index) {
        ShadowSpillObjectRecord *object = shadowspill_object_table_acquire(
            &runtime->objects, description->updates[index].object_id
        );
        if (object == NULL) {
            destroy_record(record);
            return NULL;
        }
        record->updates[index] = (ShadowSpillExecutionUpdate){
            .object = object,
            .version_delta = description->updates[index].version_delta,
        };
        record->legacy_updates[index] = description->updates[index];
    }
    for (uint32_t index = 0U; index < record->action_count; ++index) {
        ShadowSpillObjectRecord *object = shadowspill_object_table_acquire(
            &runtime->objects, description->actions[index].object_id
        );
        if (object == NULL) {
            destroy_record(record);
            return NULL;
        }
        record->actions[index] = (ShadowSpillExecutionAction){
            .object = object,
            .kind = description->actions[index].kind,
        };
        record->legacy_actions[index] = description->actions[index];
    }
    return record;
}

ShadowSpillRuntimeStatus shadowspill_admit_execution(
    ShadowSpillRuntime *runtime,
    const ShadowSpillExecutionDescription *description
) {
    if (runtime == NULL || description == NULL ||
        (description->input_count != 0U &&
         description->input_object_ids == NULL) ||
        (description->update_count != 0U && description->updates == NULL) ||
        (description->action_count != 0U && description->actions == NULL)) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    ShadowSpillRuntimeStatus status = shadowspill_current_status_locked(runtime);
    if (status != SHADOWSPILL_RUNTIME_OK) {
        return status;
    }
    ShadowSpillExecutionRecord *created = create_record(runtime, description);
    if (created == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    ShadowSpillExecutionTable *table = &runtime->execution;
    pthread_rwlock_wrlock(&table->lock);
    ShadowSpillExecutionRecord *existing = find_unlocked(
        table, description->task_id
    );
    if (existing != NULL) {
        const int matches = same_description(existing, description);
        pthread_rwlock_unlock(&table->lock);
        destroy_record(created);
        return matches
            ? SHADOWSPILL_RUNTIME_OK
            : SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    const uint64_t bucket = execution_bucket(table, created->task_id);
    created->hash_next = table->by_id[bucket];
    table->by_id[bucket] = created;
    created->ownership_next = table->owned_head;
    table->owned_head = created;
    pthread_rwlock_unlock(&table->lock);
    return SHADOWSPILL_RUNTIME_OK;
}

ShadowSpillRuntimeStatus shadowspill_before_execution(
    ShadowSpillRuntime *runtime,
    uint64_t task_id,
    ShadowSpillBackendStream compute_stream,
    ShadowSpillObjectBinding *bindings,
    uint32_t binding_capacity
) {
    if (runtime == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    ShadowSpillExecutionRecord *record = shadowspill_execution_table_acquire(
        &runtime->execution, task_id
    );
    if (record == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    return shadowspill_before_task_legacy(
        runtime,
        task_id,
        compute_stream,
        record->input_object_ids,
        record->input_count,
        bindings,
        binding_capacity
    );
}

ShadowSpillRuntimeStatus shadowspill_after_execution(
    ShadowSpillRuntime *runtime,
    uint64_t task_id,
    ShadowSpillBackendStream compute_stream
) {
    if (runtime == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    ShadowSpillExecutionRecord *record = shadowspill_execution_table_acquire(
        &runtime->execution, task_id
    );
    if (record == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    return shadowspill_after_task_legacy(
        runtime,
        task_id,
        compute_stream,
        record->legacy_updates,
        record->update_count,
        record->legacy_actions,
        record->action_count
    );
}
