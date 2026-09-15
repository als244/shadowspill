/* The task table: buckets, lookup, and the records it owns. */
#include "../internal.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *shadowspill_copy_action_trace_label(
    const ShadowSpillRuntimeAction *action,
    uint64_t task_id,
    uint64_t size_bytes
) {
    if (action->trace_label != NULL) {
        const size_t length = strnlen(
            action->trace_label,
            SHADOWSPILL_RUNTIME_TRACE_LABEL_MAX_BYTES + 1U
        );
        if (length > SHADOWSPILL_RUNTIME_TRACE_LABEL_MAX_BYTES) {
            return NULL;
        }
        char *copy = malloc(length + 1U);
        if (copy == NULL) {
            return NULL;
        }
        memcpy(copy, action->trace_label, length + 1U);
        return copy;
    }
    const char *operation = action->kind == SHADOWSPILL_RUNTIME_FETCH
        ? "fetch"
        : action->kind == SHADOWSPILL_RUNTIME_EVICT ? "evict"
        : action->kind == SHADOWSPILL_RUNTIME_WRITE_BACK ? "write_back"
        : "release";
    char fallback[256];
    const int written = snprintf(
        fallback,
        sizeof(fallback),
        "shadowspill.runtime.transfer.%s.object_%llu.bytes_%llu.trigger_task_%llu",
        operation,
        (unsigned long long)action->object_id,
        (unsigned long long)size_bytes,
        (unsigned long long)task_id
    );
    if (written < 0 || (size_t)written >= sizeof(fallback)) {
        return NULL;
    }
    return strdup(fallback);
}

uint64_t shadowspill_task_bucket(
    const ShadowSpillTaskTable *table,
    uint64_t task_id
) {
    task_id ^= task_id >> 33U;
    task_id *= UINT64_C(0xff51afd7ed558ccd);
    task_id ^= task_id >> 33U;
    return task_id % table->bucket_count;
}

ShadowSpillTaskRecord *shadowspill_task_find_unlocked(
    const ShadowSpillTaskTable *table,
    uint64_t task_id
) {
    if (table->by_id == NULL || table->bucket_count == 0U) {
        return NULL;
    }
    const uint64_t bucket = shadowspill_task_bucket(table, task_id);
    for (ShadowSpillTaskRecord *record = table->by_id[bucket];
         record != NULL; record = record->hash_next) {
        if (record->task_id == task_id) {
            return record;
        }
    }
    return NULL;
}

void shadowspill_task_destroy_record(ShadowSpillTaskRecord *record) {
    if (record == NULL) {
        return;
    }
    for (uint32_t index = 0U; index < record->input_count; ++index) {
        shadowspill_object_release(record->inputs[index]);
    }
    for (uint32_t index = 0U; index < record->update_count; ++index) {
        shadowspill_object_release(record->updates[index].object);
    }
    for (uint32_t index = 0U; index < record->publication_count; ++index) {
        shadowspill_object_release(record->publications[index].object);
    }
    for (uint32_t index = 0U; index < record->action_count; ++index) {
        shadowspill_object_release(record->actions[index].object);
        free(record->actions[index].trace_label);
    }
    free(record->inputs);
    free(record->input_plan_object_ids);
    free(record->input_consistency);
    free(record->unique_inputs);
    free(record->input_unique_indices);
    free(record->unique_first_positions);
    free(record->input_bindings);
    free(record->updates);
    free(record->publications);
    free(record->actions);
    free(record->queued_actions);
    free(record->release_bindings);
    free(record->allocation_contract_steps);
    free(record->allocation_contract_states);
    free(record->trace_label);
    free(record);
}

int shadowspill_task_compare_release_bindings(const void *left, const void *right) {
    const ShadowSpillTaskReleaseBinding *lhs = left;
    const ShadowSpillTaskReleaseBinding *rhs = right;
    const uintptr_t lhs_object = (uintptr_t)lhs->object;
    const uintptr_t rhs_object = (uintptr_t)rhs->object;
    return lhs_object < rhs_object ? -1 : lhs_object > rhs_object ? 1 : 0;
}

ShadowSpillQueuedAction *shadowspill_task_release_action(
    const ShadowSpillTaskRecord *record,
    const ShadowSpillObject *object
) {
    if (record == NULL || object == NULL) {
        return NULL;
    }
    uint32_t lower = 0U;
    uint32_t upper = record->release_binding_count;
    const uintptr_t key = (uintptr_t)object;
    while (lower < upper) {
        const uint32_t middle = lower + (upper - lower) / 2U;
        const uintptr_t candidate =
            (uintptr_t)record->release_bindings[middle].object;
        if (candidate < key) {
            lower = middle + 1U;
        } else {
            upper = middle;
        }
    }
    return lower < record->release_binding_count &&
            record->release_bindings[lower].object == object
        ? record->release_bindings[lower].action
        : NULL;
}

void shadowspill_task_clear_pending_handoffs(
    const ShadowSpillTaskRecord *record
) {
    if (record == NULL) {
        return;
    }
    for (uint32_t index = 0U; index < record->release_binding_count; ++index) {
        ShadowSpillQueuedAction *action =
            record->release_bindings[index].action;
        if (!action->active) {
            action->handoff_lease = NULL;
            action->handoff_generation = 0U;
        }
    }
}

int shadowspill_task_table_initialize(
    ShadowSpillTaskTable *table,
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

void shadowspill_task_table_destroy(ShadowSpillTaskTable *table) {
    if (table == NULL) {
        return;
    }
    ShadowSpillTaskRecord *record = table->owned_head;
    while (record != NULL) {
        ShadowSpillTaskRecord *next = record->ownership_next;
        shadowspill_task_destroy_record(record);
        record = next;
    }
    free(table->by_id);
    if (table->lock_initialized) {
        pthread_rwlock_destroy(&table->lock);
    }
    *table = (ShadowSpillTaskTable){0};
}

void shadowspill_task_table_clear(ShadowSpillTaskTable *table) {
    if (table == NULL || !table->lock_initialized) {
        return;
    }
    pthread_rwlock_wrlock(&table->lock);
    ShadowSpillTaskRecord *record = table->owned_head;
    table->owned_head = NULL;
    memset(
        table->by_id,
        0,
        (size_t)table->bucket_count * sizeof(*table->by_id)
    );
    pthread_rwlock_unlock(&table->lock);

    while (record != NULL) {
        ShadowSpillTaskRecord *next = record->ownership_next;
        shadowspill_task_destroy_record(record);
        record = next;
    }
}

ShadowSpillTaskRecord *shadowspill_task_table_acquire(
    ShadowSpillTaskTable *table,
    uint64_t task_id
) {
    if (table == NULL || !table->lock_initialized) {
        return NULL;
    }
    pthread_rwlock_rdlock(&table->lock);
    ShadowSpillTaskRecord *record = shadowspill_task_find_unlocked(table, task_id);
    pthread_rwlock_unlock(&table->lock);
    return record;
}
