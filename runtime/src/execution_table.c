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
    free(record->unique_inputs);
    free(record->updates);
    free(record->actions);
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
        if (record->inputs[index]->object_id !=
            description->input_object_ids[index]) {
            return 0;
        }
    }
    for (uint32_t index = 0U; index < record->update_count; ++index) {
        if (record->updates[index].object->object_id !=
                description->updates[index].object_id ||
            record->updates[index].version_delta !=
                description->updates[index].version_delta) {
            return 0;
        }
    }
    for (uint32_t index = 0U; index < record->action_count; ++index) {
        if (record->actions[index].object->object_id !=
                description->actions[index].object_id ||
            record->actions[index].kind !=
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
        record->unique_inputs = calloc(
            record->input_count, sizeof(*record->unique_inputs)
        );
    }
    if (record->update_count != 0U) {
        record->updates = calloc(record->update_count, sizeof(*record->updates));
    }
    if (record->action_count != 0U) {
        record->actions = calloc(record->action_count, sizeof(*record->actions));
    }
    if ((record->input_count != 0U &&
         (record->inputs == NULL || record->unique_inputs == NULL)) ||
        (record->update_count != 0U && record->updates == NULL) ||
        (record->action_count != 0U && record->actions == NULL)) {
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
        int duplicate = 0;
        for (uint32_t previous = 0U;
             previous < record->unique_input_count; ++previous) {
            if (record->unique_inputs[previous] == object) {
                duplicate = 1;
                break;
            }
        }
        if (!duplicate) {
            record->unique_inputs[record->unique_input_count++] = object;
        }
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
    if ((record->input_count != 0U && bindings == NULL) ||
        binding_capacity < record->input_count) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }

    shadowspill_append_trace_event_locked(
        runtime,
        SHADOWSPILL_TRACE_BEFORE_TASK,
        task_id,
        SHADOWSPILL_RUNTIME_NO_ID,
        SHADOWSPILL_RUNTIME_NO_ID,
        0U,
        record->input_count,
        atomic_load_explicit(&runtime->actions.count, memory_order_acquire)
    );
    ShadowSpillRuntimeStatus status = shadowspill_current_status_locked(runtime);
    for (uint32_t index = 0U;
         status == SHADOWSPILL_RUNTIME_OK &&
             index < record->unique_input_count;
         ++index) {
        ShadowSpillObjectRecord *object = record->unique_inputs[index];
        pthread_mutex_lock(&object->lock);
        while (status == SHADOWSPILL_RUNTIME_OK &&
               object->residency == SHADOWSPILL_OBJECT_HOST_ONLY &&
               object->prefetch_pending) {
            shadowspill_append_trace_event_locked(
                runtime,
                SHADOWSPILL_TRACE_READINESS_WAIT,
                task_id,
                object->object_id,
                object->allocation_id,
                object->size_bytes,
                0U,
                atomic_load_explicit(
                    &runtime->actions.count, memory_order_acquire
                )
            );
            pthread_cond_wait(&object->state_changed, &object->lock);
            status = shadowspill_current_status_locked(runtime);
        }
        ShadowSpillAllocationRecord *lease = object->device_lease;
        if (status != SHADOWSPILL_RUNTIME_OK) {
            pthread_mutex_unlock(&object->lock);
            break;
        }
        if ((object->residency != SHADOWSPILL_OBJECT_DEVICE_READY &&
             object->residency != SHADOWSPILL_OBJECT_PREFETCHING) ||
            lease == NULL || lease->pointer == NULL ||
            lease->allocation_id != object->allocation_id ||
            lease->generation != object->generation ||
            object->device_version != object->authoritative_version) {
            status = SHADOWSPILL_RUNTIME_PLAN_VIOLATION;
            const uint64_t allocation_id = object->allocation_id;
            const uint64_t size_bytes = object->size_bytes;
            pthread_mutex_unlock(&object->lock);
            shadowspill_latch_failure_locked(
                runtime,
                status,
                object->object_id,
                allocation_id,
                size_bytes
            );
            break;
        }
        ShadowSpillEventLease *readiness_event = NULL;
        if (object->residency == SHADOWSPILL_OBJECT_PREFETCHING) {
            if (!object->has_readiness_event) {
                status = SHADOWSPILL_RUNTIME_BACKEND_FAILURE;
                const uint64_t allocation_id = object->allocation_id;
                const uint64_t size_bytes = object->size_bytes;
                pthread_mutex_unlock(&object->lock);
                shadowspill_latch_failure_locked(
                    runtime,
                    status,
                    object->object_id,
                    allocation_id,
                    size_bytes
                );
                break;
            }
            readiness_event = object->readiness_event;
            shadowspill_event_lease_retain(readiness_event);
        }
        const ShadowSpillObjectBinding snapshot = {
            .object_id = object->object_id,
            .generation = object->generation,
            .allocation_id = object->allocation_id,
            .authoritative_version = object->authoritative_version,
            .pointer = lease->pointer,
        };
        pthread_mutex_unlock(&object->lock);
        if (readiness_event != NULL) {
            if (runtime->backend.wait_event(
                    runtime->backend.context,
                    compute_stream,
                    readiness_event->event
                ) != 0) {
                (void)shadowspill_event_lease_release(
                    runtime, readiness_event
                );
                status = SHADOWSPILL_RUNTIME_BACKEND_FAILURE;
                shadowspill_latch_failure_locked(
                    runtime,
                    status,
                    snapshot.object_id,
                    snapshot.allocation_id,
                    object->size_bytes
                );
                break;
            }
            const uint64_t wait_count = atomic_fetch_add_explicit(
                &runtime->wait_events_inserted, 1U, memory_order_acq_rel
            ) + 1U;
            shadowspill_append_trace_event_locked(
                runtime,
                SHADOWSPILL_TRACE_READINESS_WAIT,
                task_id,
                snapshot.object_id,
                snapshot.allocation_id,
                object->size_bytes,
                1U,
                wait_count
            );
            (void)shadowspill_event_lease_release(runtime, readiness_event);
        }
        for (uint32_t position = 0U; position < record->input_count;
             ++position) {
            if (record->inputs[position] == object) {
                bindings[position] = snapshot;
            }
        }
    }
    if (status == SHADOWSPILL_RUNTIME_OK &&
        shadowspill_enter_task_scope(runtime, task_id) != 0) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    return status;
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
    return shadowspill_after_execution_record(runtime, record, compute_stream);
}
