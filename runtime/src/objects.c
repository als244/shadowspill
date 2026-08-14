#include "internal.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

ShadowSpillObject *shadowspill_find_object(
    ShadowSpillRuntime *runtime,
    uint64_t object_id
) {
    return shadowspill_object_table_find(&runtime->objects, object_id);
}

ShadowSpillRuntimeStatus shadowspill_register_object(
    ShadowSpillRuntime *runtime,
    const ShadowSpillObjectDescription *description
) {
    if (runtime == NULL || description == NULL ||
        description->object_id == SHADOWSPILL_RUNTIME_NO_ID) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&runtime->mutex);
    ShadowSpillRuntimeStatus status = shadowspill_current_status_locked(runtime);
    if (status != SHADOWSPILL_RUNTIME_OK) {
        goto done;
    }
    if (shadowspill_find_object(runtime, description->object_id) != NULL) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
        goto done;
    }
    ShadowSpillObject *created = calloc(1U, sizeof(*created));
    if (created == NULL) {
        status = SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
        goto done;
    }
    created->locations = calloc(
        runtime->pool_count, sizeof(*created->locations)
    );
    if (created->locations == NULL) {
        free(created);
        status = SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
        goto done;
    }
    created->location_count = runtime->pool_count;
    atomic_init(&created->references, 1U);
    atomic_init(&created->detached, 0U);
    atomic_init(&created->prefetch_pending, 0U);
    if (pthread_mutex_init(&created->lock, NULL) != 0) {
        free(created->locations);
        free(created);
        status = SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
        goto done;
    }
    if (pthread_cond_init(&created->state_changed, NULL) != 0) {
        pthread_mutex_destroy(&created->lock);
        free(created->locations);
        free(created);
        status = SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
        goto done;
    }
    created->object_id = description->object_id;
    created->size_bytes = description->size_bytes;
    created->authoritative_version = description->initial_version;
    created->retain_spill_copy = description->retain_spill_copy;
    created->allocation_id = SHADOWSPILL_RUNTIME_NO_ID;
    created->handoff_destination_object_id = SHADOWSPILL_RUNTIME_NO_ID;
    created->handoff_task_id = SHADOWSPILL_RUNTIME_NO_ID;
    created->handoff_next_source_object_id = SHADOWSPILL_RUNTIME_NO_ID;
    created->residency = description->initially_spill_resident
        ? SHADOWSPILL_OBJECT_SPILL_ONLY
        : SHADOWSPILL_OBJECT_RELEASED;
    if (description->initially_spill_resident) {
        ShadowSpillMemoryLease *spill_lease = calloc(
            1U, sizeof(*spill_lease)
        );
        if (spill_lease == NULL) {
            pthread_cond_destroy(&created->state_changed);
            pthread_mutex_destroy(&created->lock);
            free(created->locations);
            free(created);
            status = SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
            goto done;
        }
        shadowspill_memory_pool_lock_foreground(shadowspill_spill_pool(runtime));
        const int reserve_status = shadowspill_memory_pool_reserve_lease_locked(
            shadowspill_spill_pool(runtime),
            spill_lease,
            description->size_bytes,
            1U,
            SHADOWSPILL_MEMORY_FIRST_FIT
        );
        shadowspill_memory_pool_unlock_foreground(shadowspill_spill_pool(runtime));
        if (reserve_status != 0) {
            free(spill_lease);
            pthread_cond_destroy(&created->state_changed);
            pthread_mutex_destroy(&created->lock);
            free(created->locations);
            free(created);
            status = SHADOWSPILL_RUNTIME_OUT_OF_MEMORY;
            goto done;
        }
        shadowspill_spill_location(runtime, created)->lease = spill_lease;
        shadowspill_spill_location(runtime, created)->owns_lease = 1U;
        shadowspill_spill_location(runtime, created)->lease->state = SHADOWSPILL_LEASE_ACTIVE;
        shadowspill_spill_location(runtime, created)->current = 1U;
        shadowspill_spill_location(runtime, created)->version = description->initial_version;
    }
    if (shadowspill_object_table_insert(&runtime->objects, created) != 0) {
        if (shadowspill_spill_location(runtime, created)->lease != NULL) {
            shadowspill_memory_pool_lock_foreground(shadowspill_spill_pool(runtime));
            (void)shadowspill_memory_pool_release_lease_locked(
                shadowspill_spill_location(runtime, created)->lease
            );
            shadowspill_memory_pool_unlock_foreground(shadowspill_spill_pool(runtime));
            free(shadowspill_spill_location(runtime, created)->lease);
            shadowspill_spill_location(runtime, created)->lease = NULL;
            shadowspill_spill_location(runtime, created)->owns_lease = 0U;
        }
        pthread_cond_destroy(&created->state_changed);
        pthread_mutex_destroy(&created->lock);
        free(created->locations);
        free(created);
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
        goto done;
    }
    ++runtime->registered_objects;

done:
    pthread_mutex_unlock(&runtime->mutex);
    return status;
}

ShadowSpillRuntimeStatus shadowspill_unregister_object(
    ShadowSpillRuntime *runtime,
    uint64_t object_id
) {
    if (runtime == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&runtime->mutex);
    ShadowSpillRuntimeStatus status = shadowspill_current_status_locked(runtime);
    ShadowSpillObject *object = shadowspill_find_object(runtime, object_id);
    if (status != SHADOWSPILL_RUNTIME_OK) {
        goto done;
    }
    if (object == NULL || object->allocation_id != SHADOWSPILL_RUNTIME_NO_ID ||
        (object->residency != SHADOWSPILL_OBJECT_SPILL_ONLY &&
         object->residency != SHADOWSPILL_OBJECT_RELEASED)) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
        goto done;
    }
    pthread_mutex_lock(&runtime->actions.lock);
    for (ShadowSpillQueuedAction *action = runtime->actions.head;
         action != NULL; action = action->next) {
        if (action->object == object) {
            status = SHADOWSPILL_RUNTIME_INVALID_STATE;
            pthread_mutex_unlock(&runtime->actions.lock);
            goto done;
        }
    }
    pthread_mutex_unlock(&runtime->actions.lock);
    if (shadowspill_spill_location(runtime, object)->lease != NULL) {
        shadowspill_memory_pool_lock_foreground(shadowspill_spill_pool(runtime));
        const int release_status = shadowspill_memory_pool_release_lease_locked(
            shadowspill_spill_location(runtime, object)->lease
        );
        shadowspill_memory_pool_unlock_foreground(shadowspill_spill_pool(runtime));
        if (release_status != 0) {
            status = SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
            goto done;
        }
        free(shadowspill_spill_location(runtime, object)->lease);
        shadowspill_spill_location(runtime, object)->lease = NULL;
        shadowspill_spill_location(runtime, object)->owns_lease = 0U;
    }
    if (shadowspill_object_table_remove(&runtime->objects, object) != 0) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
        goto done;
    }
    --runtime->registered_objects;
    shadowspill_object_release(object);

done:
    pthread_mutex_unlock(&runtime->mutex);
    return status;
}

ShadowSpillRuntimeStatus shadowspill_rekey_object(
    ShadowSpillRuntime *runtime,
    uint64_t object_id,
    uint64_t replacement_object_id
) {
    if (runtime == NULL || object_id == SHADOWSPILL_RUNTIME_NO_ID ||
        replacement_object_id == SHADOWSPILL_RUNTIME_NO_ID ||
        object_id == replacement_object_id) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&runtime->mutex);
    ShadowSpillRuntimeStatus status = shadowspill_current_status_locked(runtime);
    ShadowSpillObject *object = shadowspill_find_object(runtime, object_id);
    if (status != SHADOWSPILL_RUNTIME_OK) {
        goto done;
    }
    if (object == NULL ||
        shadowspill_find_object(runtime, replacement_object_id) != NULL ||
        object->allocation_id != SHADOWSPILL_RUNTIME_NO_ID ||
        object->has_readiness_event ||
        (object->residency != SHADOWSPILL_OBJECT_SPILL_ONLY &&
         object->residency != SHADOWSPILL_OBJECT_RELEASED)) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
        goto done;
    }
    pthread_mutex_lock(&runtime->actions.lock);
    for (ShadowSpillQueuedAction *action = runtime->actions.head;
         action != NULL; action = action->next) {
        if (action->object == object) {
            status = SHADOWSPILL_RUNTIME_INVALID_STATE;
            break;
        }
    }
    pthread_mutex_unlock(&runtime->actions.lock);
    if (status != SHADOWSPILL_RUNTIME_OK) {
        goto done;
    }
    if (shadowspill_object_table_rekey(
            &runtime->objects, object, replacement_object_id
        ) != 0) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
    }

done:
    pthread_mutex_unlock(&runtime->mutex);
    return status;
}

ShadowSpillRuntimeStatus shadowspill_write_spill_object(
    ShadowSpillRuntime *runtime,
    uint64_t object_id,
    const void *source,
    uint64_t bytes
) {
    if (runtime == NULL || bytes > SIZE_MAX ||
        (bytes != 0U && source == NULL)) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&runtime->mutex);
    ShadowSpillRuntimeStatus status = shadowspill_current_status_locked(runtime);
    ShadowSpillObject *object = shadowspill_find_object(runtime, object_id);
    if (status != SHADOWSPILL_RUNTIME_OK) {
        goto done;
    }
    if (object == NULL || bytes != object->size_bytes ||
        shadowspill_spill_location(runtime, object)->lease == NULL ||
        object->residency != SHADOWSPILL_OBJECT_SPILL_ONLY ||
        object->allocation_id != SHADOWSPILL_RUNTIME_NO_ID) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
        goto done;
    }
    if (bytes != 0U) {
        memcpy(
            shadowspill_spill_location(runtime, object)->lease->pointer,
            source,
            (size_t)bytes
        );
    }
    shadowspill_spill_location(runtime, object)->current = 1U;
    shadowspill_spill_location(runtime, object)->version = object->authoritative_version;

done:
    pthread_mutex_unlock(&runtime->mutex);
    return status;
}

ShadowSpillRuntimeStatus shadowspill_read_spill_object(
    ShadowSpillRuntime *runtime,
    uint64_t object_id,
    void *destination,
    uint64_t bytes
) {
    if (runtime == NULL || bytes > SIZE_MAX ||
        (bytes != 0U && destination == NULL)) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&runtime->mutex);
    ShadowSpillRuntimeStatus status = shadowspill_current_status_locked(runtime);
    ShadowSpillObject *object = shadowspill_find_object(runtime, object_id);
    if (status != SHADOWSPILL_RUNTIME_OK) {
        goto done;
    }
    if (object == NULL || bytes != object->size_bytes ||
        shadowspill_spill_location(runtime, object)->lease == NULL || !shadowspill_spill_location(runtime, object)->current ||
        shadowspill_spill_location(runtime, object)->version != object->authoritative_version ||
        object->residency != SHADOWSPILL_OBJECT_SPILL_ONLY) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
        goto done;
    }
    if (bytes != 0U) {
        memcpy(
            destination,
            shadowspill_spill_location(runtime, object)->lease->pointer,
            (size_t)bytes
        );
    }

done:
    pthread_mutex_unlock(&runtime->mutex);
    return status;
}

ShadowSpillRuntimeStatus shadowspill_bind_object(
    ShadowSpillRuntime *runtime,
    uint64_t object_id,
    uint64_t allocation_id
) {
    if (runtime == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&runtime->mutex);
    pthread_mutex_lock(&shadowspill_execution_pool(runtime)->lock);
    ShadowSpillObject *object = shadowspill_find_object(runtime, object_id);
    ShadowSpillMemoryLease *allocation = shadowspill_find_execution_lease(
        runtime, allocation_id
    );
    ShadowSpillRuntimeStatus status = SHADOWSPILL_RUNTIME_OK;
    if (object == NULL || allocation == NULL || allocation->logical_freed ||
        allocation->pointer == NULL || object->allocation_id !=
            SHADOWSPILL_RUNTIME_NO_ID ||
        allocation->requested_bytes < object->size_bytes) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
        goto done;
    }
    ShadowSpillObject *previous_owner = NULL;
    if (allocation->bound_object_id != SHADOWSPILL_RUNTIME_NO_ID) {
        previous_owner = shadowspill_find_object(
            runtime, allocation->bound_object_id
        );
        if (previous_owner == NULL || previous_owner == object ||
            previous_owner->allocation_id != allocation_id) {
            status = SHADOWSPILL_RUNTIME_INVALID_STATE;
            goto done;
        }
    }
    const uint64_t task_id = shadowspill_current_task_id(runtime);
    if (previous_owner != NULL &&
        (previous_owner->handoff_task_id != SHADOWSPILL_RUNTIME_NO_ID ||
         task_id == SHADOWSPILL_RUNTIME_NO_ID ||
         (previous_owner->residency != SHADOWSPILL_OBJECT_EXECUTION_READY &&
          previous_owner->residency != SHADOWSPILL_OBJECT_PREFETCHING))) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
        goto done;
    }
    allocation->plan_owned = 1;
    allocation->ever_plan_owned = 1;
    shadowspill_append_allocation_event_locked(
        runtime,
        allocation,
        SHADOWSPILL_ALLOCATION_PROMOTED,
        SHADOWSPILL_ALLOCATION_PLANNED_OBJECT
    );
    if (shadowspill_failure_status(runtime) != SHADOWSPILL_RUNTIME_OK) {
        status = shadowspill_failure_status(runtime);
        goto done;
    }
    if (previous_owner != NULL) {
        ShadowSpillObject *tail = NULL;
        if (allocation->handoff_tail_object_id !=
                SHADOWSPILL_RUNTIME_NO_ID) {
            tail = shadowspill_find_object(
                runtime, allocation->handoff_tail_object_id
            );
            if (tail == NULL || tail->handoff_next_source_object_id !=
                    SHADOWSPILL_RUNTIME_NO_ID) {
                status = SHADOWSPILL_RUNTIME_INVALID_STATE;
                goto done;
            }
        }
        previous_owner->handoff_destination_object_id = object->object_id;
        previous_owner->handoff_task_id = task_id;
        previous_owner->handoff_next_source_object_id =
            SHADOWSPILL_RUNTIME_NO_ID;
        if (tail == NULL) {
            allocation->handoff_head_object_id = previous_owner->object_id;
        } else {
            tail->handoff_next_source_object_id = previous_owner->object_id;
        }
        allocation->handoff_tail_object_id = previous_owner->object_id;
    }
    object->allocation_id = allocation_id;
    shadowspill_execution_location(runtime, object)->lease = allocation;
    allocation->bound_object_id = object->object_id;
    object->generation = allocation->generation;
    shadowspill_execution_location(runtime, object)->version = object->authoritative_version;
    object->residency = SHADOWSPILL_OBJECT_EXECUTION_READY;

done:
    pthread_mutex_unlock(&shadowspill_execution_pool(runtime)->lock);
    pthread_mutex_unlock(&runtime->mutex);
    return status;
}

ShadowSpillRuntimeStatus shadowspill_replace_object_allocation(
    ShadowSpillRuntime *runtime,
    uint64_t object_id,
    uint64_t allocation_id,
    ShadowSpillObjectBinding *binding
) {
    if (runtime == NULL || binding == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    const uint64_t task_id = shadowspill_current_task_id(runtime);
    if (task_id == SHADOWSPILL_RUNTIME_NO_ID) {
        return SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    ShadowSpillObject *object = shadowspill_object_table_acquire(
        &runtime->objects, object_id
    );
    if (object == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_STATE;
    }

    ShadowSpillEventLease *retired_readiness = NULL;
    ShadowSpillRuntimeStatus status = shadowspill_current_status_locked(runtime);
    pthread_mutex_lock(&object->lock);
    shadowspill_memory_pool_lock_foreground(shadowspill_execution_pool(runtime));
    ShadowSpillMemoryLease *replacement = shadowspill_find_execution_lease(
        runtime, allocation_id
    );
    ShadowSpillMemoryLease *prior = shadowspill_execution_location(
        runtime, object
    )->lease;
    if (status != SHADOWSPILL_RUNTIME_OK) {
        goto done;
    }
    if ((object->residency != SHADOWSPILL_OBJECT_EXECUTION_READY &&
         object->residency != SHADOWSPILL_OBJECT_PREFETCHING) ||
        prior == NULL || prior->pointer == NULL || prior->logical_freed ||
        prior->allocation_id != object->allocation_id ||
        prior->generation != object->generation || replacement == NULL ||
        replacement == prior || replacement->pointer == NULL ||
        replacement->logical_freed || replacement->plan_owned ||
        replacement->bound_object_id != SHADOWSPILL_RUNTIME_NO_ID ||
        replacement->requested_bytes < object->size_bytes) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
        goto done;
    }

    replacement->plan_owned = 1;
    replacement->ever_plan_owned = 1;
    replacement->bound_object_id = object->object_id;
    replacement->state = SHADOWSPILL_LEASE_ACTIVE;
    shadowspill_append_allocation_event_locked(
        runtime,
        replacement,
        SHADOWSPILL_ALLOCATION_PROMOTED,
        SHADOWSPILL_ALLOCATION_PLANNED_OBJECT
    );
    if (shadowspill_failure_status(runtime) != SHADOWSPILL_RUNTIME_OK) {
        status = shadowspill_failure_status(runtime);
        replacement->plan_owned = 0;
        replacement->ever_plan_owned = 0;
        replacement->bound_object_id = SHADOWSPILL_RUNTIME_NO_ID;
        goto done;
    }

    object->retired_generation = object->generation;
    object->retired_execution_pointer = prior->pointer;
    prior->bound_object_id = SHADOWSPILL_RUNTIME_NO_ID;
    prior->release_task_id = task_id;
    prior->logical_freed = 1;
    prior->state = SHADOWSPILL_LEASE_RETIRING;
    shadowspill_append_allocation_event_locked(
        runtime,
        prior,
        SHADOWSPILL_ALLOCATION_LOGICAL_FREED,
        SHADOWSPILL_ALLOCATION_PLANNED_OBJECT
    );
    (void)atomic_fetch_add_explicit(
        &runtime->pending_retirements, 1U, memory_order_acq_rel
    );

    ShadowSpillObjectLocation *execution = shadowspill_execution_location(
        runtime, object
    );
    execution->lease = replacement;
    execution->version = object->authoritative_version;
    execution->current = 1U;
    object->allocation_id = replacement->allocation_id;
    object->generation = replacement->generation;
    object->residency = SHADOWSPILL_OBJECT_EXECUTION_READY;
    if (object->has_readiness_event) {
        retired_readiness = object->readiness_event;
        object->readiness_event = NULL;
        object->has_readiness_event = 0U;
    }
    *binding = (ShadowSpillObjectBinding){
        .object_id = object->object_id,
        .generation = object->generation,
        .allocation_id = object->allocation_id,
        .authoritative_version = object->authoritative_version,
        .pointer = replacement->pointer,
    };
    pthread_cond_broadcast(&object->state_changed);

done:
    shadowspill_memory_pool_unlock_foreground(shadowspill_execution_pool(runtime));
    pthread_mutex_unlock(&object->lock);
    shadowspill_object_release(object);
    if (retired_readiness != NULL &&
        shadowspill_event_lease_release(runtime, retired_readiness) != 0 &&
        status == SHADOWSPILL_RUNTIME_OK) {
        status = SHADOWSPILL_RUNTIME_BACKEND_FAILURE;
    }
    return status;
}

ShadowSpillRuntimeStatus shadowspill_transfer_object_to_caller(
    ShadowSpillRuntime *runtime,
    uint64_t object_id,
    ShadowSpillAllocation *allocation
) {
    if (runtime == NULL || allocation == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&runtime->mutex);
    ShadowSpillRuntimeStatus status = shadowspill_current_status_locked(runtime);
    ShadowSpillObject *object = shadowspill_find_object(runtime, object_id);
    if (status != SHADOWSPILL_RUNTIME_OK) {
        goto done_runtime;
    }
    if (object == NULL ||
        object->residency != SHADOWSPILL_OBJECT_EXECUTION_READY) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
        goto done_runtime;
    }
    pthread_mutex_lock(&runtime->actions.lock);
    for (ShadowSpillQueuedAction *action = runtime->actions.head;
         action != NULL; action = action->next) {
        if (action->object == object) {
            status = SHADOWSPILL_RUNTIME_INVALID_STATE;
            pthread_mutex_unlock(&runtime->actions.lock);
            goto done_runtime;
        }
    }
    pthread_mutex_unlock(&runtime->actions.lock);
    pthread_mutex_lock(&shadowspill_execution_pool(runtime)->lock);
    ShadowSpillMemoryLease *record = shadowspill_find_execution_lease(
        runtime, object->allocation_id
    );
    if (record == NULL || record->pointer == NULL || record->logical_freed ||
        !record->plan_owned) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
        goto done_allocation;
    }
    if (shadowspill_spill_location(runtime, object)->lease != NULL) {
        shadowspill_memory_pool_lock_foreground(shadowspill_spill_pool(runtime));
        const int release_status = shadowspill_memory_pool_release_lease_locked(
            shadowspill_spill_location(runtime, object)->lease
        );
        shadowspill_memory_pool_unlock_foreground(shadowspill_spill_pool(runtime));
        if (release_status != 0) {
            status = SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
            goto done_allocation;
        }
    }
    if (shadowspill_object_table_remove(&runtime->objects, object) != 0) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
        goto done_allocation;
    }
    --runtime->registered_objects;
    record->framework_free_seen = 0;
    record->plan_owned = 0;
    record->bound_object_id = SHADOWSPILL_RUNTIME_NO_ID;
    shadowspill_append_allocation_event_locked(
        runtime,
        record,
        SHADOWSPILL_ALLOCATION_PROMOTED,
        SHADOWSPILL_ALLOCATION_CALLER_OWNED
    );
    *allocation = (ShadowSpillAllocation){
        .allocation_id = record->allocation_id,
        .generation = record->generation,
        .requested_bytes = record->requested_bytes,
        .charged_bytes = record->charged_bytes,
        .pointer = record->pointer,
    };
    shadowspill_object_release(object);

done_allocation:
    pthread_mutex_unlock(&shadowspill_execution_pool(runtime)->lock);
done_runtime:
    pthread_mutex_unlock(&runtime->mutex);
    return status;
}

ShadowSpillRuntimeStatus shadowspill_object_snapshot(
    ShadowSpillRuntime *runtime,
    uint64_t object_id,
    ShadowSpillObjectSnapshot *snapshot
) {
    if (runtime == NULL || snapshot == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&runtime->mutex);
    pthread_mutex_lock(&shadowspill_execution_pool(runtime)->lock);
    ShadowSpillObject *object = shadowspill_find_object(runtime, object_id);
    if (object == NULL) {
        pthread_mutex_unlock(&shadowspill_execution_pool(runtime)->lock);
        pthread_mutex_unlock(&runtime->mutex);
        return SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    ShadowSpillMemoryLease *allocation = shadowspill_find_execution_lease(
        runtime, object->allocation_id
    );
    *snapshot = (ShadowSpillObjectSnapshot){
        .object_id = object->object_id,
        .size_bytes = object->size_bytes,
        .generation = object->generation,
        .allocation_id = object->allocation_id,
        .authoritative_version = object->authoritative_version,
        .execution_version = shadowspill_execution_location(runtime, object)->version,
        .spill_version = shadowspill_spill_location(runtime, object)->version,
        .residency = object->residency,
        .spill_current = shadowspill_spill_location(runtime, object)->current,
        .has_spill_lease =
            shadowspill_spill_location(runtime, object)->lease != NULL,
        .execution_pointer = allocation == NULL ? NULL : allocation->pointer,
        .spill_pointer = shadowspill_spill_location(runtime, object)->lease == NULL
            ? NULL
            : shadowspill_spill_location(runtime, object)->lease->pointer,
        .retired_generation = object->retired_generation,
        .retired_execution_pointer = object->retired_execution_pointer,
    };
    pthread_mutex_unlock(&shadowspill_execution_pool(runtime)->lock);
    pthread_mutex_unlock(&runtime->mutex);
    return SHADOWSPILL_RUNTIME_OK;
}

ShadowSpillRuntimeStatus shadowspill_before_task_legacy(
    ShadowSpillRuntime *runtime,
    uint64_t task_id,
    ShadowSpillBackendStream compute_stream,
    const uint64_t *input_object_ids,
    uint32_t input_count,
    ShadowSpillObjectBinding *bindings,
    uint32_t binding_capacity
) {
    if (runtime == NULL || (input_count != 0U && input_object_ids == NULL) ||
        (input_count != 0U && bindings == NULL) ||
        binding_capacity < input_count) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&runtime->mutex);
    shadowspill_append_trace_event_locked(
        runtime,
        SHADOWSPILL_TRACE_BEFORE_TASK,
        task_id,
        SHADOWSPILL_RUNTIME_NO_ID,
        SHADOWSPILL_RUNTIME_NO_ID,
        0U,
        input_count,
        atomic_load_explicit(&runtime->actions.count, memory_order_acquire)
    );
    ShadowSpillRuntimeStatus status = shadowspill_current_status_locked(runtime);
    for (uint32_t index = 0; status == SHADOWSPILL_RUNTIME_OK &&
         index < input_count; ++index) {
        ShadowSpillObject *object = NULL;
        while (status == SHADOWSPILL_RUNTIME_OK) {
            object = shadowspill_find_object(runtime, input_object_ids[index]);
            if (object == NULL ||
                object->residency != SHADOWSPILL_OBJECT_SPILL_ONLY) {
                break;
            }
            int pending_prefetch = 0;
            pthread_mutex_lock(&runtime->actions.lock);
            for (ShadowSpillQueuedAction *action = runtime->actions.head;
                 action != NULL; action = action->next) {
                if (action->object == object &&
                    action->kind == SHADOWSPILL_RUNTIME_PREFETCH) {
                    pending_prefetch = 1;
                    break;
                }
            }
            pthread_mutex_unlock(&runtime->actions.lock);
            if (!pending_prefetch) {
                break;
            }
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
            pthread_cond_wait(&runtime->condition, &runtime->mutex);
            status = shadowspill_current_status_locked(runtime);
        }
        if (object == NULL) {
            status = SHADOWSPILL_RUNTIME_INVALID_STATE;
            break;
        }
        pthread_mutex_lock(&shadowspill_execution_pool(runtime)->lock);
        ShadowSpillMemoryLease *allocation = shadowspill_find_execution_lease(
            runtime, object->allocation_id
        );
        void *device_pointer = allocation == NULL ? NULL : allocation->pointer;
        if ((object->residency != SHADOWSPILL_OBJECT_EXECUTION_READY &&
             object->residency != SHADOWSPILL_OBJECT_PREFETCHING) ||
            allocation == NULL || device_pointer == NULL ||
            shadowspill_execution_location(runtime, object)->version != object->authoritative_version) {
            pthread_mutex_unlock(&shadowspill_execution_pool(runtime)->lock);
            status = SHADOWSPILL_RUNTIME_PLAN_VIOLATION;
            shadowspill_latch_failure_locked(
                runtime,
                status,
                object->object_id,
                object->allocation_id,
                object->size_bytes
            );
            break;
        }
        pthread_mutex_unlock(&shadowspill_execution_pool(runtime)->lock);
        int duplicate = 0;
        for (uint32_t previous = 0; previous < index; ++previous) {
            if (input_object_ids[previous] == input_object_ids[index]) {
                duplicate = 1;
                break;
            }
        }
        if (!duplicate && object->residency == SHADOWSPILL_OBJECT_PREFETCHING) {
            if (!object->has_readiness_event || runtime->backend.wait_event(
                    runtime->backend.context,
                    compute_stream,
                    object->readiness_event->event
                ) != 0) {
                status = SHADOWSPILL_RUNTIME_BACKEND_FAILURE;
                shadowspill_latch_failure_locked(
                    runtime,
                    status,
                    object->object_id,
                    object->allocation_id,
                    object->size_bytes
                );
                break;
            }
            ++runtime->wait_events_inserted;
            shadowspill_append_trace_event_locked(
                runtime,
                SHADOWSPILL_TRACE_READINESS_WAIT,
                task_id,
                object->object_id,
                object->allocation_id,
                object->size_bytes,
                1U,
                runtime->wait_events_inserted
            );
        }
        bindings[index] = (ShadowSpillObjectBinding){
            .object_id = object->object_id,
            .generation = object->generation,
            .allocation_id = object->allocation_id,
            .authoritative_version = object->authoritative_version,
            .pointer = device_pointer,
        };
    }
    if (status == SHADOWSPILL_RUNTIME_OK &&
        shadowspill_enter_task_scope(runtime, task_id) != 0) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    pthread_mutex_unlock(&runtime->mutex);
    return status;
}

static void discard_actions_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillQueuedAction *head
) {
    while (head != NULL) {
        ShadowSpillQueuedAction *next = head->next;
        if (head->kind == SHADOWSPILL_RUNTIME_PREFETCH) {
            pthread_mutex_lock(&head->object->lock);
            atomic_store_explicit(
                &head->object->prefetch_pending, 0U, memory_order_release
            );
            pthread_cond_broadcast(&head->object->state_changed);
            pthread_mutex_unlock(&head->object->lock);
        }
        shadowspill_release_task_fence_locked(runtime, head->fence);
        shadowspill_object_release(head->object);
        if (head->owns_trace_label) {
            free((void *)head->trace_label);
        }
        free(head);
        head = next;
    }
}

ShadowSpillRuntimeStatus shadowspill_after_task_legacy(
    ShadowSpillRuntime *runtime,
    uint64_t task_id,
    ShadowSpillBackendStream compute_stream,
    const ShadowSpillObjectUpdate *updates,
    uint32_t update_count,
    const ShadowSpillRuntimeAction *actions,
    uint32_t action_count
) {
    if (runtime == NULL || (action_count != 0U && actions == NULL) ||
        (update_count != 0U && updates == NULL)) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&runtime->mutex);
    ShadowSpillRuntimeStatus status = shadowspill_current_status_locked(runtime);
    ShadowSpillTaskFence *fence = NULL;
    ShadowSpillQueuedAction *head = NULL;
    ShadowSpillQueuedAction *tail = NULL;
    uint64_t failure_object_id = SHADOWSPILL_RUNTIME_NO_ID;
    uint64_t failure_allocation_id = SHADOWSPILL_RUNTIME_NO_ID;
    if (status != SHADOWSPILL_RUNTIME_OK) {
        goto done;
    }
    for (uint32_t index = 0; index < update_count; ++index) {
        ShadowSpillObject *object = shadowspill_find_object(
            runtime, updates[index].object_id
        );
        if (object == NULL || updates[index].version_delta == 0U ||
            (object->residency != SHADOWSPILL_OBJECT_EXECUTION_READY &&
             object->residency != SHADOWSPILL_OBJECT_PREFETCHING) ||
            updates[index].version_delta >
                UINT64_MAX - object->authoritative_version) {
            status = SHADOWSPILL_RUNTIME_INVALID_STATE;
            shadowspill_latch_failure_locked(
                runtime,
                status,
                updates[index].object_id,
                object == NULL ? SHADOWSPILL_RUNTIME_NO_ID : object->allocation_id,
                0U
            );
            goto done;
        }
        object->authoritative_version += updates[index].version_delta;
        shadowspill_execution_location(runtime, object)->version = object->authoritative_version;
        shadowspill_spill_location(runtime, object)->current = 0U;
    }
    pthread_mutex_lock(&shadowspill_execution_pool(runtime)->lock);
    for (ShadowSpillMemoryLease *allocation = runtime->active_execution_leases;
         allocation != NULL; allocation = allocation->active_next) {
        uint64_t source_id = allocation->handoff_head_object_id;
        uint64_t traversed = 0U;
        while (source_id != SHADOWSPILL_RUNTIME_NO_ID) {
            ShadowSpillObject *source = shadowspill_find_object(
                runtime, source_id
            );
            if (source == NULL || ++traversed >
                    atomic_load_explicit(
                        &runtime->registered_objects, memory_order_acquire
                    )) {
                status = SHADOWSPILL_RUNTIME_INVALID_STATE;
                shadowspill_latch_failure_locked(
                    runtime,
                    status,
                    source_id,
                    allocation->allocation_id,
                    allocation->requested_bytes
                );
                break;
            }
            if (source->handoff_task_id == task_id) {
                int matched_release = 0;
                for (uint32_t index = 0; index < action_count; ++index) {
                    if (actions[index].object_id == source->object_id &&
                        actions[index].kind == SHADOWSPILL_RUNTIME_RELEASE) {
                        matched_release = 1;
                        break;
                    }
                }
                if (!matched_release) {
                    status = SHADOWSPILL_RUNTIME_PLAN_VIOLATION;
                    shadowspill_latch_failure_locked(
                        runtime,
                        status,
                        source->handoff_destination_object_id,
                        allocation->allocation_id,
                        allocation->requested_bytes
                    );
                    break;
                }
            }
            source_id = source->handoff_next_source_object_id;
        }
        if (status != SHADOWSPILL_RUNTIME_OK) {
            break;
        }
    }
    pthread_mutex_unlock(&shadowspill_execution_pool(runtime)->lock);
    if (status != SHADOWSPILL_RUNTIME_OK) {
        goto done;
    }
    uint64_t task_retirement_count = 0U;
    pthread_mutex_lock(&shadowspill_execution_pool(runtime)->lock);
    for (ShadowSpillMemoryLease *allocation = runtime->active_execution_leases;
         allocation != NULL; allocation = allocation->active_next) {
        if (allocation->logical_freed && allocation->pointer != NULL &&
            allocation->release_task_id == task_id &&
            allocation->retirement_events == NULL &&
            allocation->retirement_fence == NULL) {
            ++task_retirement_count;
        }
    }
    pthread_mutex_unlock(&shadowspill_execution_pool(runtime)->lock);
    if (action_count == 0U && task_retirement_count == 0U) {
        goto done;
    }
    fence = calloc(1U, sizeof(*fence));
    const ShadowSpillRuntimeStatus event_status = fence == NULL
        ? SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE
        : shadowspill_event_lease_create_locked(runtime, &fence->event);
    if (event_status != SHADOWSPILL_RUNTIME_OK || runtime->backend.record_event(
            runtime->backend.context, fence->event->event, compute_stream
        ) != 0 || shadowspill_completion_submit(
            runtime,
            compute_stream,
            fence->event,
            SHADOWSPILL_RUNTIME_NO_ID,
            SHADOWSPILL_RUNTIME_NO_ID
        ) != SHADOWSPILL_RUNTIME_OK) {
        if (fence != NULL && fence->event != NULL) {
            (void)shadowspill_event_lease_release(runtime, fence->event);
        }
        free(fence);
        status = SHADOWSPILL_RUNTIME_BACKEND_FAILURE;
        shadowspill_latch_failure_locked(
            runtime, status, SHADOWSPILL_RUNTIME_NO_ID,
            SHADOWSPILL_RUNTIME_NO_ID, 0U
        );
        goto done;
    }
    for (uint32_t index = 0; index < action_count; ++index) {
        ShadowSpillObject *object = shadowspill_find_object(
            runtime, actions[index].object_id
        );
        failure_object_id = actions[index].object_id;
        failure_allocation_id = object == NULL
            ? SHADOWSPILL_RUNTIME_NO_ID
            : object->allocation_id;
        if (object == NULL || actions[index].kind > SHADOWSPILL_RUNTIME_PREFETCH) {
            status = SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
            break;
        }
        for (uint32_t previous = 0; previous < index; ++previous) {
            if (actions[previous].object_id == actions[index].object_id) {
                status = SHADOWSPILL_RUNTIME_PLAN_VIOLATION;
                break;
            }
        }
        if (status != SHADOWSPILL_RUNTIME_OK) {
            break;
        }
        if (actions[index].kind == SHADOWSPILL_RUNTIME_PREFETCH) {
            if (object->residency != SHADOWSPILL_OBJECT_SPILL_ONLY ||
                !shadowspill_spill_location(runtime, object)->current ||
                shadowspill_spill_location(runtime, object)->version != object->authoritative_version) {
                status = SHADOWSPILL_RUNTIME_PLAN_VIOLATION;
                break;
            }
        } else if (object->residency != SHADOWSPILL_OBJECT_EXECUTION_READY &&
                   object->residency != SHADOWSPILL_OBJECT_PREFETCHING) {
            status = SHADOWSPILL_RUNTIME_PLAN_VIOLATION;
            break;
        }
        ShadowSpillQueuedAction *created = calloc(1U, sizeof(*created));
        if (created == NULL) {
            status = SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
            break;
        }
        created->task_id = task_id;
        created->kind = actions[index].kind;
        created->object = object;
        created->trace_label = shadowspill_copy_action_trace_label(
            &actions[index], task_id, object->size_bytes
        );
        if (created->trace_label == NULL) {
            free(created);
            status = SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
            break;
        }
        created->owns_trace_label = 1U;
        shadowspill_object_retain(object);
        created->fence = fence;
        shadowspill_retain_task_fence(fence);
        if (created->kind == SHADOWSPILL_RUNTIME_PREFETCH) {
            atomic_store_explicit(
                &object->prefetch_pending, 1U, memory_order_release
            );
        }
        if (tail == NULL) {
            head = created;
        } else {
            created->previous = tail;
            tail->next = created;
        }
        tail = created;
    }
    if (status != SHADOWSPILL_RUNTIME_OK) {
        if (atomic_load_explicit(
                &fence->references, memory_order_acquire
            ) == 0U) {
            (void)shadowspill_event_lease_release(runtime, fence->event);
            free(fence);
        } else {
            discard_actions_locked(runtime, head);
        }
        shadowspill_latch_failure_locked(
            runtime, status, failure_object_id, failure_allocation_id, 0U
        );
        goto done;
    }
    pthread_mutex_lock(&shadowspill_execution_pool(runtime)->lock);
    for (ShadowSpillMemoryLease *allocation = runtime->active_execution_leases;
         allocation != NULL; allocation = allocation->active_next) {
        if (!allocation->logical_freed || allocation->pointer == NULL ||
            allocation->release_task_id != task_id ||
            allocation->retirement_events != NULL ||
            allocation->retirement_fence != NULL) {
            continue;
        }
        allocation->retirement_fence = fence;
        shadowspill_retain_task_fence(fence);
        status = shadowspill_retirement_enqueue_locked(runtime, allocation);
        if (status != SHADOWSPILL_RUNTIME_OK) {
            break;
        }
    }
    pthread_mutex_unlock(&shadowspill_execution_pool(runtime)->lock);
    if (status != SHADOWSPILL_RUNTIME_OK) {
        shadowspill_latch_failure_locked(
            runtime,
            status,
            SHADOWSPILL_RUNTIME_NO_ID,
            SHADOWSPILL_RUNTIME_NO_ID,
            0U
        );
        goto done;
    }
    if (head != NULL) {
        for (ShadowSpillQueuedAction *queued = head; queued != NULL;
             queued = queued->next) {
            ShadowSpillTransferLane *lane =
                shadowspill_transfer_lane_for_action(runtime, queued);
            if (lane != NULL) {
                shadowspill_transfer_lane_enqueue(lane, queued);
            }
            if (queued == tail) {
                break;
            }
        }
        pthread_mutex_lock(&runtime->actions.lock);
        if (runtime->actions.tail == NULL) {
            runtime->actions.head = head;
        } else {
            head->previous = runtime->actions.tail;
            runtime->actions.tail->next = head;
        }
        runtime->actions.tail = tail;
        (void)atomic_fetch_add_explicit(
            &runtime->actions.count, action_count, memory_order_release
        );
        pthread_mutex_unlock(&runtime->actions.lock);
        for (ShadowSpillQueuedAction *queued = head; queued != NULL;
             queued = queued->next) {
            if (queued->kind == SHADOWSPILL_RUNTIME_RELEASE ||
                queued->kind == SHADOWSPILL_RUNTIME_OFFLOAD) {
                (void)atomic_fetch_add_explicit(
                    &runtime->pending_capacity_actions,
                    1U,
                    memory_order_release
                );
            }
            pthread_mutex_lock(&queued->object->lock);
            const uint64_t allocation_id = queued->object->allocation_id;
            const uint64_t size_bytes = queued->object->size_bytes;
            pthread_mutex_unlock(&queued->object->lock);
            shadowspill_append_trace_event_locked(
                runtime,
                SHADOWSPILL_TRACE_ACTION_QUEUED,
                task_id,
                queued->object->object_id,
                allocation_id,
                size_bytes,
                queued->kind,
                atomic_load_explicit(
                    &runtime->actions.count, memory_order_acquire
                )
            );
            if (queued == tail) {
                break;
            }
        }
    }
    pthread_cond_broadcast(&runtime->condition);

done:
    /*
     * A failure can be latched by an allocator callback while the compiled
     * task is still unwinding. Tensor destruction after that callback may add
     * task-local retirements. They still need a compute-stream fence; leaving
     * them unfenced gives wait_idle no possible progress source during
     * rollback.
     */
    if (status != SHADOWSPILL_RUNTIME_OK) {
        pthread_mutex_lock(&shadowspill_execution_pool(runtime)->lock);
        const ShadowSpillRuntimeStatus retirement_status =
            shadowspill_fence_task_retirements_locked(
                runtime, task_id, compute_stream
            );
        pthread_mutex_unlock(&shadowspill_execution_pool(runtime)->lock);
        if (retirement_status != SHADOWSPILL_RUNTIME_OK) {
            shadowspill_latch_failure_locked(
                runtime,
                retirement_status,
                SHADOWSPILL_RUNTIME_NO_ID,
                SHADOWSPILL_RUNTIME_NO_ID,
                0U
            );
        }
    }
    shadowspill_append_trace_event_locked(
        runtime,
        SHADOWSPILL_TRACE_AFTER_TASK,
        task_id,
        failure_object_id,
        failure_allocation_id,
        0U,
        (uint64_t)status,
        action_count
    );
    pthread_mutex_unlock(&runtime->mutex);
    shadowspill_leave_task_scope(runtime);
    return status;
}
