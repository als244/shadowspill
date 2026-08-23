#include "../internal.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

ShadowSpillObject *shadowspill_find_object(
    ShadowSpillRuntime *runtime,
    uint64_t object_id
) {
    return shadowspill_object_table_find(&runtime->objects, object_id);
}

ShadowSpillStatus shadowspill_object_handle_acquire(
    ShadowSpillRuntime *runtime,
    uint64_t runtime_object_id,
    ShadowSpillObjectHandle **output
) {
    if (runtime == NULL || output == NULL ||
        runtime_object_id == SHADOWSPILL_RUNTIME_NO_ID) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    *output = NULL;
    if (atomic_load_explicit(&runtime->closing, memory_order_acquire) != 0U) {
        return SHADOWSPILL_STATUS_CLOSED;
    }
    ShadowSpillObject *object = shadowspill_object_table_acquire(
        &runtime->objects, runtime_object_id
    );
    if (object == NULL) {
        return SHADOWSPILL_STATUS_INVALID_STATE;
    }
    ShadowSpillObjectHandle *handle = malloc(sizeof(*handle));
    if (handle == NULL) {
        shadowspill_object_release(object);
        return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    }
    const ShadowSpillStatus retain_status =
        shadowspill_object_owner_retain(object);
    shadowspill_object_release(object);
    if (retain_status != SHADOWSPILL_STATUS_OK) {
        free(handle);
        return retain_status;
    }
    *handle = (ShadowSpillObjectHandle){
        .runtime = runtime,
        .object = object,
    };
    *output = handle;
    return SHADOWSPILL_STATUS_OK;
}

static ShadowSpillStatus release_object_residency(
    ShadowSpillObject *object,
    uint64_t expected_generation,
    uint8_t validate_generation
) {
    ShadowSpillRuntime *runtime = object->runtime;
    if (runtime == NULL || atomic_load_explicit(
            &runtime->closing, memory_order_acquire
        ) != 0U) {
        return SHADOWSPILL_STATUS_OK;
    }

    ShadowSpillEventLease *readiness_event = NULL;
    pthread_mutex_lock(&runtime->mutex);
    pthread_mutex_lock(&object->lock);
    if ((validate_generation && object->generation != expected_generation) ||
        object->action_head != NULL || object->action_tail != NULL ||
        shadowspill_object_has_unpublished_fetch_locked(object) ||
        object->residency == SHADOWSPILL_OBJECT_PREFETCHING ||
        object->residency == SHADOWSPILL_OBJECT_OFFLOADING) {
        pthread_mutex_unlock(&object->lock);
        pthread_mutex_unlock(&runtime->mutex);
        return SHADOWSPILL_STATUS_INVALID_STATE;
    }

    ShadowSpillStatus status = SHADOWSPILL_STATUS_OK;
    for (uint32_t pool_id = 0U;
         pool_id < object->location_count &&
             status == SHADOWSPILL_STATUS_OK;
         ++pool_id) {
        ShadowSpillObjectLocation *location = shadowspill_object_location(
            object, pool_id
        );
        ShadowSpillMemoryLease *lease = location->lease;
        if (lease == NULL) {
            continue;
        }
        ShadowSpillMemoryPool *pool = lease->pool;
        if (pool == NULL || pool->pool_id != pool_id) {
            status = SHADOWSPILL_STATUS_INVALID_STATE;
            break;
        }
        shadowspill_memory_pool_lock_foreground(pool);
        const int is_execution_generation =
            object->allocation_id != SHADOWSPILL_RUNTIME_NO_ID &&
            lease->allocation_id == object->allocation_id;
        if (is_execution_generation) {
            if (lease->logical_freed ||
                lease->bound_object != object) {
                status = SHADOWSPILL_STATUS_INVALID_STATE;
            } else {
                lease->bound_object = NULL;
                lease->plan_owned = 0U;
                object->retired_generation = object->generation;
                object->retired_execution_pointer = lease->pointer;
                object->allocation_id = SHADOWSPILL_RUNTIME_NO_ID;
                shadowspill_release_execution_lease_locked(runtime, lease);
                status = shadowspill_failure_status(runtime);
            }
        } else if (shadowspill_memory_pool_release_lease_locked(lease) != 0) {
            status = SHADOWSPILL_STATUS_INTERNAL_FAILURE;
        }
        if (status == SHADOWSPILL_STATUS_OK) {
            location->lease = NULL;
            location->version = 0U;
            location->current = 0U;
            if (location->owns_lease) {
                shadowspill_memory_pool_try_recycle_lease_record_locked(
                    lease
                );
            }
            location->owns_lease = 0U;
        }
        shadowspill_memory_pool_unlock_foreground(pool);
    }

    if (status == SHADOWSPILL_STATUS_OK) {
        object->residency = SHADOWSPILL_OBJECT_RELEASED;
        readiness_event = object->readiness_event;
        object->readiness_event = NULL;
        object->has_readiness_event = 0U;
    }
    pthread_mutex_unlock(&object->lock);
    pthread_mutex_unlock(&runtime->mutex);

    if (status == SHADOWSPILL_STATUS_OK && readiness_event != NULL &&
        shadowspill_event_lease_release(runtime, readiness_event) != 0) {
        status = SHADOWSPILL_STATUS_BACKEND_FAILURE;
    }
    return status;
}

ShadowSpillStatus shadowspill_object_owner_retain(
    ShadowSpillObject *object
) {
    if (object == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    uint32_t owners = atomic_load_explicit(
        &object->owners, memory_order_acquire
    );
    while (owners != 0U && owners != UINT32_MAX) {
        if (atomic_compare_exchange_weak_explicit(
                &object->owners,
                &owners,
                owners + 1U,
                memory_order_acq_rel,
                memory_order_acquire
            )) {
            shadowspill_object_retain(object);
            return SHADOWSPILL_STATUS_OK;
        }
    }
    return SHADOWSPILL_STATUS_INVALID_STATE;
}

ShadowSpillStatus shadowspill_object_owner_release(
    ShadowSpillObject *object
) {
    if (object == NULL) {
        return SHADOWSPILL_STATUS_OK;
    }
    uint32_t owners = atomic_load_explicit(
        &object->owners, memory_order_acquire
    );
    while (owners != 0U && !atomic_compare_exchange_weak_explicit(
               &object->owners,
               &owners,
               owners - 1U,
               memory_order_acq_rel,
               memory_order_acquire
           )) {
    }
    if (owners == 0U) {
        return SHADOWSPILL_STATUS_INVALID_STATE;
    }
    ShadowSpillStatus status = SHADOWSPILL_STATUS_OK;
    if (owners == 1U) {
        ShadowSpillRuntime *runtime = object->runtime;
        if (atomic_load_explicit(
                &object->detached, memory_order_acquire
            ) == 0U) {
            if (shadowspill_object_table_remove(
                    &runtime->objects, object
                ) != 0) {
                status = SHADOWSPILL_STATUS_INVALID_STATE;
            } else {
                (void)atomic_fetch_sub_explicit(
                    &runtime->registered_objects, 1U, memory_order_acq_rel
                );
            }
        }
        if (status == SHADOWSPILL_STATUS_OK) {
            status = release_object_residency(object, 0U, 0U);
        }
    }
    if (status != SHADOWSPILL_STATUS_OK) {
        shadowspill_latch_failure_locked(
            object->runtime,
            status,
            object->object_id,
            object->allocation_id,
            object->size_bytes
        );
    }
    shadowspill_object_release(object);
    return status;
}

ShadowSpillStatus shadowspill_object_handle_release(
    ShadowSpillObjectHandle *handle
) {
    if (handle == NULL) {
        return SHADOWSPILL_STATUS_OK;
    }
    const ShadowSpillStatus status = shadowspill_object_owner_release(
        handle->object
    );
    handle->object = NULL;
    handle->runtime = NULL;
    free(handle);
    return status;
}

ShadowSpillStatus shadowspill_object_release_generation(
    const ShadowSpillObjectHandle *handle,
    uint64_t expected_generation
) {
    if (handle == NULL || handle->runtime == NULL || handle->object == NULL ||
        handle->object->runtime != handle->runtime ||
        atomic_load_explicit(
            &handle->object->detached, memory_order_acquire
        ) != 0U) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    return release_object_residency(
        handle->object, expected_generation, 1U
    );
}

ShadowSpillStatus shadowspill_register_object(
    ShadowSpillRuntime *runtime,
    const ShadowSpillObjectDescription *description
) {
    if (runtime == NULL || description == NULL ||
        description->object_id == SHADOWSPILL_RUNTIME_NO_ID ||
        description->retain_spill_copy > 1U ||
        description->initially_resident > 1U) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&runtime->mutex);
    ShadowSpillStatus status = shadowspill_current_status_locked(runtime);
    if (status != SHADOWSPILL_STATUS_OK) {
        goto done;
    }
    if (shadowspill_find_object(runtime, description->object_id) != NULL) {
        status = SHADOWSPILL_STATUS_INVALID_STATE;
        goto done;
    }
    ShadowSpillObject *created = calloc(1U, sizeof(*created));
    if (created == NULL) {
        status = SHADOWSPILL_STATUS_INTERNAL_FAILURE;
        goto done;
    }
    created->locations = calloc(
        runtime->pool_count, sizeof(*created->locations)
    );
    if (created->locations == NULL) {
        free(created);
        status = SHADOWSPILL_STATUS_INTERNAL_FAILURE;
        goto done;
    }
    created->location_count = runtime->pool_count;
    created->runtime = runtime;
    atomic_init(&created->references, 1U);
    atomic_init(&created->owners, 1U);
    atomic_init(&created->registration_owned, 1U);
    atomic_init(&created->detached, 0U);
    atomic_init(&created->unpublished_fetch_count, 0U);
    if (pthread_mutex_init(&created->lock, NULL) != 0) {
        free(created->locations);
        free(created);
        status = SHADOWSPILL_STATUS_INTERNAL_FAILURE;
        goto done;
    }
    created->object_id = description->object_id;
    created->size_bytes = description->size_bytes;
    created->authoritative_version = description->initial_version;
    created->retain_spill_copy = description->retain_spill_copy;
    created->allocation_id = SHADOWSPILL_RUNTIME_NO_ID;
    created->residency = description->initially_resident
        ? SHADOWSPILL_OBJECT_SPILL_ONLY
        : SHADOWSPILL_OBJECT_RELEASED;
    ShadowSpillMemoryPool *initial_pool = description->initially_resident
        ? shadowspill_runtime_pool(runtime, description->initial_pool_id)
        : NULL;
    if (description->initially_resident && initial_pool == NULL) {
        pthread_mutex_destroy(&created->lock);
        free(created->locations);
        free(created);
        status = SHADOWSPILL_STATUS_INVALID_ARGUMENT;
        goto done;
    }
    if (initial_pool != NULL) {
        shadowspill_memory_pool_lock_foreground(initial_pool);
        ShadowSpillMemoryLease *initial_lease =
            shadowspill_memory_pool_acquire_lease_record_locked(
                runtime,
                initial_pool,
                SHADOWSPILL_RUNTIME_NO_ID
            );
        const int reserve_status = initial_lease == NULL
            ? -1
            : shadowspill_memory_pool_reserve_lease_locked(
                  initial_pool,
                  initial_lease,
                  description->size_bytes,
                  1U,
                  SHADOWSPILL_MEMORY_FIRST_FIT
              );
        if (reserve_status != 0 && initial_lease != NULL) {
            shadowspill_memory_pool_try_recycle_lease_record_locked(
                initial_lease
            );
        }
        shadowspill_memory_pool_unlock_foreground(initial_pool);
        if (reserve_status != 0) {
            pthread_mutex_destroy(&created->lock);
            free(created->locations);
            free(created);
            status = SHADOWSPILL_STATUS_OUT_OF_MEMORY;
            goto done;
        }
        ShadowSpillObjectLocation *initial = shadowspill_object_location(
            created, description->initial_pool_id
        );
        initial->lease = initial_lease;
        initial->owns_lease = 1U;
        initial->lease->state = SHADOWSPILL_LEASE_IN_USE;
        initial->current = 1U;
        initial->version = description->initial_version;
    }
    if (shadowspill_object_table_insert(&runtime->objects, created) != 0) {
        ShadowSpillObjectLocation *initial = initial_pool == NULL
            ? NULL : shadowspill_object_location(
                created, description->initial_pool_id
            );
        if (initial != NULL && initial->lease != NULL) {
            shadowspill_memory_pool_lock_foreground(initial_pool);
            (void)shadowspill_memory_pool_release_lease_locked(
                initial->lease
            );
            shadowspill_memory_pool_try_recycle_lease_record_locked(
                initial->lease
            );
            shadowspill_memory_pool_unlock_foreground(initial_pool);
            initial->lease = NULL;
            initial->owns_lease = 0U;
        }
        pthread_mutex_destroy(&created->lock);
        free(created->locations);
        free(created);
        status = SHADOWSPILL_STATUS_INVALID_STATE;
        goto done;
    }
    ++runtime->registered_objects;

done:
    pthread_mutex_unlock(&runtime->mutex);
    return status;
}

ShadowSpillStatus shadowspill_unregister_object(
    ShadowSpillRuntime *runtime,
    uint64_t object_id
) {
    if (runtime == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&runtime->mutex);
    ShadowSpillStatus status = shadowspill_current_status_locked(runtime);
    ShadowSpillObject *object = shadowspill_object_table_acquire(
        &runtime->objects, object_id
    );
    if (status != SHADOWSPILL_STATUS_OK) {
        goto done;
    }
    if (object == NULL) {
        status = SHADOWSPILL_STATUS_INVALID_STATE;
        goto done;
    }
    pthread_mutex_lock(&runtime->actions.lock);
    for (ShadowSpillQueuedAction *action = runtime->actions.head;
         action != NULL; action = action->next) {
        if (action->object == object) {
            status = SHADOWSPILL_STATUS_INVALID_STATE;
            pthread_mutex_unlock(&runtime->actions.lock);
            goto done;
        }
    }
    pthread_mutex_unlock(&runtime->actions.lock);
    uint8_t expected_registration = 1U;
    if (!atomic_compare_exchange_strong_explicit(
            &object->registration_owned,
            &expected_registration,
            0U,
            memory_order_acq_rel,
            memory_order_acquire
        )) {
        status = SHADOWSPILL_STATUS_INVALID_STATE;
        goto done;
    }

done:
    pthread_mutex_unlock(&runtime->mutex);
    if (status == SHADOWSPILL_STATUS_OK) {
        status = shadowspill_object_owner_release(object);
    }
    shadowspill_object_release(object);
    return status;
}

ShadowSpillStatus shadowspill_rekey_object(
    ShadowSpillRuntime *runtime,
    uint64_t object_id,
    uint64_t replacement_object_id
) {
    if (runtime == NULL || object_id == SHADOWSPILL_RUNTIME_NO_ID ||
        replacement_object_id == SHADOWSPILL_RUNTIME_NO_ID ||
        object_id == replacement_object_id) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&runtime->mutex);
    ShadowSpillStatus status = shadowspill_current_status_locked(runtime);
    ShadowSpillObject *object = shadowspill_find_object(runtime, object_id);
    if (status != SHADOWSPILL_STATUS_OK) {
        goto done;
    }
    if (object == NULL ||
        shadowspill_find_object(runtime, replacement_object_id) != NULL ||
        object->allocation_id != SHADOWSPILL_RUNTIME_NO_ID ||
        object->has_readiness_event ||
        (object->residency != SHADOWSPILL_OBJECT_SPILL_ONLY &&
         object->residency != SHADOWSPILL_OBJECT_RELEASED)) {
        status = SHADOWSPILL_STATUS_INVALID_STATE;
        goto done;
    }
    pthread_mutex_lock(&runtime->actions.lock);
    for (ShadowSpillQueuedAction *action = runtime->actions.head;
         action != NULL; action = action->next) {
        if (action->object == object) {
            status = SHADOWSPILL_STATUS_INVALID_STATE;
            break;
        }
    }
    pthread_mutex_unlock(&runtime->actions.lock);
    if (status != SHADOWSPILL_STATUS_OK) {
        goto done;
    }
    if (shadowspill_object_table_rekey(
            &runtime->objects, object, replacement_object_id
        ) != 0) {
        status = SHADOWSPILL_STATUS_INVALID_STATE;
    }

done:
    pthread_mutex_unlock(&runtime->mutex);
    return status;
}

ShadowSpillStatus shadowspill_write_object(
    ShadowSpillRuntime *runtime,
    uint64_t object_id,
    uint32_t pool_id,
    const void *source,
    uint64_t bytes
) {
    if (shadowspill_runtime_pool(runtime, pool_id) == NULL || bytes > SIZE_MAX ||
        (bytes != 0U && source == NULL)) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&runtime->mutex);
    ShadowSpillStatus status = shadowspill_current_status_locked(runtime);
    ShadowSpillObject *object = shadowspill_find_object(runtime, object_id);
    if (status != SHADOWSPILL_STATUS_OK) {
        goto done;
    }
    if (object == NULL || bytes != object->size_bytes ||
        shadowspill_object_location(object, pool_id)->lease == NULL ||
        object->residency != SHADOWSPILL_OBJECT_SPILL_ONLY ||
        object->allocation_id != SHADOWSPILL_RUNTIME_NO_ID) {
        status = SHADOWSPILL_STATUS_INVALID_STATE;
        goto done;
    }
    if (bytes != 0U) {
        ShadowSpillObjectLocation *location = shadowspill_object_location(
            object, pool_id
        );
        memcpy(
            location->lease->pointer,
            source,
            (size_t)bytes
        );
    }
    ShadowSpillObjectLocation *location = shadowspill_object_location(
        object, pool_id
    );
    location->current = 1U;
    location->version = object->authoritative_version;

done:
    pthread_mutex_unlock(&runtime->mutex);
    return status;
}

ShadowSpillStatus shadowspill_read_object(
    ShadowSpillRuntime *runtime,
    uint64_t object_id,
    uint32_t pool_id,
    void *destination,
    uint64_t bytes
) {
    if (shadowspill_runtime_pool(runtime, pool_id) == NULL || bytes > SIZE_MAX ||
        (bytes != 0U && destination == NULL)) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&runtime->mutex);
    ShadowSpillStatus status = shadowspill_current_status_locked(runtime);
    ShadowSpillObject *object = shadowspill_find_object(runtime, object_id);
    ShadowSpillObjectLocation *location = object == NULL
        ? NULL : shadowspill_object_location(object, pool_id);
    if (status != SHADOWSPILL_STATUS_OK) {
        goto read_done;
    }
    if (object == NULL || bytes != object->size_bytes ||
        location->lease == NULL || !location->current ||
        location->version != object->authoritative_version ||
        object->residency != SHADOWSPILL_OBJECT_SPILL_ONLY) {
        status = SHADOWSPILL_STATUS_INVALID_STATE;
        goto read_done;
    }
    if (bytes != 0U) {
        memcpy(
            destination,
            location->lease->pointer,
            (size_t)bytes
        );
    }
read_done:
    pthread_mutex_unlock(&runtime->mutex);
    return status;
}

ShadowSpillStatus shadowspill_object_bind_allocation(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryPool *pool,
    ShadowSpillObject *object,
    const void *pointer,
    const ShadowSpillTaskRecord *task,
    ShadowSpillObjectBinding *binding
) {
    if (runtime == NULL || pool == NULL || object == NULL || pointer == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    ShadowSpillStatus status = shadowspill_failure_status(runtime);
    if (status != SHADOWSPILL_STATUS_OK) {
        return status;
    }

    /* Snapshot the directly retained prior owner before taking object locks. */
    shadowspill_memory_pool_lock_foreground(pool);
    ShadowSpillMemoryLease *allocation =
        shadowspill_find_execution_lease_by_pointer(pool, pointer);
    ShadowSpillObject *previous_owner = allocation == NULL
        ? NULL : allocation->bound_object;
    shadowspill_memory_pool_unlock_foreground(pool);

    ShadowSpillObject *first = object;
    ShadowSpillObject *second = previous_owner;
    if (second != NULL && (uintptr_t)second < (uintptr_t)first) {
        first = previous_owner;
        second = object;
    }
    pthread_mutex_lock(&first->lock);
    if (second != NULL && second != first) {
        pthread_mutex_lock(&second->lock);
    }
    shadowspill_memory_pool_lock_foreground(pool);
    allocation = shadowspill_find_execution_lease_by_pointer(pool, pointer);
    ShadowSpillObjectLocation *location = shadowspill_object_location(
        object, pool->pool_id
    );
    status = shadowspill_failure_status(runtime);
    if (status != SHADOWSPILL_STATUS_OK) {
        goto done;
    }
    if (location == NULL || allocation == NULL || allocation->logical_freed ||
        allocation->pointer == NULL || object->allocation_id !=
            SHADOWSPILL_RUNTIME_NO_ID ||
        allocation->requested_bytes < object->size_bytes ||
        allocation->bound_object != previous_owner) {
        status = SHADOWSPILL_STATUS_INVALID_STATE;
        goto done;
    }
    ShadowSpillQueuedAction *handoff_action = previous_owner == NULL
        ? NULL : shadowspill_task_release_action(task, previous_owner);
    const uint64_t task_id = shadowspill_current_task_id(runtime);
    if (previous_owner != NULL &&
        (previous_owner == object ||
         previous_owner->allocation_id != allocation->allocation_id ||
         task == NULL || task->task_id != task_id ||
         handoff_action == NULL || handoff_action->active ||
         handoff_action->handoff_lease != NULL ||
         (previous_owner->residency != SHADOWSPILL_OBJECT_EXECUTION_READY &&
          previous_owner->residency != SHADOWSPILL_OBJECT_PREFETCHING))) {
        status = SHADOWSPILL_STATUS_PLAN_VIOLATION;
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
    if (shadowspill_failure_status(runtime) != SHADOWSPILL_STATUS_OK) {
        status = shadowspill_failure_status(runtime);
        goto done;
    }
    if (previous_owner != NULL) {
        handoff_action->handoff_lease = allocation;
        handoff_action->handoff_generation = allocation->generation;
    }
    object->allocation_id = allocation->allocation_id;
    location->lease = allocation;
    location->current = 1U;
    allocation->bound_object = object;
    object->generation = allocation->generation;
    location->version = object->authoritative_version;
    object->residency = SHADOWSPILL_OBJECT_EXECUTION_READY;
    if (binding != NULL) {
        *binding = (ShadowSpillObjectBinding){
            .object_id = object->object_id,
            .generation = object->generation,
            .allocation_id = object->allocation_id,
            .authoritative_version = object->authoritative_version,
            .pointer = allocation->pointer,
        };
    }

done:
    shadowspill_memory_pool_unlock_foreground(pool);
    if (second != NULL && second != first) {
        pthread_mutex_unlock(&second->lock);
    }
    pthread_mutex_unlock(&first->lock);
    return status;
}

ShadowSpillStatus shadowspill_plan_publish_initial_allocation(
    ShadowSpillPlan *plan,
    uint64_t plan_object_id,
    const void *pointer,
    ShadowSpillObjectBinding *binding
) {
    if (plan == NULL || pointer == NULL || binding == NULL ||
        plan->runtime == NULL || plan->execution_pool == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    ShadowSpillObject *object = shadowspill_plan_object_acquire(
        plan, plan_object_id, NULL
    );
    if (object == NULL) {
        return SHADOWSPILL_STATUS_INVALID_STATE;
    }
    const ShadowSpillStatus status = shadowspill_object_bind_allocation(
        plan->runtime,
        plan->execution_pool,
        object,
        pointer,
        NULL,
        binding
    );
    shadowspill_object_release(object);
    return status;
}

ShadowSpillStatus shadowspill_object_replace_allocation(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryPool *pool,
    ShadowSpillObject *object,
    const void *pointer,
    ShadowSpillObjectBinding *binding
) {
    if (runtime == NULL || pool == NULL || object == NULL || pointer == NULL ||
        binding == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    const uint64_t task_id = shadowspill_current_task_id(runtime);
    if (task_id == SHADOWSPILL_RUNTIME_NO_ID) {
        return SHADOWSPILL_STATUS_INVALID_STATE;
    }

    ShadowSpillEventLease *retired_readiness = NULL;
    ShadowSpillStatus status = shadowspill_current_status_locked(runtime);
    pthread_mutex_lock(&object->lock);
    shadowspill_memory_pool_lock_foreground(pool);
    ShadowSpillMemoryLease *replacement =
        shadowspill_find_execution_lease_by_pointer(pool, pointer);
    ShadowSpillObjectLocation *location = shadowspill_object_location(
        object, pool->pool_id
    );
    ShadowSpillMemoryLease *prior = location == NULL ? NULL : location->lease;
    if (status != SHADOWSPILL_STATUS_OK) {
        goto done;
    }
    if ((object->residency != SHADOWSPILL_OBJECT_EXECUTION_READY &&
         object->residency != SHADOWSPILL_OBJECT_PREFETCHING) ||
        prior == NULL || prior->pointer == NULL || prior->logical_freed ||
        prior->allocation_id != object->allocation_id ||
        prior->generation != object->generation || replacement == NULL ||
        replacement == prior || replacement->pointer == NULL ||
        replacement->logical_freed || replacement->plan_owned ||
        replacement->bound_object != NULL ||
        replacement->requested_bytes < object->size_bytes) {
        status = SHADOWSPILL_STATUS_INVALID_STATE;
        goto done;
    }

    replacement->plan_owned = 1;
    replacement->ever_plan_owned = 1;
    replacement->bound_object = object;
    replacement->state = SHADOWSPILL_LEASE_IN_USE;
    shadowspill_append_allocation_event_locked(
        runtime,
        replacement,
        SHADOWSPILL_ALLOCATION_PROMOTED,
        SHADOWSPILL_ALLOCATION_PLANNED_OBJECT
    );
    if (shadowspill_failure_status(runtime) != SHADOWSPILL_STATUS_OK) {
        status = shadowspill_failure_status(runtime);
        replacement->plan_owned = 0;
        replacement->ever_plan_owned = 0;
        replacement->bound_object = NULL;
        goto done;
    }

    object->retired_generation = object->generation;
    object->retired_execution_pointer = prior->pointer;
    prior->bound_object = NULL;
    prior->release_task_id = task_id;
    prior->logical_freed = 1;
    if (shadowspill_memory_pool_begin_retirement_locked(
            prior, NULL, 1
        ) != 0) {
        status = SHADOWSPILL_STATUS_INVALID_STATE;
        goto done;
    }
    if (shadowspill_track_task_retirement(runtime, prior) != 0) {
        status = SHADOWSPILL_STATUS_INVALID_STATE;
        goto done;
    }
    shadowspill_append_allocation_event_locked(
        runtime,
        prior,
        SHADOWSPILL_ALLOCATION_LOGICAL_FREED,
        SHADOWSPILL_ALLOCATION_PLANNED_OBJECT
    );
    (void)atomic_fetch_add_explicit(
        &runtime->pending_retirements, 1U, memory_order_acq_rel
    );
    (void)atomic_fetch_add_explicit(
        &prior->pool->pending_retirements, 1U, memory_order_acq_rel
    );

    location->lease = replacement;
    location->version = object->authoritative_version;
    location->current = 1U;
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
done:
    shadowspill_memory_pool_unlock_foreground(pool);
    pthread_mutex_unlock(&object->lock);
    if (retired_readiness != NULL &&
        shadowspill_event_lease_release(runtime, retired_readiness) != 0 &&
        status == SHADOWSPILL_STATUS_OK) {
        status = SHADOWSPILL_STATUS_BACKEND_FAILURE;
    }
    return status;
}

ShadowSpillStatus shadowspill_task_publish_allocation(
    ShadowSpillRuntime *runtime,
    const ShadowSpillTaskHandle *handle,
    uint32_t publication_ordinal,
    const void *pointer,
    ShadowSpillObjectBinding *binding
) {
    const ShadowSpillTaskRecord *record = handle;
    if (runtime == NULL || record == NULL || pointer == NULL || binding == NULL ||
        record->plan_owner == NULL || record->plan_owner->runtime != runtime ||
        record->boundary_kind != SHADOWSPILL_BOUNDARY_TASK ||
        publication_ordinal >= record->publication_count) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    if (shadowspill_current_plan(runtime) != record->plan_owner ||
        shadowspill_current_task_id(runtime) != record->task_id) {
        return SHADOWSPILL_STATUS_INVALID_STATE;
    }
    const ShadowSpillTaskPublication *publication =
        &record->publications[publication_ordinal];
    if (publication->kind == SHADOWSPILL_TASK_PUBLICATION_REPLACE) {
        return shadowspill_object_replace_allocation(
            runtime,
            record->plan_owner->execution_pool,
            publication->object,
            pointer,
            binding
        );
    }
    return shadowspill_object_bind_allocation(
        runtime,
        record->plan_owner->execution_pool,
        publication->object,
        pointer,
        record,
        binding
    );
}

ShadowSpillStatus shadowspill_task_validate_replacement_binding(
    ShadowSpillRuntime *runtime,
    const ShadowSpillTaskHandle *handle,
    uint32_t publication_ordinal,
    const void *retired_pointer,
    const void *successor_pointer
) {
    const ShadowSpillTaskRecord *record = handle;
    if (runtime == NULL || record == NULL || retired_pointer == NULL ||
        successor_pointer == NULL ||
        record->plan_owner == NULL || record->plan_owner->runtime != runtime ||
        record->boundary_kind != SHADOWSPILL_BOUNDARY_TASK ||
        publication_ordinal >= record->publication_count ||
        record->publications[publication_ordinal].kind !=
            SHADOWSPILL_TASK_PUBLICATION_REPLACE) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    if (shadowspill_current_plan(runtime) != record->plan_owner ||
        shadowspill_current_task_id(runtime) != record->task_id) {
        return SHADOWSPILL_STATUS_INVALID_STATE;
    }
    ShadowSpillObject *object = record->publications[publication_ordinal].object;
    pthread_mutex_lock(&object->lock);
    const ShadowSpillObjectLocation *location = shadowspill_object_location(
        object, record->plan_owner->execution_pool->pool_id
    );
    const int matches = location != NULL && location->lease != NULL &&
        location->lease->pointer == successor_pointer &&
        location->lease->generation == object->generation &&
        object->retired_execution_pointer == retired_pointer;
    pthread_mutex_unlock(&object->lock);
    return matches
        ? SHADOWSPILL_STATUS_OK
        : SHADOWSPILL_STATUS_INVALID_STATE;
}

ShadowSpillStatus shadowspill_object_transfer_to_caller(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryPool *execution_pool,
    ShadowSpillMemoryPool *spill_pool,
    ShadowSpillObject *object,
    ShadowSpillBackendStream consumer_stream,
    const void *required_pointer,
    uint64_t required_generation,
    uint64_t required_allocation_id,
    uint8_t validate_expected,
    ShadowSpillAllocation *allocation
) {
    if (runtime == NULL || execution_pool == NULL || spill_pool == NULL ||
        object == NULL || allocation == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    uint64_t expected_allocation_id = SHADOWSPILL_RUNTIME_NO_ID;
    uint64_t expected_generation = 0U;

    /*
     * Acquisition has already inserted a readiness-event wait into the
     * consumer stream.  Snapshot the exact generation without waiting for a
     * final fetch to complete on the dispatcher.
     */
    ShadowSpillStatus status = shadowspill_current_status_locked(runtime);
    pthread_mutex_lock(&object->lock);
    const ShadowSpillObjectLocation *initial_execution =
        shadowspill_object_location(object, execution_pool->pool_id);
    const int execution_available =
        object->residency == SHADOWSPILL_OBJECT_EXECUTION_READY ||
        (object->residency == SHADOWSPILL_OBJECT_PREFETCHING &&
         object->has_readiness_event);
    if (status == SHADOWSPILL_STATUS_OK &&
        (!execution_available || initial_execution == NULL ||
         initial_execution->lease == NULL ||
         object->allocation_id == SHADOWSPILL_RUNTIME_NO_ID)) {
        status = SHADOWSPILL_STATUS_INVALID_STATE;
    }
    if (status == SHADOWSPILL_STATUS_OK && validate_expected &&
        (initial_execution->lease->pointer != required_pointer ||
         object->generation != required_generation ||
         object->allocation_id != required_allocation_id)) {
        status = SHADOWSPILL_STATUS_INVALID_STATE;
    }
    if (status == SHADOWSPILL_STATUS_OK) {
        expected_allocation_id = object->allocation_id;
        expected_generation = object->generation;
    }
    pthread_mutex_unlock(&object->lock);
    if (status != SHADOWSPILL_STATUS_OK) {
        return status;
    }

    status = shadowspill_memory_pool_record_stream(
        runtime,
        execution_pool->pool_id,
        expected_allocation_id,
        consumer_stream
    );
    if (status != SHADOWSPILL_STATUS_OK) {
        return status;
    }

    /* Commit ownership only if the snapshotted generation is still current. */
    status = shadowspill_current_status_locked(runtime);
    pthread_mutex_lock(&object->lock);
    ShadowSpillObjectLocation *execution = shadowspill_object_location(
        object, execution_pool->pool_id
    );
    ShadowSpillQueuedAction *settling_fetch = NULL;
    if (object->action_head != NULL &&
        object->action_head == object->action_tail &&
        object->action_head->kind == SHADOWSPILL_RUNTIME_PREFETCH &&
        object->action_head->state == SHADOWSPILL_ACTION_IN_FLIGHT &&
        ((object->residency == SHADOWSPILL_OBJECT_PREFETCHING &&
          object->has_readiness_event &&
          object->action_head->completion_event == object->readiness_event) ||
         (object->residency == SHADOWSPILL_OBJECT_EXECUTION_READY &&
          !object->has_readiness_event))) {
        settling_fetch = object->action_head;
    }
    const int ready_without_actions =
        object->residency == SHADOWSPILL_OBJECT_EXECUTION_READY &&
        object->action_head == NULL && !object->has_readiness_event;
    if (status != SHADOWSPILL_STATUS_OK ||
        atomic_load_explicit(&object->detached, memory_order_acquire) != 0U ||
        object->allocation_id != expected_allocation_id ||
        object->generation != expected_generation ||
        execution == NULL || execution->lease == NULL ||
        (!ready_without_actions && settling_fetch == NULL)) {
        status = status == SHADOWSPILL_STATUS_OK
            ? SHADOWSPILL_STATUS_INVALID_STATE
            : status;
        goto done_object;
    }

    shadowspill_memory_pool_lock_foreground(execution_pool);
    ShadowSpillMemoryLease *record = shadowspill_find_execution_lease(
        execution_pool, object->allocation_id
    );
    if (record == NULL || record->pointer == NULL || record->logical_freed ||
        !record->plan_owned) {
        status = SHADOWSPILL_STATUS_INVALID_STATE;
        goto done_allocation;
    }
    if (object->residency == SHADOWSPILL_OBJECT_EXECUTION_READY &&
        shadowspill_object_location(object, spill_pool->pool_id)->lease != NULL) {
        ShadowSpillObjectLocation *spill = shadowspill_object_location(
            object, spill_pool->pool_id
        );
        shadowspill_memory_pool_lock_foreground(spill_pool);
        const int release_status = shadowspill_memory_pool_release_lease_locked(
            spill->lease
        );
        if (release_status == 0 && spill->owns_lease) {
            shadowspill_memory_pool_try_recycle_lease_record_locked(
                spill->lease
            );
        }
        shadowspill_memory_pool_unlock_foreground(spill_pool);
        if (release_status != 0) {
            status = SHADOWSPILL_STATUS_INTERNAL_FAILURE;
            goto done_allocation;
        }
        spill->lease = NULL;
        spill->owns_lease = 0U;
        spill->current = 0U;
    }
    if (settling_fetch != NULL) {
        if (settling_fetch->caller_handoff_lease != NULL) {
            status = SHADOWSPILL_STATUS_INVALID_STATE;
            goto done_allocation;
        }
    }
    record->framework_free_seen = 0;
    record->plan_owned = 0;
    record->bound_object = NULL;
    if (settling_fetch != NULL) {
        shadowspill_memory_lease_retain(record);
        settling_fetch->caller_handoff_lease = record;
        settling_fetch->caller_handoff_generation = record->generation;
    }
    object->retired_generation = object->generation;
    object->retired_execution_pointer = record->pointer;
    object->allocation_id = SHADOWSPILL_RUNTIME_NO_ID;
    execution->lease = NULL;
    execution->current = 0U;
    object->residency = SHADOWSPILL_OBJECT_RELEASED;
    shadowspill_append_allocation_event_locked(
        runtime,
        record,
        SHADOWSPILL_ALLOCATION_PROMOTED,
        SHADOWSPILL_ALLOCATION_CALLER_OWNED
    );
    *allocation = (ShadowSpillAllocation){
        .pool_id = execution_pool->pool_id,
        .allocation_id = record->allocation_id,
        .generation = record->generation,
        .requested_bytes = record->requested_bytes,
        .charged_bytes = record->charged_bytes,
        .pointer = record->pointer,
    };

done_allocation:
    shadowspill_memory_pool_unlock_foreground(execution_pool);
done_object:
    pthread_mutex_unlock(&object->lock);
    return status;
}


ShadowSpillStatus shadowspill_object_snapshot(
    ShadowSpillRuntime *runtime,
    uint64_t object_id,
    ShadowSpillObjectSnapshot *snapshot
) {
    if (runtime == NULL || snapshot == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    ShadowSpillObject *object = shadowspill_object_table_acquire(
        &runtime->objects, object_id
    );
    if (object == NULL) {
        return SHADOWSPILL_STATUS_INVALID_STATE;
    }
    pthread_mutex_lock(&object->lock);
    const ShadowSpillObjectLocation *execution = shadowspill_execution_location(
        runtime, object
    );
    const ShadowSpillObjectLocation *spill = shadowspill_spill_location(
        runtime, object
    );
    *snapshot = (ShadowSpillObjectSnapshot){
        .object_id = object->object_id,
        .size_bytes = object->size_bytes,
        .generation = object->generation,
        .allocation_id = object->allocation_id,
        .authoritative_version = object->authoritative_version,
        .execution_version = execution == NULL ? 0U : execution->version,
        .spill_version = spill == NULL ? 0U : spill->version,
        .residency = object->residency,
        .spill_current = spill == NULL ? 0U : spill->current,
        .has_spill_lease = spill != NULL && spill->lease != NULL,
        .execution_pointer = execution == NULL || execution->lease == NULL
            ? NULL : execution->lease->pointer,
        .spill_pointer = spill == NULL || spill->lease == NULL
            ? NULL : spill->lease->pointer,
        .retired_generation = object->retired_generation,
        .retired_execution_pointer = object->retired_execution_pointer,
    };
    pthread_mutex_unlock(&object->lock);
    shadowspill_object_release(object);
    return SHADOWSPILL_STATUS_OK;
}

ShadowSpillStatus shadowspill_object_location_snapshot(
    ShadowSpillRuntime *runtime,
    uint64_t object_id,
    uint32_t pool_id,
    ShadowSpillObjectLocationSnapshot *snapshot
) {
    if (runtime == NULL || snapshot == NULL ||
        shadowspill_runtime_pool(runtime, pool_id) == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    ShadowSpillObject *object = shadowspill_object_table_acquire(
        &runtime->objects, object_id
    );
    if (object == NULL) {
        return SHADOWSPILL_STATUS_INVALID_STATE;
    }
    pthread_mutex_lock(&object->lock);
    const ShadowSpillObjectLocation *location = shadowspill_object_location(
        object, pool_id
    );
    const ShadowSpillMemoryLease *lease = location == NULL
        ? NULL : location->lease;
    *snapshot = (ShadowSpillObjectLocationSnapshot){
        .object_id = object->object_id,
        .size_bytes = object->size_bytes,
        .authoritative_version = object->authoritative_version,
        .version = location == NULL ? 0U : location->version,
        .allocation_id = lease == NULL ? 0U : lease->allocation_id,
        .generation = lease == NULL ? 0U : lease->generation,
        .pool_id = pool_id,
        .current = location == NULL ? 0U : location->current,
        .has_lease = lease != NULL,
        .pointer = lease == NULL ? NULL : lease->pointer,
    };
    pthread_mutex_unlock(&object->lock);
    shadowspill_object_release(object);
    return SHADOWSPILL_STATUS_OK;
}
