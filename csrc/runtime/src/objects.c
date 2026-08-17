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

ShadowSpillRuntimeStatus shadowspill_object_handle_acquire(
    ShadowSpillRuntime *runtime,
    uint64_t runtime_object_id,
    ShadowSpillObjectHandle **output
) {
    if (runtime == NULL || output == NULL ||
        runtime_object_id == SHADOWSPILL_RUNTIME_NO_ID) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    *output = NULL;
    if (atomic_load_explicit(&runtime->closing, memory_order_acquire) != 0U) {
        return SHADOWSPILL_RUNTIME_CLOSED;
    }
    ShadowSpillObject *object = shadowspill_object_table_acquire(
        &runtime->objects, runtime_object_id
    );
    if (object == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    ShadowSpillObjectHandle *handle = malloc(sizeof(*handle));
    if (handle == NULL) {
        shadowspill_object_release(object);
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    const ShadowSpillRuntimeStatus retain_status =
        shadowspill_object_owner_retain(object);
    shadowspill_object_release(object);
    if (retain_status != SHADOWSPILL_RUNTIME_OK) {
        free(handle);
        return retain_status;
    }
    *handle = (ShadowSpillObjectHandle){
        .runtime = runtime,
        .object = object,
    };
    *output = handle;
    return SHADOWSPILL_RUNTIME_OK;
}

static ShadowSpillRuntimeStatus release_object_residency(
    ShadowSpillObject *object,
    uint64_t expected_generation,
    uint8_t validate_generation
) {
    ShadowSpillRuntime *runtime = object->runtime;
    if (runtime == NULL || atomic_load_explicit(
            &runtime->closing, memory_order_acquire
        ) != 0U) {
        return SHADOWSPILL_RUNTIME_OK;
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
        return SHADOWSPILL_RUNTIME_INVALID_STATE;
    }

    ShadowSpillRuntimeStatus status = SHADOWSPILL_RUNTIME_OK;
    ShadowSpillObjectLocation *execution = shadowspill_execution_location(
        runtime, object
    );
    ShadowSpillMemoryLease *execution_lease = execution->lease;
    if (execution_lease != NULL) {
        shadowspill_memory_pool_lock_foreground(
            shadowspill_execution_pool(runtime)
        );
        if (execution_lease->logical_freed ||
            execution_lease->bound_object_id != object->object_id ||
            object->allocation_id != execution_lease->allocation_id) {
            status = SHADOWSPILL_RUNTIME_INVALID_STATE;
        } else {
            execution_lease->bound_object_id = SHADOWSPILL_RUNTIME_NO_ID;
            execution_lease->plan_owned = 0;
            object->retired_generation = object->generation;
            object->retired_execution_pointer = execution_lease->pointer;
            object->allocation_id = SHADOWSPILL_RUNTIME_NO_ID;
            execution->lease = NULL;
            execution->version = 0U;
            execution->current = 0U;
            shadowspill_release_execution_lease_locked(
                runtime, execution_lease
            );
            status = shadowspill_failure_status(runtime);
        }
        shadowspill_memory_pool_unlock_foreground(
            shadowspill_execution_pool(runtime)
        );
    }

    ShadowSpillObjectLocation *spill = shadowspill_spill_location(
        runtime, object
    );
    if (status == SHADOWSPILL_RUNTIME_OK && spill->lease != NULL) {
        ShadowSpillMemoryLease *spill_lease = spill->lease;
        shadowspill_memory_pool_lock_foreground(shadowspill_spill_pool(runtime));
        if (shadowspill_memory_pool_release_lease_locked(spill_lease) != 0) {
            status = SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
        } else {
            spill->lease = NULL;
            spill->version = 0U;
            spill->current = 0U;
            if (spill->owns_lease) {
                free(spill_lease);
            }
            spill->owns_lease = 0U;
        }
        shadowspill_memory_pool_unlock_foreground(
            shadowspill_spill_pool(runtime)
        );
    }

    if (status == SHADOWSPILL_RUNTIME_OK) {
        object->residency = SHADOWSPILL_OBJECT_RELEASED;
        readiness_event = object->readiness_event;
        object->readiness_event = NULL;
        object->has_readiness_event = 0U;
    }
    pthread_mutex_unlock(&object->lock);
    pthread_mutex_unlock(&runtime->mutex);

    if (status == SHADOWSPILL_RUNTIME_OK && readiness_event != NULL &&
        shadowspill_event_lease_release(runtime, readiness_event) != 0) {
        status = SHADOWSPILL_RUNTIME_BACKEND_FAILURE;
    }
    return status;
}

ShadowSpillRuntimeStatus shadowspill_object_owner_retain(
    ShadowSpillObject *object
) {
    if (object == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
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
            return SHADOWSPILL_RUNTIME_OK;
        }
    }
    return SHADOWSPILL_RUNTIME_INVALID_STATE;
}

ShadowSpillRuntimeStatus shadowspill_object_owner_release(
    ShadowSpillObject *object
) {
    if (object == NULL) {
        return SHADOWSPILL_RUNTIME_OK;
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
        return SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    ShadowSpillRuntimeStatus status = SHADOWSPILL_RUNTIME_OK;
    if (owners == 1U) {
        ShadowSpillRuntime *runtime = object->runtime;
        if (atomic_load_explicit(
                &object->detached, memory_order_acquire
            ) == 0U) {
            if (shadowspill_object_table_remove(
                    &runtime->objects, object
                ) != 0) {
                status = SHADOWSPILL_RUNTIME_INVALID_STATE;
            } else {
                (void)atomic_fetch_sub_explicit(
                    &runtime->registered_objects, 1U, memory_order_acq_rel
                );
            }
        }
        if (status == SHADOWSPILL_RUNTIME_OK) {
            status = release_object_residency(object, 0U, 0U);
        }
    }
    if (status != SHADOWSPILL_RUNTIME_OK) {
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

ShadowSpillRuntimeStatus shadowspill_object_handle_release(
    ShadowSpillObjectHandle *handle
) {
    if (handle == NULL) {
        return SHADOWSPILL_RUNTIME_OK;
    }
    const ShadowSpillRuntimeStatus status = shadowspill_object_owner_release(
        handle->object
    );
    handle->object = NULL;
    handle->runtime = NULL;
    free(handle);
    return status;
}

ShadowSpillRuntimeStatus shadowspill_object_release_generation(
    const ShadowSpillObjectHandle *handle,
    uint64_t expected_generation
) {
    if (handle == NULL || handle->runtime == NULL || handle->object == NULL ||
        handle->object->runtime != handle->runtime ||
        atomic_load_explicit(
            &handle->object->detached, memory_order_acquire
        ) != 0U) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    return release_object_residency(
        handle->object, expected_generation, 1U
    );
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
    created->runtime = runtime;
    atomic_init(&created->references, 1U);
    atomic_init(&created->owners, 1U);
    atomic_init(&created->registration_owned, 1U);
    atomic_init(&created->detached, 0U);
    atomic_init(&created->unpublished_fetch_count, 0U);
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
        shadowspill_spill_location(runtime, created)->lease->state = SHADOWSPILL_LEASE_IN_USE;
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
    ShadowSpillObject *object = shadowspill_object_table_acquire(
        &runtime->objects, object_id
    );
    if (status != SHADOWSPILL_RUNTIME_OK) {
        goto done;
    }
    if (object == NULL) {
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
    uint8_t expected_registration = 1U;
    if (!atomic_compare_exchange_strong_explicit(
            &object->registration_owned,
            &expected_registration,
            0U,
            memory_order_acq_rel,
            memory_order_acquire
        )) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
        goto done;
    }

done:
    pthread_mutex_unlock(&runtime->mutex);
    if (status == SHADOWSPILL_RUNTIME_OK) {
        status = shadowspill_object_owner_release(object);
    }
    shadowspill_object_release(object);
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
        shadowspill_execution_pool(runtime), allocation_id
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
        shadowspill_execution_pool(runtime), allocation_id
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
    replacement->state = SHADOWSPILL_LEASE_IN_USE;
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
    if (shadowspill_memory_pool_begin_retirement_locked(
            prior, NULL, 1
        ) != 0) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
        goto done;
    }
    if (shadowspill_track_task_retirement(runtime, prior) != 0) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
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
    ShadowSpillBackendStream consumer_stream,
    ShadowSpillAllocation *allocation
) {
    if (runtime == NULL || allocation == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    ShadowSpillObject *object = NULL;
    uint64_t expected_allocation_id = SHADOWSPILL_RUNTIME_NO_ID;
    uint64_t expected_generation = 0U;

    /*
     * An admitted task record retains direct object pointers. Caller
     * handoff therefore drains this generation's actions but preserves the
     * object record itself for the next recurrent invocation.
     */
    pthread_mutex_lock(&runtime->mutex);
    ShadowSpillRuntimeStatus status = shadowspill_current_status_locked(runtime);
    if (status != SHADOWSPILL_RUNTIME_OK) {
        pthread_mutex_unlock(&runtime->mutex);
        return status;
    }
    object = shadowspill_find_object(runtime, object_id);
    if (object == NULL) {
        pthread_mutex_unlock(&runtime->mutex);
        return SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    shadowspill_object_retain(object);
    for (;;) {
        pthread_mutex_lock(&object->lock);
        const int settled = object->action_head == NULL &&
            object->residency != SHADOWSPILL_OBJECT_PREFETCHING &&
            object->residency != SHADOWSPILL_OBJECT_OFFLOADING;
        if (settled || status != SHADOWSPILL_RUNTIME_OK) {
            break;
        }
        pthread_mutex_unlock(&object->lock);
        pthread_cond_wait(&runtime->condition, &runtime->mutex);
        status = shadowspill_current_status_locked(runtime);
    }
    if (status == SHADOWSPILL_RUNTIME_OK &&
        (object->residency != SHADOWSPILL_OBJECT_EXECUTION_READY ||
         object->allocation_id == SHADOWSPILL_RUNTIME_NO_ID ||
         object->has_readiness_event)) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    if (status == SHADOWSPILL_RUNTIME_OK) {
        expected_allocation_id = object->allocation_id;
        expected_generation = object->generation;
    }
    pthread_mutex_unlock(&object->lock);
    pthread_mutex_unlock(&runtime->mutex);
    if (status != SHADOWSPILL_RUNTIME_OK) {
        goto release_object;
    }

    status = shadowspill_memory_pool_record_stream(
        runtime,
        SHADOWSPILL_EXECUTION_POOL_ID,
        expected_allocation_id,
        consumer_stream
    );
    if (status != SHADOWSPILL_RUNTIME_OK) {
        goto release_object;
    }

    /* Commit ownership only if the snapshotted generation is still current. */
    pthread_mutex_lock(&runtime->mutex);
    status = shadowspill_current_status_locked(runtime);
    pthread_mutex_lock(&object->lock);
    if (status != SHADOWSPILL_RUNTIME_OK ||
        atomic_load_explicit(&object->detached, memory_order_acquire) != 0U ||
        object->allocation_id != expected_allocation_id ||
        object->generation != expected_generation ||
        object->residency != SHADOWSPILL_OBJECT_EXECUTION_READY ||
        object->action_head != NULL || object->has_readiness_event) {
        status = status == SHADOWSPILL_RUNTIME_OK
            ? SHADOWSPILL_RUNTIME_INVALID_STATE
            : status;
        goto done_object;
    }

    shadowspill_memory_pool_lock_foreground(shadowspill_execution_pool(runtime));
    ShadowSpillMemoryLease *record = shadowspill_find_execution_lease(
        shadowspill_execution_pool(runtime), object->allocation_id
    );
    if (record == NULL || record->pointer == NULL || record->logical_freed ||
        !record->plan_owned) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
        goto done_allocation;
    }
    if (object->residency == SHADOWSPILL_OBJECT_EXECUTION_READY &&
        shadowspill_spill_location(runtime, object)->lease != NULL) {
        shadowspill_memory_pool_lock_foreground(shadowspill_spill_pool(runtime));
        const int release_status = shadowspill_memory_pool_release_lease_locked(
            shadowspill_spill_location(runtime, object)->lease
        );
        shadowspill_memory_pool_unlock_foreground(shadowspill_spill_pool(runtime));
        if (release_status != 0) {
            status = SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
            goto done_allocation;
        }
        if (shadowspill_spill_location(runtime, object)->owns_lease) {
            free(shadowspill_spill_location(runtime, object)->lease);
        }
        shadowspill_spill_location(runtime, object)->lease = NULL;
        shadowspill_spill_location(runtime, object)->owns_lease = 0U;
        shadowspill_spill_location(runtime, object)->current = 0U;
    }
    record->framework_free_seen = 0;
    record->plan_owned = 0;
    record->bound_object_id = SHADOWSPILL_RUNTIME_NO_ID;
    object->retired_generation = object->generation;
    object->retired_execution_pointer = record->pointer;
    object->allocation_id = SHADOWSPILL_RUNTIME_NO_ID;
    shadowspill_execution_location(runtime, object)->lease = NULL;
    shadowspill_execution_location(runtime, object)->current = 0U;
    object->residency = SHADOWSPILL_OBJECT_RELEASED;
    shadowspill_append_allocation_event_locked(
        runtime,
        record,
        SHADOWSPILL_ALLOCATION_PROMOTED,
        SHADOWSPILL_ALLOCATION_CALLER_OWNED
    );
    *allocation = (ShadowSpillAllocation){
        .pool_id = shadowspill_execution_pool(runtime)->pool_id,
        .allocation_id = record->allocation_id,
        .generation = record->generation,
        .requested_bytes = record->requested_bytes,
        .charged_bytes = record->charged_bytes,
        .pointer = record->pointer,
    };

done_allocation:
    shadowspill_memory_pool_unlock_foreground(shadowspill_execution_pool(runtime));
done_object:
    pthread_mutex_unlock(&object->lock);
    pthread_mutex_unlock(&runtime->mutex);
release_object:
    shadowspill_object_release(object);
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
        shadowspill_execution_pool(runtime), object->allocation_id
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
