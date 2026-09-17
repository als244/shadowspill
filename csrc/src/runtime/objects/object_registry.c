/* The object table: finding, registering and renaming an entry. */
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
                SHADOWSPILL_ALLOCATION_ORIGIN_RUNTIME_OBJECT
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

/* Where a lease sits in its pool.
   Both directions across the pool's edge need this and neither may use the
   lease pointer instead: for a kind whose region lives on another machine the
   pointer is meaningful only to that kind, while the offset is meaningful to
   everyone. */
static uint64_t lease_offset(
    const ShadowSpillMemoryPool *pool,
    const ShadowSpillObjectLocation *location
) {
    return (uint64_t)(
        (const char *)location->lease->pointer - (const char *)pool->base
    );
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
        ShadowSpillMemoryPool *const pool =
            shadowspill_runtime_pool(runtime, pool_id);
        if (pool->memory.write != NULL) {
            /* A kind whose region this process cannot address writes it
               itself; everyone else is an ordinary copy. */
            if (pool->memory.write(
                    pool->memory_state,
                    lease_offset(pool, location),
                    source,
                    bytes
                ) != 0) {
                status = SHADOWSPILL_STATUS_BACKEND_FAILURE;
                goto done;
            }
        } else {
            memcpy(location->lease->pointer, source, (size_t)bytes);
        }
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
    /* The requirement is that the copy being read is the authoritative one: it
       exists, it is current, and its version is the object's. Whether a copy
       also lives somewhere else does not bear on that. Requiring sole residency
       as well conflated "this copy is authoritative" with "this is the only
       copy", so a caller could not read state that was also on the device --
       which a checkpoint has to be able to do without first giving up residency
       to get at it. */
    if (object == NULL || bytes != object->size_bytes ||
        location->lease == NULL || !location->current ||
        location->version != object->authoritative_version) {
        status = SHADOWSPILL_STATUS_INVALID_STATE;
        goto read_done;
    }
    if (bytes != 0U) {
        ShadowSpillMemoryPool *const pool =
            shadowspill_runtime_pool(runtime, pool_id);
        if (pool->memory.read != NULL) {
            if (pool->memory.read(
                    pool->memory_state,
                    lease_offset(pool, location),
                    destination,
                    bytes
                ) != 0) {
                status = SHADOWSPILL_STATUS_BACKEND_FAILURE;
                goto read_done;
            }
        } else {
            memcpy(destination, location->lease->pointer, (size_t)bytes);
        }
    }

read_done:
    pthread_mutex_unlock(&runtime->mutex);
    return status;
}
