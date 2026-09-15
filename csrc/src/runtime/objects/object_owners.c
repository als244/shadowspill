/* Who holds an object, and when its residency is released. */
#include "../internal.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

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
        object->residency == SHADOWSPILL_OBJECT_FETCHING ||
        object->residency == SHADOWSPILL_OBJECT_EVICTING) {
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
                shadowspill_release_lease_locked(runtime, lease);
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
            SHADOWSPILL_FAILURE_REASON_OBJECT_STATE_REJECTED,
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
