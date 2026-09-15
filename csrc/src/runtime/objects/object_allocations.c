/* The memory behind an object, bound and rebound. */
#include "../internal.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

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
        shadowspill_find_lease_by_pointer(pool, pointer);
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
    allocation = shadowspill_find_lease_by_pointer(pool, pointer);
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
          previous_owner->residency != SHADOWSPILL_OBJECT_FETCHING))) {
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
        shadowspill_find_lease_by_pointer(pool, pointer);
    ShadowSpillObjectLocation *location = shadowspill_object_location(
        object, pool->pool_id
    );
    ShadowSpillMemoryLease *prior = location == NULL ? NULL : location->lease;
    if (status != SHADOWSPILL_STATUS_OK) {
        goto done;
    }
    if ((object->residency != SHADOWSPILL_OBJECT_EXECUTION_READY &&
         object->residency != SHADOWSPILL_OBJECT_FETCHING) ||
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
