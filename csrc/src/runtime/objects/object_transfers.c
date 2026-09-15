/* Handing an object to the caller, and reading where it is. */
#include "../internal.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

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
        (object->residency == SHADOWSPILL_OBJECT_FETCHING &&
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
        object->action_head->kind == SHADOWSPILL_RUNTIME_FETCH &&
        object->action_head->state == SHADOWSPILL_ACTION_IN_FLIGHT &&
        ((object->residency == SHADOWSPILL_OBJECT_FETCHING &&
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
    ShadowSpillMemoryLease *record = shadowspill_find_lease(
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
    /*
     * `ShadowSpillObjectSnapshot` names two pools by role in its own shape --
     * `execution_version` and `spill_version` -- so this is where the snapshot's
     * promise, not the runtime's, fixes which pools those are. Pools themselves
     * carry no roles; a plan assigns them.
     */
    const ShadowSpillObjectLocation *execution =
        shadowspill_object_location(object, 0U);
    const ShadowSpillObjectLocation *spill =
        shadowspill_object_location(object, 1U);
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
