#include "internal.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

ShadowSpillObjectRecord *shadowspill_find_object(
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
    ShadowSpillObjectRecord *created = calloc(1U, sizeof(*created));
    if (created == NULL) {
        status = SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
        goto done;
    }
    atomic_init(&created->references, 1U);
    atomic_init(&created->detached, 0U);
    if (pthread_mutex_init(&created->lock, NULL) != 0) {
        free(created);
        status = SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
        goto done;
    }
    created->object_id = description->object_id;
    created->size_bytes = description->size_bytes;
    created->authoritative_version = description->initial_version;
    created->retain_host_backing = description->retain_host_backing;
    created->allocation_id = SHADOWSPILL_RUNTIME_NO_ID;
    created->residency = description->initially_host_resident
        ? SHADOWSPILL_OBJECT_HOST_ONLY
        : SHADOWSPILL_OBJECT_RELEASED;
    if (description->initially_host_resident) {
        uint64_t charged = description->size_bytes == 0U
            ? 1U
            : description->size_bytes;
        if (shadowspill_range_allocate(
                &runtime->host_ranges, charged, 1U, &created->host_offset
            ) != 0) {
            pthread_mutex_destroy(&created->lock);
            free(created);
            status = SHADOWSPILL_RUNTIME_OUT_OF_MEMORY;
            goto done;
        }
        created->has_host_range = 1U;
        created->host_current = 1U;
        created->host_version = description->initial_version;
    }
    if (shadowspill_object_table_insert(&runtime->objects, created) != 0) {
        if (created->has_host_range) {
            uint64_t charged = created->size_bytes == 0U
                ? 1U
                : created->size_bytes;
            (void)shadowspill_range_free(
                &runtime->host_ranges, created->host_offset, charged
            );
        }
        pthread_mutex_destroy(&created->lock);
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
    ShadowSpillObjectRecord *object = shadowspill_find_object(runtime, object_id);
    if (status != SHADOWSPILL_RUNTIME_OK) {
        goto done;
    }
    if (object == NULL || object->allocation_id != SHADOWSPILL_RUNTIME_NO_ID ||
        (object->residency != SHADOWSPILL_OBJECT_HOST_ONLY &&
         object->residency != SHADOWSPILL_OBJECT_RELEASED)) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
        goto done;
    }
    for (ShadowSpillQueuedAction *action = runtime->action_head;
         action != NULL; action = action->next) {
        if (action->object == object) {
            status = SHADOWSPILL_RUNTIME_INVALID_STATE;
            goto done;
        }
    }
    if (object->has_host_range) {
        uint64_t charged = object->size_bytes == 0U ? 1U : object->size_bytes;
        if (shadowspill_range_free(
                &runtime->host_ranges, object->host_offset, charged
            ) != 0) {
            status = SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
            goto done;
        }
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

ShadowSpillRuntimeStatus shadowspill_write_host_object(
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
    ShadowSpillObjectRecord *object = shadowspill_find_object(runtime, object_id);
    if (status != SHADOWSPILL_RUNTIME_OK) {
        goto done;
    }
    if (object == NULL || bytes != object->size_bytes ||
        !object->has_host_range ||
        object->residency != SHADOWSPILL_OBJECT_HOST_ONLY ||
        object->allocation_id != SHADOWSPILL_RUNTIME_NO_ID) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
        goto done;
    }
    if (bytes != 0U) {
        memcpy(
            (unsigned char *)runtime->host_arena + object->host_offset,
            source,
            (size_t)bytes
        );
    }
    object->host_current = 1U;
    object->host_version = object->authoritative_version;

done:
    pthread_mutex_unlock(&runtime->mutex);
    return status;
}

ShadowSpillRuntimeStatus shadowspill_read_host_object(
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
    ShadowSpillObjectRecord *object = shadowspill_find_object(runtime, object_id);
    if (status != SHADOWSPILL_RUNTIME_OK) {
        goto done;
    }
    if (object == NULL || bytes != object->size_bytes ||
        !object->has_host_range || !object->host_current ||
        object->host_version != object->authoritative_version ||
        object->residency != SHADOWSPILL_OBJECT_HOST_ONLY) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
        goto done;
    }
    if (bytes != 0U) {
        memcpy(
            destination,
            (unsigned char *)runtime->host_arena + object->host_offset,
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
    pthread_mutex_lock(&runtime->allocation_pool.lock);
    ShadowSpillObjectRecord *object = shadowspill_find_object(runtime, object_id);
    ShadowSpillAllocationRecord *allocation = shadowspill_find_allocation(
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
    ShadowSpillObjectRecord *previous_owner = NULL;
    for (ShadowSpillObjectRecord *candidate = runtime->objects.owned_head;
         candidate != NULL; candidate = candidate->ownership_next) {
        if (candidate != object &&
            candidate->allocation_id == allocation_id) {
            if (previous_owner != NULL) {
                status = SHADOWSPILL_RUNTIME_INVALID_STATE;
                goto done;
            }
            previous_owner = candidate;
        }
    }
    const uint64_t task_id = shadowspill_current_task_id(runtime);
    if (previous_owner != NULL &&
        (allocation->handoff_from_object_id != SHADOWSPILL_RUNTIME_NO_ID ||
         task_id == SHADOWSPILL_RUNTIME_NO_ID ||
         (previous_owner->residency != SHADOWSPILL_OBJECT_DEVICE_READY &&
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
        allocation->handoff_from_object_id = previous_owner->object_id;
        allocation->handoff_to_object_id = object->object_id;
        allocation->handoff_task_id = task_id;
    }
    object->allocation_id = allocation_id;
    object->generation = allocation->generation;
    object->device_version = object->authoritative_version;
    object->residency = SHADOWSPILL_OBJECT_DEVICE_READY;

done:
    pthread_mutex_unlock(&runtime->allocation_pool.lock);
    pthread_mutex_unlock(&runtime->mutex);
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
    pthread_mutex_lock(&runtime->allocation_pool.lock);
    ShadowSpillRuntimeStatus status = shadowspill_current_status_locked(runtime);
    ShadowSpillObjectRecord *object = shadowspill_find_object(runtime, object_id);
    ShadowSpillAllocationRecord *record = object == NULL
        ? NULL
        : shadowspill_find_allocation(runtime, object->allocation_id);
    if (status != SHADOWSPILL_RUNTIME_OK) {
        goto done;
    }
    if (object == NULL || record == NULL || record->pointer == NULL ||
        record->logical_freed || !record->plan_owned ||
        object->residency != SHADOWSPILL_OBJECT_DEVICE_READY) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
        goto done;
    }
    for (ShadowSpillQueuedAction *action = runtime->action_head;
         action != NULL; action = action->next) {
        if (action->object == object) {
            status = SHADOWSPILL_RUNTIME_INVALID_STATE;
            goto done;
        }
    }
    if (object->has_host_range) {
        uint64_t charged = object->size_bytes == 0U ? 1U : object->size_bytes;
        if (shadowspill_range_free(
                &runtime->host_ranges, object->host_offset, charged
            ) != 0) {
            status = SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
            goto done;
        }
    }
    if (shadowspill_object_table_remove(&runtime->objects, object) != 0) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
        goto done;
    }
    --runtime->registered_objects;
    record->framework_free_seen = 0;
    record->plan_owned = 0;
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

done:
    pthread_mutex_unlock(&runtime->allocation_pool.lock);
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
    pthread_mutex_lock(&runtime->allocation_pool.lock);
    ShadowSpillObjectRecord *object = shadowspill_find_object(runtime, object_id);
    if (object == NULL) {
        pthread_mutex_unlock(&runtime->allocation_pool.lock);
        pthread_mutex_unlock(&runtime->mutex);
        return SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    ShadowSpillAllocationRecord *allocation = shadowspill_find_allocation(
        runtime, object->allocation_id
    );
    *snapshot = (ShadowSpillObjectSnapshot){
        .object_id = object->object_id,
        .size_bytes = object->size_bytes,
        .generation = object->generation,
        .allocation_id = object->allocation_id,
        .authoritative_version = object->authoritative_version,
        .device_version = object->device_version,
        .host_version = object->host_version,
        .residency = object->residency,
        .host_current = object->host_current,
        .has_host_range = object->has_host_range,
        .device_pointer = allocation == NULL ? NULL : allocation->pointer,
        .retired_generation = object->retired_generation,
        .retired_device_pointer = object->retired_device_pointer,
    };
    pthread_mutex_unlock(&runtime->allocation_pool.lock);
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
        runtime->queued_actions
    );
    ShadowSpillRuntimeStatus status = shadowspill_current_status_locked(runtime);
    for (uint32_t index = 0; status == SHADOWSPILL_RUNTIME_OK &&
         index < input_count; ++index) {
        ShadowSpillObjectRecord *object = NULL;
        while (status == SHADOWSPILL_RUNTIME_OK) {
            object = shadowspill_find_object(runtime, input_object_ids[index]);
            if (object == NULL ||
                object->residency != SHADOWSPILL_OBJECT_HOST_ONLY) {
                break;
            }
            int pending_prefetch = 0;
            for (ShadowSpillQueuedAction *action = runtime->action_head;
                 action != NULL; action = action->next) {
                if (action->object == object &&
                    action->kind == SHADOWSPILL_RUNTIME_PREFETCH) {
                    pending_prefetch = 1;
                    break;
                }
            }
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
                runtime->queued_actions
            );
            pthread_cond_wait(&runtime->condition, &runtime->mutex);
            status = shadowspill_current_status_locked(runtime);
        }
        if (object == NULL) {
            status = SHADOWSPILL_RUNTIME_INVALID_STATE;
            break;
        }
        pthread_mutex_lock(&runtime->allocation_pool.lock);
        ShadowSpillAllocationRecord *allocation = shadowspill_find_allocation(
            runtime, object->allocation_id
        );
        void *device_pointer = allocation == NULL ? NULL : allocation->pointer;
        if ((object->residency != SHADOWSPILL_OBJECT_DEVICE_READY &&
             object->residency != SHADOWSPILL_OBJECT_PREFETCHING) ||
            allocation == NULL || device_pointer == NULL ||
            object->device_version != object->authoritative_version) {
            pthread_mutex_unlock(&runtime->allocation_pool.lock);
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
        pthread_mutex_unlock(&runtime->allocation_pool.lock);
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
        shadowspill_release_task_fence_locked(runtime, head->fence);
        shadowspill_object_release(head->object);
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
        ShadowSpillObjectRecord *object = shadowspill_find_object(
            runtime, updates[index].object_id
        );
        if (object == NULL || updates[index].version_delta == 0U ||
            (object->residency != SHADOWSPILL_OBJECT_DEVICE_READY &&
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
        object->device_version = object->authoritative_version;
        object->host_current = 0U;
    }
    pthread_mutex_lock(&runtime->allocation_pool.lock);
    for (ShadowSpillAllocationRecord *allocation = runtime->active_allocations;
         allocation != NULL; allocation = allocation->active_next) {
        if (allocation->handoff_task_id != task_id) {
            continue;
        }
        int matched_release = 0;
        for (uint32_t index = 0; index < action_count; ++index) {
            if (actions[index].object_id ==
                    allocation->handoff_from_object_id &&
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
                allocation->handoff_to_object_id,
                allocation->allocation_id,
                allocation->requested_bytes
            );
            break;
        }
    }
    pthread_mutex_unlock(&runtime->allocation_pool.lock);
    if (status != SHADOWSPILL_RUNTIME_OK) {
        goto done;
    }
    uint64_t task_retirement_count = 0U;
    pthread_mutex_lock(&runtime->allocation_pool.lock);
    for (ShadowSpillAllocationRecord *allocation = runtime->active_allocations;
         allocation != NULL; allocation = allocation->active_next) {
        if (allocation->logical_freed && allocation->pointer != NULL &&
            allocation->release_task_id == task_id &&
            allocation->retirement_events == NULL &&
            allocation->retirement_fence == NULL) {
            ++task_retirement_count;
        }
    }
    pthread_mutex_unlock(&runtime->allocation_pool.lock);
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
        ShadowSpillObjectRecord *object = shadowspill_find_object(
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
            if (object->residency != SHADOWSPILL_OBJECT_HOST_ONLY ||
                !object->host_current ||
                object->host_version != object->authoritative_version) {
                status = SHADOWSPILL_RUNTIME_PLAN_VIOLATION;
                break;
            }
        } else if (object->residency != SHADOWSPILL_OBJECT_DEVICE_READY &&
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
        shadowspill_object_retain(object);
        created->fence = fence;
        shadowspill_retain_task_fence(fence);
        if (tail == NULL) {
            head = created;
        } else {
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
    pthread_mutex_lock(&runtime->allocation_pool.lock);
    for (ShadowSpillAllocationRecord *allocation = runtime->active_allocations;
         allocation != NULL; allocation = allocation->active_next) {
        if (!allocation->logical_freed || allocation->pointer == NULL ||
            allocation->release_task_id != task_id ||
            allocation->retirement_events != NULL ||
            allocation->retirement_fence != NULL) {
            continue;
        }
        allocation->retirement_fence = fence;
        shadowspill_retain_task_fence(fence);
    }
    pthread_mutex_unlock(&runtime->allocation_pool.lock);
    if (head != NULL) {
        if (runtime->action_tail == NULL) {
            runtime->action_head = head;
        } else {
            runtime->action_tail->next = head;
        }
        runtime->action_tail = tail;
        runtime->queued_actions += action_count;
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
            shadowspill_append_trace_event_locked(
                runtime,
                SHADOWSPILL_TRACE_ACTION_QUEUED,
                task_id,
                queued->object->object_id,
                queued->object->allocation_id,
                queued->object->size_bytes,
                queued->kind,
                runtime->queued_actions
            );
            if (queued == tail) {
                break;
            }
        }
    }
    pthread_cond_broadcast(&runtime->condition);

done:
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
