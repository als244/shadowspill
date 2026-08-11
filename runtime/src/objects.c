#include "internal.h"

#include <stdint.h>
#include <stdlib.h>

ShadowSpillObjectRecord *shadowspill_find_object(
    ShadowSpillRuntime *runtime,
    uint64_t object_id
) {
    for (ShadowSpillObjectRecord *record = runtime->objects; record != NULL;
         record = record->next) {
        if (record->object_id == object_id) {
            return record;
        }
    }
    return NULL;
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
            free(created);
            status = SHADOWSPILL_RUNTIME_OUT_OF_MEMORY;
            goto done;
        }
        created->has_host_range = 1U;
        created->host_current = 1U;
        created->host_version = description->initial_version;
    }
    created->next = runtime->objects;
    runtime->objects = created;
    ++runtime->registered_objects;

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
    allocation->plan_owned = 1;
    object->allocation_id = allocation_id;
    object->generation = allocation->generation;
    object->device_version = object->authoritative_version;
    object->residency = SHADOWSPILL_OBJECT_DEVICE_READY;

done:
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
    ShadowSpillObjectRecord *object = shadowspill_find_object(runtime, object_id);
    if (object == NULL) {
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
    };
    pthread_mutex_unlock(&runtime->mutex);
    return SHADOWSPILL_RUNTIME_OK;
}

ShadowSpillRuntimeStatus shadowspill_before_task(
    ShadowSpillRuntime *runtime,
    uint64_t task_id,
    ShadowSpillBackendStream compute_stream,
    const uint64_t *input_object_ids,
    uint32_t input_count,
    ShadowSpillObjectBinding *bindings,
    uint32_t binding_capacity
) {
    (void)task_id;
    if (runtime == NULL || (input_count != 0U && input_object_ids == NULL) ||
        (input_count != 0U && bindings == NULL) ||
        binding_capacity < input_count) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&runtime->mutex);
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
            pthread_cond_wait(&runtime->condition, &runtime->mutex);
            status = shadowspill_current_status_locked(runtime);
        }
        if (object == NULL) {
            status = SHADOWSPILL_RUNTIME_INVALID_STATE;
            break;
        }
        ShadowSpillAllocationRecord *allocation = shadowspill_find_allocation(
            runtime, object->allocation_id
        );
        if ((object->residency != SHADOWSPILL_OBJECT_DEVICE_READY &&
             object->residency != SHADOWSPILL_OBJECT_PREFETCHING) ||
            allocation == NULL || allocation->pointer == NULL ||
            object->device_version != object->authoritative_version) {
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
                    object->readiness_event
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
        }
        bindings[index] = (ShadowSpillObjectBinding){
            .object_id = object->object_id,
            .generation = object->generation,
            .allocation_id = object->allocation_id,
            .authoritative_version = object->authoritative_version,
            .pointer = allocation->pointer,
        };
    }
    pthread_mutex_unlock(&runtime->mutex);
    return status;
}

static void release_fence_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillTaskFence *fence
) {
    if (--fence->references == 0U) {
        if (runtime->backend.destroy_event(
                runtime->backend.context, fence->event
            ) != 0) {
            shadowspill_latch_failure_locked(
                runtime,
                SHADOWSPILL_RUNTIME_BACKEND_FAILURE,
                SHADOWSPILL_RUNTIME_NO_ID,
                SHADOWSPILL_RUNTIME_NO_ID,
                0U
            );
        }
        free(fence);
    }
}

static void discard_actions_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillQueuedAction *head
) {
    while (head != NULL) {
        ShadowSpillQueuedAction *next = head->next;
        release_fence_locked(runtime, head->fence);
        free(head);
        head = next;
    }
}

ShadowSpillRuntimeStatus shadowspill_after_task(
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
    if (action_count == 0U) {
        goto done;
    }
    fence = calloc(1U, sizeof(*fence));
    int fence_created = 0;
    if (fence != NULL && runtime->backend.create_event(
            runtime->backend.context, &fence->event
        ) == 0) {
        fence_created = 1;
    }
    if (fence == NULL || !fence_created || runtime->backend.record_event(
            runtime->backend.context, fence->event, compute_stream
        ) != 0) {
        if (fence_created) {
            (void)runtime->backend.destroy_event(
                runtime->backend.context, fence->event
            );
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
        if (actions[index].kind == SHADOWSPILL_RUNTIME_OFFLOAD &&
            !object->has_host_range) {
            uint64_t charged = object->size_bytes == 0U ? 1U : object->size_bytes;
            if (shadowspill_range_allocate(
                    &runtime->host_ranges, charged, 1U, &object->host_offset
                ) != 0) {
                status = SHADOWSPILL_RUNTIME_OUT_OF_MEMORY;
                break;
            }
            object->has_host_range = 1U;
        }
        ShadowSpillQueuedAction *created = calloc(1U, sizeof(*created));
        if (created == NULL) {
            status = SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
            break;
        }
        created->task_id = task_id;
        created->kind = actions[index].kind;
        created->object = object;
        created->fence = fence;
        ++fence->references;
        if (tail == NULL) {
            head = created;
        } else {
            tail->next = created;
        }
        tail = created;
    }
    if (status != SHADOWSPILL_RUNTIME_OK) {
        if (fence->references == 0U) {
            (void)runtime->backend.destroy_event(
                runtime->backend.context, fence->event
            );
            free(fence);
        } else {
            discard_actions_locked(runtime, head);
        }
        shadowspill_latch_failure_locked(
            runtime, status, SHADOWSPILL_RUNTIME_NO_ID,
            SHADOWSPILL_RUNTIME_NO_ID, 0U
        );
        goto done;
    }
    if (runtime->action_tail == NULL) {
        runtime->action_head = head;
    } else {
        runtime->action_tail->next = head;
    }
    runtime->action_tail = tail;
    runtime->queued_actions += action_count;
    pthread_cond_broadcast(&runtime->condition);

done:
    pthread_mutex_unlock(&runtime->mutex);
    return status;
}
