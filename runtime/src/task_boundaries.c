#include "internal.h"
#include "internal/task_boundaries.h"

#include <stdlib.h>

typedef struct ShadowSpillActionBatch {
    ShadowSpillQueuedAction *head;
    ShadowSpillQueuedAction *tail;
} ShadowSpillActionBatch;

static ShadowSpillRuntimeStatus publish_mutations_locked(
    ShadowSpillRuntime *runtime,
    const ShadowSpillExecutionRecord *record,
    uint64_t *failure_object_id,
    uint64_t *failure_allocation_id
) {
    for (uint32_t index = 0U; index < record->update_count; ++index) {
        const ShadowSpillExecutionUpdate *update = &record->updates[index];
        ShadowSpillObject *object = update->object;
        pthread_mutex_lock(&object->lock);
        if (update->version_delta == 0U ||
            (object->residency != SHADOWSPILL_OBJECT_DEVICE_READY &&
             object->residency != SHADOWSPILL_OBJECT_PREFETCHING) ||
            update->version_delta >
                UINT64_MAX - object->authoritative_version) {
            *failure_object_id = object->object_id;
            *failure_allocation_id = object->allocation_id;
            pthread_mutex_unlock(&object->lock);
            shadowspill_latch_failure_locked(
                runtime,
                SHADOWSPILL_RUNTIME_INVALID_STATE,
                *failure_object_id,
                *failure_allocation_id,
                0U
            );
            return SHADOWSPILL_RUNTIME_INVALID_STATE;
        }
        object->authoritative_version += update->version_delta;
        shadowspill_execution_location(runtime, object)->version = object->authoritative_version;
        shadowspill_spill_location(runtime, object)->current = 0U;
        pthread_mutex_unlock(&object->lock);
    }
    return SHADOWSPILL_RUNTIME_OK;
}

static int action_releases_object(
    const ShadowSpillExecutionRecord *record,
    uint64_t object_id
) {
    for (uint32_t index = 0U; index < record->action_count; ++index) {
        if (record->actions[index].object->object_id == object_id &&
            record->actions[index].kind == SHADOWSPILL_RUNTIME_RELEASE) {
            return 1;
        }
    }
    return 0;
}

static ShadowSpillRuntimeStatus validate_handoffs_locked(
    ShadowSpillRuntime *runtime,
    const ShadowSpillExecutionRecord *record,
    uint64_t *failure_object_id,
    uint64_t *failure_allocation_id
) {
    ShadowSpillRuntimeStatus status = SHADOWSPILL_RUNTIME_OK;
    pthread_mutex_lock(&shadowspill_execution_pool(runtime)->lock);
    for (ShadowSpillMemoryLease *allocation = runtime->active_execution_leases;
         allocation != NULL; allocation = allocation->active_next) {
        if (allocation->handoff_task_id != record->task_id) {
            continue;
        }
        if (!action_releases_object(
                record, allocation->handoff_from_object_id
            )) {
            *failure_object_id = allocation->handoff_to_object_id;
            *failure_allocation_id = allocation->allocation_id;
            status = SHADOWSPILL_RUNTIME_PLAN_VIOLATION;
            break;
        }
    }
    pthread_mutex_unlock(&shadowspill_execution_pool(runtime)->lock);
    if (status != SHADOWSPILL_RUNTIME_OK) {
        shadowspill_latch_failure_locked(
            runtime,
            status,
            *failure_object_id,
            *failure_allocation_id,
            0U
        );
    }
    return status;
}

static uint64_t count_task_retirements_locked(
    ShadowSpillRuntime *runtime,
    uint64_t task_id
) {
    uint64_t count = 0U;
    pthread_mutex_lock(&shadowspill_execution_pool(runtime)->lock);
    for (ShadowSpillMemoryLease *allocation = runtime->active_execution_leases;
         allocation != NULL; allocation = allocation->active_next) {
        if (allocation->logical_freed && allocation->pointer != NULL &&
            allocation->release_task_id == task_id &&
            allocation->retirement_events == NULL &&
            allocation->retirement_fence == NULL) {
            ++count;
        }
    }
    pthread_mutex_unlock(&shadowspill_execution_pool(runtime)->lock);
    return count;
}

static ShadowSpillRuntimeStatus record_task_fence_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillBackendStream compute_stream,
    ShadowSpillTaskFence **result
) {
    ShadowSpillTaskFence *fence = calloc(1U, sizeof(*fence));
    if (fence == NULL) {
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    ShadowSpillRuntimeStatus status = shadowspill_event_lease_create_locked(
        runtime, &fence->event
    );
    if (status != SHADOWSPILL_RUNTIME_OK || runtime->backend.record_event(
            runtime->backend.context, fence->event->event, compute_stream
        ) != 0 || shadowspill_completion_submit(
            runtime,
            compute_stream,
            fence->event,
            SHADOWSPILL_RUNTIME_NO_ID,
            SHADOWSPILL_RUNTIME_NO_ID
        ) != SHADOWSPILL_RUNTIME_OK) {
        if (fence->event != NULL) {
            (void)shadowspill_event_lease_release(runtime, fence->event);
        }
        free(fence);
        return SHADOWSPILL_RUNTIME_BACKEND_FAILURE;
    }
    *result = fence;
    return SHADOWSPILL_RUNTIME_OK;
}

static void discard_action_batch_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillActionBatch *batch
) {
    ShadowSpillQueuedAction *action = batch->head;
    while (action != NULL) {
        ShadowSpillQueuedAction *next = action->next;
        if (action->destination_priority_declared) {
            ShadowSpillMemoryPool *pool =
                action->kind == SHADOWSPILL_RUNTIME_PREFETCH
                ? shadowspill_execution_pool(runtime)
                : shadowspill_spill_pool(runtime);
            shadowspill_memory_pool_relinquish_transfer(pool);
            action->destination_priority_declared = 0U;
        }
        if (action->kind == SHADOWSPILL_RUNTIME_PREFETCH) {
            pthread_mutex_lock(&action->object->lock);
            action->object->prefetch_pending = 0U;
            pthread_cond_broadcast(&action->object->state_changed);
            pthread_mutex_unlock(&action->object->lock);
        }
        shadowspill_release_task_fence_locked(runtime, action->fence);
        if (action->admitted) {
            const uint8_t kind = action->kind;
            ShadowSpillObject *object = action->object;
            const uint64_t task_id = action->task_id;
            *action = (ShadowSpillQueuedAction){
                .task_id = task_id,
                .kind = kind,
                .object = object,
                .admitted = 1U,
            };
        } else {
            shadowspill_object_release(action->object);
            free(action);
        }
        action = next;
    }
    *batch = (ShadowSpillActionBatch){0};
}

static ShadowSpillRuntimeStatus instantiate_actions_locked(
    ShadowSpillRuntime *runtime,
    const ShadowSpillExecutionRecord *record,
    ShadowSpillTaskFence *fence,
    ShadowSpillActionBatch *batch,
    uint64_t *failure_object_id,
    uint64_t *failure_allocation_id
) {
    for (uint32_t index = 0U; index < record->action_count; ++index) {
        const ShadowSpillExecutionAction *action = &record->actions[index];
        ShadowSpillObject *object = action->object;
        pthread_mutex_lock(&object->lock);
        *failure_object_id = object->object_id;
        *failure_allocation_id = object->allocation_id;
        if (action->kind == SHADOWSPILL_RUNTIME_PREFETCH) {
            if (object->residency != SHADOWSPILL_OBJECT_HOST_ONLY ||
                !shadowspill_spill_location(runtime, object)->current ||
                shadowspill_spill_location(runtime, object)->version != object->authoritative_version) {
                pthread_mutex_unlock(&object->lock);
                return SHADOWSPILL_RUNTIME_PLAN_VIOLATION;
            }
        } else if (object->residency != SHADOWSPILL_OBJECT_DEVICE_READY &&
                   object->residency != SHADOWSPILL_OBJECT_PREFETCHING) {
            pthread_mutex_unlock(&object->lock);
            return SHADOWSPILL_RUNTIME_PLAN_VIOLATION;
        }
        ShadowSpillQueuedAction *queued = &record->queued_actions[index];
        if (queued->active) {
            pthread_mutex_unlock(&object->lock);
            return SHADOWSPILL_RUNTIME_INVALID_STATE;
        }
        queued->active = 1U;
        queued->state = SHADOWSPILL_ACTION_QUEUED;
        queued->fence = fence;
        shadowspill_retain_task_fence(fence);
        if (action->kind == SHADOWSPILL_RUNTIME_PREFETCH) {
            object->prefetch_pending = 1U;
        }
        pthread_mutex_unlock(&object->lock);
        if (batch->tail == NULL) {
            batch->head = queued;
        } else {
            queued->previous = batch->tail;
            batch->tail->next = queued;
        }
        batch->tail = queued;
    }
    return SHADOWSPILL_RUNTIME_OK;
}

static ShadowSpillRuntimeStatus attach_task_retirements_locked(
    ShadowSpillRuntime *runtime,
    uint64_t task_id,
    ShadowSpillTaskFence *fence
) {
    ShadowSpillRuntimeStatus status = SHADOWSPILL_RUNTIME_OK;
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
    return status;
}

static void publish_action_batch_locked(
    ShadowSpillRuntime *runtime,
    const ShadowSpillExecutionRecord *record,
    ShadowSpillActionBatch *batch
) {
    if (batch->head == NULL) {
        return;
    }
    for (ShadowSpillQueuedAction *queued = batch->head; queued != NULL;
         queued = queued->next) {
        ShadowSpillTransferLane *lane = shadowspill_transfer_lane_for_action(
            runtime, queued
        );
        if (lane != NULL) {
            shadowspill_transfer_lane_enqueue(lane, queued);
        }
        if (queued == batch->tail) {
            break;
        }
    }
    pthread_mutex_lock(&runtime->actions.lock);
    if (runtime->actions.tail == NULL) {
        runtime->actions.head = batch->head;
    } else {
        batch->head->previous = runtime->actions.tail;
        runtime->actions.tail->next = batch->head;
    }
    runtime->actions.tail = batch->tail;
    (void)atomic_fetch_add_explicit(
        &runtime->actions.count, record->action_count, memory_order_release
    );
    pthread_mutex_unlock(&runtime->actions.lock);
    for (ShadowSpillQueuedAction *queued = batch->head; queued != NULL;
         queued = queued->next) {
        if (queued->kind == SHADOWSPILL_RUNTIME_RELEASE ||
            queued->kind == SHADOWSPILL_RUNTIME_OFFLOAD) {
            (void)atomic_fetch_add_explicit(
                &runtime->pending_capacity_actions, 1U, memory_order_release
            );
        }
        pthread_mutex_lock(&queued->object->lock);
        const uint64_t allocation_id = queued->object->allocation_id;
        const uint64_t size_bytes = queued->object->size_bytes;
        pthread_mutex_unlock(&queued->object->lock);
        shadowspill_append_trace_event_locked(
            runtime,
            SHADOWSPILL_TRACE_ACTION_QUEUED,
            record->task_id,
            queued->object->object_id,
            allocation_id,
            size_bytes,
            queued->kind,
            atomic_load_explicit(
                &runtime->actions.count, memory_order_acquire
            )
        );
        if (queued == batch->tail) {
            break;
        }
    }
    *batch = (ShadowSpillActionBatch){0};
}

ShadowSpillRuntimeStatus shadowspill_after_execution_record(
    ShadowSpillRuntime *runtime,
    const ShadowSpillExecutionRecord *record,
    ShadowSpillBackendStream compute_stream
) {
    ShadowSpillRuntimeStatus status = shadowspill_current_status_locked(runtime);
    ShadowSpillTaskFence *fence = NULL;
    ShadowSpillActionBatch batch = {0};
    uint64_t failure_object_id = SHADOWSPILL_RUNTIME_NO_ID;
    uint64_t failure_allocation_id = SHADOWSPILL_RUNTIME_NO_ID;

    if (status == SHADOWSPILL_RUNTIME_OK) {
        status = publish_mutations_locked(
            runtime, record, &failure_object_id, &failure_allocation_id
        );
    }
    if (status == SHADOWSPILL_RUNTIME_OK) {
        status = validate_handoffs_locked(
            runtime, record, &failure_object_id, &failure_allocation_id
        );
    }
    const uint64_t retirement_count = status == SHADOWSPILL_RUNTIME_OK
        ? count_task_retirements_locked(runtime, record->task_id)
        : 0U;
    if (status == SHADOWSPILL_RUNTIME_OK &&
        (record->action_count != 0U || retirement_count != 0U)) {
        status = record_task_fence_locked(runtime, compute_stream, &fence);
        if (status == SHADOWSPILL_RUNTIME_OK) {
            status = instantiate_actions_locked(
                runtime,
                record,
                fence,
                &batch,
                &failure_object_id,
                &failure_allocation_id
            );
        }
        if (status == SHADOWSPILL_RUNTIME_OK) {
            status = attach_task_retirements_locked(
                runtime, record->task_id, fence
            );
        }
        if (status == SHADOWSPILL_RUNTIME_OK) {
            publish_action_batch_locked(runtime, record, &batch);
            pthread_cond_broadcast(&runtime->condition);
        }
    }
    if (status != SHADOWSPILL_RUNTIME_OK) {
        if (fence != NULL) {
            if (atomic_load_explicit(
                    &fence->references, memory_order_acquire
                ) == 0U) {
                (void)shadowspill_event_lease_release(runtime, fence->event);
                free(fence);
            } else {
                discard_action_batch_locked(runtime, &batch);
            }
        }
        shadowspill_latch_failure_locked(
            runtime,
            status,
            failure_object_id,
            failure_allocation_id,
            0U
        );
    }
    shadowspill_append_trace_event_locked(
        runtime,
        SHADOWSPILL_TRACE_AFTER_TASK,
        record->task_id,
        failure_object_id,
        failure_allocation_id,
        0U,
        (uint64_t)status,
        record->action_count
    );
    shadowspill_leave_task_scope(runtime);
    return status;
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
    return shadowspill_before_task_legacy(
        runtime,
        task_id,
        compute_stream,
        input_object_ids,
        input_count,
        bindings,
        binding_capacity
    );
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
    return shadowspill_after_task_legacy(
        runtime,
        task_id,
        compute_stream,
        updates,
        update_count,
        actions,
        action_count
    );
}
