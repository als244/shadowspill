#include "internal.h"
#include "internal/task_boundaries.h"

#include <stdlib.h>

typedef struct ShadowSpillActionBatch {
    ShadowSpillQueuedAction *head;
    ShadowSpillQueuedAction *tail;
} ShadowSpillActionBatch;

static void release_reserved_destination(
    ShadowSpillRuntime *runtime,
    ShadowSpillQueuedAction *action
) {
    ShadowSpillMemoryLease *lease = action->destination_lease;
    if (lease == NULL) {
        return;
    }
    action->destination_lease = NULL;
    if (action->kind == SHADOWSPILL_RUNTIME_PREFETCH) {
        shadowspill_memory_pool_lock_reservation(
            shadowspill_execution_pool(runtime)
        );
        shadowspill_cancel_execution_reservation_locked(runtime, lease);
        shadowspill_memory_pool_unlock_reservation(
            shadowspill_execution_pool(runtime)
        );
        shadowspill_memory_pool_relinquish_reservation(
            shadowspill_execution_pool(runtime)
        );
        return;
    }
    shadowspill_memory_pool_lock_reservation(shadowspill_spill_pool(runtime));
    (void)shadowspill_memory_pool_cancel_reservation_locked(lease);
    shadowspill_memory_pool_unlock_reservation(shadowspill_spill_pool(runtime));
    shadowspill_memory_pool_relinquish_reservation(
        shadowspill_spill_pool(runtime)
    );
    free(lease);
}

static ShadowSpillRuntimeStatus try_reserve_action_destination_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillQueuedAction *action,
    ShadowSpillMemoryPool *pool
) {
    ShadowSpillRuntimeStatus status = SHADOWSPILL_RUNTIME_OK;
    if (action->kind == SHADOWSPILL_RUNTIME_PREFETCH) {
        const ShadowSpillFixedPlacementDescription *fixed =
            action->admitted
            ? shadowspill_fixed_layout_find_placement(
                  runtime,
                  SHADOWSPILL_FIXED_ACTION_DESTINATION,
                  action->task_id,
                  action->action_ordinal,
                  action->object->object_id
              )
            : NULL;
        if (action->admitted && runtime->fixed_layout.sealed && fixed == NULL) {
            return SHADOWSPILL_RUNTIME_PLAN_VIOLATION;
        }
        status = fixed == NULL
            ? shadowspill_create_execution_lease_locked(
                  runtime,
                  action->object->size_bytes,
                  pool->minimum_alignment,
                  1,
                  SHADOWSPILL_MEMORY_BEST_FIT_LOW,
                  action->task_id,
                  &action->destination_lease
              )
            : shadowspill_create_fixed_execution_lease_locked(
                  runtime,
                  fixed,
                  1,
                  action->task_id,
                  &action->destination_lease
              );
        if (fixed == NULL && status == SHADOWSPILL_RUNTIME_OUT_OF_MEMORY) {
            status = shadowspill_create_execution_successor_locked(
                runtime,
                action->object->size_bytes,
                pool->minimum_alignment,
                action->task_id,
                &action->destination_lease
            );
        }
    } else {
        action->destination_lease = calloc(
            1U, sizeof(*action->destination_lease)
        );
        const int reserve_status = action->destination_lease == NULL
            ? -1
            : shadowspill_memory_pool_reserve_lease_locked(
                pool,
                action->destination_lease,
                action->object->size_bytes,
                pool->minimum_alignment,
                SHADOWSPILL_MEMORY_BEST_FIT_LOW
            );
        if (reserve_status != 0) {
            const int successor_status = action->destination_lease == NULL
                ? -1
                : shadowspill_memory_pool_reserve_causal_successor_locked(
                    pool,
                    action->destination_lease,
                    action->object->size_bytes,
                    pool->minimum_alignment
                );
            if (successor_status != 0) {
                free(action->destination_lease);
                action->destination_lease = NULL;
                status = successor_status > 0
                    ? SHADOWSPILL_RUNTIME_OUT_OF_MEMORY
                    : SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
            }
        }
    }
    if (status != SHADOWSPILL_RUNTIME_OK) {
        return status;
    }
    if (action->destination_lease->state !=
            SHADOWSPILL_LEASE_SUCCESSOR_RESERVED &&
        shadowspill_memory_pool_mark_reserved_locked(
            action->destination_lease
        ) != 0) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    if (status == SHADOWSPILL_RUNTIME_OK &&
        action->kind == SHADOWSPILL_RUNTIME_PREFETCH) {
        action->destination_lease->bound_object_id =
            action->object->object_id;
    }
    return status;
}

static ShadowSpillRuntimeStatus reserve_action_destination(
    ShadowSpillRuntime *runtime,
    ShadowSpillQueuedAction *action,
    int destination_required
) {
    if (!destination_required) {
        return SHADOWSPILL_RUNTIME_OK;
    }
    ShadowSpillMemoryPool *pool = action->kind == SHADOWSPILL_RUNTIME_PREFETCH
        ? shadowspill_execution_pool(runtime)
        : shadowspill_spill_pool(runtime);
    ShadowSpillRuntimeStatus status = SHADOWSPILL_RUNTIME_OK;
    shadowspill_memory_pool_lock_reservation(pool);
    for (;;) {
        status = try_reserve_action_destination_locked(runtime, action, pool);
        if (status != SHADOWSPILL_RUNTIME_OUT_OF_MEMORY) {
            break;
        }
        const int eventually_fits =
            shadowspill_memory_pool_can_reserve_after_releases_locked(
                pool,
                action->object->size_bytes,
                pool->minimum_alignment
            );
        if (eventually_fits <= 0) {
            status = eventually_fits == 0
                ? SHADOWSPILL_RUNTIME_OUT_OF_MEMORY
                : SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
            break;
        }
        /*
         * Several completed predecessors may need to coalesce before this
         * destination has one compatible range. Their events are already
         * published, so waiting here has a concrete progress source.
         */
        shadowspill_notify_worker(runtime);
        pthread_cond_wait(&pool->capacity_changed, &pool->lock);
        status = shadowspill_failure_status(runtime);
        if (status != SHADOWSPILL_RUNTIME_OK) {
            break;
        }
    }
    shadowspill_memory_pool_unlock_reservation(pool);
    shadowspill_memory_pool_relinquish_reservation(pool);

    if (status != SHADOWSPILL_RUNTIME_OK) {
        release_reserved_destination(runtime, action);
        if (status == SHADOWSPILL_RUNTIME_OUT_OF_MEMORY) {
            shadowspill_latch_task_failure(
                runtime,
                SHADOWSPILL_RUNTIME_NO_PROGRESS,
                action->task_id,
                action->object->object_id,
                SHADOWSPILL_RUNTIME_NO_ID,
                action->object->size_bytes
            );
            return SHADOWSPILL_RUNTIME_NO_PROGRESS;
        }
        shadowspill_latch_task_failure(
            runtime,
            status,
            action->task_id,
            action->object->object_id,
            SHADOWSPILL_RUNTIME_NO_ID,
            action->object->size_bytes
        );
        return status;
    }
    shadowspill_append_trace_event_locked(
        runtime,
        SHADOWSPILL_TRACE_DESTINATION_RESERVED,
        action->task_id,
        action->object->object_id,
        action->object->allocation_id,
        action->object->size_bytes,
        action->kind,
        action->destination_lease->offset
    );
    return SHADOWSPILL_RUNTIME_OK;
}

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
            (object->residency != SHADOWSPILL_OBJECT_EXECUTION_READY &&
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
                *failure_object_id = source_id;
                *failure_allocation_id = allocation->allocation_id;
                status = SHADOWSPILL_RUNTIME_INVALID_STATE;
                break;
            }
            if (source->handoff_task_id == record->task_id &&
                !action_releases_object(record, source->object_id)) {
                *failure_object_id = source->handoff_destination_object_id;
                *failure_allocation_id = allocation->allocation_id;
                status = SHADOWSPILL_RUNTIME_PLAN_VIOLATION;
                break;
            }
            source_id = source->handoff_next_source_object_id;
        }
        if (status != SHADOWSPILL_RUNTIME_OK) {
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
        release_reserved_destination(runtime, action);
        pthread_mutex_lock(&action->object->lock);
        (void)shadowspill_object_remove_action_locked(
            action->object, action
        );
        if (action->kind == SHADOWSPILL_RUNTIME_PREFETCH) {
            action->object->prefetch_pending = 0U;
        } else {
            ShadowSpillMemoryLease *source = shadowspill_execution_location(
                runtime, action->object
            )->lease;
            if (source != NULL &&
                source->state == SHADOWSPILL_LEASE_RETIRE_PENDING) {
                (void)shadowspill_memory_pool_cancel_retirement_locked(source);
            }
        }
        pthread_cond_broadcast(&action->object->state_changed);
        pthread_mutex_unlock(&action->object->lock);
        shadowspill_release_task_fence_locked(runtime, action->fence);
        if (action->admitted) {
            const uint8_t kind = action->kind;
            ShadowSpillObject *object = action->object;
            const uint64_t task_id = action->task_id;
            const uint64_t action_ordinal = action->action_ordinal;
            const uint64_t completed_generation =
                action->completed_generation;
            const char *trace_label = action->trace_label;
            *action = (ShadowSpillQueuedAction){
                .task_id = task_id,
                .action_ordinal = action_ordinal,
                .completed_generation = completed_generation,
                .kind = kind,
                .object = object,
                .trace_label = trace_label,
                .admitted = 1U,
            };
        } else {
            shadowspill_object_release(action->object);
            if (action->owns_trace_label) {
                free((void *)action->trace_label);
            }
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
        ShadowSpillQueuedAction *queued = &record->queued_actions[index];
        if (queued->active) {
            pthread_mutex_unlock(&object->lock);
            return SHADOWSPILL_RUNTIME_INVALID_STATE;
        }
        ShadowSpillRuntimeStatus status =
            shadowspill_object_schedule_action_locked(
                runtime, object, queued
            );
        if (status != SHADOWSPILL_RUNTIME_OK) {
            pthread_mutex_unlock(&object->lock);
            return status;
        }
        const uint64_t expected_generation = object->generation;
        const uint64_t expected_version = object->authoritative_version;
        /*
         * A non-retained spill lease belongs to the preceding fetch and is
         * retired when that fetch completes.  It may still be visible here
         * while the dispatcher runs ahead, but it cannot serve a later
         * eviction.  Reserve the next spill generation at this trigger.
         */
        const int destination_required =
            action->kind == SHADOWSPILL_RUNTIME_PREFETCH ||
            (action->kind == SHADOWSPILL_RUNTIME_OFFLOAD &&
             (shadowspill_spill_location(runtime, object)->lease == NULL ||
              !object->retain_spill_copy));
        queued->active = 1U;
        queued->activation_generation =
            shadowspill_current_task_invocation(runtime);
        queued->state = SHADOWSPILL_ACTION_QUEUED;
        queued->fence = fence;
        shadowspill_retain_task_fence(fence);
        if (batch->tail == NULL) {
            batch->head = queued;
        } else {
            queued->previous = batch->tail;
            batch->tail->next = queued;
        }
        batch->tail = queued;
        if (action->kind == SHADOWSPILL_RUNTIME_PREFETCH) {
            object->prefetch_pending = 1U;
        } else {
            ShadowSpillMemoryLease *source =
                shadowspill_execution_location(runtime, object)->lease;
            if (source == NULL) {
                pthread_mutex_unlock(&object->lock);
                return SHADOWSPILL_RUNTIME_INVALID_STATE;
            }
            const int keeps_lease_for_handoff =
                action->kind == SHADOWSPILL_RUNTIME_RELEASE &&
                object->handoff_task_id == record->task_id;
            if (!keeps_lease_for_handoff) {
                pthread_mutex_lock(&shadowspill_execution_pool(runtime)->lock);
                const int retirement_status =
                    shadowspill_memory_pool_begin_retirement_locked(
                        source,
                        action->kind == SHADOWSPILL_RUNTIME_RELEASE
                            ? fence->event
                            : NULL,
                        action->kind == SHADOWSPILL_RUNTIME_OFFLOAD
                    );
                pthread_mutex_unlock(
                    &shadowspill_execution_pool(runtime)->lock
                );
                if (retirement_status != 0) {
                    pthread_mutex_unlock(&object->lock);
                    return SHADOWSPILL_RUNTIME_INVALID_STATE;
                }
            }
        }
        pthread_mutex_unlock(&object->lock);
        status = reserve_action_destination(
            runtime, queued, destination_required
        );
        if (status != SHADOWSPILL_RUNTIME_OK) {
            return status;
        }
        pthread_mutex_lock(&object->lock);
        const int unchanged = object->generation == expected_generation &&
            object->authoritative_version == expected_version;
        pthread_mutex_unlock(&object->lock);
        if (!unchanged) {
            return SHADOWSPILL_RUNTIME_INVALID_STATE;
        }
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
        if (shadowspill_memory_pool_publish_retirement_dependency_locked(
                allocation, fence->event
            ) != 0) {
            status = SHADOWSPILL_RUNTIME_INVALID_STATE;
            break;
        }
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
        status = shadowspill_validate_task_allocation_complete(runtime);
    }
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
            status = attach_task_retirements_locked(
                runtime, record->task_id, fence
            );
        }
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
            publish_action_batch_locked(runtime, record, &batch);
            shadowspill_notify_worker(runtime);
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
        /*
         * Preserve causal retirement even when the first failure came from
         * an allocator callback inside this task. Such frees occur after the
         * failure is latched and otherwise have no completion source.
         */
        pthread_mutex_lock(&shadowspill_execution_pool(runtime)->lock);
        const ShadowSpillRuntimeStatus retirement_status =
            shadowspill_fence_task_retirements_locked(
                runtime, record->task_id, compute_stream
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
    const ShadowSpillExecutionDescription description = {
        .task_id = task_id,
        .updates = updates,
        .update_count = update_count,
        .actions = actions,
        .action_count = action_count,
    };
    ShadowSpillRuntimeStatus status = shadowspill_admit_execution(
        runtime, &description
    );
    if (status != SHADOWSPILL_RUNTIME_OK) {
        /*
         * An allocator callback may have latched this failure inside the
         * active task. Publish completion for its logical frees even though
         * failure state prevents admitting a new execution record.
         */
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
        shadowspill_leave_task_scope(runtime);
        return status;
    }
    const ShadowSpillExecutionRecord *record =
        shadowspill_execution_table_acquire(&runtime->execution, task_id);
    if (record == NULL) {
        shadowspill_leave_task_scope(runtime);
        return SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    return shadowspill_after_execution_record(runtime, record, compute_stream);
}
