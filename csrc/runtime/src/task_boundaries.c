#include "internal.h"

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
    ShadowSpillPlan *plan = action->plan_owner;
    if (action->kind == SHADOWSPILL_RUNTIME_PREFETCH) {
        shadowspill_memory_pool_lock_reservation(
            plan->execution_pool
        );
        shadowspill_cancel_execution_reservation_locked(runtime, lease);
        shadowspill_memory_pool_unlock_reservation(
            plan->execution_pool
        );
        shadowspill_memory_pool_relinquish_reservation(
            plan->execution_pool
        );
        return;
    }
    shadowspill_memory_pool_lock_reservation(plan->spill_pool);
    (void)shadowspill_memory_pool_cancel_reservation_locked(lease);
    shadowspill_memory_pool_try_recycle_lease_record_locked(lease);
    shadowspill_memory_pool_unlock_reservation(plan->spill_pool);
    shadowspill_memory_pool_relinquish_reservation(plan->spill_pool);
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
                  action->plan_owner,
                  SHADOWSPILL_FIXED_ACTION_DESTINATION,
                  action->task_id,
                  action->action_ordinal,
                  action->plan_object_id
              )
            : NULL;
        const ShadowSpillFixedPlacementDescription *dynamic =
            action->admitted && fixed == NULL
            ? shadowspill_fixed_layout_find_placement(
                  action->plan_owner,
                  SHADOWSPILL_DYNAMIC_ACTION_DESTINATION,
                  action->task_id,
                  action->action_ordinal,
                  action->plan_object_id
              )
            : NULL;
        if (action->admitted && action->plan_owner->fixed_layout.sealed &&
            fixed == NULL && dynamic == NULL) {
            return SHADOWSPILL_RUNTIME_PLAN_VIOLATION;
        }
        status = fixed == NULL
            ? shadowspill_create_execution_lease_locked(
                  runtime,
                  pool,
                  action->object->size_bytes,
                  pool->minimum_alignment,
                  1,
                  SHADOWSPILL_MEMORY_BEST_FIT_LOW,
                  action->task_id,
                  &action->destination_lease
              )
            : shadowspill_create_fixed_execution_lease_locked(
                  action->plan_owner,
                  fixed,
                  1,
                  action->task_id,
                  &action->destination_lease
              );
        if (fixed == NULL && status == SHADOWSPILL_RUNTIME_OUT_OF_MEMORY) {
            status = shadowspill_create_execution_successor_locked(
                runtime,
                pool,
                action->object->size_bytes,
                pool->minimum_alignment,
                action->task_id,
                &action->destination_lease
            );
        }
    } else {
        action->destination_lease =
            shadowspill_memory_pool_acquire_lease_record_locked(
                runtime, pool, action->task_id
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
                shadowspill_memory_pool_try_recycle_lease_record_locked(
                    action->destination_lease
                );
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
        action->destination_lease->bound_object = action->object;
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
        ? action->plan_owner->execution_pool
        : action->plan_owner->spill_pool;
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
         * destination has one compatible range. Keep reservation priority,
         * release the pool lock so the worker can reclaim those ranges, and
         * actively poll the monotonic capacity epoch. Neither thread sleeps.
         */
        const uint64_t capacity_epoch = atomic_load_explicit(
            &pool->capacity_epoch, memory_order_acquire
        );
        shadowspill_memory_pool_unlock_reservation(pool);
        status = SHADOWSPILL_RUNTIME_OK;
        while (atomic_load_explicit(
                   &pool->capacity_epoch, memory_order_acquire
               ) == capacity_epoch) {
            status = shadowspill_failure_status(runtime);
            if (status != SHADOWSPILL_RUNTIME_OK ||
                atomic_load_explicit(
                    &runtime->worker_stop, memory_order_acquire
                ) != 0U) {
                break;
            }
            shadowspill_cpu_relax();
        }
        while (!shadowspill_memory_pool_try_lock_reservation(pool)) {
            shadowspill_cpu_relax();
        }
        if (status == SHADOWSPILL_RUNTIME_OK && atomic_load_explicit(
                &runtime->worker_stop, memory_order_acquire
            ) != 0U) {
            status = SHADOWSPILL_RUNTIME_CLOSED;
        }
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
    const ShadowSpillTaskRecord *record,
    uint64_t *failure_object_id,
    uint64_t *failure_allocation_id
) {
    for (uint32_t index = 0U; index < record->update_count; ++index) {
        const ShadowSpillTaskUpdate *update = &record->updates[index];
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
        shadowspill_plan_execution_location(
            record->plan_owner, object
        )->version = object->authoritative_version;
        shadowspill_plan_spill_location(
            record->plan_owner, object
        )->current = 0U;
        pthread_mutex_unlock(&object->lock);
    }
    return SHADOWSPILL_RUNTIME_OK;
}

static uint64_t count_task_retirements_locked(
    ShadowSpillRuntime *runtime,
    uint64_t task_id
) {
    uint64_t count = 0U;
    ShadowSpillMemoryPool *pool = shadowspill_current_allocation_pool(runtime);
    if (pool == NULL) {
        return 0U;
    }
    pthread_mutex_lock(&pool->lock);
    for (ShadowSpillMemoryLease *allocation =
             shadowspill_current_task_retirements(runtime);
         allocation != NULL;
         allocation = allocation->task_retirement_next) {
        if (allocation->logical_freed && allocation->pointer != NULL &&
            allocation->release_task_id == task_id &&
            allocation->retirement_events == NULL &&
            allocation->retirement_event == NULL) {
            ++count;
        }
    }
    pthread_mutex_unlock(&pool->lock);
    return count;
}

static ShadowSpillRuntimeStatus record_task_completion_event(
    ShadowSpillRuntime *runtime,
    ShadowSpillBackendStream compute_stream,
    ShadowSpillEventLease **result
) {
    *result = NULL;
    ShadowSpillEventLease *event = NULL;
    ShadowSpillRuntimeStatus status = shadowspill_event_lease_create_locked(
        runtime, &event
    );
    if (status != SHADOWSPILL_RUNTIME_OK ||
        runtime->synchronization.record_event(
            runtime->synchronization.context, event->event, compute_stream
        ) != 0 || shadowspill_completion_submit(
            runtime,
            compute_stream,
            event,
            SHADOWSPILL_RUNTIME_NO_ID,
            SHADOWSPILL_RUNTIME_NO_ID
        ) != SHADOWSPILL_RUNTIME_OK) {
        if (event != NULL) {
            (void)shadowspill_event_lease_release(runtime, event);
        }
        return SHADOWSPILL_RUNTIME_BACKEND_FAILURE;
    }
    *result = event;
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
            (void)shadowspill_object_note_fetch_discarded_locked(
                action->object
            );
        } else {
            ShadowSpillMemoryLease *source =
                shadowspill_plan_execution_location(
                    action->plan_owner, action->object
                )->lease;
            if (source != NULL &&
                source->state == SHADOWSPILL_LEASE_RETIRE_PENDING) {
                (void)shadowspill_memory_pool_cancel_retirement_locked(source);
            }
        }
        ShadowSpillEventLease *trigger_event = action->trigger_event;
        action->trigger_event = NULL;
        ShadowSpillObject *object = action->object;
        const uint64_t task_id = action->task_id;
        const uint64_t object_id = object->object_id;
        const uint64_t allocation_id = object->allocation_id;
        if (action->admitted) {
            if (shadowspill_object_reset_admitted_action_locked(
                    object, action
                ) != 0) {
                shadowspill_latch_task_failure(
                    runtime,
                    SHADOWSPILL_RUNTIME_INVALID_STATE,
                    action->task_id,
                    object->object_id,
                    object->allocation_id,
                    0U
                );
            }
            pthread_mutex_unlock(&object->lock);
        } else {
            pthread_mutex_unlock(&object->lock);
        }
        if (shadowspill_event_lease_release(runtime, trigger_event) != 0) {
            shadowspill_latch_task_failure(
                runtime,
                SHADOWSPILL_RUNTIME_BACKEND_FAILURE,
                task_id,
                object_id,
                allocation_id,
                0U
            );
        }
        if (!action->admitted) {
            shadowspill_object_release(object);
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
    const ShadowSpillTaskRecord *record,
    ShadowSpillEventLease *task_completion_event,
    ShadowSpillActionBatch *batch,
    uint64_t *failure_object_id,
    uint64_t *failure_allocation_id
) {
    for (uint32_t index = 0U; index < record->action_count; ++index) {
        const ShadowSpillTaskAction *action = &record->actions[index];
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
                object, queued
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
             (shadowspill_plan_spill_location(
                  record->plan_owner, object
              )->lease == NULL ||
              !object->retain_spill_copy));
        queued->active = 1U;
        queued->activation_generation =
            shadowspill_current_task_invocation(runtime);
        queued->state = SHADOWSPILL_ACTION_QUEUED;
        queued->trigger_event = task_completion_event;
        shadowspill_event_lease_retain(task_completion_event);
        if (batch->tail == NULL) {
            batch->head = queued;
        } else {
            queued->previous = batch->tail;
            batch->tail->next = queued;
        }
        batch->tail = queued;
        if (action->kind == SHADOWSPILL_RUNTIME_PREFETCH) {
            shadowspill_object_note_fetch_queued_locked(object);
        } else {
            ShadowSpillMemoryLease *source =
                shadowspill_plan_execution_location(
                    record->plan_owner, object
                )->lease;
            if (source == NULL) {
                pthread_mutex_unlock(&object->lock);
                return SHADOWSPILL_RUNTIME_INVALID_STATE;
            }
            const int keeps_lease_for_handoff =
                action->kind == SHADOWSPILL_RUNTIME_RELEASE &&
                queued->handoff_lease == source &&
                queued->handoff_generation == source->generation;
            if (!keeps_lease_for_handoff) {
                pthread_mutex_lock(
                    &record->plan_owner->execution_pool->lock
                );
                const int retirement_status =
                    shadowspill_memory_pool_begin_retirement_locked(
                        source,
                        action->kind == SHADOWSPILL_RUNTIME_RELEASE
                            ? task_completion_event
                            : NULL,
                        action->kind == SHADOWSPILL_RUNTIME_OFFLOAD
                    );
                pthread_mutex_unlock(
                    &record->plan_owner->execution_pool->lock
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
    ShadowSpillEventLease *task_completion_event
) {
    ShadowSpillRuntimeStatus status = SHADOWSPILL_RUNTIME_OK;
    ShadowSpillMemoryPool *pool = shadowspill_current_allocation_pool(runtime);
    if (pool == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    pthread_mutex_lock(&pool->lock);
    for (ShadowSpillMemoryLease *allocation =
             shadowspill_current_task_retirements(runtime);
         allocation != NULL;
         allocation = allocation->task_retirement_next) {
        if (!allocation->logical_freed || allocation->pointer == NULL ||
            allocation->release_task_id != task_id ||
            allocation->retirement_events != NULL ||
            allocation->retirement_event != NULL) {
            continue;
        }
        if (shadowspill_memory_pool_publish_retirement_dependency_locked(
                allocation, task_completion_event
            ) != 0) {
            status = SHADOWSPILL_RUNTIME_INVALID_STATE;
            break;
        }
        allocation->retirement_event = task_completion_event;
        shadowspill_event_lease_retain(task_completion_event);
        status = shadowspill_retirement_enqueue_locked(runtime, allocation);
        if (status != SHADOWSPILL_RUNTIME_OK) {
            break;
        }
    }
    pthread_mutex_unlock(&pool->lock);
    return status;
}

static void publish_action_batch_locked(
    ShadowSpillRuntime *runtime,
    const ShadowSpillTaskRecord *record,
    ShadowSpillActionBatch *batch
) {
    if (batch->head == NULL) {
        return;
    }
    const uint64_t published_count = atomic_load_explicit(
        &runtime->actions.count, memory_order_acquire
    ) + record->action_count;
    for (ShadowSpillQueuedAction *queued = batch->head; queued != NULL;
         queued = queued->next) {
        /* Complete dispatcher bookkeeping before the worker can see it. */
        if (queued->kind == SHADOWSPILL_RUNTIME_RELEASE ||
            queued->kind == SHADOWSPILL_RUNTIME_OFFLOAD) {
            (void)atomic_fetch_add_explicit(
                &runtime->pending_capacity_actions, 1U, memory_order_release
            );
            (void)atomic_fetch_add_explicit(
                &record->plan_owner->execution_pool->pending_capacity_actions,
                1U,
                memory_order_release
            );
        }
        pthread_mutex_lock(&queued->object->lock);
        const uint64_t object_id = queued->object->object_id;
        const uint64_t allocation_id = queued->object->allocation_id;
        const uint64_t size_bytes = queued->object->size_bytes;
        pthread_mutex_unlock(&queued->object->lock);
        shadowspill_append_trace_event_locked(
            runtime,
            SHADOWSPILL_TRACE_ACTION_QUEUED,
            record->task_id,
            object_id,
            allocation_id,
            size_bytes,
            queued->kind,
            published_count
        );
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
    /* Publishing the action count is the final mutable batch operation. */
    *batch = (ShadowSpillActionBatch){0};
}

static ShadowSpillRuntimeStatus await_worker_submission(
    ShadowSpillRuntime *runtime,
    const ShadowSpillTaskRecord *record
) {
    if (record->action_count == 0U) {
        return SHADOWSPILL_RUNTIME_OK;
    }
    ShadowSpillTaskRecord *mutable_record =
        (ShadowSpillTaskRecord *)record;
    const uint64_t sequence = atomic_fetch_add_explicit(
        &runtime->next_worker_submission_sequence, 1U, memory_order_acq_rel
    ) + 1U;
    atomic_store_explicit(
        &mutable_record->submission_invocation,
        shadowspill_current_task_invocation(runtime),
        memory_order_relaxed
    );
    atomic_store_explicit(
        &mutable_record->submission_sequence, sequence, memory_order_release
    );

    ShadowSpillTaskRecord *expected = NULL;
    while (!atomic_compare_exchange_weak_explicit(
        &runtime->worker_submission,
        &expected,
        mutable_record,
        memory_order_release,
        memory_order_acquire
    )) {
        expected = NULL;
        const ShadowSpillRuntimeStatus status =
            shadowspill_failure_status(runtime);
        if (status != SHADOWSPILL_RUNTIME_OK) {
            return status;
        }
        if (atomic_load_explicit(
                &runtime->worker_stop, memory_order_acquire
            ) != 0U) {
            return SHADOWSPILL_RUNTIME_CLOSED;
        }
        shadowspill_cpu_relax();
    }

    while (atomic_load_explicit(
        &mutable_record->acknowledgement_sequence, memory_order_acquire
    ) < sequence) {
        const ShadowSpillRuntimeStatus status =
            shadowspill_failure_status(runtime);
        if (status != SHADOWSPILL_RUNTIME_OK) {
            return status;
        }
        if (atomic_load_explicit(
                &runtime->worker_stop, memory_order_acquire
            ) != 0U) {
            return SHADOWSPILL_RUNTIME_CLOSED;
        }
        shadowspill_cpu_relax();
    }
    return SHADOWSPILL_RUNTIME_OK;
}

ShadowSpillRuntimeStatus shadowspill_after_task_record(
    ShadowSpillRuntime *runtime,
    const ShadowSpillTaskRecord *record,
    ShadowSpillBackendStream compute_stream
) {
    ShadowSpillRuntimeStatus status = shadowspill_current_status_locked(runtime);
    ShadowSpillEventLease *task_completion_event = NULL;
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
    const uint64_t retirement_count = status == SHADOWSPILL_RUNTIME_OK
        ? count_task_retirements_locked(runtime, record->task_id)
        : 0U;
    if (status == SHADOWSPILL_RUNTIME_OK &&
        (record->action_count != 0U || retirement_count != 0U)) {
        status = record_task_completion_event(
            runtime, compute_stream, &task_completion_event
        );
        if (status == SHADOWSPILL_RUNTIME_OK) {
            status = attach_task_retirements_locked(
                runtime, record->task_id, task_completion_event
            );
        }
        if (status == SHADOWSPILL_RUNTIME_OK) {
            status = instantiate_actions_locked(
                runtime,
                record,
                task_completion_event,
                &batch,
                &failure_object_id,
                &failure_allocation_id
            );
        }
        if (status == SHADOWSPILL_RUNTIME_OK) {
            publish_action_batch_locked(runtime, record, &batch);
            shadowspill_notify_worker(runtime);
            status = await_worker_submission(runtime, record);
        }
    }
    if (status != SHADOWSPILL_RUNTIME_OK) {
        if (task_completion_event != NULL) {
            discard_action_batch_locked(runtime, &batch);
        }
        shadowspill_task_clear_pending_handoffs(record);
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
        const ShadowSpillRuntimeStatus retirement_status =
            shadowspill_publish_task_retirement_event(
                runtime, record->task_id, compute_stream
            );
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
    if (task_completion_event != NULL && shadowspill_event_lease_release(
            runtime, task_completion_event
        ) != 0) {
        shadowspill_latch_failure_locked(
            runtime,
            SHADOWSPILL_RUNTIME_BACKEND_FAILURE,
            failure_object_id,
            failure_allocation_id,
            0U
        );
        if (status == SHADOWSPILL_RUNTIME_OK) {
            status = SHADOWSPILL_RUNTIME_BACKEND_FAILURE;
        }
    }
    if (record->boundary_kind == SHADOWSPILL_BOUNDARY_TASK) {
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
    }
    shadowspill_leave_task_scope(runtime);
    return status;
}
