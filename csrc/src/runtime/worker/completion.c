/* What a finished action commits, unlinks and releases. */
#define _GNU_SOURCE

#include "internal.h"

#include <stdint.h>
#include <stdlib.h>

static void unlink_action_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillQueuedAction *action
) {
    if (action->previous == NULL) {
        runtime->actions.head = action->next;
    } else {
        action->previous->next = action->next;
    }
    if (action->next != NULL) {
        action->next->previous = action->previous;
    }
    if (runtime->actions.tail == action) {
        runtime->actions.tail = action->previous;
    }
    action->previous = NULL;
    action->next = NULL;
}

void shadowspill_action_latch_failure(
    ShadowSpillRuntime *runtime,
    const ShadowSpillQueuedAction *action,
    ShadowSpillStatus status,
    ShadowSpillFailureReason reason,
    uint64_t object_id,
    uint64_t allocation_id,
    uint64_t requested_bytes
) {
    shadowspill_latch_task_failure(
        runtime,
        status,
        reason,
        action->task_id,
        object_id,
        allocation_id,
        requested_bytes
    );
}

void shadowspill_action_complete(
    ShadowSpillRuntime *runtime,
    ShadowSpillQueuedAction *action
) {
    ShadowSpillPlan *plan_owner = action->plan_owner;
    ShadowSpillObject *object = action->object;
    pthread_mutex_lock(&object->lock);
    if (shadowspill_object_remove_action_locked(
            object, action
        ) != 0) {
        pthread_mutex_unlock(&object->lock);
        shadowspill_action_latch_failure(
            runtime,
            action,
            SHADOWSPILL_STATUS_INVALID_STATE,
            SHADOWSPILL_FAILURE_REASON_OBJECT_STATE_REJECTED,
            object->object_id,
            object->allocation_id,
            0U
        );
        return;
    }
    action->state = SHADOWSPILL_ACTION_FINISHED;
    action->completed_generation = action->activation_generation;
    /*
     * Detach every per-invocation event while the action's object lock still
     * protects its published fields, so no concurrent reader can retain or
     * query an event through a field that outlives its release.
     */
    ShadowSpillEventLease *trigger_event = action->trigger_event;
    ShadowSpillEventLease *completion_event = action->completion_event;
    ShadowSpillEventLease *dependency_event = action->dependency_event;
    const uint8_t had_completion_event = action->has_completion_event;
    action->trigger_event = NULL;
    action->completion_event = NULL;
    action->dependency_event = NULL;
    action->has_completion_event = 0U;
    /* Whatever the lane kept for this transfer is released by the one query
       below; reaching here without having made it means nothing was kept. */
    action->lane_handle = 0U;
    const uint64_t task_id = action->task_id;
    const uint64_t object_id = object->object_id;
    const uint64_t allocation_id = object->allocation_id;
    const uint8_t kind = action->kind;
    const uint8_t admitted = action->admitted;
    ShadowSpillMemoryLease *caller_handoff_lease =
        action->caller_handoff_lease;
    action->caller_handoff_lease = NULL;
    action->caller_handoff_generation = 0U;
    pthread_mutex_unlock(&object->lock);
    pthread_mutex_lock(&runtime->actions.lock);
    unlink_action_locked(runtime, action);
    pthread_mutex_unlock(&runtime->actions.lock);
    if (kind == SHADOWSPILL_RUNTIME_RELEASE ||
        kind == SHADOWSPILL_RUNTIME_EVICT) {
        (void)atomic_fetch_sub_explicit(
            &runtime->pending_capacity_actions, 1U, memory_order_release
        );
        if (action->plan_owner != NULL) {
            (void)atomic_fetch_sub_explicit(
                &action->plan_owner->execution_pool->pending_capacity_actions,
                1U,
                memory_order_release
            );
        }
    }
    if (admitted) {
        pthread_mutex_lock(&object->lock);
        const int reset_status =
            shadowspill_object_reset_admitted_action_locked(object, action);
        pthread_mutex_unlock(&object->lock);
        if (reset_status != 0) {
            shadowspill_action_latch_failure(
                runtime,
                action,
                SHADOWSPILL_STATUS_INVALID_STATE,
                SHADOWSPILL_FAILURE_REASON_OBJECT_STATE_REJECTED,
                object->object_id,
                object->allocation_id,
                0U
            );
        }
    }
    const int trigger_release_failed = shadowspill_event_lease_release(
        runtime, trigger_event
    ) != 0;
    const int completion_release_failed = had_completion_event &&
        shadowspill_event_lease_release(runtime, completion_event) != 0;
    const int dependency_release_failed = dependency_event != NULL &&
        shadowspill_event_lease_release(runtime, dependency_event) != 0;
    if (trigger_release_failed || completion_release_failed ||
        dependency_release_failed) {
        shadowspill_latch_task_failure(
            runtime,
            SHADOWSPILL_STATUS_BACKEND_FAILURE,
            SHADOWSPILL_FAILURE_REASON_EVENT_RELEASE_REJECTED,
            task_id,
            object_id,
            allocation_id,
            0U
        );
    }
    if (!admitted) {
        shadowspill_object_release(object);
        if (action->owns_trace_label) {
            free((void *)action->trace_label);
        }
        free(action);
    }
    shadowspill_memory_lease_release(caller_handoff_lease);
    if (plan_owner != NULL) {
        (void)atomic_fetch_sub_explicit(
            &plan_owner->pending_actions, 1U, memory_order_release
        );
    }
    const uint64_t previous_actions = atomic_fetch_sub_explicit(
        &runtime->actions.count, 1U, memory_order_release
    );
    if (previous_actions == 1U) {
        shadowspill_idle_notify(runtime);
    }
}

static int event_complete_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillEventLease *event,
    uint64_t object_id,
    int *complete
) {
    (void)runtime;
    (void)object_id;
    *complete = atomic_load_explicit(
        &event->backend_complete, memory_order_acquire
    ) != 0U;
    return 0;
}


/*
 * The transfer is in flight: commit it if the backend says the copy is done,
 * and otherwise leave it exactly as it was for the next pass.
 */
int shadowspill_action_finish_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillQueuedAction *action
) {
    ShadowSpillObject *object = action->object;
    ShadowSpillTransferQueue *queue =
        shadowspill_transfer_queue_for_action(runtime, action);
    /*
     * The backend may have completed a whole FIFO prefix while an earlier
     * action's object lock is briefly unavailable.  Commit that
     * prefix strictly from the queue head: skipping a busy predecessor
     * must never let a later transfer publish residency first.
     */
    if (!shadowspill_transfer_queue_is_inflight_head(queue, action)) {
        pthread_mutex_unlock(&object->lock);
        return 0;
    }
    int complete = 0;
    if (event_complete_locked(
            runtime, action->completion_event,
            object->object_id, &complete
        ) != 0) {
        pthread_mutex_unlock(&object->lock);
        return -1;
    }
    if (complete) {
        ShadowSpillEventLease *readiness_to_release = NULL;
        ShadowSpillMemoryPool *release_pool = NULL;
        const int caller_handoff =
            action->caller_handoff_lease != NULL;
        if (caller_handoff &&
            action->caller_handoff_lease->generation !=
                action->caller_handoff_generation) {
            shadowspill_action_latch_failure(
                runtime,
                action,
                SHADOWSPILL_STATUS_INVALID_STATE,
                SHADOWSPILL_FAILURE_REASON_OBJECT_STATE_REJECTED,
                object->object_id,
                action->caller_handoff_lease->allocation_id,
                object->size_bytes
            );
            pthread_mutex_unlock(&object->lock);
            return -1;
        }
        if (action->kind == SHADOWSPILL_RUNTIME_EVICT) {
            release_pool = action->plan_owner->execution_pool;
        } else if (action->kind == SHADOWSPILL_RUNTIME_FETCH &&
                   (caller_handoff || !object->retain_spill_copy)) {
            release_pool = action->plan_owner->spill_pool;
        }
        if (release_pool != NULL &&
            !shadowspill_memory_pool_try_lock_reclamation(
                release_pool
            )) {
            pthread_mutex_unlock(&object->lock);
            return 0;
        }
        if (action->kind == SHADOWSPILL_RUNTIME_EVICT) {
            ShadowSpillMemoryLease *allocation =
                shadowspill_find_lease(
                    action->plan_owner->execution_pool,
                    object->allocation_id
                );
            if (allocation == NULL) {
                shadowspill_memory_pool_unlock_reclamation(
                    action->plan_owner->execution_pool
                );
                shadowspill_action_latch_failure(
                    runtime,
                    action,
                    SHADOWSPILL_STATUS_INVALID_STATE,
                    SHADOWSPILL_FAILURE_REASON_OBJECT_STATE_REJECTED,
                    object->object_id,
                    object->allocation_id,
                    0U
                );
                pthread_mutex_unlock(&object->lock);
                return -1;
            }
            object->retired_generation = object->generation;
            object->retired_execution_pointer = allocation->pointer;
            allocation->release_task_id = action->task_id;
            shadowspill_release_lease_locked(runtime, allocation);
            shadowspill_memory_pool_unlock_reclamation(
                action->plan_owner->execution_pool
            );
            object->allocation_id = SHADOWSPILL_RUNTIME_NO_ID;
            ShadowSpillObjectLocation *execution =
                shadowspill_plan_execution_location(
                    action->plan_owner, object
                );
            ShadowSpillObjectLocation *spill =
                shadowspill_plan_spill_location(
                    action->plan_owner, object
                );
            execution->lease = NULL;
            execution->current = 0U;
            spill->current = 1U;
            spill->version = object->authoritative_version;
            object->residency = SHADOWSPILL_OBJECT_SPILL_ONLY;
        } else if (action->kind == SHADOWSPILL_RUNTIME_WRITE_BACK) {
            /* The copy carries the version the action was scheduled
             * at; the execution copy stays, and stays authoritative. */
            ShadowSpillObjectLocation *spill =
                shadowspill_plan_spill_location(
                    action->plan_owner, object
                );
            spill->current = 1U;
            spill->version = action->scheduled_version;
        } else {
            ShadowSpillObjectLocation *execution =
                shadowspill_plan_execution_location(
                    action->plan_owner, object
                );
            if (execution->lease != NULL &&
                execution->lease->generation == object->generation) {
                object->residency = SHADOWSPILL_OBJECT_EXECUTION_READY;
                execution->current = 1U;
            }
            /*
             * The compute stream may already have waited on this
             * transfer and launched a task.  In that case
             * after_task has advanced execution_version while the
             * worker still observes FETCHING. The fetch
             * completion only changes readiness; it must not roll
             * the execution version back to the copied spill version.
             */
            object->has_readiness_event = 0U;
            readiness_to_release = object->readiness_event;
            object->readiness_event = NULL;
            if (caller_handoff || !object->retain_spill_copy) {
                ShadowSpillObjectLocation *spill =
                    shadowspill_plan_spill_location(
                        action->plan_owner, object
                    );
                ShadowSpillMemoryLease *lease = spill->lease;
                const int range_status =
                    shadowspill_memory_pool_release_lease_locked(lease);
                if (range_status == 0) {
                    shadowspill_memory_pool_try_recycle_lease_record_locked(
                        lease
                    );
                }
                shadowspill_memory_pool_unlock_reclamation(
                    action->plan_owner->spill_pool
                );
                if (range_status != 0) {
                    shadowspill_action_latch_failure(
                        runtime,
                        action,
                        SHADOWSPILL_STATUS_INTERNAL_FAILURE,
                        SHADOWSPILL_FAILURE_REASON_OBJECT_STATE_REJECTED,
                        object->object_id,
                        object->allocation_id,
                        object->size_bytes
                    );
                    pthread_mutex_unlock(&object->lock);
                    return -1;
                }
                spill->lease = NULL;
                spill->owns_lease = 0U;
                spill->current = 0U;
            }
        }
        /*
         * What the lane says this transfer did. Asked once, after its event
         * has completed, and only for a handle `copy` kept -- the query is
         * what retires it, so there is nothing to release afterwards.
         *
         * A lane that reports no instants leaves them unset, and the trace
         * records the transfer with no times rather than with wrong ones.
         */
        uint64_t lane_issued_at_ns = SHADOWSPILL_TRACE_NO_STREAM_TIME;
        uint64_t lane_started_at_ns = SHADOWSPILL_TRACE_NO_STREAM_TIME;
        uint64_t lane_finished_at_ns = SHADOWSPILL_TRACE_NO_STREAM_TIME;
        ShadowSpillLaneTransfer moved = {
            .issued_at_nanoseconds = SHADOWSPILL_LANE_NO_TIME,
            .started_at_nanoseconds = SHADOWSPILL_LANE_NO_TIME,
            .finished_at_nanoseconds = SHADOWSPILL_LANE_NO_TIME,
            .bytes = 0U,
            .chunks = 0U,
        };
        const ShadowSpillRouteState *const lane_route = action->route;
        if (action->lane_handle != 0U && lane_route != NULL &&
            lane_route->operations->transfer != NULL &&
            lane_route->operations->transfer(
                lane_route->lane, action->lane_handle, &moved
            ) == 0) {
            if (moved.issued_at_nanoseconds != SHADOWSPILL_LANE_NO_TIME) {
                lane_issued_at_ns = moved.issued_at_nanoseconds;
            }
            if (moved.started_at_nanoseconds != SHADOWSPILL_LANE_NO_TIME) {
                lane_started_at_ns = moved.started_at_nanoseconds;
            }
            if (moved.finished_at_nanoseconds != SHADOWSPILL_LANE_NO_TIME) {
                lane_finished_at_ns = moved.finished_at_nanoseconds;
            }
        }
        action->lane_handle = 0U;
        shadowspill_append_stamped_trace_event_locked(
            runtime,
            SHADOWSPILL_TRACE_TRANSFER_COMPLETED,
            action->task_id,
            object->object_id,
            caller_handoff
                ? action->caller_handoff_lease->allocation_id
                : object->allocation_id,
            object->size_bytes,
            action->kind == SHADOWSPILL_RUNTIME_FETCH
                ? SHADOWSPILL_TRANSFER_FETCH
                : SHADOWSPILL_TRANSFER_EVICT,
            atomic_load_explicit(
                &runtime->actions.count, memory_order_acquire
            ),
            lane_issued_at_ns,
            lane_started_at_ns,
            lane_finished_at_ns
        );
        pthread_mutex_unlock(&object->lock);
        if (readiness_to_release != NULL &&
            shadowspill_event_lease_release(
                runtime, readiness_to_release
            ) != 0) {
            shadowspill_action_latch_failure(
                runtime,
                action,
                SHADOWSPILL_STATUS_BACKEND_FAILURE,
                SHADOWSPILL_FAILURE_REASON_BACKEND_CALL_REJECTED,
                object->object_id,
                object->allocation_id,
                0U
            );
            return -1;
        }
        if (shadowspill_transfer_queue_complete(queue, action) != 0) {
            shadowspill_action_latch_failure(
                runtime,
                action,
                SHADOWSPILL_STATUS_INVALID_STATE,
                SHADOWSPILL_FAILURE_REASON_OBJECT_STATE_REJECTED,
                object->object_id,
                object->allocation_id,
                0U
            );
            return -1;
        }
        shadowspill_action_complete(runtime, action);
        return 2;
    }
    pthread_mutex_unlock(&object->lock);
    return 0;
}
