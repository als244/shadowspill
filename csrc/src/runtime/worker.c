#define _GNU_SOURCE

#include "internal.h"

#include <stdint.h>
#include <stdlib.h>

static int submit_transfer_copy(
    ShadowSpillRuntime *runtime,
    const ShadowSpillQueuedAction *action,
    const ShadowSpillRouteState *route,
    void *destination,
    const void *source,
    uint64_t bytes,
    ShadowSpillBackendStream stream
) {
    const char *fallback = action->kind == SHADOWSPILL_RUNTIME_FETCH
        ? "shadowspill.runtime.transfer.fetch.unlabeled"
        : "shadowspill.runtime.transfer.evict.unlabeled";
    const ShadowSpillProfilerRange range = shadowspill_profiler_range_begin(
        &runtime->backend,
        action->trace_label == NULL ? fallback : action->trace_label
    );
    const int status = shadowspill_route_copy_async(
        runtime, route, destination, source, bytes, stream
    );
    shadowspill_profiler_range_end(&runtime->backend, range);
    return status;
}

/*
 * A traced transfer is measured on its lane by the worker that dispatches
 * it: the interval opens just before the copy and closes just after it,
 * ahead of the completion event, so observing the completion guarantees the
 * interval is readable. An untraced step pays the acquire load and nothing
 * else. An interval that cannot be measured is a gap in the trace, never a
 * transfer failure.
 */
static int submit_traced_copy(
    ShadowSpillRuntime *runtime,
    ShadowSpillQueuedAction *action,
    const ShadowSpillRouteState *route,
    void *source,
    void *destination,
    uint64_t bytes,
    ShadowSpillBackendStream lane
) {
    const int traced =
        atomic_load_explicit(&runtime->trace_active, memory_order_acquire) != 0U &&
        runtime->trace_origin_present;
    if (traced) {
        (void)shadowspill_stream_interval_open(
            runtime, &action->stream_interval, lane
        );
    }
    if (submit_transfer_copy(
            runtime, action, route, source, destination, bytes, lane
        ) != 0) {
        shadowspill_stream_interval_discard(runtime, &action->stream_interval);
        return -1;
    }
    if (traced) {
        (void)shadowspill_stream_interval_close(
            runtime, &action->stream_interval, lane
        );
    }
    return 0;
}

static ShadowSpillRouteState *route_for_action(
    const ShadowSpillQueuedAction *action
) {
    return action == NULL ? NULL : action->route;
}

void shadowspill_notify_worker(ShadowSpillRuntime *runtime) {
    (void)runtime;
}

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

static void latch_action_failure(
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

static void complete_action(
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
        latch_action_failure(
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
     * protects its published fields.  The old implementation released the
     * trigger fence first and cleared the pointer later, leaving a window in
     * which another reader could retain or query freed storage.
     */
    ShadowSpillEventLease *trigger_event = action->trigger_event;
    ShadowSpillEventLease *completion_event = action->completion_event;
    ShadowSpillEventLease *dependency_event = action->dependency_event;
    const uint8_t had_completion_event = action->has_completion_event;
    action->trigger_event = NULL;
    action->completion_event = NULL;
    action->dependency_event = NULL;
    action->has_completion_event = 0U;
    shadowspill_stream_interval_discard(runtime, &action->stream_interval);
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
            latch_action_failure(
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

static void release_action_claim(
    ShadowSpillRuntime *runtime,
    ShadowSpillQueuedAction *action
) {
    pthread_mutex_lock(&runtime->actions.lock);
    action->processing = 0U;
    pthread_mutex_unlock(&runtime->actions.lock);
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

static int destination_dependency_is_published(
    ShadowSpillQueuedAction *action
) {
    const int fixed_ready =
        shadowspill_fixed_layout_dependencies_published(
            action->plan_owner,
            SHADOWSPILL_FIXED_ACTION_DESTINATION,
            action->task_id,
            action->action_ordinal,
            action->activation_generation
        );
    if (fixed_ready <= 0) {
        return fixed_ready;
    }
    ShadowSpillMemoryLease *lease = action->destination_lease;
    if (lease == NULL) {
        return 1;
    }
    if (lease->pool == NULL) {
        return 0;
    }
    ShadowSpillMemoryPool *pool = lease->pool;
    if (!shadowspill_memory_pool_try_lock_reservation(pool)) {
        return 0;
    }
    const int ready = lease->state != SHADOWSPILL_LEASE_SUCCESSOR_RESERVED ||
        (lease->causal_predecessor != NULL &&
         lease->causal_predecessor->causal_event != NULL);
    shadowspill_memory_pool_unlock_reservation(pool);
    return ready;
}

static int acquire_reserved_destination(
    ShadowSpillRuntime *runtime,
    ShadowSpillQueuedAction *action
) {
    ShadowSpillMemoryLease *lease = action->destination_lease;
    if (lease == NULL || lease->pool == NULL) {
        return -1;
    }
    ShadowSpillMemoryPool *pool = lease->pool;
    shadowspill_memory_pool_lock_reservation(pool);
    ShadowSpillEventLease *dependency_event = NULL;
    const int status = action->kind == SHADOWSPILL_RUNTIME_FETCH
        ? shadowspill_acquire_reserved_execution_lease_locked(
            runtime, lease, &dependency_event
        )
        : shadowspill_memory_pool_acquire_reserved_lease_locked(
            lease, &dependency_event
        );
    shadowspill_memory_pool_unlock_reservation(pool);
    shadowspill_memory_pool_relinquish_reservation(pool);
    if (status != 0) {
        if (dependency_event != NULL) {
            (void)shadowspill_event_lease_release(runtime, dependency_event);
        }
        return status;
    }
    if (dependency_event != NULL) {
        ShadowSpillRouteState *route = route_for_action(action);
        if (route == NULL || runtime->backend.wait_event(
                runtime->backend.state,
                route->lane,
                dependency_event->event
            ) != 0) {
            (void)shadowspill_event_lease_release(runtime, dependency_event);
            return -1;
        }
        action->dependency_event = dependency_event;
    }
    return status;
}

static int dispatch_evict_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillQueuedAction *action
) {
    ShadowSpillObject *object = action->object;
    if (object->residency == SHADOWSPILL_OBJECT_FETCHING) {
        return 0;
    }
    ShadowSpillObjectLocation *execution = shadowspill_plan_execution_location(
        action->plan_owner, object
    );
    ShadowSpillObjectLocation *spill = shadowspill_plan_spill_location(
        action->plan_owner, object
    );
    ShadowSpillMemoryLease *allocation = execution->lease;
    if (allocation == NULL || allocation->pointer == NULL ||
        allocation->allocation_id != object->allocation_id ||
        allocation->generation != object->generation) {
        latch_action_failure(
            runtime,
            action,
            SHADOWSPILL_STATUS_INVALID_STATE,
            SHADOWSPILL_FAILURE_REASON_OBJECT_STATE_REJECTED,
            object->object_id,
            object->allocation_id,
            object->size_bytes
        );
        return -1;
    }
    const uint64_t object_id = object->object_id;
    const uint64_t object_generation = object->generation;
    const uint64_t allocation_id = allocation->allocation_id;
    const uint64_t bytes = object->size_bytes;
    void *execution_pointer = allocation->pointer;
    int spill_lease_created = 0;
    if (spill->lease == NULL) {
        if (action->destination_lease == NULL) {
            latch_action_failure(
                runtime,
                action,
                SHADOWSPILL_STATUS_INVALID_STATE,
                SHADOWSPILL_FAILURE_REASON_OBJECT_STATE_REJECTED,
                object->object_id,
                object->allocation_id,
                object->size_bytes
            );
            return -1;
        }
        if (acquire_reserved_destination(runtime, action) != 0) {
            latch_action_failure(
                runtime,
                action,
                SHADOWSPILL_STATUS_INVALID_STATE,
                SHADOWSPILL_FAILURE_REASON_OBJECT_STATE_REJECTED,
                object->object_id,
                object->allocation_id,
                object->size_bytes
            );
            return -1;
        }
        spill->lease = action->destination_lease;
        spill->owns_lease = 1U;
        action->destination_lease = NULL;
        spill_lease_created = 1;
    }
    ShadowSpillMemoryLease *spill_lease = spill->lease;
    ShadowSpillEventLease *trigger_event = action->trigger_event;
    if (trigger_event == NULL) {
        latch_action_failure(
            runtime,
            action,
            SHADOWSPILL_STATUS_INVALID_STATE,
            SHADOWSPILL_FAILURE_REASON_OBJECT_STATE_REJECTED,
            object_id,
            allocation_id,
            bytes
        );
        return -1;
    }
    shadowspill_event_lease_retain(trigger_event);
    pthread_mutex_unlock(&object->lock);

    ShadowSpillEventLease *completion_event = NULL;
    ShadowSpillRouteState *route = route_for_action(action);
    ShadowSpillStatus event_status = shadowspill_event_lease_create_locked(
        runtime, &completion_event
    );
    int backend_failed = event_status != SHADOWSPILL_STATUS_OK || route == NULL;
    if (!backend_failed && runtime->backend.wait_event(
            runtime->backend.state,
            route->lane,
            trigger_event->event
        ) != 0) {
        backend_failed = 1;
    }
    if (!backend_failed && (submit_traced_copy(
            runtime,
            action,
            route,
            spill_lease->pointer,
            execution_pointer,
            bytes,
            route->lane
        ) != 0 || runtime->backend.record_event(
                runtime->backend.state,
                completion_event->event,
                route->lane
            ) != 0 || shadowspill_completion_submit(
                runtime,
                route->lane,
                completion_event,
                object_id,
                allocation_id
            ) != SHADOWSPILL_STATUS_OK)) {
        backend_failed = 1;
    }
    if (!backend_failed) {
        pthread_mutex_lock(&allocation->pool->lock);
        if (shadowspill_memory_pool_publish_retirement_dependency_locked(
                allocation, completion_event
            ) != 0) {
            backend_failed = 1;
        }
        pthread_mutex_unlock(&allocation->pool->lock);
    }
    if (shadowspill_event_lease_release(runtime, trigger_event) != 0) {
        backend_failed = 1;
    }
    pthread_mutex_lock(&object->lock);
    if (backend_failed || object->generation != object_generation ||
        execution->lease != allocation ||
        object->allocation_id != allocation_id) {
        if (completion_event != NULL) {
            (void)shadowspill_event_lease_release(runtime, completion_event);
        }
        /*
         * Once a causal destination has accepted a predecessor dependency,
         * it cannot safely re-enter the free list on submission failure.
         * Failure latches the runtime; close drains the lane and tears down
         * the owning pool without exposing this range to another allocation.
         */
        (void)spill_lease_created;
        latch_action_failure(
            runtime,
            action,
            backend_failed ? SHADOWSPILL_STATUS_BACKEND_FAILURE
                           : SHADOWSPILL_STATUS_INVALID_STATE,
            SHADOWSPILL_FAILURE_REASON_BACKEND_CALL_REJECTED,
            object_id,
            allocation_id,
            bytes
        );
        return -1;
    }
    action->completion_event = completion_event;
    action->has_completion_event = 1U;
    action->state = SHADOWSPILL_ACTION_IN_FLIGHT;
    object->residency = SHADOWSPILL_OBJECT_EVICTING;
    (void)atomic_fetch_add_explicit(
        &runtime->evict_transfers, 1U, memory_order_acq_rel
    );
    (void)atomic_fetch_add_explicit(
        &runtime->bytes_evicted, bytes, memory_order_acq_rel
    );
    shadowspill_append_trace_event_locked(
        runtime,
        SHADOWSPILL_TRACE_TRANSFER_DISPATCHED,
        action->task_id,
        object->object_id,
        allocation_id,
        bytes,
        SHADOWSPILL_TRANSFER_EVICT,
        atomic_load_explicit(&runtime->actions.count, memory_order_acquire)
    );
    return 1;
}



static int dispatch_fetch_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillQueuedAction *action
) {
    if (action->destination_lease == NULL ||
        shadowspill_plan_spill_location(
            action->plan_owner, action->object
        )->lease == NULL) {
        latch_action_failure(
            runtime,
            action,
            SHADOWSPILL_STATUS_INVALID_STATE,
            SHADOWSPILL_FAILURE_REASON_OBJECT_STATE_REJECTED,
            action->object->object_id,
            SHADOWSPILL_RUNTIME_NO_ID,
            action->object->size_bytes
        );
        return -1;
    }
    ShadowSpillObject *object = action->object;
    ShadowSpillObjectLocation *execution = shadowspill_plan_execution_location(
        action->plan_owner, object
    );
    ShadowSpillObjectLocation *spill = shadowspill_plan_spill_location(
        action->plan_owner, object
    );
    const uint64_t object_id = object->object_id;
    const uint64_t previous_generation = object->generation;
    const uint64_t authoritative_version = object->authoritative_version;
    const uint64_t spill_version = spill->version;
    const uint64_t bytes = object->size_bytes;
    ShadowSpillMemoryLease *allocation = action->destination_lease;
    if (acquire_reserved_destination(runtime, action) != 0) {
        latch_action_failure(
            runtime,
            action,
            SHADOWSPILL_STATUS_INVALID_STATE,
            SHADOWSPILL_FAILURE_REASON_OBJECT_STATE_REJECTED,
            object_id,
            allocation->allocation_id,
            bytes
        );
        return -1;
    }
    action->destination_lease = NULL;
    ShadowSpillEventLease *trigger_event = action->trigger_event;
    if (trigger_event == NULL) {
        latch_action_failure(
            runtime,
            action,
            SHADOWSPILL_STATUS_INVALID_STATE,
            SHADOWSPILL_FAILURE_REASON_OBJECT_STATE_REJECTED,
            object_id,
            allocation->allocation_id,
            bytes
        );
        return -1;
    }
    shadowspill_event_lease_retain(trigger_event);
    pthread_mutex_unlock(&object->lock);
    ShadowSpillEventLease *completion_event = NULL;
    ShadowSpillRouteState *route = route_for_action(action);
    ShadowSpillStatus event_status = shadowspill_event_lease_create_locked(
        runtime, &completion_event
    );
    int backend_failed = event_status != SHADOWSPILL_STATUS_OK || route == NULL;
    if (!backend_failed && runtime->backend.wait_event(
            runtime->backend.state,
            route->lane,
            trigger_event->event
        ) != 0) {
        backend_failed = 1;
    }
    if (!backend_failed && (submit_traced_copy(
            runtime,
            action,
            route,
            allocation->pointer,
            spill->lease->pointer,
            bytes,
            route->lane
        ) != 0 || runtime->backend.record_event(
                runtime->backend.state,
                completion_event->event,
                route->lane
            ) != 0 || shadowspill_completion_submit(
                runtime,
                route->lane,
                completion_event,
                object_id,
                allocation->allocation_id
            ) != SHADOWSPILL_STATUS_OK)) {
        backend_failed = 1;
    }
    if (!backend_failed && !object->retain_spill_copy) {
        pthread_mutex_lock(&spill->lease->pool->lock);
        if (shadowspill_memory_pool_begin_retirement_locked(
                spill->lease, completion_event, 0
            ) != 0) {
            backend_failed = 1;
        }
        pthread_mutex_unlock(&spill->lease->pool->lock);
    }
    if (shadowspill_event_lease_release(runtime, trigger_event) != 0) {
        backend_failed = 1;
    }
    pthread_mutex_lock(&object->lock);
    if (backend_failed || object->residency != SHADOWSPILL_OBJECT_SPILL_ONLY ||
        object->generation != previous_generation ||
        object->authoritative_version != authoritative_version ||
        spill->version != spill_version || !spill->current) {
        if (completion_event != NULL) {
            (void)shadowspill_event_lease_release(runtime, completion_event);
        }
        /* Retain the activated range until failure teardown; see evict. */
        latch_action_failure(
            runtime,
            action,
            backend_failed ? SHADOWSPILL_STATUS_BACKEND_FAILURE
                           : SHADOWSPILL_STATUS_INVALID_STATE,
            SHADOWSPILL_FAILURE_REASON_BACKEND_CALL_REJECTED,
            object_id,
            allocation->allocation_id,
            bytes
        );
        return -1;
    }
    action->completion_event = completion_event;
    action->has_completion_event = 1U;
    action->state = SHADOWSPILL_ACTION_IN_FLIGHT;
    object->allocation_id = allocation->allocation_id;
    execution->lease = allocation;
    object->generation = allocation->generation;
    execution->version = spill->version;
    execution->current = 0U;
    object->readiness_event = completion_event;
    shadowspill_event_lease_retain(object->readiness_event);
    object->has_readiness_event = 1U;
    object->residency = SHADOWSPILL_OBJECT_FETCHING;
    if (shadowspill_object_note_fetch_published_locked(object) != 0) {
        latch_action_failure(
            runtime,
            action,
            SHADOWSPILL_STATUS_INVALID_STATE,
            SHADOWSPILL_FAILURE_REASON_OBJECT_STATE_REJECTED,
            object_id,
            allocation->allocation_id,
            bytes
        );
        return -1;
    }
    (void)atomic_fetch_add_explicit(
        &runtime->fetch_transfers, 1U, memory_order_acq_rel
    );
    (void)atomic_fetch_add_explicit(
        &runtime->bytes_fetched, bytes, memory_order_acq_rel
    );
    shadowspill_append_trace_event_locked(
        runtime,
        SHADOWSPILL_TRACE_TRANSFER_DISPATCHED,
        action->task_id,
        object_id,
        allocation->allocation_id,
        bytes,
        SHADOWSPILL_TRANSFER_FETCH,
        atomic_load_explicit(&runtime->actions.count, memory_order_acquire)
    );
    return 1;
}

static int handle_action(
    ShadowSpillRuntime *runtime,
    ShadowSpillQueuedAction *action
) {
    int changed = 0;
    ShadowSpillObject *object = action->object;
    if (pthread_mutex_trylock(&object->lock) != 0) {
        return 0;
    }
    if (!shadowspill_object_action_is_head_locked(object, action)) {
        pthread_mutex_unlock(&object->lock);
        return 0;
    }
    if (action->state == SHADOWSPILL_ACTION_QUEUED) {
            if (action->kind == SHADOWSPILL_RUNTIME_RELEASE) {
                if (action->trigger_event == NULL) {
                    latch_action_failure(
                        runtime,
                        action,
                        SHADOWSPILL_STATUS_BACKEND_FAILURE,
                        SHADOWSPILL_FAILURE_REASON_BACKEND_CALL_REJECTED,
                        object->object_id,
                        object->allocation_id,
                        0U
                    );
                    pthread_mutex_unlock(&object->lock);
                    return -1;
                }
                const int complete = shadowspill_event_lease_is_complete(
                    action->trigger_event
                );
                if (complete) {
                    if (!shadowspill_memory_pool_try_lock_reclamation(
                            action->plan_owner->execution_pool
                        )) {
                        pthread_mutex_unlock(&object->lock);
                        return 0;
                    }
                    ShadowSpillMemoryLease *allocation =
                        shadowspill_find_execution_lease(
                            action->plan_owner->execution_pool,
                            object->allocation_id
                        );
                    if (allocation == NULL) {
                        shadowspill_memory_pool_unlock_reclamation(
                            action->plan_owner->execution_pool
                        );
                        latch_action_failure(
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
                    if (action->handoff_lease != NULL) {
                        if (allocation != action->handoff_lease ||
                            allocation->generation !=
                                action->handoff_generation ||
                            allocation->bound_object == NULL ||
                            allocation->bound_object == object) {
                            shadowspill_memory_pool_unlock_reclamation(
                                action->plan_owner->execution_pool
                            );
                            latch_action_failure(
                                runtime,
                                action,
                                SHADOWSPILL_STATUS_INVALID_STATE,
                                SHADOWSPILL_FAILURE_REASON_OBJECT_STATE_REJECTED,
                                object->object_id,
                                allocation->allocation_id,
                                allocation->requested_bytes
                            );
                            pthread_mutex_unlock(&object->lock);
                            return -1;
                        }
                        object->retired_generation = object->generation;
                        object->retired_execution_pointer = allocation->pointer;
                        object->allocation_id = SHADOWSPILL_RUNTIME_NO_ID;
                        ShadowSpillObjectLocation *execution =
                            shadowspill_plan_execution_location(
                                action->plan_owner, object
                            );
                        const ShadowSpillObjectLocation *spill =
                            shadowspill_plan_spill_location(
                                action->plan_owner, object
                            );
                        execution->lease = NULL;
                        execution->current = 0U;
                        object->residency = spill->current
                            ? SHADOWSPILL_OBJECT_SPILL_ONLY
                            : SHADOWSPILL_OBJECT_RELEASED;
                        shadowspill_memory_pool_unlock_reclamation(
                            action->plan_owner->execution_pool
                        );
                        pthread_mutex_unlock(&object->lock);
                        complete_action(runtime, action);
                        return 2;
                    }
                    if (allocation->bound_object != object) {
                        shadowspill_memory_pool_unlock_reclamation(
                            action->plan_owner->execution_pool
                        );
                        latch_action_failure(
                            runtime,
                            action,
                            SHADOWSPILL_STATUS_INVALID_STATE,
                            SHADOWSPILL_FAILURE_REASON_OBJECT_STATE_REJECTED,
                            object->object_id,
                            allocation->allocation_id,
                            allocation->requested_bytes
                        );
                        pthread_mutex_unlock(&object->lock);
                        return -1;
                    }
                    object->retired_generation = object->generation;
                    object->retired_execution_pointer = allocation->pointer;
                    allocation->release_task_id = action->task_id;
                    shadowspill_release_execution_lease_locked(runtime, allocation);
                    shadowspill_memory_pool_unlock_reclamation(
                        action->plan_owner->execution_pool
                    );
                    object->allocation_id = SHADOWSPILL_RUNTIME_NO_ID;
                    ShadowSpillObjectLocation *execution =
                        shadowspill_plan_execution_location(
                            action->plan_owner, object
                        );
                    const ShadowSpillObjectLocation *spill =
                        shadowspill_plan_spill_location(
                            action->plan_owner, object
                        );
                    execution->lease = NULL;
                    execution->current = 0U;
                    object->residency = spill->current
                        ? SHADOWSPILL_OBJECT_SPILL_ONLY
                        : SHADOWSPILL_OBJECT_RELEASED;
                    pthread_mutex_unlock(&object->lock);
                    complete_action(runtime, action);
                    return 2;
                }
            } else {
                ShadowSpillObjectLocation *spill =
                    shadowspill_plan_spill_location(
                        action->plan_owner, object
                    );
                if ((action->kind == SHADOWSPILL_RUNTIME_FETCH &&
                     action->destination_lease == NULL) ||
                    (action->kind == SHADOWSPILL_RUNTIME_EVICT &&
                     spill->lease == NULL &&
                     action->destination_lease == NULL)) {
                    latch_action_failure(
                        runtime,
                        action,
                        SHADOWSPILL_STATUS_INVALID_STATE,
                        SHADOWSPILL_FAILURE_REASON_OBJECT_STATE_REJECTED,
                        object->object_id,
                        object->allocation_id,
                        object->size_bytes
                    );
                    pthread_mutex_unlock(&object->lock);
                    return -1;
                }
                if (action->kind == SHADOWSPILL_RUNTIME_EVICT &&
                    object->residency == SHADOWSPILL_OBJECT_FETCHING) {
                    pthread_mutex_unlock(&object->lock);
                    return 0;
                }
                /* Capacity is owned; transfer dispatch still obeys the task. */
                if (action->trigger_event == NULL) {
                    latch_action_failure(
                        runtime,
                        action,
                        SHADOWSPILL_STATUS_BACKEND_FAILURE,
                        SHADOWSPILL_FAILURE_REASON_BACKEND_CALL_REJECTED,
                        object->object_id,
                        object->allocation_id,
                        0U
                    );
                    pthread_mutex_unlock(&object->lock);
                    return -1;
                }
                const int trigger_complete =
                    action->kind == SHADOWSPILL_RUNTIME_FETCH ||
                    shadowspill_event_lease_is_complete(action->trigger_event);
                if (!trigger_complete) {
                    pthread_mutex_unlock(&object->lock);
                    return 0;
                }
                /*
                 * A causal destination owns capacity from the trigger, but it
                 * cannot enter a transfer lane until its predecessor has
                 * published the event that makes address reuse safe.  Leave
                 * it queued so it cannot occupy and stall the lane head.
                 */
                pthread_mutex_unlock(&object->lock);
                const int dependency_ready =
                    destination_dependency_is_published(action);
                /*
                 * Insert fixed-range reuse waits without owning the current
                 * object.  A later fetch commonly reuses the range freed by
                 * an earlier eviction of this same object; resolving that
                 * predecessor therefore locks this exact object internally.
                 */
                const ShadowSpillStatus dependency_wait_status =
                    dependency_ready > 0 &&
                        action->kind == SHADOWSPILL_RUNTIME_FETCH
                    ? shadowspill_fixed_layout_insert_dependency_waits(
                          action->plan_owner,
                          SHADOWSPILL_FIXED_ACTION_DESTINATION,
                          action->task_id,
                          action->action_ordinal,
                          action->activation_generation,
                          action->route->lane
                      )
                    : SHADOWSPILL_STATUS_OK;
                pthread_mutex_lock(&object->lock);
                if (dependency_ready < 0) {
                    latch_action_failure(
                        runtime,
                        action,
                        SHADOWSPILL_STATUS_PLAN_VIOLATION,
                        SHADOWSPILL_FAILURE_REASON_OBJECT_STATE_REJECTED,
                        object->object_id,
                        object->allocation_id,
                        object->size_bytes
                    );
                    pthread_mutex_unlock(&object->lock);
                    return -1;
                }
                if (dependency_wait_status != SHADOWSPILL_STATUS_OK) {
                    latch_action_failure(
                        runtime,
                        action,
                        dependency_wait_status,
                        SHADOWSPILL_FAILURE_REASON_BACKEND_CALL_REJECTED,
                        object->object_id,
                        object->allocation_id,
                        object->size_bytes
                    );
                    pthread_mutex_unlock(&object->lock);
                    return -1;
                }
                if (!dependency_ready) {
                    pthread_mutex_unlock(&object->lock);
                    return 0;
                }
                if (!shadowspill_object_action_is_head_locked(object, action) ||
                    action->state != SHADOWSPILL_ACTION_QUEUED) {
                    pthread_mutex_unlock(&object->lock);
                    return 0;
                }
                if (action->kind == SHADOWSPILL_RUNTIME_FETCH) {
                    ShadowSpillObjectLocation *spill =
                        shadowspill_plan_spill_location(
                            action->plan_owner, object
                        );
                    if (object->residency !=
                            SHADOWSPILL_OBJECT_SPILL_ONLY ||
                        spill->lease == NULL || !spill->current ||
                        spill->version != object->authoritative_version) {
                        latch_action_failure(
                            runtime,
                            action,
                            SHADOWSPILL_STATUS_PLAN_VIOLATION,
                            SHADOWSPILL_FAILURE_REASON_OBJECT_STATE_REJECTED,
                            object->object_id,
                            object->allocation_id,
                            object->size_bytes
                        );
                        pthread_mutex_unlock(&object->lock);
                        return -1;
                    }
                }
                ShadowSpillTransferLane *lane =
                    shadowspill_transfer_lane_for_action(runtime, action);
                if (!shadowspill_transfer_lane_claim(lane, action)) {
                    pthread_mutex_unlock(&object->lock);
                    return 0;
                }
                int dispatched = action->kind == SHADOWSPILL_RUNTIME_EVICT
                    ? dispatch_evict_locked(runtime, action)
                    : dispatch_fetch_locked(runtime, action);
                if (dispatched < 0) {
                    pthread_mutex_unlock(&object->lock);
                    return -1;
                }
                if (dispatched != 0) {
                    shadowspill_transfer_lane_publish_inflight(lane, action);
                }
                pthread_mutex_unlock(&object->lock);
                return dispatched;
            }
    } else {
            ShadowSpillTransferLane *lane =
                shadowspill_transfer_lane_for_action(runtime, action);
            /*
             * The backend may have completed a whole FIFO prefix while an earlier
             * action's object lock is briefly unavailable.  Commit that
             * prefix strictly from the lane head: skipping a busy predecessor
             * must never let a later transfer publish residency first.
             */
            if (!shadowspill_transfer_lane_is_inflight_head(lane, action)) {
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
                    latch_action_failure(
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
                } else if (caller_handoff || !object->retain_spill_copy) {
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
                        shadowspill_find_execution_lease(
                            action->plan_owner->execution_pool,
                            object->allocation_id
                        );
                    if (allocation == NULL) {
                        shadowspill_memory_pool_unlock_reclamation(
                            action->plan_owner->execution_pool
                        );
                        latch_action_failure(
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
                    shadowspill_release_execution_lease_locked(runtime, allocation);
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
                            latch_action_failure(
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
                uint64_t stream_start_ns = SHADOWSPILL_TRACE_NO_STREAM_TIME;
                uint64_t stream_end_ns = SHADOWSPILL_TRACE_NO_STREAM_TIME;
                if (shadowspill_stream_interval_read(
                        runtime, &action->stream_interval,
                        runtime->trace_origin_event,
                        &stream_start_ns, &stream_end_ns
                    ) != 0) {
                    stream_start_ns = SHADOWSPILL_TRACE_NO_STREAM_TIME;
                    stream_end_ns = SHADOWSPILL_TRACE_NO_STREAM_TIME;
                }
                shadowspill_stream_interval_discard(
                    runtime, &action->stream_interval
                );
                shadowspill_append_stamped_trace_event_locked(
                    runtime,
                    SHADOWSPILL_TRACE_TRANSFER_COMPLETED,
                    action->task_id,
                    object->object_id,
                    caller_handoff
                        ? action->caller_handoff_lease->allocation_id
                        : object->allocation_id,
                    object->size_bytes,
                    action->kind == SHADOWSPILL_RUNTIME_EVICT
                        ? SHADOWSPILL_TRANSFER_EVICT
                        : SHADOWSPILL_TRANSFER_FETCH,
                    atomic_load_explicit(
                        &runtime->actions.count, memory_order_acquire
                    ),
                    stream_start_ns,
                    stream_end_ns
                );
                pthread_mutex_unlock(&object->lock);
                if (readiness_to_release != NULL &&
                    shadowspill_event_lease_release(
                        runtime, readiness_to_release
                    ) != 0) {
                    latch_action_failure(
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
                if (shadowspill_transfer_lane_complete(lane, action) != 0) {
                    latch_action_failure(
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
                complete_action(runtime, action);
                return 2;
            }
    }
    pthread_mutex_unlock(&object->lock);
    return changed;
}

static int handle_actions(ShadowSpillRuntime *runtime) {
    const uint64_t limit = atomic_load_explicit(
        &runtime->actions.count, memory_order_acquire
    );
    int changed = 0;
    ShadowSpillQueuedAction *cursor = NULL;
    for (uint64_t visited = 0U; visited < limit; ++visited) {
        pthread_mutex_lock(&runtime->actions.lock);
        ShadowSpillQueuedAction *action = cursor == NULL
            ? runtime->actions.head
            : cursor;
        while (action != NULL && action->processing) {
            action = action->next;
        }
        if (action == NULL) {
            pthread_mutex_unlock(&runtime->actions.lock);
            break;
        }
        action->processing = 1U;
        cursor = action->next;
        pthread_mutex_unlock(&runtime->actions.lock);
        const int action_status = handle_action(runtime, action);
        if (action_status < 0) {
            release_action_claim(runtime, action);
            return changed;
        }
        if (action_status < 2) {
            release_action_claim(runtime, action);
        }
        if (action_status != 0) {
            changed = 1;
        }
    }
    return changed;
}

static int handle_submission_actions(
    ShadowSpillRuntime *runtime,
    ShadowSpillTaskRecord *record,
    uint64_t invocation
) {
    int changed = 0;
    for (uint32_t index = 0U; index < record->action_count; ++index) {
        ShadowSpillQueuedAction *action = &record->queued_actions[index];
        pthread_mutex_lock(&runtime->actions.lock);
        const int claim = action->active && !action->processing &&
            action->activation_generation == invocation;
        if (claim) {
            action->processing = 1U;
        }
        pthread_mutex_unlock(&runtime->actions.lock);
        if (!claim) {
            continue;
        }
        const int action_status = handle_action(runtime, action);
        if (action_status < 2) {
            release_action_claim(runtime, action);
        }
        if (action_status != 0) {
            changed = 1;
        }
        if (action_status < 0) {
            break;
        }
    }
    return changed;
}

static int handle_newly_published_submission(ShadowSpillRuntime *runtime) {
    ShadowSpillTaskRecord *record = atomic_load_explicit(
        &runtime->worker_submission, memory_order_acquire
    );
    if (record == NULL) {
        return 0;
    }
    const uint64_t sequence = atomic_load_explicit(
        &record->submission_sequence, memory_order_acquire
    );
    const uint64_t invocation = atomic_load_explicit(
        &record->submission_invocation, memory_order_relaxed
    );

    /* Attempt only this predecoded batch before testing acknowledgement. */
    const int changed = handle_submission_actions(
        runtime, record, invocation
    );
    int fetches_published = 1;
    for (uint32_t index = 0U; index < record->action_count; ++index) {
        ShadowSpillQueuedAction *action = &record->queued_actions[index];
        if (action->kind != SHADOWSPILL_RUNTIME_FETCH) {
            continue;
        }
        ShadowSpillObject *object = action->object;
        if (pthread_mutex_trylock(&object->lock) != 0) {
            fetches_published = 0;
            break;
        }
        const int published =
            (action->active && action->activation_generation == invocation &&
             action->state != SHADOWSPILL_ACTION_QUEUED) ||
            (!action->active &&
             action->completed_generation == invocation);
        pthread_mutex_unlock(&object->lock);
        if (!published) {
            fetches_published = 0;
            break;
        }
    }
    if (!fetches_published &&
        shadowspill_failure_status(runtime) == SHADOWSPILL_STATUS_OK) {
        return changed;
    }

    atomic_store_explicit(
        &record->acknowledgement_sequence, sequence, memory_order_release
    );
    ShadowSpillTaskRecord *expected = record;
    (void)atomic_compare_exchange_strong_explicit(
        &runtime->worker_submission,
        &expected,
        NULL,
        memory_order_release,
        memory_order_relaxed
    );
    return 1;
}

void *shadowspill_worker_main(void *pointer) {
    ShadowSpillRuntime *runtime = pointer;
    shadowspill_profiler_name_current_thread(
        &runtime->backend, "shadowspill_worker"
    );
    while (atomic_load_explicit(
        &runtime->worker_stop, memory_order_acquire
    ) == 0U) {
        /* Observe one dispatcher batch and publish every fetch readiness event. */
        const int submission_changed =
            handle_newly_published_submission(runtime);
        uint64_t next_completion_poll = 0U;
        uint64_t failure_object_id = SHADOWSPILL_RUNTIME_NO_ID;
        uint64_t failure_allocation_id = SHADOWSPILL_RUNTIME_NO_ID;
        /* Advance the FIFO completion frontier without holding pool locks. */
        const int completion_status = shadowspill_completion_poll(
            runtime,
            &next_completion_poll,
            &failure_object_id,
            &failure_allocation_id
        );
        if (completion_status < 0) {
            shadowspill_latch_failure_locked(
                runtime,
                SHADOWSPILL_STATUS_BACKEND_FAILURE,
                SHADOWSPILL_FAILURE_REASON_BACKEND_CALL_REJECTED,
                failure_object_id,
                failure_allocation_id,
                0U
            );
        }
        /* Reclaim completed leases while yielding pool priority to malloc. */
        const ShadowSpillRetirementWork retirement_work =
            shadowspill_handle_retirements(runtime);
        /* Dispatch or complete ready release, fetch, and evict actions. */
        if (!retirement_work.pool_busy &&
            shadowspill_failure_status(runtime) == SHADOWSPILL_STATUS_OK) {
            (void)handle_actions(runtime);
        }
        /* Failed actions remain parked while the always-active worker polls. */
        if (shadowspill_failure_status(runtime) != SHADOWSPILL_STATUS_OK) {
            shadowspill_cpu_relax();
            continue;
        }
        if (!submission_changed && completion_status == 0 &&
            !retirement_work.pool_busy) {
            shadowspill_cpu_relax();
        }
    }
    return NULL;
}
