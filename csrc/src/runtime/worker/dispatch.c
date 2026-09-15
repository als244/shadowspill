/* A transfer onto its lane: the destination's dependency, the copy, and
 * the events that order both against the compute stream. */
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

int shadowspill_action_destination_ready(ShadowSpillQueuedAction *action) {
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
        ? shadowspill_acquire_reserved_lease_locked(
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

int shadowspill_action_dispatch_evict_locked(
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
        shadowspill_action_latch_failure(
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
            shadowspill_action_latch_failure(
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
            shadowspill_action_latch_failure(
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
        shadowspill_action_latch_failure(
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
    if (!backend_failed && action->kind == SHADOWSPILL_RUNTIME_EVICT) {
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
        shadowspill_action_latch_failure(
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
    if (action->kind == SHADOWSPILL_RUNTIME_EVICT) {
        /* A write-back leaves the object readable while it copies. */
        object->residency = SHADOWSPILL_OBJECT_EVICTING;
    }
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



int shadowspill_action_dispatch_fetch_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillQueuedAction *action
) {
    if (action->destination_lease == NULL ||
        shadowspill_plan_spill_location(
            action->plan_owner, action->object
        )->lease == NULL) {
        shadowspill_action_latch_failure(
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
        shadowspill_action_latch_failure(
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
        shadowspill_action_latch_failure(
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
        shadowspill_action_latch_failure(
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
        shadowspill_action_latch_failure(
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
