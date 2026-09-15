/* One action handled: which stage it is in, and the two queued paths --
 * a release that gives capacity back, or a transfer onto a lane. */
#define _GNU_SOURCE

#include "internal.h"

#include <stdint.h>
#include <stdlib.h>

/*
 * A release gives capacity back: it waits for the task that last read the
 * object, then retires the execution lease. Nothing is copied, so the action
 * finishes here rather than on a transfer lane.
 */
static int handle_release_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillQueuedAction *action
) {
    ShadowSpillObject *object = action->object;
    if (action->trigger_event == NULL) {
        shadowspill_action_latch_failure(
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
        if (action->handoff_lease != NULL) {
            if (allocation != action->handoff_lease ||
                allocation->generation !=
                    action->handoff_generation ||
                allocation->bound_object == NULL ||
                allocation->bound_object == object) {
                shadowspill_memory_pool_unlock_reclamation(
                    action->plan_owner->execution_pool
                );
                shadowspill_action_latch_failure(
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
            shadowspill_action_complete(runtime, action);
            return 2;
        }
        if (allocation->bound_object != object) {
            shadowspill_memory_pool_unlock_reclamation(
                action->plan_owner->execution_pool
            );
            shadowspill_action_latch_failure(
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
        if (action->retires_when_processed &&
            shadowspill_memory_pool_begin_retirement_locked(
                allocation, NULL, 0
            ) != 0) {
            shadowspill_memory_pool_unlock_reclamation(
                action->plan_owner->execution_pool
            );
            shadowspill_action_latch_failure(
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
        shadowspill_release_lease_locked(runtime, allocation);
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
        shadowspill_action_complete(runtime, action);
        return 2;
    }
    pthread_mutex_unlock(&object->lock);
    return 0;
}

/*
 * A fetch, an eviction or a write-back: capacity is owned by the time the
 * action is queued, so what remains is waiting for its trigger, proving the
 * destination may be written, claiming the lane, and dispatching the copy.
 */
static int handle_transfer_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillQueuedAction *action
) {
    ShadowSpillObject *object = action->object;
    ShadowSpillObjectLocation *spill =
        shadowspill_plan_spill_location(
            action->plan_owner, object
        );
    const int copies_to_spill =
        action->kind == SHADOWSPILL_RUNTIME_EVICT ||
        (action->kind == SHADOWSPILL_RUNTIME_WRITE_BACK &&
         !action->skips_copy);
    if ((action->kind == SHADOWSPILL_RUNTIME_FETCH &&
         action->destination_lease == NULL) ||
        (copies_to_spill && spill->lease == NULL &&
         action->destination_lease == NULL)) {
        shadowspill_action_latch_failure(
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
    if (copies_to_spill &&
        object->residency == SHADOWSPILL_OBJECT_FETCHING) {
        pthread_mutex_unlock(&object->lock);
        return 0;
    }
    /* Capacity is owned; transfer dispatch still obeys the task. */
    if (action->trigger_event == NULL) {
        shadowspill_action_latch_failure(
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
    if (action->kind == SHADOWSPILL_RUNTIME_WRITE_BACK &&
        action->skips_copy) {
        /* The spill copy already held this version when the
         * action was scheduled: nothing to copy. */
        pthread_mutex_unlock(&object->lock);
        shadowspill_action_complete(runtime, action);
        return 2;
    }
    /*
     * A causal destination owns capacity from the trigger, but it
     * cannot enter a transfer lane until its predecessor has
     * published the event that makes address reuse safe.  Leave
     * it queued so it cannot occupy and stall the lane head.
     */
    pthread_mutex_unlock(&object->lock);
    const int dependency_ready =
        shadowspill_action_destination_ready(action);
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
        shadowspill_action_latch_failure(
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
        shadowspill_action_latch_failure(
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
            shadowspill_action_latch_failure(
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
    int dispatched = action->kind == SHADOWSPILL_RUNTIME_FETCH
        ? shadowspill_action_dispatch_fetch_locked(runtime, action)
        : shadowspill_action_dispatch_evict_locked(runtime, action);
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

int shadowspill_action_handle(
    ShadowSpillRuntime *runtime,
    ShadowSpillQueuedAction *action
) {
    ShadowSpillObject *object = action->object;
    if (pthread_mutex_trylock(&object->lock) != 0) {
        return 0;
    }
    if (!shadowspill_object_action_is_head_locked(object, action)) {
        pthread_mutex_unlock(&object->lock);
        return 0;
    }
    if (action->state != SHADOWSPILL_ACTION_QUEUED) {
        return shadowspill_action_finish_locked(runtime, action);
    }
    return action->kind == SHADOWSPILL_RUNTIME_RELEASE
        ? handle_release_locked(runtime, action)
        : handle_transfer_locked(runtime, action);
}
