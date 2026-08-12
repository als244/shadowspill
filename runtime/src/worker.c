#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "internal.h"

#include <stdint.h>
#include <stdlib.h>
#include <time.h>

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

static void complete_action(
    ShadowSpillRuntime *runtime,
    ShadowSpillQueuedAction *action
) {
    pthread_mutex_lock(&runtime->actions.lock);
    unlink_action_locked(runtime, action);
    pthread_mutex_unlock(&runtime->actions.lock);
    if (action->kind == SHADOWSPILL_RUNTIME_RELEASE ||
        action->kind == SHADOWSPILL_RUNTIME_OFFLOAD) {
        (void)atomic_fetch_sub_explicit(
            &runtime->pending_capacity_actions, 1U, memory_order_release
        );
    }
    shadowspill_release_task_fence_locked(runtime, action->fence);
    if (action->has_completion_event) {
        if (shadowspill_event_lease_release(
                runtime, action->completion_event
            ) != 0) {
            shadowspill_latch_failure_locked(
                runtime,
                SHADOWSPILL_RUNTIME_BACKEND_FAILURE,
                action->object->object_id,
                action->object->allocation_id,
                0U
            );
        }
    }
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
    const uint64_t previous_actions = atomic_fetch_sub_explicit(
        &runtime->actions.count, 1U, memory_order_release
    );
    pthread_cond_broadcast(&runtime->condition);
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
        &event->completion_known, memory_order_acquire
    ) != 0U;
    return 0;
}

static int reserve_destination_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillQueuedAction *action
) {
    ShadowSpillObjectLocation *spill = shadowspill_spill_location(
        runtime, action->object
    );
    if (action->destination_lease != NULL ||
        (action->kind == SHADOWSPILL_RUNTIME_OFFLOAD &&
         spill->lease != NULL)) {
        return 1;
    }
    int trigger_complete = 0;
    if (shadowspill_task_fence_complete_locked(
            runtime, action->fence, &trigger_complete
        ) != 0) {
        shadowspill_latch_failure_locked(
            runtime,
            SHADOWSPILL_RUNTIME_BACKEND_FAILURE,
            action->object->object_id,
            action->object->allocation_id,
            0U
        );
        return -1;
    }
    if (!trigger_complete) {
        return 0;
    }
    ShadowSpillMemoryPool *pool = action->kind == SHADOWSPILL_RUNTIME_PREFETCH
        ? shadowspill_execution_pool(runtime)
        : shadowspill_spill_pool(runtime);
    if (!action->destination_priority_declared) {
        shadowspill_memory_pool_declare_transfer(pool);
        action->destination_priority_declared = 1U;
    }
    if (!shadowspill_memory_pool_try_lock_transfer(pool)) {
        return 0;
    }
    int range_status;
    if (action->kind == SHADOWSPILL_RUNTIME_PREFETCH) {
        ShadowSpillRuntimeStatus lease_status =
            shadowspill_create_execution_lease_locked(
                runtime,
                action->object->size_bytes,
                shadowspill_execution_pool(runtime)->minimum_alignment,
                1,
                action->task_id,
                &action->destination_lease
            );
        range_status = lease_status == SHADOWSPILL_RUNTIME_OK
            ? 0
            : (lease_status == SHADOWSPILL_RUNTIME_OUT_OF_MEMORY ? 1 : -1);
        if (range_status == 0) {
            action->destination_lease->state = SHADOWSPILL_LEASE_RESERVED;
            action->destination_lease->bound_object_id =
                action->object->object_id;
        }
    } else {
        action->destination_lease = calloc(
            1U, sizeof(*action->destination_lease)
        );
        range_status = action->destination_lease == NULL
            ? -1
            : shadowspill_memory_pool_reserve_lease_locked(
                shadowspill_spill_pool(runtime),
                action->destination_lease,
                action->object->size_bytes,
                1U,
                SHADOWSPILL_MEMORY_FIRST_FIT
            );
        if (range_status != 0) {
            free(action->destination_lease);
            action->destination_lease = NULL;
        }
    }
    shadowspill_memory_pool_relinquish_transfer(pool);
    action->destination_priority_declared = 0U;
    shadowspill_memory_pool_unlock_transfer(pool);
    if (range_status != 0) {
        ShadowSpillRuntimeStatus status = range_status < 0
            ? SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE
            : SHADOWSPILL_RUNTIME_PLAN_VIOLATION;
        shadowspill_latch_failure_locked(
            runtime,
            status,
            action->object->object_id,
            SHADOWSPILL_RUNTIME_NO_ID,
            action->object->size_bytes
        );
        return -1;
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
    pthread_cond_broadcast(&runtime->condition);
    return 1;
}

static int dispatch_offload_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillQueuedAction *action
) {
    ShadowSpillObject *object = action->object;
    if (object->residency == SHADOWSPILL_OBJECT_PREFETCHING) {
        return 0;
    }
    ShadowSpillObjectLocation *execution = shadowspill_execution_location(
        runtime, object
    );
    ShadowSpillObjectLocation *spill = shadowspill_spill_location(
        runtime, object
    );
    ShadowSpillMemoryLease *allocation = execution->lease;
    if (allocation == NULL || allocation->pointer == NULL ||
        allocation->allocation_id != object->allocation_id ||
        allocation->generation != object->generation) {
        shadowspill_latch_failure_locked(
            runtime,
            SHADOWSPILL_RUNTIME_INVALID_STATE,
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
            shadowspill_latch_failure_locked(
                runtime,
                SHADOWSPILL_RUNTIME_INVALID_STATE,
                object->object_id,
                object->allocation_id,
                object->size_bytes
            );
            return -1;
        }
        spill->lease = action->destination_lease;
        spill->owns_lease = 1U;
        spill->lease->state = SHADOWSPILL_LEASE_TRANSFERRING;
        action->destination_lease = NULL;
        spill_lease_created = 1;
    }
    ShadowSpillMemoryLease *spill_lease = spill->lease;
    spill_lease->state = SHADOWSPILL_LEASE_TRANSFERRING;
    pthread_mutex_unlock(&object->lock);

    ShadowSpillEventLease *completion_event = NULL;
    ShadowSpillRuntimeStatus event_status = shadowspill_event_lease_create_locked(
        runtime, &completion_event
    );
    int backend_failed = event_status != SHADOWSPILL_RUNTIME_OK;
    if (!backend_failed && runtime->backend.wait_event(
            runtime->backend.context,
            runtime->evict_stream,
            action->fence->event->event
        ) != 0) {
        backend_failed = 1;
    }
    if (!backend_failed && (runtime->evict_route.copy_async(
            runtime->evict_route.context,
            spill_lease->pointer,
            execution_pointer,
            bytes,
            runtime->evict_stream
        ) != 0 || runtime->backend.record_event(
                runtime->backend.context,
                completion_event->event,
                runtime->evict_stream
            ) != 0 || shadowspill_completion_submit(
                runtime,
                runtime->evict_stream,
                completion_event,
                object_id,
                allocation_id
            ) != SHADOWSPILL_RUNTIME_OK)) {
        backend_failed = 1;
    }
    pthread_mutex_lock(&object->lock);
    if (backend_failed || object->generation != object_generation ||
        execution->lease != allocation ||
        object->allocation_id != allocation_id) {
        if (completion_event != NULL) {
            (void)shadowspill_event_lease_release(runtime, completion_event);
        }
        if (spill_lease_created) {
            pthread_mutex_lock(&shadowspill_spill_pool(runtime)->lock);
            (void)shadowspill_memory_pool_release_lease_locked(spill_lease);
            pthread_mutex_unlock(&shadowspill_spill_pool(runtime)->lock);
            free(spill_lease);
            spill->lease = NULL;
            spill->owns_lease = 0U;
        }
        shadowspill_latch_failure_locked(
            runtime,
            backend_failed ? SHADOWSPILL_RUNTIME_BACKEND_FAILURE
                           : SHADOWSPILL_RUNTIME_INVALID_STATE,
            object_id,
            allocation_id,
            bytes
        );
        return -1;
    }
    action->completion_event = completion_event;
    action->has_completion_event = 1U;
    action->state = SHADOWSPILL_ACTION_IN_FLIGHT;
    object->residency = SHADOWSPILL_OBJECT_OFFLOADING;
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

static int dispatch_prefetch_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillQueuedAction *action
) {
    if (action->destination_lease == NULL ||
        shadowspill_spill_location(runtime, action->object)->lease == NULL) {
        shadowspill_latch_failure_locked(
            runtime,
            SHADOWSPILL_RUNTIME_INVALID_STATE,
            action->object->object_id,
            SHADOWSPILL_RUNTIME_NO_ID,
            action->object->size_bytes
        );
        return -1;
    }
    ShadowSpillObject *object = action->object;
    ShadowSpillObjectLocation *execution = shadowspill_execution_location(
        runtime, object
    );
    ShadowSpillObjectLocation *spill = shadowspill_spill_location(
        runtime, object
    );
    const uint64_t object_id = object->object_id;
    const uint64_t previous_generation = object->generation;
    const uint64_t authoritative_version = object->authoritative_version;
    const uint64_t spill_version = spill->version;
    const uint64_t bytes = object->size_bytes;
    ShadowSpillMemoryLease *allocation = action->destination_lease;
    action->destination_lease = NULL;
    allocation->state = SHADOWSPILL_LEASE_TRANSFERRING;
    pthread_mutex_unlock(&object->lock);
    ShadowSpillEventLease *completion_event = NULL;
    ShadowSpillRuntimeStatus event_status = shadowspill_event_lease_create_locked(
        runtime, &completion_event
    );
    int backend_failed = event_status != SHADOWSPILL_RUNTIME_OK;
    if (!backend_failed && runtime->backend.wait_event(
            runtime->backend.context,
            runtime->fetch_stream,
            action->fence->event->event
        ) != 0) {
        backend_failed = 1;
    }
    if (!backend_failed && (runtime->fetch_route.copy_async(
            runtime->fetch_route.context,
            allocation->pointer,
            spill->lease->pointer,
            bytes,
            runtime->fetch_stream
        ) != 0 || runtime->backend.record_event(
                runtime->backend.context,
                completion_event->event,
                runtime->fetch_stream
            ) != 0 || shadowspill_completion_submit(
                runtime,
                runtime->fetch_stream,
                completion_event,
                object_id,
                allocation->allocation_id
            ) != SHADOWSPILL_RUNTIME_OK)) {
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
        pthread_mutex_lock(&shadowspill_execution_pool(runtime)->lock);
        allocation->release_task_id = action->task_id;
        shadowspill_release_execution_lease_locked(runtime, allocation);
        pthread_mutex_unlock(&shadowspill_execution_pool(runtime)->lock);
        shadowspill_latch_failure_locked(
            runtime,
            backend_failed ? SHADOWSPILL_RUNTIME_BACKEND_FAILURE
                           : SHADOWSPILL_RUNTIME_INVALID_STATE,
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
    object->residency = SHADOWSPILL_OBJECT_PREFETCHING;
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
    if (action->state == SHADOWSPILL_ACTION_QUEUED) {
            if (action->kind == SHADOWSPILL_RUNTIME_RELEASE) {
                int complete = 0;
                if (shadowspill_task_fence_complete_locked(
                        runtime, action->fence, &complete
                    ) != 0) {
                    shadowspill_latch_failure_locked(
                        runtime,
                        SHADOWSPILL_RUNTIME_BACKEND_FAILURE,
                        object->object_id,
                        object->allocation_id,
                        0U
                    );
                    pthread_mutex_unlock(&object->lock);
                    return -1;
                }
                if (complete) {
                    if (!shadowspill_memory_pool_try_lock_reclamation(
                            shadowspill_execution_pool(runtime)
                        )) {
                        pthread_mutex_unlock(&object->lock);
                        return 0;
                    }
                    ShadowSpillMemoryLease *allocation =
                        shadowspill_find_execution_lease(
                            runtime, object->allocation_id
                        );
                    if (allocation == NULL) {
                        shadowspill_memory_pool_unlock_reclamation(
                            shadowspill_execution_pool(runtime)
                        );
                        shadowspill_latch_failure_locked(
                            runtime,
                            SHADOWSPILL_RUNTIME_INVALID_STATE,
                            object->object_id,
                            object->allocation_id,
                            0U
                        );
                        pthread_mutex_unlock(&object->lock);
                        return -1;
                    }
                    if (allocation->handoff_from_object_id ==
                            object->object_id) {
                        ShadowSpillObject *target =
                            shadowspill_find_object(
                                runtime,
                                allocation->handoff_to_object_id
                            );
                        if (allocation->handoff_task_id != action->task_id ||
                            target == NULL ||
                            target->allocation_id != allocation->allocation_id) {
                            shadowspill_memory_pool_unlock_reclamation(
                                shadowspill_execution_pool(runtime)
                            );
                            shadowspill_latch_failure_locked(
                                runtime,
                                SHADOWSPILL_RUNTIME_INVALID_STATE,
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
                        shadowspill_execution_location(runtime, object)->lease = NULL;
                        shadowspill_execution_location(runtime, object)->current = 0U;
                        object->residency = shadowspill_spill_location(runtime, object)->current
                            ? SHADOWSPILL_OBJECT_SPILL_ONLY
                            : SHADOWSPILL_OBJECT_RELEASED;
                        allocation->handoff_from_object_id =
                            SHADOWSPILL_RUNTIME_NO_ID;
                        allocation->handoff_to_object_id =
                            SHADOWSPILL_RUNTIME_NO_ID;
                        allocation->handoff_task_id =
                            SHADOWSPILL_RUNTIME_NO_ID;
                        shadowspill_memory_pool_unlock_reclamation(
                            shadowspill_execution_pool(runtime)
                        );
                        pthread_cond_broadcast(&object->state_changed);
                        pthread_mutex_unlock(&object->lock);
                        complete_action(runtime, action);
                        return 2;
                    }
                    object->retired_generation = object->generation;
                    object->retired_execution_pointer = allocation->pointer;
                    allocation->release_task_id = action->task_id;
                    shadowspill_release_execution_lease_locked(runtime, allocation);
                    shadowspill_memory_pool_unlock_reclamation(
                        shadowspill_execution_pool(runtime)
                    );
                    object->allocation_id = SHADOWSPILL_RUNTIME_NO_ID;
                    shadowspill_execution_location(runtime, object)->lease = NULL;
                    shadowspill_execution_location(runtime, object)->current = 0U;
                    object->residency = shadowspill_spill_location(runtime, object)->current
                        ? SHADOWSPILL_OBJECT_SPILL_ONLY
                        : SHADOWSPILL_OBJECT_RELEASED;
                    pthread_cond_broadcast(&object->state_changed);
                    pthread_mutex_unlock(&object->lock);
                    complete_action(runtime, action);
                    return 2;
                }
            } else {
                int reserved = reserve_destination_locked(runtime, action);
                if (reserved < 0) {
                    if (action->kind == SHADOWSPILL_RUNTIME_PREFETCH) {
                        object->prefetch_pending = 0U;
                        pthread_cond_broadcast(&object->state_changed);
                    }
                    pthread_mutex_unlock(&object->lock);
                    return -1;
                }
                if (reserved == 0) {
                    pthread_mutex_unlock(&object->lock);
                    return 0;
                }
                if (action->kind == SHADOWSPILL_RUNTIME_OFFLOAD &&
                    object->residency == SHADOWSPILL_OBJECT_PREFETCHING) {
                    pthread_mutex_unlock(&object->lock);
                    return 0;
                }
                ShadowSpillTransferLane *lane =
                    shadowspill_transfer_lane_for_action(runtime, action);
                if (!shadowspill_transfer_lane_claim(lane, action)) {
                    pthread_mutex_unlock(&object->lock);
                    return 0;
                }
                int dispatched = action->kind == SHADOWSPILL_RUNTIME_OFFLOAD
                    ? dispatch_offload_locked(runtime, action)
                    : dispatch_prefetch_locked(runtime, action);
                if (dispatched < 0) {
                    if (action->kind == SHADOWSPILL_RUNTIME_PREFETCH) {
                        object->prefetch_pending = 0U;
                        pthread_cond_broadcast(&object->state_changed);
                    }
                    pthread_mutex_unlock(&object->lock);
                    return -1;
                }
                if (dispatched != 0) {
                    shadowspill_transfer_lane_publish_inflight(lane, action);
                    pthread_cond_broadcast(&runtime->condition);
                    pthread_cond_broadcast(&object->state_changed);
                }
                pthread_mutex_unlock(&object->lock);
                return dispatched;
            }
    } else {
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
                if (action->kind == SHADOWSPILL_RUNTIME_OFFLOAD) {
                    release_pool = shadowspill_execution_pool(runtime);
                } else if (!object->retain_spill_copy) {
                    release_pool = shadowspill_spill_pool(runtime);
                }
                if (release_pool != NULL &&
                    !shadowspill_memory_pool_try_lock_reclamation(
                        release_pool
                    )) {
                    pthread_mutex_unlock(&object->lock);
                    return 0;
                }
                if (action->kind == SHADOWSPILL_RUNTIME_OFFLOAD) {
                    ShadowSpillMemoryLease *allocation =
                        shadowspill_find_execution_lease(
                            runtime, object->allocation_id
                        );
                    if (allocation == NULL) {
                        shadowspill_memory_pool_unlock_reclamation(
                            shadowspill_execution_pool(runtime)
                        );
                        shadowspill_latch_failure_locked(
                            runtime,
                            SHADOWSPILL_RUNTIME_INVALID_STATE,
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
                        shadowspill_execution_pool(runtime)
                    );
                    object->allocation_id = SHADOWSPILL_RUNTIME_NO_ID;
                    shadowspill_execution_location(runtime, object)->lease = NULL;
                    shadowspill_execution_location(runtime, object)->current = 0U;
                    shadowspill_spill_location(runtime, object)->current = 1U;
                    shadowspill_spill_location(runtime, object)->version = object->authoritative_version;
                    shadowspill_spill_location(runtime, object)->lease->state =
                        SHADOWSPILL_LEASE_ACTIVE;
                    object->residency = SHADOWSPILL_OBJECT_SPILL_ONLY;
                } else {
                    ShadowSpillObjectLocation *execution =
                        shadowspill_execution_location(runtime, object);
                    if (execution->lease != NULL &&
                        execution->lease->generation == object->generation) {
                        object->residency = SHADOWSPILL_OBJECT_EXECUTION_READY;
                        execution->current = 1U;
                        execution->lease->state = SHADOWSPILL_LEASE_ACTIVE;
                    }
                    /*
                     * The compute stream may already have waited on this
                     * transfer and launched a task.  In that case
                     * after_task has advanced execution_version while the
                     * worker still observes PREFETCHING. The fetch
                     * completion only changes readiness; it must not roll
                     * the execution version back to the copied spill version.
                     */
                    object->has_readiness_event = 0U;
                    readiness_to_release = object->readiness_event;
                    object->readiness_event = NULL;
                    if (!object->retain_spill_copy) {
                        ShadowSpillObjectLocation *spill =
                            shadowspill_spill_location(runtime, object);
                        ShadowSpillMemoryLease *lease = spill->lease;
                        const int range_status =
                            shadowspill_memory_pool_release_lease_locked(lease);
                        shadowspill_memory_pool_unlock_reclamation(
                            shadowspill_spill_pool(runtime)
                        );
                        if (range_status != 0) {
                            shadowspill_latch_failure_locked(
                                runtime,
                                SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE,
                                object->object_id,
                                object->allocation_id,
                                object->size_bytes
                            );
                            pthread_mutex_unlock(&object->lock);
                            return -1;
                        }
                        free(lease);
                        spill->lease = NULL;
                        spill->owns_lease = 0U;
                        spill->current = 0U;
                    }
                }
                shadowspill_append_trace_event_locked(
                    runtime,
                    SHADOWSPILL_TRACE_TRANSFER_COMPLETED,
                    action->task_id,
                    object->object_id,
                    object->allocation_id,
                    object->size_bytes,
                    action->kind == SHADOWSPILL_RUNTIME_OFFLOAD
                        ? SHADOWSPILL_TRANSFER_EVICT
                        : SHADOWSPILL_TRANSFER_FETCH,
                    atomic_load_explicit(
                        &runtime->actions.count, memory_order_acquire
                    )
                );
                object->prefetch_pending = 0U;
                pthread_cond_broadcast(&object->state_changed);
                pthread_mutex_unlock(&object->lock);
                if (readiness_to_release != NULL &&
                    shadowspill_event_lease_release(
                        runtime, readiness_to_release
                    ) != 0) {
                    shadowspill_latch_failure_locked(
                        runtime,
                        SHADOWSPILL_RUNTIME_BACKEND_FAILURE,
                        object->object_id,
                        object->allocation_id,
                        0U
                    );
                    return -1;
                }
                ShadowSpillTransferLane *lane =
                    shadowspill_transfer_lane_for_action(runtime, action);
                if (shadowspill_transfer_lane_complete(lane, action) != 0) {
                    shadowspill_latch_failure_locked(
                        runtime,
                        SHADOWSPILL_RUNTIME_INVALID_STATE,
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

static void timed_wait_locked(
    ShadowSpillRuntime *runtime,
    uint64_t completion_wait_nanoseconds
) {
    uint64_t wait_ns = completion_wait_nanoseconds;
    if (wait_ns == 0U) {
        wait_ns = runtime->worker_poll_nanoseconds;
    }
    if (wait_ns == 0U) {
        wait_ns = 1000000U;
    }
    struct timespec deadline;
    if (clock_gettime(CLOCK_REALTIME, &deadline) != 0) {
        pthread_cond_wait(&runtime->condition, &runtime->mutex);
        return;
    }
    uint64_t nanoseconds = (uint64_t)deadline.tv_nsec + wait_ns;
    deadline.tv_sec += (time_t)(nanoseconds / 1000000000U);
    deadline.tv_nsec = (long)(nanoseconds % 1000000000U);
    (void)pthread_cond_timedwait(
        &runtime->condition, &runtime->mutex, &deadline
    );
}

void *shadowspill_worker_main(void *pointer) {
    ShadowSpillRuntime *runtime = pointer;
    shadowspill_profiler_name_current_thread(
        &runtime->profiler, "shadowspill_worker"
    );
    while (atomic_load_explicit(
        &runtime->worker_stop, memory_order_acquire
    ) == 0U) {
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
                SHADOWSPILL_RUNTIME_BACKEND_FAILURE,
                failure_object_id,
                failure_allocation_id,
                0U
            );
        }
        /* Reclaim completed leases while yielding pool priority to malloc. */
        const ShadowSpillRetirementWork retirement_work =
            shadowspill_handle_retirements(runtime);
        /* Dispatch or complete ready release, fetch, and evict actions. */
        if (!retirement_work.pool_busy) {
            (void)handle_actions(runtime);
        }
        /* Wake blocked callers immediately after a worker failure. */
        if (shadowspill_failure_status(runtime) != SHADOWSPILL_RUNTIME_OK) {
            pthread_cond_broadcast(&runtime->condition);
        }
        /* Park only when idle; otherwise poll the next causal frontier. */
        if (atomic_load_explicit(
            &runtime->worker_stop, memory_order_acquire
        ) == 0U) {
            pthread_mutex_lock(&runtime->mutex);
            if (atomic_load_explicit(
                &runtime->actions.count, memory_order_acquire
                ) == 0U && !shadowspill_has_actionable_retirement(runtime)) {
                pthread_cond_wait(&runtime->condition, &runtime->mutex);
            } else {
                /*
                 * Always release the runtime lock between worker passes.
                 * A FIFO transfer window can complete one item per pass for
                 * many consecutive passes.  Immediately rescanning after
                 * each completion otherwise starves framework malloc/free
                 * callbacks that need the same state lock.
                 */
                timed_wait_locked(runtime, next_completion_poll);
            }
            pthread_mutex_unlock(&runtime->mutex);
        }
    }
    return NULL;
}
