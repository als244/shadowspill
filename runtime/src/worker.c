#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "internal.h"

#include <stdint.h>
#include <stdlib.h>
#include <time.h>

static int submit_transfer_copy(
    ShadowSpillRuntime *runtime,
    const ShadowSpillQueuedAction *action,
    const ShadowSpillTransferRoute *route,
    void *destination,
    const void *source,
    uint64_t bytes,
    ShadowSpillBackendStream stream
) {
    const char *fallback = action->kind == SHADOWSPILL_RUNTIME_PREFETCH
        ? "shadowspill.runtime.transfer.fetch.unlabeled"
        : "shadowspill.runtime.transfer.evict.unlabeled";
    const ShadowSpillProfilerRange range = shadowspill_profiler_range_begin(
        &runtime->profiler,
        action->trace_label == NULL ? fallback : action->trace_label
    );
    const int status = route->copy_async(
        route->context, destination, source, bytes, stream
    );
    shadowspill_profiler_range_end(&runtime->profiler, range);
    return status;
}

void shadowspill_notify_worker(ShadowSpillRuntime *runtime) {
    pthread_mutex_lock(&runtime->mutex);
    pthread_cond_broadcast(&runtime->condition);
    pthread_mutex_unlock(&runtime->mutex);
}

static void wait_while_idle(ShadowSpillRuntime *runtime) {
    struct timespec deadline;
    if (clock_gettime(CLOCK_REALTIME, &deadline) != 0) {
        pthread_cond_wait(&runtime->condition, &runtime->mutex);
        return;
    }
    uint64_t nanoseconds = (uint64_t)deadline.tv_nsec + UINT64_C(1000000);
    deadline.tv_sec += (time_t)(nanoseconds / UINT64_C(1000000000));
    deadline.tv_nsec = (long)(nanoseconds % UINT64_C(1000000000));
    (void)pthread_cond_timedwait(
        &runtime->condition, &runtime->mutex, &deadline
    );
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
    ShadowSpillRuntimeStatus status,
    uint64_t object_id,
    uint64_t allocation_id,
    uint64_t requested_bytes
) {
    shadowspill_latch_task_failure(
        runtime,
        status,
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
    pthread_mutex_lock(&action->object->lock);
    if (shadowspill_object_remove_action_locked(
            action->object, action
        ) != 0) {
        pthread_mutex_unlock(&action->object->lock);
        latch_action_failure(
            runtime,
            action,
            SHADOWSPILL_RUNTIME_INVALID_STATE,
            action->object->object_id,
            action->object->allocation_id,
            0U
        );
        return;
    }
    action->state = SHADOWSPILL_ACTION_FINISHED;
    pthread_mutex_unlock(&action->object->lock);
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
            latch_action_failure(
                runtime,
                action,
                SHADOWSPILL_RUNTIME_BACKEND_FAILURE,
                action->object->object_id,
                action->object->allocation_id,
                0U
            );
        }
    }
    if (action->dependency_event != NULL) {
        if (shadowspill_event_lease_release(
                runtime, action->dependency_event
            ) != 0) {
            latch_action_failure(
                runtime,
                action,
                SHADOWSPILL_RUNTIME_BACKEND_FAILURE,
                action->object->object_id,
                action->object->allocation_id,
                0U
            );
        }
        action->dependency_event = NULL;
    }
    if (action->admitted) {
        const uint8_t kind = action->kind;
        ShadowSpillObject *object = action->object;
        const uint64_t task_id = action->task_id;
        const char *trace_label = action->trace_label;
        *action = (ShadowSpillQueuedAction){
            .task_id = task_id,
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

static void wait_for_failure_recovery_or_close(ShadowSpillRuntime *runtime) {
    pthread_mutex_lock(&runtime->mutex);
    while (shadowspill_failure_status(runtime) != SHADOWSPILL_RUNTIME_OK &&
           atomic_load_explicit(
               &runtime->worker_stop, memory_order_acquire
           ) == 0U) {
        pthread_cond_wait(&runtime->condition, &runtime->mutex);
    }
    pthread_mutex_unlock(&runtime->mutex);
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
    const int status = pool->pool_id == runtime->execution_pool_id
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
        const ShadowSpillBackendStream lane =
            action->kind == SHADOWSPILL_RUNTIME_PREFETCH
            ? runtime->fetch_stream
            : runtime->evict_stream;
        if (runtime->backend.wait_event(
                runtime->backend.context,
                lane,
                dependency_event->event
            ) != 0) {
            (void)shadowspill_event_lease_release(runtime, dependency_event);
            return -1;
        }
        action->dependency_event = dependency_event;
    }
    return status;
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
        latch_action_failure(
            runtime,
            action,
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
            latch_action_failure(
                runtime,
                action,
                SHADOWSPILL_RUNTIME_INVALID_STATE,
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
                SHADOWSPILL_RUNTIME_INVALID_STATE,
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
    if (!backend_failed && (submit_transfer_copy(
            runtime,
            action,
            &runtime->evict_route,
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
    if (!backend_failed) {
        pthread_mutex_lock(&shadowspill_execution_pool(runtime)->lock);
        if (shadowspill_memory_pool_publish_retirement_dependency_locked(
                allocation, completion_event
            ) != 0) {
            backend_failed = 1;
        }
        pthread_mutex_unlock(&shadowspill_execution_pool(runtime)->lock);
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
        latch_action_failure(
            runtime,
            action,
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
    if (acquire_reserved_destination(runtime, action) != 0) {
        latch_action_failure(
            runtime,
            action,
            SHADOWSPILL_RUNTIME_INVALID_STATE,
            object_id,
            allocation->allocation_id,
            bytes
        );
        return -1;
    }
    action->destination_lease = NULL;
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
    if (!backend_failed && (submit_transfer_copy(
            runtime,
            action,
            &runtime->fetch_route,
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
    if (!backend_failed && !object->retain_spill_copy) {
        pthread_mutex_lock(&shadowspill_spill_pool(runtime)->lock);
        if (shadowspill_memory_pool_begin_retirement_locked(
                spill->lease, completion_event, 0
            ) != 0) {
            backend_failed = 1;
        }
        pthread_mutex_unlock(&shadowspill_spill_pool(runtime)->lock);
    }
    pthread_mutex_lock(&object->lock);
    if (backend_failed || object->residency != SHADOWSPILL_OBJECT_SPILL_ONLY ||
        object->generation != previous_generation ||
        object->authoritative_version != authoritative_version ||
        spill->version != spill_version || !spill->current) {
        if (completion_event != NULL) {
            (void)shadowspill_event_lease_release(runtime, completion_event);
        }
        /* Retain the activated range until failure teardown; see offload. */
        latch_action_failure(
            runtime,
            action,
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
    if (!shadowspill_object_action_is_head_locked(object, action)) {
        pthread_mutex_unlock(&object->lock);
        return 0;
    }
    if (action->state == SHADOWSPILL_ACTION_QUEUED) {
            if (action->kind == SHADOWSPILL_RUNTIME_RELEASE) {
                int complete = 0;
                if (shadowspill_task_fence_complete_locked(
                        runtime, action->fence, &complete
                    ) != 0) {
                    latch_action_failure(
                        runtime,
                        action,
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
                        latch_action_failure(
                            runtime,
                            action,
                            SHADOWSPILL_RUNTIME_INVALID_STATE,
                            object->object_id,
                            object->allocation_id,
                            0U
                        );
                        pthread_mutex_unlock(&object->lock);
                        return -1;
                    }
                    if (object->handoff_task_id !=
                            SHADOWSPILL_RUNTIME_NO_ID) {
                        if (allocation->handoff_head_object_id !=
                                object->object_id) {
                            shadowspill_memory_pool_unlock_reclamation(
                                shadowspill_execution_pool(runtime)
                            );
                            pthread_mutex_unlock(&object->lock);
                            return 0;
                        }
                        ShadowSpillObject *target =
                            shadowspill_find_object(
                                runtime,
                                object->handoff_destination_object_id
                            );
                        if (object->handoff_task_id != action->task_id ||
                            target == NULL ||
                            target->allocation_id != allocation->allocation_id) {
                            shadowspill_memory_pool_unlock_reclamation(
                                shadowspill_execution_pool(runtime)
                            );
                            latch_action_failure(
                                runtime,
                                action,
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
                        allocation->handoff_head_object_id =
                            object->handoff_next_source_object_id;
                        if (allocation->handoff_head_object_id ==
                                SHADOWSPILL_RUNTIME_NO_ID) {
                            allocation->handoff_tail_object_id =
                                SHADOWSPILL_RUNTIME_NO_ID;
                        }
                        object->handoff_destination_object_id =
                            SHADOWSPILL_RUNTIME_NO_ID;
                        object->handoff_task_id =
                            SHADOWSPILL_RUNTIME_NO_ID;
                        object->handoff_next_source_object_id =
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
                ShadowSpillObjectLocation *spill = shadowspill_spill_location(
                    runtime, object
                );
                if ((action->kind == SHADOWSPILL_RUNTIME_PREFETCH &&
                     action->destination_lease == NULL) ||
                    (action->kind == SHADOWSPILL_RUNTIME_OFFLOAD &&
                     spill->lease == NULL &&
                     action->destination_lease == NULL)) {
                    latch_action_failure(
                        runtime,
                        action,
                        SHADOWSPILL_RUNTIME_INVALID_STATE,
                        object->object_id,
                        object->allocation_id,
                        object->size_bytes
                    );
                    if (action->kind == SHADOWSPILL_RUNTIME_PREFETCH) {
                        object->prefetch_pending = 0U;
                        pthread_cond_broadcast(&object->state_changed);
                    }
                    pthread_mutex_unlock(&object->lock);
                    return -1;
                }
                if (action->kind == SHADOWSPILL_RUNTIME_OFFLOAD &&
                    object->residency == SHADOWSPILL_OBJECT_PREFETCHING) {
                    pthread_mutex_unlock(&object->lock);
                    return 0;
                }
                /* Capacity is owned; transfer dispatch still obeys the task. */
                int trigger_complete = 0;
                if (shadowspill_task_fence_complete_locked(
                        runtime, action->fence, &trigger_complete
                    ) != 0) {
                    latch_action_failure(
                        runtime,
                        action,
                        SHADOWSPILL_RUNTIME_BACKEND_FAILURE,
                        object->object_id,
                        object->allocation_id,
                        0U
                    );
                    pthread_mutex_unlock(&object->lock);
                    return -1;
                }
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
                if (!destination_dependency_is_published(action)) {
                    pthread_mutex_unlock(&object->lock);
                    return 0;
                }
                if (action->kind == SHADOWSPILL_RUNTIME_PREFETCH) {
                    ShadowSpillObjectLocation *spill =
                        shadowspill_spill_location(runtime, object);
                    if (object->residency !=
                            SHADOWSPILL_OBJECT_SPILL_ONLY ||
                        spill->lease == NULL || !spill->current ||
                        spill->version != object->authoritative_version) {
                        latch_action_failure(
                            runtime,
                            action,
                            SHADOWSPILL_RUNTIME_PLAN_VIOLATION,
                            object->object_id,
                            object->allocation_id,
                            object->size_bytes
                        );
                        object->prefetch_pending = 0U;
                        pthread_cond_broadcast(&object->state_changed);
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
                        latch_action_failure(
                            runtime,
                            action,
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
                    object->residency = SHADOWSPILL_OBJECT_SPILL_ONLY;
                } else {
                    ShadowSpillObjectLocation *execution =
                        shadowspill_execution_location(runtime, object);
                    if (execution->lease != NULL &&
                        execution->lease->generation == object->generation) {
                        object->residency = SHADOWSPILL_OBJECT_EXECUTION_READY;
                        execution->current = 1U;
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
                            latch_action_failure(
                                runtime,
                                action,
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
                    latch_action_failure(
                        runtime,
                        action,
                        SHADOWSPILL_RUNTIME_BACKEND_FAILURE,
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
        if (!retirement_work.pool_busy &&
            shadowspill_failure_status(runtime) == SHADOWSPILL_RUNTIME_OK) {
            (void)handle_actions(runtime);
        }
        /* Failed actions remain parked until recovery or teardown. */
        if (shadowspill_failure_status(runtime) != SHADOWSPILL_RUNTIME_OK) {
            pthread_cond_broadcast(&runtime->condition);
            wait_for_failure_recovery_or_close(runtime);
            continue;
        }
        /*
         * Park only when no work exists. With actions or retirements active,
         * immediately run another nonblocking pass. Completion polling checks
         * each stream's FIFO head at a short fixed cadence and drains its
         * already-complete prefix, so an actionable transition incurs neither
         * exponential backoff nor a sleep/wake round trip.
         */
        if (atomic_load_explicit(
            &runtime->worker_stop, memory_order_acquire
        ) == 0U) {
            const int idle = atomic_load_explicit(
                &runtime->actions.count, memory_order_acquire
            ) == 0U && !shadowspill_has_actionable_retirement(runtime);
            if (idle) {
                pthread_mutex_lock(&runtime->mutex);
                const int still_idle = atomic_load_explicit(
                    &runtime->actions.count, memory_order_acquire
                ) == 0U && !shadowspill_has_actionable_retirement(runtime);
                if (still_idle && atomic_load_explicit(
                        &runtime->worker_stop, memory_order_acquire
                    ) == 0U) {
                    wait_while_idle(runtime);
                }
                pthread_mutex_unlock(&runtime->mutex);
            }
        }
    }
    return NULL;
}
