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
    shadowspill_object_release(action->object);
    free(action);
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
    if (action->destination_reserved ||
        (action->kind == SHADOWSPILL_RUNTIME_OFFLOAD &&
         action->object->has_host_range)) {
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
    const uint64_t charged = action->object->size_bytes == 0U
        ? 1U
        : action->object->size_bytes;
    uint64_t offset = 0U;
    int range_status;
    if (action->kind == SHADOWSPILL_RUNTIME_PREFETCH) {
        pthread_mutex_lock(&runtime->device_pool.lock);
        range_status = shadowspill_memory_pool_reserve_locked(
            &runtime->device_pool,
            charged,
            runtime->device_pool.minimum_alignment,
            SHADOWSPILL_MEMORY_BEST_FIT_LOW,
            &offset
        );
        if (range_status == 0) {
            shadowspill_publish_device_geometry_locked(runtime);
        }
        pthread_mutex_unlock(&runtime->device_pool.lock);
    } else {
        pthread_mutex_lock(&runtime->host_pool.lock);
        range_status = shadowspill_memory_pool_reserve_locked(
            &runtime->host_pool,
            charged,
            1U,
            SHADOWSPILL_MEMORY_FIRST_FIT,
            &offset
        );
        pthread_mutex_unlock(&runtime->host_pool.lock);
    }
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
    action->destination_reserved = 1U;
    action->destination_offset = offset;
    action->destination_bytes = charged;
    shadowspill_append_trace_event_locked(
        runtime,
        SHADOWSPILL_TRACE_DESTINATION_RESERVED,
        action->task_id,
        action->object->object_id,
        action->object->allocation_id,
        action->object->size_bytes,
        action->kind,
        offset
    );
    pthread_cond_broadcast(&runtime->condition);
    return 1;
}

static int dispatch_offload_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillQueuedAction *action
) {
    ShadowSpillObjectRecord *object = action->object;
    if (object->residency == SHADOWSPILL_OBJECT_PREFETCHING) {
        return 0;
    }
    ShadowSpillAllocationRecord *allocation = object->device_lease;
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
    void *device_pointer = allocation->pointer;
    int host_range_created = 0;
    if (!object->has_host_range) {
        if (!action->destination_reserved) {
            shadowspill_latch_failure_locked(
                runtime,
                SHADOWSPILL_RUNTIME_INVALID_STATE,
                object->object_id,
                object->allocation_id,
                object->size_bytes
            );
            return -1;
        }
        object->host_offset = action->destination_offset;
        object->has_host_range = 1U;
        action->destination_reserved = 0U;
        host_range_created = 1;
    }
    const uint64_t host_offset = object->host_offset;
    pthread_mutex_unlock(&object->lock);

    ShadowSpillEventLease *completion_event = NULL;
    ShadowSpillRuntimeStatus event_status = shadowspill_event_lease_create_locked(
        runtime, &completion_event
    );
    int backend_failed = event_status != SHADOWSPILL_RUNTIME_OK;
    if (!backend_failed && runtime->backend.wait_event(
            runtime->backend.context,
            runtime->d2h_stream,
            action->fence->event->event
        ) != 0) {
        backend_failed = 1;
    }
    if (!backend_failed && (runtime->backend.copy_async(
            runtime->backend.context,
            shadowspill_memory_pool_pointer(
                &runtime->host_pool, host_offset
            ),
            device_pointer,
            bytes,
            SHADOWSPILL_TRANSFER_TO_HOST,
            runtime->d2h_stream
        ) != 0 || runtime->backend.record_event(
                runtime->backend.context,
                completion_event->event,
                runtime->d2h_stream
            ) != 0 || shadowspill_completion_submit(
                runtime,
                runtime->d2h_stream,
                completion_event,
                object_id,
                allocation_id
            ) != SHADOWSPILL_RUNTIME_OK)) {
        backend_failed = 1;
    }
    pthread_mutex_lock(&object->lock);
    if (backend_failed || object->generation != object_generation ||
        object->device_lease != allocation ||
        object->allocation_id != allocation_id) {
        if (completion_event != NULL) {
            (void)shadowspill_event_lease_release(runtime, completion_event);
        }
        if (host_range_created) {
            const uint64_t charged = bytes == 0U ? 1U : bytes;
            pthread_mutex_lock(&runtime->host_pool.lock);
            (void)shadowspill_memory_pool_release_locked(
                &runtime->host_pool, host_offset, charged
            );
            pthread_mutex_unlock(&runtime->host_pool.lock);
            object->has_host_range = 0U;
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
        &runtime->transfers_to_host, 1U, memory_order_acq_rel
    );
    (void)atomic_fetch_add_explicit(
        &runtime->bytes_to_host, bytes, memory_order_acq_rel
    );
    shadowspill_append_trace_event_locked(
        runtime,
        SHADOWSPILL_TRACE_TRANSFER_DISPATCHED,
        action->task_id,
        object->object_id,
        allocation_id,
        bytes,
        SHADOWSPILL_TRANSFER_TO_HOST,
        atomic_load_explicit(&runtime->actions.count, memory_order_acquire)
    );
    return 1;
}

static int dispatch_prefetch_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillQueuedAction *action
) {
    if (!action->destination_reserved) {
        shadowspill_latch_failure_locked(
            runtime,
            SHADOWSPILL_RUNTIME_INVALID_STATE,
            action->object->object_id,
            SHADOWSPILL_RUNTIME_NO_ID,
            action->object->size_bytes
        );
        return -1;
    }
    ShadowSpillObjectRecord *object = action->object;
    const uint64_t object_id = object->object_id;
    const uint64_t previous_generation = object->generation;
    const uint64_t authoritative_version = object->authoritative_version;
    const uint64_t host_version = object->host_version;
    const uint64_t host_offset = object->host_offset;
    const uint64_t bytes = object->size_bytes;
    const uint64_t destination_offset = action->destination_offset;
    action->destination_reserved = 0U;
    pthread_mutex_unlock(&object->lock);

    ShadowSpillAllocationRecord *allocation = NULL;
    pthread_mutex_lock(&runtime->device_pool.lock);
    ShadowSpillRuntimeStatus allocation_status =
        shadowspill_adopt_reserved_device_range_locked(
        runtime,
        bytes,
        destination_offset,
        1,
            action->task_id,
            &allocation
        );
    pthread_mutex_unlock(&runtime->device_pool.lock);
    if (allocation_status != SHADOWSPILL_RUNTIME_OK) {
        pthread_mutex_lock(&object->lock);
        shadowspill_latch_failure_locked(
            runtime,
            allocation_status,
            object_id,
            SHADOWSPILL_RUNTIME_NO_ID,
            bytes
        );
        return -1;
    }
    ShadowSpillEventLease *completion_event = NULL;
    ShadowSpillRuntimeStatus event_status = shadowspill_event_lease_create_locked(
        runtime, &completion_event
    );
    int backend_failed = event_status != SHADOWSPILL_RUNTIME_OK;
    if (!backend_failed && runtime->backend.wait_event(
            runtime->backend.context,
            runtime->h2d_stream,
            action->fence->event->event
        ) != 0) {
        backend_failed = 1;
    }
    if (!backend_failed && (runtime->backend.copy_async(
            runtime->backend.context,
            allocation->pointer,
            shadowspill_memory_pool_pointer(
                &runtime->host_pool, host_offset
            ),
            bytes,
            SHADOWSPILL_TRANSFER_TO_DEVICE,
            runtime->h2d_stream
        ) != 0 || runtime->backend.record_event(
                runtime->backend.context,
                completion_event->event,
                runtime->h2d_stream
            ) != 0 || shadowspill_completion_submit(
                runtime,
                runtime->h2d_stream,
                completion_event,
                object_id,
                allocation->allocation_id
            ) != SHADOWSPILL_RUNTIME_OK)) {
        backend_failed = 1;
    }
    pthread_mutex_lock(&object->lock);
    if (backend_failed || object->residency != SHADOWSPILL_OBJECT_HOST_ONLY ||
        object->generation != previous_generation ||
        object->authoritative_version != authoritative_version ||
        object->host_version != host_version || !object->host_current) {
        if (completion_event != NULL) {
            (void)shadowspill_event_lease_release(runtime, completion_event);
        }
        pthread_mutex_lock(&runtime->device_pool.lock);
        allocation->release_task_id = action->task_id;
        shadowspill_release_allocation_locked(runtime, allocation);
        pthread_mutex_unlock(&runtime->device_pool.lock);
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
    pthread_mutex_lock(&runtime->device_pool.lock);
    allocation->bound_object_id = object_id;
    pthread_mutex_unlock(&runtime->device_pool.lock);
    object->allocation_id = allocation->allocation_id;
    object->device_lease = allocation;
    object->generation = allocation->generation;
    object->device_version = object->host_version;
    object->readiness_event = completion_event;
    shadowspill_event_lease_retain(object->readiness_event);
    object->has_readiness_event = 1U;
    object->residency = SHADOWSPILL_OBJECT_PREFETCHING;
    (void)atomic_fetch_add_explicit(
        &runtime->transfers_to_device, 1U, memory_order_acq_rel
    );
    (void)atomic_fetch_add_explicit(
        &runtime->bytes_to_device, bytes, memory_order_acq_rel
    );
    shadowspill_append_trace_event_locked(
        runtime,
        SHADOWSPILL_TRACE_TRANSFER_DISPATCHED,
        action->task_id,
        object_id,
        allocation->allocation_id,
        bytes,
        SHADOWSPILL_TRANSFER_TO_DEVICE,
        atomic_load_explicit(&runtime->actions.count, memory_order_acquire)
    );
    return 1;
}

static int progress_action(
    ShadowSpillRuntime *runtime,
    ShadowSpillQueuedAction *action
) {
    int changed = 0;
    ShadowSpillObjectRecord *object = action->object;
    pthread_mutex_lock(&object->lock);
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
                    pthread_mutex_lock(&runtime->device_pool.lock);
                    ShadowSpillAllocationRecord *allocation =
                        shadowspill_find_allocation(
                            runtime, object->allocation_id
                        );
                    if (allocation == NULL) {
                        pthread_mutex_unlock(&runtime->device_pool.lock);
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
                        ShadowSpillObjectRecord *target =
                            shadowspill_find_object(
                                runtime,
                                allocation->handoff_to_object_id
                            );
                        if (allocation->handoff_task_id != action->task_id ||
                            target == NULL ||
                            target->allocation_id != allocation->allocation_id) {
                            pthread_mutex_unlock(
                                &runtime->device_pool.lock
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
                        object->retired_device_pointer = allocation->pointer;
                        object->allocation_id = SHADOWSPILL_RUNTIME_NO_ID;
                        object->device_lease = NULL;
                        object->residency = object->host_current
                            ? SHADOWSPILL_OBJECT_HOST_ONLY
                            : SHADOWSPILL_OBJECT_RELEASED;
                        allocation->handoff_from_object_id =
                            SHADOWSPILL_RUNTIME_NO_ID;
                        allocation->handoff_to_object_id =
                            SHADOWSPILL_RUNTIME_NO_ID;
                        allocation->handoff_task_id =
                            SHADOWSPILL_RUNTIME_NO_ID;
                        pthread_mutex_unlock(&runtime->device_pool.lock);
                        pthread_cond_broadcast(&object->state_changed);
                        pthread_mutex_unlock(&object->lock);
                        complete_action(runtime, action);
                        return 2;
                    }
                    object->retired_generation = object->generation;
                    object->retired_device_pointer = allocation->pointer;
                    allocation->release_task_id = action->task_id;
                    shadowspill_release_allocation_locked(runtime, allocation);
                    pthread_mutex_unlock(&runtime->device_pool.lock);
                    object->allocation_id = SHADOWSPILL_RUNTIME_NO_ID;
                    object->device_lease = NULL;
                    object->residency = object->host_current
                        ? SHADOWSPILL_OBJECT_HOST_ONLY
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
                shadowspill_append_trace_event_locked(
                    runtime,
                    SHADOWSPILL_TRACE_TRANSFER_COMPLETED,
                    action->task_id,
                    object->object_id,
                    object->allocation_id,
                    object->size_bytes,
                    action->kind == SHADOWSPILL_RUNTIME_OFFLOAD
                        ? SHADOWSPILL_TRANSFER_TO_HOST
                        : SHADOWSPILL_TRANSFER_TO_DEVICE,
                    atomic_load_explicit(
                        &runtime->actions.count, memory_order_acquire
                    )
                );
                if (action->kind == SHADOWSPILL_RUNTIME_OFFLOAD) {
                    pthread_mutex_lock(&runtime->device_pool.lock);
                    ShadowSpillAllocationRecord *allocation =
                        shadowspill_find_allocation(
                            runtime, object->allocation_id
                        );
                    if (allocation == NULL) {
                        pthread_mutex_unlock(&runtime->device_pool.lock);
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
                    object->retired_device_pointer = allocation->pointer;
                    allocation->release_task_id = action->task_id;
                    shadowspill_release_allocation_locked(runtime, allocation);
                    pthread_mutex_unlock(&runtime->device_pool.lock);
                    object->allocation_id = SHADOWSPILL_RUNTIME_NO_ID;
                    object->device_lease = NULL;
                    object->host_current = 1U;
                    object->host_version = object->authoritative_version;
                    object->residency = SHADOWSPILL_OBJECT_HOST_ONLY;
                } else {
                    object->residency = SHADOWSPILL_OBJECT_DEVICE_READY;
                    /*
                     * The compute stream may already have waited on this
                     * transfer and launched a task.  In that case
                     * after_task has advanced device_version while the
                     * progress thread still observes PREFETCHING.  The H2D
                     * completion only changes readiness; it must not roll
                     * the device version back to the copied host version.
                     */
                    object->has_readiness_event = 0U;
                    readiness_to_release = object->readiness_event;
                    object->readiness_event = NULL;
                    if (!object->retain_host_backing) {
                        uint64_t charged = object->size_bytes == 0U
                            ? 1U
                            : object->size_bytes;
                        pthread_mutex_lock(&runtime->host_pool.lock);
                        const int range_status =
                            shadowspill_memory_pool_release_locked(
                                &runtime->host_pool,
                                object->host_offset,
                                charged
                            );
                        pthread_mutex_unlock(&runtime->host_pool.lock);
                        if (range_status != 0) {
                            shadowspill_latch_failure_locked(
                                runtime,
                                SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE,
                                object->object_id,
                                object->allocation_id,
                                charged
                            );
                            pthread_mutex_unlock(&object->lock);
                            return -1;
                        }
                        object->has_host_range = 0U;
                        object->host_current = 0U;
                    }
                }
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

static int progress_actions(ShadowSpillRuntime *runtime) {
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
        const int action_status = progress_action(runtime, action);
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
        wait_ns = runtime->progress_poll_nanoseconds;
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

void *shadowspill_progress_main(void *pointer) {
    ShadowSpillRuntime *runtime = pointer;
#if defined(__linux__)
    (void)pthread_setname_np(pthread_self(), "shadowspill.wkr");
#endif
    while (atomic_load_explicit(
        &runtime->worker_stop, memory_order_acquire
    ) == 0U) {
        uint64_t next_completion_poll = 0U;
        uint64_t failure_object_id = SHADOWSPILL_RUNTIME_NO_ID;
        uint64_t failure_allocation_id = SHADOWSPILL_RUNTIME_NO_ID;
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
        const ShadowSpillRetirementProgress retirement_progress =
            shadowspill_progress_retirements(runtime);
        if (!retirement_progress.pool_busy) {
            (void)progress_actions(runtime);
        }
        if (shadowspill_failure_status(runtime) != SHADOWSPILL_RUNTIME_OK) {
            pthread_cond_broadcast(&runtime->condition);
        }
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
                 * Always release the runtime lock between progress passes.
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
