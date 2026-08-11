#define _POSIX_C_SOURCE 200809L

#include "internal.h"

#include <stdint.h>
#include <stdlib.h>
#include <time.h>

static void destroy_retirement_events_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillAllocationRecord *allocation
) {
    ShadowSpillEventRecord *event = allocation->retirement_events;
    while (event != NULL) {
        ShadowSpillEventRecord *next = event->next;
        if (runtime->backend.destroy_event(
                runtime->backend.context, event->event
            ) != 0) {
            shadowspill_latch_failure_locked(
                runtime,
                SHADOWSPILL_RUNTIME_BACKEND_FAILURE,
                SHADOWSPILL_RUNTIME_NO_ID,
                allocation->allocation_id,
                0U
            );
        }
        free(event);
        event = next;
    }
    allocation->retirement_events = NULL;
}

static int progress_retirements_locked(ShadowSpillRuntime *runtime) {
    int changed = 0;
    for (ShadowSpillAllocationRecord *allocation = runtime->allocations;
         allocation != NULL; allocation = allocation->next) {
        if (!allocation->logical_freed ||
            allocation->retirement_events == NULL ||
            allocation->pointer == NULL) {
            continue;
        }
        int complete = 1;
        for (ShadowSpillEventRecord *event = allocation->retirement_events;
             event != NULL; event = event->next) {
            int event_complete = 0;
            if (runtime->backend.query_event(
                    runtime->backend.context, event->event, &event_complete
                ) != 0) {
                shadowspill_latch_failure_locked(
                    runtime,
                    SHADOWSPILL_RUNTIME_BACKEND_FAILURE,
                    SHADOWSPILL_RUNTIME_NO_ID,
                    allocation->allocation_id,
                    0U
                );
                return changed;
            }
            if (!event_complete) {
                complete = 0;
            }
        }
        if (!complete) {
            continue;
        }
        destroy_retirement_events_locked(runtime, allocation);
        shadowspill_release_allocation_locked(runtime, allocation);
        if (runtime->pending_retirements != 0U) {
            --runtime->pending_retirements;
        }
        changed = 1;
    }
    return changed;
}

static void release_fence_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillTaskFence *fence
) {
    if (--fence->references != 0U) {
        return;
    }
    if (runtime->backend.destroy_event(
            runtime->backend.context, fence->event
        ) != 0) {
        shadowspill_latch_failure_locked(
            runtime,
            SHADOWSPILL_RUNTIME_BACKEND_FAILURE,
            SHADOWSPILL_RUNTIME_NO_ID,
            SHADOWSPILL_RUNTIME_NO_ID,
            0U
        );
    }
    free(fence);
}

static void complete_action_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillQueuedAction *previous,
    ShadowSpillQueuedAction *action
) {
    if (previous == NULL) {
        runtime->action_head = action->next;
    } else {
        previous->next = action->next;
    }
    if (runtime->action_tail == action) {
        runtime->action_tail = previous;
    }
    release_fence_locked(runtime, action->fence);
    if (action->has_completion_event) {
        if (runtime->backend.destroy_event(
                runtime->backend.context, action->completion_event
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
    free(action);
    if (runtime->queued_actions != 0U) {
        --runtime->queued_actions;
    }
    pthread_cond_broadcast(&runtime->condition);
}

static int event_complete_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillBackendEvent event,
    uint64_t object_id,
    int *complete
) {
    if (runtime->backend.query_event(
            runtime->backend.context, event, complete
        ) != 0) {
        shadowspill_latch_failure_locked(
            runtime,
            SHADOWSPILL_RUNTIME_BACKEND_FAILURE,
            object_id,
            SHADOWSPILL_RUNTIME_NO_ID,
            0U
        );
        return -1;
    }
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
    if (event_complete_locked(
            runtime,
            action->fence->event,
            action->object->object_id,
            &trigger_complete
        ) != 0) {
        return -1;
    }
    if (!trigger_complete) {
        return 0;
    }
    const uint64_t charged = action->object->size_bytes == 0U
        ? 1U
        : action->object->size_bytes;
    uint64_t offset = 0U;
    int range_status = action->kind == SHADOWSPILL_RUNTIME_PREFETCH
        ? shadowspill_range_allocate_best_fit_low(
            &runtime->device_ranges,
            charged,
            runtime->minimum_alignment,
            &offset
        )
        : shadowspill_range_allocate(
            &runtime->host_ranges, charged, 1U, &offset
        );
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
    pthread_cond_broadcast(&runtime->condition);
    return 1;
}

static int dispatch_offload_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillQueuedAction *action
) {
    for (ShadowSpillQueuedAction *earlier = runtime->action_head;
         earlier != NULL && earlier != action; earlier = earlier->next) {
        if (earlier->kind == SHADOWSPILL_RUNTIME_OFFLOAD) {
            return 0;
        }
    }
    ShadowSpillObjectRecord *object = action->object;
    if (object->residency == SHADOWSPILL_OBJECT_PREFETCHING) {
        return 0;
    }
    ShadowSpillAllocationRecord *allocation = shadowspill_find_allocation(
        runtime, object->allocation_id
    );
    if (allocation == NULL || allocation->pointer == NULL) {
        shadowspill_latch_failure_locked(
            runtime,
            SHADOWSPILL_RUNTIME_INVALID_STATE,
            object->object_id,
            object->allocation_id,
            object->size_bytes
        );
        return -1;
    }
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
    int completion_created = 0;
    if (runtime->backend.wait_event(
            runtime->backend.context,
            runtime->d2h_stream,
            action->fence->event
        ) != 0) {
        goto backend_failure;
    }
    if (runtime->backend.create_event(
            runtime->backend.context, &action->completion_event
        ) != 0) {
        goto backend_failure;
    }
    completion_created = 1;
    if (runtime->backend.copy_async(
            runtime->backend.context,
            (unsigned char *)runtime->host_arena + object->host_offset,
            allocation->pointer,
            object->size_bytes,
            SHADOWSPILL_TRANSFER_TO_HOST,
            runtime->d2h_stream
        ) != 0 || runtime->backend.record_event(
            runtime->backend.context,
            action->completion_event,
            runtime->d2h_stream
        ) != 0) {
        goto backend_failure;
    }
    action->has_completion_event = 1U;
    action->state = SHADOWSPILL_ACTION_IN_FLIGHT;
    object->residency = SHADOWSPILL_OBJECT_OFFLOADING;
    ++runtime->transfers_to_host;
    runtime->bytes_to_host += object->size_bytes;
    return 1;

backend_failure:
    if (completion_created) {
        (void)runtime->backend.destroy_event(
            runtime->backend.context, action->completion_event
        );
    }
    if (host_range_created) {
        const uint64_t charged = object->size_bytes == 0U
            ? 1U
            : object->size_bytes;
        (void)shadowspill_range_free(
            &runtime->host_ranges, object->host_offset, charged
        );
        object->has_host_range = 0U;
    }
    shadowspill_latch_failure_locked(
        runtime,
        SHADOWSPILL_RUNTIME_BACKEND_FAILURE,
        object->object_id,
        object->allocation_id,
        object->size_bytes
    );
    return -1;
}

static int dispatch_prefetch_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillQueuedAction *action
) {
    for (ShadowSpillQueuedAction *earlier = runtime->action_head;
         earlier != NULL && earlier != action; earlier = earlier->next) {
        if (earlier->kind == SHADOWSPILL_RUNTIME_PREFETCH) {
            return 0;
        }
    }
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
    ShadowSpillAllocationRecord *allocation = NULL;
    ShadowSpillRuntimeStatus allocation_status =
        shadowspill_adopt_reserved_device_range_locked(
        runtime,
        object->size_bytes,
        action->destination_offset,
        1,
        action->task_id,
        &allocation
    );
    if (allocation_status != SHADOWSPILL_RUNTIME_OK) {
        shadowspill_latch_failure_locked(
            runtime,
            allocation_status,
            object->object_id,
            SHADOWSPILL_RUNTIME_NO_ID,
            object->size_bytes
        );
        return -1;
    }
    action->destination_reserved = 0U;
    int completion_created = 0;
    if (runtime->backend.wait_event(
            runtime->backend.context,
            runtime->h2d_stream,
            action->fence->event
        ) != 0) {
        goto backend_failure;
    }
    if (runtime->backend.create_event(
            runtime->backend.context, &action->completion_event
        ) != 0) {
        goto backend_failure;
    }
    completion_created = 1;
    if (runtime->backend.copy_async(
            runtime->backend.context,
            allocation->pointer,
            (unsigned char *)runtime->host_arena + object->host_offset,
            object->size_bytes,
            SHADOWSPILL_TRANSFER_TO_DEVICE,
            runtime->h2d_stream
        ) != 0 || runtime->backend.record_event(
            runtime->backend.context,
            action->completion_event,
            runtime->h2d_stream
        ) != 0) {
        goto backend_failure;
    }
    action->has_completion_event = 1U;
    action->state = SHADOWSPILL_ACTION_IN_FLIGHT;
    object->allocation_id = allocation->allocation_id;
    object->generation = allocation->generation;
    object->device_version = object->host_version;
    object->readiness_event = action->completion_event;
    object->has_readiness_event = 1U;
    object->residency = SHADOWSPILL_OBJECT_PREFETCHING;
    ++runtime->transfers_to_device;
    runtime->bytes_to_device += object->size_bytes;
    return 1;

backend_failure:
    if (completion_created) {
        (void)runtime->backend.destroy_event(
            runtime->backend.context, action->completion_event
        );
    }
    allocation->release_task_id = action->task_id;
    shadowspill_release_allocation_locked(runtime, allocation);
    shadowspill_latch_failure_locked(
        runtime,
        SHADOWSPILL_RUNTIME_BACKEND_FAILURE,
        object->object_id,
        allocation->allocation_id,
        object->size_bytes
    );
    return -1;
}

static int progress_actions_locked(ShadowSpillRuntime *runtime) {
    int changed = 0;
    ShadowSpillQueuedAction *previous = NULL;
    ShadowSpillQueuedAction *action = runtime->action_head;
    while (action != NULL) {
        ShadowSpillQueuedAction *next = action->next;
        ShadowSpillObjectRecord *object = action->object;
        if (action->state == SHADOWSPILL_ACTION_QUEUED) {
            if (action->kind == SHADOWSPILL_RUNTIME_RELEASE) {
                int complete = 0;
                if (event_complete_locked(
                        runtime, action->fence->event,
                        object->object_id, &complete
                    ) != 0) {
                    return changed;
                }
                if (complete) {
                    ShadowSpillAllocationRecord *allocation =
                        shadowspill_find_allocation(
                            runtime, object->allocation_id
                        );
                    if (allocation == NULL) {
                        shadowspill_latch_failure_locked(
                            runtime,
                            SHADOWSPILL_RUNTIME_INVALID_STATE,
                            object->object_id,
                            object->allocation_id,
                            0U
                        );
                        return changed;
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
                            shadowspill_latch_failure_locked(
                                runtime,
                                SHADOWSPILL_RUNTIME_INVALID_STATE,
                                object->object_id,
                                allocation->allocation_id,
                                allocation->requested_bytes
                            );
                            return changed;
                        }
                        object->retired_generation = object->generation;
                        object->retired_device_pointer = allocation->pointer;
                        object->allocation_id = SHADOWSPILL_RUNTIME_NO_ID;
                        object->residency = object->host_current
                            ? SHADOWSPILL_OBJECT_HOST_ONLY
                            : SHADOWSPILL_OBJECT_RELEASED;
                        allocation->handoff_from_object_id =
                            SHADOWSPILL_RUNTIME_NO_ID;
                        allocation->handoff_to_object_id =
                            SHADOWSPILL_RUNTIME_NO_ID;
                        allocation->handoff_task_id =
                            SHADOWSPILL_RUNTIME_NO_ID;
                        complete_action_locked(runtime, previous, action);
                        changed = 1;
                        action = next;
                        continue;
                    }
                    object->retired_generation = object->generation;
                    object->retired_device_pointer = allocation->pointer;
                    allocation->release_task_id = action->task_id;
                    shadowspill_release_allocation_locked(runtime, allocation);
                    object->allocation_id = SHADOWSPILL_RUNTIME_NO_ID;
                    object->residency = object->host_current
                        ? SHADOWSPILL_OBJECT_HOST_ONLY
                        : SHADOWSPILL_OBJECT_RELEASED;
                    complete_action_locked(runtime, previous, action);
                    changed = 1;
                    action = next;
                    continue;
                }
            } else {
                int reserved = reserve_destination_locked(runtime, action);
                if (reserved < 0) {
                    return changed;
                }
                if (reserved == 0) {
                    previous = action;
                    action = next;
                    continue;
                }
                int dispatched = action->kind == SHADOWSPILL_RUNTIME_OFFLOAD
                    ? dispatch_offload_locked(runtime, action)
                    : dispatch_prefetch_locked(runtime, action);
                if (dispatched < 0) {
                    return changed;
                }
                changed |= dispatched;
                if (dispatched != 0) {
                    pthread_cond_broadcast(&runtime->condition);
                }
            }
        } else {
            int complete = 0;
            if (event_complete_locked(
                    runtime, action->completion_event,
                    object->object_id, &complete
                ) != 0) {
                return changed;
            }
            if (complete) {
                if (action->kind == SHADOWSPILL_RUNTIME_OFFLOAD) {
                    ShadowSpillAllocationRecord *allocation =
                        shadowspill_find_allocation(
                            runtime, object->allocation_id
                        );
                    if (allocation == NULL) {
                        shadowspill_latch_failure_locked(
                            runtime,
                            SHADOWSPILL_RUNTIME_INVALID_STATE,
                            object->object_id,
                            object->allocation_id,
                            0U
                        );
                        return changed;
                    }
                    object->retired_generation = object->generation;
                    object->retired_device_pointer = allocation->pointer;
                    allocation->release_task_id = action->task_id;
                    shadowspill_release_allocation_locked(runtime, allocation);
                    object->allocation_id = SHADOWSPILL_RUNTIME_NO_ID;
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
                    if (!object->retain_host_backing) {
                        uint64_t charged = object->size_bytes == 0U
                            ? 1U
                            : object->size_bytes;
                        if (shadowspill_range_free(
                                &runtime->host_ranges,
                                object->host_offset,
                                charged
                            ) != 0) {
                            shadowspill_latch_failure_locked(
                                runtime,
                                SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE,
                                object->object_id,
                                object->allocation_id,
                                charged
                            );
                            return changed;
                        }
                        object->has_host_range = 0U;
                        object->host_current = 0U;
                    }
                }
                complete_action_locked(runtime, previous, action);
                changed = 1;
                action = next;
                continue;
            }
        }
        previous = action;
        action = next;
    }
    return changed;
}

static void timed_wait_locked(ShadowSpillRuntime *runtime) {
    uint64_t wait_ns = runtime->progress_poll_nanoseconds;
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
    pthread_mutex_lock(&runtime->mutex);
    while (!runtime->worker_stop) {
        int changed = progress_retirements_locked(runtime);
        changed |= progress_actions_locked(runtime);
        if (runtime->failure.status != SHADOWSPILL_RUNTIME_OK) {
            pthread_cond_broadcast(&runtime->condition);
        }
        if (!changed && !runtime->worker_stop) {
            if (runtime->queued_actions == 0U &&
                runtime->pending_retirements == 0U) {
                pthread_cond_wait(&runtime->condition, &runtime->mutex);
            } else {
                timed_wait_locked(runtime);
            }
        }
    }
    pthread_mutex_unlock(&runtime->mutex);
    return NULL;
}
