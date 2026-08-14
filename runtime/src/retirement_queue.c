#include "internal.h"

#include <stdlib.h>

static void release_event_references(
    ShadowSpillRuntime *runtime,
    ShadowSpillEventLease **events,
    uint32_t event_count,
    uint64_t allocation_id
) {
    for (uint32_t index = 0U; index < event_count; ++index) {
        if (shadowspill_event_lease_release(runtime, events[index]) != 0) {
            shadowspill_latch_failure_locked(
                runtime,
                SHADOWSPILL_RUNTIME_BACKEND_FAILURE,
                SHADOWSPILL_RUNTIME_NO_ID,
                allocation_id,
                0U
            );
        }
    }
    free(events);
}

static void destroy_retirement_record(
    ShadowSpillRuntime *runtime,
    ShadowSpillRetirementRecord *record
) {
    if (record == NULL) {
        return;
    }
    release_event_references(
        runtime,
        record->events,
        record->event_count,
        record->allocation->allocation_id
    );
    shadowspill_release_task_fence_locked(runtime, record->fence);
    free(record);
}

int shadowspill_retirement_queue_initialize(
    ShadowSpillRetirementQueue *queue
) {
    if (queue == NULL) {
        return -1;
    }
    *queue = (ShadowSpillRetirementQueue){0};
    if (pthread_mutex_init(&queue->lock, NULL) != 0) {
        return -1;
    }
    queue->lock_initialized = 1U;
    atomic_init(&queue->count, 0U);
    return 0;
}

void shadowspill_retirement_queue_destroy(
    ShadowSpillRuntime *runtime,
    ShadowSpillRetirementQueue *queue
) {
    if (runtime == NULL || queue == NULL || !queue->lock_initialized) {
        return;
    }
    ShadowSpillRetirementRecord *record = queue->head;
    while (record != NULL) {
        ShadowSpillRetirementRecord *next = record->next;
        destroy_retirement_record(runtime, record);
        record = next;
    }
    queue->head = NULL;
    queue->tail = NULL;
    atomic_store_explicit(&queue->count, 0U, memory_order_release);
    pthread_mutex_destroy(&queue->lock);
    queue->lock_initialized = 0U;
}

ShadowSpillRuntimeStatus shadowspill_retirement_enqueue_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryLease *allocation
) {
    if (runtime == NULL || allocation == NULL ||
        !allocation->logical_freed || allocation->pointer == NULL ||
        (allocation->retirement_events == NULL &&
         allocation->retirement_fence == NULL)) {
        return SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    if (allocation->retirement_enqueued_generation == allocation->generation) {
        return SHADOWSPILL_RUNTIME_OK;
    }

    uint32_t event_count = 0U;
    for (ShadowSpillEventRecord *event = allocation->retirement_events;
         event != NULL; event = event->next) {
        if (event_count == UINT32_MAX) {
            return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
        }
        ++event_count;
    }
    ShadowSpillRetirementRecord *record = calloc(1U, sizeof(*record));
    ShadowSpillEventLease **events = event_count == 0U
        ? NULL
        : calloc((size_t)event_count, sizeof(*events));
    if (record == NULL || (event_count != 0U && events == NULL)) {
        free(record);
        free(events);
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    uint32_t index = 0U;
    for (ShadowSpillEventRecord *event = allocation->retirement_events;
         event != NULL; event = event->next) {
        events[index++] = event->event;
        shadowspill_event_lease_retain(event->event);
    }
    record->allocation = allocation;
    record->allocation_generation = allocation->generation;
    record->events = events;
    record->event_count = event_count;
    record->fence = allocation->retirement_fence;
    shadowspill_retain_task_fence(record->fence);

    ShadowSpillRetirementQueue *queue = &runtime->retirements;
    pthread_mutex_lock(&queue->lock);
    if (queue->tail == NULL) {
        queue->head = record;
    } else {
        queue->tail->next = record;
    }
    queue->tail = record;
    (void)atomic_fetch_add_explicit(
        &queue->count, 1U, memory_order_release
    );
    pthread_mutex_unlock(&queue->lock);
    allocation->retirement_enqueued_generation = allocation->generation;
    return SHADOWSPILL_RUNTIME_OK;
}

static int retirement_complete(const ShadowSpillRetirementRecord *record) {
    if (record->fence != NULL && atomic_load_explicit(
            &record->fence->event->backend_complete, memory_order_acquire
        ) == 0U) {
        return 0;
    }
    for (uint32_t index = 0U; index < record->event_count; ++index) {
        if (atomic_load_explicit(
                &record->events[index]->backend_complete,
                memory_order_acquire
            ) == 0U) {
            return 0;
        }
    }
    return 1;
}

static void release_owned_retirement_requirements(
    ShadowSpillRuntime *runtime,
    ShadowSpillEventRecord *events,
    ShadowSpillTaskFence *fence,
    uint64_t allocation_id
) {
    while (events != NULL) {
        ShadowSpillEventRecord *next = events->next;
        if (shadowspill_event_lease_release(runtime, events->event) != 0) {
            shadowspill_latch_failure_locked(
                runtime,
                SHADOWSPILL_RUNTIME_BACKEND_FAILURE,
                SHADOWSPILL_RUNTIME_NO_ID,
                allocation_id,
                0U
            );
        }
        free(events);
        events = next;
    }
    shadowspill_release_task_fence_locked(runtime, fence);
}

static void append_retry(
    ShadowSpillRetirementRecord **head,
    ShadowSpillRetirementRecord **tail,
    ShadowSpillRetirementRecord *record
) {
    record->next = NULL;
    if (*tail == NULL) {
        *head = record;
    } else {
        (*tail)->next = record;
    }
    *tail = record;
}

ShadowSpillRetirementWork shadowspill_handle_retirements(
    ShadowSpillRuntime *runtime
) {
    ShadowSpillRetirementWork work = {0};
    if (runtime == NULL) {
        return work;
    }
    ShadowSpillRetirementQueue *queue = &runtime->retirements;
    pthread_mutex_lock(&queue->lock);
    ShadowSpillRetirementRecord *record = queue->head;
    queue->head = NULL;
    queue->tail = NULL;
    pthread_mutex_unlock(&queue->lock);

    ShadowSpillRetirementRecord *retry_head = NULL;
    ShadowSpillRetirementRecord *retry_tail = NULL;
    while (record != NULL) {
        ShadowSpillRetirementRecord *next = record->next;
        const int complete = retirement_complete(record);
        if (!complete) {
            append_retry(&retry_head, &retry_tail, record);
            record = next;
            continue;
        }

        /*
         * Completion discovery never owns the memory pool.  Background
         * reclamation enters only through the pool's generic priority-aware
         * interface, which declines ownership while a foreground allocator
         * client is waiting.  Requeue the remaining completed records and let
         * the foreground call run before another background attempt.
         */
        if (!shadowspill_memory_pool_try_lock_reclamation(
                shadowspill_execution_pool(runtime)
            )) {
            work.pool_busy = 1U;
            append_retry(&retry_head, &retry_tail, record);
            while (next != NULL) {
                ShadowSpillRetirementRecord *remaining = next->next;
                append_retry(&retry_head, &retry_tail, next);
                next = remaining;
            }
            break;
        }
        ShadowSpillEventRecord *owned_events = NULL;
        ShadowSpillTaskFence *owned_fence = NULL;
        int released = 0;
        ShadowSpillMemoryLease *allocation = record->allocation;
        if (allocation->generation == record->allocation_generation &&
            allocation->logical_freed && allocation->pointer != NULL &&
            allocation->retirement_enqueued_generation ==
                record->allocation_generation) {
            owned_events = allocation->retirement_events;
            owned_fence = allocation->retirement_fence;
            allocation->retirement_events = NULL;
            allocation->retirement_fence = NULL;
            shadowspill_append_trace_event_locked(
                runtime,
                SHADOWSPILL_TRACE_RETIREMENT_COMPLETED,
                allocation->release_task_id,
                SHADOWSPILL_RUNTIME_NO_ID,
                allocation->allocation_id,
                allocation->requested_bytes,
                allocation->offset,
                allocation->charged_bytes
            );
            shadowspill_release_execution_lease_locked(runtime, allocation);
            released = 1;
        }
        shadowspill_memory_pool_unlock_reclamation(shadowspill_execution_pool(runtime));

        release_owned_retirement_requirements(
            runtime,
            owned_events,
            owned_fence,
            record->allocation->allocation_id
        );
        if (released && atomic_fetch_sub_explicit(
                &runtime->pending_retirements, 1U, memory_order_release
            ) == 1U) {
            shadowspill_idle_notify(runtime);
        }
        destroy_retirement_record(runtime, record);
        (void)atomic_fetch_sub_explicit(
            &queue->count, 1U, memory_order_release
        );
        record = next;
    }

    if (retry_head != NULL) {
        pthread_mutex_lock(&queue->lock);
        if (queue->tail == NULL) {
            queue->head = retry_head;
        } else {
            queue->tail->next = retry_head;
        }
        queue->tail = retry_tail;
        pthread_mutex_unlock(&queue->lock);
    }
    return work;
}

int shadowspill_has_actionable_retirement(ShadowSpillRuntime *runtime) {
    return runtime != NULL && atomic_load_explicit(
        &runtime->retirements.count, memory_order_acquire
    ) != 0U;
}
