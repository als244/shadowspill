#include "internal.h"

#include <stdlib.h>
#include <string.h>

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

static void release_retirement_requirements(
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
        record->allocation_id
    );
    if (shadowspill_event_lease_release(
            runtime, record->task_completion_event
        ) != 0) {
        shadowspill_latch_failure_locked(
            runtime,
            SHADOWSPILL_RUNTIME_BACKEND_FAILURE,
            SHADOWSPILL_RUNTIME_NO_ID,
            record->allocation_id,
            0U
        );
    }
    record->events = NULL;
    record->event_count = 0U;
    record->task_completion_event = NULL;
}

static ShadowSpillRetirementRecord *acquire_retirement_record(
    ShadowSpillRetirementQueue *queue
) {
    pthread_mutex_lock(&queue->lock);
    ShadowSpillRetirementRecord *record = queue->free_head;
    const uint8_t sealed = queue->sealed;
    if (record != NULL) {
        queue->free_head = record->free_next;
        record->free_next = NULL;
        --queue->available;
        ++queue->in_use;
        if (queue->in_use > queue->peak_in_use) {
            queue->peak_in_use = queue->in_use;
        }
    } else if (sealed) {
        ++queue->growth_rejections;
    }
    pthread_mutex_unlock(&queue->lock);
    if (record != NULL || sealed) {
        return record;
    }
    return calloc(1U, sizeof(*record));
}

static void release_retirement_record(
    ShadowSpillRetirementQueue *queue,
    ShadowSpillRetirementRecord *record
) {
    if (!record->pool_owned) {
        free(record);
        return;
    }
    memset(record, 0, sizeof(*record));
    record->pool_owned = 1U;
    pthread_mutex_lock(&queue->lock);
    record->free_next = queue->free_head;
    queue->free_head = record;
    ++queue->available;
    if (queue->in_use != 0U) {
        --queue->in_use;
    }
    pthread_mutex_unlock(&queue->lock);
}

static void destroy_retirement_record(
    ShadowSpillRuntime *runtime,
    ShadowSpillRetirementRecord *record
) {
    if (record == NULL) {
        return;
    }
    release_retirement_requirements(runtime, record);
    release_retirement_record(&runtime->retirements, record);
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

ShadowSpillRuntimeStatus shadowspill_retirement_queue_reserve(
    ShadowSpillRetirementQueue *queue,
    uint64_t minimum_free_records
) {
    if (queue == NULL || !queue->lock_initialized ||
        minimum_free_records == 0U) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&queue->lock);
    if (queue->available >= minimum_free_records) {
        queue->sealed = 1U;
        pthread_mutex_unlock(&queue->lock);
        return SHADOWSPILL_RUNTIME_OK;
    }
    const uint64_t additional = minimum_free_records - queue->available;
    pthread_mutex_unlock(&queue->lock);

    if (additional > SIZE_MAX / sizeof(ShadowSpillRetirementRecord)) {
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    ShadowSpillRetirementRecordBlock *block = calloc(1U, sizeof(*block));
    ShadowSpillRetirementRecord *records = calloc(
        (size_t)additional, sizeof(*records)
    );
    if (block == NULL || records == NULL) {
        free(records);
        free(block);
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    block->records = records;
    block->count = additional;

    pthread_mutex_lock(&queue->lock);
    block->next = queue->blocks;
    queue->blocks = block;
    for (uint64_t index = 0U; index < additional; ++index) {
        records[index].pool_owned = 1U;
        records[index].free_next = queue->free_head;
        queue->free_head = &records[index];
    }
    queue->capacity += additional;
    queue->available += additional;
    queue->sealed = 1U;
    pthread_mutex_unlock(&queue->lock);
    return SHADOWSPILL_RUNTIME_OK;
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
        release_retirement_requirements(runtime, record);
        if (!record->pool_owned) {
            free(record);
        }
        record = next;
    }
    ShadowSpillRetirementRecordBlock *block = queue->blocks;
    while (block != NULL) {
        ShadowSpillRetirementRecordBlock *next = block->next;
        free(block->records);
        free(block);
        block = next;
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
         allocation->retirement_event == NULL)) {
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
    ShadowSpillRetirementRecord *record = acquire_retirement_record(
        &runtime->retirements
    );
    ShadowSpillEventLease **events = event_count == 0U
        ? NULL
        : calloc((size_t)event_count, sizeof(*events));
    if (record == NULL || (event_count != 0U && events == NULL)) {
        if (record != NULL) {
            release_retirement_record(&runtime->retirements, record);
        }
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
    record->pool = allocation->pool;
    record->allocation_id = allocation->allocation_id;
    record->allocation_generation = allocation->generation;
    record->events = events;
    record->event_count = event_count;
    record->task_completion_event = allocation->retirement_event;
    shadowspill_event_lease_retain(record->task_completion_event);

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
    if (record->task_completion_event != NULL &&
        !shadowspill_event_lease_is_complete(record->task_completion_event)) {
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
    ShadowSpillEventLease *task_completion_event,
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
    if (shadowspill_event_lease_release(
            runtime, task_completion_event
        ) != 0) {
        shadowspill_latch_failure_locked(
            runtime,
            SHADOWSPILL_RUNTIME_BACKEND_FAILURE,
            SHADOWSPILL_RUNTIME_NO_ID,
            allocation_id,
            0U
        );
    }
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
        ShadowSpillMemoryPool *pool = record->pool;
        if (!shadowspill_memory_pool_try_lock_reclamation(pool)) {
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
        ShadowSpillEventLease *owned_task_completion_event = NULL;
        int released = 0;
        ShadowSpillMemoryLease *allocation = record->allocation;
        if (allocation->pool == pool &&
            allocation->generation == record->allocation_generation &&
            allocation->logical_freed && allocation->pointer != NULL &&
            allocation->retirement_enqueued_generation ==
                record->allocation_generation) {
            owned_events = allocation->retirement_events;
            owned_task_completion_event = allocation->retirement_event;
            allocation->retirement_events = NULL;
            allocation->retirement_event = NULL;
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
        shadowspill_memory_pool_unlock_reclamation(pool);

        release_owned_retirement_requirements(
            runtime,
            owned_events,
            owned_task_completion_event,
            record->allocation_id
        );
        if (released && atomic_fetch_sub_explicit(
                &runtime->pending_retirements, 1U, memory_order_release
            ) == 1U) {
            shadowspill_idle_notify(runtime);
        }
        if (released) {
            (void)atomic_fetch_sub_explicit(
                &pool->pending_retirements, 1U, memory_order_release
            );
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
