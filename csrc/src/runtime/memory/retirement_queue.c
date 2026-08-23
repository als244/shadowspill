#include "../internal.h"

#include <stdlib.h>
#include <string.h>

static int release_event_requirements(
    ShadowSpillRuntime *runtime,
    ShadowSpillLeaseUseRecord *requirements,
    uint64_t allocation_id
) {
    int status = 0;
    for (ShadowSpillLeaseUseRecord *requirement = requirements;
         requirement != NULL; requirement = requirement->next) {
        if (requirement->event == NULL) {
            continue;
        }
        if (shadowspill_event_lease_release(
                runtime, requirement->event
            ) != 0) {
            shadowspill_latch_failure_locked(
                runtime,
                SHADOWSPILL_STATUS_BACKEND_FAILURE,
                SHADOWSPILL_RUNTIME_NO_ID,
                allocation_id,
                0U
            );
            status = -1;
            continue;
        }
        requirement->event = NULL;
    }
    return status;
}

static void release_retirement_requirements(
    ShadowSpillRuntime *runtime,
    ShadowSpillRetirementRecord *record
) {
    if (record == NULL) {
        return;
    }
    (void)release_event_requirements(
        runtime,
        record->requirements,
        record->allocation_id
    );
    if (record->requirements != NULL) {
        shadowspill_memory_pool_lock_foreground(record->pool);
        const int release_status =
            shadowspill_memory_pool_release_use_records_locked(
                record->pool, record->requirements
            );
        shadowspill_memory_pool_unlock_foreground(record->pool);
        if (release_status != 0) {
            shadowspill_latch_pool_failure_locked(
                runtime,
                record->pool,
                SHADOWSPILL_STATUS_INVALID_STATE,
                SHADOWSPILL_RUNTIME_NO_ID,
                record->allocation_id,
                0U
            );
        } else {
            record->requirements = NULL;
        }
    }
    if (shadowspill_event_lease_release(
            runtime, record->task_completion_event
        ) != 0) {
        shadowspill_latch_failure_locked(
            runtime,
            SHADOWSPILL_STATUS_BACKEND_FAILURE,
            SHADOWSPILL_RUNTIME_NO_ID,
            record->allocation_id,
            0U
        );
    }
    record->task_completion_event = NULL;
}

static void detach_borrowed_requirements_for_teardown(
    ShadowSpillRetirementRecord *record
) {
    if (record == NULL || record->allocation == NULL ||
        record->pool == NULL) {
        return;
    }
    ShadowSpillMemoryLease *allocation = record->allocation;
    shadowspill_memory_pool_lock_foreground(record->pool);
    if (allocation->pool == record->pool &&
        allocation->generation == record->allocation_generation) {
        if (allocation->retirement_requirements == record->requirements) {
            allocation->retirement_requirements = NULL;
        }
        if (allocation->uses == record->requirements) {
            allocation->uses = NULL;
        }
        if (allocation->retirement_event == record->task_completion_event) {
            allocation->retirement_event = NULL;
        }
    }
    shadowspill_memory_pool_unlock_foreground(record->pool);
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
    ShadowSpillMemoryLease *allocation = record->allocation;
    ShadowSpillPlan *plan_owner = record->plan_owner;
    record->allocation = NULL;
    record->plan_owner = NULL;
    release_retirement_requirements(runtime, record);
    release_retirement_record(&runtime->retirements, record);
    shadowspill_memory_lease_release(allocation);
    if (plan_owner != NULL) {
        (void)atomic_fetch_sub_explicit(
            &plan_owner->pending_retirements, 1U, memory_order_release
        );
    }
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

ShadowSpillStatus shadowspill_retirement_queue_reserve(
    ShadowSpillRetirementQueue *queue,
    uint64_t minimum_free_records
) {
    if (queue == NULL || !queue->lock_initialized ||
        minimum_free_records == 0U) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&queue->lock);
    if (queue->available >= minimum_free_records) {
        queue->sealed = 1U;
        pthread_mutex_unlock(&queue->lock);
        return SHADOWSPILL_STATUS_OK;
    }
    const uint64_t additional = minimum_free_records - queue->available;
    pthread_mutex_unlock(&queue->lock);

    if (additional > SIZE_MAX / sizeof(ShadowSpillRetirementRecord)) {
        return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    }
    ShadowSpillRetirementRecordBlock *block = calloc(1U, sizeof(*block));
    ShadowSpillRetirementRecord *records = calloc(
        (size_t)additional, sizeof(*records)
    );
    if (block == NULL || records == NULL) {
        free(records);
        free(block);
        return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
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
    return SHADOWSPILL_STATUS_OK;
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
        ShadowSpillMemoryLease *allocation = record->allocation;
        ShadowSpillPlan *plan_owner = record->plan_owner;
        detach_borrowed_requirements_for_teardown(record);
        record->allocation = NULL;
        record->plan_owner = NULL;
        release_retirement_requirements(runtime, record);
        shadowspill_memory_lease_release(allocation);
        if (plan_owner != NULL) {
            (void)atomic_fetch_sub_explicit(
                &plan_owner->pending_retirements, 1U, memory_order_release
            );
        }
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

ShadowSpillStatus shadowspill_retirement_enqueue_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryLease *allocation
) {
    if (runtime == NULL || allocation == NULL ||
        !allocation->logical_freed || allocation->pointer == NULL ||
        (allocation->retirement_requirements == NULL &&
         allocation->retirement_event == NULL)) {
        return SHADOWSPILL_STATUS_INVALID_STATE;
    }
    if (allocation->retirement_enqueued_generation == allocation->generation) {
        return SHADOWSPILL_STATUS_OK;
    }

    ShadowSpillRetirementRecord *record = acquire_retirement_record(
        &runtime->retirements
    );
    if (record == NULL) {
        return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    }
    record->allocation = allocation;
    record->plan_owner = shadowspill_current_plan(runtime);
    shadowspill_memory_lease_retain(allocation);
    record->pool = allocation->pool;
    record->allocation_id = allocation->allocation_id;
    record->allocation_generation = allocation->generation;
    record->requirements = allocation->retirement_requirements;
    record->task_completion_event = allocation->retirement_event;
    allocation->retirement_enqueued_generation = allocation->generation;
    if (record->plan_owner != NULL) {
        (void)atomic_fetch_add_explicit(
            &record->plan_owner->pending_retirements,
            1U,
            memory_order_release
        );
    }

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
    return SHADOWSPILL_STATUS_OK;
}

static int retirement_complete(const ShadowSpillRetirementRecord *record) {
    if (record->task_completion_event != NULL &&
        !shadowspill_event_lease_is_complete(record->task_completion_event)) {
        return 0;
    }
    for (const ShadowSpillLeaseUseRecord *requirement = record->requirements;
         requirement != NULL; requirement = requirement->next) {
        if (requirement->event != NULL && atomic_load_explicit(
                &requirement->event->backend_complete,
                memory_order_acquire
            ) == 0U) {
            return 0;
        }
    }
    return 1;
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

        /* Retired event handles are backend resources, so release them before
         * entering the pool. The immutable requirement records stay owned by
         * this queue entry until the same reclamation critical section that
         * frees the physical range. */
        if (release_event_requirements(
                runtime, record->requirements, record->allocation_id
            ) != 0) {
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
        int released = 0;
        ShadowSpillMemoryLease *allocation = record->allocation;
        if (allocation->pool == pool &&
            allocation->generation == record->allocation_generation &&
            allocation->logical_freed && allocation->pointer != NULL &&
            allocation->retirement_enqueued_generation ==
                record->allocation_generation) {
            if (allocation->uses == record->requirements) {
                allocation->uses = NULL;
            }
            allocation->retirement_requirements = NULL;
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
        if (allocation->pool == pool &&
            allocation->generation == record->allocation_generation) {
            if (allocation->retirement_requirements ==
                record->requirements) {
                allocation->retirement_requirements = NULL;
            }
            if (allocation->uses == record->requirements) {
                allocation->uses = NULL;
            }
            if (allocation->retirement_event ==
                record->task_completion_event) {
                allocation->retirement_event = NULL;
            }
        }
        if (record->requirements != NULL) {
            if (shadowspill_memory_pool_release_use_records_locked(
                    pool, record->requirements
                ) != 0) {
                shadowspill_latch_pool_failure_locked(
                    runtime,
                    pool,
                    SHADOWSPILL_STATUS_INVALID_STATE,
                    SHADOWSPILL_RUNTIME_NO_ID,
                    record->allocation_id,
                    0U
                );
            } else {
                record->requirements = NULL;
            }
        }
        shadowspill_memory_pool_unlock_reclamation(pool);

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
