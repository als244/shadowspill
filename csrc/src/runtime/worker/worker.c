/* The worker thread: one pass over the queued actions, then the
 * dispatcher's newly published batch. */
#define _GNU_SOURCE

#include "internal.h"

#include <stdint.h>
#include <stdlib.h>

void shadowspill_notify_worker(ShadowSpillRuntime *runtime) {
    (void)runtime;
}

static void release_action_claim(
    ShadowSpillRuntime *runtime,
    ShadowSpillQueuedAction *action
) {
    pthread_mutex_lock(&runtime->actions.lock);
    action->processing = 0U;
    pthread_mutex_unlock(&runtime->actions.lock);
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
        const int action_status = shadowspill_action_handle(runtime, action);
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
        const int action_status = shadowspill_action_handle(runtime, action);
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
        &runtime->backend, "shadowspill.wkr"
    );
    while (atomic_load_explicit(
        &runtime->worker_stop, memory_order_acquire
    ) == 0U) {
        /* Observe one dispatcher batch and publish every fetch readiness event. */
        const int submission_changed =
            handle_newly_published_submission(runtime);
        uint64_t failure_object_id = SHADOWSPILL_RUNTIME_NO_ID;
        uint64_t failure_allocation_id = SHADOWSPILL_RUNTIME_NO_ID;
        /* Advance the FIFO completion frontier without holding pool locks. */
        const int completion_status = shadowspill_completion_poll(
            runtime, &failure_object_id, &failure_allocation_id
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
