/* Retain one completion event per producer, shared by all dependent streams. */
#include "../internal.h"

ShadowSpillStatus shadowspill_task_publish_completion(
    const ShadowSpillTaskRecord *record, ShadowSpillEventLease *event
) {
    if (!record->completion_required) {
        return SHADOWSPILL_STATUS_OK;
    }
    ShadowSpillTaskRecord *task = (ShadowSpillTaskRecord *)record;
    ShadowSpillRuntime *runtime = task->plan_owner->runtime;
    const uint64_t invocation = shadowspill_current_task_invocation(runtime);
    if (event == NULL || invocation == 0U) {
        return SHADOWSPILL_STATUS_INVALID_STATE;
    }
    shadowspill_event_lease_retain(event);
    pthread_mutex_lock(&task->completion_lock);
    ShadowSpillEventLease *previous = task->completion_event;
    task->completion_event = event;
    task->completion_invocation = invocation;
    pthread_mutex_unlock(&task->completion_lock);
    if (previous != NULL && shadowspill_event_lease_release(runtime, previous) != 0) {
        return SHADOWSPILL_STATUS_BACKEND_FAILURE;
    }
    shadowspill_notify_worker(runtime);
    return SHADOWSPILL_STATUS_OK;
}

int shadowspill_task_snapshot_completion(
    ShadowSpillTaskRecord *task, uint64_t invocation, ShadowSpillEventLease **event
) {
    *event = NULL;
    if (invocation == 0U) {
        return -1;
    }
    pthread_mutex_lock(&task->completion_lock);
    int ready = 0;
    if (task->completion_invocation > invocation) {
        ready = -1;
    } else if (task->completion_invocation == invocation && task->completion_event != NULL) {
        *event = task->completion_event;
        shadowspill_event_lease_retain(*event);
        ready = 1;
    }
    pthread_mutex_unlock(&task->completion_lock);
    return ready;
}
