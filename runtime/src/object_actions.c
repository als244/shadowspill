#include "internal.h"

typedef struct ShadowSpillProjectedObjectState {
    uint64_t version;
    uint8_t execution_current;
    uint8_t spill_current;
} ShadowSpillProjectedObjectState;

static ShadowSpillProjectedObjectState projected_state_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillObject *object
) {
    ShadowSpillQueuedAction *tail = object->action_tail;
    if (tail != NULL && tail->scheduled_version ==
            object->authoritative_version) {
        return (ShadowSpillProjectedObjectState){
            .version = tail->scheduled_version,
            .execution_current = tail->produces_current_execution,
            .spill_current = tail->produces_current_spill,
        };
    }

    const ShadowSpillObjectLocation *execution =
        shadowspill_execution_location(runtime, object);
    const ShadowSpillObjectLocation *spill = shadowspill_spill_location(
        runtime, object
    );
    return (ShadowSpillProjectedObjectState){
        .version = object->authoritative_version,
        .execution_current =
            (object->residency == SHADOWSPILL_OBJECT_EXECUTION_READY ||
             object->residency == SHADOWSPILL_OBJECT_PREFETCHING) &&
            execution->lease != NULL &&
            execution->version == object->authoritative_version,
        .spill_current = spill->lease != NULL && spill->current &&
            spill->version == object->authoritative_version,
    };
}

static void append_action_locked(
    ShadowSpillObject *object,
    ShadowSpillQueuedAction *action
) {
    action->object_previous = object->action_tail;
    action->object_next = NULL;
    if (object->action_tail == NULL) {
        object->action_head = action;
    } else {
        object->action_tail->object_next = action;
    }
    object->action_tail = action;
}

ShadowSpillRuntimeStatus shadowspill_object_schedule_action_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillObject *object,
    ShadowSpillQueuedAction *action
) {
    if (runtime == NULL || object == NULL || action == NULL ||
        action->object != object || action->object_previous != NULL ||
        action->object_next != NULL || action == object->action_head ||
        action == object->action_tail) {
        return SHADOWSPILL_RUNTIME_INVALID_STATE;
    }

    const ShadowSpillProjectedObjectState before = projected_state_locked(
        runtime, object
    );
    action->scheduled_version = before.version;
    action->produces_current_execution = 0U;
    action->produces_current_spill = 0U;

    switch (action->kind) {
        case SHADOWSPILL_RUNTIME_PREFETCH:
            if (!before.spill_current || before.execution_current) {
                return SHADOWSPILL_RUNTIME_PLAN_VIOLATION;
            }
            action->produces_current_execution = 1U;
            action->produces_current_spill =
                object->retain_spill_copy ? 1U : 0U;
            break;
        case SHADOWSPILL_RUNTIME_RELEASE:
            if (!before.execution_current) {
                return SHADOWSPILL_RUNTIME_PLAN_VIOLATION;
            }
            action->produces_current_spill = before.spill_current;
            break;
        case SHADOWSPILL_RUNTIME_OFFLOAD:
            if (!before.execution_current) {
                return SHADOWSPILL_RUNTIME_PLAN_VIOLATION;
            }
            action->produces_current_spill = 1U;
            break;
        default:
            return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }

    append_action_locked(object, action);
    return SHADOWSPILL_RUNTIME_OK;
}

int shadowspill_object_action_is_head_locked(
    const ShadowSpillObject *object,
    const ShadowSpillQueuedAction *action
) {
    return object != NULL && action != NULL && object->action_head == action;
}

int shadowspill_object_fetch_event_unpublished_locked(
    const ShadowSpillObject *object
) {
    if (object == NULL || !object->prefetch_pending ||
        object->residency == SHADOWSPILL_OBJECT_PREFETCHING) {
        return 0;
    }
    /*
     * A functional output may replace an in-flight fetch generation. In that
     * case the current execution lease is already authoritative and the old
     * fetch head must not stall its consumer.
     */
    if (object->residency == SHADOWSPILL_OBJECT_EXECUTION_READY &&
        (object->action_head == NULL ||
         object->action_head->kind == SHADOWSPILL_RUNTIME_PREFETCH)) {
        return 0;
    }
    return 1;
}

int shadowspill_object_remove_action_locked(
    ShadowSpillObject *object,
    ShadowSpillQueuedAction *action
) {
    if (object == NULL || action == NULL || action->object != object ||
        (action->object_previous == NULL && object->action_head != action) ||
        (action->object_next == NULL && object->action_tail != action)) {
        return -1;
    }
    if (action->object_previous == NULL) {
        object->action_head = action->object_next;
    } else {
        action->object_previous->object_next = action->object_next;
    }
    if (action->object_next == NULL) {
        object->action_tail = action->object_previous;
    } else {
        action->object_next->object_previous = action->object_previous;
    }
    action->object_previous = NULL;
    action->object_next = NULL;
    return 0;
}
