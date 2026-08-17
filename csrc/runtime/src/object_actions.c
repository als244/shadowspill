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

int shadowspill_object_reset_admitted_action_locked(
    ShadowSpillObject *object,
    ShadowSpillQueuedAction *action
) {
    if (object == NULL || action == NULL || !action->admitted ||
        action->object != object || action->object_previous != NULL ||
        action->object_next != NULL) {
        return -1;
    }
    /*
     * Keep the admitted identity immutable. Fixed-layout dependencies retain
     * direct pointers to these records across invocations, so an aggregate
     * zero-and-rebuild would expose transient NULL identity to concurrent
     * readers. Only per-invocation state is reset here, under the object lock.
     */
    action->activation_generation = 0U;
    action->state = SHADOWSPILL_ACTION_QUEUED;
    action->destination_lease = NULL;
    action->trigger_event = NULL;
    action->completion_event = NULL;
    action->dependency_event = NULL;
    action->owns_trace_label = 0U;
    action->has_completion_event = 0U;
    action->processing = 0U;
    action->active = 0U;
    action->produces_current_execution = 0U;
    action->produces_current_spill = 0U;
    action->scheduled_version = 0U;
    action->previous = NULL;
    action->next = NULL;
    action->object_previous = NULL;
    action->object_next = NULL;
    action->lane_previous = NULL;
    action->lane_next = NULL;
    action->lane_state = 0U;
    return 0;
}

void shadowspill_object_note_fetch_queued_locked(ShadowSpillObject *object) {
    if (object == NULL) {
        return;
    }
    (void)atomic_fetch_add_explicit(
        &object->unpublished_fetch_count, 1U, memory_order_release
    );
}

static int remove_unpublished_fetch_locked(ShadowSpillObject *object) {
    if (object == NULL || atomic_load_explicit(
            &object->unpublished_fetch_count, memory_order_acquire
        ) == 0U) {
        return -1;
    }
    (void)atomic_fetch_sub_explicit(
        &object->unpublished_fetch_count, 1U, memory_order_release
    );
    return 0;
}

int shadowspill_object_note_fetch_published_locked(
    ShadowSpillObject *object
) {
    return remove_unpublished_fetch_locked(object);
}

int shadowspill_object_note_fetch_discarded_locked(
    ShadowSpillObject *object
) {
    return remove_unpublished_fetch_locked(object);
}

int shadowspill_object_has_unpublished_fetch_locked(
    const ShadowSpillObject *object
) {
    return object != NULL && atomic_load_explicit(
        &object->unpublished_fetch_count, memory_order_acquire
    ) != 0U;
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
