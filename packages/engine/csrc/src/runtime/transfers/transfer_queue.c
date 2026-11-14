#include "../internal.h"


int shadowspill_transfer_queue_initialize(ShadowSpillTransferQueue *queue) {
    if (queue == NULL || pthread_mutex_init(&queue->lock, NULL) != 0) {
        return -1;
    }
    queue->lock_initialized = 1U;
    return 0;
}

void shadowspill_transfer_queue_destroy(ShadowSpillTransferQueue *queue) {
    if (queue == NULL || !queue->lock_initialized) {
        return;
    }
    queue->pending_head = NULL;
    queue->pending_tail = NULL;
    queue->background_head = NULL;
    queue->background_tail = NULL;
    queue->inflight_head = NULL;
    queue->inflight_tail = NULL;
    queue->background_inflight_bytes = 0U;
    atomic_store_explicit(&queue->inflight_count, 0U, memory_order_relaxed);
    pthread_mutex_destroy(&queue->lock);
    queue->lock_initialized = 0U;
}

ShadowSpillTransferQueue *shadowspill_transfer_queue_for_action(
    ShadowSpillRuntime *runtime,
    const ShadowSpillQueuedAction *action
) {
    (void)runtime;
    if (action == NULL || action->route == NULL ||
        action->kind == SHADOWSPILL_RUNTIME_RELEASE ||
        (action->kind == SHADOWSPILL_RUNTIME_WRITE_BACK &&
         action->skips_copy)) {
        return NULL;
    }
    return &action->route->queue;
}

void shadowspill_transfer_queue_enqueue(
    ShadowSpillTransferQueue *queue,
    ShadowSpillQueuedAction *action
) {
    if (queue == NULL || action == NULL) {
        return;
    }
    pthread_mutex_lock(&queue->lock);
    ShadowSpillQueuedAction **head = action->background
        ? &queue->background_head : &queue->pending_head;
    ShadowSpillQueuedAction **tail = action->background
        ? &queue->background_tail : &queue->pending_tail;
    action->queue_previous = *tail;
    action->queue_next = NULL;
    action->queue_state = SHADOWSPILL_QUEUE_PENDING;
    if (*tail == NULL) {
        *head = action;
    } else {
        (*tail)->queue_next = action;
    }
    *tail = action;
    pthread_mutex_unlock(&queue->lock);
}

/* Whether the window admits one more background copy of this size: always
   when the queue holds none, so a copy larger than the window runs alone. */
static int background_window_admits(
    const ShadowSpillTransferQueue *queue, uint64_t bytes
) {
    return queue->background_window_bytes == 0U ||
        queue->background_inflight_bytes == 0U ||
        queue->background_inflight_bytes + bytes <= queue->background_window_bytes;
}

int shadowspill_transfer_queue_claim(
    ShadowSpillTransferQueue *queue,
    ShadowSpillQueuedAction *action
) {
    if (queue == NULL || action == NULL) {
        return 0;
    }
    pthread_mutex_lock(&queue->lock);
    ShadowSpillQueuedAction **head = action->background
        ? &queue->background_head : &queue->pending_head;
    ShadowSpillQueuedAction **tail = action->background
        ? &queue->background_tail : &queue->pending_tail;
    if (*head != action || action->queue_state != SHADOWSPILL_QUEUE_PENDING ||
        (action->background &&
         !background_window_admits(
             queue, action->object == NULL ? 0U : action->object->size_bytes
         ))) {
        pthread_mutex_unlock(&queue->lock);
        return 0;
    }
    *head = action->queue_next;
    if (*head == NULL) {
        *tail = NULL;
    } else {
        (*head)->queue_previous = NULL;
    }
    action->queue_previous = NULL;
    action->queue_next = NULL;
    action->queue_state = SHADOWSPILL_QUEUE_NONE;
    pthread_mutex_unlock(&queue->lock);
    return 1;
}

/*
 * Put a claimed action back at the head it came from, still pending. For a lane
 * whose `wait` cannot enqueue a dependency and asks to be retried: the action
 * has already been popped, and it has to go back in front rather than behind,
 * or the queue's order stops being the order the boundaries triggered.
 */
void shadowspill_transfer_queue_return(
    ShadowSpillTransferQueue *queue,
    ShadowSpillQueuedAction *action
) {
    if (queue == NULL || action == NULL) {
        return;
    }
    pthread_mutex_lock(&queue->lock);
    ShadowSpillQueuedAction **head = action->background
        ? &queue->background_head : &queue->pending_head;
    ShadowSpillQueuedAction **tail = action->background
        ? &queue->background_tail : &queue->pending_tail;
    action->queue_previous = NULL;
    action->queue_next = *head;
    if (*head != NULL) {
        (*head)->queue_previous = action;
    } else {
        *tail = action;
    }
    *head = action;
    action->queue_state = SHADOWSPILL_QUEUE_PENDING;
    pthread_mutex_unlock(&queue->lock);
}

void shadowspill_transfer_queue_publish_inflight(
    ShadowSpillTransferQueue *queue,
    ShadowSpillQueuedAction *action
) {
    if (queue == NULL || action == NULL) {
        return;
    }
    pthread_mutex_lock(&queue->lock);
    action->queue_previous = queue->inflight_tail;
    action->queue_next = NULL;
    action->queue_state = SHADOWSPILL_QUEUE_INFLIGHT;
    if (queue->inflight_tail == NULL) {
        queue->inflight_head = action;
    } else {
        queue->inflight_tail->queue_next = action;
    }
    queue->inflight_tail = action;
    atomic_fetch_add_explicit(&queue->inflight_count, 1U, memory_order_relaxed);
    if (action->background) {
        queue->background_inflight_bytes +=
            action->object == NULL ? 0U : action->object->size_bytes;
    }
    pthread_mutex_unlock(&queue->lock);
}

int shadowspill_transfer_queue_is_inflight_head(
    ShadowSpillTransferQueue *queue,
    const ShadowSpillQueuedAction *action
) {
    if (queue == NULL || action == NULL) {
        return 0;
    }
    pthread_mutex_lock(&queue->lock);
    const int is_head = queue->inflight_head == action &&
        action->queue_state == SHADOWSPILL_QUEUE_INFLIGHT;
    pthread_mutex_unlock(&queue->lock);
    return is_head;
}

int shadowspill_transfer_queue_complete(
    ShadowSpillTransferQueue *queue,
    ShadowSpillQueuedAction *action
) {
    if (queue == NULL || action == NULL) {
        return -1;
    }
    pthread_mutex_lock(&queue->lock);
    if (queue->inflight_head != action ||
        action->queue_state != SHADOWSPILL_QUEUE_INFLIGHT) {
        pthread_mutex_unlock(&queue->lock);
        return -1;
    }
    queue->inflight_head = action->queue_next;
    if (queue->inflight_head == NULL) {
        queue->inflight_tail = NULL;
    } else {
        queue->inflight_head->queue_previous = NULL;
    }
    action->queue_previous = NULL;
    action->queue_next = NULL;
    action->queue_state = SHADOWSPILL_QUEUE_NONE;
    atomic_fetch_sub_explicit(&queue->inflight_count, 1U, memory_order_relaxed);
    if (action->background) {
        const uint64_t bytes = action->object == NULL
            ? 0U : action->object->size_bytes;
        queue->background_inflight_bytes = bytes > queue->background_inflight_bytes
            ? 0U : queue->background_inflight_bytes - bytes;
    }
    pthread_mutex_unlock(&queue->lock);
    return 0;
}
