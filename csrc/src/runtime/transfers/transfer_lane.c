#include "../internal.h"

enum {
    SHADOWSPILL_LANE_NONE = 0,
    SHADOWSPILL_LANE_PENDING = 1,
    SHADOWSPILL_LANE_INFLIGHT = 2,
};

int shadowspill_transfer_lane_initialize(ShadowSpillTransferLane *lane) {
    if (lane == NULL || pthread_mutex_init(&lane->lock, NULL) != 0) {
        return -1;
    }
    lane->lock_initialized = 1U;
    return 0;
}

void shadowspill_transfer_lane_destroy(ShadowSpillTransferLane *lane) {
    if (lane == NULL || !lane->lock_initialized) {
        return;
    }
    lane->pending_head = NULL;
    lane->pending_tail = NULL;
    lane->background_head = NULL;
    lane->background_tail = NULL;
    lane->inflight_head = NULL;
    lane->inflight_tail = NULL;
    lane->background_inflight_bytes = 0U;
    pthread_mutex_destroy(&lane->lock);
    lane->lock_initialized = 0U;
}

ShadowSpillTransferLane *shadowspill_transfer_lane_for_action(
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
    return &action->route->transfers;
}

void shadowspill_transfer_lane_enqueue(
    ShadowSpillTransferLane *lane,
    ShadowSpillQueuedAction *action
) {
    if (lane == NULL || action == NULL) {
        return;
    }
    pthread_mutex_lock(&lane->lock);
    ShadowSpillQueuedAction **head = action->background
        ? &lane->background_head : &lane->pending_head;
    ShadowSpillQueuedAction **tail = action->background
        ? &lane->background_tail : &lane->pending_tail;
    action->lane_previous = *tail;
    action->lane_next = NULL;
    action->lane_state = SHADOWSPILL_LANE_PENDING;
    if (*tail == NULL) {
        *head = action;
    } else {
        (*tail)->lane_next = action;
    }
    *tail = action;
    pthread_mutex_unlock(&lane->lock);
}

/* Whether the window admits one more background copy of this size: always
   when the lane holds none, so a copy larger than the window runs alone. */
static int background_window_admits(
    const ShadowSpillTransferLane *lane, uint64_t bytes
) {
    return lane->background_window_bytes == 0U ||
        lane->background_inflight_bytes == 0U ||
        lane->background_inflight_bytes + bytes <= lane->background_window_bytes;
}

int shadowspill_transfer_lane_claim(
    ShadowSpillTransferLane *lane,
    ShadowSpillQueuedAction *action
) {
    if (lane == NULL || action == NULL) {
        return 0;
    }
    pthread_mutex_lock(&lane->lock);
    ShadowSpillQueuedAction **head = action->background
        ? &lane->background_head : &lane->pending_head;
    ShadowSpillQueuedAction **tail = action->background
        ? &lane->background_tail : &lane->pending_tail;
    if (*head != action || action->lane_state != SHADOWSPILL_LANE_PENDING ||
        (action->background &&
         !background_window_admits(
             lane, action->object == NULL ? 0U : action->object->size_bytes
         ))) {
        pthread_mutex_unlock(&lane->lock);
        return 0;
    }
    *head = action->lane_next;
    if (*head == NULL) {
        *tail = NULL;
    } else {
        (*head)->lane_previous = NULL;
    }
    action->lane_previous = NULL;
    action->lane_next = NULL;
    action->lane_state = SHADOWSPILL_LANE_NONE;
    pthread_mutex_unlock(&lane->lock);
    return 1;
}

void shadowspill_transfer_lane_publish_inflight(
    ShadowSpillTransferLane *lane,
    ShadowSpillQueuedAction *action
) {
    if (lane == NULL || action == NULL) {
        return;
    }
    pthread_mutex_lock(&lane->lock);
    action->lane_previous = lane->inflight_tail;
    action->lane_next = NULL;
    action->lane_state = SHADOWSPILL_LANE_INFLIGHT;
    if (lane->inflight_tail == NULL) {
        lane->inflight_head = action;
    } else {
        lane->inflight_tail->lane_next = action;
    }
    lane->inflight_tail = action;
    if (action->background) {
        lane->background_inflight_bytes +=
            action->object == NULL ? 0U : action->object->size_bytes;
    }
    pthread_mutex_unlock(&lane->lock);
}

int shadowspill_transfer_lane_is_inflight_head(
    ShadowSpillTransferLane *lane,
    const ShadowSpillQueuedAction *action
) {
    if (lane == NULL || action == NULL) {
        return 0;
    }
    pthread_mutex_lock(&lane->lock);
    const int is_head = lane->inflight_head == action &&
        action->lane_state == SHADOWSPILL_LANE_INFLIGHT;
    pthread_mutex_unlock(&lane->lock);
    return is_head;
}

int shadowspill_transfer_lane_complete(
    ShadowSpillTransferLane *lane,
    ShadowSpillQueuedAction *action
) {
    if (lane == NULL || action == NULL) {
        return -1;
    }
    pthread_mutex_lock(&lane->lock);
    if (lane->inflight_head != action ||
        action->lane_state != SHADOWSPILL_LANE_INFLIGHT) {
        pthread_mutex_unlock(&lane->lock);
        return -1;
    }
    lane->inflight_head = action->lane_next;
    if (lane->inflight_head == NULL) {
        lane->inflight_tail = NULL;
    } else {
        lane->inflight_head->lane_previous = NULL;
    }
    action->lane_previous = NULL;
    action->lane_next = NULL;
    action->lane_state = SHADOWSPILL_LANE_NONE;
    if (action->background) {
        const uint64_t bytes = action->object == NULL
            ? 0U : action->object->size_bytes;
        lane->background_inflight_bytes = bytes > lane->background_inflight_bytes
            ? 0U : lane->background_inflight_bytes - bytes;
    }
    pthread_mutex_unlock(&lane->lock);
    return 0;
}
