#include "internal.h"

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
    lane->inflight_head = NULL;
    lane->inflight_tail = NULL;
    pthread_mutex_destroy(&lane->lock);
    lane->lock_initialized = 0U;
}

ShadowSpillTransferLane *shadowspill_transfer_lane_for_action(
    ShadowSpillRuntime *runtime,
    const ShadowSpillQueuedAction *action
) {
    if (runtime == NULL || action == NULL ||
        action->kind == SHADOWSPILL_RUNTIME_RELEASE) {
        return NULL;
    }
    return action->kind == SHADOWSPILL_RUNTIME_PREFETCH
        ? &runtime->fetch_lane
        : &runtime->evict_lane;
}

void shadowspill_transfer_lane_enqueue(
    ShadowSpillTransferLane *lane,
    ShadowSpillQueuedAction *action
) {
    if (lane == NULL || action == NULL) {
        return;
    }
    pthread_mutex_lock(&lane->lock);
    action->lane_previous = lane->pending_tail;
    action->lane_next = NULL;
    action->lane_state = SHADOWSPILL_LANE_PENDING;
    if (lane->pending_tail == NULL) {
        lane->pending_head = action;
    } else {
        lane->pending_tail->lane_next = action;
    }
    lane->pending_tail = action;
    pthread_mutex_unlock(&lane->lock);
}

int shadowspill_transfer_lane_claim(
    ShadowSpillTransferLane *lane,
    ShadowSpillQueuedAction *action
) {
    if (lane == NULL || action == NULL) {
        return 0;
    }
    pthread_mutex_lock(&lane->lock);
    if (lane->pending_head != action ||
        action->lane_state != SHADOWSPILL_LANE_PENDING) {
        pthread_mutex_unlock(&lane->lock);
        return 0;
    }
    lane->pending_head = action->lane_next;
    if (lane->pending_head == NULL) {
        lane->pending_tail = NULL;
    } else {
        lane->pending_head->lane_previous = NULL;
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
    pthread_mutex_unlock(&lane->lock);
    return 0;
}
