
#include <stdatomic.h>
#include <stdlib.h>

#include "internal.h"

/*
 * The count the worker reads to skip an idle lane must track the list exactly,
 * and a claimed action a lane refused must go back in front of the queue and
 * not behind it -- otherwise a retry reorders what the boundaries triggered.
 */
static int inflight_count_and_return(void) {
    ShadowSpillTransferQueue queue = {0};
    ShadowSpillQueuedAction first = {0};
    ShadowSpillQueuedAction second = {0};
    if (shadowspill_transfer_queue_initialize(&queue) != 0) {
        return -1;
    }
    if (atomic_load(&queue.inflight_count) != 0U) {
        return -1;
    }
    shadowspill_transfer_queue_enqueue(&queue, &first);
    shadowspill_transfer_queue_enqueue(&queue, &second);

    /* Claimed then handed back: still the head, still pending, still nothing
       in flight. */
    if (!shadowspill_transfer_queue_claim(&queue, &first)) {
        return -1;
    }
    shadowspill_transfer_queue_return(&queue, &first);
    if (queue.pending_head != &first || queue.pending_tail != &second ||
        first.queue_state != SHADOWSPILL_QUEUE_PENDING ||
        atomic_load(&queue.inflight_count) != 0U) {
        return -1;
    }
    /* The order survived the retry: `second` still cannot be claimed first. */
    if (shadowspill_transfer_queue_claim(&queue, &second)) {
        return -1;
    }

    if (!shadowspill_transfer_queue_claim(&queue, &first)) {
        return -1;
    }
    shadowspill_transfer_queue_publish_inflight(&queue, &first);
    if (atomic_load(&queue.inflight_count) != 1U) {
        return -1;
    }
    if (shadowspill_transfer_queue_complete(&queue, &first) != 0 ||
        atomic_load(&queue.inflight_count) != 0U) {
        return -1;
    }
    shadowspill_transfer_queue_destroy(&queue);
    return 0;
}

int main(void) {
    if (inflight_count_and_return() != 0) {
        return EXIT_FAILURE;
    }
    ShadowSpillTransferQueue queue = {0};
    ShadowSpillQueuedAction first = {0};
    ShadowSpillQueuedAction second = {0};

    if (shadowspill_transfer_queue_initialize(&queue) != 0) {
        return EXIT_FAILURE;
    }
    shadowspill_transfer_queue_enqueue(&queue, &first);
    shadowspill_transfer_queue_enqueue(&queue, &second);
    if (!shadowspill_transfer_queue_claim(&queue, &first)) {
        return EXIT_FAILURE;
    }
    shadowspill_transfer_queue_publish_inflight(&queue, &first);
    if (!shadowspill_transfer_queue_claim(&queue, &second)) {
        return EXIT_FAILURE;
    }
    shadowspill_transfer_queue_publish_inflight(&queue, &second);

    /* A ready successor cannot commit while its FIFO predecessor is pending. */
    if (!shadowspill_transfer_queue_is_inflight_head(&queue, &first) ||
        shadowspill_transfer_queue_is_inflight_head(&queue, &second) ||
        shadowspill_transfer_queue_complete(&queue, &first) != 0 ||
        !shadowspill_transfer_queue_is_inflight_head(&queue, &second) ||
        shadowspill_transfer_queue_complete(&queue, &second) != 0) {
        return EXIT_FAILURE;
    }

    shadowspill_transfer_queue_destroy(&queue);

    /* Pending transfers are claimed in the order their boundaries queued
       them: one behind the head waits for it. */
    ShadowSpillTransferQueue ordered = {0};
    ShadowSpillQueuedAction earlier = {0};
    ShadowSpillQueuedAction later = {0};
    if (shadowspill_transfer_queue_initialize(&ordered) != 0) {
        return EXIT_FAILURE;
    }
    shadowspill_transfer_queue_enqueue(&ordered, &earlier);
    shadowspill_transfer_queue_enqueue(&ordered, &later);
    if (shadowspill_transfer_queue_claim(&ordered, &later) ||
        !shadowspill_transfer_queue_claim(&ordered, &earlier) ||
        !shadowspill_transfer_queue_claim(&ordered, &later)) {
        return EXIT_FAILURE;
    }
    shadowspill_transfer_queue_destroy(&ordered);
    return EXIT_SUCCESS;
}
