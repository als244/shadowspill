
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

    /* Background transfers fill the queue only up to the window, and a plan
       transfer queued behind them is claimable at once. */
    ShadowSpillTransferQueue bounded = {0};
    ShadowSpillObject big = {.size_bytes = 48U};
    ShadowSpillObject small = {.size_bytes = 8U};
    ShadowSpillQueuedAction restore_a = {.object = &big, .background = 1U};
    ShadowSpillQueuedAction restore_b = {.object = &big, .background = 1U};
    ShadowSpillQueuedAction restore_c = {.object = &small, .background = 1U};
    ShadowSpillQueuedAction planned = {.object = &small};
    if (shadowspill_transfer_queue_initialize(&bounded) != 0) {
        return EXIT_FAILURE;
    }
    bounded.background_window_bytes = 64U;
    shadowspill_transfer_queue_enqueue(&bounded, &restore_a);
    shadowspill_transfer_queue_enqueue(&bounded, &restore_b);
    shadowspill_transfer_queue_enqueue(&bounded, &restore_c);
    shadowspill_transfer_queue_enqueue(&bounded, &planned);
    /* The first background copy runs; the second would exceed the window. */
    if (!shadowspill_transfer_queue_claim(&bounded, &restore_a)) {
        return EXIT_FAILURE;
    }
    shadowspill_transfer_queue_publish_inflight(&bounded, &restore_a);
    if (shadowspill_transfer_queue_claim(&bounded, &restore_b)) {
        return EXIT_FAILURE;
    }
    /* The plan's transfer does not wait behind the background queue. */
    if (!shadowspill_transfer_queue_claim(&bounded, &planned)) {
        return EXIT_FAILURE;
    }
    shadowspill_transfer_queue_publish_inflight(&bounded, &planned);
    /* Completing the background copy reopens the window in FIFO order. */
    if (shadowspill_transfer_queue_complete(&bounded, &restore_a) != 0 ||
        bounded.background_inflight_bytes != 0U ||
        !shadowspill_transfer_queue_claim(&bounded, &restore_b)) {
        return EXIT_FAILURE;
    }
    shadowspill_transfer_queue_publish_inflight(&bounded, &restore_b);
    /* A small background copy still fits beside a large one within the window. */
    if (!shadowspill_transfer_queue_claim(&bounded, &restore_c)) {
        return EXIT_FAILURE;
    }
    shadowspill_transfer_queue_publish_inflight(&bounded, &restore_c);
    if (bounded.background_inflight_bytes != 56U ||
        shadowspill_transfer_queue_complete(&bounded, &planned) != 0 ||
        shadowspill_transfer_queue_complete(&bounded, &restore_b) != 0 ||
        shadowspill_transfer_queue_complete(&bounded, &restore_c) != 0 ||
        bounded.background_inflight_bytes != 0U) {
        return EXIT_FAILURE;
    }
    shadowspill_transfer_queue_destroy(&bounded);
    return EXIT_SUCCESS;
}
