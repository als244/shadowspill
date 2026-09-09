
#include <stdlib.h>

#include "internal.h"

int main(void) {
    ShadowSpillTransferLane lane = {0};
    ShadowSpillQueuedAction first = {0};
    ShadowSpillQueuedAction second = {0};

    if (shadowspill_transfer_lane_initialize(&lane) != 0) {
        return EXIT_FAILURE;
    }
    shadowspill_transfer_lane_enqueue(&lane, &first);
    shadowspill_transfer_lane_enqueue(&lane, &second);
    if (!shadowspill_transfer_lane_claim(&lane, &first)) {
        return EXIT_FAILURE;
    }
    shadowspill_transfer_lane_publish_inflight(&lane, &first);
    if (!shadowspill_transfer_lane_claim(&lane, &second)) {
        return EXIT_FAILURE;
    }
    shadowspill_transfer_lane_publish_inflight(&lane, &second);

    /* A ready successor cannot commit while its FIFO predecessor is pending. */
    if (!shadowspill_transfer_lane_is_inflight_head(&lane, &first) ||
        shadowspill_transfer_lane_is_inflight_head(&lane, &second) ||
        shadowspill_transfer_lane_complete(&lane, &first) != 0 ||
        !shadowspill_transfer_lane_is_inflight_head(&lane, &second) ||
        shadowspill_transfer_lane_complete(&lane, &second) != 0) {
        return EXIT_FAILURE;
    }

    shadowspill_transfer_lane_destroy(&lane);

    /* Background transfers fill the lane only up to the window, and a plan
       transfer queued behind them is claimable at once. */
    ShadowSpillTransferLane bounded = {0};
    ShadowSpillObject big = {.size_bytes = 48U};
    ShadowSpillObject small = {.size_bytes = 8U};
    ShadowSpillQueuedAction restore_a = {.object = &big, .background = 1U};
    ShadowSpillQueuedAction restore_b = {.object = &big, .background = 1U};
    ShadowSpillQueuedAction restore_c = {.object = &small, .background = 1U};
    ShadowSpillQueuedAction planned = {.object = &small};
    if (shadowspill_transfer_lane_initialize(&bounded) != 0) {
        return EXIT_FAILURE;
    }
    bounded.background_window_bytes = 64U;
    shadowspill_transfer_lane_enqueue(&bounded, &restore_a);
    shadowspill_transfer_lane_enqueue(&bounded, &restore_b);
    shadowspill_transfer_lane_enqueue(&bounded, &restore_c);
    shadowspill_transfer_lane_enqueue(&bounded, &planned);
    /* The first background copy runs; the second would exceed the window. */
    if (!shadowspill_transfer_lane_claim(&bounded, &restore_a)) {
        return EXIT_FAILURE;
    }
    shadowspill_transfer_lane_publish_inflight(&bounded, &restore_a);
    if (shadowspill_transfer_lane_claim(&bounded, &restore_b)) {
        return EXIT_FAILURE;
    }
    /* The plan's transfer does not wait behind the background queue. */
    if (!shadowspill_transfer_lane_claim(&bounded, &planned)) {
        return EXIT_FAILURE;
    }
    shadowspill_transfer_lane_publish_inflight(&bounded, &planned);
    /* Completing the background copy reopens the window in FIFO order. */
    if (shadowspill_transfer_lane_complete(&bounded, &restore_a) != 0 ||
        bounded.background_inflight_bytes != 0U ||
        !shadowspill_transfer_lane_claim(&bounded, &restore_b)) {
        return EXIT_FAILURE;
    }
    shadowspill_transfer_lane_publish_inflight(&bounded, &restore_b);
    /* A small background copy still fits beside a large one within the window. */
    if (!shadowspill_transfer_lane_claim(&bounded, &restore_c)) {
        return EXIT_FAILURE;
    }
    shadowspill_transfer_lane_publish_inflight(&bounded, &restore_c);
    if (bounded.background_inflight_bytes != 56U ||
        shadowspill_transfer_lane_complete(&bounded, &planned) != 0 ||
        shadowspill_transfer_lane_complete(&bounded, &restore_b) != 0 ||
        shadowspill_transfer_lane_complete(&bounded, &restore_c) != 0 ||
        bounded.background_inflight_bytes != 0U) {
        return EXIT_FAILURE;
    }
    shadowspill_transfer_lane_destroy(&bounded);
    return EXIT_SUCCESS;
}
