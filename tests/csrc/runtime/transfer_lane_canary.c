
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
    return EXIT_SUCCESS;
}
