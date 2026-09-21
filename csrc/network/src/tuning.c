/* Every knob that changes how bytes move, read once from the environment. */
#include "../internal.h"

#include <stdio.h>
#include <stdlib.h>

/*
 * The defaults are what this hardware wanted, and each is a number somebody
 * may need to change while looking at a measurement. They are read from the
 * environment rather than taken as arguments because an investigation's shape
 * is "the same run with one thing different", which should not mean editing a
 * configuration and rebuilding.
 */

static uint64_t number(const char *name, uint64_t fallback) {
    const char *text = getenv(name);
    if (text == NULL || text[0] == '\0') {
        return fallback;
    }
    char *end = NULL;
    const unsigned long long value = strtoull(text, &end, 0);
    /* A value that is not a number at all is a typo, and silently taking the
       default would hide it during exactly the investigation it was set for. */
    if (end == text || *end != '\0') {
        fprintf(
            stderr,
            "shadowspill network: %s=\"%s\" is not a number; using %llu\n",
            name, text, (unsigned long long)fallback
        );
        return fallback;
    }
    return (uint64_t)value;
}

void shadowspill_network_tuning_read(ShadowSpillNetworkTuning *tuning) {
    if (tuning == NULL) {
        return;
    }
    *tuning = (ShadowSpillNetworkTuning){
        /*
         * Two, not one, and not for bandwidth: a remote pool is served by two
         * lanes, one per direction, and they run at the same time. Each takes
         * a queue pair and its completion queue for its own use, because
         * sharing one lets them consume each other's completions.
         *
         * One still saturates 25 Gb/s. More than two is the bandwidth knob.
         */
        .queue_pairs = (uint32_t)number("SHADOWSPILL_NETWORK_QUEUE_PAIRS", 2U),
        .send_depth = (uint32_t)number("SHADOWSPILL_NETWORK_SEND_DEPTH", 128U),
        .receive_depth =
            (uint32_t)number("SHADOWSPILL_NETWORK_RECEIVE_DEPTH", 16U),
        /* 16 is what perftest asks for by default and what this HCA granted.
           It bounds reads in flight, so it bounds the fetch direction. */
        .outstanding_reads =
            (uint32_t)number("SHADOWSPILL_NETWORK_OUTSTANDING_READS", 16U),
        /*
         * Measured, not guessed. Each chunk is posted and waited for before
         * the next, so the chunk is also the pipeline's whole depth, and a
         * small one pays a round trip per chunk:
         *
         *   256 KiB -> 1866 MiB/s     1 MiB -> 2104 MiB/s
         *     4 MiB -> 2351 MiB/s     8 MiB -> 2412 MiB/s
         *
         * against `ib_read_bw`'s 2921 MiB/s. 4 MiB is 80 % of line rate for
         * 4 MiB of ring per lane, which is nothing beside a spill pool; the
         * remaining 20 % is the host copy, which does not overlap.
         */
        .chunk_bytes = number("SHADOWSPILL_NETWORK_CHUNK_BYTES", 4U << 20U),
        /* Zero is "ask the port", which is what anyone not testing the pieces
           wants; the endpoint lowers it to the port's limit either way. */
        .message_bytes = number("SHADOWSPILL_NETWORK_MESSAGE_BYTES", 0U),
        /* Two overlaps the host copy with the transfer, which is the whole
           point; a third stage would be needed for a third slot to help. */
        .ring_slots = (uint32_t)number("SHADOWSPILL_NETWORK_RING_SLOTS", 2U),
        .signal_every =
            (uint32_t)number("SHADOWSPILL_NETWORK_SIGNAL_EVERY", 8U),
        /* Watch, do not sleep. Twenty microseconds of spinning used to be the
           default and a small transfer still paid a full wake on both
           handoffs -- measured 26.71 us to wake the lane thread and 19.86 us
           to hand back, against a NIC that answers in 2.05. Set the variable
           to bound it again, or to zero to block immediately. */
        .spin_nanoseconds = number(
            "SHADOWSPILL_NETWORK_SPIN_NANOSECONDS",
            SHADOWSPILL_NETWORK_SPIN_FOREVER
        ),
        .traffic_class =
            (uint32_t)number("SHADOWSPILL_NETWORK_TRAFFIC_CLASS", 0U),
        .service_level =
            (uint32_t)number("SHADOWSPILL_NETWORK_SERVICE_LEVEL", 0U),
        .path_mtu = (uint32_t)number("SHADOWSPILL_NETWORK_PATH_MTU", 0U),
        .timeout = (uint32_t)number("SHADOWSPILL_NETWORK_TIMEOUT", 14U),
        .retry_count = (uint32_t)number("SHADOWSPILL_NETWORK_RETRY_COUNT", 7U),
        .rnr_retry = (uint32_t)number("SHADOWSPILL_NETWORK_RNR_RETRY", 7U),
        .device = getenv("SHADOWSPILL_NETWORK_DEVICE"),
        .gid_index = (int)number("SHADOWSPILL_NETWORK_GID_INDEX", (uint64_t)(-1)),
    };
    /* A zero here would mean "no queue pairs" and "never signal", neither of
       which anyone means. Clamp rather than refuse: this is a tuning knob, and
       failing to start over a typo in one helps nobody. */
    /* One per direction is a floor, not a preference: below it a lane cannot
       claim a queue pair of its own and the two would share completions. */
    if (tuning->queue_pairs < 2U) {
        tuning->queue_pairs = 2U;
    }
    if (tuning->queue_pairs > SHADOWSPILL_NETWORK_MAX_QUEUE_PAIRS) {
        tuning->queue_pairs = SHADOWSPILL_NETWORK_MAX_QUEUE_PAIRS;
    }
    if (tuning->signal_every == 0U) {
        tuning->signal_every = 1U;
    }
    if (tuning->ring_slots == 0U) {
        tuning->ring_slots = 1U;
    }
    if (tuning->ring_slots > SHADOWSPILL_NETWORK_MAX_RING_SLOTS) {
        tuning->ring_slots = SHADOWSPILL_NETWORK_MAX_RING_SLOTS;
    }
    if (tuning->chunk_bytes == 0U) {
        tuning->chunk_bytes = 64U << 10U;
    }
    /* The environment spells "discover" as any negative number; normalise so
       nothing downstream has to know which one. */
    if (tuning->gid_index < 0) {
        tuning->gid_index = -1;
    }
}

void shadowspill_network_tuning_report(
    const ShadowSpillNetworkTuning *tuning
) {
    if (tuning == NULL) {
        return;
    }
    fprintf(
        stderr,
        "shadowspill network: queue pairs %u, send depth %u, outstanding reads "
        "%u, chunk %llu KiB x %u slots, signal every %u, traffic class %u, "
        "service level %u, message %s, mtu %s, device %s, gid %s\n",
        tuning->queue_pairs, tuning->send_depth, tuning->outstanding_reads,
        (unsigned long long)(tuning->chunk_bytes >> 10U), tuning->ring_slots,
        tuning->signal_every, tuning->traffic_class, tuning->service_level,
        tuning->message_bytes == 0U ? "from the port" : "overridden",
        tuning->path_mtu == 0U ? "from the port" : "overridden",
        tuning->device != NULL ? tuning->device : "detected",
        tuning->gid_index < 0 ? "detected" : "overridden"
    );
}
