#include <stdint.h>

#include "internal.h"

#define NS_PER_SECOND 1000000000U

static uint64_t saturated_add(uint64_t left, uint64_t right) {
    return left > UINT64_MAX - right ? UINT64_MAX : left + right;
}

/* Divide multiplicand * 1e9 without overflowing uint64_t. The multiplicand
 * is smaller than divisor, so the quotient fits in 1e9. Most copies take
 * the direct path; the bounded division covers the full public ABI range. */
static uint64_t scaled_divide(uint64_t multiplicand, uint64_t divisor,
                              uint64_t *remainder) {
    if (multiplicand <= UINT64_MAX / NS_PER_SECOND) {
        uint64_t product = multiplicand * NS_PER_SECOND;
        *remainder = product % divisor;
        return product / divisor;
    }
    uint64_t quotient = 0U;
    *remainder = 0U;
    for (uint32_t mask = 1U << 29U; mask != 0U; mask >>= 1U) {
        quotient *= 2U;
        if (*remainder >= divisor - *remainder) {
            *remainder -= divisor - *remainder;
            quotient += 1U;
        } else {
            *remainder *= 2U;
        }
        if ((NS_PER_SECOND & mask) != 0U) {
            if (*remainder >= divisor - multiplicand) {
                *remainder -= divisor - multiplicand;
                quotient += 1U;
            } else {
                *remainder += multiplicand;
            }
        }
    }
    return quotient;
}

static uint64_t remaining_time_ns(const ShadowSpillTransferState *transfer) {
    uint64_t rate = transfer->rate_bytes_per_second;
    uint64_t seconds = transfer->remaining_bytes / rate;
    if (seconds > UINT64_MAX / NS_PER_SECOND) {
        return UINT64_MAX;
    }
    uint64_t remainder = 0U;
    uint64_t partial = scaled_divide(
        transfer->remaining_bytes % rate, rate, &remainder
    );
    /* Fractional bytes are in billionths. Keeping them across rate changes
     * avoids rounding the transferred bytes at every overlap boundary. */
    partial += transfer->remaining_fraction / rate;
    uint64_t fraction = transfer->remaining_fraction % rate;
    if (remainder >= rate - fraction) {
        remainder -= rate - fraction;
        partial += 1U;
    } else {
        remainder += fraction;
    }
    partial += remainder != 0U ? 1U : 0U;
    return saturated_add(seconds * NS_PER_SECOND, partial);
}

static void subtract_product(ShadowSpillTransferState *transfer,
                             uint64_t left, uint64_t right) {
    if (left != 0U && right > transfer->remaining_bytes / left) {
        transfer->remaining_bytes = 0U;
        transfer->remaining_fraction = 0U;
    } else {
        transfer->remaining_bytes -= left * right;
    }
}

static void advance_progress(ShadowSpillTransferState *transfer, uint64_t now) {
    uint64_t elapsed = now - transfer->progress_ns;
    uint64_t rate = transfer->rate_bytes_per_second;
    uint64_t rate_fraction = rate % NS_PER_SECOND;
    uint64_t product = (elapsed % NS_PER_SECOND) * rate_fraction;
    subtract_product(transfer, elapsed, rate / NS_PER_SECOND);
    subtract_product(transfer, elapsed / NS_PER_SECOND, rate_fraction);
    subtract_product(transfer, product / NS_PER_SECOND, 1U);
    uint64_t fraction = product % NS_PER_SECOND;
    if (fraction <= transfer->remaining_fraction) {
        transfer->remaining_fraction -= fraction;
    } else if (transfer->remaining_bytes != 0U) {
        transfer->remaining_bytes -= 1U;
        transfer->remaining_fraction += NS_PER_SECOND - fraction;
    } else {
        transfer->remaining_fraction = 0U;
    }
}

void shadowspill_start_transfer_timing(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillTransferState *transfer,
    uint64_t now
) {
    const ShadowSpillSimulationDevice *device = &program->devices[transfer->device];
    uint64_t latency = transfer->direction == SHADOWSPILL_TRANSFER_FETCH
        ? device->fetch_latency_ns : device->evict_latency_ns;
    transfer->start_ns = now;
    transfer->payload_start_ns = saturated_add(now, latency);
    transfer->remaining_bytes = program->alias_size_bytes[transfer->alias];
    transfer->remaining_fraction = 0U;
    transfer->progress_ns = now;
    transfer->rate_bytes_per_second = 0U;
    transfer->end_ns = transfer->remaining_bytes == 0U
        ? transfer->payload_start_ns : UINT64_MAX;
}

static int payload_active(const ShadowSpillTransferState *transfer, uint64_t now) {
    return transfer != NULL && transfer->payload_start_ns <= now &&
        (transfer->remaining_bytes != 0U || transfer->remaining_fraction != 0U);
}

static void set_rate(ShadowSpillTransferState *transfer, uint64_t rate, uint64_t now) {
    if (transfer == NULL || rate == transfer->rate_bytes_per_second) {
        return;
    }
    if (transfer->rate_bytes_per_second != 0U) {
        advance_progress(transfer, now);
    }
    transfer->progress_ns = now;
    transfer->rate_bytes_per_second = rate;
    if (rate != 0U) {
        transfer->end_ns = saturated_add(now, remaining_time_ns(transfer));
    }
}

void shadowspill_refresh_transfer_rates(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work
) {
    for (uint32_t device = 0U; device < program->device_count; ++device) {
        ShadowSpillTransferState *fetch = work->active_fetch[device] < 0 ? NULL
            : &work->transfers[work->active_fetch[device]];
        ShadowSpillTransferState *evict = work->active_evict[device] < 0 ? NULL
            : &work->transfers[work->active_evict[device]];
        int fetching = payload_active(fetch, work->now_ns);
        int evicting = payload_active(evict, work->now_ns);
        const ShadowSpillSimulationDevice *config = &program->devices[device];
        set_rate(fetch, !fetching ? 0U : evicting
            ? config->fetch_concurrent_bandwidth_bytes_per_second
            : config->fetch_solo_bandwidth_bytes_per_second, work->now_ns);
        set_rate(evict, !evicting ? 0U : fetching
            ? config->evict_concurrent_bandwidth_bytes_per_second
            : config->evict_solo_bandwidth_bytes_per_second, work->now_ns);
    }
}
