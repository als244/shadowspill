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

/* Return ceil(numerator * 1e9 / denominator) without overflowing uint64_t. */
static int copy_duration_ns(
    uint64_t numerator,
    uint64_t denominator,
    uint64_t *result
) {
    const uint64_t scale = UINT64_C(1000000000);
    if (denominator == 0U) {
        return -1;
    }
    uint64_t whole = numerator / denominator;
    uint64_t remainder = numerator % denominator;
    if (whole > UINT64_MAX / scale) {
        return -1;
    }
    uint64_t quotient = 0U;
    uint64_t fractional_remainder = 0U;
    uint64_t highest_bit = UINT64_C(1) << 29U;
    for (uint64_t bit = highest_bit; bit != 0U; bit >>= 1U) {
        if (quotient > UINT64_MAX / 2U) {
            return -1;
        }
        quotient *= 2U;
        if (fractional_remainder >= denominator - fractional_remainder) {
            fractional_remainder -= denominator - fractional_remainder;
            ++quotient;
        } else {
            fractional_remainder *= 2U;
        }
        if ((scale & bit) == 0U) {
            continue;
        }
        if (fractional_remainder >= denominator - remainder) {
            fractional_remainder -= denominator - remainder;
            ++quotient;
        } else {
            fractional_remainder += remainder;
        }
    }
    if (fractional_remainder != 0U) {
        ++quotient;
    }
    uint64_t duration = whole * scale;
    /* Keep heuristic transfer costs consistent with bandwidth-only simulation. */
    return shadowspill_add_overflow_u64(duration, quotient, result) ? -1 : 0;
}

int shadowspill_transfer_duration_ns(
    const ShadowSpillSimulationProgram *program, uint32_t alias, uint8_t direction,
    uint64_t copy_rate_bytes_per_second, uint64_t *duration_ns
) {
    if (program == NULL || duration_ns == NULL || alias >= program->alias_count ||
        direction > SHADOWSPILL_TRANSFER_EVICT || copy_rate_bytes_per_second == 0U) {
        return -1;
    }
    uint32_t begin = program->alias_transfer_offsets == NULL ? 0U
        : program->alias_transfer_offsets[alias * 2U + direction];
    uint32_t end = program->alias_transfer_offsets == NULL ? 0U
        : program->alias_transfer_offsets[alias * 2U + direction + 1U];
    if (begin > end || end > program->transfer_phase_count ||
        (end > begin && program->transfer_phases == NULL)) {
        return -1;
    }
    if (begin == end) {
        return copy_duration_ns(
            program->alias_size_bytes[alias], copy_rate_bytes_per_second, duration_ns
        );
    }
    *duration_ns = 0U;
    for (uint32_t index = begin; index < end; ++index) {
        const ShadowSpillTransferPhase *phase = &program->transfer_phases[index];
        uint64_t duration = phase->duration_ns;
        if (phase->kind > SHADOWSPILL_TRANSFER_PHASE_TIMED ||
            (phase->kind == SHADOWSPILL_TRANSFER_PHASE_COPY && copy_duration_ns(
                phase->bytes, copy_rate_bytes_per_second, &duration
            ) != 0) || shadowspill_add_overflow_u64(*duration_ns, duration, duration_ns)) {
            return -1;
        }
    }
    return 0;
}

static void start_phase(
    const ShadowSpillSimulationProgram *program, ShadowSpillSimulationWork *work,
    ShadowSpillTransferState *transfer, uint64_t now
) {
    const ShadowSpillTransferPhase *phase = transfer->phase_index < transfer->phase_end
        ? &program->transfer_phases[transfer->phase_index] : NULL;
    transfer->timed_phase = phase != NULL && phase->kind == SHADOWSPILL_TRANSFER_PHASE_TIMED;
    transfer->awaiting_resource_order = 0U;
    transfer->payload_start_ns = now;
    transfer->remaining_bytes = phase == NULL ? program->alias_size_bytes[transfer->alias]
        : transfer->timed_phase ? 0U : phase->bytes;
    transfer->remaining_fraction = 0U;
    transfer->progress_ns = now;
    transfer->rate_bytes_per_second = 0U;
    transfer->end_ns = transfer->remaining_bytes == 0U ? now : UINT64_MAX;
    if (transfer->timed_phase) {
        transfer->remaining_work_ns = phase->duration_ns;
        transfer->work_progress_ns = now;
        transfer->work_rate = 1.0;
        if (phase->submission_ordered && transfer->resource_order !=
            work->transfer_resource_completed[phase->resource] + 1U) {
            transfer->awaiting_resource_order = 1U;
            transfer->payload_start_ns = UINT64_MAX;
            transfer->end_ns = UINT64_MAX;
            transfer->stall_mask |= SHADOWSPILL_STALL_TRANSFER_RESOURCE;
            return;
        }
        if (!shadowspill_phase_task_free(program, work, phase->resource)) {
            transfer->awaiting_resource_order = 1;
            transfer->payload_start_ns = transfer->end_ns = UINT64_MAX;
            transfer->stall_mask |= SHADOWSPILL_STALL_TRANSFER_RESOURCE;
            return;
        }
        if (program->overlap_category_count != 0U || program->task_phase_count != 0U) {
            /* A measured overlap rate can change the owner's finish time.
               Retry on actual completion, never reserve a future interval. */
            for (uint32_t device = 0; device < program->device_count; ++device) {
                int32_t peers[2] = {work->active_fetch[device], work->active_evict[device]};
                for (uint32_t lane = 0; lane < 2; ++lane) {
                    if (peers[lane] < 0) continue;
                    ShadowSpillTransferState *other = &work->transfers[peers[lane]];
                    if (other != transfer && other->timed_phase &&
                        !other->awaiting_resource_order && other->end_ns > now &&
                        program->transfer_phases[other->phase_index].resource == phase->resource) {
                        transfer->awaiting_resource_order = 1;
                        transfer->payload_start_ns = transfer->end_ns = UINT64_MAX;
                        transfer->stall_mask |= SHADOWSPILL_STALL_TRANSFER_RESOURCE;
                        return;
                    }
                }
            }
            transfer->end_ns = saturated_add(now, phase->duration_ns);
            return;
        }
        uint64_t *available = &work->transfer_resource_free_ns[phase->resource];
        if (*available > now) {
            transfer->payload_start_ns = *available;
            transfer->stall_mask |= SHADOWSPILL_STALL_TRANSFER_RESOURCE;
        }
        transfer->end_ns = saturated_add(transfer->payload_start_ns, phase->duration_ns);
        *available = transfer->end_ns;
    }
}

void shadowspill_start_transfer_timing(
    const ShadowSpillSimulationProgram *program, ShadowSpillSimulationWork *work,
    ShadowSpillTransferState *transfer, uint64_t now
) {
    transfer->start_ns = now;
    transfer->phase_index = program->alias_transfer_offsets == NULL ? 0U
        : program->alias_transfer_offsets[transfer->alias * 2U + transfer->direction];
    transfer->phase_end = program->alias_transfer_offsets == NULL ? 0U
        : program->alias_transfer_offsets[transfer->alias * 2U + transfer->direction + 1U];
    start_phase(program, work, transfer, now);
}

int shadowspill_advance_transfer_phase(
    const ShadowSpillSimulationProgram *program, ShadowSpillSimulationWork *work,
    ShadowSpillTransferState *transfer
) {
    if (transfer->phase_index < transfer->phase_end) {
        const ShadowSpillTransferPhase *phase = &program->transfer_phases[transfer->phase_index];
        if (phase->submission_ordered) {
            work->transfer_resource_completed[phase->resource] = transfer->resource_order;
        }
    }
    if (transfer->phase_index + 1U >= transfer->phase_end) {
        return 0;
    }
    ++transfer->phase_index;
    start_phase(program, work, transfer, work->now_ns);
    return 1;
}

static int payload_active(const ShadowSpillTransferState *transfer, uint64_t now) {
    return transfer != NULL && !transfer->timed_phase && transfer->payload_start_ns <= now &&
        (transfer->remaining_bytes != 0U || transfer->remaining_fraction != 0U);
}

static void set_rate(ShadowSpillTransferState *transfer, uint64_t rate, uint64_t now) {
    if (transfer == NULL || transfer->timed_phase || rate == transfer->rate_bytes_per_second) {
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
        if (fetch != NULL && fetch->awaiting_resource_order) {
            start_phase(program, work, fetch, work->now_ns);
        }
        if (evict != NULL && evict->awaiting_resource_order) {
            start_phase(program, work, evict, work->now_ns);
        }
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
