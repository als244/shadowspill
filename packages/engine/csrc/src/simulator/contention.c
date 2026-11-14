/* Advance isolated work only at the measured rate of the active overlap.
 * Queue waits do not consume work; a participant's rate returns to one as
 * soon as its peer completes. Inputs are immutable and simulation-local. */
#include <stdint.h>

#include "internal.h"

static uint64_t finish(uint64_t now, long double remaining, double rate) {
    if (remaining <= 0.0L) {
        return now;
    }
    long double duration = remaining / rate;
    if (duration >= (long double)(UINT64_MAX - now)) {
        return UINT64_MAX;
    }
    uint64_t rounded = (uint64_t)duration;
    if ((long double)rounded < duration) {
        ++rounded;
    }
    return now + rounded;
}

static int active_modeled_transfer(
    const ShadowSpillSimulationProgram *program,
    const ShadowSpillTransferState *state
) {
    return state->timed_phase && !state->awaiting_resource_order &&
        program->transfer_overlap_categories[state->phase_index] !=
            SHADOWSPILL_SIMULATOR_NO_INDEX;
}

static double task_rate(
    const ShadowSpillSimulationProgram *program,
    const ShadowSpillSimulationWork *work,
    uint32_t task
) {
    uint32_t device = program->task_device[task];
    int32_t peers[2] = {
        work->active_fetch[device], work->active_evict[device]
    };
    uint64_t row = (uint64_t)program->task_overlap_profiles[task] *
        program->overlap_category_count;
    double rate = 1.0;
    for (uint32_t lane = 0; lane < 2U; ++lane) {
        if (peers[lane] < 0) {
            continue;
        }
        const ShadowSpillTransferState *peer = &work->transfers[peers[lane]];
        if (active_modeled_transfer(program, peer)) {
            uint32_t category = program->transfer_overlap_categories[
                peer->phase_index
            ];
            double candidate = program->task_overlap_rates[row + category];
            if (candidate < rate) {
                rate = candidate;
            }
        }
    }
    return rate;
}

static double transfer_rate(
    const ShadowSpillSimulationProgram *program,
    const ShadowSpillSimulationWork *work,
    uint32_t device,
    uint32_t category
) {
    double rate = 1.0;
    for (uint32_t word = 0; word < work->task_word_count; ++word) {
        uint64_t active = work->active_tasks[word];
        while (active != 0U) {
            uint32_t task = word * 64U + (uint32_t)__builtin_ctzll(active);
            active &= active - 1U;
            if (program->task_device[task] != device ||
                !shadowspill_computing_task(program, &work->tasks[task], task)) {
                continue;
            }
            uint64_t row = (uint64_t)program->task_overlap_profiles[task] *
                program->overlap_category_count;
            double candidate = program->transfer_overlap_rates[row + category];
            if (candidate < rate) {
                rate = candidate;
            }
        }
    }
    return rate;
}

void shadowspill_refresh_contention(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work
) {
    shadowspill_refresh_task_phases(program, work);
    if (program->overlap_category_count == 0U) {
        return;
    }
    for (uint32_t word = 0; word < work->task_word_count; ++word) {
        uint64_t active = work->active_tasks[word];
        while (active != 0U) {
            uint32_t task = word * 64U + (uint32_t)__builtin_ctzll(active);
            active &= active - 1U;
            ShadowSpillTaskState *state = &work->tasks[task];
            if (!shadowspill_computing_task(program, state, task)) {
                continue;
            }
            state->remaining_work_ns -= (work->now_ns - state->progress_ns) *
                (long double)state->work_rate;
            state->progress_ns = work->now_ns;
            state->work_rate = task_rate(program, work, task);
            state->end_ns = finish(
                work->now_ns, state->remaining_work_ns, state->work_rate
            );
        }
    }
    for (uint32_t device = 0; device < program->device_count; ++device) {
        int32_t peers[2] = {
            work->active_fetch[device], work->active_evict[device]
        };
        for (uint32_t lane = 0; lane < 2U; ++lane) {
            if (peers[lane] < 0) {
                continue;
            }
            ShadowSpillTransferState *state = &work->transfers[peers[lane]];
            if (!active_modeled_transfer(program, state)) {
                continue;
            }
            state->remaining_work_ns -=
                (work->now_ns - state->work_progress_ns) *
                (long double)state->work_rate;
            state->work_progress_ns = work->now_ns;
            state->work_rate = transfer_rate(
                program, work, device,
                program->transfer_overlap_categories[state->phase_index]
            );
            state->end_ns = finish(
                work->now_ns, state->remaining_work_ns, state->work_rate
            );
        }
    }
}
