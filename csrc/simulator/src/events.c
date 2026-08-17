#include <stdint.h>

#include "internal.h"

uint64_t shadowspill_next_event_time(
    const ShadowSpillSimulationProgram *program,
    const ShadowSpillSimulationWork *work
) {
    uint64_t next = UINT64_MAX;
    for (uint32_t word = 0U; word < work->task_word_count; ++word) {
        uint64_t active = work->active_tasks[word];
        while (active != 0U) {
            uint32_t bit = (uint32_t)__builtin_ctzll(active);
            uint32_t task = word * 64U + bit;
            active &= active - 1U;
            if (work->tasks[task].end_ns < next) {
                next = work->tasks[task].end_ns;
            }
        }
    }
    for (uint32_t device = 0; device < program->device_count; ++device) {
        if (work->active_fetch[device] >= 0) {
            uint32_t index = (uint32_t)work->active_fetch[device];
            if (work->transfers[index].end_ns < next) {
                next = work->transfers[index].end_ns;
            }
        }
        if (work->active_evict[device] >= 0) {
            uint32_t index = (uint32_t)work->active_evict[device];
            if (work->transfers[index].end_ns < next) {
                next = work->transfers[index].end_ns;
            }
        }
    }
    return next;
}

int shadowspill_complete_events(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work,
    ShadowSpillSimulationResult *result
) {
    for (uint32_t device = 0; device < program->device_count; ++device) {
        if (work->active_fetch[device] >= 0) {
            uint32_t index = (uint32_t)work->active_fetch[device];
            if (work->transfers[index].end_ns == work->now_ns &&
                !shadowspill_complete_transfer(
                    program,
                    work,
                    result,
                    device,
                    SHADOWSPILL_TRANSFER_FETCH
                )) {
                return 0;
            }
        }
    }
    for (uint32_t device = 0; device < program->device_count; ++device) {
        if (work->active_evict[device] >= 0) {
            uint32_t index = (uint32_t)work->active_evict[device];
            if (work->transfers[index].end_ns == work->now_ns &&
                !shadowspill_complete_transfer(
                    program,
                    work,
                    result,
                    device,
                    SHADOWSPILL_TRANSFER_EVICT
                )) {
                return 0;
            }
        }
    }
    for (uint32_t word = 0U; word < work->task_word_count; ++word) {
        uint64_t active = work->active_tasks[word];
        while (active != 0U) {
            uint32_t bit = (uint32_t)__builtin_ctzll(active);
            uint32_t task = word * 64U + bit;
            active &= active - 1U;
            if (work->tasks[task].end_ns == work->now_ns &&
                !shadowspill_complete_task(program, work, result, task)) {
                return 0;
            }
        }
    }
    return 1;
}

int shadowspill_has_pending_work(
    const ShadowSpillSimulationProgram *program,
    const ShadowSpillSimulationWork *work
) {
    if (work->completed_tasks < program->task_count ||
        work->submitted_actions < program->action_count ||
        work->pending_transfers != 0U) {
        return 1;
    }
    return 0;
}
