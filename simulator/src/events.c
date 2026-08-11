#include <stdint.h>

#include "internal.h"

uint64_t shadowspill_next_event_time(
    const ShadowSpillSimulationProgram *program,
    const ShadowSpillSimulationWork *work
) {
    uint64_t next = UINT64_MAX;
    for (uint32_t task = 0; task < program->task_count; ++task) {
        if (work->tasks[task].state == SHADOWSPILL_TASK_ACTIVE &&
            work->tasks[task].end_ns < next) {
            next = work->tasks[task].end_ns;
        }
    }
    for (uint32_t device = 0; device < program->device_count; ++device) {
        if (work->active_h2d[device] >= 0) {
            uint32_t index = (uint32_t)work->active_h2d[device];
            if (work->transfers[index].end_ns < next) {
                next = work->transfers[index].end_ns;
            }
        }
        if (work->active_d2h[device] >= 0) {
            uint32_t index = (uint32_t)work->active_d2h[device];
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
        if (work->active_h2d[device] >= 0) {
            uint32_t index = (uint32_t)work->active_h2d[device];
            if (work->transfers[index].end_ns == work->now_ns &&
                !shadowspill_complete_transfer(
                    program,
                    work,
                    result,
                    device,
                    SHADOWSPILL_TRANSFER_HOST_TO_DEVICE
                )) {
                return 0;
            }
        }
    }
    for (uint32_t device = 0; device < program->device_count; ++device) {
        if (work->active_d2h[device] >= 0) {
            uint32_t index = (uint32_t)work->active_d2h[device];
            if (work->transfers[index].end_ns == work->now_ns &&
                !shadowspill_complete_transfer(
                    program,
                    work,
                    result,
                    device,
                    SHADOWSPILL_TRANSFER_DEVICE_TO_HOST
                )) {
                return 0;
            }
        }
    }
    for (uint32_t task = 0; task < program->task_count; ++task) {
        if (work->tasks[task].state == SHADOWSPILL_TASK_ACTIVE &&
            work->tasks[task].end_ns == work->now_ns &&
            !shadowspill_complete_task(program, work, result, task)) {
            return 0;
        }
    }
    return 1;
}

int shadowspill_has_pending_work(
    const ShadowSpillSimulationProgram *program,
    const ShadowSpillSimulationWork *work
) {
    if (work->completed_tasks < program->task_count ||
        work->submitted_actions < program->action_count) {
        return 1;
    }
    for (uint32_t index = 0; index < program->action_count; ++index) {
        if (work->transfers[index].state == SHADOWSPILL_TRANSFER_QUEUED ||
            work->transfers[index].state == SHADOWSPILL_TRANSFER_ACTIVE) {
            return 1;
        }
    }
    return 0;
}
