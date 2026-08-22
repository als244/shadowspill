#include <limits.h>
#include <stdint.h>

#include "internal.h"

uint64_t shadowspill_device_used_bytes(
    const ShadowSpillSimulationProgram *program,
    const ShadowSpillSimulationWork *work,
    uint32_t device
) {
    if (program->use_admission_accounting != 0U) {
        return work->device_physical_bytes[device];
    }
    return work->device_object_bytes[device] +
        work->device_workspace_bytes[device];
}

int shadowspill_resolve_physical_delta(
    const ShadowSpillSimulationProgram *program,
    const int64_t *deltas,
    uint32_t index,
    int64_t default_delta,
    int64_t *result
) {
    if (result == NULL) {
        return 0;
    }
    if (program->use_admission_accounting != 0U &&
        deltas[index] != INT64_MIN) {
        *result = deltas[index];
    } else {
        *result = default_delta;
    }
    return 1;
}

int shadowspill_physical_delta_fits(
    const ShadowSpillSimulationProgram *program,
    const ShadowSpillSimulationWork *work,
    uint32_t device,
    int64_t delta
) {
    if (delta <= 0) {
        return 1;
    }
    uint64_t used = shadowspill_device_used_bytes(program, work, device);
    uint64_t requested = (uint64_t)delta;
    return requested <= program->devices[device].capacity_bytes &&
        used <= program->devices[device].capacity_bytes - requested;
}

int shadowspill_apply_physical_delta(
    const ShadowSpillSimulationProgram *program,
    ShadowSpillSimulationWork *work,
    uint32_t device,
    int64_t delta
) {
    uint64_t current = work->device_physical_bytes[device];
    if (delta >= 0) {
        uint64_t increase = (uint64_t)delta;
        if (increase > UINT64_MAX - current) {
            return 0;
        }
        work->device_physical_bytes[device] = current + increase;
        return 1;
    }
    uint64_t decrease = (uint64_t)(-(delta + 1)) + 1U;
    if (decrease > current) {
        return 0;
    }
    work->device_physical_bytes[device] = current - decrease;
    (void)program;
    return 1;
}

static int dependencies_complete(
    const ShadowSpillSimulationProgram *program,
    const ShadowSpillSimulationWork *work,
    uint32_t successor,
    int successor_is_task
) {
    for (uint32_t index = 0; index < program->reuse_dependency_count; ++index) {
        uint32_t candidate = successor_is_task != 0
            ? program->reuse_successor_tasks[index]
            : program->reuse_successor_actions[index];
        if (candidate != successor) {
            continue;
        }
        uint32_t predecessor = program->reuse_predecessor_actions[index];
        if (work->transfers[predecessor].state !=
            SHADOWSPILL_TRANSFER_COMPLETE) {
            return 0;
        }
    }
    return 1;
}

int shadowspill_task_reuse_dependencies_complete(
    const ShadowSpillSimulationProgram *program,
    const ShadowSpillSimulationWork *work,
    uint32_t task
) {
    return dependencies_complete(program, work, task, 1);
}

int shadowspill_action_reuse_dependencies_complete(
    const ShadowSpillSimulationProgram *program,
    const ShadowSpillSimulationWork *work,
    uint32_t action
) {
    return dependencies_complete(program, work, action, 0);
}
