#ifndef SHADOWSPILL_PLANNER_RESIDENCY_INTERNAL_H
#define SHADOWSPILL_PLANNER_RESIDENCY_INTERNAL_H

#include <shadowspill/planner.h>

typedef struct ShadowSpillResidencyWorkspace ShadowSpillResidencyWorkspace;

int shadowspill_residency_workspace_create(
    const ShadowSpillResidencyProblem *problem,
    ShadowSpillResidencyWorkspace **workspace
);

void shadowspill_residency_workspace_destroy(
    ShadowSpillResidencyWorkspace *workspace
);

int shadowspill_residency_pressure_at(
    const ShadowSpillResidencyProblem *problem,
    const ShadowSpillResidencyOptions *options,
    const uint8_t *resident,
    const uint8_t *breaks,
    uint32_t device,
    uint32_t boundary,
    ShadowSpillResidencyWorkspace *workspace,
    uint64_t *pressure_bytes
);

/*
 * Remove one alias from a boundary using the same legal-cut rules as the
 * residency reducer. Returns 1 when changed, 2 when already absent, 0 when
 * the boundary is semantically required, and -1 for invalid input.
 */
int shadowspill_residency_force_absent(
    const ShadowSpillResidencyProblem *problem,
    uint8_t *resident,
    uint8_t *breaks,
    uint32_t alias,
    uint32_t boundary,
    ShadowSpillResidencyWorkspace *workspace
);

int shadowspill_residency_mark_removable(
    const ShadowSpillResidencyProblem *problem,
    const uint8_t *resident,
    const uint8_t *breaks,
    uint32_t boundary,
    ShadowSpillResidencyWorkspace *workspace,
    uint8_t *removable,
    uint32_t removable_capacity
);

ShadowSpillStatus shadowspill_reduce_residency_reusing(
    const ShadowSpillResidencyProblem *problem,
    const ShadowSpillResidencyOptions *options,
    ShadowSpillResidencyResult *result,
    ShadowSpillResidencyWorkspace *workspace
);

#endif
