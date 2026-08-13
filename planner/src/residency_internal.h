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

ShadowSpillPlannerStatus shadowspill_reduce_residency_reusing(
    const ShadowSpillResidencyProblem *problem,
    const ShadowSpillResidencyOptions *options,
    ShadowSpillResidencyResult *result,
    ShadowSpillResidencyWorkspace *workspace
);

#endif
