#ifndef SHADOWSPILL_PLANNER_INTERNAL_H
#define SHADOWSPILL_PLANNER_INTERNAL_H

#include <shadowspill/planner.h>

static inline uint64_t shadowspill_boundary_capacity(
    const ShadowSpillResidencyProblem *problem,
    uint32_t device,
    uint32_t boundary
) {
    return problem->boundary_capacity_bytes[
        (uint64_t)device * problem->boundary_count + boundary
    ];
}

void shadowspill_planner_reset_result(ShadowSpillPlanSelectionResult *result);

#endif
