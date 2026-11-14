/* The loop: cut, repressure, and stop when the budget is met. */
#include "internal.h"

#include "../trace.h"

#define APPLY_CUT apply_cut_and_repressure
#define PRESSURE_ADD shadowspill_residency_pressure_add
#include "apply.h"
#undef PRESSURE_ADD
#undef APPLY_CUT

#define APPLY_CUT apply_traced_cut
#define PRESSURE_ADD(...) shadowspill_candidate_trace_pressure_add(result->trace, __VA_ARGS__)
#include "apply.h"
#undef PRESSURE_ADD
#undef APPLY_CUT

/* No legal cut relieves this boundary, so no residency this strategy can
 * reach fits it. That is a fact about the problem rather than a failure. */
static ShadowSpillStatus report_analytic_infeasible(
    const ShadowSpillPressureFitResidencyProblem *problem,
    ShadowSpillPressureFitResidencyResult *result,
    uint32_t device,
    uint32_t boundary,
    uint64_t used_bytes
) {
    result->status = SHADOWSPILL_STATUS_ANALYTIC_INFEASIBLE;
    result->error_device = device;
    result->error_boundary = (int32_t)boundary - 1;
    result->required_bytes = used_bytes;
    result->capacity_bytes =
        shadowspill_boundary_capacity(problem, device, boundary);
    return SHADOWSPILL_STATUS_ANALYTIC_INFEASIBLE;
}

/*
 * Remove objects until every boundary fits.
 *
 * One step: take the boundary that is furthest over capacity, choose the cut
 * that relieves it best under the strategy's score, apply it, and let the
 * pressure map absorb the change. Repeat until nothing is over capacity, or
 * until a boundary has no legal cut left.
 */
ShadowSpillStatus shadowspill_pressurefit_reduce_residency_reusing(
    const ShadowSpillPressureFitResidencyProblem *problem,
    const ShadowSpillPressureFitResidencyOptions *options,
    ShadowSpillPressureFitResidencyResult *result,
    ShadowSpillPressureFitResidencyWorkspace *workspace
) {
    if (!shadowspill_residency_valid_problem(problem, options, result) || workspace == NULL ||
        workspace->alias_count != problem->alias_count ||
        workspace->boundary_count != problem->boundary_count ||
        workspace->device_count != problem->device_count) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    shadowspill_residency_reset_residency_result(result);
    if (shadowspill_residency_seed_residency(options, result) != 0) {
        return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    }
    if (shadowspill_residency_prepare_seed_geometry(problem, options, workspace) != 0 ||
        shadowspill_residency_prepare_base_pressure(problem, options, workspace) != 0) {
        return SHADOWSPILL_STATUS_PLANNER_INTERNAL_ERROR;
    }
    if (shadowspill_residency_reset_cut_candidates(workspace) != 0) {
        return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    }
    shadowspill_residency_reset_working_pressure(problem, options, workspace);
    if (shadowspill_residency_seed_excess_heap(problem, options, workspace) != 0) {
        return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    }

    while (1) {
        uint32_t device = UINT32_MAX;
        uint32_t boundary = UINT32_MAX;
        uint64_t used_bytes = 0U;
        const int over_capacity = shadowspill_residency_pop_worst_boundary(
            problem, options, workspace, &device, &boundary, &used_bytes
        );
        if (over_capacity < 0) {
            return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
        }
        if (over_capacity == 0) {
            result->status = SHADOWSPILL_STATUS_OK;
            return SHADOWSPILL_STATUS_OK;
        }
        ResidencyCut chosen;
        uint32_t chosen_id = UINT32_MAX;
        if (!shadowspill_residency_select_cut(
                problem,
                device,
                (int32_t)boundary - 1,
                options->minimize_transfer != 0U,
                &workspace->cut_index,
                workspace->cut_active,
                workspace->node_cursors,
                &chosen,
                &chosen_id
            )) {
            return report_analytic_infeasible(
                problem, result, device, boundary, used_bytes
            );
        }
        int applied;
        if (result->trace != NULL) {
            if (shadowspill_candidate_trace_cut(result->trace, problem, workspace,
                    device, boundary, used_bytes, chosen_id,
                    options->minimize_transfer != 0U) != 0) {
                return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
            }
            applied = apply_traced_cut(problem, options, result, workspace, &chosen, chosen_id);
        } else {
            applied = apply_cut_and_repressure(problem, options, result, workspace, &chosen, chosen_id);
        }
        if (applied != 0) {
            return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
        }
    }
}
