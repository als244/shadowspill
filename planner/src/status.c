#include "internal.h"

#include <string.h>

uint32_t shadowspill_planner_abi_version(void) {
    return SHADOWSPILL_PLANNER_ABI_VERSION;
}

const char *shadowspill_planner_status_string(ShadowSpillPlannerStatus status) {
    switch (status) {
        case SHADOWSPILL_PLANNER_OK:
            return "ok";
        case SHADOWSPILL_PLANNER_INVALID_ARGUMENT:
            return "invalid planner argument";
        case SHADOWSPILL_PLANNER_ALLOCATION_FAILURE:
            return "planner allocation failure";
        case SHADOWSPILL_PLANNER_NO_FEASIBLE_CANDIDATE:
            return "no simulator-valid candidate";
        case SHADOWSPILL_PLANNER_INTERNAL_ERROR:
            return "planner internal error";
        case SHADOWSPILL_PLANNER_ANALYTIC_INFEASIBLE:
            return "analytic residency is infeasible";
    }
    return "unknown planner status";
}

void shadowspill_planner_reset_result(ShadowSpillPlanSelectionResult *result) {
    ShadowSpillCandidateResult *candidate_results = result->candidate_results;
    uint32_t candidate_result_capacity = result->candidate_result_capacity;
    memset(result, 0, sizeof(*result));
    result->status = SHADOWSPILL_PLANNER_INVALID_ARGUMENT;
    result->selected_index = SHADOWSPILL_PLANNER_NO_INDEX;
    result->selected_candidate_id = SHADOWSPILL_PLANNER_NO_INDEX;
    result->selected_selection_id = SHADOWSPILL_PLANNER_NO_INDEX;
    result->first_failure_index = SHADOWSPILL_PLANNER_NO_INDEX;
    result->first_failure_status = SHADOWSPILL_SIMULATION_OK;
    result->candidate_results = candidate_results;
    result->candidate_result_capacity = candidate_result_capacity;
}
