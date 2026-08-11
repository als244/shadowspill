#ifndef SHADOWSPILL_PLANNER_H
#define SHADOWSPILL_PLANNER_H

#include <stdint.h>

#include <shadowspill/simulator.h>

#if defined(_WIN32)
#define SHADOWSPILL_PLANNER_API __declspec(dllexport)
#else
#define SHADOWSPILL_PLANNER_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SHADOWSPILL_PLANNER_ABI_VERSION 1U
#define SHADOWSPILL_PLANNER_NO_INDEX UINT32_MAX

typedef enum ShadowSpillPlannerStatus {
    SHADOWSPILL_PLANNER_OK = 0,
    SHADOWSPILL_PLANNER_INVALID_ARGUMENT = 1,
    SHADOWSPILL_PLANNER_ALLOCATION_FAILURE = 2,
    SHADOWSPILL_PLANNER_NO_FEASIBLE_CANDIDATE = 3,
    SHADOWSPILL_PLANNER_INTERNAL_ERROR = 4,
} ShadowSpillPlannerStatus;

/*
 * One fully materialized PressureFit candidate. `program` includes the
 * candidate's exact initial residency, ordered actions, and final residency.
 * Identifiers are opaque to the planner and are copied into the result.
 */
typedef struct ShadowSpillPlanCandidate {
    const ShadowSpillSimulationProgram *program;
    uint32_t candidate_id;
    uint32_t selection_id;
} ShadowSpillPlanCandidate;

typedef struct ShadowSpillCandidateResult {
    uint32_t simulation_status;
    uint8_t valid;
    uint64_t makespan_ns;
} ShadowSpillCandidateResult;

typedef struct ShadowSpillPlanSelectionResult {
    uint32_t status;
    uint32_t selected_index;
    uint32_t selected_candidate_id;
    uint32_t selected_selection_id;
    uint32_t valid_candidate_count;
    uint32_t first_failure_index;
    uint32_t first_failure_status;
    uint64_t selected_makespan_ns;

    /* Optional caller-owned buffer, one entry per input candidate. */
    ShadowSpillCandidateResult *candidate_results;
    uint32_t candidate_result_capacity;
    uint32_t candidate_result_count;
} ShadowSpillPlanSelectionResult;

/*
 * Simulator-verify and deterministically select a PressureFit candidate.
 *
 * Candidate order is the stable policy tie-break: the lowest makespan wins,
 * then the lowest input index. All input programs and pointed-to arrays are
 * borrowed for the duration of the call. Output buffers are caller-owned.
 * The function has no global mutable state and is thread-safe for distinct
 * result buffers.
 */
SHADOWSPILL_PLANNER_API ShadowSpillPlannerStatus shadowspill_select_plan(
    const ShadowSpillPlanCandidate *candidates,
    uint32_t candidate_count,
    ShadowSpillPlanSelectionResult *result
);

SHADOWSPILL_PLANNER_API uint32_t shadowspill_planner_abi_version(void);

SHADOWSPILL_PLANNER_API const char *shadowspill_planner_status_string(
    ShadowSpillPlannerStatus status
);

#ifdef __cplusplus
}
#endif

#endif
