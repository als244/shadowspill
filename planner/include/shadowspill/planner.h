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

#define SHADOWSPILL_PLANNER_ABI_VERSION 2U
#define SHADOWSPILL_PLANNER_NO_INDEX UINT32_MAX

typedef enum ShadowSpillPlannerStatus {
    SHADOWSPILL_PLANNER_OK = 0,
    SHADOWSPILL_PLANNER_INVALID_ARGUMENT = 1,
    SHADOWSPILL_PLANNER_ALLOCATION_FAILURE = 2,
    SHADOWSPILL_PLANNER_NO_FEASIBLE_CANDIDATE = 3,
    SHADOWSPILL_PLANNER_INTERNAL_ERROR = 4,
    SHADOWSPILL_PLANNER_ANALYTIC_INFEASIBLE = 5,
} ShadowSpillPlannerStatus;

typedef struct ShadowSpillResidencyProblem {
    uint32_t abi_version;
    uint32_t alias_count;
    uint32_t boundary_count;
    uint32_t device_count;

    const uint64_t *alias_size_bytes;
    const uint32_t *alias_device;
    const uint8_t *alias_retain_spill_copy;
    const int8_t *initial_location;
    const int8_t *final_location;
    const uint8_t *anchors;
    const uint8_t *productions;
    const uint32_t *latest_access_task;
    const uint8_t *output_reservations;
    const uint8_t *write_prefix;
    const uint32_t *first_input_task;
    const uint64_t *h2d_runtime_ns;
    const uint64_t *d2h_runtime_ns;
    const uint64_t *task_ideal_end_ns;
    const uint64_t *device_capacity_bytes;
    const uint32_t *device_priority;
} ShadowSpillResidencyProblem;

typedef struct ShadowSpillResidencyOptions {
    uint8_t minimize_transfer;
    uint8_t prefetch_headroom;
    const uint8_t *seed_resident;
    const uint8_t *seed_breaks;
    const uint64_t *extra_pressure_bytes;
} ShadowSpillResidencyOptions;

typedef struct ShadowSpillResidencyResult {
    uint32_t status;
    uint32_t error_device;
    int32_t error_boundary;
    uint64_t required_bytes;
    uint64_t capacity_bytes;
    uint8_t *resident;
    uint64_t resident_capacity;
    uint8_t *breaks;
    uint64_t break_capacity;
} ShadowSpillResidencyResult;

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

/*
 * Apply the PressureFit analytic residency reduction to dense boundary data.
 * All input arrays are borrowed for the call. Output buffers are caller-owned
 * and require alias_count * boundary_count bytes each. `breaks` records a
 * logical span boundary after a resident boundary; its final column is unused.
 * The function is thread-safe for distinct output buffers.
 */
SHADOWSPILL_PLANNER_API ShadowSpillPlannerStatus
shadowspill_reduce_residency(
    const ShadowSpillResidencyProblem *problem,
    const ShadowSpillResidencyOptions *options,
    ShadowSpillResidencyResult *result
);

SHADOWSPILL_PLANNER_API uint32_t shadowspill_planner_abi_version(void);

SHADOWSPILL_PLANNER_API const char *shadowspill_planner_status_string(
    ShadowSpillPlannerStatus status
);

#ifdef __cplusplus
}
#endif

#endif
