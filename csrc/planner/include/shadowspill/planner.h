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

#define SHADOWSPILL_PLANNER_ABI_VERSION 11U
#define SHADOWSPILL_PLANNER_NO_INDEX UINT32_MAX
#define SHADOWSPILL_PLANNER_DIGEST_BYTES 32U

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
    const uint64_t *fetch_runtime_ns;
    const uint64_t *evict_runtime_ns;
    const uint64_t *task_ideal_end_ns;
    const uint64_t *device_capacity_bytes;
    /* Maximum task-object pressure at each [device][boundary] cell. */
    const uint64_t *boundary_capacity_bytes;
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

typedef enum ShadowSpillResidencyStrategy {
    SHADOWSPILL_RESIDENCY_HEADROOM_STALL = 0,
    SHADOWSPILL_RESIDENCY_HEADROOM_TRANSFER = 1,
    SHADOWSPILL_RESIDENCY_TIGHT_STALL = 2,
    SHADOWSPILL_RESIDENCY_TIGHT_TRANSFER = 3,
    SHADOWSPILL_RESIDENCY_RELAXED_STALL = 4,
} ShadowSpillResidencyStrategy;

typedef enum ShadowSpillPrefetchRule {
    SHADOWSPILL_PREFETCH_PACKED_FIFO = 0,
    SHADOWSPILL_PREFETCH_PACKED_FIT = 1,
    SHADOWSPILL_PREFETCH_INTERVAL_ENTRY = 2,
    SHADOWSPILL_PREFETCH_LATEST_SAFE = 3,
    SHADOWSPILL_PREFETCH_DEMAND = 4,
} ShadowSpillPrefetchRule;

typedef enum ShadowSpillInitialPlacement {
    SHADOWSPILL_INITIAL_PLACEMENT_REQUIRED = 0,
    SHADOWSPILL_INITIAL_PLACEMENT_GREEDY = 1,
} ShadowSpillInitialPlacement;

/*
 * Dense schedule storage used by the complete compiled candidate evaluator.
 * Every identifier is the corresponding dense task or alias index in the
 * supplied simulation program. Arrays are owned by the result and remain
 * valid until shadowspill_pressurefit_context_result_destroy().
 */
typedef struct ShadowSpillDenseSchedule {
    uint32_t action_count;
    uint32_t *action_trigger_tasks;
    uint32_t *action_aliases;
    uint8_t *action_kinds;
    uint32_t initial_count;
    uint32_t *initial_aliases;
    uint8_t *initial_locations;
    uint32_t final_count;
    uint32_t *final_aliases;
    uint8_t *final_locations;
} ShadowSpillDenseSchedule;

/*
 * Schedule-invariant physical ownership facts for one execution pool.
 * Offsets have task_count + 1 entries and index the corresponding flattened
 * workspace-extent or alias arrays. Workspace extents are the simultaneously
 * live anonymous allocation multiset, not one artificial contiguous range.
 * Storage handoffs transfer a live lease from source to destination without
 * allocating. The arrays are borrowed for evaluation.
 */
typedef struct ShadowSpillAdmissionTopology {
    uint32_t abi_version;
    uint32_t task_count;
    uint32_t alias_count;
    uint64_t pool_capacity_bytes;
    uint64_t object_capacity_bytes;
    uint64_t minimum_alignment;
    const uint32_t *task_workspace_offsets;
    const uint64_t *task_workspace_extent_bytes;
    const uint32_t *fresh_output_offsets;
    const uint32_t *fresh_output_aliases;
    const uint32_t *replacement_offsets;
    const uint32_t *replacement_aliases;
    const uint32_t *handoff_offsets;
    const uint32_t *handoff_source_aliases;
    const uint32_t *handoff_destination_aliases;
    uint32_t allocation_slot_count;
    const uint32_t *task_allocation_offsets;
    const uint32_t *task_allocation_slots;
    const uint64_t *task_allocation_bytes;
    const uint32_t *task_allocation_aliases;
    const uint8_t *task_allocation_kinds;
} ShadowSpillAdmissionTopology;

typedef struct ShadowSpillPressureFitContext {
    uint32_t abi_version;
    const ShadowSpillResidencyProblem *residency;
    const ShadowSpillSimulationProgram *simulation;
    const uint8_t *seed_resident;
    const uint8_t *seed_breaks;
    const ShadowSpillAdmissionTopology *admission;

    /* JSON-escaped identifier payloads, without surrounding quotes. */
    const char *const *alias_json_names;
    const char *const *task_json_names;
} ShadowSpillPressureFitContext;

typedef struct ShadowSpillPressureFitContextOptions {
    const uint8_t *residency_strategies;
    uint32_t residency_strategy_count;
    const uint8_t *prefetch_rules;
    uint32_t prefetch_rule_count;
    uint8_t evaluate_coalesced;
    uint32_t max_repair_attempts;
    uint8_t initial_placement;
} ShadowSpillPressureFitContextOptions;

/*
 * Schedule-invariant input for the high-level compiled PressureFit path.
 * The simulation topology carries the selected tasks plus the declared
 * initial/final residency.  The planner derives the dense analytic residency
 * problem and initial seed internally before evaluating the unchanged
 * candidate portfolio.
 */
typedef struct ShadowSpillPressureFitProgramContext {
    uint32_t abi_version;
    const ShadowSpillSimulationProgram *simulation;
    const uint32_t *device_priority;
    const ShadowSpillAdmissionTopology *admission;

    /* JSON-escaped identifier payloads, without surrounding quotes. */
    const char *const *alias_json_names;
    const char *const *task_json_names;
} ShadowSpillPressureFitProgramContext;

typedef enum ShadowSpillCandidateStatus {
    SHADOWSPILL_CANDIDATE_VALID = 0,
    SHADOWSPILL_CANDIDATE_ANALYTIC_INFEASIBLE = 1,
    SHADOWSPILL_CANDIDATE_SIMULATION_INFEASIBLE = 2,
    SHADOWSPILL_CANDIDATE_ADMISSION_INFEASIBLE = 3,
    SHADOWSPILL_CANDIDATE_INTERNAL_ERROR = 4,
    SHADOWSPILL_CANDIDATE_REPAIR_EXHAUSTED = 5,
} ShadowSpillCandidateStatus;

/* Categorized monotonic repair operations for one candidate evaluation. */
typedef struct ShadowSpillPressureFitRepairDiagnostics {
    uint64_t admission_prefetch_advance_attempts;
    uint64_t admission_prefetch_delay_attempts;
    uint64_t admission_pressure_boundary_attempts;
    uint64_t simulation_prefetch_delay_attempts;
    uint64_t simulation_pressure_boundary_attempts;
} ShadowSpillPressureFitRepairDiagnostics;

/* Exact operations and summed component work for a candidate or context. */
typedef struct ShadowSpillPressureFitWorkDiagnostics {
    uint64_t evaluation_time_ns;
    uint64_t residency_cache_hits;
    uint64_t residency_cache_misses;
    uint64_t schedule_emissions;
    uint64_t schedule_cache_hits;
    uint64_t simulation_calls;
    uint64_t simulation_cache_hits;
    uint64_t admission_calls;
    uint64_t residency_time_ns;
    uint64_t schedule_time_ns;
    uint64_t simulation_time_ns;
    uint64_t admission_time_ns;
    uint64_t digest_time_ns;
} ShadowSpillPressureFitWorkDiagnostics;

typedef struct ShadowSpillPressureFitCandidateDiagnostic {
    uint8_t status;
    uint8_t residency_strategy;
    uint8_t prefetch_rule;
    uint8_t coalesced;
    ShadowSpillPressureFitRepairDiagnostics repairs;
    ShadowSpillPressureFitWorkDiagnostics work;
    uint32_t simulation_status;
    uint64_t makespan_ns;
    uint8_t schedule_digest[SHADOWSPILL_PLANNER_DIGEST_BYTES];

    uint32_t error_task;
    uint32_t error_alias;
    uint32_t error_device;
    uint8_t error_location;
    int32_t error_boundary;
    uint64_t error_time_ns;
    uint64_t error_capacity_bytes;
    uint64_t error_used_bytes;
    uint64_t error_requested_bytes;
    uint64_t error_required_bytes;
} ShadowSpillPressureFitCandidateDiagnostic;

typedef struct ShadowSpillPressureFitContextResult {
    uint32_t status;
    uint32_t selected_candidate_index;
    uint64_t selected_makespan_ns;
    ShadowSpillDenseSchedule selected_schedule;
    ShadowSpillPressureFitCandidateDiagnostic *candidates;
    uint32_t candidate_count;
    ShadowSpillPressureFitRepairDiagnostics repairs;
    ShadowSpillPressureFitWorkDiagnostics work;
} ShadowSpillPressureFitContextResult;

/* Caller-owned output buffers for one selected schedule's exact admission. */
typedef struct ShadowSpillScheduleAdmissionResult {
    uint32_t status;
    uint64_t decision_digest;
    uint64_t peak_allocated_bytes;
    uint64_t peak_reserved_bytes;
    uint64_t peak_fragmentation_bytes;
    uint64_t error_operation_index;
    uint64_t error_requested_bytes;
    uint64_t error_free_bytes;
    uint64_t error_largest_free_range_bytes;
    uint64_t initial_physical_bytes;

    int64_t *task_start_deltas;
    int64_t *task_completion_deltas;
    uint32_t task_capacity;
    int64_t *action_trigger_deltas;
    int64_t *action_completion_deltas;
    uint32_t action_capacity;
    uint32_t *reuse_predecessor_actions;
    uint32_t *reuse_successor_tasks;
    uint32_t *reuse_successor_actions;
    uint32_t reuse_capacity;
    uint32_t reuse_count;
} ShadowSpillScheduleAdmissionResult;

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

/*
 * Evaluate the complete deterministic candidate portfolio for one already
 * resolved recomputation selection. The function performs no Python calls and
 * retains dense residency and schedule records throughout evaluation. Result
 * storage is owned by the caller after success or a no-feasible result and
 * must be released with the matching destroy function.
 */
SHADOWSPILL_PLANNER_API ShadowSpillPlannerStatus
shadowspill_evaluate_pressurefit_context(
    const ShadowSpillPressureFitContext *context,
    const ShadowSpillPressureFitContextOptions *options,
    ShadowSpillPressureFitContextResult *result
);

/*
 * Derive one dense residency context from a selected simulation program and
 * evaluate the complete deterministic PressureFit candidate portfolio.
 * This is equivalent to constructing ShadowSpillResidencyProblem and its seed
 * explicitly, but avoids materializing alias-by-boundary matrices in Python.
 */
SHADOWSPILL_PLANNER_API ShadowSpillPlannerStatus
shadowspill_evaluate_pressurefit_program_context(
    const ShadowSpillPressureFitProgramContext *context,
    const ShadowSpillPressureFitContextOptions *options,
    ShadowSpillPressureFitContextResult *result
);

SHADOWSPILL_PLANNER_API ShadowSpillPlannerStatus
shadowspill_evaluate_schedule_admission(
    const ShadowSpillSimulationProgram *simulation,
    const ShadowSpillAdmissionTopology *admission,
    const ShadowSpillDenseSchedule *schedule,
    ShadowSpillScheduleAdmissionResult *result
);

SHADOWSPILL_PLANNER_API void
shadowspill_pressurefit_context_result_destroy(
    ShadowSpillPressureFitContextResult *result
);

SHADOWSPILL_PLANNER_API uint32_t shadowspill_planner_abi_version(void);

SHADOWSPILL_PLANNER_API const char *shadowspill_planner_status_string(
    ShadowSpillPlannerStatus status
);

#ifdef __cplusplus
}
#endif

#endif
