#ifndef SHADOWSPILL_PLANNER_H
#define SHADOWSPILL_PLANNER_H

#include <stdint.h>
#include <shadowspill/shadowspill.h>

#include <shadowspill/simulator.h>

#if defined(_WIN32)
#define SHADOWSPILL_PLANNER_API __declspec(dllexport)
#else
#define SHADOWSPILL_PLANNER_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SHADOWSPILL_PLANNER_NO_INDEX UINT32_MAX
#define SHADOWSPILL_PLANNER_DIGEST_BYTES 32U
#define SHADOWSPILL_ADMISSION_NO_DEPENDENCY UINT64_MAX
#define SHADOWSPILL_ADMISSION_NO_OPERATION UINT64_MAX
#define SHADOWSPILL_ADMISSION_NO_LEASE UINT64_MAX

/* Planner names for the shared statuses; see <shadowspill/status.h>. */

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
 * Indexed schedule storage used by the complete compiled candidate evaluator.
 * Every identifier is the corresponding contiguous task or alias index in the
 * supplied simulation program. Arrays are owned by the result and remain
 * valid until shadowspill_pressurefit_problem_result_destroy().
 */
typedef struct ShadowSpillIndexedSchedule {
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
} ShadowSpillIndexedSchedule;

/*
 * Schedule-invariant physical ownership facts for one execution pool.
 * Offsets have task_count + 1 entries and index the corresponding flattened
 * workspace-extent or alias arrays. Workspace extents are the simultaneously
 * live anonymous allocation multiset, not one artificial contiguous range.
 * Storage handoffs transfer a live lease from source to destination without
 * allocating. The arrays are borrowed for evaluation.
 */
typedef struct ShadowSpillAdmissionFacts {
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
} ShadowSpillAdmissionFacts;

typedef struct ShadowSpillPressureFitProblem {
    uint32_t abi_version;
    const ShadowSpillResidencyProblem *residency;
    const ShadowSpillSimulationProgram *simulation;
    const uint8_t *seed_resident;
    const uint8_t *seed_breaks;
    const ShadowSpillAdmissionFacts *admission;

    /* JSON-escaped identifier payloads, without surrounding quotes. */
    const char *const *alias_json_names;
    const char *const *task_json_names;
} ShadowSpillPressureFitProblem;

typedef struct ShadowSpillPressureFitProblemOptions {
    const uint8_t *residency_strategies;
    uint32_t residency_strategy_count;
    const uint8_t *prefetch_rules;
    uint32_t prefetch_rule_count;
    uint8_t evaluate_coalesced;
    uint32_t max_repair_attempts;
    uint8_t initial_placement;
} ShadowSpillPressureFitProblemOptions;

/*
 * Schedule-invariant input for the high-level PressureFit path.
 * The simulation topology carries the selected tasks plus the declared
 * initial/final residency.  The planner derives the indexed analytic residency
 * problem and initial seed internally before evaluating the unchanged
 * candidate portfolio.
 */
typedef struct ShadowSpillPressureFitProgramProblem {
    uint32_t abi_version;
    const ShadowSpillSimulationProgram *simulation;
    const uint32_t *device_priority;
    const ShadowSpillAdmissionFacts *admission;

    /* JSON-escaped identifier payloads, without surrounding quotes. */
    const char *const *alias_json_names;
    const char *const *task_json_names;
} ShadowSpillPressureFitProgramProblem;

typedef enum ShadowSpillPressureFitPreflightFailureKind {
    SHADOWSPILL_PREFLIGHT_NONE = 0,
    SHADOWSPILL_PREFLIGHT_WORKSPACE_CAPACITY = 1,
    SHADOWSPILL_PREFLIGHT_REQUIRED_CAPACITY = 2,
    SHADOWSPILL_PREFLIGHT_MISSING_INITIAL_RESIDENCY = 3,
} ShadowSpillPressureFitPreflightFailureKind;

/* Structured semantic feasibility result produced before candidate search. */
typedef struct ShadowSpillPressureFitPreflightResult {
    uint32_t status;
    uint8_t failure_kind;
    uint32_t error_device;
    uint32_t error_alias;
    int32_t error_boundary;
    uint64_t required_bytes;
    uint64_t capacity_bytes;
} ShadowSpillPressureFitPreflightResult;

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

/* Exact operations and summed component work for a candidate or problem. */
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

typedef struct ShadowSpillPressureFitProblemResult {
    uint32_t status;
    uint32_t selected_candidate_index;
    uint64_t selected_makespan_ns;
    ShadowSpillIndexedSchedule selected_schedule;
    ShadowSpillPressureFitCandidateDiagnostic *candidates;
    uint32_t candidate_count;
    ShadowSpillPressureFitRepairDiagnostics repairs;
    ShadowSpillPressureFitWorkDiagnostics work;
} ShadowSpillPressureFitProblemResult;

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
SHADOWSPILL_PLANNER_API ShadowSpillStatus shadowspill_select_plan(
    const ShadowSpillPlanCandidate *candidates,
    uint32_t candidate_count,
    ShadowSpillPlanSelectionResult *result
);

/*
 * Apply the PressureFit analytic residency reduction to indexed boundary data.
 * All input arrays are borrowed for the call. Output buffers are caller-owned
 * and require alias_count * boundary_count bytes each. `breaks` records a
 * logical span boundary after a resident boundary; its final column is unused.
 * The function is thread-safe for distinct output buffers.
 */
SHADOWSPILL_PLANNER_API ShadowSpillStatus
shadowspill_reduce_residency(
    const ShadowSpillResidencyProblem *problem,
    const ShadowSpillResidencyOptions *options,
    ShadowSpillResidencyResult *result
);

/*
 * Evaluate the complete deterministic candidate portfolio for one already
 * resolved recomputation selection. The function performs no Python calls and
 * retains indexed residency and schedule records throughout evaluation. Result
 * storage is owned by the caller after success or a no-feasible result and
 * must be released with the matching destroy function.
 */
SHADOWSPILL_PLANNER_API ShadowSpillStatus
shadowspill_evaluate_pressurefit_problem(
    const ShadowSpillPressureFitProblem *problem,
    const ShadowSpillPressureFitProblemOptions *options,
    ShadowSpillPressureFitProblemResult *result
);

/*
 * Derive one indexed residency problem from a selected simulation program and
 * evaluate the complete deterministic PressureFit candidate portfolio.
 * This is equivalent to constructing ShadowSpillResidencyProblem and its seed
 * explicitly, but avoids materializing alias-by-boundary matrices in Python.
 */
SHADOWSPILL_PLANNER_API ShadowSpillStatus
shadowspill_evaluate_pressurefit_program_problem(
    const ShadowSpillPressureFitProgramProblem *problem,
    const ShadowSpillPressureFitProblemOptions *options,
    ShadowSpillPressureFitProblemResult *result
);

SHADOWSPILL_PLANNER_API ShadowSpillStatus
shadowspill_validate_pressurefit_program_problem(
    const ShadowSpillPressureFitProgramProblem *problem,
    ShadowSpillPressureFitPreflightResult *result
);

SHADOWSPILL_PLANNER_API ShadowSpillStatus
shadowspill_evaluate_schedule_admission(
    const ShadowSpillSimulationProgram *simulation,
    const ShadowSpillAdmissionFacts *admission,
    const ShadowSpillIndexedSchedule *schedule,
    ShadowSpillScheduleAdmissionResult *result
);

SHADOWSPILL_PLANNER_API void
shadowspill_pressurefit_problem_result_destroy(
    ShadowSpillPressureFitProblemResult *result
);

/* The pool operations a schedule implies, with the provenance a fixed layout
 * needs: why each lease exists, and which task or action it belongs to.
 *
 * `shadowspill_admission_operation_bounds` reports how many entries the arrays
 * below must hold; `shadowspill_build_admission_operations` fills them. All
 * arrays are caller-owned, so the builder allocates nothing the caller must
 * release.
 */
typedef struct ShadowSpillAdmissionOperations {
    /* Caller-owned, `operation_capacity` entries each, indexed alike. An
     * operation's sequence is its index. */
    uint64_t *lease_ids;
    /* The completion a reuse of this lease's address must wait for, or
     * SHADOWSPILL_ADMISSION_NO_DEPENDENCY where the operation publishes none. */
    uint64_t *dependency_ids;
    uint64_t *bytes;
    uint64_t *alignments;
    uint8_t *kinds;       /* ShadowSpillAdmissionReplayOperationKind */
    uint8_t *purposes;    /* why the lease exists */
    uint8_t *boundaries;  /* where in the step it sits */
    uint32_t *indices;    /* which task or action, per the boundary */
    /* For a task allocation, its offset into the topology's flattened
     * allocation arrays; SHADOWSPILL_PLANNER_NO_INDEX otherwise. This is what
     * ties a lease back to the allocation step that produced it. */
    uint32_t *allocation_offsets;
    uint64_t operation_capacity;

    /* Caller-owned, `lease_capacity` entries each. `lease_aliases` is the
     * alias a lease carries, or SHADOWSPILL_PLANNER_NO_INDEX for anonymous
     * task workspace. The other two are the operations that create and retire
     * it, so a reader can go straight to a lease without scanning: several
     * operations touch each lease and most touch none that matters.
     * `lease_retires` is SHADOWSPILL_ADMISSION_NO_OPERATION for a lease that
     * outlives the step. */
    uint32_t *lease_aliases;
    uint64_t *lease_starts;
    uint64_t *lease_retires;
    uint64_t lease_capacity;

    /* Filled by the builder. */
    uint64_t operation_count;
    uint64_t lease_count;
    uint64_t dependency_count;

    /* Bytes each transfer lane must move. A schedule cannot finish sooner
     * than its busiest lane, so these bound its makespan without simulating. */
    uint64_t fetch_bytes;
    uint64_t evict_bytes;
} ShadowSpillAdmissionOperations;

SHADOWSPILL_PLANNER_API ShadowSpillStatus
shadowspill_admission_operation_bounds(
    const ShadowSpillSimulationProgram *simulation,
    const ShadowSpillAdmissionFacts *admission,
    const ShadowSpillIndexedSchedule *schedule,
    uint64_t *operation_capacity,
    uint64_t *lease_capacity
);

SHADOWSPILL_PLANNER_API ShadowSpillStatus
shadowspill_build_admission_operations(
    const ShadowSpillSimulationProgram *simulation,
    const ShadowSpillAdmissionFacts *admission,
    const ShadowSpillIndexedSchedule *schedule,
    ShadowSpillAdmissionOperations *result
);

/* One lease to place: how much space it needs, how that space must be
 * aligned, and the half-open interval over which it is live. Two leases
 * conflict when their intervals intersect, so leases whose intervals merely
 * touch may share an offset.
 *
 * Placement is told nothing else. It never sees lease identity: offsets come
 * back in input order, and the input index breaks every tie, so the result
 * depends only on the records and the order they arrive in.
 */
typedef struct ShadowSpillLeaseLifetime {
    uint64_t bytes;
    uint64_t alignment;
    uint64_t start_ns;
    uint64_t end_ns;
} ShadowSpillLeaseLifetime;

/* Everything about a lease except when it is live: why it exists, what it
 * belongs to, and where it sits in the operation order. Every identifier is an
 * index into the caller's own tables — no strings enter the planner.
 *
 * `causal_start` and `causal_end` are operation sequence numbers, not times.
 * They are what makes a shared offset safe: a layout may only reuse an address
 * when the predecessor's `causal_end` precedes the successor's `causal_start`,
 * which no amount of timing drift can change.
 */
typedef struct ShadowSpillLeaseIdentity {
    uint64_t lease_id;
    uint64_t causal_start;
    uint64_t causal_end;
    uint32_t task;    /* SHADOWSPILL_PLANNER_NO_INDEX where it names no task */
    uint32_t alias;   /* SHADOWSPILL_PLANNER_NO_INDEX for anonymous workspace */
    uint32_t action;  /* SHADOWSPILL_PLANNER_NO_INDEX unless an action made it */
    uint8_t purpose;  /* ShadowSpillAdmissionPurpose */
} ShadowSpillLeaseIdentity;

/* Resolving one schedule's operations into the lifetimes a layout places.
 *
 * The operations say which lease each one creates and retires; the simulated
 * intervals say when. Joining them is all this does. `dynamic_aliases` names
 * the caller-owned terminal aliases whose final lease must stay out of the
 * reusable fixed slice.
 */
typedef struct ShadowSpillLeaseLifetimeProblem {
    uint32_t abi_version;
    const ShadowSpillAdmissionOperations *operations;
    const ShadowSpillAdmissionFacts *admission;
    const ShadowSpillIndexedSchedule *schedule;
    const ShadowSpillTaskInterval *task_intervals;
    uint32_t task_interval_count;
    const ShadowSpillTransferInterval *transfer_intervals;
    uint32_t transfer_interval_count;
    uint64_t makespan_ns;
    const uint32_t *dynamic_aliases;
    uint32_t dynamic_alias_count;
} ShadowSpillLeaseLifetimeProblem;

/* Caller-owned throughout; the builder allocates nothing the caller frees.
 *
 * `lifetimes` and `identities` hold `operations->lease_count` entries and are
 * indexed alike. Fixed leases occupy `[0, fixed_count)` and dynamic ones
 * follow, so placement runs on the prefix without a copy. Lease order is
 * preserved within each part.
 *
 * `allocation_step_leases` has one entry per flattened allocation step and
 * `alias_leases` one per alias: the lease each names when the step ends, or
 * SHADOWSPILL_ADMISSION_NO_LEASE. They are what a certificate's lookup tables
 * are built from.
 */
typedef struct ShadowSpillLeaseLifetimeResult {
    ShadowSpillLeaseLifetime *lifetimes;
    ShadowSpillLeaseIdentity *identities;
    uint64_t *allocation_step_leases;
    uint64_t *alias_leases;
    uint64_t lifetime_count;
    uint64_t fixed_count;
} ShadowSpillLeaseLifetimeResult;

SHADOWSPILL_PLANNER_API ShadowSpillStatus shadowspill_build_lease_lifetimes(
    const ShadowSpillLeaseLifetimeProblem *problem,
    ShadowSpillLeaseLifetimeResult *result
);

/* Fixed-offset placement of lease lifetimes within one execution-pool slice. */
typedef struct ShadowSpillPlacementProblem {
    uint32_t abi_version;
    uint32_t lifetime_count;
    const ShadowSpillLeaseLifetime *lifetimes;
} ShadowSpillPlacementProblem;

/* `offsets` is caller-owned and must hold `lifetime_count` entries, written in
 * input order. `required_bytes` is the span the assignment covers. */
typedef struct ShadowSpillPlacementResult {
    uint64_t required_bytes;
    uint64_t *offsets;
} ShadowSpillPlacementResult;

SHADOWSPILL_PLANNER_API ShadowSpillStatus shadowspill_place_lifetimes(
    const ShadowSpillPlacementProblem *problem,
    ShadowSpillPlacementResult *result
);

#ifdef __cplusplus
}
#endif

#endif
