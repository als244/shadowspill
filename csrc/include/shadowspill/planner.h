#ifndef SHADOWSPILL_PLANNER_H
#define SHADOWSPILL_PLANNER_H

#include <stdint.h>
#include <shadowspill/shadowspill.h>

#include <shadowspill/simulator.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SHADOWSPILL_PLANNER_NO_INDEX UINT32_MAX
#define SHADOWSPILL_PLANNER_DIGEST_BYTES 32U

/* Declared here so problem options can name it; defined further down. */
typedef struct ShadowSpillBestPlaced ShadowSpillBestPlaced;
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
    /* The anchors of each alias as a sorted list: anchor_offsets[alias] ..
       anchor_offsets[alias + 1] index anchor_positions (boundary) and
       anchor_tasks (that cell's latest_access_task). The sparse companion
       of `anchors` and `latest_access_task`. */
    const uint32_t *anchor_offsets;
    const uint32_t *anchor_positions;
    const uint32_t *anchor_tasks;
    /* The boundaries each alias reserves for a produced output, as a sorted
       list: reserved_offsets[alias] .. reserved_offsets[alias + 1] index
       reserved_positions. The sparse companion of `output_reservations`. */
    const uint32_t *reserved_offsets;
    const uint32_t *reserved_positions;
    /* Per alias, whether the reducer may cut its residency; NULL means every
       alias may be cut. An alias that may not be cut stays resident from its
       first to its last access and is charged in the required floor; the
       emitter still produces its opening fetch, its release, and its
       terminal writeback. */
    const uint8_t *alias_evict_eligible;
    /* Per alias the reducer may not cut and that starts the step in spill:
       the task after which it is fetched, chosen once at preparation so the
       resident slice is sized for it. UINT32_MAX elsewhere. */
    const uint32_t *fixed_fetch_trigger;
} ShadowSpillResidencyProblem;

typedef struct ShadowSpillResidencyOptions {
    uint8_t minimize_transfer;
    uint8_t fetch_headroom;
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
    /* Optional: the aliases this reduction cut, in the order it cut them.
       NULL asks for no record, which is what planning normally wants. */
    uint32_t *cut_aliases;
    uint64_t cut_capacity;
    uint64_t cut_count;
} ShadowSpillResidencyResult;

typedef enum ShadowSpillResidencyStrategy {
    SHADOWSPILL_RESIDENCY_HEADROOM_STALL = 0,
    SHADOWSPILL_RESIDENCY_HEADROOM_TRANSFER = 1,
    SHADOWSPILL_RESIDENCY_TIGHT_STALL = 2,
    SHADOWSPILL_RESIDENCY_TIGHT_TRANSFER = 3,
    SHADOWSPILL_RESIDENCY_RELAXED_STALL = 4,
} ShadowSpillResidencyStrategy;

typedef enum ShadowSpillFetchRule {
    SHADOWSPILL_FETCH_PACKED_FIFO = 0,
    SHADOWSPILL_FETCH_PACKED_FIT = 1,
    SHADOWSPILL_FETCH_INTERVAL_ENTRY = 2,
    SHADOWSPILL_FETCH_LATEST_SAFE = 3,
    SHADOWSPILL_FETCH_DEMAND = 4,
} ShadowSpillFetchRule;

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
    /* The same topology, supplied for placement alone.
     *
     * `admission` above switches on the dynamic-pool replay, which is a
     * stricter and different question: it rejects schedules that certified
     * fixed placement accepts, so a search that prefilters through it
     * discards plans that would have run. Placement needs the topology
     * without that prefilter, so it is passed separately and the two are
     * deliberately not the same field. NULL leaves plans unplaced, which is
     * how a caller opts out of measuring layouts during the search. */
    const ShadowSpillAdmissionFacts *placement;

    /* JSON-escaped identifier payloads, without surrounding quotes. */
    const char *const *alias_json_names;
    const char *const *task_json_names;
} ShadowSpillPressureFitProblem;

typedef struct ShadowSpillPressureFitProblemOptions {
    const uint8_t *residency_strategies;
    uint32_t residency_strategy_count;
    const uint8_t *fetch_rules;
    uint32_t fetch_rule_count;
    /* Which coalescing modes to evaluate: 0 plain, 1 coalesced. A list like
       the two axes above, so a caller evaluates exactly the combinations it
       asks for; the product of the three counts is the candidate count. */
    const uint8_t *coalescing_modes;
    uint32_t coalescing_mode_count;
    uint32_t max_repair_attempts;
    uint8_t initial_placement;
    /* How much capacity a plan gives back at a time when its layout does
       not fit; zero hands back exactly what it overran. */
    uint64_t capacity_refinement_bytes;
    /* Record every plan each candidate held, at 48 bytes a step. Off by
       default: a corpus sweep would otherwise carry millions of steps it
       never reads. */
    uint8_t record_reduction_steps;
    /* The shared best-placed plan, or NULL to place without a gate. The
       planner never owns it: one object passed to several searches shares the
       gate between them, separate objects keep them independent. */
    ShadowSpillBestPlaced *best_placed;
    /* How many threads evaluate candidates. Zero means one per logical CPU,
       one means evaluate serially on the calling thread. Scheduling rather
       than search: it changes neither which plans are legal nor how they
       simulate, but it does change how many candidates the shared record
       lets a search skip, so per-candidate counters move with it. */
    uint32_t workers;
    /* Nonzero makes every candidate's outcome a pure function of its
       inputs: the placement gate consults only the candidate's own placed
       plans, never the shared record, so parallel evaluation is
       reproducible run to run. Costs additional placement measurements. */
    uint8_t deterministic;
    /* Aliases smaller than this many bytes are not eligible to be cut: they
       stay resident from their first to their last access. Zero makes every
       alias eligible. */
    uint64_t minimum_object_bytes_evict_eligible;
} ShadowSpillPressureFitProblemOptions;

/*
 * Schedule-invariant input for the high-level PressureFit path.
 * The simulation topology carries the selected tasks plus the declared
 * initial/final residency.  The planner derives the indexed analytic residency
 * problem and initial seed internally before evaluating the unchanged
 * candidate set.
 */
typedef struct ShadowSpillPressureFitProgramProblem {
    uint32_t abi_version;
    const ShadowSpillSimulationProgram *simulation;
    const uint32_t *device_priority;
    const ShadowSpillAdmissionFacts *admission;
    /* The topology, for placement alone. See the note on the same field of
       ShadowSpillPressureFitProblem: supplying `admission` switches on the
       dynamic-pool replay, which rejects plans certified fixed placement
       accepts, so the two are separate fields on purpose. */
    const ShadowSpillAdmissionFacts *placement;

    /* JSON-escaped identifier payloads, without surrounding quotes. */
    const char *const *alias_json_names;
    const char *const *task_json_names;
} ShadowSpillPressureFitProgramProblem;

typedef enum ShadowSpillPressureFitPreflightFailureKind {
    SHADOWSPILL_PREFLIGHT_NONE = 0,
    SHADOWSPILL_PREFLIGHT_WORKSPACE_CAPACITY = 1,
    SHADOWSPILL_PREFLIGHT_REQUIRED_CAPACITY = 2,
    SHADOWSPILL_PREFLIGHT_MISSING_INITIAL_RESIDENCY = 3,
    /* The resident slice -- a static home for every lease of an alias the
       reducer may not cut -- does not fit the device on its own. */
    SHADOWSPILL_PREFLIGHT_RESIDENT_SLICE_CAPACITY = 4,
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
    /* Every plan this candidate reached needed more contiguous pool than the
     * pool has. Its makespan was never the question: a plan with no layout
     * cannot run, so the candidate has no answer to offer. */
    SHADOWSPILL_CANDIDATE_UNPLACEABLE = 7,
} ShadowSpillCandidateStatus;

/* Categorized monotonic repair operations for one candidate evaluation. */
typedef struct ShadowSpillPressureFitRepairDiagnostics {
    uint64_t admission_fetch_advance_attempts;
    uint64_t admission_fetch_delay_attempts;
    uint64_t admission_pressure_boundary_attempts;
    uint64_t simulation_fetch_delay_attempts;
    uint64_t simulation_pressure_boundary_attempts;
} ShadowSpillPressureFitRepairDiagnostics;

/* Exact operations and summed component work for a candidate or problem. */
/*
 * Where a candidate's, or a problem's, time went.
 *
 * The sections are disjoint and are opened and closed by whichever function
 * orchestrates them, never by the work itself, so a reader can see the whole
 * partition in one place rather than inferring it from counters scattered
 * through the code that does the work.
 *
 * `total_ns` is what the orchestrator measured around everything below it, and
 * `residual_ns` is the part of that total the named sections do not claim --
 * allocation, bookkeeping, and the glue between one section and the next. It
 * is reported rather than left implicit so that the parts always add up to the
 * whole, and so an unexplained residual is visible as a number rather than as
 * a discrepancy someone has to notice.
 *
 * `admit_ns` is the one exception to disjointness and is marked as such: the
 * dynamic-pool replay happens inside a simulation, so its time is also part of
 * `simulate_ns` and must not be added again.
 */
typedef struct ShadowSpillPressureFitSectionTiming {
    uint64_t total_ns;
    /* Deriving the residency problem from the Program. Problem level only. */
    uint64_t prepare_ns;
    /* Schedule facts and the candidate workspace. */
    uint64_t setup_ns;
    /* Choosing what stays resident, before any candidate repairs it. */
    uint64_t reduce_ns;
    /* Turning residency gaps into an ordered schedule. */
    uint64_t emit_ns;
    /* Replaying the schedule for a makespan. */
    uint64_t simulate_ns;
    /* Moving a transfer or making room for one, and reducing again when that
       is what it took. */
    uint64_t repair_ns;
    /* Naming the schedule. */
    uint64_t digest_ns;
    /* Measuring whether the plan has a layout that fits. */
    uint64_t place_ns;
    /* Deciding what to answer with, and materialising it. */
    uint64_t select_ns;
    /* Releasing everything the evaluation held. */
    uint64_t teardown_ns;
    /* Inside `simulate_ns`, not beside it. */
    uint64_t admit_ns;
    /* `total_ns` less every disjoint section above. */
    uint64_t residual_ns;
} ShadowSpillPressureFitSectionTiming;

/*
 * One step of a candidate's descent: the plan it held, and what became of it.
 *
 * A candidate reduces, emits, simulates, and sometimes measures a layout,
 * over and over. Only the last plan survives in the result, so without a
 * record of the steps the reasons a candidate ended where it did are gone by
 * the time anyone asks. The step is kept small deliberately -- a candidate
 * can take hundreds, and a corpus sweep runs millions -- so it holds indices
 * and flags rather than anything it would have to allocate.
 *
 * `cut_alias` is the object whose residency the reduction gave up to reach
 * this step, or SHADOWSPILL_PLANNER_NO_INDEX for the first plan, which was
 * not reached by cutting anything.
 */
typedef struct ShadowSpillPressureFitReductionStep {
    uint64_t makespan_ns;
    /* Zero unless this step's layout was measured. */
    uint64_t required_bytes;
    /* The capacity this step was planned at. */
    uint64_t capacity_bytes;
    /* Where this step's cuts sit in the candidate's flat cut record, and
       how many there were. A reduction gives up several objects, so the
       aliases live once in one array rather than per step. */
    uint32_t cut_offset;
    uint32_t cut_count;
    /* Repairs spent when this step was reached. */
    uint32_t repairs;
    /* Simulator status, so a step that failed says how. */
    uint32_t simulation_status;
    /* Where the plan came up short, if it did. */
    uint32_t capacity_violations;
    /* A ShadowSpillReductionStepFlags bitmask. */
    uint32_t flags;
} ShadowSpillPressureFitReductionStep;

enum ShadowSpillReductionStepFlags {
    /* The plan simulated without error. */
    SHADOWSPILL_STEP_SIMULATED = 1U << 0U,
    /* Its layout was measured, so `required_bytes` means something. */
    SHADOWSPILL_STEP_MEASURED = 1U << 1U,
    /* That layout fit the pool. */
    SHADOWSPILL_STEP_PLACED = 1U << 2U,
    /* The plan gave capacity back after this step. */
    SHADOWSPILL_STEP_REFINED = 1U << 3U,
    /* This step was, when it was reached, the best plan the candidate had. */
    SHADOWSPILL_STEP_BEST = 1U << 4U,
    /* This step is the plan the candidate answered with. */
    SHADOWSPILL_STEP_ANSWER = 1U << 5U,
};

typedef struct ShadowSpillPressureFitWorkDiagnostics {
    uint64_t schedule_emissions;
    uint64_t schedule_cache_hits;
    uint64_t simulation_calls;
    uint64_t simulation_cache_hits;
    uint64_t admission_calls;
    ShadowSpillPressureFitSectionTiming sections;
} ShadowSpillPressureFitWorkDiagnostics;

typedef struct ShadowSpillPressureFitCandidateDiagnostic {
    uint8_t status;
    uint8_t residency_strategy;
    uint8_t fetch_rule;
    uint8_t coalesced;
    ShadowSpillPressureFitRepairDiagnostics repairs;
    ShadowSpillPressureFitWorkDiagnostics work;
    uint32_t simulation_status;
    uint64_t makespan_ns;
    /* Every plan this candidate held, in the order it held them. Owned by
       the result and released with it. Empty unless the caller asked for a
       trajectory, because a sweep does not want millions of these. */
    ShadowSpillPressureFitReductionStep *steps;
    uint32_t step_count;
    uint32_t step_capacity;
    /* Every alias this candidate ever cut, in order; steps index into it. */
    uint32_t *cut_aliases;
    uint32_t cut_count;
    uint32_t cut_capacity;
    /* How many places the accepted plan came up short of capacity and
       waited. Zero means it never waited for memory. */
    uint32_t capacity_violation_count;
    /* What placing this candidate's plans cost, and what it bought.
       `placements_attempted` counts measurements actually taken -- the gate
       and the plateau rule decide how few that is -- and
       `capacity_refinements` counts the times a plan gave back what it
       overran and reduced again. */
    uint32_t placements_attempted;
    uint32_t placements_admitted;
    uint32_t capacity_refinements;
    /* Repairs spent when the plan this candidate answers with was placed;
       UINT32_MAX when it placed none. */
    uint32_t repairs_at_best;
    uint8_t schedule_digest[SHADOWSPILL_PLANNER_DIGEST_BYTES];

    /* When this candidate ran, in nanoseconds from the start of the call
       that evaluated it. Unlike `work.sections`, which is work done, these
       are wall clock: two candidates ran at the same time exactly when
       their spans overlap, which is what makes a timeline of the workers
       readable. Both are zero for a candidate no worker reached. */
    uint64_t started_ns;
    uint64_t finished_ns;

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
    /* The problem's span on the same clock its candidates use: from the
       first candidate a worker started to the last one it finished. With
       several problems in one call these overlap, because workers take
       whatever task is next rather than finishing a problem first. */
    uint64_t started_ns;
    uint64_t finished_ns;
    /* The aliases `minimum_object_bytes_evict_eligible` kept resident: how
       many, their bytes, the resident slice reserved for them (bytes per
       device), and which they are (zero where an alias may not be cut, by
       alias index). Both arrays are owned by the result. */
    uint32_t evict_ineligible_aliases;
    uint64_t evict_ineligible_bytes;
    uint64_t *resident_slice_bytes;
    uint8_t *alias_evict_eligible;
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
SHADOWSPILL_API ShadowSpillStatus shadowspill_select_plan(
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
SHADOWSPILL_API ShadowSpillStatus
shadowspill_reduce_residency(
    const ShadowSpillResidencyProblem *problem,
    const ShadowSpillResidencyOptions *options,
    ShadowSpillResidencyResult *result
);

/*
 * Evaluate several resolved programs on one set of worker threads.
 *
 * A candidate of a problem is the unit of work, and every candidate of every
 * problem here competes for the same workers, so worker count and problem
 * count are independent. Results are written one per problem, in input order.
 * Worker count is scheduling only and never an input to an answer.
 *
 * The placement record in `options` is shared across all of them, which is
 * the point of evaluating them together: a plan placed under any resolved
 * program bounds the search under every other.
 *
 * `results` must have `problem_count` entries, each owned by the caller
 * afterwards and released with the matching destroy function -- including
 * when this returns a failure, since problems that completed still own
 * their storage.
 */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_evaluate_pressurefit_program_problems(
    const ShadowSpillPressureFitProgramProblem *problems,
    uint32_t problem_count,
    const ShadowSpillPressureFitProblemOptions *options,
    ShadowSpillPressureFitProblemResult *results
);

/*
 * Evaluate several already-derived residency problems together. Same
 * contract as above, for a caller that built the problems itself.
 */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_evaluate_pressurefit_problems(
    const ShadowSpillPressureFitProblem *problems,
    uint32_t problem_count,
    const ShadowSpillPressureFitProblemOptions *options,
    ShadowSpillPressureFitProblemResult *results
);

SHADOWSPILL_API ShadowSpillStatus
shadowspill_validate_pressurefit_program_problem(
    const ShadowSpillPressureFitProgramProblem *problem,
    ShadowSpillPressureFitPreflightResult *result
);

SHADOWSPILL_API ShadowSpillStatus
shadowspill_evaluate_schedule_admission(
    const ShadowSpillSimulationProgram *simulation,
    const ShadowSpillAdmissionFacts *admission,
    const ShadowSpillIndexedSchedule *schedule,
    ShadowSpillScheduleAdmissionResult *result
);

SHADOWSPILL_API void
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

SHADOWSPILL_API ShadowSpillStatus
shadowspill_admission_operation_bounds(
    const ShadowSpillSimulationProgram *simulation,
    const ShadowSpillAdmissionFacts *admission,
    const ShadowSpillIndexedSchedule *schedule,
    uint64_t *operation_capacity,
    uint64_t *lease_capacity
);

SHADOWSPILL_API ShadowSpillStatus
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

SHADOWSPILL_API ShadowSpillStatus shadowspill_build_lease_lifetimes(
    const ShadowSpillLeaseLifetimeProblem *problem,
    ShadowSpillLeaseLifetimeResult *result
);

/*
 * The best makespan any caller has actually placed, shared between searches.
 *
 * A plan no better than one already placed cannot win even if it places, so a
 * search consults this before paying for a placement. The object knows nothing
 * about candidates, resolved programs or calls: passing one object to several
 * concurrent searches shares the gate between them, and passing separate
 * objects keeps them independent. Safe to use from several threads at once.
 *
 * `offer` returns non-zero when the value replaced the previous best. `admits`
 * is the question a search asks: is this plan worth measuring at all.
 */

/* Which plan the best makespan belongs to.
 *
 * A bare makespan answers "is this worth measuring" but not "what won", and
 * the two are the same question asked at different times: whatever holds this
 * record at the end is the plan the search selected. `makespan_ns` of zero
 * means nothing has been placed. */
typedef struct ShadowSpillBestPlacedRecord {
    uint64_t makespan_ns;
    /* The capacity the plan was built against, which is a property of the
       plan rather than of the search that produced it. */
    uint64_t object_capacity_bytes;
    /* How much device capacity this plan gave back, applied to every device.
       Anything that re-times or re-measures the plan has to use the capacity
       it was built at: at full capacity it stalls less, so its timeline --
       and the lease lifetimes derived from that timeline -- are not the ones
       the plan was chosen on. */
    uint64_t capacity_given_back_bytes;
    uint8_t residency_strategy;
    uint8_t fetch_rule;
    uint8_t coalesced;
    uint8_t schedule_digest[SHADOWSPILL_PLANNER_DIGEST_BYTES];
} ShadowSpillBestPlacedRecord;

SHADOWSPILL_API ShadowSpillBestPlaced *shadowspill_best_placed_create(void);
SHADOWSPILL_API void shadowspill_best_placed_destroy(
    ShadowSpillBestPlaced *best
);
/* Copies out what is held. `makespan_ns` is zero when nothing was placed. */
SHADOWSPILL_API void shadowspill_best_placed_read(
    const ShadowSpillBestPlaced *best,
    ShadowSpillBestPlacedRecord *record
);

/* Fixed-offset placement of lease lifetimes within one execution-pool slice. */
typedef struct ShadowSpillPlacementProblem {
    uint32_t abi_version;
    uint32_t lifetime_count;
    const ShadowSpillLeaseLifetime *lifetimes;
    /* Per lifetime, nonzero leaves the lease out: its offset is not written
       and it is outside the span reported. NULL places every lease. */
    const uint8_t *excluded;
} ShadowSpillPlacementProblem;

/* `offsets` is caller-owned and must hold `lifetime_count` entries, written in
 * input order. `required_bytes` is the span the assignment covers. */
typedef struct ShadowSpillPlacementResult {
    uint64_t required_bytes;
    uint64_t *offsets;
} ShadowSpillPlacementResult;

/*
 * The size of one planner structure, so a caller mirroring these layouts can
 * check its mirror rather than discover a mismatch as corrupted fields. Takes
 * a ShadowSpillPlannerStruct; returns zero for anything it does not know.
 */
SHADOWSPILL_API uint64_t shadowspill_planner_struct_size(uint32_t which);

enum ShadowSpillPlannerStruct {
    SHADOWSPILL_STRUCT_PROBLEM_OPTIONS = 0,
    SHADOWSPILL_STRUCT_WORK_DIAGNOSTICS = 1,
    SHADOWSPILL_STRUCT_CANDIDATE_DIAGNOSTIC = 2,
    SHADOWSPILL_STRUCT_SECTION_TIMING = 3,
    SHADOWSPILL_STRUCT_REDUCTION_STEP = 4,
    SHADOWSPILL_STRUCT_ADMISSION_FACTS = 5,
    SHADOWSPILL_STRUCT_BEST_PLACED_RECORD = 6,
    SHADOWSPILL_STRUCT_RESIDENCY_PROBLEM = 7,
    SHADOWSPILL_STRUCT_RESIDENCY_RESULT = 8,
    SHADOWSPILL_STRUCT_PROBLEM_RESULT = 9,
};

SHADOWSPILL_API ShadowSpillStatus shadowspill_place_lifetimes(
    const ShadowSpillPlacementProblem *problem,
    ShadowSpillPlacementResult *result
);

#ifdef __cplusplus
}
#endif

#endif
