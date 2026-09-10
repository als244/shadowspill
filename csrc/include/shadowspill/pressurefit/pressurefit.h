/*
 * PressureFit: the search that ships with ShadowSpill.
 *
 * One implementation of the planning question <shadowspill/planner.h> poses.
 * It takes one ShadowSpillIndexedProblem per resolved program, places a
 * schedule for each under a shared best-placed record, and answers with the
 * fastest schedule that the simulator accepts and physical admission
 * certifies.
 *
 * Everything here is PressureFit's own: its options, its diagnostics, and the
 * record it shares across resolved programs. A caller that only wants to
 * plan, certify a schedule, or place leases needs <shadowspill/planner.h>
 * alone. A second search would ship its own header beside this one and reuse
 * that generic half unchanged.
 */

#ifndef SHADOWSPILL_PRESSUREFIT_H
#define SHADOWSPILL_PRESSUREFIT_H

#include <stdint.h>

#include <shadowspill/planner.h>
#include <shadowspill/shadowspill.h>
#include <shadowspill/simulator.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Declared here so problem options can name it; defined further down. */
typedef struct ShadowSpillPressureFitBestPlaced ShadowSpillPressureFitBestPlaced;
typedef enum ShadowSpillPressureFitResidencyStrategy {
    SHADOWSPILL_PRESSUREFIT_RESIDENCY_HEADROOM_STALL = 0,
    SHADOWSPILL_PRESSUREFIT_RESIDENCY_HEADROOM_TRANSFER = 1,
    SHADOWSPILL_PRESSUREFIT_RESIDENCY_TIGHT_STALL = 2,
    SHADOWSPILL_PRESSUREFIT_RESIDENCY_TIGHT_TRANSFER = 3,
    SHADOWSPILL_PRESSUREFIT_RESIDENCY_RELAXED_STALL = 4,
} ShadowSpillPressureFitResidencyStrategy;
typedef enum ShadowSpillPressureFitFetchRule {
    SHADOWSPILL_PRESSUREFIT_FETCH_PACKED_FIFO = 0,
    SHADOWSPILL_PRESSUREFIT_FETCH_PACKED_FIT = 1,
    SHADOWSPILL_PRESSUREFIT_FETCH_INTERVAL_ENTRY = 2,
    SHADOWSPILL_PRESSUREFIT_FETCH_LATEST_SAFE = 3,
    SHADOWSPILL_PRESSUREFIT_FETCH_DEMAND = 4,
} ShadowSpillPressureFitFetchRule;
typedef enum ShadowSpillPressureFitInitialPlacement {
    SHADOWSPILL_PRESSUREFIT_INITIAL_PLACEMENT_REQUIRED = 0,
    SHADOWSPILL_PRESSUREFIT_INITIAL_PLACEMENT_GREEDY = 1,
} ShadowSpillPressureFitInitialPlacement;
typedef struct ShadowSpillPressureFitOptions {
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
    ShadowSpillPressureFitBestPlaced *best_placed;
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
    /* Nonzero lets a plan that has simulated split an eviction whose copy
       fits in idle evict-lane time: a write-back where the lane is free, a
       release where the eviction was. The plan is simulated again and the
       split kept only if it got faster, so this widens what the search may
       consider rather than deciding anything by itself. */
    uint8_t split_write_backs;
    /* Aliases smaller than this many bytes are not eligible to be cut: they
       stay resident from their first to their last access. Zero makes every
       alias eligible. */
    uint64_t minimum_object_bytes_evict_eligible;
} ShadowSpillPressureFitOptions;
typedef enum ShadowSpillPressureFitPreflightFailureKind {
    SHADOWSPILL_PRESSUREFIT_PREFLIGHT_NONE = 0,
    SHADOWSPILL_PRESSUREFIT_PREFLIGHT_WORKSPACE_CAPACITY = 1,
    SHADOWSPILL_PRESSUREFIT_PREFLIGHT_REQUIRED_CAPACITY = 2,
    SHADOWSPILL_PRESSUREFIT_PREFLIGHT_MISSING_INITIAL_RESIDENCY = 3,
    /* The resident slice -- a static home for every lease of an alias the
       reducer may not cut -- does not fit the device on its own. */
    SHADOWSPILL_PRESSUREFIT_PREFLIGHT_RESIDENT_SLICE_CAPACITY = 4,
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
typedef enum ShadowSpillPressureFitCandidateStatus {
    SHADOWSPILL_PRESSUREFIT_CANDIDATE_VALID = 0,
    SHADOWSPILL_PRESSUREFIT_CANDIDATE_ANALYTIC_INFEASIBLE = 1,
    SHADOWSPILL_PRESSUREFIT_CANDIDATE_SIMULATION_INFEASIBLE = 2,
    SHADOWSPILL_PRESSUREFIT_CANDIDATE_ADMISSION_INFEASIBLE = 3,
    SHADOWSPILL_PRESSUREFIT_CANDIDATE_INTERNAL_ERROR = 4,
    SHADOWSPILL_PRESSUREFIT_CANDIDATE_REPAIR_EXHAUSTED = 5,
    /* Every plan this candidate reached needed more contiguous pool than the
     * pool has. Its makespan was never the question: a plan with no layout
     * cannot run, so the candidate has no answer to offer. */
    SHADOWSPILL_PRESSUREFIT_CANDIDATE_UNPLACEABLE = 6,
} ShadowSpillPressureFitCandidateStatus;
/* Categorized monotonic repair operations for one candidate evaluation. */
typedef struct ShadowSpillPressureFitRepairDiagnostics {
    uint64_t admission_fetch_advance_attempts;
    uint64_t admission_fetch_delay_attempts;
    uint64_t admission_pressure_boundary_attempts;
    uint64_t simulation_fetch_delay_attempts;
    uint64_t simulation_pressure_boundary_attempts;
} ShadowSpillPressureFitRepairDiagnostics;
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
 * `cut_offset` and `cut_count` say where this step's cuts sit in the
 * candidate's flat cut record: the objects whose residency the reduction gave
 * up to reach this step. A count of zero is the first plan, which was not
 * reached by cutting anything.
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
    /* A ShadowSpillPressureFitReductionStepFlags bitmask. */
    uint32_t flags;
} ShadowSpillPressureFitReductionStep;
enum ShadowSpillPressureFitReductionStepFlags {
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
    /* Pressure repairs that asked for more than the shortfall because the
       same failure had repeated, and how many of those asks no cut could
       meet and were taken back. */
    uint32_t pressure_escalations;
    uint32_t escalations_taken_back;
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
typedef struct ShadowSpillPressureFitResult {
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

    /* What became of the plan to beat, when the problem carried one:
       whether it did, the plan's status in the candidate vocabulary (valid,
       unplaceable, or the infeasibility that stopped it), the makespan it
       simulated to at this capacity, the pool bytes its layout needed, and
       whether it is the answer. When it is, `selected_candidate_index` is
       SHADOWSPILL_PLANNER_NO_INDEX and the selected schedule and makespan
       are its own. */
    uint8_t incumbent_given;
    uint8_t incumbent_status;
    uint8_t incumbent_selected;
    uint64_t incumbent_makespan_ns;
    uint64_t incumbent_required_bytes;
} ShadowSpillPressureFitResult;
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
shadowspill_pressurefit_search(
    const ShadowSpillIndexedProblem *problems,
    uint32_t problem_count,
    const ShadowSpillPressureFitOptions *options,
    ShadowSpillPressureFitResult *results
);
SHADOWSPILL_API ShadowSpillStatus
shadowspill_pressurefit_preflight(
    const ShadowSpillIndexedProblem *problem,
    ShadowSpillPressureFitPreflightResult *result
);
SHADOWSPILL_API void
shadowspill_pressurefit_result_destroy(
    ShadowSpillPressureFitResult *result
);
/* Which plan the best makespan belongs to.
 *
 * A bare makespan answers "is this worth measuring" but not "what won", and
 * the two are the same question asked at different times: whatever holds this
 * record at the end is the plan the search selected. `makespan_ns` of zero
 * means nothing has been placed. */
typedef struct ShadowSpillPressureFitBestPlacedRecord {
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
} ShadowSpillPressureFitBestPlacedRecord;
SHADOWSPILL_API ShadowSpillPressureFitBestPlaced *shadowspill_pressurefit_best_placed_create(void);
SHADOWSPILL_API void shadowspill_pressurefit_best_placed_destroy(
    ShadowSpillPressureFitBestPlaced *best
);
/* Copies out what is held. `makespan_ns` is zero when nothing was placed. */
SHADOWSPILL_API void shadowspill_pressurefit_best_placed_read(
    const ShadowSpillPressureFitBestPlaced *best,
    ShadowSpillPressureFitBestPlacedRecord *record
);

/* PressureFit's own structures, for shadowspill_planner_struct_size(). These
 * continue enum ShadowSpillPlannerStruct rather than restarting, so one call
 * answers for the generic planner and for this search. */
enum ShadowSpillPressureFitStruct {
    SHADOWSPILL_PRESSUREFIT_STRUCT_OPTIONS = 1,
    SHADOWSPILL_PRESSUREFIT_STRUCT_WORK_DIAGNOSTICS = 2,
    SHADOWSPILL_PRESSUREFIT_STRUCT_CANDIDATE_DIAGNOSTIC = 3,
    SHADOWSPILL_PRESSUREFIT_STRUCT_SECTION_TIMING = 4,
    SHADOWSPILL_PRESSUREFIT_STRUCT_REDUCTION_STEP = 5,
    SHADOWSPILL_PRESSUREFIT_STRUCT_BEST_PLACED_RECORD = 6,
    SHADOWSPILL_PRESSUREFIT_STRUCT_RESULT = 7,
};

#ifdef __cplusplus
}
#endif

#endif
