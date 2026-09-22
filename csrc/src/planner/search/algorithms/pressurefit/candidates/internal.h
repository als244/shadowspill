#ifndef SHADOWSPILL_PRESSUREFIT_CANDIDATES_INTERNAL_H
#define SHADOWSPILL_PRESSUREFIT_CANDIDATES_INTERNAL_H

/*
 * One resolved problem searched: the stages a candidate passes through, and
 * the state they share.
 *
 * A candidate is a residency -- which cells are kept at which boundary --
 * turned into a plan and measured. The search walks stages over it: reduce
 * the residency, emit a schedule, simulate it, place it physically, repair
 * what the simulation or the admission refused, and settle. Each stage is
 * one function returning a StageOutcome, and this directory holds one file
 * per layer beneath them: the buffers a candidate reuses, the caches that
 * keep a schedule from being emitted or simulated twice, the plan built from
 * a residency, the repairs, and the work accounting every stage records.
 */

#include "../../../../admission/internal.h"
#include "../../../../../common/platform.h"
#include "../../../../internal.h"
#include "../candidates_internal.h"
#include "../residency_internal.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The candidate index that names the incumbent rather than a candidate. */
#define INCUMBENT_CANDIDATE (SHADOWSPILL_PLANNER_NO_INDEX - 1U)

typedef struct SimulationWorkspace {
    ShadowSpillTaskInterval *tasks;
    ShadowSpillTransferInterval *transfers;
    ShadowSpillDevicePeak *peaks;
    uint32_t task_capacity;
    uint32_t transfer_capacity;
    uint32_t device_capacity;
} SimulationWorkspace;


typedef struct HashSlot {
    uint64_t hash;
    uint32_t entry_plus_one;
} HashSlot;

typedef struct HashIndex {
    HashSlot *slots;
    uint32_t capacity;
    uint32_t count;
} HashIndex;

/* A 128-bit fingerprint: of a residency (the cells it keeps) or of a
 * schedule (its actions and boundary residency).
 *
 * Two independent 64-bit hashes rather than one, because this is compared
 * instead of the bytes themselves. A single 64-bit hash would collide about
 * once in 2^32 distinct values, which a long search would reach; 128 bits
 * puts it beyond reach, and comparing sixteen bytes replaces comparing
 * megabytes. */

typedef struct Fingerprint {
    uint64_t low;
    uint64_t high;
} Fingerprint;


typedef struct ScheduleMemoEntry {
    uint64_t hash;
    Fingerprint residency;
    uint8_t rule;
    uint8_t coalesced;
    uint8_t fetch_headroom;
    ShadowSpillIndexedSchedule schedule;
} ScheduleMemoEntry;

/* The last few emitted schedules, by (residency, rule, coalescing, headroom).
 *
 * Re-emission hits are recency-local: a candidate that keeps its residency
 * while it adjusts placement asks for the same schedule again within a few
 * cycles. A fixed ring keeps those hits and bounds the memory by
 * construction; an evicted schedule is simply emitted again. */
#define SCHEDULE_MEMO_CAPACITY 16U
typedef struct ScheduleMemo {
    ScheduleMemoEntry entries[SCHEDULE_MEMO_CAPACITY];
    uint32_t count;
    uint32_t next;
} ScheduleMemo;

/* One simulated schedule's outcome, identified by the schedule's fingerprint. */
typedef struct SimulationMemoEntry {
    Fingerprint identity;
    ShadowSpillSimulationResult result;
    /* Held by value, because the buffer the simulator wrote it into belongs
     * to the workspace and the next candidate overwrites it. */
    ShadowSpillCapacityViolation first_violation;
    ShadowSpillAdmissionReplayResult admission;
    uint32_t admission_status;
    uint8_t digest[SHADOWSPILL_PLANNER_DIGEST_BYTES];
    uint8_t digest_valid;
} SimulationMemoEntry;

typedef struct SimulationMemo {
    SimulationMemoEntry *entries;
    uint32_t count;
    uint32_t capacity;
    HashIndex index;
} SimulationMemo;


/*
 * Buffers for deciding whether one plan can be placed in the execution pool.
 *
 * Placement runs on the plan a candidate currently holds, so a candidate
 * places many plans over its life and the sizes barely change between them.
 * The arrays are grown on demand and reused rather than allocated per plan.
 */

typedef struct PlacementWorkspace {
    ShadowSpillAdmissionOperations operations;
    uint64_t *lease_ids;
    uint64_t *dependency_ids;
    uint64_t *bytes;
    uint64_t *alignments;
    uint8_t *kinds;
    uint8_t *purposes;
    uint8_t *boundaries;
    uint32_t *indices;
    uint32_t *allocation_offsets;
    uint32_t *lease_aliases;
    uint64_t *lease_starts;
    uint64_t *lease_retires;
    ShadowSpillLeaseLifetime *lifetimes;
    ShadowSpillLeaseIdentity *identities;
    uint64_t *allocation_step_leases;
    uint64_t *alias_leases;
    uint64_t *offsets;
    /* Per lease, whether placement leaves it out: the lease of an alias the
     * reducer may not cut takes a static home in the resident slice instead. */
    uint8_t *excluded;
    uint32_t *dynamic_aliases;
    /* What the last measurement placed: how many leases, the extent they
     * span, and the lease whose end is that extent -- what a layout that
     * overran the pool is answered around. */
    uint64_t placed_count;
    uint64_t extent_bytes;
    uint64_t extent_lease;
    uint64_t operation_capacity;
    uint64_t lease_capacity;
    uint32_t alias_capacity;
    uint32_t allocation_slot_capacity;
} PlacementWorkspace;


typedef struct CandidateWorkspace {
    /* What this plan has given back: the bytes its layout overran. It
     * shapes the plan -- the reducer charges it through `extra_pressure`
     * and the emitter measures against it -- and is reported on the plan so
     * a reader can tell what it was built for. The simulator is deliberately
     * not one of its readers: the plan runs on the machine the caller
     * described, so that is the capacity it is timed at. */
    uint64_t plan_capacity_given_back;
    /* Where the last repair pressure went, so an ask that cannot be met can
     * be taken back from the boundary it was made at. */
    uint64_t last_pressure_position;
    uint8_t *resident;
    uint8_t *breaks;
    uint8_t *base_resident;
    uint8_t *base_breaks;
    uint8_t *repair_resident;
    uint8_t *repair_breaks;
    uint8_t *removable_aliases;
    uint64_t *extra_pressure;
    ShadowSpillScheduleStorage schedule;
    ShadowSpillScheduleStorage selected;
    /* The best plan this candidate has reached, kept because repairing
     * past a success can make it worse before it makes it better. */
    ShadowSpillScheduleStorage best;
    SimulationWorkspace simulation;
    /* Where a plan first came up short, which is what repair aims at
     * when the plan simulates but waits for memory. */
    ShadowSpillCapacityViolation first_violation;
    /* Buffers for placing a plan physically. Grown on demand and reused,
     * because a candidate places many plans and the sizes barely move. */
    PlacementWorkspace placement;
    ShadowSpillCandidateAdmissionWorkspace admission;
    /* Cell geometry, and the problem's seed residency packed once per
     * workspace so every reduction starts from packed bitmaps. */
    size_t cell_count;
    size_t packed_cell_count;
    uint8_t *packed_seed_resident;
    uint8_t *packed_seed_breaks;
    ScheduleMemo schedule_memo;
    SimulationMemo simulation_memo;
    ShadowSpillPressureFitResidencyWorkspace *residency_workspace;
    ShadowSpillFetchTriggerConstraint *fetch_constraints;
    uint32_t fetch_constraint_count;
    uint32_t fetch_constraint_capacity;
    /* Which residency the workspace currently holds, and the one every
     * candidate of this strategy starts from. Content, not position, so a
     * memo below can drop entries without invalidating anything. */
    Fingerprint current_residency;
    Fingerprint base_residency;
    uint64_t schedule_emissions;
    uint64_t schedule_cache_hits;
    uint64_t simulation_calls;
    uint64_t simulation_cache_hits;
    /* Where the time went. Written only by the functions that orchestrate
     * the work, never by the work itself. */
    ShadowSpillPressureFitSectionTiming sections;
    /* Scratch the reducer appends its cuts to, when a trajectory is being
     * recorded. Drained into the candidate's record after each reduction. */
    uint32_t *cut_scratch;
    uint64_t cut_scratch_capacity;
    uint64_t cut_scratch_count;
} CandidateWorkspace;

/*
 * One section of an orchestrator's time.
 *
 * Opened and closed around a call, so the partition of a function's time is
 * written where that function orchestrates the work rather than inside the
 * work. A section may be opened against any sink; nesting is expressed by
 * which sink it is given, not by the helper.
 */

typedef struct Section {
    uint64_t started;
    uint64_t *sink;
} Section;


typedef enum StageOutcome {
    /* Run the next stage of this round. */
    STAGE_NEXT = 0,
    /* Something changed; start a new round. */
    STAGE_REPEAT,
    /* The candidate is finished. `search->answer` is what to report. */
    STAGE_DONE,
} StageOutcome;


typedef struct CandidateSearch {
    /* Fixed for the whole search. */
    const ShadowSpillPressureFitProblem *problem;
    /* The problem's facts, held by value so this candidate can point them at
     * the capacity *it* has given back. The problem shares one set of facts
     * between workers, but what a plan gave back belongs to the worker
     * building it, so the pointer cannot live in the shared copy. */
    ShadowSpillScheduleFacts facts;
    const ShadowSpillPressureFitOptions *options;
    CandidateWorkspace *workspace;
    ShadowSpillPressureFitCandidateDiagnostic *diagnostic;
    ShadowSpillPressureFitResidencyOptions reduce_options;
    uint8_t strategy;
    uint8_t rule;
    uint8_t coalesced;
    /* Whether there is a pool to place into at all. Without one the
     * candidate answers with its fastest plan, which is what a caller that
     * supplied no topology can be told. */
    int placing;
    uint64_t cells;
    uint64_t pressure_cells;

    /* Moves as the search runs. */
    int need_emit;
    /* Capacity is a property of the plan, so it travels with it. */
    uint64_t plan_capacity_bytes;
    /* The best plan the search set aside, held in `workspace->best`. */
    uint64_t best_makespan_ns;
    uint8_t best_digest[SHADOWSPILL_PLANNER_DIGEST_BYTES];
    /* The plan this candidate would answer with: the best it has placed,
     * which is not always the best it has simulated. */
    uint64_t placed_makespan_ns;
    /* The plan last placed, by fingerprint: placing it again is skipped. */
    Fingerprint placed_identity;
    /* Where the last simulation came up short, and how many times in a row
     * it has come up short there: a repeat asks the reducer for more. */
    uint32_t last_error_task;
    uint64_t last_error_time_ns;
    uint32_t failure_repeats;
    /* The layout move in flight: whether the plan being measured followed a
     * fetch delay, and the extent the plan before it needed. A delay that
     * did not make the extent fall is the last one at this capacity. */
    int layout_moved;
    uint64_t layout_required_before_move;

    /* The round in hand: what the last simulation produced. */
    ShadowSpillSimulationResult simulation;
    ShadowSpillStatus simulation_status;
    ShadowSpillStatus admission_status;
    ShadowSpillAdmissionReplayResult admission_result;
    ShadowSpillAdmissionAnnotation admission_annotation;
    SimulationMemoEntry *simulation_entry;
    int improves;

    /* What `evaluate_candidate` returns, set with STAGE_DONE. */
    int answer;
} CandidateSearch;


typedef struct SearchedProblem {
    const ShadowSpillPressureFitProblem *problem;
    /* A caller may hand in a residency problem without its sparse anchor and
     * reservation lists; the search then derives them once, here, and works
     * from its own copy of the problem. */
    ShadowSpillPressureFitProblem owned_problem;
    ShadowSpillPressureFitResidencyProblem owned_residency;
    ShadowSpillPressureFitResidencySparseLists lists;
    int derived;
    ShadowSpillScheduleFacts facts;
    ShadowSpillPressureFitResult *result;
    /* Global index of this problem's first candidate. */
    uint32_t first_task;
    uint32_t candidate_count;
    /* The best plan any worker placed for this problem, and its schedule.
     * Guarded because several workers can beat it at once. */
    ShadowSpillScheduleStorage selected;
    uint64_t selected_makespan_ns;
    uint32_t selected_candidate;
    atomic_flag guard;
    int ready;
} SearchedProblem;

typedef struct ProgramSearch {
    const ShadowSpillPressureFitOptions *options;
    SearchedProblem *problems;
    uint32_t problem_count;
    uint32_t total_tasks;
    _Atomic uint32_t next_task;
    /* The clock every reported timestamp is measured from, so a caller
     * reading several problems out of one call sees one timeline. */
    uint64_t origin_ns;
} ProgramSearch;

typedef struct SearchWorker {
    ProgramSearch *search;
    CandidateWorkspace workspace;
    /* Which problem the workspace is sized for, or NO_INDEX before the first
     * task. Sizes come from the problem, so this cannot be shared across
     * problems without rebuilding. */
    uint32_t workspace_problem;
    int failed;
} SearchWorker;

/* Which resolved problem owns a global task index, and which of its
 * candidates the task names. */


Section shadowspill_candidate_section_open(uint64_t *sink);
void shadowspill_candidate_section_close(Section section);
void shadowspill_candidate_section_close_total( ShadowSpillPressureFitSectionTiming *timing, uint64_t started );
int shadowspill_candidate_record_step( ShadowSpillPressureFitCandidateDiagnostic *diagnostic, ShadowSpillPressureFitReductionStep step );
int shadowspill_candidate_drain_cuts( ShadowSpillPressureFitCandidateDiagnostic *diagnostic, CandidateWorkspace *workspace );
void shadowspill_candidate_mark_last_step( ShadowSpillPressureFitCandidateDiagnostic *diagnostic, uint32_t flags, uint64_t required_bytes );
uint64_t shadowspill_candidate_repair_total( const ShadowSpillPressureFitRepairDiagnostics *repairs );
ShadowSpillPressureFitWorkDiagnostics shadowspill_candidate_workspace_work( const CandidateWorkspace *workspace );
ShadowSpillPressureFitWorkDiagnostics shadowspill_candidate_work_delta( ShadowSpillPressureFitWorkDiagnostics after, ShadowSpillPressureFitWorkDiagnostics before );
ShadowSpillPressureFitWorkDiagnostics shadowspill_candidate_add_work( ShadowSpillPressureFitWorkDiagnostics total, ShadowSpillPressureFitWorkDiagnostics part );
void shadowspill_candidate_add_repairs( ShadowSpillPressureFitRepairDiagnostics *destination, const ShadowSpillPressureFitRepairDiagnostics *source );
int shadowspill_candidate_fingerprint_equal(Fingerprint left, Fingerprint right);
uint64_t shadowspill_candidate_hash_bytes(uint64_t hash, const void *data, size_t size);
uint64_t shadowspill_candidate_hash_bytes_high(uint64_t hash, const void *data, size_t size);
int shadowspill_candidate_simulate_cached( const ShadowSpillPressureFitProblem *problem, CandidateWorkspace *workspace, ShadowSpillSimulationResult *result, ShadowSpillStatus *admission_status, ShadowSpillAdmissionReplayResult *admission_result, ShadowSpillAdmissionAnnotation *admission_error_annotation, SimulationMemoEntry **selected_entry );
int shadowspill_candidate_emit_cached( const ShadowSpillScheduleFacts *facts, CandidateWorkspace *workspace, const uint8_t *resident, const uint8_t *breaks, uint8_t rule, uint8_t coalesced, uint8_t fetch_headroom );
void shadowspill_candidate_free_indexed_schedule(ShadowSpillIndexedSchedule *schedule);
int shadowspill_candidate_simulation_workspace_reserve_transfers( SimulationWorkspace *workspace, uint32_t capacity );
int shadowspill_candidate_placement_reserve( PlacementWorkspace *workspace, uint64_t operations, uint64_t leases, uint32_t aliases, uint32_t allocation_slots );
int shadowspill_candidate_candidate_workspace_create( const ShadowSpillPressureFitProblem *problem, CandidateWorkspace *workspace );
void shadowspill_candidate_candidate_workspace_destroy(CandidateWorkspace *workspace);
int shadowspill_candidate_simulate_schedule( const ShadowSpillPressureFitProblem *problem, const ShadowSpillIndexedSchedule *schedule, SimulationWorkspace *workspace, ShadowSpillCandidateAdmissionWorkspace *admission_workspace, ShadowSpillCapacityViolation *first_violation, ShadowSpillSimulationResult *result, ShadowSpillStatus *admission_status, ShadowSpillAdmissionReplayResult *admission_result );
int shadowspill_candidate_place_plan( const ShadowSpillPressureFitProblem *problem, CandidateWorkspace *workspace, const ShadowSpillSimulationResult *simulation, uint64_t *required_bytes );
int shadowspill_candidate_record_fetch_constraint( CandidateWorkspace *workspace, ShadowSpillFetchTriggerConstraint incoming );

/* A layout that overran the pool, answered where the miss is: one fetch
 * whose destination overlaps the lease that set the extent is delayed a
 * task, recorded as a trigger constraint. Returns 1 when a delay was
 * recorded, 0 when no delay frees at least `shortfall_bytes` of overlap or
 * the delay is one the schedule already carries, -1 on failure. */
void shadowspill_candidate_trace_measurement(
    const CandidateWorkspace *workspace,
    const ShadowSpillSimulationResult *simulation,
    uint64_t required_bytes,
    uint64_t pool_bytes,
    uint32_t cuts
);

int shadowspill_candidate_move_for_layout(
    const ShadowSpillScheduleFacts *facts,
    CandidateWorkspace *workspace,
    const ShadowSpillSimulationResult *simulation,
    uint64_t shortfall_bytes
);
void shadowspill_candidate_residency_options( CandidateWorkspace *workspace, uint8_t strategy, ShadowSpillPressureFitResidencyOptions *options );
ShadowSpillStatus shadowspill_candidate_reduce_residency( const ShadowSpillPressureFitProblem *problem, CandidateWorkspace *workspace, const ShadowSpillPressureFitResidencyOptions *options, uint8_t strategy, uint8_t *resident, uint8_t *breaks, ShadowSpillPressureFitResidencyResult *result );
void shadowspill_candidate_copy_simulation_error( ShadowSpillPressureFitCandidateDiagnostic *diagnostic, const ShadowSpillSimulationResult *simulation );
void shadowspill_candidate_copy_analytic_error( ShadowSpillPressureFitCandidateDiagnostic *diagnostic, const ShadowSpillPressureFitResidencyResult *residency );
int shadowspill_candidate_add_repair_pressure( const ShadowSpillPressureFitProblem *problem, CandidateWorkspace *workspace, const ShadowSpillSimulationResult *failure, uint32_t escalation, uint64_t *asked_beyond_shortfall );
int shadowspill_candidate_simulation_failure_may_be_repairable( ShadowSpillStatus status );
int shadowspill_candidate_delay_admission_fetch( const ShadowSpillScheduleFacts *facts, const ShadowSpillAdmissionReplayResult *failure, ShadowSpillAdmissionAnnotation annotation, ShadowSpillScheduleStorage *schedule, ShadowSpillFetchTriggerConstraint *constraint );
int shadowspill_candidate_advance_admission_fetch( const ShadowSpillScheduleFacts *facts, ShadowSpillAdmissionAnnotation annotation, ShadowSpillScheduleStorage *schedule, ShadowSpillFetchTriggerConstraint *constraint );
int shadowspill_candidate_add_admission_repair_pressure( const ShadowSpillPressureFitProblem *problem, CandidateWorkspace *workspace, const ShadowSpillPressureFitResidencyOptions *options, const ShadowSpillAdmissionReplayResult *failure, ShadowSpillAdmissionAnnotation annotation, const ShadowSpillIndexedSchedule *schedule );
void shadowspill_candidate_copy_admission_error( const ShadowSpillPressureFitProblem *problem, const ShadowSpillIndexedSchedule *schedule, const ShadowSpillAdmissionReplayResult *failure, ShadowSpillAdmissionAnnotation annotation, ShadowSpillPressureFitCandidateDiagnostic *diagnostic );
int shadowspill_candidate_reduce_repaired_candidate( const ShadowSpillPressureFitProblem *problem, CandidateWorkspace *workspace, const ShadowSpillPressureFitResidencyOptions *options, uint8_t strategy, ShadowSpillPressureFitCandidateDiagnostic *diagnostic );
void shadowspill_candidate_initialize_diagnostic( ShadowSpillPressureFitCandidateDiagnostic *diagnostic, uint8_t strategy, uint8_t rule, uint8_t coalesced );
int shadowspill_candidate_may_repair_again( const ShadowSpillPressureFitOptions *candidate_options, const ShadowSpillPressureFitCandidateDiagnostic *diagnostic );
int shadowspill_candidate_evaluate_candidate( const ShadowSpillPressureFitProblem *problem, const ShadowSpillScheduleFacts *facts, const ShadowSpillPressureFitOptions *candidate_options, CandidateWorkspace *workspace, uint8_t strategy, uint8_t rule, uint8_t coalesced, ShadowSpillPressureFitCandidateDiagnostic *diagnostic );
int shadowspill_candidate_adopt_selected_schedule( ShadowSpillPressureFitResult *result, ShadowSpillScheduleStorage *selected );
int shadowspill_candidate_evaluate_incumbent(SearchWorker *worker, uint32_t index);
void *shadowspill_candidate_worker_main(void *argument);
uint32_t shadowspill_candidate_worker_count_for( const ShadowSpillPressureFitOptions *options, uint32_t tasks );
void shadowspill_candidate_program_search_destroy( ProgramSearch *search, SearchWorker *workers, uint32_t worker_count );

#endif
