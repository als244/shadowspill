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

#define SHADOWSPILL_ADMISSION_NO_DEPENDENCY UINT64_MAX
#define SHADOWSPILL_ADMISSION_NO_OPERATION UINT64_MAX
#define SHADOWSPILL_ADMISSION_NO_LEASE UINT64_MAX

/* Planning statuses are in the shared vocabulary; see <shadowspill/status.h>. */

/*
 * A schedule in indexed form: every identifier is the contiguous task or
 * alias index in the simulation program it was placed over. A search returns
 * these inside its own result, which owns the arrays and says when they are
 * released.
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

/* The part of a planning problem that is not about how it is searched.
 *
 * Certifying a schedule, digesting it, and replaying it through the pool
 * are the same questions whichever search produced the schedule: they need
 * the machine it runs on, the topology it must fit, and the names its
 * aliases and tasks are written under. Every search embeds one of these, so
 * generic code takes a pointer to it and never sees the search's own input.
 */
typedef struct ShadowSpillScheduleContext {
    const ShadowSpillSimulationProgram *simulation;
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
} ShadowSpillScheduleContext;

/*
 * Schedule-invariant input to a search: the problem before any schedule
 * exists for it. A search resolves this into whatever it evaluates.
 */
typedef struct ShadowSpillIndexedProblem {
    uint32_t abi_version;
    /* What any code judging this problem's schedules needs. */
    ShadowSpillScheduleContext context;
    /* A rank per device, which breaks ties between devices so an answer does
       not depend on the order the devices arrived in. */
    const uint32_t *device_priority;

    /* The plan to beat: a plan for this problem already in hand -- found at
       a smaller capacity, say -- or NULL. A search given one measures it
       before its own candidates and never answers worse than it. Its aliases
       and tasks index this problem. */
    const ShadowSpillIndexedSchedule *incumbent;
} ShadowSpillIndexedProblem;

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

SHADOWSPILL_API ShadowSpillStatus
shadowspill_evaluate_schedule_admission(
    const ShadowSpillSimulationProgram *simulation,
    const ShadowSpillAdmissionFacts *admission,
    const ShadowSpillIndexedSchedule *schedule,
    ShadowSpillScheduleAdmissionResult *result
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

/* Which structure shadowspill_planner_struct_size() is asked about. A search
 * that ships its own structures continues this numbering in its own header, so
 * one call answers for the generic planner and for the search. */
enum ShadowSpillPlannerStruct {
    SHADOWSPILL_STRUCT_ADMISSION_FACTS = 0,
};

SHADOWSPILL_API ShadowSpillStatus shadowspill_place_lifetimes(
    const ShadowSpillPlacementProblem *problem,
    ShadowSpillPlacementResult *result
);

#ifdef __cplusplus
}
#endif

#endif
