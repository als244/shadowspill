#ifndef SHADOWSPILL_PLANNER_ADMISSION_INTERNAL_H
#define SHADOWSPILL_PLANNER_ADMISSION_INTERNAL_H

#include <stdint.h>

#include <shadowspill/admission_replay.h>
#include <shadowspill/planner.h>

typedef enum ShadowSpillAdmissionBoundaryKind {
    SHADOWSPILL_ADMISSION_BOUNDARY_INITIAL = 0,
    SHADOWSPILL_ADMISSION_BOUNDARY_TASK_START = 1,
    SHADOWSPILL_ADMISSION_BOUNDARY_TASK_COMPLETION = 2,
    SHADOWSPILL_ADMISSION_BOUNDARY_ACTION_TRIGGER = 3,
    SHADOWSPILL_ADMISSION_BOUNDARY_ACTION_COMPLETION = 4,
} ShadowSpillAdmissionBoundaryKind;

typedef struct ShadowSpillAdmissionAnnotation {
    uint32_t index;
    uint8_t boundary;
} ShadowSpillAdmissionAnnotation;

/* Why a lease exists. The boundary above says where an operation sits in the
 * step; this says what it is for, which is what lifetime construction and the
 * fixed/dynamic split need. Values match
 * `pytorch.planning.admission.AdmissionReplayPurpose`. */
typedef enum ShadowSpillAdmissionPurpose {
    SHADOWSPILL_ADMISSION_PURPOSE_INITIAL_OBJECT = 0,
    SHADOWSPILL_ADMISSION_PURPOSE_TASK_WORKSPACE = 1,
    SHADOWSPILL_ADMISSION_PURPOSE_TASK_OUTPUT = 2,
    SHADOWSPILL_ADMISSION_PURPOSE_MUTATION_REPLACEMENT = 3,
    SHADOWSPILL_ADMISSION_PURPOSE_RELEASE = 4,
    SHADOWSPILL_ADMISSION_PURPOSE_EVICTION = 5,
    SHADOWSPILL_ADMISSION_PURPOSE_FETCH_DESTINATION = 6,
    SHADOWSPILL_ADMISSION_PURPOSE_TERMINAL_COMPLETION = 7,
} ShadowSpillAdmissionPurpose;

typedef struct ShadowSpillPendingRetirement {
    uint64_t lease_id;
    uint64_t dependency_id;
    uint32_t completion_index;
    uint8_t completion_boundary;
    uint8_t completion_purpose;
} ShadowSpillPendingRetirement;

typedef struct ShadowSpillCandidateAdmissionWorkspace {
    ShadowSpillAdmissionReplayOperation *operations;
    ShadowSpillAdmissionReplayDecision *decisions;
    ShadowSpillAdmissionReuseDependency *dependencies;
    ShadowSpillAdmissionReplayLiveLease *live_leases;
    ShadowSpillAdmissionAnnotation *annotations;
    uint8_t *purposes;
    uint32_t *allocation_offsets;
    uint64_t operation_capacity;
    uint64_t lease_capacity;
    uint64_t dependency_capacity;
    uint64_t reuse_dependency_capacity;

    uint64_t *active_alias_leases;
    uint64_t *new_alias_leases;
    uint64_t *task_allocation_leases;
    uint8_t *task_allocation_live;
    uint32_t *lease_aliases;
    uint64_t *lease_start_operations;
    uint64_t *lease_retire_operations;
    uint64_t *repair_candidate_starts;
    uint64_t *repair_blocked_prefix;
    uint32_t *repair_unremovable_prefix;
    uint32_t *predecessor_actions;
    uint32_t *predecessor_tasks;
    uint8_t *handoff_sources;
    ShadowSpillPendingRetirement *pending_retirements;

    int64_t *task_start_deltas;
    int64_t *task_completion_deltas;
    int64_t *action_trigger_deltas;
    int64_t *action_completion_deltas;
    uint32_t *reuse_predecessor_actions;
    uint32_t *reuse_successor_tasks;
    uint32_t *reuse_successor_actions;
    uint32_t action_capacity;
    uint32_t projected_reuse_count;
    ShadowSpillSimulationDevice physical_device;

    ShadowSpillAdmissionReplayWorkspace *replay;
    uint64_t initial_physical_bytes;
    uint64_t decision_digest;
    uint64_t peak_allocated_bytes;
    uint64_t peak_reserved_bytes;
    uint64_t peak_fragmentation_bytes;
    uint64_t calls;
    uint64_t time_ns;
} ShadowSpillCandidateAdmissionWorkspace;

int shadowspill_candidate_admission_workspace_create(
    const ShadowSpillPressureFitContext *context,
    ShadowSpillCandidateAdmissionWorkspace *workspace
);

void shadowspill_candidate_admission_workspace_destroy(
    ShadowSpillCandidateAdmissionWorkspace *workspace
);

ShadowSpillAdmissionReplayStatus shadowspill_admit_indexed_schedule(
    const ShadowSpillPressureFitContext *context,
    const ShadowSpillIndexedSchedule *schedule,
    ShadowSpillCandidateAdmissionWorkspace *workspace,
    ShadowSpillSimulationProgram *program,
    ShadowSpillAdmissionReplayResult *replay_result
);

/* A running count of everything one schedule has produced so far, and the
 * workspace it is being written into. Each count is also the index the next
 * entry takes, so the tally is both the total and the cursor. `operations.c`
 * fills one; `candidate.c` owns it so it can replay and project the result. */
typedef struct OperationTally {
    ShadowSpillCandidateAdmissionWorkspace *workspace;
    uint64_t operation_count;
    uint64_t lease_count;
    uint64_t dependency_count;
    uint64_t pending_count;
    /* Bytes each transfer lane must move. A schedule cannot finish sooner
     * than its busiest lane, so these bound the makespan without simulating. */
    uint64_t fetch_bytes;
    uint64_t evict_bytes;
} OperationTally;

/* topology.c */
int shadowspill_admission_counts(
    const ShadowSpillPressureFitContext *context,
    const ShadowSpillIndexedSchedule *schedule,
    uint64_t *lease_count,
    uint64_t *operation_count
);
int shadowspill_admission_topology_valid(
    const ShadowSpillPressureFitContext *context
);
int shadowspill_admission_reserve_buffers(
    const ShadowSpillPressureFitContext *context,
    const ShadowSpillIndexedSchedule *schedule,
    ShadowSpillCandidateAdmissionWorkspace *workspace
);

/* operations.c */
int shadowspill_admission_build_operations(
    const ShadowSpillPressureFitContext *context,
    const ShadowSpillIndexedSchedule *schedule,
    ShadowSpillCandidateAdmissionWorkspace *workspace,
    OperationTally *tally
);


#endif
