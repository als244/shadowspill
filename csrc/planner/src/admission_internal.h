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

typedef struct ShadowSpillPendingRetirement {
    uint64_t lease_id;
    uint64_t dependency_id;
    uint32_t completion_index;
    uint8_t completion_boundary;
} ShadowSpillPendingRetirement;

typedef struct ShadowSpillCandidateAdmissionWorkspace {
    ShadowSpillAdmissionReplayOperation *operations;
    ShadowSpillAdmissionReplayDecision *decisions;
    ShadowSpillAdmissionReuseDependency *dependencies;
    ShadowSpillAdmissionReplayLiveLease *live_leases;
    ShadowSpillAdmissionAnnotation *annotations;
    uint64_t operation_capacity;
    uint64_t lease_capacity;
    uint64_t dependency_capacity;
    uint64_t reuse_dependency_capacity;

    uint64_t *active_alias_leases;
    uint64_t *new_alias_leases;
    uint64_t *task_allocation_leases;
    uint8_t *task_allocation_live;
    uint32_t *lease_aliases;
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

/* Running totals while `operations.c` derives a schedule's operations.
 * `candidate.c` owns one so it can replay and project what was produced. */
typedef struct ScriptState {
    ShadowSpillCandidateAdmissionWorkspace *workspace;
    uint64_t operation_count;
    uint64_t lease_count;
    uint64_t dependency_count;
    uint64_t pending_count;
} ScriptState;

/* topology.c */
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
    ScriptState *state
);


#endif
