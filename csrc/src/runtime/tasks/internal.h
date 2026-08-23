#ifndef SHADOWSPILL_RUNTIME_TASKS_INTERNAL_H
#define SHADOWSPILL_RUNTIME_TASKS_INTERNAL_H

/*
 * Task records and the dispatcher-thread task scope.
 *
 * A record is the runtime's copy of one task's plan: its inputs, the objects
 * it publishes, the actions it triggers, and the allocation contract it must
 * honour. The scope is thread-local state naming the task a dispatcher thread
 * is currently inside, which is how an allocation finds its origin.
 */

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>

#include <shadowspill/runtime.h>

#include "../objects/internal.h"

typedef struct ShadowSpillTaskUpdate {
    ShadowSpillObject *object;
    uint64_t plan_object_id;
    uint64_t version_delta;
} ShadowSpillTaskUpdate;

typedef struct ShadowSpillTaskPublication {
    ShadowSpillObject *object;
    uint64_t plan_object_id;
    uint8_t kind;
} ShadowSpillTaskPublication;

typedef struct ShadowSpillTaskAction {
    ShadowSpillObject *object;
    uint64_t plan_object_id;
    uint8_t kind;
    char *trace_label;
} ShadowSpillTaskAction;

typedef struct ShadowSpillTaskReleaseBinding {
    ShadowSpillObject *object;
    ShadowSpillQueuedAction *action;
} ShadowSpillTaskReleaseBinding;

struct ShadowSpillTaskRecord {
    ShadowSpillPlan *plan_owner;
    uint64_t task_id;
    char *trace_label;
    ShadowSpillObject **inputs;
    uint64_t *input_plan_object_ids;
    uint8_t *input_consistency;
    uint32_t input_count;
    ShadowSpillObject **unique_inputs;
    uint32_t unique_input_count;
    uint32_t *input_unique_indices;
    uint32_t *unique_first_positions;
    ShadowSpillObjectBinding *input_bindings;
    ShadowSpillTaskUpdate *updates;
    uint32_t update_count;
    ShadowSpillTaskPublication *publications;
    uint32_t publication_count;
    ShadowSpillTaskAction *actions;
    ShadowSpillQueuedAction *queued_actions;
    uint32_t action_count;
    ShadowSpillTaskReleaseBinding *release_bindings;
    uint32_t release_binding_count;
    ShadowSpillTaskAllocationContractStep *allocation_contract_steps;
    uint8_t *allocation_contract_states;
    uint32_t allocation_contract_step_count;
    uint32_t allocation_contract_allocation_count;
    uint8_t enforce_allocation_contract;
    uint64_t maximum_requested_allocation_bytes;
    uint64_t maximum_charged_allocation_bytes;
    uint64_t live_requested_allocation_limit_bytes;
    uint64_t live_charged_allocation_limit_bytes;
    uint64_t dynamic_scratch_maximum_allocation_bytes;
    uint64_t dynamic_scratch_live_limit_bytes;
    _Atomic uint64_t invocation_count;
    _Atomic uint64_t submission_sequence;
    _Atomic uint64_t submission_invocation;
    _Atomic uint64_t acknowledgement_sequence;
    _Atomic uint8_t invocation_active;
    uint8_t boundary_kind;
    struct ShadowSpillTaskRecord *hash_next;
    struct ShadowSpillTaskRecord *ownership_next;
};

enum {
    SHADOWSPILL_BOUNDARY_TASK = 0U,
    SHADOWSPILL_BOUNDARY_ACTION_BATCH = 1U,
};

typedef struct ShadowSpillTaskTable {
    pthread_rwlock_t lock;
    ShadowSpillTaskRecord **by_id;
    ShadowSpillTaskRecord *owned_head;
    uint64_t bucket_count;
    uint8_t lock_initialized;
} ShadowSpillTaskTable;

void shadowspill_abort_current_task(ShadowSpillRuntime *runtime);

uint64_t shadowspill_current_task_id(ShadowSpillRuntime *runtime);

uint64_t shadowspill_current_task_allocation_ordinal(
    ShadowSpillRuntime *runtime
);

uint64_t shadowspill_current_task_core_allocation_ordinal(
    ShadowSpillRuntime *runtime
);

int shadowspill_current_task_allocation_is_scratch(
    ShadowSpillRuntime *runtime
);

uint64_t shadowspill_current_task_invocation(ShadowSpillRuntime *runtime);

ShadowSpillPlan *shadowspill_current_plan(ShadowSpillRuntime *runtime);

ShadowSpillMemoryPool *shadowspill_current_allocation_pool(
    ShadowSpillRuntime *runtime
);

int shadowspill_track_task_retirement(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryLease *lease
);

ShadowSpillMemoryLease *shadowspill_current_task_retirements(
    ShadowSpillRuntime *runtime
);

int shadowspill_enter_allocation_scope(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryPool *pool,
    uint64_t task_id
);

int shadowspill_enter_task_scope(
    ShadowSpillRuntime *runtime,
    const ShadowSpillTaskRecord *record
);

int shadowspill_claim_task_invocation(
    const ShadowSpillTaskRecord *record
);

int shadowspill_enter_claimed_task_scope(
    ShadowSpillRuntime *runtime,
    const ShadowSpillTaskRecord *record
);

void shadowspill_release_task_invocation(
    const ShadowSpillTaskRecord *record
);

ShadowSpillStatus shadowspill_validate_task_allocation(
    ShadowSpillRuntime *runtime,
    uint64_t requested_bytes,
    uint64_t charged_bytes,
    uint64_t alignment_bytes
);

uint64_t shadowspill_commit_task_allocation(
    ShadowSpillRuntime *runtime,
    uint64_t requested_bytes,
    uint64_t charged_bytes
);

ShadowSpillStatus shadowspill_release_task_allocation(
    ShadowSpillRuntime *runtime,
    uint64_t origin_task_id,
    uint64_t origin_task_invocation,
    uint64_t allocation_ordinal,
    int allocation_is_scratch,
    uint64_t requested_bytes,
    uint64_t charged_bytes,
    uint64_t alignment_bytes
);

ShadowSpillStatus shadowspill_validate_task_allocation_complete(
    ShadowSpillRuntime *runtime
);

void shadowspill_leave_task_scope(ShadowSpillRuntime *runtime);

int shadowspill_task_table_initialize(
    ShadowSpillTaskTable *table,
    uint64_t bucket_count
);

void shadowspill_task_table_destroy(ShadowSpillTaskTable *table);

void shadowspill_task_table_clear(ShadowSpillTaskTable *table);

ShadowSpillTaskRecord *shadowspill_task_table_acquire(
    ShadowSpillTaskTable *table,
    uint64_t task_id
);

ShadowSpillQueuedAction *shadowspill_task_release_action(
    const ShadowSpillTaskRecord *record,
    const ShadowSpillObject *object
);

void shadowspill_task_clear_pending_handoffs(
    const ShadowSpillTaskRecord *record
);

char *shadowspill_copy_action_trace_label(
    const ShadowSpillRuntimeAction *action,
    uint64_t task_id,
    uint64_t size_bytes
);

ShadowSpillStatus shadowspill_after_task_record(
    ShadowSpillRuntime *runtime,
    const ShadowSpillTaskRecord *record,
    ShadowSpillBackendStream compute_stream
);

#endif
