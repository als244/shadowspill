#ifndef SHADOWSPILL_RUNTIME_PLAN_INTERNAL_H
#define SHADOWSPILL_RUNTIME_PLAN_INTERNAL_H

/*
 * A plan: its objects, its tasks, and the fixed layout it was admitted under.
 *
 * The plan owns the tables that map plan-scoped identifiers to runtime state,
 * and the placement each execution lease was certified at.
 */

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>

#include <shadowspill/runtime.h>

#include "../tasks/internal.h"
#include "../transfers/internal.h"

typedef struct ShadowSpillFixedRuntimeDependency {
    ShadowSpillFixedDependencyDescription description;
    struct ShadowSpillQueuedAction *predecessor_action;
} ShadowSpillFixedRuntimeDependency;

typedef struct ShadowSpillFixedLayoutState {
    uint64_t slice_offset;
    uint64_t slice_bytes;
    ShadowSpillFixedPlacementDescription *placements;
    uint64_t placement_count;
    ShadowSpillFixedRuntimeDependency *dependencies;
    uint64_t dependency_count;
    uint8_t active;
    uint8_t sealed;
} ShadowSpillFixedLayoutState;

typedef struct ShadowSpillObjectAcquisitionRecord
    ShadowSpillObjectAcquisitionRecord;

struct ShadowSpillObjectAcquisitionRecord {
    ShadowSpillPlan *plan_owner;
    ShadowSpillObject **objects;
    uint32_t object_count;
    ShadowSpillObject **unique_objects;
    uint32_t unique_object_count;
    uint32_t *object_unique_indices;
    uint32_t *unique_first_positions;
    struct ShadowSpillObjectAcquisitionRecord *ownership_next;
};

typedef struct ShadowSpillPlanObjectBinding {
    uint64_t plan_object_id;
    ShadowSpillObject *object;
    uint8_t consistency;
    struct ShadowSpillPlanObjectBinding *hash_next;
    struct ShadowSpillPlanObjectBinding *ownership_next;
} ShadowSpillPlanObjectBinding;

typedef struct ShadowSpillPlanObjectTable {
    pthread_rwlock_t lock;
    ShadowSpillPlanObjectBinding **by_id;
    ShadowSpillPlanObjectBinding *owned_head;
    uint64_t bucket_count;
    uint8_t lock_initialized;
} ShadowSpillPlanObjectTable;

struct ShadowSpillPlan {
    ShadowSpillRuntime *runtime;
    ShadowSpillMemoryPool *execution_pool;
    ShadowSpillMemoryPool *spill_pool;
    ShadowSpillRouteState *fetch_route;
    ShadowSpillRouteState *evict_route;
    ShadowSpillPlanObjectTable object_bindings;
    ShadowSpillTaskTable tasks;
    ShadowSpillObjectAcquisitionRecord *object_acquisitions;
    ShadowSpillFixedLayoutState fixed_layout;
    pthread_mutex_t lifecycle_lock;
    _Atomic uint32_t active_task_scopes;
    _Atomic uint64_t pending_actions;
    _Atomic uint64_t pending_retirements;
    _Atomic uint8_t closing;
    _Atomic uint8_t closed;
    uint8_t lifecycle_lock_initialized;
    uint8_t object_bindings_initialized;
    uint8_t tasks_initialized;
    struct ShadowSpillPlan *ownership_next;
    struct ShadowSpillPlan **ownership_previous_link;
};

int shadowspill_plan_object_table_initialize(
    ShadowSpillPlanObjectTable *table,
    uint64_t bucket_count
);

void shadowspill_plan_object_table_destroy(ShadowSpillPlanObjectTable *table);

void shadowspill_plan_object_table_clear(ShadowSpillPlanObjectTable *table);

ShadowSpillObject *shadowspill_plan_object_acquire(
    ShadowSpillPlan *plan,
    uint64_t plan_object_id,
    uint8_t *consistency
);

ShadowSpillRuntimeStatus shadowspill_fixed_layout_reserve_slice(
    ShadowSpillPlan *plan,
    uint64_t bytes
);

ShadowSpillRuntimeStatus shadowspill_fixed_layout_clear(
    ShadowSpillPlan *plan
);

void shadowspill_fixed_layout_destroy(ShadowSpillPlan *plan);

const ShadowSpillFixedPlacementDescription *
shadowspill_fixed_layout_find_placement(
    const ShadowSpillPlan *plan,
    uint8_t kind,
    uint64_t task_id,
    uint64_t ordinal,
    uint64_t object_id
);

ShadowSpillRuntimeStatus shadowspill_fixed_layout_adopt_execution_lease_locked(
    ShadowSpillPlan *plan,
    ShadowSpillMemoryLease *lease,
    uint64_t relative_offset,
    uint64_t bytes,
    uint64_t alignment
);

int shadowspill_fixed_layout_dependencies_published(
    ShadowSpillPlan *plan,
    uint8_t successor_kind,
    uint64_t task_id,
    uint64_t ordinal,
    uint64_t invocation
);

ShadowSpillRuntimeStatus shadowspill_fixed_layout_insert_dependency_waits(
    ShadowSpillPlan *plan,
    uint8_t successor_kind,
    uint64_t task_id,
    uint64_t ordinal,
    uint64_t invocation,
    ShadowSpillBackendStream stream
);

ShadowSpillRuntimeStatus shadowspill_fixed_layout_wait_for_dependencies(
    ShadowSpillPlan *plan,
    uint8_t successor_kind,
    uint64_t task_id,
    uint64_t ordinal,
    uint64_t invocation,
    ShadowSpillBackendStream stream
);

void shadowspill_plan_destroy_all(ShadowSpillRuntime *runtime);

#endif
