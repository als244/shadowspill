#ifndef SHADOWSPILL_RUNTIME_OBJECTS_INTERNAL_H
#define SHADOWSPILL_RUNTIME_OBJECTS_INTERNAL_H

/*
 * Logical objects and the actions queued against them.
 *
 * An object is the unit a plan names. Its bytes may sit in more than one pool
 * at once, so it holds one location per pool and one authoritative version.
 * A queued action is a pending fetch, eviction or release against an object.
 */

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>

#include <shadowspill/runtime.h>

#include "../memory/internal.h"

/* Objects and transfers refer to each other: a queued action names the route
 * that will carry it, and a lane holds the actions waiting on that route. */
typedef struct ShadowSpillTaskRecord ShadowSpillTaskRecord;

typedef struct ShadowSpillTaskRecord ShadowSpillTaskRecord;

typedef struct ShadowSpillQueuedAction ShadowSpillQueuedAction;

typedef struct ShadowSpillRouteState ShadowSpillRouteState;

struct ShadowSpillObject {
    ShadowSpillRuntime *runtime;
    uint64_t object_id;
    uint64_t size_bytes;
    _Atomic uint32_t references;
    _Atomic uint32_t owners;
    _Atomic uint8_t registration_owned;
    _Atomic uint8_t detached;
    pthread_mutex_t lock;
    ShadowSpillObjectLocation *locations;
    uint32_t location_count;
    uint64_t generation;
    uint64_t authoritative_version;
    uint64_t allocation_id;
    uint8_t retain_spill_copy;
    uint8_t residency;
    /*
     * Number of queued fetch generations that have not yet published a
     * readiness event.  This must be a count rather than a Boolean: the
     * dispatcher can enqueue a later release/fetch cycle while an earlier fetch
     * for the same object is still in flight.
     */
    _Atomic uint32_t unpublished_fetch_count;
    ShadowSpillEventLease *readiness_event;
    uint8_t has_readiness_event;
    uint64_t retired_generation;
    void *retired_execution_pointer;
    ShadowSpillQueuedAction *action_head;
    ShadowSpillQueuedAction *action_tail;
    struct ShadowSpillObject *ownership_next;
    struct ShadowSpillObject **ownership_previous_link;
    struct ShadowSpillObject *id_index_next;
};

struct ShadowSpillObjectHandle {
    ShadowSpillRuntime *runtime;
    ShadowSpillObject *object;
};

typedef struct ShadowSpillObjectTable {
    pthread_rwlock_t lock;
    uint8_t lock_initialized;
    ShadowSpillObject *owned_head;
    ShadowSpillObject **by_id;
    uint64_t bucket_count;
} ShadowSpillObjectTable;

typedef enum ShadowSpillQueuedActionState {
    SHADOWSPILL_ACTION_QUEUED = 0,
    SHADOWSPILL_ACTION_IN_FLIGHT = 1,
    SHADOWSPILL_ACTION_FINISHED = 2,
} ShadowSpillQueuedActionState;

struct ShadowSpillQueuedAction {
    uint64_t task_id;
    uint64_t plan_object_id;
    uint64_t action_ordinal;
    uint64_t activation_generation;
    uint64_t completed_generation;
    uint8_t kind;
    uint8_t state;
    ShadowSpillMemoryLease *destination_lease;
    ShadowSpillObject *object;
    ShadowSpillPlan *plan_owner;
    ShadowSpillRouteState *route;
    ShadowSpillEventLease *trigger_event;
    ShadowSpillEventLease *completion_event;
    ShadowSpillEventLease *dependency_event;
    /* The copy's interval on its lane, open only while a trace measures it. */
    ShadowSpillStreamInterval stream_interval;
    const char *trace_label;
    uint8_t owns_trace_label;
    uint8_t has_completion_event;
    uint8_t processing;
    uint8_t admitted;
    uint8_t active;
    uint8_t produces_current_execution;
    uint8_t produces_current_spill;
    /*
     * A zero-copy publication may replace this action's logical source with
     * another object while preserving the same physical lease. Admission
     * resolves the release action directly; publication records the exact
     * lease generation here without building an object-ID handoff chain.
     */
    ShadowSpillMemoryLease *handoff_lease;
    uint64_t handoff_generation;
    /*
     * A final fetch may be handed to the caller after its readiness event is
     * published but before the worker observes completion.  The caller owns
     * the execution lease immediately; this retained metadata lets the worker
     * finish source cleanup without keeping the logical object resident.
     */
    ShadowSpillMemoryLease *caller_handoff_lease;
    uint64_t caller_handoff_generation;
    uint64_t scheduled_version;
    struct ShadowSpillQueuedAction *previous;
    struct ShadowSpillQueuedAction *next;
    struct ShadowSpillQueuedAction *object_previous;
    struct ShadowSpillQueuedAction *object_next;
    struct ShadowSpillQueuedAction *lane_previous;
    struct ShadowSpillQueuedAction *lane_next;
    uint8_t lane_state;
};

typedef struct ShadowSpillActionQueue {
    pthread_mutex_t lock;
    ShadowSpillQueuedAction *head;
    ShadowSpillQueuedAction *tail;
    _Atomic uint64_t count;
    uint8_t lock_initialized;
} ShadowSpillActionQueue;

ShadowSpillObject *shadowspill_find_object(
    ShadowSpillRuntime *runtime,
    uint64_t object_id
);

int shadowspill_object_table_initialize(
    ShadowSpillObjectTable *table,
    uint64_t bucket_count
);

void shadowspill_object_table_destroy(ShadowSpillObjectTable *table);

ShadowSpillObject *shadowspill_object_table_find(
    const ShadowSpillObjectTable *table,
    uint64_t object_id
);

ShadowSpillObject *shadowspill_object_table_acquire(
    ShadowSpillObjectTable *table,
    uint64_t object_id
);

int shadowspill_object_table_insert(
    ShadowSpillObjectTable *table,
    ShadowSpillObject *object
);

int shadowspill_object_table_remove(
    ShadowSpillObjectTable *table,
    ShadowSpillObject *object
);

int shadowspill_object_table_rekey(
    ShadowSpillObjectTable *table,
    ShadowSpillObject *object,
    uint64_t replacement_object_id
);

void shadowspill_object_retain(ShadowSpillObject *object);

void shadowspill_object_release(ShadowSpillObject *object);

ShadowSpillStatus shadowspill_object_owner_retain(
    ShadowSpillObject *object
);

ShadowSpillStatus shadowspill_object_owner_release(
    ShadowSpillObject *object
);

ShadowSpillStatus shadowspill_object_schedule_action_locked(
    ShadowSpillObject *object,
    ShadowSpillQueuedAction *action
);

int shadowspill_object_action_is_head_locked(
    const ShadowSpillObject *object,
    const ShadowSpillQueuedAction *action
);

int shadowspill_object_reset_admitted_action_locked(
    ShadowSpillObject *object,
    ShadowSpillQueuedAction *action
);

void shadowspill_object_note_fetch_queued_locked(ShadowSpillObject *object);

int shadowspill_object_note_fetch_published_locked(ShadowSpillObject *object);

int shadowspill_object_note_fetch_discarded_locked(ShadowSpillObject *object);

int shadowspill_object_has_unpublished_fetch_locked(
    const ShadowSpillObject *object
);

int shadowspill_object_remove_action_locked(
    ShadowSpillObject *object,
    ShadowSpillQueuedAction *action
);

void shadowspill_object_acquisitions_clear(ShadowSpillPlan *plan);

ShadowSpillStatus shadowspill_acquire_object_bindings(
    ShadowSpillRuntime *runtime,
    const ShadowSpillPlan *plan,
    uint64_t trace_task_id,
    ShadowSpillObject *const *unique_objects,
    uint32_t unique_object_count,
    const uint32_t *object_unique_indices,
    const uint32_t *unique_first_positions,
    uint32_t object_count,
    ShadowSpillBackendStream consumer_stream,
    ShadowSpillObjectBinding *bindings,
    uint32_t binding_capacity
);

ShadowSpillStatus shadowspill_object_transfer_to_caller(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryPool *execution_pool,
    ShadowSpillMemoryPool *spill_pool,
    ShadowSpillObject *object,
    ShadowSpillBackendStream consumer_stream,
    const void *expected_pointer,
    uint64_t expected_generation,
    uint64_t expected_allocation_id,
    uint8_t validate_expected,
    ShadowSpillAllocation *allocation
);

ShadowSpillStatus shadowspill_object_bind_allocation(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryPool *pool,
    ShadowSpillObject *object,
    const void *pointer,
    const ShadowSpillTaskRecord *task,
    ShadowSpillObjectBinding *binding
);

ShadowSpillStatus shadowspill_object_replace_allocation(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryPool *pool,
    ShadowSpillObject *object,
    const void *pointer,
    ShadowSpillObjectBinding *binding
);

#endif
