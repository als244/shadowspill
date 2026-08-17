#ifndef SHADOWSPILL_RUNTIME_INTERNAL_H
#define SHADOWSPILL_RUNTIME_INTERNAL_H

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <pthread.h>
#include <stddef.h>
#include <stdatomic.h>
#include <stdint.h>

#include <shadowspill/runtime.h>

#include "internal/failure_state.h"
#include "internal/event_pool.h"
#include "internal/completion_tracker.h"

typedef struct ShadowSpillRange {
    uint64_t offset;
    uint64_t bytes;
    struct ShadowSpillRange *next;
} ShadowSpillRange;

typedef struct ShadowSpillRangeAllocator {
    uint64_t capacity;
    uint64_t allocated;
    uint64_t peak_allocated;
    ShadowSpillRange *free_ranges;
    /*
     * Production pools leave these fields null and allocate range nodes from
     * the process heap. Admission workspaces supply a bounded borrowed arena
     * so repeated candidate evaluation executes the identical range policy
     * without per-operation heap traffic.
     */
    ShadowSpillRange *node_storage;
    uint64_t node_capacity;
    ShadowSpillRange *available_nodes;
} ShadowSpillRangeAllocator;

typedef struct ShadowSpillStreamRecord {
    ShadowSpillBackendStream stream;
    struct ShadowSpillStreamRecord *next;
} ShadowSpillStreamRecord;

typedef struct ShadowSpillObject ShadowSpillObject;
typedef struct ShadowSpillTaskRecord ShadowSpillTaskRecord;
typedef struct ShadowSpillEventPool ShadowSpillEventPool;

struct ShadowSpillEventLease {
    ShadowSpillBackendEvent event;
    uint64_t generation;
    uint64_t completion_object_id;
    uint64_t completion_allocation_id;
    _Atomic uint32_t references;
    /*
     * The backend has reported that the event completed. This is deliberately
     * not a memory-availability flag: a range becomes reusable only when its
     * owning MemoryPool commits the matching lease transition under pool.lock.
     */
    _Atomic uint8_t backend_complete;
    struct ShadowSpillEventLease *completion_next;
    struct ShadowSpillEventLease *free_next;
    uint8_t pool_owned;
    uint8_t completion_linked;
};

typedef struct ShadowSpillEventPoolBlock {
    ShadowSpillEventLease *leases;
    uint64_t count;
    struct ShadowSpillEventPoolBlock *next;
} ShadowSpillEventPoolBlock;

/*
 * Owns reusable neutral event records. Backend event handles are still leased
 * through the synchronization backend, whose implementation may maintain its
 * own driver-event pool. Explicit cold-path reserve calls may grow this owner;
 * once reserved, a hot-path shortage fails instead of allocating from libc.
 */
struct ShadowSpillEventPool {
    pthread_mutex_t lock;
    ShadowSpillEventPoolBlock *blocks;
    ShadowSpillEventLease *free_head;
    uint64_t capacity;
    uint64_t available;
    uint64_t in_use;
    uint64_t peak_in_use;
    uint64_t growth_rejections;
    uint8_t initialized;
    uint8_t sealed;
};

typedef struct ShadowSpillEventRecord {
    ShadowSpillEventLease *event;
    struct ShadowSpillEventRecord *next;
} ShadowSpillEventRecord;

typedef struct ShadowSpillCompletionStream {
    ShadowSpillBackendStream stream;
    ShadowSpillEventLease *head;
    ShadowSpillEventLease *tail;
    uint64_t next_poll_timestamp_ns;
    struct ShadowSpillCompletionStream *next;
} ShadowSpillCompletionStream;

typedef struct ShadowSpillCompletionTracker {
    pthread_mutex_t lock;
    ShadowSpillCompletionStream *streams;
    uint64_t pending;
} ShadowSpillCompletionTracker;

typedef enum ShadowSpillMemoryPlacement {
    SHADOWSPILL_MEMORY_FIRST_FIT = 0,
    SHADOWSPILL_MEMORY_BEST_FIT_LOW = 1,
    SHADOWSPILL_MEMORY_BEST_FIT_HIGH = 2,
} ShadowSpillMemoryPlacement;

enum {
    SHADOWSPILL_EXECUTION_POOL_ID = 0U,
    SHADOWSPILL_SPILL_POOL_ID = 1U,
    SHADOWSPILL_FETCH_ROUTE_ID = 0U,
    SHADOWSPILL_EVICT_ROUTE_ID = 1U,
    SHADOWSPILL_TRANSFER_FETCH = 0U,
    SHADOWSPILL_TRANSFER_EVICT = 1U,
};

/*
 * Owns one bounded arena and its suballocation geometry. A pool does not know
 * whether its storage is local, remote, accelerator, host, or persistent.
 * Concrete backends attach that meaning when they instantiate the pool.
 */
typedef struct ShadowSpillMemoryPool {
    pthread_mutex_t lock;
    _Atomic uint64_t foreground_waiters;
    _Atomic uint64_t reservation_waiters;
    _Atomic uint64_t capacity_epoch;
    ShadowSpillRangeAllocator ranges;
    /* Physical range owners currently registered with the range allocator. */
    struct ShadowSpillMemoryLease *range_leases;
    /* Complete lease metadata owned by this pool until runtime teardown. */
    struct ShadowSpillMemoryLease *owned_leases;
    /* Reusable metadata records; these own no physical range while linked. */
    struct ShadowSpillMemoryLease *free_lease_records;
    struct ShadowSpillMemoryLease *active_leases;
    struct ShadowSpillMemoryLease **leases_by_id;
    struct ShadowSpillMemoryLease **leases_by_pointer;
    struct ShadowSpillMemoryLease **reusable_leases_by_size;
    ShadowSpillMemoryPoolBackend backend;
    void *base;
    uint32_t pool_id;
    uint64_t minimum_alignment;
    uint64_t next_request_sequence;
    uint64_t next_release_sequence;
    uint64_t reserved_bytes;
    uint64_t allocation_index_bucket_count;
    uint64_t reusable_index_bucket_count;
    uint64_t requested_allocated_bytes;
    uint64_t peak_requested_allocated_bytes;
    uint64_t live_allocations;
    uint64_t blocked_allocators;
    uint64_t lease_record_capacity;
    uint64_t lease_record_available;
    uint64_t lease_record_in_use;
    uint64_t lease_record_peak_in_use;
    uint64_t lease_record_growth_rejections;
    _Atomic uint64_t pending_retirements;
    _Atomic uint64_t pending_capacity_actions;
    _Atomic uint64_t free_bytes_snapshot;
    _Atomic uint64_t largest_free_bytes_snapshot;
    uint8_t initialized;
    uint8_t lease_records_sealed;
} ShadowSpillMemoryPool;

typedef enum ShadowSpillMemoryLeaseState {
    SHADOWSPILL_LEASE_FREE = 0,
    SHADOWSPILL_LEASE_IN_USE = 1,
    SHADOWSPILL_LEASE_RETIRE_PENDING = 2,
    SHADOWSPILL_LEASE_RESERVED = 3,
    SHADOWSPILL_LEASE_SUCCESSOR_RESERVED = 4,
    SHADOWSPILL_LEASE_PREDECESSOR_TRANSFERRED = 5,
} ShadowSpillMemoryLeaseState;

typedef struct ShadowSpillMemoryLease {
    /* Stable metadata owner; unlike ``pool``, this survives physical release. */
    ShadowSpillMemoryPool *metadata_owner;
    ShadowSpillMemoryPool *pool;
    ShadowSpillMemoryLeaseState state;
    uint64_t allocation_id;
    uint64_t generation;
    uint64_t requested_bytes;
    uint64_t charged_bytes;
    uint64_t alignment_bytes;
    uint64_t offset;
    uint64_t origin_task_id;
    uint64_t origin_task_invocation;
    uint64_t origin_task_allocation_sequence;
    uint64_t origin_task_allocation_ordinal;
    uint8_t origin_task_allocation_is_scratch;
    uint64_t release_task_id;
    uint64_t request_sequence;
    uint64_t release_sequence;
    void *pointer;
    void *retired_pointer;
    _Atomic uint32_t references;
    int logical_freed;
    uint8_t retirement_preparing;
    int plan_owned;
    int ever_plan_owned;
    int framework_free_seen;
    ShadowSpillObject *bound_object;
    ShadowSpillStreamRecord *streams;
    ShadowSpillEventRecord *retirement_events;
    ShadowSpillEventLease *retirement_event;
    uint64_t retirement_enqueued_generation;
    /*
     * A causal successor owns a future claim on this exact physical range.
     * The predecessor remains the pool's range owner until either its
     * completion is observed or the successor's stream has accepted the
     * published dependency. Exactly one successor may exist.
     */
    struct ShadowSpillMemoryLease *causal_predecessor;
    struct ShadowSpillMemoryLease *causal_successor;
    uint64_t causal_predecessor_generation;
    ShadowSpillEventLease *causal_event;
    uint8_t causal_dependency_expected;
    struct ShadowSpillMemoryLease *ownership_next;
    struct ShadowSpillMemoryLease *free_record_next;
    struct ShadowSpillMemoryLease *id_index_next;
    struct ShadowSpillMemoryLease *pointer_index_next;
    struct ShadowSpillMemoryLease *reusable_index_next;
    struct ShadowSpillMemoryLease *active_next;
    struct ShadowSpillMemoryLease **active_previous_link;
    /*
     * Dispatcher-local frees are linked directly to the active task scope.
     * A lease stays linked when it is reused within that task; after_task()
     * examines its final state exactly once instead of scanning every active
     * execution lease.
     */
    struct ShadowSpillMemoryLease *task_retirement_next;
    uint8_t task_retirement_linked;
    struct ShadowSpillMemoryLease *pool_next;
    struct ShadowSpillMemoryLease **pool_previous_link;
    uint8_t in_reusable_index;
    uint8_t in_id_index;
    uint8_t in_pointer_index;
    uint8_t metadata_in_use;
    /*
     * The pool range allocator owns ordinary leases.  A borrowed lease names
     * bytes inside a separately owned parent range (for example, one admitted
     * plan slice) and therefore must never free those bytes independently.
     */
    uint8_t owns_pool_range;
} ShadowSpillMemoryLease;

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

typedef struct ShadowSpillRetirementRecord {
    ShadowSpillMemoryLease *allocation;
    ShadowSpillMemoryPool *pool;
    uint64_t allocation_id;
    uint64_t allocation_generation;
    ShadowSpillEventLease **events;
    uint32_t event_count;
    ShadowSpillEventLease *task_completion_event;
    struct ShadowSpillRetirementRecord *next;
    struct ShadowSpillRetirementRecord *free_next;
    uint8_t pool_owned;
} ShadowSpillRetirementRecord;

typedef struct ShadowSpillRetirementRecordBlock {
    ShadowSpillRetirementRecord *records;
    uint64_t count;
    struct ShadowSpillRetirementRecordBlock *next;
} ShadowSpillRetirementRecordBlock;

/*
 * Producers publish only fully described retirements.  The worker thread
 * detaches the complete list before inspecting event state, so neither event
 * polling nor pending-list traversal holds this lock.  Device-pool ownership
 * is entered separately and only for the final validated range release.
 */
typedef struct ShadowSpillRetirementQueue {
    pthread_mutex_t lock;
    ShadowSpillRetirementRecord *head;
    ShadowSpillRetirementRecord *tail;
    ShadowSpillRetirementRecord *free_head;
    ShadowSpillRetirementRecordBlock *blocks;
    _Atomic uint64_t count;
    uint64_t capacity;
    uint64_t available;
    uint64_t in_use;
    uint64_t peak_in_use;
    uint64_t growth_rejections;
    uint8_t lock_initialized;
    uint8_t sealed;
} ShadowSpillRetirementQueue;

typedef struct ShadowSpillRetirementWork {
    uint8_t pool_busy;
} ShadowSpillRetirementWork;

typedef struct ShadowSpillObjectLocation {
    ShadowSpillMemoryLease *lease;
    uint64_t version;
    uint8_t current;
    uint8_t owns_lease;
} ShadowSpillObjectLocation;

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
    pthread_cond_t state_changed;
    ShadowSpillObjectLocation *locations;
    uint32_t location_count;
    uint64_t generation;
    uint64_t authoritative_version;
    uint64_t allocation_id;
    uint8_t retain_spill_copy;
    uint8_t residency;
    /*
     * Number of queued fetch generations that have not yet published a
     * readiness event.  This must be a count rather than a Boolean: host
     * dispatch can enqueue a later release/fetch cycle while an earlier fetch
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

typedef struct ShadowSpillTransferLane {
    pthread_mutex_t lock;
    ShadowSpillQueuedAction *pending_head;
    ShadowSpillQueuedAction *pending_tail;
    ShadowSpillQueuedAction *inflight_head;
    ShadowSpillQueuedAction *inflight_tail;
    uint8_t lock_initialized;
} ShadowSpillTransferLane;

struct ShadowSpillRouteState {
    ShadowSpillTransferRoute route;
    ShadowSpillTransferLane transfers;
    ShadowSpillBackendStream lane;
    uint8_t lane_created;
};

/*
 * Owns only the quiescence notification consumed by runtime_wait_idle. The
 * final action and retirement transitions advance this epoch after their
 * counters reach zero. It deliberately does not drive the worker thread or
 * alter transfer/retirement dispatch cadence.
 */
typedef struct ShadowSpillIdleWakeup {
    pthread_mutex_t lock;
    pthread_cond_t condition;
    uint64_t epoch;
    uint8_t initialized;
} ShadowSpillIdleWakeup;

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
    ShadowSpillObject **inputs;
    uint64_t *input_plan_object_ids;
    uint8_t *input_consistency;
    uint32_t input_count;
    ShadowSpillObject **unique_inputs;
    uint32_t unique_input_count;
    uint32_t *input_unique_indices;
    uint32_t *unique_first_positions;
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
    _Atomic uint32_t active_invocations;
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

void shadowspill_abort_current_task(ShadowSpillRuntime *runtime);

struct ShadowSpillRuntime {
    /* Cold lifecycle and the still-unmigrated action-list owner. */
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    pthread_mutex_t failure_lock;
    ShadowSpillIdleWakeup idle_wakeup;
    pthread_t worker_thread;
    int worker_started;
    _Atomic uint8_t closing;
    _Atomic uint8_t closed;
    _Atomic uint8_t worker_stop;
    _Atomic uint32_t failure_status;
    uint64_t worker_poll_nanoseconds;

    ShadowSpillSynchronizationBackend synchronization;
    ShadowSpillProfiler profiler;
    ShadowSpillRouteState *routes;
    uint32_t route_count;

    pthread_rwlock_t transfer_profiles_lock;
    ShadowSpillTransferProfile *transfer_profiles;
    uint32_t transfer_profile_count;
    uint64_t transfer_profile_generation;
    uint8_t transfer_profiles_initialized;

    ShadowSpillMemoryPool *pools;
    uint32_t pool_count;
    ShadowSpillObjectTable objects;
    pthread_mutex_t plans_lock;
    ShadowSpillPlan *plans;
    uint8_t plans_lock_initialized;
    ShadowSpillEventPool events;
    ShadowSpillCompletionTracker completions;
    uint8_t completions_initialized;
    ShadowSpillRetirementQueue retirements;
    ShadowSpillActionQueue actions;
    _Atomic(ShadowSpillTaskRecord *) worker_submission;
    _Atomic uint64_t next_worker_submission_sequence;

    _Atomic uint64_t next_allocation_id;
    _Atomic uint64_t next_generation;
    _Atomic uint64_t next_event_generation;
    _Atomic uint64_t pending_retirements;
    _Atomic uint64_t pending_capacity_actions;
    _Atomic uint64_t registered_objects;
    _Atomic uint64_t fetch_transfers;
    _Atomic uint64_t evict_transfers;
    _Atomic uint64_t bytes_fetched;
    _Atomic uint64_t bytes_evicted;
    _Atomic uint64_t wait_events_inserted;
    uint64_t event_query_epoch;
    ShadowSpillAllocationEvent *allocation_events;
    _Atomic uint64_t allocation_event_count;
    uint64_t allocation_event_capacity;
    _Atomic uint64_t next_allocation_event_sequence;
    _Atomic uint8_t allocation_telemetry_active;
    _Atomic uint8_t allocation_event_overflow;
    ShadowSpillTraceEvent *trace_events;
    _Atomic uint64_t trace_event_count;
    uint64_t trace_event_capacity;
    _Atomic uint64_t next_trace_event_sequence;
    uint64_t trace_step_id;
    uint64_t trace_begin_timestamp_ns;
    uint64_t trace_end_timestamp_ns;
    uint64_t trace_allocation_event_capacity;
    _Atomic uint8_t trace_prepared;
    _Atomic uint8_t trace_active;
    _Atomic uint8_t trace_event_overflow;
    ShadowSpillRuntimeFailure failure;
};

static inline void shadowspill_cpu_relax(void) {
#if defined(__x86_64__) || defined(__i386__)
    __asm__ volatile("pause" ::: "memory");
#elif defined(__aarch64__) || defined(__arm__)
    __asm__ volatile("yield" ::: "memory");
#else
    atomic_signal_fence(memory_order_seq_cst);
#endif
}

int shadowspill_idle_wakeup_initialize(
    ShadowSpillIdleWakeup *wakeup
);
void shadowspill_idle_wakeup_destroy(ShadowSpillIdleWakeup *wakeup);
void shadowspill_idle_notify(ShadowSpillRuntime *runtime);

int shadowspill_range_initialize(
    ShadowSpillRangeAllocator *allocator,
    uint64_t capacity
);
int shadowspill_range_initialize_with_nodes(
    ShadowSpillRangeAllocator *allocator,
    uint64_t capacity,
    ShadowSpillRange *nodes,
    uint64_t node_capacity
);
int shadowspill_range_clone_extended(
    const ShadowSpillRangeAllocator *source,
    uint64_t capacity,
    ShadowSpillRangeAllocator *destination
);
void shadowspill_range_destroy(ShadowSpillRangeAllocator *allocator);
int shadowspill_range_allocate(
    ShadowSpillRangeAllocator *allocator,
    uint64_t bytes,
    uint64_t alignment,
    uint64_t *offset
);
int shadowspill_range_allocate_best_fit_low(
    ShadowSpillRangeAllocator *allocator,
    uint64_t bytes,
    uint64_t alignment,
    uint64_t *offset
);
int shadowspill_range_allocate_best_fit_high(
    ShadowSpillRangeAllocator *allocator,
    uint64_t bytes,
    uint64_t alignment,
    uint64_t *offset
);
int shadowspill_range_allocate_highest(
    ShadowSpillRangeAllocator *allocator,
    uint64_t bytes,
    uint64_t alignment,
    uint64_t minimum_offset,
    uint64_t *offset
);
int shadowspill_range_allocate_at(
    ShadowSpillRangeAllocator *allocator,
    uint64_t offset,
    uint64_t bytes
);
int shadowspill_range_free(
    ShadowSpillRangeAllocator *allocator,
    uint64_t offset,
    uint64_t bytes
);
uint64_t shadowspill_range_free_bytes(
    const ShadowSpillRangeAllocator *allocator
);
uint64_t shadowspill_range_free_prefix(
    const ShadowSpillRangeAllocator *allocator
);
uint64_t shadowspill_range_largest_free(
    const ShadowSpillRangeAllocator *allocator
);

int shadowspill_memory_pool_initialize(
    ShadowSpillMemoryPool *pool,
    uint32_t pool_id,
    const ShadowSpillMemoryPoolBackend *backend,
    uint64_t capacity,
    uint64_t minimum_alignment
);
void shadowspill_memory_pool_close(ShadowSpillMemoryPool *pool);
ShadowSpillRuntimeStatus shadowspill_memory_pool_reserve_lease_records(
    ShadowSpillMemoryPool *pool,
    uint64_t minimum_free_records
);
void shadowspill_memory_pool_lock_foreground(ShadowSpillMemoryPool *pool);
void shadowspill_memory_pool_unlock_foreground(ShadowSpillMemoryPool *pool);
void shadowspill_memory_pool_declare_reservation(ShadowSpillMemoryPool *pool);
void shadowspill_memory_pool_relinquish_reservation(ShadowSpillMemoryPool *pool);
void shadowspill_memory_pool_lock_reservation(ShadowSpillMemoryPool *pool);
int shadowspill_memory_pool_try_lock_reservation(ShadowSpillMemoryPool *pool);
void shadowspill_memory_pool_unlock_reservation(ShadowSpillMemoryPool *pool);
int shadowspill_memory_pool_try_lock_reclamation(ShadowSpillMemoryPool *pool);
void shadowspill_memory_pool_unlock_reclamation(ShadowSpillMemoryPool *pool);
int shadowspill_memory_pool_reserve_locked(
    ShadowSpillMemoryPool *pool,
    uint64_t bytes,
    uint64_t alignment,
    ShadowSpillMemoryPlacement placement,
    uint64_t *offset
);
int shadowspill_memory_pool_release_locked(
    ShadowSpillMemoryPool *pool,
    uint64_t offset,
    uint64_t bytes
);
int shadowspill_memory_pool_reserve_lease_locked(
    ShadowSpillMemoryPool *pool,
    ShadowSpillMemoryLease *lease,
    uint64_t bytes,
    uint64_t alignment,
    ShadowSpillMemoryPlacement placement
);
int shadowspill_memory_pool_release_lease_locked(
    ShadowSpillMemoryLease *lease
);
int shadowspill_memory_pool_mark_reserved_locked(
    ShadowSpillMemoryLease *lease
);
int shadowspill_memory_pool_begin_retirement_locked(
    ShadowSpillMemoryLease *lease,
    ShadowSpillEventLease *dependency_event,
    int dependency_expected
);
int shadowspill_memory_pool_publish_retirement_dependency_locked(
    ShadowSpillMemoryLease *lease,
    ShadowSpillEventLease *dependency_event
);
int shadowspill_memory_pool_cancel_retirement_locked(
    ShadowSpillMemoryLease *lease
);
int shadowspill_memory_pool_reserve_causal_successor_locked(
    ShadowSpillMemoryPool *pool,
    ShadowSpillMemoryLease *successor,
    uint64_t bytes,
    uint64_t alignment
);
int shadowspill_memory_pool_can_reserve_after_releases_locked(
    const ShadowSpillMemoryPool *pool,
    uint64_t bytes,
    uint64_t alignment
);
int shadowspill_memory_pool_find_release_frontier_locked(
    const ShadowSpillMemoryPool *pool,
    uint64_t bytes,
    uint64_t alignment,
    ShadowSpillMemoryLease **frontier,
    uint64_t frontier_capacity,
    uint64_t *frontier_count
);
int shadowspill_memory_pool_acquire_reserved_lease_locked(
    ShadowSpillMemoryLease *lease,
    ShadowSpillEventLease **dependency_event
);
int shadowspill_memory_pool_cancel_reservation_locked(
    ShadowSpillMemoryLease *lease
);
int shadowspill_memory_pool_adopt_lease_locked(
    ShadowSpillMemoryPool *pool,
    ShadowSpillMemoryLease *lease,
    uint64_t bytes,
    uint64_t alignment,
    uint64_t offset
);
int shadowspill_memory_pool_adopt_borrowed_lease_locked(
    ShadowSpillMemoryPool *pool,
    ShadowSpillMemoryLease *lease,
    uint64_t bytes,
    uint64_t alignment,
    uint64_t offset
);
void shadowspill_memory_pool_rebase_locked(
    ShadowSpillMemoryPool *pool,
    void *new_base
);
uint64_t shadowspill_memory_pool_free_bytes_locked(
    const ShadowSpillMemoryPool *pool
);
uint64_t shadowspill_memory_pool_free_prefix_locked(
    const ShadowSpillMemoryPool *pool
);
uint64_t shadowspill_memory_pool_largest_free_locked(
    const ShadowSpillMemoryPool *pool
);
void *shadowspill_memory_pool_pointer(
    const ShadowSpillMemoryPool *pool,
    uint64_t offset
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

ShadowSpillMemoryPool *shadowspill_runtime_pool(
    ShadowSpillRuntime *runtime,
    uint32_t pool_id
);
const ShadowSpillMemoryPool *shadowspill_runtime_pool_const(
    const ShadowSpillRuntime *runtime,
    uint32_t pool_id
);
ShadowSpillMemoryPool *shadowspill_execution_pool(ShadowSpillRuntime *runtime);
const ShadowSpillMemoryPool *shadowspill_execution_pool_const(
    const ShadowSpillRuntime *runtime
);
ShadowSpillMemoryPool *shadowspill_spill_pool(ShadowSpillRuntime *runtime);
ShadowSpillObjectLocation *shadowspill_object_location(
    ShadowSpillObject *object,
    uint32_t pool_id
);
ShadowSpillObjectLocation *shadowspill_execution_location(
    ShadowSpillRuntime *runtime,
    ShadowSpillObject *object
);
ShadowSpillObjectLocation *shadowspill_spill_location(
    ShadowSpillRuntime *runtime,
    ShadowSpillObject *object
);
ShadowSpillObjectLocation *shadowspill_plan_execution_location(
    const ShadowSpillPlan *plan,
    ShadowSpillObject *object
);
ShadowSpillObjectLocation *shadowspill_plan_spill_location(
    const ShadowSpillPlan *plan,
    ShadowSpillObject *object
);
ShadowSpillMemoryLease *shadowspill_find_execution_lease(
    ShadowSpillMemoryPool *pool,
    uint64_t allocation_id
);
ShadowSpillMemoryLease *shadowspill_find_execution_lease_by_pointer(
    ShadowSpillMemoryPool *pool,
    const void *pointer
);
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
ShadowSpillRuntimeStatus shadowspill_object_owner_retain(
    ShadowSpillObject *object
);
ShadowSpillRuntimeStatus shadowspill_object_owner_release(
    ShadowSpillObject *object
);
ShadowSpillRuntimeStatus shadowspill_object_schedule_action_locked(
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
ShadowSpillRuntimeStatus shadowspill_create_execution_lease_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryPool *pool,
    uint64_t bytes,
    uint64_t alignment,
    int plan_owned,
    ShadowSpillMemoryPlacement placement,
    uint64_t origin_task_id,
    ShadowSpillMemoryLease **record
);
ShadowSpillRuntimeStatus shadowspill_create_fixed_execution_lease_locked(
    ShadowSpillPlan *plan,
    const ShadowSpillFixedPlacementDescription *placement,
    int plan_owned,
    uint64_t origin_task_id,
    ShadowSpillMemoryLease **record
);
ShadowSpillRuntimeStatus shadowspill_create_execution_successor_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryPool *pool,
    uint64_t bytes,
    uint64_t alignment,
    uint64_t origin_task_id,
    ShadowSpillMemoryLease **record
);
int shadowspill_acquire_reserved_execution_lease_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryLease *lease,
    ShadowSpillEventLease **dependency_event
);
void shadowspill_cancel_execution_reservation_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryLease *lease
);
void shadowspill_release_execution_lease_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryLease *allocation
);
ShadowSpillMemoryLease *shadowspill_memory_pool_acquire_lease_record_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryPool *pool,
    uint64_t origin_task_id
);
void shadowspill_memory_lease_retain(ShadowSpillMemoryLease *lease);
void shadowspill_memory_lease_release(ShadowSpillMemoryLease *lease);
void shadowspill_memory_pool_try_recycle_lease_record_locked(
    ShadowSpillMemoryLease *lease
);
ShadowSpillRuntimeStatus shadowspill_publish_task_retirement_event(
    ShadowSpillRuntime *runtime,
    uint64_t task_id,
    ShadowSpillBackendStream stream
);
void shadowspill_finalize_aborted_task_retirements(
    ShadowSpillRuntime *runtime,
    uint64_t task_id
);
int shadowspill_retirement_queue_initialize(
    ShadowSpillRetirementQueue *queue
);
ShadowSpillRuntimeStatus shadowspill_retirement_queue_reserve(
    ShadowSpillRetirementQueue *queue,
    uint64_t minimum_free_records
);
void shadowspill_retirement_queue_destroy(
    ShadowSpillRuntime *runtime,
    ShadowSpillRetirementQueue *queue
);
ShadowSpillRuntimeStatus shadowspill_retirement_enqueue_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryLease *allocation
);
ShadowSpillRetirementWork shadowspill_handle_retirements(
    ShadowSpillRuntime *runtime
);
int shadowspill_has_actionable_retirement(ShadowSpillRuntime *runtime);
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
ShadowSpillRuntimeStatus shadowspill_validate_task_allocation(
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
ShadowSpillRuntimeStatus shadowspill_release_task_allocation(
    ShadowSpillRuntime *runtime,
    uint64_t origin_task_id,
    uint64_t origin_task_invocation,
    uint64_t allocation_ordinal,
    int allocation_is_scratch,
    uint64_t requested_bytes,
    uint64_t charged_bytes,
    uint64_t alignment_bytes
);
ShadowSpillRuntimeStatus shadowspill_validate_task_allocation_complete(
    ShadowSpillRuntime *runtime
);
void shadowspill_leave_task_scope(ShadowSpillRuntime *runtime);
void shadowspill_append_allocation_event_locked(
    ShadowSpillRuntime *runtime,
    const ShadowSpillMemoryLease *allocation,
    ShadowSpillAllocationEventKind kind,
    ShadowSpillAllocationCategory category
);
void shadowspill_trace_append_enabled(
    ShadowSpillRuntime *runtime,
    ShadowSpillTraceEventKind kind,
    uint64_t task_id,
    uint64_t object_id,
    uint64_t allocation_id,
    uint64_t bytes,
    uint64_t detail_0,
    uint64_t detail_1
);
#define shadowspill_append_trace_event_locked(runtime_, ...)                  \
    do {                                                                       \
        ShadowSpillRuntime *const shadowspill_trace_runtime__ = (runtime_);    \
        if (atomic_load_explicit(                                              \
                &shadowspill_trace_runtime__->trace_active,                    \
                memory_order_acquire                                           \
            ) != 0U) {                                                         \
            shadowspill_trace_append_enabled(                                 \
                shadowspill_trace_runtime__, __VA_ARGS__                       \
            );                                                                 \
        }                                                                      \
    } while (0)
int shadowspill_memory_pool_backend_is_valid(
    const ShadowSpillMemoryPoolBackend *backend
);
int shadowspill_transfer_route_is_valid(
    const ShadowSpillTransferRoute *route
);
int shadowspill_synchronization_backend_is_valid(
    const ShadowSpillSynchronizationBackend *backend
);
ShadowSpillTransferRoute *shadowspill_transfer_route(
    ShadowSpillRuntime *runtime,
    uint32_t source_pool_id,
    uint32_t destination_pool_id
);
ShadowSpillBackendStream *shadowspill_transfer_route_lane(
    ShadowSpillRuntime *runtime,
    const ShadowSpillTransferRoute *route
);
ShadowSpillRouteState *shadowspill_runtime_route(
    ShadowSpillRuntime *runtime,
    uint32_t route_id
);
void shadowspill_plan_destroy_all(ShadowSpillRuntime *runtime);
int shadowspill_transfer_profiles_initialize(ShadowSpillRuntime *runtime);
void shadowspill_transfer_profiles_destroy(ShadowSpillRuntime *runtime);
void shadowspill_publish_pool_geometry_locked(ShadowSpillMemoryPool *pool);
int shadowspill_task_table_initialize(
    ShadowSpillTaskTable *table,
    uint64_t bucket_count
);
void shadowspill_object_acquisitions_clear(ShadowSpillPlan *plan);
ShadowSpillRuntimeStatus shadowspill_acquire_object_bindings(
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
ShadowSpillRuntimeStatus shadowspill_object_transfer_to_caller(
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
ShadowSpillRuntimeStatus shadowspill_object_bind_allocation(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryPool *pool,
    ShadowSpillObject *object,
    const void *pointer,
    const ShadowSpillTaskRecord *task,
    ShadowSpillObjectBinding *binding
);
ShadowSpillRuntimeStatus shadowspill_object_replace_allocation(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryPool *pool,
    ShadowSpillObject *object,
    const void *pointer,
    ShadowSpillObjectBinding *binding
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
ShadowSpillRuntimeStatus shadowspill_after_task_record(
    ShadowSpillRuntime *runtime,
    const ShadowSpillTaskRecord *record,
    ShadowSpillBackendStream compute_stream
);
int shadowspill_transfer_lane_initialize(ShadowSpillTransferLane *lane);
void shadowspill_transfer_lane_destroy(ShadowSpillTransferLane *lane);
ShadowSpillTransferLane *shadowspill_transfer_lane_for_action(
    ShadowSpillRuntime *runtime,
    const ShadowSpillQueuedAction *action
);
void shadowspill_transfer_lane_enqueue(
    ShadowSpillTransferLane *lane,
    ShadowSpillQueuedAction *action
);
int shadowspill_transfer_lane_claim(
    ShadowSpillTransferLane *lane,
    ShadowSpillQueuedAction *action
);
void shadowspill_transfer_lane_publish_inflight(
    ShadowSpillTransferLane *lane,
    ShadowSpillQueuedAction *action
);
int shadowspill_transfer_lane_is_inflight_head(
    ShadowSpillTransferLane *lane,
    const ShadowSpillQueuedAction *action
);
int shadowspill_transfer_lane_complete(
    ShadowSpillTransferLane *lane,
    ShadowSpillQueuedAction *action
);
void *shadowspill_worker_main(void *pointer);
void shadowspill_notify_worker(ShadowSpillRuntime *runtime);

int shadowspill_profiler_is_valid(const ShadowSpillProfiler *profiler);
void shadowspill_profiler_set_enabled(
    const ShadowSpillProfiler *profiler, uint8_t enabled
);
void shadowspill_profiler_name_current_thread(
    const ShadowSpillProfiler *profiler, const char *name
);
void shadowspill_profiler_name_stream(
    const ShadowSpillProfiler *profiler,
    ShadowSpillBackendStream stream,
    const char *name
);
ShadowSpillProfilerRange shadowspill_profiler_range_begin(
    const ShadowSpillProfiler *profiler, const char *name
);
void shadowspill_profiler_range_end(
    const ShadowSpillProfiler *profiler, ShadowSpillProfilerRange range
);

#endif
