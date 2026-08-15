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

struct ShadowSpillEventLease {
    ShadowSpillBackendEvent event;
    uint64_t generation;
    _Atomic uint32_t references;
    /*
     * The backend has reported that the event completed. This is deliberately
     * not a memory-availability flag: a range becomes reusable only when its
     * owning MemoryPool commits the matching lease transition under pool.lock.
     */
    _Atomic uint8_t backend_complete;
};

typedef struct ShadowSpillEventRecord {
    ShadowSpillEventLease *event;
    struct ShadowSpillEventRecord *next;
} ShadowSpillEventRecord;

typedef struct ShadowSpillCompletionRecord {
    ShadowSpillEventLease *event;
    uint64_t object_id;
    uint64_t allocation_id;
    struct ShadowSpillCompletionRecord *next;
} ShadowSpillCompletionRecord;

typedef struct ShadowSpillCompletionStream {
    ShadowSpillBackendStream stream;
    ShadowSpillCompletionRecord *head;
    ShadowSpillCompletionRecord *tail;
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
    SHADOWSPILL_INITIAL_POOL_COUNT = 2U,
};

/*
 * Owns one bounded arena and its suballocation geometry. A pool does not know
 * whether its storage is local, remote, accelerator, host, or persistent.
 * Concrete backends attach that meaning when they instantiate the pool.
 */
typedef struct ShadowSpillMemoryPool {
    pthread_mutex_t lock;
    pthread_cond_t capacity_changed;
    _Atomic uint64_t foreground_waiters;
    _Atomic uint64_t reservation_waiters;
    ShadowSpillRangeAllocator ranges;
    struct ShadowSpillMemoryLease *leases;
    ShadowSpillMemoryPoolBackend backend;
    void *base;
    uint32_t pool_id;
    uint64_t minimum_alignment;
    uint64_t next_request_sequence;
    uint64_t next_release_sequence;
    uint64_t reserved_bytes;
    uint8_t initialized;
} ShadowSpillMemoryPool;

typedef struct ShadowSpillTaskFence ShadowSpillTaskFence;

typedef enum ShadowSpillMemoryLeaseState {
    SHADOWSPILL_LEASE_FREE = 0,
    SHADOWSPILL_LEASE_IN_USE = 1,
    SHADOWSPILL_LEASE_RETIRE_PENDING = 2,
    SHADOWSPILL_LEASE_RESERVED = 3,
    SHADOWSPILL_LEASE_SUCCESSOR_RESERVED = 4,
    SHADOWSPILL_LEASE_PREDECESSOR_TRANSFERRED = 5,
} ShadowSpillMemoryLeaseState;

typedef struct ShadowSpillMemoryLease {
    ShadowSpillMemoryPool *pool;
    ShadowSpillMemoryLeaseState state;
    uint64_t allocation_id;
    uint64_t generation;
    uint64_t requested_bytes;
    uint64_t charged_bytes;
    uint64_t alignment_bytes;
    uint64_t offset;
    uint64_t origin_task_id;
    uint64_t origin_task_allocation_ordinal;
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
    uint64_t bound_object_id;
    /*
     * Zero-copy task outputs may hand this lease through several logical
     * objects before the worker observes the first completion fence.  The
     * source objects form an intrusive FIFO; the lease owns only its ends.
     */
    uint64_t handoff_head_object_id;
    uint64_t handoff_tail_object_id;
    ShadowSpillStreamRecord *streams;
    ShadowSpillEventRecord *retirement_events;
    ShadowSpillTaskFence *retirement_fence;
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
    struct ShadowSpillMemoryLease *next;
    struct ShadowSpillMemoryLease *id_index_next;
    struct ShadowSpillMemoryLease *pointer_index_next;
    struct ShadowSpillMemoryLease *reusable_index_next;
    struct ShadowSpillMemoryLease *active_next;
    struct ShadowSpillMemoryLease **active_previous_link;
    struct ShadowSpillMemoryLease *pool_next;
    struct ShadowSpillMemoryLease **pool_previous_link;
    uint8_t in_reusable_index;
} ShadowSpillMemoryLease;

typedef struct ShadowSpillRetirementRecord {
    ShadowSpillMemoryLease *allocation;
    uint64_t allocation_generation;
    ShadowSpillEventLease **events;
    uint32_t event_count;
    ShadowSpillTaskFence *fence;
    struct ShadowSpillRetirementRecord *next;
} ShadowSpillRetirementRecord;

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
    _Atomic uint64_t count;
    uint8_t lock_initialized;
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

typedef struct ShadowSpillObject {
    uint64_t object_id;
    uint64_t size_bytes;
    _Atomic uint32_t references;
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
    _Atomic uint8_t prefetch_pending;
    ShadowSpillEventLease *readiness_event;
    uint8_t has_readiness_event;
    uint64_t retired_generation;
    void *retired_execution_pointer;
    uint64_t handoff_destination_object_id;
    uint64_t handoff_task_id;
    uint64_t handoff_next_source_object_id;
    ShadowSpillQueuedAction *action_head;
    ShadowSpillQueuedAction *action_tail;
    struct ShadowSpillObject *ownership_next;
    struct ShadowSpillObject **ownership_previous_link;
    struct ShadowSpillObject *id_index_next;
} ShadowSpillObject;

typedef struct ShadowSpillObjectTable {
    pthread_rwlock_t lock;
    uint8_t lock_initialized;
    ShadowSpillObject *owned_head;
    ShadowSpillObject **by_id;
    uint64_t bucket_count;
} ShadowSpillObjectTable;

struct ShadowSpillTaskFence {
    ShadowSpillEventLease *event;
    _Atomic uint32_t references;
    _Atomic uint8_t completion_known;
    uint8_t last_query_complete;
    uint64_t last_query_epoch;
};

typedef enum ShadowSpillQueuedActionState {
    SHADOWSPILL_ACTION_QUEUED = 0,
    SHADOWSPILL_ACTION_IN_FLIGHT = 1,
    SHADOWSPILL_ACTION_FINISHED = 2,
} ShadowSpillQueuedActionState;

struct ShadowSpillQueuedAction {
    uint64_t task_id;
    uint8_t kind;
    uint8_t state;
    ShadowSpillMemoryLease *destination_lease;
    ShadowSpillObject *object;
    ShadowSpillTaskFence *fence;
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

typedef struct ShadowSpillExecutionUpdate {
    ShadowSpillObject *object;
    uint64_t version_delta;
} ShadowSpillExecutionUpdate;

typedef struct ShadowSpillExecutionAction {
    ShadowSpillObject *object;
    uint8_t kind;
    char *trace_label;
} ShadowSpillExecutionAction;

typedef struct ShadowSpillExecutionRecord {
    ShadowSpillRuntime *runtime_owner;
    uint64_t task_id;
    ShadowSpillObject **inputs;
    uint32_t input_count;
    ShadowSpillObject **unique_inputs;
    uint32_t unique_input_count;
    uint32_t *input_unique_indices;
    uint32_t *unique_first_positions;
    ShadowSpillExecutionUpdate *updates;
    uint32_t update_count;
    ShadowSpillExecutionAction *actions;
    ShadowSpillQueuedAction *queued_actions;
    uint32_t action_count;
    ShadowSpillTaskAllocationABIStep *allocation_abi_steps;
    uint32_t allocation_abi_step_count;
    uint8_t enforce_allocation_abi;
    uint64_t maximum_requested_allocation_bytes;
    uint64_t maximum_charged_allocation_bytes;
    uint64_t live_requested_allocation_limit_bytes;
    uint64_t live_charged_allocation_limit_bytes;
    struct ShadowSpillExecutionRecord *hash_next;
    struct ShadowSpillExecutionRecord *ownership_next;
} ShadowSpillExecutionRecord;

typedef struct ShadowSpillExecutionTable {
    pthread_rwlock_t lock;
    ShadowSpillExecutionRecord **by_id;
    ShadowSpillExecutionRecord *owned_head;
    uint64_t bucket_count;
    uint8_t lock_initialized;
} ShadowSpillExecutionTable;

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

    ShadowSpillBackend backend;
    ShadowSpillProfiler profiler;
    ShadowSpillTransferRoute fetch_route;
    ShadowSpillTransferRoute evict_route;
    ShadowSpillBackendStream fetch_stream;
    ShadowSpillBackendStream evict_stream;
    int fetch_stream_created;
    int evict_stream_created;

    pthread_rwlock_t transfer_profiles_lock;
    ShadowSpillTransferProfile *transfer_profiles;
    uint32_t transfer_profile_count;
    uint64_t transfer_profile_generation;
    uint8_t transfer_profiles_initialized;

    ShadowSpillMemoryPool *pools;
    uint32_t pool_count;
    uint32_t execution_pool_id;
    uint32_t spill_pool_id;
    _Atomic uint64_t execution_free_bytes_snapshot;
    _Atomic uint64_t execution_largest_free_snapshot;
    ShadowSpillMemoryLease *execution_leases;
    ShadowSpillMemoryLease *active_execution_leases;
    ShadowSpillMemoryLease **execution_leases_by_id;
    ShadowSpillMemoryLease **execution_leases_by_pointer;
    ShadowSpillMemoryLease **reusable_execution_leases_by_size;
    uint64_t allocation_index_bucket_count;
    uint64_t reusable_index_bucket_count;
    ShadowSpillObjectTable objects;
    ShadowSpillExecutionTable execution;
    ShadowSpillCompletionTracker completions;
    uint8_t completions_initialized;
    ShadowSpillRetirementQueue retirements;
    ShadowSpillActionQueue actions;
    ShadowSpillTransferLane fetch_lane;
    ShadowSpillTransferLane evict_lane;

    uint64_t next_allocation_id;
    uint64_t next_generation;
    _Atomic uint64_t next_event_generation;
    uint64_t requested_allocated_bytes;
    uint64_t peak_requested_allocated_bytes;
    uint64_t live_allocations;
    uint64_t blocked_allocators;
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
ShadowSpillMemoryLease *shadowspill_find_execution_lease(
    ShadowSpillRuntime *runtime,
    uint64_t allocation_id
);
ShadowSpillMemoryLease *shadowspill_find_execution_lease_by_pointer(
    ShadowSpillRuntime *runtime,
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
ShadowSpillRuntimeStatus shadowspill_object_schedule_action_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillObject *object,
    ShadowSpillQueuedAction *action
);
int shadowspill_object_action_is_head_locked(
    const ShadowSpillObject *object,
    const ShadowSpillQueuedAction *action
);
int shadowspill_object_fetch_event_unpublished_locked(
    const ShadowSpillObject *object
);
int shadowspill_object_remove_action_locked(
    ShadowSpillObject *object,
    ShadowSpillQueuedAction *action
);
ShadowSpillRuntimeStatus shadowspill_create_execution_lease_locked(
    ShadowSpillRuntime *runtime,
    uint64_t bytes,
    uint64_t alignment,
    int plan_owned,
    ShadowSpillMemoryPlacement placement,
    uint64_t origin_task_id,
    ShadowSpillMemoryLease **record
);
ShadowSpillRuntimeStatus shadowspill_create_execution_successor_locked(
    ShadowSpillRuntime *runtime,
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
void shadowspill_release_task_fence_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillTaskFence *fence
);
void shadowspill_retain_task_fence(ShadowSpillTaskFence *fence);
int shadowspill_task_fence_complete_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillTaskFence *fence,
    int *complete
);
ShadowSpillRuntimeStatus shadowspill_fence_task_retirements_locked(
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
int shadowspill_enter_task_scope(
    ShadowSpillRuntime *runtime,
    uint64_t task_id
);
int shadowspill_enter_execution_scope(
    ShadowSpillRuntime *runtime,
    const ShadowSpillExecutionRecord *record
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
    uint64_t allocation_ordinal,
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
int shadowspill_backend_is_valid(const ShadowSpillBackend *backend);
int shadowspill_memory_pool_backend_is_valid(
    const ShadowSpillMemoryPoolBackend *backend
);
int shadowspill_transfer_route_is_valid(
    const ShadowSpillTransferRoute *route
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
int shadowspill_transfer_profiles_initialize(ShadowSpillRuntime *runtime);
void shadowspill_transfer_profiles_destroy(ShadowSpillRuntime *runtime);
void shadowspill_publish_execution_geometry_locked(ShadowSpillRuntime *runtime);
int shadowspill_execution_table_initialize(
    ShadowSpillExecutionTable *table,
    uint64_t bucket_count
);
void shadowspill_execution_table_destroy(ShadowSpillExecutionTable *table);
void shadowspill_execution_table_clear(ShadowSpillExecutionTable *table);
ShadowSpillExecutionRecord *shadowspill_execution_table_acquire(
    ShadowSpillExecutionTable *table,
    uint64_t task_id
);
char *shadowspill_copy_action_trace_label(
    const ShadowSpillRuntimeAction *action,
    uint64_t task_id,
    uint64_t size_bytes
);
ShadowSpillRuntimeStatus shadowspill_after_execution_record(
    ShadowSpillRuntime *runtime,
    const ShadowSpillExecutionRecord *record,
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
