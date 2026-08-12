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
} ShadowSpillRangeAllocator;

typedef struct ShadowSpillStreamRecord {
    ShadowSpillBackendStream stream;
    struct ShadowSpillStreamRecord *next;
} ShadowSpillStreamRecord;

struct ShadowSpillEventLease {
    ShadowSpillBackendEvent event;
    uint64_t generation;
    _Atomic uint32_t references;
    _Atomic uint8_t completion_known;
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
    uint64_t poll_misses;
    uint64_t next_poll_timestamp_ns;
    struct ShadowSpillCompletionStream *next;
} ShadowSpillCompletionStream;

typedef struct ShadowSpillCompletionTracker {
    pthread_mutex_t lock;
    ShadowSpillCompletionStream *streams;
    uint64_t pending;
} ShadowSpillCompletionTracker;

typedef enum ShadowSpillMemoryKind {
    SHADOWSPILL_MEMORY_DEVICE = 0,
    SHADOWSPILL_MEMORY_PINNED_HOST = 1,
} ShadowSpillMemoryKind;

typedef enum ShadowSpillMemoryPlacement {
    SHADOWSPILL_MEMORY_FIRST_FIT = 0,
    SHADOWSPILL_MEMORY_BEST_FIT_LOW = 1,
    SHADOWSPILL_MEMORY_BEST_FIT_HIGH = 2,
} ShadowSpillMemoryPlacement;

/*
 * Owns one bounded physical arena and its suballocation geometry. Device and
 * pinned-host memory are two configured instances of this same abstraction.
 * Allocation records, stream retirement, and object residency deliberately
 * remain clients of the pool rather than being embedded in it. Foreground
 * allocator callbacks have priority over background reclamation whenever
 * both need to mutate range geometry.
 */
typedef struct ShadowSpillMemoryPool {
    pthread_mutex_t lock;
    pthread_cond_t capacity_changed;
    _Atomic uint64_t foreground_waiters;
    ShadowSpillRangeAllocator ranges;
    void *base;
    uint64_t minimum_alignment;
    ShadowSpillMemoryKind kind;
    uint8_t initialized;
} ShadowSpillMemoryPool;

typedef struct ShadowSpillTaskFence ShadowSpillTaskFence;

typedef struct ShadowSpillAllocationRecord {
    uint64_t allocation_id;
    uint64_t generation;
    uint64_t requested_bytes;
    uint64_t charged_bytes;
    uint64_t offset;
    uint64_t origin_task_id;
    uint64_t release_task_id;
    void *pointer;
    _Atomic uint32_t references;
    int logical_freed;
    uint8_t retirement_preparing;
    int plan_owned;
    int ever_plan_owned;
    int framework_free_seen;
    uint64_t bound_object_id;
    uint64_t handoff_from_object_id;
    uint64_t handoff_to_object_id;
    uint64_t handoff_task_id;
    ShadowSpillStreamRecord *streams;
    ShadowSpillEventRecord *retirement_events;
    ShadowSpillTaskFence *retirement_fence;
    uint64_t retirement_enqueued_generation;
    struct ShadowSpillAllocationRecord *next;
    struct ShadowSpillAllocationRecord *id_index_next;
    struct ShadowSpillAllocationRecord *pointer_index_next;
    struct ShadowSpillAllocationRecord *reusable_index_next;
    struct ShadowSpillAllocationRecord *active_next;
    struct ShadowSpillAllocationRecord **active_previous_link;
    uint8_t in_reusable_index;
} ShadowSpillAllocationRecord;

typedef struct ShadowSpillRetirementRecord {
    ShadowSpillAllocationRecord *allocation;
    uint64_t allocation_generation;
    ShadowSpillEventLease **events;
    uint32_t event_count;
    ShadowSpillTaskFence *fence;
    struct ShadowSpillRetirementRecord *next;
} ShadowSpillRetirementRecord;

/*
 * Producers publish only fully described retirements.  The progress worker
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

typedef struct ShadowSpillRetirementProgress {
    uint8_t pool_busy;
} ShadowSpillRetirementProgress;

typedef struct ShadowSpillObjectRecord {
    uint64_t object_id;
    uint64_t size_bytes;
    _Atomic uint32_t references;
    _Atomic uint8_t detached;
    pthread_mutex_t lock;
    pthread_cond_t state_changed;
    uint64_t generation;
    uint64_t authoritative_version;
    uint64_t device_version;
    uint64_t host_version;
    uint64_t allocation_id;
    ShadowSpillAllocationRecord *device_lease;
    uint64_t host_offset;
    uint8_t retain_host_backing;
    uint8_t host_current;
    uint8_t has_host_range;
    uint8_t residency;
    _Atomic uint8_t prefetch_pending;
    ShadowSpillEventLease *readiness_event;
    uint8_t has_readiness_event;
    uint64_t retired_generation;
    void *retired_device_pointer;
    struct ShadowSpillObjectRecord *ownership_next;
    struct ShadowSpillObjectRecord **ownership_previous_link;
    struct ShadowSpillObjectRecord *id_index_next;
} ShadowSpillObjectRecord;

typedef struct ShadowSpillObjectTable {
    pthread_rwlock_t lock;
    uint8_t lock_initialized;
    ShadowSpillObjectRecord *owned_head;
    ShadowSpillObjectRecord **by_id;
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
} ShadowSpillQueuedActionState;

typedef struct ShadowSpillQueuedAction {
    uint64_t task_id;
    uint8_t kind;
    uint8_t state;
    uint8_t destination_reserved;
    uint64_t destination_offset;
    uint64_t destination_bytes;
    ShadowSpillObjectRecord *object;
    ShadowSpillTaskFence *fence;
    ShadowSpillEventLease *completion_event;
    uint8_t has_completion_event;
    uint8_t processing;
    struct ShadowSpillQueuedAction *previous;
    struct ShadowSpillQueuedAction *next;
    struct ShadowSpillQueuedAction *lane_previous;
    struct ShadowSpillQueuedAction *lane_next;
    uint8_t lane_state;
} ShadowSpillQueuedAction;

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
 * counters reach zero. It deliberately does not drive the progress worker or
 * alter transfer/retirement dispatch cadence.
 */
typedef struct ShadowSpillIdleWakeup {
    pthread_mutex_t lock;
    pthread_cond_t condition;
    uint64_t epoch;
    uint8_t initialized;
} ShadowSpillIdleWakeup;

typedef struct ShadowSpillExecutionUpdate {
    ShadowSpillObjectRecord *object;
    uint64_t version_delta;
} ShadowSpillExecutionUpdate;

typedef struct ShadowSpillExecutionAction {
    ShadowSpillObjectRecord *object;
    uint8_t kind;
} ShadowSpillExecutionAction;

typedef struct ShadowSpillExecutionRecord {
    ShadowSpillRuntime *runtime_owner;
    uint64_t task_id;
    ShadowSpillObjectRecord **inputs;
    uint32_t input_count;
    ShadowSpillObjectRecord **unique_inputs;
    uint32_t unique_input_count;
    ShadowSpillExecutionUpdate *updates;
    uint32_t update_count;
    ShadowSpillExecutionAction *actions;
    uint32_t action_count;
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
    pthread_t progress_thread;
    int progress_started;
    _Atomic uint8_t closing;
    _Atomic uint8_t closed;
    _Atomic uint8_t worker_stop;
    _Atomic uint32_t failure_status;
    uint64_t progress_poll_nanoseconds;

    ShadowSpillBackend backend;
    ShadowSpillBackendStream h2d_stream;
    ShadowSpillBackendStream d2h_stream;
    int h2d_stream_created;
    int d2h_stream_created;

    ShadowSpillMemoryPool device_pool;
    _Atomic uint64_t device_free_bytes_snapshot;
    _Atomic uint64_t device_largest_free_snapshot;
    ShadowSpillMemoryPool host_pool;
    ShadowSpillAllocationRecord *allocations;
    ShadowSpillAllocationRecord *active_allocations;
    ShadowSpillAllocationRecord **allocations_by_id;
    ShadowSpillAllocationRecord **allocations_by_pointer;
    ShadowSpillAllocationRecord **reusable_by_size;
    uint64_t allocation_index_bucket_count;
    uint64_t reusable_index_bucket_count;
    ShadowSpillObjectTable objects;
    ShadowSpillExecutionTable execution;
    ShadowSpillCompletionTracker completions;
    uint8_t completions_initialized;
    ShadowSpillRetirementQueue retirements;
    ShadowSpillActionQueue actions;
    ShadowSpillTransferLane h2d_lane;
    ShadowSpillTransferLane d2h_lane;

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
    _Atomic uint64_t transfers_to_device;
    _Atomic uint64_t transfers_to_host;
    _Atomic uint64_t bytes_to_device;
    _Atomic uint64_t bytes_to_host;
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
int shadowspill_range_free(
    ShadowSpillRangeAllocator *allocator,
    uint64_t offset,
    uint64_t bytes
);
uint64_t shadowspill_range_free_bytes(
    const ShadowSpillRangeAllocator *allocator
);
uint64_t shadowspill_range_largest_free(
    const ShadowSpillRangeAllocator *allocator
);

int shadowspill_memory_pool_initialize(
    ShadowSpillMemoryPool *pool,
    ShadowSpillMemoryKind kind,
    void *base,
    uint64_t capacity,
    uint64_t minimum_alignment
);
void shadowspill_memory_pool_destroy(ShadowSpillMemoryPool *pool);
void shadowspill_memory_pool_lock_foreground(ShadowSpillMemoryPool *pool);
void shadowspill_memory_pool_unlock_foreground(ShadowSpillMemoryPool *pool);
int shadowspill_memory_pool_try_lock_background(ShadowSpillMemoryPool *pool);
void shadowspill_memory_pool_unlock_background(ShadowSpillMemoryPool *pool);
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
uint64_t shadowspill_memory_pool_free_bytes_locked(
    const ShadowSpillMemoryPool *pool
);
uint64_t shadowspill_memory_pool_largest_free_locked(
    const ShadowSpillMemoryPool *pool
);
void *shadowspill_memory_pool_pointer(
    const ShadowSpillMemoryPool *pool,
    uint64_t offset
);

ShadowSpillAllocationRecord *shadowspill_find_allocation(
    ShadowSpillRuntime *runtime,
    uint64_t allocation_id
);
ShadowSpillAllocationRecord *shadowspill_find_allocation_by_pointer(
    ShadowSpillRuntime *runtime,
    const void *pointer
);
ShadowSpillObjectRecord *shadowspill_find_object(
    ShadowSpillRuntime *runtime,
    uint64_t object_id
);
int shadowspill_object_table_initialize(
    ShadowSpillObjectTable *table,
    uint64_t bucket_count
);
void shadowspill_object_table_destroy(ShadowSpillObjectTable *table);
ShadowSpillObjectRecord *shadowspill_object_table_find(
    const ShadowSpillObjectTable *table,
    uint64_t object_id
);
ShadowSpillObjectRecord *shadowspill_object_table_acquire(
    ShadowSpillObjectTable *table,
    uint64_t object_id
);
int shadowspill_object_table_insert(
    ShadowSpillObjectTable *table,
    ShadowSpillObjectRecord *object
);
int shadowspill_object_table_remove(
    ShadowSpillObjectTable *table,
    ShadowSpillObjectRecord *object
);
void shadowspill_object_retain(ShadowSpillObjectRecord *object);
void shadowspill_object_release(ShadowSpillObjectRecord *object);
ShadowSpillRuntimeStatus shadowspill_allocate_locked(
    ShadowSpillRuntime *runtime,
    uint64_t bytes,
    uint64_t alignment,
    int plan_owned,
    uint64_t origin_task_id,
    ShadowSpillAllocationRecord **record
);
ShadowSpillRuntimeStatus shadowspill_adopt_reserved_device_range_locked(
    ShadowSpillRuntime *runtime,
    uint64_t bytes,
    uint64_t offset,
    int plan_owned,
    uint64_t origin_task_id,
    ShadowSpillAllocationRecord **record
);
void shadowspill_release_allocation_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillAllocationRecord *allocation
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
    ShadowSpillAllocationRecord *allocation
);
ShadowSpillRetirementProgress shadowspill_progress_retirements(
    ShadowSpillRuntime *runtime
);
int shadowspill_has_actionable_retirement(ShadowSpillRuntime *runtime);
uint64_t shadowspill_current_task_id(ShadowSpillRuntime *runtime);
int shadowspill_enter_task_scope(
    ShadowSpillRuntime *runtime,
    uint64_t task_id
);
void shadowspill_leave_task_scope(ShadowSpillRuntime *runtime);
void shadowspill_append_allocation_event_locked(
    ShadowSpillRuntime *runtime,
    const ShadowSpillAllocationRecord *allocation,
    ShadowSpillAllocationEventKind kind,
    ShadowSpillAllocationCategory category
);
void shadowspill_append_trace_event_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillTraceEventKind kind,
    uint64_t task_id,
    uint64_t object_id,
    uint64_t allocation_id,
    uint64_t bytes,
    uint64_t detail_0,
    uint64_t detail_1
);
int shadowspill_backend_is_valid(const ShadowSpillBackend *backend);
void shadowspill_publish_device_geometry_locked(ShadowSpillRuntime *runtime);
int shadowspill_execution_table_initialize(
    ShadowSpillExecutionTable *table,
    uint64_t bucket_count
);
void shadowspill_execution_table_destroy(ShadowSpillExecutionTable *table);
ShadowSpillExecutionRecord *shadowspill_execution_table_acquire(
    ShadowSpillExecutionTable *table,
    uint64_t task_id
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
int shadowspill_transfer_lane_complete(
    ShadowSpillTransferLane *lane,
    ShadowSpillQueuedAction *action
);
void *shadowspill_progress_main(void *pointer);

#endif
