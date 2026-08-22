#ifndef SHADOWSPILL_RUNTIME_MEMORY_INTERNAL_H
#define SHADOWSPILL_RUNTIME_MEMORY_INTERNAL_H

/*
 * Ranges, pools, leases and retirement.
 *
 * A pool owns one bounded arena and the leases suballocated from it. A lease
 * outlives its physical range: the range returns to the allocator once the
 * work that touched it completes, while the metadata record stays with the
 * pool for reuse.
 */

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>

#include <shadowspill/runtime.h>

#include "../sync/internal.h"

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

typedef struct ShadowSpillObject ShadowSpillObject;

/*
 * One reusable record follows a lease use from stream attribution through
 * retirement completion.  The record is never copied: while the lease is
 * live, ``event`` is null; an ordinary asynchronous free fills the event in
 * place and transfers ownership of the complete list to the retirement queue.
 */
typedef struct ShadowSpillLeaseUseRecord {
    ShadowSpillBackendStream stream;
    ShadowSpillEventLease *event;
    struct ShadowSpillLeaseUseRecord *next;
    struct ShadowSpillLeaseUseRecord *ownership_next;
    struct ShadowSpillLeaseUseRecord *free_next;
} ShadowSpillLeaseUseRecord;

typedef enum ShadowSpillMemoryPlacement {
    SHADOWSPILL_MEMORY_FIRST_FIT = 0,
    SHADOWSPILL_MEMORY_BEST_FIT_LOW = 1,
    SHADOWSPILL_MEMORY_BEST_FIT_HIGH = 2,
} ShadowSpillMemoryPlacement;

enum {
    SHADOWSPILL_EXECUTION_POOL_ID = 0U,
    SHADOWSPILL_SPILL_POOL_ID = 1U,
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
    /* Reusable stream-use and retirement-requirement records. */
    ShadowSpillLeaseUseRecord *owned_use_records;
    ShadowSpillLeaseUseRecord *free_use_records;
    struct ShadowSpillMemoryLease *active_leases;
    struct ShadowSpillMemoryLease **leases_by_id;
    struct ShadowSpillMemoryLease **leases_by_pointer;
    struct ShadowSpillMemoryLease **reusable_leases_by_size;
    /* Cold-reserved workspace for prospective causal-release queries. */
    struct ShadowSpillMemoryLease **release_frontier_workspace;
    ShadowSpillRange *release_range_workspace;
    ShadowSpillMemoryPoolBackend backend;
    void *base;
    uint32_t pool_id;
    uint64_t minimum_alignment;
    uint64_t next_request_sequence;
    uint64_t next_release_sequence;
    uint64_t reserved_bytes;
    uint64_t allocation_index_bucket_count;
    uint64_t reusable_index_bucket_count;
    uint64_t release_frontier_capacity;
    uint64_t release_range_capacity;
    uint64_t requested_allocated_bytes;
    uint64_t peak_requested_allocated_bytes;
    uint64_t live_allocations;
    uint64_t blocked_allocators;
    uint64_t lease_record_capacity;
    uint64_t lease_record_available;
    uint64_t lease_record_in_use;
    uint64_t lease_record_peak_in_use;
    uint64_t lease_record_growth_rejections;
    uint64_t use_record_capacity;
    uint64_t use_record_available;
    uint64_t use_record_in_use;
    uint64_t use_record_peak_in_use;
    uint64_t use_record_growth_rejections;
    _Atomic uint64_t pending_retirements;
    _Atomic uint64_t pending_capacity_actions;
    _Atomic uint64_t free_bytes_snapshot;
    _Atomic uint64_t largest_free_bytes_snapshot;
    uint8_t initialized;
    uint8_t lease_records_sealed;
    uint8_t use_records_sealed;
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
    ShadowSpillLeaseUseRecord *uses;
    /* Queue-owned alias of ``uses`` while ordinary events are pending. */
    ShadowSpillLeaseUseRecord *retirement_requirements;
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

typedef struct ShadowSpillRetirementRecord {
    ShadowSpillMemoryLease *allocation;
    ShadowSpillMemoryPool *pool;
    struct ShadowSpillPlan *plan_owner;
    uint64_t allocation_id;
    uint64_t allocation_generation;
    /*
     * The queue owns these immutable requirements.  The allocation borrows
     * the same pointers while its generation remains retirement-pending so
     * the allocator can make causal-reuse decisions without copying them.
     */
    ShadowSpillLeaseUseRecord *requirements;
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

int shadowspill_range_clone_extended_with_nodes(
    const ShadowSpillRangeAllocator *source,
    uint64_t capacity,
    ShadowSpillRangeAllocator *destination,
    ShadowSpillRange *nodes,
    uint64_t node_capacity
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

ShadowSpillLeaseUseRecord *shadowspill_memory_pool_acquire_use_record_locked(
    ShadowSpillMemoryPool *pool
);

int shadowspill_memory_pool_release_use_records_locked(
    ShadowSpillMemoryPool *pool,
    ShadowSpillLeaseUseRecord *records
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

void shadowspill_publish_pool_geometry_locked(ShadowSpillMemoryPool *pool);

#endif
