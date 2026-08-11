#ifndef SHADOWSPILL_RUNTIME_INTERNAL_H
#define SHADOWSPILL_RUNTIME_INTERNAL_H

#include <pthread.h>
#include <stddef.h>
#include <stdint.h>

#include <shadowspill/runtime.h>

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

typedef struct ShadowSpillEventRecord {
    ShadowSpillBackendEvent event;
    struct ShadowSpillEventRecord *next;
} ShadowSpillEventRecord;

typedef struct ShadowSpillAllocationRecord {
    uint64_t allocation_id;
    uint64_t generation;
    uint64_t requested_bytes;
    uint64_t charged_bytes;
    uint64_t offset;
    uint64_t origin_task_id;
    uint64_t release_task_id;
    void *pointer;
    int logical_freed;
    int plan_owned;
    int ever_plan_owned;
    int framework_free_seen;
    uint64_t handoff_from_object_id;
    uint64_t handoff_to_object_id;
    uint64_t handoff_task_id;
    ShadowSpillStreamRecord *streams;
    ShadowSpillEventRecord *retirement_events;
    struct ShadowSpillAllocationRecord *next;
} ShadowSpillAllocationRecord;

typedef struct ShadowSpillObjectRecord {
    uint64_t object_id;
    uint64_t size_bytes;
    uint64_t generation;
    uint64_t authoritative_version;
    uint64_t device_version;
    uint64_t host_version;
    uint64_t allocation_id;
    uint64_t host_offset;
    uint8_t retain_host_backing;
    uint8_t host_current;
    uint8_t has_host_range;
    uint8_t residency;
    ShadowSpillBackendEvent readiness_event;
    uint8_t has_readiness_event;
    uint64_t retired_generation;
    void *retired_device_pointer;
    struct ShadowSpillObjectRecord *next;
} ShadowSpillObjectRecord;

typedef struct ShadowSpillTaskFence {
    ShadowSpillBackendEvent event;
    uint32_t references;
} ShadowSpillTaskFence;

typedef enum ShadowSpillQueuedActionState {
    SHADOWSPILL_ACTION_QUEUED = 0,
    SHADOWSPILL_ACTION_IN_FLIGHT = 1,
} ShadowSpillQueuedActionState;

typedef struct ShadowSpillQueuedAction {
    uint64_t task_id;
    uint8_t kind;
    uint8_t state;
    ShadowSpillObjectRecord *object;
    ShadowSpillTaskFence *fence;
    ShadowSpillBackendEvent completion_event;
    uint8_t has_completion_event;
    struct ShadowSpillQueuedAction *next;
} ShadowSpillQueuedAction;

struct ShadowSpillRuntime {
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    pthread_t progress_thread;
    int progress_started;
    int closing;
    int closed;
    int worker_stop;
    uint64_t progress_poll_nanoseconds;
    uint64_t minimum_alignment;

    ShadowSpillBackend backend;
    void *device_slab;
    void *host_arena;
    ShadowSpillBackendStream h2d_stream;
    ShadowSpillBackendStream d2h_stream;
    int h2d_stream_created;
    int d2h_stream_created;

    ShadowSpillRangeAllocator device_ranges;
    ShadowSpillRangeAllocator host_ranges;
    ShadowSpillAllocationRecord *allocations;
    ShadowSpillObjectRecord *objects;
    ShadowSpillQueuedAction *action_head;
    ShadowSpillQueuedAction *action_tail;

    uint64_t next_allocation_id;
    uint64_t next_generation;
    uint64_t requested_allocated_bytes;
    uint64_t peak_requested_allocated_bytes;
    uint64_t live_allocations;
    uint64_t blocked_allocators;
    uint64_t pending_retirements;
    uint64_t registered_objects;
    uint64_t queued_actions;
    uint64_t transfers_to_device;
    uint64_t transfers_to_host;
    uint64_t bytes_to_device;
    uint64_t bytes_to_host;
    uint64_t wait_events_inserted;
    ShadowSpillAllocationEvent *allocation_events;
    uint64_t allocation_event_count;
    uint64_t allocation_event_capacity;
    uint64_t next_allocation_event_sequence;
    int allocation_telemetry_active;
    int allocation_event_overflow;
    ShadowSpillRuntimeFailure failure;
};

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
ShadowSpillRuntimeStatus shadowspill_allocate_locked(
    ShadowSpillRuntime *runtime,
    uint64_t bytes,
    uint64_t alignment,
    int plan_owned,
    uint64_t origin_task_id,
    ShadowSpillAllocationRecord **record
);
void shadowspill_release_allocation_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillAllocationRecord *allocation
);
void shadowspill_latch_failure_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillRuntimeStatus status,
    uint64_t object_id,
    uint64_t allocation_id,
    uint64_t requested_bytes
);
ShadowSpillRuntimeStatus shadowspill_current_status_locked(
    ShadowSpillRuntime *runtime
);
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
int shadowspill_backend_is_valid(const ShadowSpillBackend *backend);
void *shadowspill_progress_main(void *pointer);

#endif
