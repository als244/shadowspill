/* The allocation callback, and the pointer queries beside it. */
#include "internal.h"

/*
 * What one allocation asks for. The four attempts at a lease below each read
 * the same seven values, so they are named once here rather than threaded
 * through four calls.
 */
typedef struct TaskAllocationRequest {
    uint64_t bytes;
    uint64_t alignment;
    ShadowSpillBackendStream stream;
    ShadowSpillAllocationOrigin origin;
    ShadowSpillPlan *plan;
    const ShadowSpillFixedPlacementDescription *fixed_placement;
    ShadowSpillMemoryPlacement dynamic_placement;
} TaskAllocationRequest;

/*
 * A lease for one request, in the order the pool prefers to give one: the
 * fixed placement the plan named, a pending lease of this task's that can be
 * recycled on the same stream, a fresh range, and -- only when the pool is
 * out -- a pending lease from any task. Called with the pool's foreground
 * lock held, and returns with it held.
 */
static ShadowSpillStatus take_lease_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryPool *pool,
    const TaskAllocationRequest *request,
    ShadowSpillMemoryLease **record
) {
    ShadowSpillStatus status = SHADOWSPILL_STATUS_OK;
    if (request->fixed_placement != NULL) {
        status = shadowspill_create_fixed_execution_lease_locked(
            request->plan,
            request->fixed_placement,
            0,
            request->origin,
            record
        );
    }
    /*
     * A logically freed exact-size lease from this task is immediately
     * reusable on the same stream: stream order already places the new
     * consumer after the prior use.  Recycle it before consuming another
     * slab range so allocation-heavy compiled tasks retain caching-
     * allocator behavior without fixed offsets or backend events.
     */
    if (request->fixed_placement == NULL) {
        status = shadowspill_allocations_reuse_pending_allocation_locked(
            runtime,
            pool,
            request->bytes,
            request->alignment,
            request->stream,
            request->origin,
            1,
            record
        );
    }
    if (request->fixed_placement == NULL &&
        status == SHADOWSPILL_STATUS_OK && *record == NULL) {
        status = shadowspill_create_lease_locked(
            runtime,
            pool,
            request->bytes,
            request->alignment,
            0,
            request->dynamic_placement,
            request->origin,
            record
        );
    }
    if (request->fixed_placement == NULL &&
        status == SHADOWSPILL_STATUS_OUT_OF_MEMORY) {
        status = shadowspill_allocations_reuse_pending_allocation_locked(
            runtime,
            pool,
            request->bytes,
            request->alignment,
            request->stream,
            request->origin,
            0,
            record
        );
        if (status == SHADOWSPILL_STATUS_OK && *record == NULL) {
            status = SHADOWSPILL_STATUS_OUT_OF_MEMORY;
        }
    }
    return status;
}

/*
 * Wait for capacity this pool was told to expect, then take the lock again.
 * Called with the pool's foreground lock held; the lock is dropped for the
 * wait and held again on return.
 */
static ShadowSpillStatus wait_for_capacity_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryPool *pool,
    uint64_t task_id,
    uint64_t bytes,
    ShadowSpillBackendStream stream
) {
    ShadowSpillStatus status = SHADOWSPILL_STATUS_OK;
    const uint64_t capacity_epoch = atomic_load_explicit(
        &pool->capacity_epoch, memory_order_acquire
    );
    shadowspill_append_trace_event_locked(
        runtime,
        SHADOWSPILL_TRACE_ALLOCATION_WAIT_BEGIN,
        task_id,
        SHADOWSPILL_RUNTIME_NO_ID,
        SHADOWSPILL_RUNTIME_NO_ID,
        bytes,
        shadowspill_memory_pool_free_bytes_locked(pool),
        shadowspill_memory_pool_largest_free_locked(pool)
    );
    ++pool->blocked_allocators;
    shadowspill_memory_pool_unlock_foreground(pool);
    status = SHADOWSPILL_STATUS_OK;
    if (task_id != SHADOWSPILL_RUNTIME_NO_ID) {
        status = shadowspill_publish_task_retirement_event(
            runtime, task_id, stream
        );
        if (status != SHADOWSPILL_STATUS_OK) {
            shadowspill_latch_failure_locked(
                runtime,
                status,
                SHADOWSPILL_FAILURE_REASON_RETIREMENT_PUBLICATION_REJECTED,
                SHADOWSPILL_RUNTIME_NO_ID,
                SHADOWSPILL_RUNTIME_NO_ID,
                bytes
            );
        }
    }
    /*
     * Wait for the capacity this pool was told to expect. The epoch is
     * what says capacity moved, but it is not what says capacity is
     * still coming: a retirement can drain without freeing a range, and
     * then the epoch never moves again. Waiting on the epoch alone is
     * therefore a wait for an event that has already happened, and the
     * only exits left were a runtime failure and a closing worker --
     * neither of which an allocation failure raises. So the release
     * source is re-read every turn, and the loop gives up the moment
     * there is nothing left to wait for. The retry above then finds the
     * pool still full with no release source and latches NO_PROGRESS,
     * which is the answer this wait exists to reach.
     */
    while (status == SHADOWSPILL_STATUS_OK && atomic_load_explicit(
               &pool->capacity_epoch, memory_order_acquire
           ) == capacity_epoch && shadowspill_memory_pool_has_release_source(pool)) {
        status = shadowspill_failure_status(runtime);
        if (status == SHADOWSPILL_STATUS_OK && atomic_load_explicit(
                &runtime->worker_stop, memory_order_acquire
            ) != 0U) {
            status = SHADOWSPILL_STATUS_CLOSED;
        }
        shadowspill_cpu_relax();
    }
    shadowspill_memory_pool_lock_foreground(pool);
    --pool->blocked_allocators;
    shadowspill_append_trace_event_locked(
        runtime,
        SHADOWSPILL_TRACE_ALLOCATION_WAIT_END,
        task_id,
        SHADOWSPILL_RUNTIME_NO_ID,
        SHADOWSPILL_RUNTIME_NO_ID,
        bytes,
        shadowspill_memory_pool_free_bytes_locked(pool),
        shadowspill_memory_pool_largest_free_locked(pool)
    );
    return status;
}

ShadowSpillStatus shadowspill_memory_pool_allocate(
    ShadowSpillRuntime *runtime,
    uint32_t pool_id,
    uint64_t bytes,
    uint64_t alignment,
    ShadowSpillBackendStream stream,
    ShadowSpillAllocation *allocation
) {
    ShadowSpillMemoryPool *pool = shadowspill_runtime_pool(runtime, pool_id);
    if (pool == NULL || allocation == NULL || alignment == 0U) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    ShadowSpillMemoryPool *scope_pool =
        shadowspill_current_allocation_pool(runtime);
    if (scope_pool != NULL && scope_pool != pool) {
        return SHADOWSPILL_STATUS_INVALID_STATE;
    }
    if (alignment < pool->minimum_alignment) {
        alignment = pool->minimum_alignment;
    }
    const uint64_t charged_bytes = bytes == 0U ? 0U : bytes;
    ShadowSpillStatus status = bytes == 0U
        ? SHADOWSPILL_STATUS_OK
        : shadowspill_validate_task_allocation(
            runtime, bytes, charged_bytes, alignment
        );
    if (status != SHADOWSPILL_STATUS_OK) {
        return status;
    }
    const uint64_t task_id = shadowspill_current_task_id(runtime);
    const uint64_t allocation_ordinal =
        shadowspill_current_task_invariant_allocation_ordinal(runtime);
    const int allocation_is_scratch =
        shadowspill_current_task_allocation_is_scratch(runtime);
    const uint64_t task_invocation =
        shadowspill_current_task_invocation(runtime);
    /* Here the work is the running thread's own scope, so the scope is the
     * right source for both halves. */
    const ShadowSpillAllocationOrigin origin = {
        .plan_id = shadowspill_current_plan_id(runtime),
        .task_id = task_id,
    };
    ShadowSpillPlan *plan = shadowspill_current_plan(runtime);
    if (plan != NULL && plan->execution_pool != pool) {
        return SHADOWSPILL_STATUS_INVALID_STATE;
    }
    /*
     * Keep large, short-lived framework values at the high end of an
     * unsealed pool while small provider state grows from the low end.  A
     * provider cache retained by an isolated profiling task then occupies a
     * compact prefix instead of pinning a tiny range behind a multi-gigabyte
     * representative input.  Fixed-layout allocations bypass this policy.
     */
    const ShadowSpillMemoryPlacement dynamic_placement =
        bytes >= (UINT64_C(64) << 20U)
        ? SHADOWSPILL_MEMORY_BEST_FIT_HIGH
        : SHADOWSPILL_MEMORY_BEST_FIT_LOW;
    const ShadowSpillFixedPlacementDescription *fixed_placement =
        shadowspill_fixed_layout_find_placement(
            plan,
            SHADOWSPILL_FIXED_TASK_ALLOCATION,
            task_id,
            allocation_ordinal,
            SHADOWSPILL_RUNTIME_NO_ID
        );
    if (fixed_placement != NULL) {
        status = shadowspill_fixed_layout_wait_for_dependencies(
            plan,
            SHADOWSPILL_FIXED_TASK_ALLOCATION,
            task_id,
            allocation_ordinal,
            task_invocation,
            stream
        );
        if (status != SHADOWSPILL_STATUS_OK) {
            shadowspill_latch_task_failure(
                runtime,
                status,
                SHADOWSPILL_FAILURE_REASON_TASK_ALLOCATION_REJECTED,
                task_id,
                SHADOWSPILL_RUNTIME_NO_ID,
                SHADOWSPILL_RUNTIME_NO_ID,
                bytes
            );
            return status;
        }
    }
    const TaskAllocationRequest request = {
        .bytes = bytes,
        .alignment = alignment,
        .stream = stream,
        .origin = origin,
        .plan = plan,
        .fixed_placement = fixed_placement,
        .dynamic_placement = dynamic_placement,
    };
    shadowspill_memory_pool_lock_foreground(pool);
    status = shadowspill_current_status_locked(runtime);
    while (status == SHADOWSPILL_STATUS_OK) {
        ShadowSpillMemoryLease *record = NULL;
        status = take_lease_locked(runtime, pool, &request, &record);
        if (status == SHADOWSPILL_STATUS_OK) {
            *allocation = (ShadowSpillAllocation){
                .pool_id = pool_id,
                .allocation_id = record->allocation_id,
                .generation = record->generation,
                .requested_bytes = record->requested_bytes,
                .charged_bytes = record->charged_bytes,
                .pointer = record->pointer,
            };
            record->origin_task_allocation_sequence =
                shadowspill_commit_task_allocation(
                    runtime, record->requested_bytes, record->charged_bytes
                );
            record->origin_task_allocation_ordinal = allocation_ordinal;
            record->origin_task_allocation_is_scratch =
                allocation_is_scratch ? 1U : 0U;
            record->origin_task_invocation = task_invocation;
            break;
        }
        if (status != SHADOWSPILL_STATUS_OUT_OF_MEMORY) {
            break;
        }
        if (!shadowspill_memory_pool_has_release_source(pool)) {
            shadowspill_latch_pool_failure_locked(
                runtime,
                pool,
                SHADOWSPILL_STATUS_NO_PROGRESS,
                SHADOWSPILL_FAILURE_REASON_POOL_EXHAUSTED,
                SHADOWSPILL_RUNTIME_NO_ID,
                SHADOWSPILL_RUNTIME_NO_ID,
                bytes
            );
            status = SHADOWSPILL_STATUS_NO_PROGRESS;
            break;
        }
        status = wait_for_capacity_locked(runtime, pool, task_id, bytes, stream);
        if (status == SHADOWSPILL_STATUS_OK) {
            status = shadowspill_current_status_locked(runtime);
        }
    }
    shadowspill_memory_pool_unlock_foreground(pool);
    return status;
}

ShadowSpillStatus shadowspill_memory_pool_allocation_for_pointer(
    ShadowSpillRuntime *runtime,
    uint32_t pool_id,
    const void *pointer,
    ShadowSpillAllocation *allocation
) {
    ShadowSpillMemoryPool *pool = shadowspill_runtime_pool(runtime, pool_id);
    if (pool == NULL || pointer == NULL || allocation == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    shadowspill_memory_pool_lock_foreground(pool);
    ShadowSpillMemoryLease *record =
        shadowspill_find_lease_by_pointer(pool, pointer);
    if (record == NULL) {
        shadowspill_memory_pool_unlock_foreground(pool);
        return SHADOWSPILL_STATUS_INVALID_STATE;
    }
    *allocation = (ShadowSpillAllocation){
        .pool_id = pool_id,
        .allocation_id = record->allocation_id,
        .generation = record->generation,
        .requested_bytes = record->requested_bytes,
        .charged_bytes = record->charged_bytes,
        .pointer = record->pointer,
    };
    shadowspill_memory_pool_unlock_foreground(pool);
    return SHADOWSPILL_STATUS_OK;
}

int shadowspill_allocations_append_lease_use_locked(
    ShadowSpillMemoryLease *allocation,
    ShadowSpillBackendStream stream
) {
    for (ShadowSpillLeaseUseRecord *item = allocation->uses; item != NULL;
         item = item->next) {
        if (shadowspill_allocations_stream_equal(item->stream, stream)) {
            return 0;
        }
    }
    ShadowSpillLeaseUseRecord *created =
        shadowspill_memory_pool_acquire_use_record_locked(allocation->pool);
    if (created == NULL) {
        return -1;
    }
    created->stream = stream;
    created->next = allocation->uses;
    allocation->uses = created;
    return 0;
}

ShadowSpillStatus shadowspill_memory_pool_record_stream(
    ShadowSpillRuntime *runtime,
    uint32_t pool_id,
    uint64_t allocation_id,
    ShadowSpillBackendStream stream
) {
    ShadowSpillMemoryPool *pool = shadowspill_runtime_pool(runtime, pool_id);
    if (pool == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    shadowspill_memory_pool_lock_foreground(pool);
    ShadowSpillMemoryLease *allocation = shadowspill_find_lease(
        pool, allocation_id
    );
    ShadowSpillStatus status = SHADOWSPILL_STATUS_OK;
    if (allocation == NULL || allocation->logical_freed) {
        status = SHADOWSPILL_STATUS_INVALID_STATE;
    } else if (shadowspill_allocations_append_lease_use_locked(
                   allocation, stream
               ) != 0) {
        status = SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    }
    shadowspill_memory_pool_unlock_foreground(pool);
    return status;
}
