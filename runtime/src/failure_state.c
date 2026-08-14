#include "internal.h"

#include <stdint.h>

static void latch_failure(
    ShadowSpillRuntime *runtime,
    ShadowSpillRuntimeStatus status,
    uint64_t object_id,
    uint64_t allocation_id,
    uint64_t requested_bytes,
    uint64_t allocation_ordinal,
    uint64_t expected_allocation_ordinal,
    uint64_t expected_requested_bytes
) {
    pthread_mutex_lock(&runtime->failure_lock);
    if (atomic_load_explicit(
            &runtime->failure_status, memory_order_acquire
        ) != SHADOWSPILL_RUNTIME_OK) {
        pthread_mutex_unlock(&runtime->failure_lock);
        return;
    }
    runtime->failure = (ShadowSpillRuntimeFailure){
        .status = (uint32_t)status,
        .task_id = shadowspill_current_task_id(runtime),
        .object_id = object_id,
        .allocation_id = allocation_id,
        .requested_bytes = requested_bytes,
        .free_bytes = atomic_load_explicit(
            &runtime->execution_free_bytes_snapshot, memory_order_acquire
        ),
        .largest_free_range_bytes = atomic_load_explicit(
            &runtime->execution_largest_free_snapshot, memory_order_acquire
        ),
        .allocation_ordinal = allocation_ordinal,
        .expected_allocation_ordinal = expected_allocation_ordinal,
        .expected_requested_bytes = expected_requested_bytes,
    };
    atomic_store_explicit(
        &runtime->failure_status, (uint32_t)status, memory_order_release
    );
    pthread_mutex_unlock(&runtime->failure_lock);
    shadowspill_append_trace_event_locked(
        runtime,
        SHADOWSPILL_TRACE_FAILURE_LATCHED,
        shadowspill_current_task_id(runtime),
        object_id,
        allocation_id,
        requested_bytes,
        (uint64_t)status,
        runtime->failure.free_bytes
    );
    pthread_cond_broadcast(&runtime->condition);
    shadowspill_idle_notify(runtime);
    if (shadowspill_execution_pool(runtime)->initialized) {
        pthread_cond_broadcast(
            &shadowspill_execution_pool(runtime)->capacity_changed
        );
    }
}

void shadowspill_latch_failure_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillRuntimeStatus status,
    uint64_t object_id,
    uint64_t allocation_id,
    uint64_t requested_bytes
) {
    latch_failure(
        runtime,
        status,
        object_id,
        allocation_id,
        requested_bytes,
        SHADOWSPILL_RUNTIME_NO_ID,
        SHADOWSPILL_RUNTIME_NO_ID,
        0U
    );
}

void shadowspill_latch_placement_failure(
    ShadowSpillRuntime *runtime,
    uint64_t requested_bytes,
    uint64_t allocation_ordinal,
    uint64_t expected_allocation_ordinal,
    uint64_t expected_requested_bytes
) {
    latch_failure(
        runtime,
        SHADOWSPILL_RUNTIME_PLAN_VIOLATION,
        SHADOWSPILL_RUNTIME_NO_ID,
        SHADOWSPILL_RUNTIME_NO_ID,
        requested_bytes,
        allocation_ordinal,
        expected_allocation_ordinal,
        expected_requested_bytes
    );
}

ShadowSpillRuntimeStatus shadowspill_failure_status(
    const ShadowSpillRuntime *runtime
) {
    return (ShadowSpillRuntimeStatus)atomic_load_explicit(
        &runtime->failure_status, memory_order_acquire
    );
}

ShadowSpillRuntimeStatus shadowspill_current_status_locked(
    ShadowSpillRuntime *runtime
) {
    if (atomic_load_explicit(&runtime->closed, memory_order_acquire) != 0U ||
        atomic_load_explicit(&runtime->closing, memory_order_acquire) != 0U) {
        return SHADOWSPILL_RUNTIME_CLOSED;
    }
    const ShadowSpillRuntimeStatus failure = shadowspill_failure_status(runtime);
    if (failure != SHADOWSPILL_RUNTIME_OK) {
        return failure;
    }
    return SHADOWSPILL_RUNTIME_OK;
}

ShadowSpillRuntimeStatus shadowspill_runtime_failure(
    ShadowSpillRuntime *runtime,
    ShadowSpillRuntimeFailure *failure
) {
    if (runtime == NULL || failure == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&runtime->failure_lock);
    *failure = runtime->failure;
    failure->status = (uint32_t)shadowspill_failure_status(runtime);
    pthread_mutex_unlock(&runtime->failure_lock);
    return SHADOWSPILL_RUNTIME_OK;
}

ShadowSpillRuntimeStatus shadowspill_runtime_recover_no_progress(
    ShadowSpillRuntime *runtime
) {
    if (runtime == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    ShadowSpillMemoryPool *pool = shadowspill_execution_pool(runtime);
    shadowspill_memory_pool_lock_foreground(pool);
    pthread_mutex_lock(&runtime->failure_lock);
    const ShadowSpillRuntimeStatus failure = shadowspill_failure_status(runtime);
    ShadowSpillRuntimeStatus status = failure;
    if (failure == SHADOWSPILL_RUNTIME_OK) {
        status = SHADOWSPILL_RUNTIME_OK;
    } else if (failure != SHADOWSPILL_RUNTIME_NO_PROGRESS) {
        status = failure;
    } else if (runtime->blocked_allocators != 0U) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
    } else {
        /*
         * Never clear the latch while a logically freed execution lease has
         * no published retirement record. A fence without a queued record is
         * not a progress source either. The task boundary or abort path must
         * publish the complete causal retirement before recovery.
         */
        for (const ShadowSpillMemoryLease *allocation =
                 runtime->active_execution_leases;
             allocation != NULL; allocation = allocation->active_next) {
            if (allocation->logical_freed && allocation->pointer != NULL &&
                !allocation->retirement_preparing &&
                allocation->retirement_enqueued_generation !=
                    allocation->generation) {
                status = SHADOWSPILL_RUNTIME_INVALID_STATE;
                break;
            }
        }
    }
    if (status == SHADOWSPILL_RUNTIME_NO_PROGRESS) {
        runtime->failure = (ShadowSpillRuntimeFailure){
            .status = SHADOWSPILL_RUNTIME_OK,
            .task_id = SHADOWSPILL_RUNTIME_NO_ID,
            .object_id = SHADOWSPILL_RUNTIME_NO_ID,
            .allocation_id = SHADOWSPILL_RUNTIME_NO_ID,
            .allocation_ordinal = SHADOWSPILL_RUNTIME_NO_ID,
            .expected_allocation_ordinal = SHADOWSPILL_RUNTIME_NO_ID,
        };
        atomic_store_explicit(
            &runtime->failure_status,
            SHADOWSPILL_RUNTIME_OK,
            memory_order_release
        );
        status = SHADOWSPILL_RUNTIME_OK;
    }
    pthread_mutex_unlock(&runtime->failure_lock);
    shadowspill_memory_pool_unlock_foreground(pool);
    if (status == SHADOWSPILL_RUNTIME_OK) {
        pthread_cond_broadcast(&runtime->condition);
        pthread_cond_broadcast(&pool->capacity_changed);
        shadowspill_idle_notify(runtime);
        shadowspill_notify_worker(runtime);
    }
    return status;
}
