#include "internal.h"

#include <stdint.h>

static void latch_failure(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryPool *pool,
    ShadowSpillRuntimeStatus status,
    uint64_t task_id,
    uint64_t object_id,
    uint64_t allocation_id,
    uint64_t requested_bytes,
    uint64_t task_live_requested_bytes,
    uint64_t task_live_charged_bytes,
    uint64_t task_live_requested_limit_bytes,
    uint64_t task_live_charged_limit_bytes,
    uint64_t task_maximum_requested_allocation_bytes,
    uint64_t task_maximum_charged_allocation_bytes,
    const ShadowSpillTaskAllocationMismatch *allocation_mismatch
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
        .pool_id = pool == NULL ? UINT32_MAX : pool->pool_id,
        .task_id = task_id,
        .object_id = object_id,
        .allocation_id = allocation_id,
        .requested_bytes = requested_bytes,
        .free_bytes = pool == NULL ? 0U : atomic_load_explicit(
            &pool->free_bytes_snapshot, memory_order_acquire
        ),
        .largest_free_range_bytes = pool == NULL ? 0U : atomic_load_explicit(
            &pool->largest_free_bytes_snapshot, memory_order_acquire
        ),
        .task_live_requested_bytes = task_live_requested_bytes,
        .task_live_charged_bytes = task_live_charged_bytes,
        .task_live_requested_limit_bytes = task_live_requested_limit_bytes,
        .task_live_charged_limit_bytes = task_live_charged_limit_bytes,
        .task_maximum_requested_allocation_bytes =
            task_maximum_requested_allocation_bytes,
        .task_maximum_charged_allocation_bytes =
            task_maximum_charged_allocation_bytes,
        .task_allocation_operation_index = allocation_mismatch == NULL
            ? 0U : allocation_mismatch->operation_index,
        .task_allocation_expected_ordinal = allocation_mismatch == NULL
            ? SHADOWSPILL_RUNTIME_NO_ID : allocation_mismatch->expected_ordinal,
        .task_allocation_actual_ordinal = allocation_mismatch == NULL
            ? SHADOWSPILL_RUNTIME_NO_ID : allocation_mismatch->actual_ordinal,
        .task_allocation_expected_requested_bytes = allocation_mismatch == NULL
            ? 0U : allocation_mismatch->expected_requested_bytes,
        .task_allocation_actual_requested_bytes = allocation_mismatch == NULL
            ? 0U : allocation_mismatch->actual_requested_bytes,
        .task_allocation_expected_charged_bytes = allocation_mismatch == NULL
            ? 0U : allocation_mismatch->expected_charged_bytes,
        .task_allocation_actual_charged_bytes = allocation_mismatch == NULL
            ? 0U : allocation_mismatch->actual_charged_bytes,
        .task_allocation_expected_alignment_bytes = allocation_mismatch == NULL
            ? 0U : allocation_mismatch->expected_alignment_bytes,
        .task_allocation_actual_alignment_bytes = allocation_mismatch == NULL
            ? 0U : allocation_mismatch->actual_alignment_bytes,
        .task_allocation_expected_operation = allocation_mismatch == NULL
            ? UINT8_MAX : allocation_mismatch->expected_operation,
        .task_allocation_actual_operation = allocation_mismatch == NULL
            ? UINT8_MAX : allocation_mismatch->actual_operation,
    };
    atomic_store_explicit(
        &runtime->failure_status, (uint32_t)status, memory_order_release
    );
    pthread_mutex_unlock(&runtime->failure_lock);
    shadowspill_append_trace_event_locked(
        runtime,
        SHADOWSPILL_TRACE_FAILURE_LATCHED,
        task_id,
        object_id,
        allocation_id,
        requested_bytes,
        (uint64_t)status,
        runtime->failure.free_bytes
    );
    shadowspill_idle_notify(runtime);
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
        shadowspill_current_allocation_pool(runtime),
        status,
        shadowspill_current_task_id(runtime),
        object_id,
        allocation_id,
        requested_bytes,
        0U,
        0U,
        0U,
        0U,
        0U,
        0U,
        NULL
    );
}

void shadowspill_latch_pool_failure_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryPool *pool,
    ShadowSpillRuntimeStatus status,
    uint64_t object_id,
    uint64_t allocation_id,
    uint64_t requested_bytes
) {
    latch_failure(
        runtime,
        pool,
        status,
        shadowspill_current_task_id(runtime),
        object_id,
        allocation_id,
        requested_bytes,
        0U,
        0U,
        0U,
        0U,
        0U,
        0U,
        NULL
    );
}

void shadowspill_latch_task_failure(
    ShadowSpillRuntime *runtime,
    ShadowSpillRuntimeStatus status,
    uint64_t task_id,
    uint64_t object_id,
    uint64_t allocation_id,
    uint64_t requested_bytes
) {
    latch_failure(
        runtime,
        shadowspill_current_allocation_pool(runtime),
        status,
        task_id,
        object_id,
        allocation_id,
        requested_bytes,
        0U,
        0U,
        0U,
        0U,
        0U,
        0U,
        NULL
    );
}

void shadowspill_latch_task_envelope_failure(
    ShadowSpillRuntime *runtime,
    uint64_t requested_bytes,
    uint64_t charged_bytes,
    uint64_t live_requested_bytes,
    uint64_t live_charged_bytes,
    uint64_t live_requested_limit_bytes,
    uint64_t live_charged_limit_bytes,
    uint64_t maximum_requested_allocation_bytes,
    uint64_t maximum_charged_allocation_bytes
) {
    latch_failure(
        runtime,
        shadowspill_current_allocation_pool(runtime),
        SHADOWSPILL_RUNTIME_TASK_ALLOCATION_ENVELOPE_EXCEEDED,
        shadowspill_current_task_id(runtime),
        SHADOWSPILL_RUNTIME_NO_ID,
        SHADOWSPILL_RUNTIME_NO_ID,
        requested_bytes,
        live_requested_bytes,
        live_charged_bytes,
        live_requested_limit_bytes,
        live_charged_limit_bytes,
        maximum_requested_allocation_bytes,
        maximum_charged_allocation_bytes,
        NULL
    );
    (void)charged_bytes;
}

void shadowspill_latch_task_allocation_contract_failure(
    ShadowSpillRuntime *runtime,
    const ShadowSpillTaskAllocationMismatch *mismatch
) {
    if (runtime == NULL || mismatch == NULL) {
        return;
    }
    latch_failure(
        runtime,
        shadowspill_current_allocation_pool(runtime),
        SHADOWSPILL_RUNTIME_TASK_ALLOCATION_CONTRACT_MISMATCH,
        shadowspill_current_task_id(runtime),
        SHADOWSPILL_RUNTIME_NO_ID,
        SHADOWSPILL_RUNTIME_NO_ID,
        mismatch->actual_requested_bytes,
        0U,
        0U,
        0U,
        0U,
        0U,
        0U,
        mismatch
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
    pthread_mutex_lock(&runtime->failure_lock);
    const uint32_t failure_pool_id = runtime->failure.pool_id;
    pthread_mutex_unlock(&runtime->failure_lock);
    ShadowSpillMemoryPool *pool = shadowspill_runtime_pool(
        runtime, failure_pool_id
    );
    if (pool == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    shadowspill_memory_pool_lock_foreground(pool);
    pthread_mutex_lock(&runtime->failure_lock);
    const ShadowSpillRuntimeStatus failure = shadowspill_failure_status(runtime);
    ShadowSpillRuntimeStatus status = failure;
    if (failure == SHADOWSPILL_RUNTIME_OK) {
        status = SHADOWSPILL_RUNTIME_OK;
    } else if (failure != SHADOWSPILL_RUNTIME_NO_PROGRESS) {
        status = failure;
    } else if (pool->blocked_allocators != 0U) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
    } else {
        /*
         * Never clear the latch while a logically freed execution lease has
         * no published retirement record. A fence without a queued record is
         * not a progress source either. The task boundary or abort path must
         * publish the complete causal retirement before recovery.
         */
        for (const ShadowSpillMemoryLease *allocation =
                 pool->active_leases;
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
            .pool_id = UINT32_MAX,
            .task_id = SHADOWSPILL_RUNTIME_NO_ID,
            .object_id = SHADOWSPILL_RUNTIME_NO_ID,
            .allocation_id = SHADOWSPILL_RUNTIME_NO_ID,
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
        shadowspill_idle_notify(runtime);
        shadowspill_notify_worker(runtime);
    }
    return status;
}
