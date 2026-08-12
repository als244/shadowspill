#include "internal.h"

#include <stdint.h>

void shadowspill_latch_failure_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillRuntimeStatus status,
    uint64_t object_id,
    uint64_t allocation_id,
    uint64_t requested_bytes
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
        .object_id = object_id,
        .allocation_id = allocation_id,
        .requested_bytes = requested_bytes,
        .free_bytes = atomic_load_explicit(
            &runtime->device_free_bytes_snapshot, memory_order_acquire
        ),
        .largest_free_range_bytes = atomic_load_explicit(
            &runtime->device_largest_free_snapshot, memory_order_acquire
        ),
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
    if (runtime->device_pool.initialized) {
        pthread_cond_broadcast(&runtime->device_pool.capacity_changed);
    }
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
