#include "internal.h"

#include <stdint.h>

uint32_t shadowspill_runtime_abi_version(void) {
    return SHADOWSPILL_RUNTIME_ABI_VERSION;
}

const char *shadowspill_runtime_status_string(ShadowSpillRuntimeStatus status) {
    switch (status) {
        case SHADOWSPILL_RUNTIME_OK:
            return "ok";
        case SHADOWSPILL_RUNTIME_INVALID_ARGUMENT:
            return "invalid runtime argument";
        case SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE:
            return "runtime allocation failure";
        case SHADOWSPILL_RUNTIME_OUT_OF_MEMORY:
            return "device slab out of memory";
        case SHADOWSPILL_RUNTIME_NO_PROGRESS:
            return "allocation cannot make progress";
        case SHADOWSPILL_RUNTIME_INVALID_STATE:
            return "invalid runtime state";
        case SHADOWSPILL_RUNTIME_PLAN_VIOLATION:
            return "execution plan violation";
        case SHADOWSPILL_RUNTIME_BACKEND_FAILURE:
            return "backend failure";
        case SHADOWSPILL_RUNTIME_WORKER_FAILURE:
            return "progress worker failure";
        case SHADOWSPILL_RUNTIME_CLOSED:
            return "runtime is closed";
    }
    return "unknown runtime status";
}

void shadowspill_latch_failure_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillRuntimeStatus status,
    uint64_t object_id,
    uint64_t allocation_id,
    uint64_t requested_bytes
) {
    if (runtime->failure.status != SHADOWSPILL_RUNTIME_OK) {
        return;
    }
    runtime->failure = (ShadowSpillRuntimeFailure){
        .status = (uint32_t)status,
        .object_id = object_id,
        .allocation_id = allocation_id,
        .requested_bytes = requested_bytes,
        .free_bytes = shadowspill_range_free_bytes(&runtime->device_ranges),
        .largest_free_range_bytes =
            shadowspill_range_largest_free(&runtime->device_ranges),
    };
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
}

ShadowSpillRuntimeStatus shadowspill_current_status_locked(
    ShadowSpillRuntime *runtime
) {
    if (runtime->closed || runtime->closing) {
        return SHADOWSPILL_RUNTIME_CLOSED;
    }
    if (runtime->failure.status != SHADOWSPILL_RUNTIME_OK) {
        return (ShadowSpillRuntimeStatus)runtime->failure.status;
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
    pthread_mutex_lock(&runtime->mutex);
    *failure = runtime->failure;
    pthread_mutex_unlock(&runtime->mutex);
    return SHADOWSPILL_RUNTIME_OK;
}
