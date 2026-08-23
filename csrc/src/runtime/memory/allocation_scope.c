#include "../internal.h"

ShadowSpillStatus shadowspill_allocation_scope_begin(
    ShadowSpillRuntime *runtime,
    uint32_t pool_id,
    uint64_t scope_id
) {
    ShadowSpillMemoryPool *pool = shadowspill_runtime_pool(runtime, pool_id);
    if (pool == NULL || scope_id == SHADOWSPILL_RUNTIME_NO_ID) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    const ShadowSpillStatus status =
        shadowspill_failure_status(runtime);
    if (status != SHADOWSPILL_STATUS_OK) {
        return status;
    }
    return shadowspill_enter_allocation_scope(runtime, pool, scope_id) == 0
        ? SHADOWSPILL_STATUS_OK
        : SHADOWSPILL_STATUS_INVALID_STATE;
}

ShadowSpillStatus shadowspill_allocation_scope_end(
    ShadowSpillRuntime *runtime,
    uint64_t scope_id,
    ShadowSpillBackendStream stream
) {
    if (runtime == NULL || scope_id == SHADOWSPILL_RUNTIME_NO_ID) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    if (shadowspill_current_task_id(runtime) != scope_id ||
        shadowspill_current_plan(runtime) != NULL) {
        return SHADOWSPILL_STATUS_INVALID_STATE;
    }

    ShadowSpillStatus status = shadowspill_failure_status(runtime);
    ShadowSpillMemoryPool *pool = shadowspill_current_allocation_pool(runtime);
    if (pool == NULL) {
        return SHADOWSPILL_STATUS_INVALID_STATE;
    }
    const ShadowSpillStatus retirement_status =
        shadowspill_publish_task_retirement_event(
            runtime, scope_id, stream
        );
    if (retirement_status != SHADOWSPILL_STATUS_OK) {
        shadowspill_latch_task_failure(
            runtime,
            retirement_status,
            scope_id,
            SHADOWSPILL_RUNTIME_NO_ID,
            SHADOWSPILL_RUNTIME_NO_ID,
            0U
        );
        if (status == SHADOWSPILL_STATUS_OK) {
            status = retirement_status;
        }
    }
    shadowspill_leave_task_scope(runtime);
    return status;
}

void shadowspill_allocation_scope_abort(ShadowSpillRuntime *runtime) {
    if (runtime == NULL || shadowspill_current_plan(runtime) != NULL) {
        return;
    }
    const uint64_t scope_id = shadowspill_current_task_id(runtime);
    if (scope_id == SHADOWSPILL_RUNTIME_NO_ID) {
        return;
    }
    shadowspill_finalize_aborted_task_retirements(
        runtime, scope_id
    );
    shadowspill_leave_task_scope(runtime);
}
