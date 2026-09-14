#include "../internal.h"

ShadowSpillStatus shadowspill_allocation_scope_begin(
    ShadowSpillRuntime *runtime,
    uint32_t pool_id,
    uint64_t plan_id,
    uint64_t scope_id
) {
    ShadowSpillMemoryPool *pool = shadowspill_runtime_pool(runtime, pool_id);
    /* A scope must name its plan. Allowing it not to would leave allocations
     * made here attributable to nothing, which is the case this id exists to
     * remove. */
    if (pool == NULL || plan_id == 0U ||
        scope_id == SHADOWSPILL_RUNTIME_NO_ID) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    const ShadowSpillStatus status =
        shadowspill_failure_status(runtime);
    if (status != SHADOWSPILL_STATUS_OK) {
        return status;
    }
    /*
     * The id must be one this runtime issued. It need not name a plan that
     * exists: profiling legitimately runs before a plan is created, and in a
     * standalone measurement no plan is created at all. What matters for
     * attribution is that the id came from the minter, so it is unique for the
     * life of the runtime and cannot collide with another plan's.
     */
    if (plan_id >=
        atomic_load_explicit(&runtime->next_plan_id, memory_order_acquire)) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    return shadowspill_enter_allocation_scope(
               runtime, pool, plan_id, scope_id
           ) == 0
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
            SHADOWSPILL_FAILURE_REASON_RETIREMENT_PUBLICATION_REJECTED,
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
