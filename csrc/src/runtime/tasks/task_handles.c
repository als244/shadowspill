/* The handle a caller holds across one task, and its boundaries. */
#include "../internal.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

ShadowSpillStatus shadowspill_before_task_handle(
    ShadowSpillRuntime *runtime,
    const ShadowSpillTaskHandle *handle,
    ShadowSpillBackendStream compute_stream,
    const ShadowSpillObjectBinding **bindings,
    uint32_t *binding_count
) {
    const ShadowSpillTaskRecord *record = handle;
    if (runtime == NULL || record == NULL ||
        record->plan_owner == NULL || record->plan_owner->runtime != runtime ||
        record->boundary_kind != SHADOWSPILL_BOUNDARY_TASK) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    if (bindings == NULL || binding_count == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    *bindings = NULL;
    *binding_count = 0U;
    if (shadowspill_claim_task_invocation(record) != 0) {
        return SHADOWSPILL_STATUS_INVALID_STATE;
    }

    shadowspill_append_trace_event_locked(
        runtime,
        SHADOWSPILL_TRACE_BEFORE_TASK,
        record->task_id,
        SHADOWSPILL_RUNTIME_NO_ID,
        SHADOWSPILL_RUNTIME_NO_ID,
        0U,
        record->input_count,
        atomic_load_explicit(&runtime->actions.count, memory_order_acquire)
    );
    ShadowSpillStatus status = shadowspill_acquire_object_bindings(
        runtime,
        record->plan_owner,
        record->task_id,
        record->unique_inputs,
        record->unique_input_count,
        record->input_unique_indices,
        record->unique_first_positions,
        record->input_count,
        compute_stream,
        record->input_bindings,
        record->input_count
    );
    if (status == SHADOWSPILL_STATUS_OK &&
        shadowspill_enter_claimed_task_scope(runtime, record) != 0) {
        status = SHADOWSPILL_STATUS_INVALID_STATE;
    }
    if (status != SHADOWSPILL_STATUS_OK) {
        shadowspill_release_task_invocation(record);
        return status;
    }
    *bindings = record->input_bindings;
    *binding_count = record->input_count;
    return status;
}

ShadowSpillStatus shadowspill_wait_task_allocations_handle(
    ShadowSpillRuntime *runtime,
    const ShadowSpillTaskHandle *handle,
    ShadowSpillBackendStream compute_stream
) {
    const ShadowSpillTaskRecord *record = handle;
    if (runtime == NULL || record == NULL ||
        record->plan_owner == NULL || record->plan_owner->runtime != runtime ||
        record->boundary_kind != SHADOWSPILL_BOUNDARY_TASK) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    /*
     * Every allocation this task will make reuses a range the plan already
     * assigned it.  Resolving those here, while the boundary still owns the
     * interval, keeps the wait out of the task's own compute span.  The
     * allocator keeps its call, which then finds the dependency published.
     */
    return shadowspill_fixed_layout_wait_for_task_allocations(
        record->plan_owner,
        record->task_id,
        shadowspill_current_task_invocation(runtime),
        compute_stream
    );
}

ShadowSpillStatus shadowspill_after_task_handle(
    ShadowSpillRuntime *runtime,
    const ShadowSpillTaskHandle *handle,
    ShadowSpillBackendStream compute_stream
) {
    const ShadowSpillTaskRecord *record = handle;
    if (runtime == NULL || record == NULL ||
        record->plan_owner == NULL || record->plan_owner->runtime != runtime ||
        record->boundary_kind != SHADOWSPILL_BOUNDARY_TASK) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    return shadowspill_after_task_record(runtime, record, compute_stream);
}

ShadowSpillStatus shadowspill_submit_action_batch_handle(
    ShadowSpillRuntime *runtime,
    const ShadowSpillActionBatchHandle *handle,
    ShadowSpillBackendStream trigger_stream
) {
    const ShadowSpillTaskRecord *record = handle;
    if (runtime == NULL || record == NULL || record->plan_owner == NULL ||
        record->plan_owner->runtime != runtime ||
        record->boundary_kind != SHADOWSPILL_BOUNDARY_ACTION_BATCH) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    if (shadowspill_enter_task_scope(runtime, record) != 0) {
        return SHADOWSPILL_STATUS_INVALID_STATE;
    }
    return shadowspill_after_task_record(
        runtime, record, trigger_stream
    );
}
