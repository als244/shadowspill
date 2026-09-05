#include "internal.h"

#include <stdatomic.h>

ShadowSpillStatus shadowspill_pytorch_submit_action_batch_handle(
    uintptr_t action_batch_handle,
    uintptr_t trigger_stream_address
) {
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    if (runtime == NULL) {
        return SHADOWSPILL_STATUS_CLOSED;
    }
    if (action_batch_handle == 0U) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    const ShadowSpillProfilerRange range =
        shadowspill_pytorch_profile_range_begin(
            "shadowspill.pytorch.initial_actions"
        );
    const ShadowSpillStatus status =
        shadowspill_submit_action_batch_handle(
            runtime,
            (const ShadowSpillActionBatchHandle *)action_batch_handle,
            adapter_stream(trigger_stream_address)
        );
    shadowspill_pytorch_profile_range_end(range);
    return status;
}

ShadowSpillStatus shadowspill_pytorch_before_task_handle(
    uintptr_t task_handle,
    uintptr_t compute_stream_address,
    const ShadowSpillObjectBinding **bindings,
    uint32_t *binding_count
) {
    const ShadowSpillTaskHandle *handle =
        (const ShadowSpillTaskHandle *)task_handle;
    const uint64_t task_id = shadowspill_task_id(handle);
    if (task_range_active || task_handle == 0U ||
        task_id == SHADOWSPILL_RUNTIME_NO_ID) {
        return SHADOWSPILL_STATUS_INVALID_STATE;
    }
    if (atomic_load_explicit(
            &adapter.profiler_annotations_enabled, memory_order_relaxed
        ) != 0U) {
        char range_name[384];
        shadowspill_pytorch_format_task_range_name(range_name, sizeof(range_name), "task", handle);
        task_range_id = shadowspill_pytorch_profile_range_begin(range_name);
    } else {
        task_range_id = 0U;
    }
    task_range_active = 1;
    active_task_label = shadowspill_task_trace_label(handle);
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    ShadowSpillStatus status = runtime == NULL
        ? SHADOWSPILL_STATUS_CLOSED
        : shadowspill_before_task_handle(
            runtime,
            handle,
            adapter_stream(compute_stream_address),
            bindings,
            binding_count
        );
    if (status != SHADOWSPILL_STATUS_OK) {
        shadowspill_pytorch_end_task_range();
    }
    return status;
}

ShadowSpillStatus shadowspill_pytorch_wait_task_allocations(
    uintptr_t task_handle,
    uintptr_t compute_stream_address
) {
    const ShadowSpillTaskHandle *handle =
        (const ShadowSpillTaskHandle *)task_handle;
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    return runtime == NULL || task_handle == 0U
        ? SHADOWSPILL_STATUS_CLOSED
        : shadowspill_wait_task_allocations_handle(
            runtime,
            handle,
            adapter_stream(compute_stream_address)
        );
}

ShadowSpillStatus shadowspill_pytorch_after_task_handle(
    uintptr_t task_handle,
    uintptr_t compute_stream_address
) {
    const ShadowSpillTaskHandle *handle =
        (const ShadowSpillTaskHandle *)task_handle;
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    ShadowSpillStatus status =
        runtime == NULL || task_handle == 0U
        ? SHADOWSPILL_STATUS_CLOSED
        : shadowspill_after_task_handle(
            runtime,
            handle,
            adapter_stream(compute_stream_address)
        );
    shadowspill_pytorch_end_task_range();
    return status;
}

ShadowSpillStatus shadowspill_pytorch_abort_task_handle(
    uintptr_t task_handle
) {
    const ShadowSpillTaskHandle *handle =
        (const ShadowSpillTaskHandle *)task_handle;
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    const ShadowSpillStatus status = runtime == NULL
        ? SHADOWSPILL_STATUS_CLOSED
        : shadowspill_abort_task_handle(
              runtime, handle
          );
    shadowspill_pytorch_end_task_range();
    return status;
}
