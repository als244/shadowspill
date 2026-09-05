#include "internal.h"

#include <stdatomic.h>
#include <stdio.h>

/* The profiler range name of a task: its label, or its id when it has none. */
static void format_task_range_name(
    char *destination,
    size_t destination_bytes,
    const char *operation,
    const ShadowSpillTaskHandle *handle
) {
    const uint64_t task_id = shadowspill_task_id(handle);
    const char *label = shadowspill_task_trace_label(handle);
    if (label != NULL && label[0] != '\0') {
        (void)snprintf(
            destination,
            destination_bytes,
            "shadowspill.pytorch.%s.%s",
            operation,
            label
        );
    } else {
        (void)snprintf(
            destination,
            destination_bytes,
            "shadowspill.pytorch.%s.canonical_%llu",
            operation,
            (unsigned long long)task_id
        );
    }
}

ShadowSpillStatus shadowspill_pytorch_submit_action_batch_handle(
    uintptr_t action_batch_handle,
    uintptr_t trigger_stream_address
) {
    ShadowSpillRuntime *runtime = shadowspill_pytorch_runtime();
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
            shadowspill_pytorch_stream(trigger_stream_address)
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
    if (shadowspill_pytorch_task_range_active() || task_handle == 0U ||
        task_id == SHADOWSPILL_RUNTIME_NO_ID) {
        return SHADOWSPILL_STATUS_INVALID_STATE;
    }
    /* Naming the range costs a format; skip it when nothing would show. */
    char range_name[384];
    const char *name = NULL;
    if (atomic_load_explicit(
            &adapter.profiler_annotations_enabled, memory_order_relaxed
        ) != 0U) {
        format_task_range_name(range_name, sizeof(range_name), "task", handle);
        name = range_name;
    }
    shadowspill_pytorch_task_range_begin(
        name, shadowspill_task_trace_label(handle)
    );
    ShadowSpillRuntime *runtime = shadowspill_pytorch_runtime();
    ShadowSpillStatus status = runtime == NULL
        ? SHADOWSPILL_STATUS_CLOSED
        : shadowspill_before_task_handle(
            runtime,
            handle,
            shadowspill_pytorch_stream(compute_stream_address),
            bindings,
            binding_count
        );
    if (status != SHADOWSPILL_STATUS_OK) {
        shadowspill_pytorch_task_range_end();
    }
    return status;
}

ShadowSpillStatus shadowspill_pytorch_wait_task_allocations(
    uintptr_t task_handle,
    uintptr_t compute_stream_address
) {
    const ShadowSpillTaskHandle *handle =
        (const ShadowSpillTaskHandle *)task_handle;
    ShadowSpillRuntime *runtime = shadowspill_pytorch_runtime();
    return runtime == NULL || task_handle == 0U
        ? SHADOWSPILL_STATUS_CLOSED
        : shadowspill_wait_task_allocations_handle(
            runtime,
            handle,
            shadowspill_pytorch_stream(compute_stream_address)
        );
}

ShadowSpillStatus shadowspill_pytorch_after_task_handle(
    uintptr_t task_handle,
    uintptr_t compute_stream_address
) {
    const ShadowSpillTaskHandle *handle =
        (const ShadowSpillTaskHandle *)task_handle;
    ShadowSpillRuntime *runtime = shadowspill_pytorch_runtime();
    ShadowSpillStatus status =
        runtime == NULL || task_handle == 0U
        ? SHADOWSPILL_STATUS_CLOSED
        : shadowspill_after_task_handle(
            runtime,
            handle,
            shadowspill_pytorch_stream(compute_stream_address)
        );
    shadowspill_pytorch_task_range_end();
    return status;
}

ShadowSpillStatus shadowspill_pytorch_abort_task_handle(
    uintptr_t task_handle
) {
    const ShadowSpillTaskHandle *handle =
        (const ShadowSpillTaskHandle *)task_handle;
    ShadowSpillRuntime *runtime = shadowspill_pytorch_runtime();
    const ShadowSpillStatus status = runtime == NULL
        ? SHADOWSPILL_STATUS_CLOSED
        : shadowspill_abort_task_handle(
              runtime, handle
          );
    shadowspill_pytorch_task_range_end();
    return status;
}
