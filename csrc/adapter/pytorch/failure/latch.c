#include "internal.h"

#include "../internal.h"
#include "../tasks/internal.h"

#include <pthread.h>
#include <stdio.h>
#include <string.h>

void shadowspill_pytorch_failure_clear_locked(int32_t device_ordinal) {
    memset(&adapter.failure, 0, sizeof(adapter.failure));
    adapter.failure_task_label[0] = '\0';
    adapter.failure.device_ordinal = device_ordinal;
    adapter.failure.runtime.task_id = SHADOWSPILL_RUNTIME_NO_ID;
    adapter.failure.runtime.object_id = SHADOWSPILL_RUNTIME_NO_ID;
    adapter.failure.runtime.allocation_id = SHADOWSPILL_RUNTIME_NO_ID;
    adapter.failure.runtime.pool_id = UINT32_MAX;
}

void shadowspill_pytorch_failure_latch_physical_locked(
    ShadowSpillStatus status,
    uint64_t requested_bytes,
    uint64_t free_bytes
) {
    if (adapter.failure.status != SHADOWSPILL_STATUS_OK) {
        return;
    }
    adapter.failure.status = (uint32_t)status;
    adapter.failure.requested_bytes = requested_bytes;
    adapter.failure.runtime.status = (uint32_t)status;
    adapter.failure.runtime.requested_bytes = requested_bytes;
    adapter.failure.runtime.free_bytes = free_bytes;
}

void shadowspill_pytorch_latch_failure(
    ShadowSpillStatus status,
    int32_t device_ordinal,
    const void *address,
    uint64_t requested_bytes
) {
    char task_label[SHADOWSPILL_RUNTIME_TRACE_LABEL_MAX_BYTES + 1U] = {0};
    const char *const active_label = shadowspill_pytorch_task_range_label();
    if (active_label != NULL) {
        (void)snprintf(task_label, sizeof(task_label), "%s", active_label);
    }
    pthread_mutex_lock(&adapter.mutex);
    ++adapter.callback_failures;
    adapter.recent = (ShadowSpillPytorchAdapterFailure){
        .status = (uint32_t)status,
        .device_ordinal = device_ordinal,
        .address = (uint64_t)(uintptr_t)address,
        .requested_bytes = requested_bytes,
    };
    adapter.recent_valid = 1U;
    if (adapter.failure.status == SHADOWSPILL_STATUS_OK) {
        adapter.failure.status = (uint32_t)status;
        adapter.failure.device_ordinal = device_ordinal;
        adapter.failure.address = (uint64_t)(uintptr_t)address;
        adapter.failure.requested_bytes = requested_bytes;
        (void)snprintf(
            adapter.failure_task_label,
            sizeof(adapter.failure_task_label),
            "%s",
            task_label
        );
        if (adapter.runtime != NULL) {
            (void)shadowspill_runtime_failure(
                adapter.runtime, &adapter.failure.runtime
            );
        }
    }
    pthread_mutex_unlock(&adapter.mutex);
}

ShadowSpillStatus shadowspill_pytorch_allocator_statistics(
    ShadowSpillPytorchAdapterStatistics *statistics
) {
    if (statistics == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&adapter.mutex);
    ShadowSpillRuntime *runtime = adapter.runtime;
    const ShadowSpillBackend backend = adapter.backend.table;
    *statistics = (ShadowSpillPytorchAdapterStatistics){
        .allocation_callbacks = adapter.allocation_callbacks,
        .zero_size_allocation_callbacks =
            adapter.zero_size_allocation_callbacks,
        .free_callbacks = adapter.free_callbacks,
        .record_stream_callbacks = adapter.record_stream_callbacks,
        .pointer_lookup_failures = adapter.pointer_lookup_failures,
        .callback_failures = adapter.callback_failures,
        .physical_checks = adapter.physical_checks,
        .peak_process_physical_bytes = adapter.peak_process_physical_bytes,
        .observed_external_high_water_bytes =
            adapter.observed_external_high_water_bytes,
        .physical_budget_sealed = adapter.physical_budget_sealed,
    };
    pthread_mutex_unlock(&adapter.mutex);
    if (runtime == NULL || backend.state == NULL) {
        return SHADOWSPILL_STATUS_CLOSED;
    }
    ShadowSpillStatus status = shadowspill_runtime_statistics(
        runtime, &statistics->runtime
    );
    backend.statistics(backend.state, &statistics->backend);
    return status;
}

ShadowSpillStatus shadowspill_pytorch_allocator_failure(
    ShadowSpillPytorchAdapterFailure *failure
) {
    if (failure == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&adapter.mutex);
    *failure = adapter.failure;
    ShadowSpillRuntime *runtime = adapter.runtime;
    int32_t device_ordinal = adapter.device_ordinal;
    pthread_mutex_unlock(&adapter.mutex);
    if (failure->status == SHADOWSPILL_STATUS_OK && runtime != NULL) {
        ShadowSpillRuntimeFailure runtime_failure = {0};
        if (shadowspill_runtime_failure(runtime, &runtime_failure) ==
                SHADOWSPILL_STATUS_OK &&
            runtime_failure.status != SHADOWSPILL_STATUS_OK) {
            failure->status = runtime_failure.status;
            failure->device_ordinal = device_ordinal;
            failure->runtime = runtime_failure;
        }
    }
    return (ShadowSpillStatus)failure->status;
}

ShadowSpillStatus shadowspill_pytorch_recover_no_progress(void) {
    ShadowSpillRuntime *runtime = shadowspill_pytorch_runtime();
    if (runtime == NULL) {
        return SHADOWSPILL_STATUS_CLOSED;
    }
    ShadowSpillStatus status =
        shadowspill_runtime_recover_no_progress(runtime);
    if (status != SHADOWSPILL_STATUS_OK) {
        return status;
    }
    pthread_mutex_lock(&adapter.mutex);
    if (adapter.failure.status == SHADOWSPILL_STATUS_NO_PROGRESS) {
        shadowspill_pytorch_failure_clear_locked(
            shadowspill_pytorch_device_ordinal()
        );
    } else if (adapter.failure.status != SHADOWSPILL_STATUS_OK) {
        status = (ShadowSpillStatus)adapter.failure.status;
    }
    pthread_mutex_unlock(&adapter.mutex);
    return status;
}
