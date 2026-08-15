#include "internal.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct ShadowSpillTaskScope {
    ShadowSpillRuntime *runtime;
    uint64_t task_id;
    const ShadowSpillExecutionRecord *execution;
    uint64_t invocation;
    uint64_t operation_index;
    uint64_t allocation_ordinal;
    uint64_t live_requested_bytes;
    uint64_t live_charged_bytes;
    uint64_t peak_requested_bytes;
    uint64_t peak_charged_bytes;
    uint64_t allocation_count;
    uint64_t free_count;
} ShadowSpillTaskScope;

static _Thread_local ShadowSpillTaskScope task_scope = {
    .runtime = NULL,
    .task_id = SHADOWSPILL_RUNTIME_NO_ID,
    .execution = NULL,
};

uint64_t shadowspill_current_task_id(ShadowSpillRuntime *runtime) {
    return task_scope.runtime == runtime
        ? task_scope.task_id
        : SHADOWSPILL_RUNTIME_NO_ID;
}

uint64_t shadowspill_current_task_allocation_ordinal(
    ShadowSpillRuntime *runtime
) {
    return task_scope.runtime == runtime && task_scope.execution != NULL
        ? task_scope.allocation_ordinal
        : SHADOWSPILL_RUNTIME_NO_ID;
}

uint64_t shadowspill_current_task_invocation(ShadowSpillRuntime *runtime) {
    return task_scope.runtime == runtime && task_scope.execution != NULL
        ? task_scope.invocation
        : 0U;
}

int shadowspill_enter_task_scope(
    ShadowSpillRuntime *runtime,
    uint64_t task_id
) {
    if (task_scope.runtime != NULL || task_id == SHADOWSPILL_RUNTIME_NO_ID) {
        return -1;
    }
    task_scope.runtime = runtime;
    task_scope.task_id = task_id;
    task_scope.execution = NULL;
    task_scope.invocation = 0U;
    task_scope.operation_index = 0U;
    task_scope.allocation_ordinal = 0U;
    task_scope.live_requested_bytes = 0U;
    task_scope.live_charged_bytes = 0U;
    task_scope.peak_requested_bytes = 0U;
    task_scope.peak_charged_bytes = 0U;
    task_scope.allocation_count = 0U;
    task_scope.free_count = 0U;
    return 0;
}

static const ShadowSpillTaskAllocationABIStep *expected_allocation_step(void) {
    if (task_scope.execution == NULL ||
        !task_scope.execution->enforce_allocation_abi ||
        task_scope.operation_index >=
            task_scope.execution->allocation_abi_step_count) {
        return NULL;
    }
    return &task_scope.execution->allocation_abi_steps[
        task_scope.operation_index
    ];
}

static ShadowSpillRuntimeStatus latch_allocation_abi_mismatch(
    ShadowSpillRuntime *runtime,
    uint8_t actual_operation,
    uint64_t actual_ordinal,
    uint64_t requested_bytes,
    uint64_t charged_bytes,
    uint64_t alignment_bytes
) {
    const ShadowSpillTaskAllocationABIStep *expected =
        expected_allocation_step();
    const ShadowSpillTaskAllocationMismatch mismatch = {
        .operation_index = task_scope.operation_index,
        .expected_ordinal = expected == NULL
            ? SHADOWSPILL_RUNTIME_NO_ID : expected->allocation_ordinal,
        .actual_ordinal = actual_ordinal,
        .expected_requested_bytes = expected == NULL
            ? 0U : expected->requested_bytes,
        .actual_requested_bytes = requested_bytes,
        .expected_charged_bytes = expected == NULL
            ? 0U : expected->charged_bytes,
        .actual_charged_bytes = charged_bytes,
        .expected_alignment_bytes = expected == NULL
            ? 0U : expected->alignment_bytes,
        .actual_alignment_bytes = alignment_bytes,
        .expected_operation = expected == NULL
            ? UINT8_MAX : expected->operation,
        .actual_operation = actual_operation,
    };
    shadowspill_latch_task_allocation_abi_failure(runtime, &mismatch);
    return SHADOWSPILL_RUNTIME_TASK_ALLOCATION_ABI_MISMATCH;
}

int shadowspill_enter_execution_scope(
    ShadowSpillRuntime *runtime,
    const ShadowSpillExecutionRecord *record
) {
    if (record == NULL || record->runtime_owner != runtime ||
        shadowspill_enter_task_scope(runtime, record->task_id) != 0) {
        return -1;
    }
    task_scope.execution = record;
    task_scope.invocation = atomic_fetch_add_explicit(
        &((ShadowSpillExecutionRecord *)record)->invocation_count,
        1U,
        memory_order_acq_rel
    ) + 1U;
    return 0;
}

ShadowSpillRuntimeStatus shadowspill_validate_task_allocation(
    ShadowSpillRuntime *runtime,
    uint64_t requested_bytes,
    uint64_t charged_bytes,
    uint64_t alignment_bytes
) {
    if (runtime == NULL || requested_bytes == 0U || charged_bytes == 0U ||
        alignment_bytes == 0U) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    if (task_scope.runtime != runtime || task_scope.execution == NULL) {
        return SHADOWSPILL_RUNTIME_OK;
    }
    const ShadowSpillExecutionRecord *record = task_scope.execution;
    const uint64_t projected_requested =
        task_scope.live_requested_bytes + requested_bytes;
    const uint64_t projected_charged =
        task_scope.live_charged_bytes + charged_bytes;
    const int request_exceeded =
        (record->maximum_requested_allocation_bytes != 0U &&
         requested_bytes > record->maximum_requested_allocation_bytes) ||
        (record->maximum_charged_allocation_bytes != 0U &&
         charged_bytes > record->maximum_charged_allocation_bytes);
    const int live_exceeded =
        (record->live_requested_allocation_limit_bytes != 0U &&
         projected_requested > record->live_requested_allocation_limit_bytes) ||
        (record->live_charged_allocation_limit_bytes != 0U &&
         projected_charged > record->live_charged_allocation_limit_bytes);
    if (request_exceeded || live_exceeded) {
        shadowspill_latch_task_envelope_failure(
            runtime,
            requested_bytes,
            charged_bytes,
            projected_requested,
            projected_charged,
            record->live_requested_allocation_limit_bytes,
            record->live_charged_allocation_limit_bytes,
            record->maximum_requested_allocation_bytes,
            record->maximum_charged_allocation_bytes
        );
        return SHADOWSPILL_RUNTIME_TASK_ALLOCATION_ENVELOPE_EXCEEDED;
    }
    if (!record->enforce_allocation_abi) {
        return SHADOWSPILL_RUNTIME_OK;
    }
    const ShadowSpillTaskAllocationABIStep *expected =
        expected_allocation_step();
    if (expected == NULL ||
        expected->operation != SHADOWSPILL_TASK_ALLOCATION_ALLOCATE ||
        expected->allocation_ordinal != task_scope.allocation_ordinal ||
        expected->requested_bytes != requested_bytes ||
        expected->charged_bytes != charged_bytes ||
        expected->alignment_bytes != alignment_bytes) {
        return latch_allocation_abi_mismatch(
            runtime,
            SHADOWSPILL_TASK_ALLOCATION_ALLOCATE,
            task_scope.allocation_ordinal,
            requested_bytes,
            charged_bytes,
            alignment_bytes
        );
    }
    return SHADOWSPILL_RUNTIME_OK;
}

uint64_t shadowspill_commit_task_allocation(
    ShadowSpillRuntime *runtime,
    uint64_t requested_bytes,
    uint64_t charged_bytes
) {
    if (task_scope.runtime != runtime || task_scope.execution == NULL) {
        return SHADOWSPILL_RUNTIME_NO_ID;
    }
    const uint64_t allocation_ordinal = task_scope.allocation_ordinal++;
    task_scope.live_requested_bytes += requested_bytes;
    task_scope.live_charged_bytes += charged_bytes;
    if (task_scope.live_requested_bytes > task_scope.peak_requested_bytes) {
        task_scope.peak_requested_bytes = task_scope.live_requested_bytes;
    }
    if (task_scope.live_charged_bytes > task_scope.peak_charged_bytes) {
        task_scope.peak_charged_bytes = task_scope.live_charged_bytes;
    }
    ++task_scope.allocation_count;
    if (task_scope.execution->enforce_allocation_abi) {
        ++task_scope.operation_index;
    }
    return allocation_ordinal;
}

ShadowSpillRuntimeStatus shadowspill_release_task_allocation(
    ShadowSpillRuntime *runtime,
    uint64_t origin_task_id,
    uint64_t origin_task_invocation,
    uint64_t allocation_ordinal,
    uint64_t requested_bytes,
    uint64_t charged_bytes,
    uint64_t alignment_bytes
) {
    if (task_scope.runtime != runtime || task_scope.execution == NULL ||
        task_scope.task_id != origin_task_id ||
        task_scope.invocation != origin_task_invocation) {
        return SHADOWSPILL_RUNTIME_OK;
    }
    if (task_scope.execution->enforce_allocation_abi) {
        const ShadowSpillTaskAllocationABIStep *expected =
            expected_allocation_step();
        if (expected == NULL ||
            expected->operation != SHADOWSPILL_TASK_ALLOCATION_FREE ||
            expected->allocation_ordinal != allocation_ordinal ||
            expected->requested_bytes != requested_bytes ||
            expected->charged_bytes != charged_bytes ||
            expected->alignment_bytes != alignment_bytes) {
            return latch_allocation_abi_mismatch(
                runtime,
                SHADOWSPILL_TASK_ALLOCATION_FREE,
                allocation_ordinal,
                requested_bytes,
                charged_bytes,
                alignment_bytes
            );
        }
        ++task_scope.operation_index;
    }
    if (requested_bytes <= task_scope.live_requested_bytes) {
        task_scope.live_requested_bytes -= requested_bytes;
    } else {
        task_scope.live_requested_bytes = 0U;
    }
    if (charged_bytes <= task_scope.live_charged_bytes) {
        task_scope.live_charged_bytes -= charged_bytes;
    } else {
        task_scope.live_charged_bytes = 0U;
    }
    ++task_scope.free_count;
    return SHADOWSPILL_RUNTIME_OK;
}

ShadowSpillRuntimeStatus shadowspill_validate_task_allocation_complete(
    ShadowSpillRuntime *runtime
) {
    if (task_scope.runtime != runtime || task_scope.execution == NULL ||
        !task_scope.execution->enforce_allocation_abi) {
        return SHADOWSPILL_RUNTIME_OK;
    }
    if (task_scope.operation_index ==
        task_scope.execution->allocation_abi_step_count) {
        return SHADOWSPILL_RUNTIME_OK;
    }
    return latch_allocation_abi_mismatch(
        runtime,
        UINT8_MAX,
        SHADOWSPILL_RUNTIME_NO_ID,
        0U,
        0U,
        0U
    );
}

void shadowspill_leave_task_scope(ShadowSpillRuntime *runtime) {
    if (task_scope.runtime == runtime) {
        task_scope.runtime = NULL;
        task_scope.task_id = SHADOWSPILL_RUNTIME_NO_ID;
        task_scope.execution = NULL;
        task_scope.invocation = 0U;
        task_scope.operation_index = 0U;
        task_scope.allocation_ordinal = 0U;
        task_scope.live_requested_bytes = 0U;
        task_scope.live_charged_bytes = 0U;
        task_scope.peak_requested_bytes = 0U;
        task_scope.peak_charged_bytes = 0U;
        task_scope.allocation_count = 0U;
        task_scope.free_count = 0U;
    }
}

void shadowspill_append_allocation_event_locked(
    ShadowSpillRuntime *runtime,
    const ShadowSpillMemoryLease *allocation,
    ShadowSpillAllocationEventKind kind,
    ShadowSpillAllocationCategory category
) {
    if (atomic_load_explicit(
            &runtime->allocation_telemetry_active, memory_order_acquire
        ) == 0U || atomic_load_explicit(
            &runtime->allocation_event_overflow, memory_order_relaxed
        ) != 0U) {
        return;
    }
    uint64_t slot = atomic_load_explicit(
        &runtime->allocation_event_count, memory_order_relaxed
    );
    while (slot < runtime->allocation_event_capacity &&
           !atomic_compare_exchange_weak_explicit(
               &runtime->allocation_event_count,
               &slot,
               slot + 1U,
               memory_order_acq_rel,
               memory_order_relaxed
           )) {
    }
    if (slot >= runtime->allocation_event_capacity) {
        atomic_store_explicit(
            &runtime->allocation_event_overflow, 1U, memory_order_release
        );
        if (atomic_load_explicit(
                &runtime->trace_active, memory_order_acquire
            ) == 0U) {
            shadowspill_latch_failure_locked(
                runtime,
                SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE,
                SHADOWSPILL_RUNTIME_NO_ID,
                allocation->allocation_id,
                allocation->requested_bytes
            );
        }
        return;
    }
    uint64_t task_id = shadowspill_current_task_id(runtime);
    if (kind == SHADOWSPILL_ALLOCATION_CREATED) {
        task_id = allocation->origin_task_id;
    } else if (kind == SHADOWSPILL_ALLOCATION_LOGICAL_FREED) {
        task_id = allocation->origin_task_id;
    } else if (kind == SHADOWSPILL_ALLOCATION_RELEASED) {
        task_id = allocation->release_task_id;
    }
    runtime->allocation_events[slot] =
        (ShadowSpillAllocationEvent){
            .sequence = slot,
            .task_id = task_id,
            .allocation_id = allocation->allocation_id,
            .generation = allocation->generation,
            .requested_bytes = allocation->requested_bytes,
            .charged_bytes = allocation->charged_bytes,
            .alignment_bytes = allocation->alignment_bytes,
            .slab_offset = allocation->offset,
            .kind = (uint8_t)kind,
            .category = (uint8_t)category,
        };
}

ShadowSpillRuntimeStatus shadowspill_allocation_telemetry_start(
    ShadowSpillRuntime *runtime,
    uint64_t capacity
) {
    if (runtime == NULL || capacity == 0U ||
        capacity > SIZE_MAX / sizeof(ShadowSpillAllocationEvent)) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&runtime->mutex);
    ShadowSpillRuntimeStatus status = shadowspill_current_status_locked(runtime);
    if (status == SHADOWSPILL_RUNTIME_OK &&
        runtime->allocation_telemetry_active) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    const int needs_growth = status == SHADOWSPILL_RUNTIME_OK &&
        runtime->allocation_event_capacity < capacity;
    pthread_mutex_unlock(&runtime->mutex);
    ShadowSpillAllocationEvent *events = NULL;
    if (needs_growth) {
        events = calloc((size_t)capacity, sizeof(*events));
        if (events == NULL) {
            return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
        }
    }
    pthread_mutex_lock(&runtime->mutex);
    status = shadowspill_current_status_locked(runtime);
    if (status == SHADOWSPILL_RUNTIME_OK &&
        runtime->allocation_telemetry_active) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    if (status == SHADOWSPILL_RUNTIME_OK) {
        if (events != NULL && runtime->allocation_event_capacity < capacity) {
            free(runtime->allocation_events);
            runtime->allocation_events = events;
            runtime->allocation_event_capacity = capacity;
            events = NULL;
        }
        runtime->allocation_event_count = 0U;
        runtime->next_allocation_event_sequence = 0U;
        runtime->allocation_event_overflow = 0;
        runtime->allocation_telemetry_active = 1;
        events = NULL;
    }
    pthread_mutex_unlock(&runtime->mutex);
    free(events);
    return status;
}

ShadowSpillRuntimeStatus shadowspill_allocation_telemetry_stop(
    ShadowSpillRuntime *runtime
) {
    if (runtime == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&runtime->mutex);
    ShadowSpillRuntimeStatus status = shadowspill_current_status_locked(runtime);
    if (status == SHADOWSPILL_RUNTIME_OK &&
        !runtime->allocation_telemetry_active) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    runtime->allocation_telemetry_active = 0;
    pthread_mutex_unlock(&runtime->mutex);
    shadowspill_leave_task_scope(runtime);
    return status;
}

ShadowSpillRuntimeStatus shadowspill_allocation_telemetry_read(
    ShadowSpillRuntime *runtime,
    ShadowSpillAllocationEvent *events,
    uint64_t capacity,
    uint64_t *count
) {
    if (runtime == NULL || count == NULL ||
        (events == NULL && capacity != 0U)) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&runtime->mutex);
    *count = runtime->allocation_event_count;
    ShadowSpillRuntimeStatus status = SHADOWSPILL_RUNTIME_OK;
    if (events != NULL && capacity < runtime->allocation_event_count) {
        status = SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    } else if (events != NULL && runtime->allocation_event_count != 0U) {
        memcpy(
            events,
            runtime->allocation_events,
            (size_t)runtime->allocation_event_count * sizeof(*events)
        );
    }
    pthread_mutex_unlock(&runtime->mutex);
    return status;
}

void shadowspill_abort_task(ShadowSpillRuntime *runtime) {
    if (runtime != NULL) {
        shadowspill_finalize_aborted_task_retirements(
            runtime, shadowspill_current_task_id(runtime)
        );
        shadowspill_leave_task_scope(runtime);
    }
}
