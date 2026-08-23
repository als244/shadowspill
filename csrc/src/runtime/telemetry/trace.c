
#include "../internal.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t monotonic_nanoseconds(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        return 0U;
    }
    return (uint64_t)value.tv_sec * 1000000000U + (uint64_t)value.tv_nsec;
}

void shadowspill_trace_append_enabled(
    ShadowSpillRuntime *runtime,
    ShadowSpillTraceEventKind kind,
    uint64_t task_id,
    uint64_t object_id,
    uint64_t allocation_id,
    uint64_t bytes,
    uint64_t detail_0,
    uint64_t detail_1
) {
    if (atomic_load_explicit(&runtime->trace_active, memory_order_acquire) == 0U ||
        atomic_load_explicit(
            &runtime->trace_event_overflow, memory_order_relaxed
        ) != 0U) {
        return;
    }
    uint64_t slot = atomic_load_explicit(
        &runtime->trace_event_count, memory_order_relaxed
    );
    while (slot < runtime->trace_event_capacity &&
           !atomic_compare_exchange_weak_explicit(
               &runtime->trace_event_count,
               &slot,
               slot + 1U,
               memory_order_acq_rel,
               memory_order_relaxed
           )) {
    }
    if (slot >= runtime->trace_event_capacity) {
        atomic_store_explicit(
            &runtime->trace_event_overflow, 1U, memory_order_release
        );
        return;
    }
    const uint64_t timestamp = monotonic_nanoseconds();
    runtime->trace_events[slot] =
        (ShadowSpillTraceEvent){
            .sequence = slot,
            .timestamp_ns = timestamp,
            .step_id = runtime->trace_step_id,
            .task_id = task_id,
            .object_id = object_id,
            .allocation_id = allocation_id,
            .bytes = bytes,
            .detail_0 = detail_0,
            .detail_1 = detail_1,
            .kind = (uint8_t)kind,
        };
    if (kind == SHADOWSPILL_TRACE_SESSION_BEGIN) {
        runtime->trace_begin_timestamp_ns = timestamp;
    } else if (kind == SHADOWSPILL_TRACE_SESSION_END) {
        runtime->trace_end_timestamp_ns = timestamp;
    }
}

ShadowSpillRuntimeStatus shadowspill_trace_prepare(
    ShadowSpillRuntime *runtime,
    const ShadowSpillTraceConfig *config
) {
    if (runtime == NULL || config == NULL ||
        config->abi_version != SHADOWSPILL_ABI_VERSION ||
        config->event_capacity == 0U ||
        config->allocation_event_capacity == 0U ||
        config->event_capacity > SIZE_MAX / sizeof(ShadowSpillTraceEvent) ||
        config->allocation_event_capacity >
            SIZE_MAX / sizeof(ShadowSpillAllocationEvent)) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&runtime->mutex);
    ShadowSpillRuntimeStatus status = shadowspill_current_status_locked(runtime);
    if (status == SHADOWSPILL_RUNTIME_OK &&
        (runtime->trace_active || runtime->allocation_telemetry_active)) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    const int grow_events = status == SHADOWSPILL_RUNTIME_OK &&
        runtime->trace_event_capacity < config->event_capacity;
    const int grow_allocations = status == SHADOWSPILL_RUNTIME_OK &&
        runtime->allocation_event_capacity < config->allocation_event_capacity;
    pthread_mutex_unlock(&runtime->mutex);

    ShadowSpillTraceEvent *events = grow_events
        ? calloc((size_t)config->event_capacity, sizeof(*events))
        : NULL;
    ShadowSpillAllocationEvent *execution_leases = grow_allocations
        ? calloc(
            (size_t)config->allocation_event_capacity,
            sizeof(*execution_leases)
        )
        : NULL;
    if ((grow_events && events == NULL) ||
        (grow_allocations && execution_leases == NULL)) {
        free(events);
        free(execution_leases);
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }

    pthread_mutex_lock(&runtime->mutex);
    status = shadowspill_current_status_locked(runtime);
    if (status == SHADOWSPILL_RUNTIME_OK &&
        (runtime->trace_active || runtime->allocation_telemetry_active)) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    if (status == SHADOWSPILL_RUNTIME_OK) {
        if (events != NULL &&
            runtime->trace_event_capacity < config->event_capacity) {
            free(runtime->trace_events);
            runtime->trace_events = events;
            runtime->trace_event_capacity = config->event_capacity;
            events = NULL;
        }
        if (execution_leases != NULL &&
            runtime->allocation_event_capacity <
                config->allocation_event_capacity) {
            free(runtime->allocation_events);
            runtime->allocation_events = execution_leases;
            runtime->allocation_event_capacity =
                config->allocation_event_capacity;
            execution_leases = NULL;
        }
        runtime->trace_allocation_event_capacity =
            config->allocation_event_capacity;
        runtime->trace_prepared = 1;
    }
    pthread_mutex_unlock(&runtime->mutex);
    free(events);
    free(execution_leases);
    return status;
}

ShadowSpillRuntimeStatus shadowspill_trace_begin(
    ShadowSpillRuntime *runtime,
    uint64_t step_id
) {
    if (runtime == NULL || step_id == SHADOWSPILL_RUNTIME_NO_ID) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&runtime->mutex);
    ShadowSpillRuntimeStatus status = shadowspill_current_status_locked(runtime);
    if (status == SHADOWSPILL_RUNTIME_OK &&
        (!runtime->trace_prepared || runtime->trace_active ||
         runtime->allocation_telemetry_active)) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    if (status == SHADOWSPILL_RUNTIME_OK) {
        runtime->trace_event_count = 0U;
        runtime->next_trace_event_sequence = 0U;
        runtime->trace_step_id = step_id;
        runtime->trace_begin_timestamp_ns = 0U;
        runtime->trace_end_timestamp_ns = 0U;
        runtime->trace_event_overflow = 0;
        runtime->allocation_event_count = 0U;
        runtime->next_allocation_event_sequence = 0U;
        runtime->allocation_event_overflow = 0;
        runtime->allocation_telemetry_active = 1;
        runtime->trace_active = 1;
        shadowspill_append_trace_event_locked(
            runtime,
            SHADOWSPILL_TRACE_SESSION_BEGIN,
            SHADOWSPILL_RUNTIME_NO_ID,
            SHADOWSPILL_RUNTIME_NO_ID,
            SHADOWSPILL_RUNTIME_NO_ID,
            0U,
            0U,
            0U
        );
    }
    pthread_mutex_unlock(&runtime->mutex);
    return status;
}

ShadowSpillRuntimeStatus shadowspill_trace_end(ShadowSpillRuntime *runtime) {
    if (runtime == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&runtime->mutex);
    ShadowSpillRuntimeStatus status = SHADOWSPILL_RUNTIME_OK;
    if (!runtime->trace_active) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
    } else {
        shadowspill_append_trace_event_locked(
            runtime,
            SHADOWSPILL_TRACE_SESSION_END,
            SHADOWSPILL_RUNTIME_NO_ID,
            SHADOWSPILL_RUNTIME_NO_ID,
            SHADOWSPILL_RUNTIME_NO_ID,
            0U,
            0U,
            0U
        );
        runtime->trace_active = 0;
        runtime->allocation_telemetry_active = 0;
    }
    pthread_mutex_unlock(&runtime->mutex);
    shadowspill_leave_task_scope(runtime);
    return status;
}

ShadowSpillRuntimeStatus shadowspill_trace_read(
    ShadowSpillRuntime *runtime,
    ShadowSpillTraceSummary *summary,
    ShadowSpillTraceEvent *events,
    uint64_t event_capacity,
    ShadowSpillAllocationEvent *allocation_events,
    uint64_t allocation_event_capacity
) {
    if (runtime == NULL || summary == NULL ||
        (events == NULL && event_capacity != 0U) ||
        (allocation_events == NULL && allocation_event_capacity != 0U)) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&runtime->mutex);
    ShadowSpillRuntimeStatus status = SHADOWSPILL_RUNTIME_OK;
    if (runtime->trace_active) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
    } else if ((events != NULL && event_capacity < runtime->trace_event_count) ||
               (allocation_events != NULL &&
                allocation_event_capacity < runtime->allocation_event_count)) {
        status = SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    *summary = (ShadowSpillTraceSummary){
        .abi_version = SHADOWSPILL_ABI_VERSION,
        .step_id = runtime->trace_step_id,
        .event_count = runtime->trace_event_count,
        .allocation_event_count = runtime->allocation_event_count,
        .event_capacity = runtime->trace_event_capacity,
        .allocation_event_capacity = runtime->allocation_event_capacity,
        .begin_timestamp_ns = runtime->trace_begin_timestamp_ns,
        .end_timestamp_ns = runtime->trace_end_timestamp_ns,
        .active = (uint8_t)(runtime->trace_active != 0),
        .event_overflow = (uint8_t)(runtime->trace_event_overflow != 0),
        .allocation_event_overflow =
            (uint8_t)(runtime->allocation_event_overflow != 0),
    };
    if (status == SHADOWSPILL_RUNTIME_OK && events != NULL &&
        runtime->trace_event_count != 0U) {
        memcpy(
            events,
            runtime->trace_events,
            (size_t)runtime->trace_event_count * sizeof(*events)
        );
    }
    if (status == SHADOWSPILL_RUNTIME_OK && allocation_events != NULL &&
        runtime->allocation_event_count != 0U) {
        memcpy(
            allocation_events,
            runtime->allocation_events,
            (size_t)runtime->allocation_event_count * sizeof(*allocation_events)
        );
    }
    pthread_mutex_unlock(&runtime->mutex);
    return status;
}
