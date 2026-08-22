#include "../internal.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

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
        /* Allocation events are a diagnostic record, not a resource the step
         * depends on. Running out of room to describe what happened says
         * nothing about whether it succeeded, so stop recording and let the
         * step continue: `allocation_event_overflow` reports the gap, and a
         * caller reading telemetry can see its record is incomplete. Latching
         * a failure here instead made every later allocation in the process
         * fail, and reported it as a memory failure with the operands of
         * whichever allocation happened to fill the last slot. */
        atomic_store_explicit(
            &runtime->allocation_event_overflow, 1U, memory_order_release
        );
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
            .pool_id = allocation->pool->pool_id,
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
