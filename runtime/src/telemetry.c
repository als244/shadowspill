#include "internal.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct ShadowSpillTaskScope {
    ShadowSpillRuntime *runtime;
    uint64_t task_id;
} ShadowSpillTaskScope;

static _Thread_local ShadowSpillTaskScope task_scope = {
    .runtime = NULL,
    .task_id = SHADOWSPILL_RUNTIME_NO_ID,
};

uint64_t shadowspill_current_task_id(ShadowSpillRuntime *runtime) {
    return task_scope.runtime == runtime
        ? task_scope.task_id
        : SHADOWSPILL_RUNTIME_NO_ID;
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
    return 0;
}

void shadowspill_leave_task_scope(ShadowSpillRuntime *runtime) {
    if (task_scope.runtime == runtime) {
        task_scope.runtime = NULL;
        task_scope.task_id = SHADOWSPILL_RUNTIME_NO_ID;
    }
}

void shadowspill_append_allocation_event_locked(
    ShadowSpillRuntime *runtime,
    const ShadowSpillAllocationRecord *allocation,
    ShadowSpillAllocationEventKind kind,
    ShadowSpillAllocationCategory category
) {
    if (!runtime->allocation_telemetry_active ||
        runtime->allocation_event_overflow) {
        return;
    }
    if (runtime->allocation_event_count >=
        runtime->allocation_event_capacity) {
        runtime->allocation_event_overflow = 1;
        shadowspill_latch_failure_locked(
            runtime,
            SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE,
            SHADOWSPILL_RUNTIME_NO_ID,
            allocation->allocation_id,
            allocation->requested_bytes
        );
        return;
    }
    uint64_t task_id = shadowspill_current_task_id(runtime);
    if (kind == SHADOWSPILL_ALLOCATION_CREATED) {
        task_id = allocation->origin_task_id;
    } else if (kind == SHADOWSPILL_ALLOCATION_RELEASED) {
        task_id = allocation->release_task_id;
    }
    runtime->allocation_events[runtime->allocation_event_count++] =
        (ShadowSpillAllocationEvent){
            .sequence = runtime->next_allocation_event_sequence++,
            .task_id = task_id,
            .allocation_id = allocation->allocation_id,
            .generation = allocation->generation,
            .requested_bytes = allocation->requested_bytes,
            .charged_bytes = allocation->charged_bytes,
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
    ShadowSpillAllocationEvent *events = calloc(
        (size_t)capacity, sizeof(*events)
    );
    if (events == NULL) {
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    pthread_mutex_lock(&runtime->mutex);
    ShadowSpillRuntimeStatus status = shadowspill_current_status_locked(runtime);
    if (status == SHADOWSPILL_RUNTIME_OK &&
        runtime->allocation_telemetry_active) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    if (status == SHADOWSPILL_RUNTIME_OK) {
        free(runtime->allocation_events);
        runtime->allocation_events = events;
        runtime->allocation_event_count = 0U;
        runtime->allocation_event_capacity = capacity;
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
        shadowspill_leave_task_scope(runtime);
    }
}
