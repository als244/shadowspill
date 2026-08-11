#include "internal.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int stream_equal(
    ShadowSpillBackendStream left,
    ShadowSpillBackendStream right
) {
    return memcmp(&left, &right, sizeof(left)) == 0;
}

ShadowSpillAllocationRecord *shadowspill_find_allocation(
    ShadowSpillRuntime *runtime,
    uint64_t allocation_id
) {
    for (ShadowSpillAllocationRecord *record = runtime->allocations;
         record != NULL; record = record->next) {
        if (record->allocation_id == allocation_id) {
            return record;
        }
    }
    return NULL;
}

ShadowSpillAllocationRecord *shadowspill_find_allocation_by_pointer(
    ShadowSpillRuntime *runtime,
    const void *pointer
) {
    for (ShadowSpillAllocationRecord *record = runtime->allocations;
         record != NULL; record = record->next) {
        if (record->pointer == pointer && !record->logical_freed) {
            return record;
        }
    }
    return NULL;
}

static int has_release_source(const ShadowSpillRuntime *runtime) {
    if (runtime->pending_retirements != 0U) {
        return 1;
    }
    for (const ShadowSpillQueuedAction *action = runtime->action_head;
         action != NULL; action = action->next) {
        if (action->kind == SHADOWSPILL_RUNTIME_RELEASE ||
            action->kind == SHADOWSPILL_RUNTIME_OFFLOAD) {
            return 1;
        }
    }
    return 0;
}

ShadowSpillRuntimeStatus shadowspill_allocate_locked(
    ShadowSpillRuntime *runtime,
    uint64_t bytes,
    uint64_t alignment,
    int plan_owned,
    ShadowSpillAllocationRecord **record
) {
    uint64_t charged = bytes == 0U ? 1U : bytes;
    if (alignment < runtime->minimum_alignment) {
        alignment = runtime->minimum_alignment;
    }
    uint64_t offset = 0U;
    int range_status = shadowspill_range_allocate(
        &runtime->device_ranges, charged, alignment, &offset
    );
    if (range_status > 0) {
        return SHADOWSPILL_RUNTIME_OUT_OF_MEMORY;
    }
    if (range_status < 0) {
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    ShadowSpillAllocationRecord *created = calloc(1U, sizeof(*created));
    if (created == NULL) {
        (void)shadowspill_range_free(&runtime->device_ranges, offset, charged);
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    created->allocation_id = runtime->next_allocation_id++;
    created->generation = runtime->next_generation++;
    created->requested_bytes = bytes;
    created->charged_bytes = charged;
    created->offset = offset;
    created->pointer = (void *)((unsigned char *)runtime->device_slab + offset);
    created->plan_owned = plan_owned;
    created->next = runtime->allocations;
    runtime->allocations = created;
    ++runtime->live_allocations;
    *record = created;
    return SHADOWSPILL_RUNTIME_OK;
}

void shadowspill_release_allocation_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillAllocationRecord *allocation
) {
    if (allocation->pointer == NULL) {
        return;
    }
    if (shadowspill_range_free(
            &runtime->device_ranges,
            allocation->offset,
            allocation->charged_bytes
        ) != 0) {
        shadowspill_latch_failure_locked(
            runtime,
            SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE,
            SHADOWSPILL_RUNTIME_NO_ID,
            allocation->allocation_id,
            allocation->charged_bytes
        );
        return;
    }
    allocation->pointer = NULL;
    allocation->logical_freed = 1;
    allocation->plan_owned = 0;
    if (runtime->live_allocations != 0U) {
        --runtime->live_allocations;
    }
    pthread_cond_broadcast(&runtime->condition);
}

ShadowSpillRuntimeStatus shadowspill_allocate(
    ShadowSpillRuntime *runtime,
    uint64_t bytes,
    uint64_t alignment,
    ShadowSpillBackendStream stream,
    ShadowSpillAllocation *allocation
) {
    (void)stream;
    if (runtime == NULL || allocation == NULL || alignment == 0U) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&runtime->mutex);
    ShadowSpillRuntimeStatus status = shadowspill_current_status_locked(runtime);
    while (status == SHADOWSPILL_RUNTIME_OK) {
        ShadowSpillAllocationRecord *record = NULL;
        status = shadowspill_allocate_locked(
            runtime, bytes, alignment, 0, &record
        );
        if (status == SHADOWSPILL_RUNTIME_OK) {
            *allocation = (ShadowSpillAllocation){
                .allocation_id = record->allocation_id,
                .generation = record->generation,
                .requested_bytes = record->requested_bytes,
                .charged_bytes = record->charged_bytes,
                .pointer = record->pointer,
            };
            break;
        }
        if (status != SHADOWSPILL_RUNTIME_OUT_OF_MEMORY) {
            break;
        }
        if (!has_release_source(runtime)) {
            shadowspill_latch_failure_locked(
                runtime,
                SHADOWSPILL_RUNTIME_NO_PROGRESS,
                SHADOWSPILL_RUNTIME_NO_ID,
                SHADOWSPILL_RUNTIME_NO_ID,
                bytes
            );
            status = SHADOWSPILL_RUNTIME_NO_PROGRESS;
            break;
        }
        ++runtime->blocked_allocators;
        pthread_cond_wait(&runtime->condition, &runtime->mutex);
        --runtime->blocked_allocators;
        status = shadowspill_current_status_locked(runtime);
    }
    pthread_mutex_unlock(&runtime->mutex);
    return status;
}

ShadowSpillRuntimeStatus shadowspill_allocation_for_pointer(
    ShadowSpillRuntime *runtime,
    const void *pointer,
    ShadowSpillAllocation *allocation
) {
    if (runtime == NULL || pointer == NULL || allocation == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&runtime->mutex);
    ShadowSpillAllocationRecord *record =
        shadowspill_find_allocation_by_pointer(runtime, pointer);
    if (record == NULL) {
        pthread_mutex_unlock(&runtime->mutex);
        return SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    *allocation = (ShadowSpillAllocation){
        .allocation_id = record->allocation_id,
        .generation = record->generation,
        .requested_bytes = record->requested_bytes,
        .charged_bytes = record->charged_bytes,
        .pointer = record->pointer,
    };
    pthread_mutex_unlock(&runtime->mutex);
    return SHADOWSPILL_RUNTIME_OK;
}

static int append_stream(
    ShadowSpillAllocationRecord *allocation,
    ShadowSpillBackendStream stream
) {
    for (ShadowSpillStreamRecord *item = allocation->streams; item != NULL;
         item = item->next) {
        if (stream_equal(item->stream, stream)) {
            return 0;
        }
    }
    ShadowSpillStreamRecord *created = calloc(1U, sizeof(*created));
    if (created == NULL) {
        return -1;
    }
    created->stream = stream;
    created->next = allocation->streams;
    allocation->streams = created;
    return 0;
}

ShadowSpillRuntimeStatus shadowspill_record_stream(
    ShadowSpillRuntime *runtime,
    uint64_t allocation_id,
    ShadowSpillBackendStream stream
) {
    if (runtime == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&runtime->mutex);
    ShadowSpillAllocationRecord *allocation = shadowspill_find_allocation(
        runtime, allocation_id
    );
    ShadowSpillRuntimeStatus status = SHADOWSPILL_RUNTIME_OK;
    if (allocation == NULL || allocation->logical_freed) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
    } else if (append_stream(allocation, stream) != 0) {
        status = SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    pthread_mutex_unlock(&runtime->mutex);
    return status;
}

static void destroy_event_list(
    ShadowSpillRuntime *runtime,
    ShadowSpillEventRecord *events
) {
    while (events != NULL) {
        ShadowSpillEventRecord *next = events->next;
        (void)runtime->backend.destroy_event(
            runtime->backend.context, events->event
        );
        free(events);
        events = next;
    }
}

ShadowSpillRuntimeStatus shadowspill_free(
    ShadowSpillRuntime *runtime,
    uint64_t allocation_id,
    ShadowSpillBackendStream stream
) {
    if (runtime == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&runtime->mutex);
    ShadowSpillAllocationRecord *allocation = shadowspill_find_allocation(
        runtime, allocation_id
    );
    ShadowSpillRuntimeStatus status = SHADOWSPILL_RUNTIME_OK;
    if (allocation == NULL || allocation->logical_freed) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
        goto done;
    }
    if (allocation->plan_owned) {
        goto done;
    }
    if (append_stream(allocation, stream) != 0) {
        status = SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
        goto done;
    }
    ShadowSpillEventRecord *events = NULL;
    for (ShadowSpillStreamRecord *item = allocation->streams; item != NULL;
         item = item->next) {
        ShadowSpillEventRecord *event = calloc(1U, sizeof(*event));
        int event_created = 0;
        if (event != NULL && runtime->backend.create_event(
                runtime->backend.context, &event->event
            ) == 0) {
            event_created = 1;
        }
        if (event == NULL || !event_created || runtime->backend.record_event(
                runtime->backend.context, event->event, item->stream
            ) != 0) {
            if (event_created) {
                (void)runtime->backend.destroy_event(
                    runtime->backend.context, event->event
                );
            }
            free(event);
            destroy_event_list(runtime, events);
            status = SHADOWSPILL_RUNTIME_BACKEND_FAILURE;
            shadowspill_latch_failure_locked(
                runtime, status, SHADOWSPILL_RUNTIME_NO_ID, allocation_id, 0U
            );
            goto done;
        }
        event->next = events;
        events = event;
    }
    allocation->retirement_events = events;
    allocation->logical_freed = 1;
    ++runtime->pending_retirements;
    pthread_cond_broadcast(&runtime->condition);

done:
    pthread_mutex_unlock(&runtime->mutex);
    return status;
}
