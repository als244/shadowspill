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

static uint64_t mix_index(uint64_t value, uint64_t bucket_count) {
    value ^= value >> 33U;
    value *= UINT64_C(0xff51afd7ed558ccd);
    value ^= value >> 33U;
    return value % bucket_count;
}

static void *allocation_lookup_pointer(
    const ShadowSpillRuntime *runtime,
    const ShadowSpillAllocationRecord *allocation
) {
    return allocation->pointer != NULL
        ? allocation->pointer
        : (void *)((unsigned char *)runtime->device_slab + allocation->offset);
}

static void index_allocation_id_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillAllocationRecord *allocation
) {
    const uint64_t bucket = mix_index(
        allocation->allocation_id, runtime->allocation_index_bucket_count
    );
    allocation->id_index_next = runtime->allocations_by_id[bucket];
    runtime->allocations_by_id[bucket] = allocation;
}

static void unindex_allocation_id_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillAllocationRecord *allocation
) {
    const uint64_t bucket = mix_index(
        allocation->allocation_id, runtime->allocation_index_bucket_count
    );
    ShadowSpillAllocationRecord **link = &runtime->allocations_by_id[bucket];
    while (*link != NULL && *link != allocation) {
        link = &(*link)->id_index_next;
    }
    if (*link == allocation) {
        *link = allocation->id_index_next;
    }
    allocation->id_index_next = NULL;
}

static void index_allocation_pointer_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillAllocationRecord *allocation
) {
    const uint64_t address = (uint64_t)(uintptr_t)allocation_lookup_pointer(
        runtime, allocation
    );
    const uint64_t bucket = mix_index(
        address, runtime->allocation_index_bucket_count
    );
    allocation->pointer_index_next = runtime->allocations_by_pointer[bucket];
    runtime->allocations_by_pointer[bucket] = allocation;
}

static void unindex_allocation_pointer_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillAllocationRecord *allocation
) {
    const uint64_t address = (uint64_t)(uintptr_t)allocation_lookup_pointer(
        runtime, allocation
    );
    const uint64_t bucket = mix_index(
        address, runtime->allocation_index_bucket_count
    );
    ShadowSpillAllocationRecord **link =
        &runtime->allocations_by_pointer[bucket];
    while (*link != NULL && *link != allocation) {
        link = &(*link)->pointer_index_next;
    }
    if (*link == allocation) {
        *link = allocation->pointer_index_next;
    }
    allocation->pointer_index_next = NULL;
}

static void index_reusable_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillAllocationRecord *allocation
) {
    if (allocation->in_reusable_index) {
        return;
    }
    const uint64_t bucket = mix_index(
        allocation->charged_bytes, runtime->reusable_index_bucket_count
    );
    allocation->reusable_index_next = runtime->reusable_by_size[bucket];
    runtime->reusable_by_size[bucket] = allocation;
    allocation->in_reusable_index = 1U;
}

static void unindex_reusable_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillAllocationRecord *allocation
) {
    if (!allocation->in_reusable_index) {
        return;
    }
    const uint64_t bucket = mix_index(
        allocation->charged_bytes, runtime->reusable_index_bucket_count
    );
    ShadowSpillAllocationRecord **link = &runtime->reusable_by_size[bucket];
    while (*link != NULL && *link != allocation) {
        link = &(*link)->reusable_index_next;
    }
    if (*link == allocation) {
        *link = allocation->reusable_index_next;
    }
    allocation->reusable_index_next = NULL;
    allocation->in_reusable_index = 0U;
}

static void activate_allocation_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillAllocationRecord *allocation
) {
    allocation->active_next = runtime->active_allocations;
    allocation->active_previous_link = &runtime->active_allocations;
    if (allocation->active_next != NULL) {
        allocation->active_next->active_previous_link = &allocation->active_next;
    }
    runtime->active_allocations = allocation;
}

static void deactivate_allocation_locked(
    ShadowSpillAllocationRecord *allocation
) {
    if (allocation->active_previous_link == NULL) {
        return;
    }
    *allocation->active_previous_link = allocation->active_next;
    if (allocation->active_next != NULL) {
        allocation->active_next->active_previous_link =
            allocation->active_previous_link;
    }
    allocation->active_next = NULL;
    allocation->active_previous_link = NULL;
}

static void free_stream_records(ShadowSpillStreamRecord *streams) {
    while (streams != NULL) {
        ShadowSpillStreamRecord *next = streams->next;
        free(streams);
        streams = next;
    }
}

void shadowspill_release_task_fence_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillTaskFence *fence
) {
    if (fence == NULL || --fence->references != 0U) {
        return;
    }
    if (runtime->backend.destroy_event(
            runtime->backend.context, fence->event
        ) != 0) {
        shadowspill_latch_failure_locked(
            runtime,
            SHADOWSPILL_RUNTIME_BACKEND_FAILURE,
            SHADOWSPILL_RUNTIME_NO_ID,
            SHADOWSPILL_RUNTIME_NO_ID,
            0U
        );
    }
    free(fence);
}

int shadowspill_task_fence_complete_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillTaskFence *fence,
    int *complete
) {
    if (fence == NULL || complete == NULL) {
        return -1;
    }
    if (fence->completion_known) {
        *complete = 1;
        return 0;
    }
    if (fence->last_query_epoch == runtime->event_query_epoch) {
        *complete = fence->last_query_complete != 0U;
        return 0;
    }
    if (runtime->backend.query_event(
            runtime->backend.context, fence->event, complete
        ) != 0) {
        return -1;
    }
    fence->last_query_epoch = runtime->event_query_epoch;
    fence->last_query_complete = (uint8_t)(*complete != 0);
    if (*complete) {
        fence->completion_known = 1U;
    }
    return 0;
}

ShadowSpillAllocationRecord *shadowspill_find_allocation(
    ShadowSpillRuntime *runtime,
    uint64_t allocation_id
) {
    const uint64_t bucket = mix_index(
        allocation_id, runtime->allocation_index_bucket_count
    );
    for (ShadowSpillAllocationRecord *record =
             runtime->allocations_by_id[bucket];
         record != NULL; record = record->id_index_next) {
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
    const uint64_t bucket = mix_index(
        (uint64_t)(uintptr_t)pointer,
        runtime->allocation_index_bucket_count
    );
    for (ShadowSpillAllocationRecord *record =
             runtime->allocations_by_pointer[bucket];
         record != NULL; record = record->pointer_index_next) {
        if (record->pointer == pointer && !record->logical_freed) {
            return record;
        }
        const void *retired_pointer =
            (const unsigned char *)runtime->device_slab + record->offset;
        if (record->logical_freed && record->ever_plan_owned &&
            !record->framework_free_seen && retired_pointer == pointer) {
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

static ShadowSpillRuntimeStatus fence_task_retirements_locked(
    ShadowSpillRuntime *runtime,
    uint64_t task_id,
    ShadowSpillBackendStream stream
) {
    uint64_t count = 0U;
    for (ShadowSpillAllocationRecord *allocation = runtime->active_allocations;
         allocation != NULL; allocation = allocation->active_next) {
        if (allocation->logical_freed && allocation->pointer != NULL &&
            allocation->release_task_id == task_id &&
            allocation->retirement_events == NULL &&
            allocation->retirement_fence == NULL) {
            ++count;
        }
    }
    if (count == 0U) {
        return SHADOWSPILL_RUNTIME_OK;
    }
    ShadowSpillTaskFence *fence = calloc(1U, sizeof(*fence));
    int created = 0;
    if (fence != NULL && runtime->backend.create_event(
            runtime->backend.context, &fence->event
        ) == 0) {
        created = 1;
    }
    if (fence == NULL || !created || runtime->backend.record_event(
            runtime->backend.context, fence->event, stream
        ) != 0) {
        if (created) {
            (void)runtime->backend.destroy_event(
                runtime->backend.context, fence->event
            );
        }
        free(fence);
        return SHADOWSPILL_RUNTIME_BACKEND_FAILURE;
    }
    for (ShadowSpillAllocationRecord *allocation = runtime->active_allocations;
         allocation != NULL; allocation = allocation->active_next) {
        if (!allocation->logical_freed || allocation->pointer == NULL ||
            allocation->release_task_id != task_id ||
            allocation->retirement_events != NULL ||
            allocation->retirement_fence != NULL) {
            continue;
        }
        allocation->retirement_fence = fence;
        ++fence->references;
    }
    pthread_cond_broadcast(&runtime->condition);
    return SHADOWSPILL_RUNTIME_OK;
}

ShadowSpillRuntimeStatus shadowspill_allocate_locked(
    ShadowSpillRuntime *runtime,
    uint64_t bytes,
    uint64_t alignment,
    int plan_owned,
    uint64_t origin_task_id,
    ShadowSpillAllocationRecord **record
) {
    uint64_t charged = bytes == 0U ? 1U : bytes;
    if (alignment < runtime->minimum_alignment) {
        alignment = runtime->minimum_alignment;
    }
    uint64_t offset = 0U;
    int range_status = plan_owned
        ? shadowspill_range_allocate_best_fit_low(
            &runtime->device_ranges, charged, alignment, &offset
        )
        : shadowspill_range_allocate_best_fit_high(
            &runtime->device_ranges, charged, alignment, &offset
        );
    if (range_status > 0) {
        return SHADOWSPILL_RUNTIME_OUT_OF_MEMORY;
    }
    if (range_status < 0) {
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    ShadowSpillRuntimeStatus status =
        shadowspill_adopt_reserved_device_range_locked(
            runtime, bytes, offset, plan_owned, origin_task_id, record
        );
    if (status != SHADOWSPILL_RUNTIME_OK) {
        (void)shadowspill_range_free(
            &runtime->device_ranges, offset, charged
        );
    }
    return status;
}

ShadowSpillRuntimeStatus shadowspill_adopt_reserved_device_range_locked(
    ShadowSpillRuntime *runtime,
    uint64_t bytes,
    uint64_t offset,
    int plan_owned,
    uint64_t origin_task_id,
    ShadowSpillAllocationRecord **record
) {
    const uint64_t charged = bytes == 0U ? 1U : bytes;
    ShadowSpillAllocationRecord *created = calloc(1U, sizeof(*created));
    if (created == NULL) {
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    created->allocation_id = runtime->next_allocation_id++;
    created->generation = runtime->next_generation++;
    created->requested_bytes = bytes;
    created->charged_bytes = charged;
    created->offset = offset;
    created->origin_task_id = origin_task_id;
    created->release_task_id = SHADOWSPILL_RUNTIME_NO_ID;
    created->handoff_from_object_id = SHADOWSPILL_RUNTIME_NO_ID;
    created->handoff_to_object_id = SHADOWSPILL_RUNTIME_NO_ID;
    created->handoff_task_id = SHADOWSPILL_RUNTIME_NO_ID;
    created->pointer = (void *)((unsigned char *)runtime->device_slab + offset);
    created->plan_owned = plan_owned;
    created->next = runtime->allocations;
    runtime->allocations = created;
    activate_allocation_locked(runtime, created);
    index_allocation_id_locked(runtime, created);
    index_allocation_pointer_locked(runtime, created);
    runtime->requested_allocated_bytes += bytes;
    if (runtime->requested_allocated_bytes >
        runtime->peak_requested_allocated_bytes) {
        runtime->peak_requested_allocated_bytes =
            runtime->requested_allocated_bytes;
    }
    ++runtime->live_allocations;
    shadowspill_append_allocation_event_locked(
        runtime,
        created,
        SHADOWSPILL_ALLOCATION_CREATED,
        plan_owned ? SHADOWSPILL_ALLOCATION_PLANNED_OBJECT
                   : SHADOWSPILL_ALLOCATION_ANONYMOUS
    );
    if (runtime->failure.status != SHADOWSPILL_RUNTIME_OK) {
        unindex_allocation_pointer_locked(runtime, created);
        unindex_allocation_id_locked(runtime, created);
        runtime->allocations = created->next;
        deactivate_allocation_locked(created);
        runtime->requested_allocated_bytes -= bytes;
        --runtime->live_allocations;
        free(created);
        return (ShadowSpillRuntimeStatus)runtime->failure.status;
    }
    *record = created;
    return SHADOWSPILL_RUNTIME_OK;
}

static ShadowSpillRuntimeStatus reuse_pending_allocation_locked(
    ShadowSpillRuntime *runtime,
    uint64_t bytes,
    uint64_t alignment,
    ShadowSpillBackendStream stream,
    uint64_t origin_task_id,
    int exact_only,
    ShadowSpillAllocationRecord **record
) {
    const uint64_t required = bytes == 0U ? 1U : bytes;
    if (alignment < runtime->minimum_alignment) {
        alignment = runtime->minimum_alignment;
    }
    ShadowSpillAllocationRecord *selected = NULL;
    const uint64_t reusable_bucket = mix_index(
        required, runtime->reusable_index_bucket_count
    );
    for (ShadowSpillAllocationRecord *candidate = exact_only
             ? runtime->reusable_by_size[reusable_bucket]
             : runtime->active_allocations;
         candidate != NULL;
         candidate = exact_only
             ? candidate->reusable_index_next
             : candidate->active_next) {
        if (!candidate->logical_freed || candidate->pointer == NULL ||
            candidate->ever_plan_owned || candidate->charged_bytes < required ||
            candidate->offset % alignment != 0U ||
            (exact_only && candidate->charged_bytes != required)) {
            continue;
        }
        const int task_local = candidate->retirement_events == NULL &&
            candidate->retirement_fence == NULL &&
            candidate->release_task_id == origin_task_id &&
            origin_task_id != SHADOWSPILL_RUNTIME_NO_ID;
        if (!task_local && candidate->retirement_events == NULL &&
            candidate->retirement_fence == NULL) {
            continue;
        }
        int stream_compatible = 1;
        for (ShadowSpillStreamRecord *used = candidate->streams;
             used != NULL; used = used->next) {
            if (!stream_equal(used->stream, stream)) {
                stream_compatible = 0;
                break;
            }
        }
        if (!stream_compatible) {
            continue;
        }
        if (selected == NULL ||
            candidate->charged_bytes < selected->charged_bytes ||
            (candidate->charged_bytes == selected->charged_bytes &&
             candidate->offset > selected->offset)) {
            selected = candidate;
        }
    }
    if (selected == NULL) {
        *record = NULL;
        return SHADOWSPILL_RUNTIME_OK;
    }
    ShadowSpillAllocationRecord *split = NULL;
    if (selected->charged_bytes > required) {
        split = calloc(1U, sizeof(*split));
        if (split == NULL) {
            return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
        }
    }
    for (ShadowSpillEventRecord *event = selected->retirement_events;
         event != NULL; event = event->next) {
        if (runtime->backend.wait_event(
                runtime->backend.context, stream, event->event
            ) != 0) {
            shadowspill_latch_failure_locked(
                runtime,
                SHADOWSPILL_RUNTIME_BACKEND_FAILURE,
                SHADOWSPILL_RUNTIME_NO_ID,
                selected->allocation_id,
                bytes
            );
            free(split);
            return SHADOWSPILL_RUNTIME_BACKEND_FAILURE;
        }
    }
    unindex_reusable_locked(runtime, selected);
    if (split != NULL) {
        const uint64_t allocation_offset = selected->offset;
        unindex_allocation_pointer_locked(runtime, selected);
        runtime->requested_allocated_bytes -= selected->requested_bytes;
        selected->requested_bytes = 0U;
        selected->charged_bytes -= required;
        selected->offset += required;
        selected->pointer =
            (void *)((unsigned char *)runtime->device_slab + selected->offset);
        index_allocation_pointer_locked(runtime, selected);
        index_reusable_locked(runtime, selected);

        split->allocation_id = runtime->next_allocation_id++;
        split->generation = runtime->next_generation++;
        split->requested_bytes = bytes;
        split->charged_bytes = required;
        split->offset = allocation_offset;
        split->origin_task_id = origin_task_id;
        split->release_task_id = SHADOWSPILL_RUNTIME_NO_ID;
        split->handoff_from_object_id = SHADOWSPILL_RUNTIME_NO_ID;
        split->handoff_to_object_id = SHADOWSPILL_RUNTIME_NO_ID;
        split->handoff_task_id = SHADOWSPILL_RUNTIME_NO_ID;
        split->pointer =
            (void *)((unsigned char *)runtime->device_slab + allocation_offset);
        split->next = runtime->allocations;
        runtime->allocations = split;
        activate_allocation_locked(runtime, split);
        index_allocation_id_locked(runtime, split);
        index_allocation_pointer_locked(runtime, split);
        runtime->requested_allocated_bytes += bytes;
        if (runtime->requested_allocated_bytes >
            runtime->peak_requested_allocated_bytes) {
            runtime->peak_requested_allocated_bytes =
                runtime->requested_allocated_bytes;
        }
        ++runtime->live_allocations;
        shadowspill_append_allocation_event_locked(
            runtime,
            split,
            SHADOWSPILL_ALLOCATION_CREATED,
            SHADOWSPILL_ALLOCATION_ANONYMOUS
        );
        *record = split;
        return runtime->failure.status == SHADOWSPILL_RUNTIME_OK
            ? SHADOWSPILL_RUNTIME_OK
            : (ShadowSpillRuntimeStatus)runtime->failure.status;
    }
    ShadowSpillEventRecord *event = selected->retirement_events;
    selected->retirement_events = NULL;
    int destroy_failed = 0;
    while (event != NULL) {
        ShadowSpillEventRecord *next = event->next;
        if (runtime->backend.destroy_event(
                runtime->backend.context, event->event
            ) != 0) {
            destroy_failed = 1;
        }
        free(event);
        event = next;
    }
    if (runtime->pending_retirements != 0U) {
        --runtime->pending_retirements;
    }
    if (selected->retirement_fence != NULL) {
        shadowspill_release_task_fence_locked(
            runtime, selected->retirement_fence
        );
        selected->retirement_fence = NULL;
    }
    if (destroy_failed) {
        shadowspill_latch_failure_locked(
            runtime,
            SHADOWSPILL_RUNTIME_BACKEND_FAILURE,
            SHADOWSPILL_RUNTIME_NO_ID,
            selected->allocation_id,
            bytes
        );
        return SHADOWSPILL_RUNTIME_BACKEND_FAILURE;
    }
    runtime->requested_allocated_bytes -= selected->requested_bytes;
    free_stream_records(selected->streams);
    selected->streams = NULL;
    unindex_allocation_id_locked(runtime, selected);
    selected->allocation_id = runtime->next_allocation_id++;
    index_allocation_id_locked(runtime, selected);
    selected->generation = runtime->next_generation++;
    selected->requested_bytes = bytes;
    selected->origin_task_id = origin_task_id;
    selected->release_task_id = SHADOWSPILL_RUNTIME_NO_ID;
    selected->handoff_from_object_id = SHADOWSPILL_RUNTIME_NO_ID;
    selected->handoff_to_object_id = SHADOWSPILL_RUNTIME_NO_ID;
    selected->handoff_task_id = SHADOWSPILL_RUNTIME_NO_ID;
    selected->logical_freed = 0;
    selected->framework_free_seen = 0;
    selected->plan_owned = 0;
    selected->ever_plan_owned = 0;
    runtime->requested_allocated_bytes += bytes;
    if (runtime->requested_allocated_bytes >
        runtime->peak_requested_allocated_bytes) {
        runtime->peak_requested_allocated_bytes =
            runtime->requested_allocated_bytes;
    }
    shadowspill_append_allocation_event_locked(
        runtime,
        selected,
        SHADOWSPILL_ALLOCATION_CREATED,
        SHADOWSPILL_ALLOCATION_ANONYMOUS
    );
    *record = selected;
    return runtime->failure.status == SHADOWSPILL_RUNTIME_OK
        ? SHADOWSPILL_RUNTIME_OK
        : (ShadowSpillRuntimeStatus)runtime->failure.status;
}

void shadowspill_release_allocation_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillAllocationRecord *allocation
) {
    if (allocation->pointer == NULL) {
        return;
    }
    unindex_reusable_locked(runtime, allocation);
    shadowspill_append_allocation_event_locked(
        runtime,
        allocation,
        SHADOWSPILL_ALLOCATION_RELEASED,
        allocation->plan_owned ? SHADOWSPILL_ALLOCATION_PLANNED_OBJECT
                               : SHADOWSPILL_ALLOCATION_ANONYMOUS
    );
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
    const int retain_framework_lookup = allocation->ever_plan_owned &&
        !allocation->framework_free_seen;
    if (!retain_framework_lookup) {
        unindex_allocation_pointer_locked(runtime, allocation);
        unindex_allocation_id_locked(runtime, allocation);
    }
    deactivate_allocation_locked(allocation);
    allocation->pointer = NULL;
    allocation->logical_freed = 1;
    allocation->plan_owned = 0;
    runtime->requested_allocated_bytes -= allocation->requested_bytes;
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
    if (runtime == NULL || allocation == NULL || alignment == 0U) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&runtime->mutex);
    ShadowSpillRuntimeStatus status = shadowspill_current_status_locked(runtime);
    while (status == SHADOWSPILL_RUNTIME_OK) {
        ShadowSpillAllocationRecord *record = NULL;
        status = reuse_pending_allocation_locked(
            runtime,
            bytes,
            alignment,
            stream,
            shadowspill_current_task_id(runtime),
            1,
            &record
        );
        if (status == SHADOWSPILL_RUNTIME_OK && record == NULL) {
            status = shadowspill_allocate_locked(
                runtime,
                bytes,
                alignment,
                0,
                shadowspill_current_task_id(runtime),
                &record
            );
        }
        if (status == SHADOWSPILL_RUNTIME_OUT_OF_MEMORY) {
            status = reuse_pending_allocation_locked(
                runtime,
                bytes,
                alignment,
                stream,
                shadowspill_current_task_id(runtime),
                0,
                &record
            );
            if (status == SHADOWSPILL_RUNTIME_OK && record == NULL) {
                status = SHADOWSPILL_RUNTIME_OUT_OF_MEMORY;
            }
        }
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
        const uint64_t task_id = shadowspill_current_task_id(runtime);
        if (task_id != SHADOWSPILL_RUNTIME_NO_ID) {
            status = fence_task_retirements_locked(
                runtime, task_id, stream
            );
            if (status != SHADOWSPILL_RUNTIME_OK) {
                shadowspill_latch_failure_locked(
                    runtime,
                    status,
                    SHADOWSPILL_RUNTIME_NO_ID,
                    SHADOWSPILL_RUNTIME_NO_ID,
                    bytes
                );
                break;
            }
        }
        shadowspill_append_trace_event_locked(
            runtime,
            SHADOWSPILL_TRACE_ALLOCATION_WAIT_BEGIN,
            task_id,
            SHADOWSPILL_RUNTIME_NO_ID,
            SHADOWSPILL_RUNTIME_NO_ID,
            bytes,
            shadowspill_range_free_bytes(&runtime->device_ranges),
            shadowspill_range_largest_free(&runtime->device_ranges)
        );
        ++runtime->blocked_allocators;
        pthread_cond_wait(&runtime->condition, &runtime->mutex);
        --runtime->blocked_allocators;
        shadowspill_append_trace_event_locked(
            runtime,
            SHADOWSPILL_TRACE_ALLOCATION_WAIT_END,
            task_id,
            SHADOWSPILL_RUNTIME_NO_ID,
            SHADOWSPILL_RUNTIME_NO_ID,
            bytes,
            shadowspill_range_free_bytes(&runtime->device_ranges),
            shadowspill_range_largest_free(&runtime->device_ranges)
        );
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
    if (allocation == NULL) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
        goto done;
    }
    if (allocation->logical_freed) {
        if (allocation->ever_plan_owned && !allocation->framework_free_seen) {
            allocation->framework_free_seen = 1;
            unindex_allocation_pointer_locked(runtime, allocation);
            unindex_allocation_id_locked(runtime, allocation);
            goto done;
        }
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
        goto done;
    }
    if (allocation->plan_owned) {
        if (allocation->framework_free_seen) {
            status = SHADOWSPILL_RUNTIME_INVALID_STATE;
            goto done;
        }
        allocation->framework_free_seen = 1;
        goto done;
    }
    if (allocation->ever_plan_owned) {
        if (allocation->framework_free_seen) {
            status = SHADOWSPILL_RUNTIME_INVALID_STATE;
            goto done;
        }
        allocation->framework_free_seen = 1;
    }
    if (append_stream(allocation, stream) != 0) {
        status = SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
        goto done;
    }
    const uint64_t task_id = shadowspill_current_task_id(runtime);
    const int task_local_same_stream =
        task_id != SHADOWSPILL_RUNTIME_NO_ID &&
        allocation->streams != NULL &&
        allocation->streams->next == NULL &&
        stream_equal(allocation->streams->stream, stream);
    ShadowSpillEventRecord *events = NULL;
    if (task_local_same_stream) {
        allocation->release_task_id = task_id;
        allocation->logical_freed = 1;
        index_reusable_locked(runtime, allocation);
        shadowspill_append_allocation_event_locked(
            runtime,
            allocation,
            SHADOWSPILL_ALLOCATION_LOGICAL_FREED,
            SHADOWSPILL_ALLOCATION_ANONYMOUS
        );
        ++runtime->pending_retirements;
        if (runtime->failure.status != SHADOWSPILL_RUNTIME_OK) {
            status = (ShadowSpillRuntimeStatus)runtime->failure.status;
        }
        goto done;
    }
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
    allocation->release_task_id = task_id;
    allocation->logical_freed = 1;
    index_reusable_locked(runtime, allocation);
    shadowspill_append_allocation_event_locked(
        runtime,
        allocation,
        SHADOWSPILL_ALLOCATION_LOGICAL_FREED,
        SHADOWSPILL_ALLOCATION_ANONYMOUS
    );
    ++runtime->pending_retirements;
    pthread_cond_broadcast(&runtime->condition);
    if (runtime->failure.status != SHADOWSPILL_RUNTIME_OK) {
        status = (ShadowSpillRuntimeStatus)runtime->failure.status;
    }

done:
    pthread_mutex_unlock(&runtime->mutex);
    return status;
}

void shadowspill_finalize_aborted_task_retirements(
    ShadowSpillRuntime *runtime,
    uint64_t task_id
) {
    if (runtime == NULL || task_id == SHADOWSPILL_RUNTIME_NO_ID) {
        return;
    }
    pthread_mutex_lock(&runtime->mutex);
    for (ShadowSpillAllocationRecord *allocation = runtime->active_allocations;
         allocation != NULL; allocation = allocation->active_next) {
        if (!allocation->logical_freed || allocation->pointer == NULL ||
            allocation->release_task_id != task_id ||
            allocation->retirement_events != NULL ||
            allocation->retirement_fence != NULL) {
            continue;
        }
        ShadowSpillEventRecord *events = NULL;
        for (ShadowSpillStreamRecord *item = allocation->streams;
             item != NULL; item = item->next) {
            ShadowSpillEventRecord *event = calloc(1U, sizeof(*event));
            int created = 0;
            if (event != NULL && runtime->backend.create_event(
                    runtime->backend.context, &event->event
                ) == 0) {
                created = 1;
            }
            if (event == NULL || !created || runtime->backend.record_event(
                    runtime->backend.context, event->event, item->stream
                ) != 0) {
                if (created) {
                    (void)runtime->backend.destroy_event(
                        runtime->backend.context, event->event
                    );
                }
                free(event);
                destroy_event_list(runtime, events);
                shadowspill_latch_failure_locked(
                    runtime,
                    SHADOWSPILL_RUNTIME_BACKEND_FAILURE,
                    SHADOWSPILL_RUNTIME_NO_ID,
                    allocation->allocation_id,
                    0U
                );
                break;
            }
            event->next = events;
            events = event;
        }
        allocation->retirement_events = events;
        if (runtime->failure.status != SHADOWSPILL_RUNTIME_OK) {
            break;
        }
    }
    pthread_cond_broadcast(&runtime->condition);
    pthread_mutex_unlock(&runtime->mutex);
}
