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

void shadowspill_publish_device_geometry_locked(ShadowSpillRuntime *runtime) {
    atomic_store_explicit(
        &runtime->device_free_bytes_snapshot,
        shadowspill_memory_pool_free_bytes_locked(&runtime->device_pool),
        memory_order_release
    );
    atomic_store_explicit(
        &runtime->device_largest_free_snapshot,
        shadowspill_memory_pool_largest_free_locked(&runtime->device_pool),
        memory_order_release
    );
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
        : shadowspill_memory_pool_pointer(
            &runtime->device_pool, allocation->offset
        );
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
    if (fence == NULL || atomic_fetch_sub_explicit(
            &fence->references, 1U, memory_order_acq_rel
        ) != 1U) {
        return;
    }
    if (shadowspill_event_lease_release(runtime, fence->event) != 0) {
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

void shadowspill_retain_task_fence(ShadowSpillTaskFence *fence) {
    if (fence != NULL) {
        (void)atomic_fetch_add_explicit(
            &fence->references, 1U, memory_order_relaxed
        );
    }
}

int shadowspill_task_fence_complete_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillTaskFence *fence,
    int *complete
) {
    if (fence == NULL || complete == NULL) {
        return -1;
    }
    (void)runtime;
    *complete = atomic_load_explicit(
        &fence->event->completion_known, memory_order_acquire
    ) != 0U;
    if (*complete) {
        atomic_store_explicit(
            &fence->completion_known, 1U, memory_order_release
        );
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
            shadowspill_memory_pool_pointer(
                &runtime->device_pool, record->offset
            );
        if (record->logical_freed && record->ever_plan_owned &&
            !record->framework_free_seen && retired_pointer == pointer) {
            return record;
        }
    }
    return NULL;
}

static int has_release_source(const ShadowSpillRuntime *runtime) {
    return atomic_load_explicit(
        &runtime->pending_retirements, memory_order_acquire
    ) != 0U || atomic_load_explicit(
        &runtime->pending_capacity_actions, memory_order_acquire
    ) != 0U;
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
    const ShadowSpillRuntimeStatus event_status = fence == NULL
        ? SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE
        : shadowspill_event_lease_create_locked(runtime, &fence->event);
    if (event_status != SHADOWSPILL_RUNTIME_OK || runtime->backend.record_event(
            runtime->backend.context, fence->event->event, stream
        ) != 0 || shadowspill_completion_submit(
            runtime,
            stream,
            fence->event,
            SHADOWSPILL_RUNTIME_NO_ID,
            SHADOWSPILL_RUNTIME_NO_ID
        ) != SHADOWSPILL_RUNTIME_OK) {
        if (fence != NULL && fence->event != NULL) {
            (void)shadowspill_event_lease_release(runtime, fence->event);
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
        shadowspill_retain_task_fence(fence);
        const ShadowSpillRuntimeStatus enqueue_status =
            shadowspill_retirement_enqueue_locked(runtime, allocation);
        if (enqueue_status != SHADOWSPILL_RUNTIME_OK) {
            return enqueue_status;
        }
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
    if (alignment < runtime->device_pool.minimum_alignment) {
        alignment = runtime->device_pool.minimum_alignment;
    }
    uint64_t offset = 0U;
    int range_status = shadowspill_memory_pool_reserve_locked(
        &runtime->device_pool,
        charged,
        alignment,
        plan_owned ? SHADOWSPILL_MEMORY_BEST_FIT_LOW
                   : SHADOWSPILL_MEMORY_BEST_FIT_HIGH,
        &offset
    );
    if (range_status > 0) {
        return SHADOWSPILL_RUNTIME_OUT_OF_MEMORY;
    }
    if (range_status < 0) {
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    shadowspill_publish_device_geometry_locked(runtime);
    ShadowSpillRuntimeStatus status =
        shadowspill_adopt_reserved_device_range_locked(
            runtime, bytes, offset, plan_owned, origin_task_id, record
        );
    if (status != SHADOWSPILL_RUNTIME_OK) {
        (void)shadowspill_memory_pool_release_locked(
            &runtime->device_pool, offset, charged
        );
        shadowspill_publish_device_geometry_locked(runtime);
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
    atomic_init(&created->references, 1U);
    created->generation = runtime->next_generation++;
    created->requested_bytes = bytes;
    created->charged_bytes = charged;
    created->offset = offset;
    created->origin_task_id = origin_task_id;
    created->release_task_id = SHADOWSPILL_RUNTIME_NO_ID;
    created->bound_object_id = SHADOWSPILL_RUNTIME_NO_ID;
    created->handoff_from_object_id = SHADOWSPILL_RUNTIME_NO_ID;
    created->handoff_to_object_id = SHADOWSPILL_RUNTIME_NO_ID;
    created->handoff_task_id = SHADOWSPILL_RUNTIME_NO_ID;
    created->pointer = shadowspill_memory_pool_pointer(
        &runtime->device_pool, offset
    );
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
    if (shadowspill_failure_status(runtime) != SHADOWSPILL_RUNTIME_OK) {
        unindex_allocation_pointer_locked(runtime, created);
        unindex_allocation_id_locked(runtime, created);
        runtime->allocations = created->next;
        deactivate_allocation_locked(created);
        runtime->requested_allocated_bytes -= bytes;
        --runtime->live_allocations;
        free(created);
        return shadowspill_failure_status(runtime);
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
    if (alignment < runtime->device_pool.minimum_alignment) {
        alignment = runtime->device_pool.minimum_alignment;
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
            candidate->retirement_preparing || candidate->ever_plan_owned ||
            candidate->charged_bytes < required ||
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
    /*
     * Every recorded use was on ``stream`` (checked above). CUDA stream order
     * already retires the old use before later work that consumes the reused
     * range, so adding a wait for an event recorded on the same stream is both
     * redundant and an unnecessary driver call in the allocator hot path.
     */
    unindex_reusable_locked(runtime, selected);
    if (split != NULL) {
        const uint64_t allocation_offset = selected->offset;
        unindex_allocation_pointer_locked(runtime, selected);
        runtime->requested_allocated_bytes -= selected->requested_bytes;
        selected->requested_bytes = 0U;
        selected->charged_bytes -= required;
        selected->offset += required;
        selected->pointer =
            shadowspill_memory_pool_pointer(
                &runtime->device_pool, selected->offset
            );
        index_allocation_pointer_locked(runtime, selected);
        index_reusable_locked(runtime, selected);

        split->allocation_id = runtime->next_allocation_id++;
        atomic_init(&split->references, 1U);
        split->generation = runtime->next_generation++;
        split->requested_bytes = bytes;
        split->charged_bytes = required;
        split->offset = allocation_offset;
        split->origin_task_id = origin_task_id;
        split->release_task_id = SHADOWSPILL_RUNTIME_NO_ID;
        split->bound_object_id = SHADOWSPILL_RUNTIME_NO_ID;
        split->handoff_from_object_id = SHADOWSPILL_RUNTIME_NO_ID;
        split->handoff_to_object_id = SHADOWSPILL_RUNTIME_NO_ID;
        split->handoff_task_id = SHADOWSPILL_RUNTIME_NO_ID;
        split->pointer =
            shadowspill_memory_pool_pointer(
                &runtime->device_pool, allocation_offset
            );
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
        return shadowspill_failure_status(runtime);
    }
    ShadowSpillEventRecord *event = selected->retirement_events;
    selected->retirement_events = NULL;
    int destroy_failed = 0;
    while (event != NULL) {
        ShadowSpillEventRecord *next = event->next;
        if (shadowspill_event_lease_release(runtime, event->event) != 0) {
            destroy_failed = 1;
        }
        free(event);
        event = next;
    }
    if (atomic_fetch_sub_explicit(
            &runtime->pending_retirements, 1U, memory_order_release
        ) == 1U) {
        shadowspill_idle_notify(runtime);
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
    selected->bound_object_id = SHADOWSPILL_RUNTIME_NO_ID;
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
    return shadowspill_failure_status(runtime);
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
    if (shadowspill_memory_pool_release_locked(
            &runtime->device_pool,
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
    shadowspill_publish_device_geometry_locked(runtime);
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
    allocation->bound_object_id = SHADOWSPILL_RUNTIME_NO_ID;
    runtime->requested_allocated_bytes -= allocation->requested_bytes;
    if (runtime->live_allocations != 0U) {
        --runtime->live_allocations;
    }
    pthread_cond_broadcast(&runtime->condition);
    pthread_cond_broadcast(&runtime->device_pool.capacity_changed);
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
    shadowspill_memory_pool_lock_foreground(&runtime->device_pool);
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
            shadowspill_memory_pool_free_bytes_locked(&runtime->device_pool),
            shadowspill_memory_pool_largest_free_locked(&runtime->device_pool)
        );
        ++runtime->blocked_allocators;
        pthread_cond_wait(
            &runtime->device_pool.capacity_changed,
            &runtime->device_pool.lock
        );
        --runtime->blocked_allocators;
        shadowspill_append_trace_event_locked(
            runtime,
            SHADOWSPILL_TRACE_ALLOCATION_WAIT_END,
            task_id,
            SHADOWSPILL_RUNTIME_NO_ID,
            SHADOWSPILL_RUNTIME_NO_ID,
            bytes,
            shadowspill_memory_pool_free_bytes_locked(&runtime->device_pool),
            shadowspill_memory_pool_largest_free_locked(&runtime->device_pool)
        );
        status = shadowspill_current_status_locked(runtime);
    }
    shadowspill_memory_pool_unlock_foreground(&runtime->device_pool);
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
    shadowspill_memory_pool_lock_foreground(&runtime->device_pool);
    ShadowSpillAllocationRecord *record =
        shadowspill_find_allocation_by_pointer(runtime, pointer);
    if (record == NULL) {
        shadowspill_memory_pool_unlock_foreground(&runtime->device_pool);
        return SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    *allocation = (ShadowSpillAllocation){
        .allocation_id = record->allocation_id,
        .generation = record->generation,
        .requested_bytes = record->requested_bytes,
        .charged_bytes = record->charged_bytes,
        .pointer = record->pointer,
    };
    shadowspill_memory_pool_unlock_foreground(&runtime->device_pool);
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
    shadowspill_memory_pool_lock_foreground(&runtime->device_pool);
    ShadowSpillAllocationRecord *allocation = shadowspill_find_allocation(
        runtime, allocation_id
    );
    ShadowSpillRuntimeStatus status = SHADOWSPILL_RUNTIME_OK;
    if (allocation == NULL || allocation->logical_freed) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
    } else if (append_stream(allocation, stream) != 0) {
        status = SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    shadowspill_memory_pool_unlock_foreground(&runtime->device_pool);
    return status;
}

static void destroy_event_list(
    ShadowSpillRuntime *runtime,
    ShadowSpillEventRecord *events
) {
    while (events != NULL) {
        ShadowSpillEventRecord *next = events->next;
        (void)shadowspill_event_lease_release(runtime, events->event);
        free(events);
        events = next;
    }
}

typedef struct ShadowSpillStreamSnapshot {
    ShadowSpillBackendStream *streams;
    uint64_t count;
} ShadowSpillStreamSnapshot;

static int snapshot_streams_locked(
    const ShadowSpillAllocationRecord *allocation,
    ShadowSpillStreamSnapshot *snapshot
) {
    uint64_t count = 0U;
    for (const ShadowSpillStreamRecord *item = allocation->streams;
         item != NULL; item = item->next) {
        ++count;
    }
    ShadowSpillBackendStream *streams = count == 0U
        ? NULL
        : calloc((size_t)count, sizeof(*streams));
    if (count != 0U && streams == NULL) {
        return -1;
    }
    uint64_t index = 0U;
    for (const ShadowSpillStreamRecord *item = allocation->streams;
         item != NULL; item = item->next) {
        streams[index++] = item->stream;
    }
    *snapshot = (ShadowSpillStreamSnapshot){
        .streams = streams,
        .count = count,
    };
    return 0;
}

static ShadowSpillRuntimeStatus record_retirement_events(
    ShadowSpillRuntime *runtime,
    const ShadowSpillStreamSnapshot *snapshot,
    uint64_t allocation_id,
    ShadowSpillEventRecord **events
) {
    *events = NULL;
    for (uint64_t index = 0U; index < snapshot->count; ++index) {
        ShadowSpillEventRecord *event = calloc(1U, sizeof(*event));
        const ShadowSpillRuntimeStatus event_status = event == NULL
            ? SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE
            : shadowspill_event_lease_create_locked(runtime, &event->event);
        if (event_status != SHADOWSPILL_RUNTIME_OK ||
            runtime->backend.record_event(
                runtime->backend.context,
                event->event->event,
                snapshot->streams[index]
            ) != 0 || shadowspill_completion_submit(
                runtime,
                snapshot->streams[index],
                event->event,
                SHADOWSPILL_RUNTIME_NO_ID,
                allocation_id
            ) != SHADOWSPILL_RUNTIME_OK) {
            if (event != NULL && event->event != NULL) {
                (void)shadowspill_event_lease_release(runtime, event->event);
            }
            free(event);
            destroy_event_list(runtime, *events);
            *events = NULL;
            return event_status == SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE
                ? event_status
                : SHADOWSPILL_RUNTIME_BACKEND_FAILURE;
        }
        event->next = *events;
        *events = event;
    }
    return SHADOWSPILL_RUNTIME_OK;
}

ShadowSpillRuntimeStatus shadowspill_free(
    ShadowSpillRuntime *runtime,
    uint64_t allocation_id,
    ShadowSpillBackendStream stream
) {
    if (runtime == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    shadowspill_memory_pool_lock_foreground(&runtime->device_pool);
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
        if (shadowspill_failure_status(runtime) != SHADOWSPILL_RUNTIME_OK) {
            status = shadowspill_failure_status(runtime);
        }
        goto done;
    }
    ShadowSpillStreamSnapshot snapshot = {0};
    if (snapshot_streams_locked(allocation, &snapshot) != 0) {
        status = SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
        goto done;
    }
    const uint64_t generation = allocation->generation;
    allocation->release_task_id = task_id;
    allocation->logical_freed = 1;
    allocation->retirement_preparing = 1U;
    shadowspill_append_allocation_event_locked(
        runtime,
        allocation,
        SHADOWSPILL_ALLOCATION_LOGICAL_FREED,
        SHADOWSPILL_ALLOCATION_ANONYMOUS
    );
    ++runtime->pending_retirements;
    shadowspill_memory_pool_unlock_foreground(&runtime->device_pool);

    ShadowSpillEventRecord *events = NULL;
    status = record_retirement_events(
        runtime, &snapshot, allocation_id, &events
    );
    free(snapshot.streams);

    shadowspill_memory_pool_lock_foreground(&runtime->device_pool);
    allocation = shadowspill_find_allocation(runtime, allocation_id);
    if (allocation == NULL || allocation->generation != generation ||
        !allocation->retirement_preparing) {
        destroy_event_list(runtime, events);
        if (status == SHADOWSPILL_RUNTIME_OK) {
            status = SHADOWSPILL_RUNTIME_INVALID_STATE;
        }
    } else {
        allocation->retirement_events = events;
        allocation->retirement_preparing = 0U;
        if (status == SHADOWSPILL_RUNTIME_OK) {
            index_reusable_locked(runtime, allocation);
            status = shadowspill_retirement_enqueue_locked(
                runtime, allocation
            );
        }
    }
    if (status != SHADOWSPILL_RUNTIME_OK) {
        shadowspill_latch_failure_locked(
            runtime, status, SHADOWSPILL_RUNTIME_NO_ID, allocation_id, 0U
        );
    }
    pthread_cond_broadcast(&runtime->condition);
    if (shadowspill_failure_status(runtime) != SHADOWSPILL_RUNTIME_OK) {
        status = shadowspill_failure_status(runtime);
    }

done:
    shadowspill_memory_pool_unlock_foreground(&runtime->device_pool);
    return status;
}

void shadowspill_finalize_aborted_task_retirements(
    ShadowSpillRuntime *runtime,
    uint64_t task_id
) {
    if (runtime == NULL || task_id == SHADOWSPILL_RUNTIME_NO_ID) {
        return;
    }
    shadowspill_memory_pool_lock_foreground(&runtime->device_pool);
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
            const ShadowSpillRuntimeStatus event_status = event == NULL
                ? SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE
                : shadowspill_event_lease_create_locked(runtime, &event->event);
            if (event_status != SHADOWSPILL_RUNTIME_OK ||
                runtime->backend.record_event(
                    runtime->backend.context, event->event->event, item->stream
                ) != 0 || shadowspill_completion_submit(
                    runtime,
                    item->stream,
                    event->event,
                    SHADOWSPILL_RUNTIME_NO_ID,
                    allocation->allocation_id
                ) != SHADOWSPILL_RUNTIME_OK) {
                if (event != NULL && event->event != NULL) {
                    (void)shadowspill_event_lease_release(runtime, event->event);
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
        if (shadowspill_failure_status(runtime) != SHADOWSPILL_RUNTIME_OK) {
            break;
        }
        const ShadowSpillRuntimeStatus enqueue_status =
            shadowspill_retirement_enqueue_locked(runtime, allocation);
        if (enqueue_status != SHADOWSPILL_RUNTIME_OK) {
            shadowspill_latch_failure_locked(
                runtime,
                enqueue_status,
                SHADOWSPILL_RUNTIME_NO_ID,
                allocation->allocation_id,
                0U
            );
            break;
        }
    }
    pthread_cond_broadcast(&runtime->condition);
    pthread_cond_broadcast(&runtime->device_pool.capacity_changed);
    shadowspill_memory_pool_unlock_foreground(&runtime->device_pool);
}
