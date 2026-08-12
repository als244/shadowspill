#include "internal.h"

#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int shadowspill_backend_is_valid(const ShadowSpillBackend *backend) {
    return backend != NULL &&
        backend->abi_version == SHADOWSPILL_BACKEND_ABI_VERSION &&
        backend->allocate_device != NULL && backend->free_device != NULL &&
        backend->allocate_host != NULL && backend->free_host != NULL &&
        backend->create_stream != NULL && backend->destroy_stream != NULL &&
        backend->create_event != NULL && backend->destroy_event != NULL &&
        backend->record_event != NULL && backend->query_event != NULL &&
        backend->wait_event != NULL && backend->copy_async != NULL &&
        backend->synchronize_stream != NULL;
}

static void destroy_allocations(ShadowSpillRuntime *runtime) {
    ShadowSpillAllocationRecord *allocation = runtime->allocations;
    while (allocation != NULL) {
        ShadowSpillAllocationRecord *next = allocation->next;
        ShadowSpillStreamRecord *stream = allocation->streams;
        while (stream != NULL) {
            ShadowSpillStreamRecord *stream_next = stream->next;
            free(stream);
            stream = stream_next;
        }
        ShadowSpillEventRecord *event = allocation->retirement_events;
        while (event != NULL) {
            ShadowSpillEventRecord *event_next = event->next;
            (void)shadowspill_event_lease_release(runtime, event->event);
            free(event);
            event = event_next;
        }
        if (allocation->retirement_fence != NULL) {
            shadowspill_release_task_fence_locked(
                runtime, allocation->retirement_fence
            );
            allocation->retirement_fence = NULL;
        }
        free(allocation);
        allocation = next;
    }
    runtime->allocations = NULL;
}

static void destroy_objects(ShadowSpillRuntime *runtime) {
    for (ShadowSpillObjectRecord *object = runtime->objects.owned_head;
         object != NULL; object = object->ownership_next) {
        if (object->readiness_event != NULL) {
            (void)shadowspill_event_lease_release(
                runtime, object->readiness_event
            );
            object->readiness_event = NULL;
            object->has_readiness_event = 0U;
        }
    }
    shadowspill_object_table_destroy(&runtime->objects);
}

static void destroy_actions(ShadowSpillRuntime *runtime) {
    ShadowSpillQueuedAction *action = runtime->action_head;
    while (action != NULL) {
        ShadowSpillQueuedAction *next = action->next;
        if (action->has_completion_event) {
            (void)shadowspill_event_lease_release(
                runtime, action->completion_event
            );
        }
        if (action->destination_reserved) {
            ShadowSpillRangeAllocator *ranges =
                action->kind == SHADOWSPILL_RUNTIME_PREFETCH
                ? &runtime->device_ranges
                : &runtime->host_ranges;
            (void)shadowspill_range_free(
                ranges,
                action->destination_offset,
                action->destination_bytes
            );
        }
        shadowspill_release_task_fence_locked(runtime, action->fence);
        shadowspill_object_release(action->object);
        free(action);
        action = next;
    }
    runtime->action_head = NULL;
    runtime->action_tail = NULL;
}

static void release_resources(ShadowSpillRuntime *runtime) {
    if (runtime->completions_initialized) {
        shadowspill_completion_tracker_destroy(
            runtime, &runtime->completions
        );
        runtime->completions_initialized = 0U;
    }
    destroy_actions(runtime);
    destroy_allocations(runtime);
    destroy_objects(runtime);
    free(runtime->allocations_by_id);
    free(runtime->allocations_by_pointer);
    free(runtime->reusable_by_size);
    runtime->allocations_by_id = NULL;
    runtime->allocations_by_pointer = NULL;
    runtime->reusable_by_size = NULL;
    free(runtime->allocation_events);
    runtime->allocation_events = NULL;
    runtime->allocation_event_count = 0U;
    runtime->allocation_event_capacity = 0U;
    free(runtime->trace_events);
    runtime->trace_events = NULL;
    runtime->trace_event_count = 0U;
    runtime->trace_event_capacity = 0U;
    shadowspill_range_destroy(&runtime->device_ranges);
    shadowspill_range_destroy(&runtime->host_ranges);
    if (runtime->h2d_stream_created) {
        (void)runtime->backend.destroy_stream(
            runtime->backend.context, runtime->h2d_stream
        );
        runtime->h2d_stream_created = 0;
    }
    if (runtime->d2h_stream_created) {
        (void)runtime->backend.destroy_stream(
            runtime->backend.context, runtime->d2h_stream
        );
        runtime->d2h_stream_created = 0;
    }
    if (runtime->device_slab != NULL) {
        (void)runtime->backend.free_device(
            runtime->backend.context, runtime->device_slab
        );
        runtime->device_slab = NULL;
    }
    if (runtime->host_arena != NULL) {
        (void)runtime->backend.free_host(
            runtime->backend.context, runtime->host_arena
        );
        runtime->host_arena = NULL;
    }
}

ShadowSpillRuntimeStatus shadowspill_runtime_create_legacy(
    const ShadowSpillRuntimeConfig *config,
    ShadowSpillRuntime **output
) {
    if (config == NULL || output == NULL ||
        config->abi_version != SHADOWSPILL_RUNTIME_ABI_VERSION ||
        config->device_slab_bytes == 0U || config->minimum_alignment == 0U ||
        !shadowspill_backend_is_valid(&config->backend)) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    *output = NULL;
    ShadowSpillRuntime *runtime = calloc(1U, sizeof(*runtime));
    if (runtime == NULL) {
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    runtime->backend = config->backend;
    runtime->progress_poll_nanoseconds = config->progress_poll_nanoseconds;
    runtime->minimum_alignment = config->minimum_alignment;
    runtime->next_allocation_id = 1U;
    runtime->next_generation = 1U;
    atomic_init(&runtime->next_event_generation, 1U);
    runtime->failure.object_id = SHADOWSPILL_RUNTIME_NO_ID;
    runtime->failure.allocation_id = SHADOWSPILL_RUNTIME_NO_ID;
    atomic_init(&runtime->closing, 0U);
    atomic_init(&runtime->closed, 0U);
    atomic_init(&runtime->worker_stop, 0U);
    atomic_init(&runtime->failure_status, SHADOWSPILL_RUNTIME_OK);
    atomic_init(&runtime->pending_retirements, 0U);
    atomic_init(&runtime->pending_capacity_actions, 0U);
    atomic_init(&runtime->device_free_bytes_snapshot, 0U);
    atomic_init(&runtime->device_largest_free_snapshot, 0U);
    atomic_init(&runtime->allocation_event_count, 0U);
    atomic_init(&runtime->next_allocation_event_sequence, 0U);
    atomic_init(&runtime->allocation_telemetry_active, 0U);
    atomic_init(&runtime->allocation_event_overflow, 0U);
    atomic_init(&runtime->trace_event_count, 0U);
    atomic_init(&runtime->next_trace_event_sequence, 0U);
    atomic_init(&runtime->trace_prepared, 0U);
    atomic_init(&runtime->trace_active, 0U);
    atomic_init(&runtime->trace_event_overflow, 0U);
    runtime->allocation_index_bucket_count = 65536U;
    runtime->reusable_index_bucket_count = 8192U;
    const uint64_t object_index_bucket_count = 16384U;
    runtime->allocations_by_id = calloc(
        (size_t)runtime->allocation_index_bucket_count,
        sizeof(*runtime->allocations_by_id)
    );
    runtime->allocations_by_pointer = calloc(
        (size_t)runtime->allocation_index_bucket_count,
        sizeof(*runtime->allocations_by_pointer)
    );
    runtime->reusable_by_size = calloc(
        (size_t)runtime->reusable_index_bucket_count,
        sizeof(*runtime->reusable_by_size)
    );
    if (runtime->allocations_by_id == NULL ||
        runtime->allocations_by_pointer == NULL ||
        runtime->reusable_by_size == NULL ||
        shadowspill_object_table_initialize(
            &runtime->objects, object_index_bucket_count
        ) != 0 || shadowspill_completion_tracker_initialize(
            &runtime->completions
        ) != 0) {
        free(runtime->allocations_by_id);
        free(runtime->allocations_by_pointer);
        free(runtime->reusable_by_size);
        shadowspill_object_table_destroy(&runtime->objects);
        free(runtime);
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    runtime->completions_initialized = 1U;
    if (pthread_mutex_init(&runtime->failure_lock, NULL) != 0) {
        release_resources(runtime);
        free(runtime);
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    if (pthread_mutex_init(&runtime->allocation_pool.lock, NULL) != 0) {
        pthread_mutex_destroy(&runtime->failure_lock);
        release_resources(runtime);
        free(runtime);
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    if (pthread_cond_init(
            &runtime->allocation_pool.capacity_changed, NULL
        ) != 0) {
        pthread_mutex_destroy(&runtime->allocation_pool.lock);
        pthread_mutex_destroy(&runtime->failure_lock);
        release_resources(runtime);
        free(runtime);
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    runtime->allocation_pool_initialized = 1U;
    if (pthread_mutex_init(&runtime->mutex, NULL) != 0) {
        pthread_cond_destroy(&runtime->allocation_pool.capacity_changed);
        pthread_mutex_destroy(&runtime->allocation_pool.lock);
        runtime->allocation_pool_initialized = 0U;
        pthread_mutex_destroy(&runtime->failure_lock);
        release_resources(runtime);
        free(runtime);
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    if (pthread_cond_init(&runtime->condition, NULL) != 0) {
        pthread_mutex_destroy(&runtime->mutex);
        pthread_cond_destroy(&runtime->allocation_pool.capacity_changed);
        pthread_mutex_destroy(&runtime->allocation_pool.lock);
        runtime->allocation_pool_initialized = 0U;
        pthread_mutex_destroy(&runtime->failure_lock);
        release_resources(runtime);
        free(runtime);
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    ShadowSpillRuntimeStatus status = SHADOWSPILL_RUNTIME_BACKEND_FAILURE;
    if (runtime->backend.allocate_device(
            runtime->backend.context,
            config->device_slab_bytes,
            &runtime->device_slab
        ) != 0 ||
        (config->host_arena_bytes != 0U && runtime->backend.allocate_host(
             runtime->backend.context,
             config->host_arena_bytes,
             &runtime->host_arena
         ) != 0)) {
        goto fail;
    }
    if (shadowspill_range_initialize(
            &runtime->device_ranges, config->device_slab_bytes
        ) != 0 || shadowspill_range_initialize(
            &runtime->host_ranges, config->host_arena_bytes
        ) != 0) {
        status = SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
        goto fail;
    }
    shadowspill_publish_device_geometry_locked(runtime);
    if (runtime->backend.create_stream(
            runtime->backend.context,
            SHADOWSPILL_TRANSFER_TO_DEVICE,
            &runtime->h2d_stream
        ) != 0) {
        goto fail;
    }
    runtime->h2d_stream_created = 1;
    if (runtime->backend.create_stream(
            runtime->backend.context,
            SHADOWSPILL_TRANSFER_TO_HOST,
            &runtime->d2h_stream
        ) != 0) {
        goto fail;
    }
    runtime->d2h_stream_created = 1;
    if (pthread_create(
            &runtime->progress_thread, NULL, shadowspill_progress_main, runtime
        ) != 0) {
        status = SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
        goto fail;
    }
    runtime->progress_started = 1;
    *output = runtime;
    return SHADOWSPILL_RUNTIME_OK;

fail:
    release_resources(runtime);
    pthread_cond_destroy(&runtime->condition);
    pthread_mutex_destroy(&runtime->mutex);
    pthread_cond_destroy(&runtime->allocation_pool.capacity_changed);
    pthread_mutex_destroy(&runtime->allocation_pool.lock);
    runtime->allocation_pool_initialized = 0U;
    pthread_mutex_destroy(&runtime->failure_lock);
    free(runtime);
    return status;
}

ShadowSpillRuntimeStatus shadowspill_runtime_wait_idle_legacy(
    ShadowSpillRuntime *runtime
) {
    if (runtime == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&runtime->mutex);
    while (atomic_load_explicit(&runtime->closed, memory_order_acquire) == 0U &&
           shadowspill_failure_status(runtime) == SHADOWSPILL_RUNTIME_OK &&
           (runtime->queued_actions != 0U ||
            runtime->pending_retirements != 0U)) {
        pthread_cond_wait(&runtime->condition, &runtime->mutex);
    }
    ShadowSpillRuntimeStatus status = shadowspill_failure_status(runtime);
    pthread_mutex_unlock(&runtime->mutex);
    return status;
}

ShadowSpillRuntimeStatus shadowspill_runtime_resize_host_arena_legacy(
    ShadowSpillRuntime *runtime,
    uint64_t host_arena_bytes
) {
    if (runtime == NULL || host_arena_bytes > SIZE_MAX) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    ShadowSpillRuntimeStatus status = shadowspill_runtime_wait_idle_legacy(runtime);
    if (status != SHADOWSPILL_RUNTIME_OK) {
        return status;
    }
    pthread_mutex_lock(&runtime->mutex);
    status = shadowspill_current_status_locked(runtime);
    uint64_t current_bytes = runtime->host_ranges.capacity;
    if (status != SHADOWSPILL_RUNTIME_OK) {
        goto done;
    }
    if (atomic_load_explicit(&runtime->closing, memory_order_acquire) != 0U ||
        runtime->queued_actions != 0U ||
        runtime->pending_retirements != 0U) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
        goto done;
    }
    if (host_arena_bytes < current_bytes) {
        status = SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
        goto done;
    }
    if (host_arena_bytes == current_bytes) {
        goto done;
    }

    void *replacement = NULL;
    if (runtime->backend.allocate_host(
            runtime->backend.context, host_arena_bytes, &replacement
        ) != 0) {
        status = SHADOWSPILL_RUNTIME_BACKEND_FAILURE;
        goto done;
    }
    if (current_bytes != 0U) {
        memcpy(replacement, runtime->host_arena, (size_t)current_bytes);
    }
    ShadowSpillRangeAllocator ranges = {0};
    if (shadowspill_range_clone_extended(
            &runtime->host_ranges, host_arena_bytes, &ranges
        ) != 0) {
        (void)runtime->backend.free_host(runtime->backend.context, replacement);
        status = SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
        goto done;
    }
    if (runtime->host_arena != NULL && runtime->backend.free_host(
            runtime->backend.context, runtime->host_arena
        ) != 0) {
        shadowspill_range_destroy(&ranges);
        (void)runtime->backend.free_host(runtime->backend.context, replacement);
        status = SHADOWSPILL_RUNTIME_BACKEND_FAILURE;
        goto done;
    }
    runtime->host_arena = replacement;
    shadowspill_range_destroy(&runtime->host_ranges);
    runtime->host_ranges = ranges;

done:
    pthread_mutex_unlock(&runtime->mutex);
    return status;
}

ShadowSpillRuntimeStatus shadowspill_runtime_close_legacy(
    ShadowSpillRuntime *runtime
) {
    if (runtime == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&runtime->mutex);
    if (atomic_load_explicit(&runtime->closed, memory_order_acquire) != 0U) {
        pthread_mutex_unlock(&runtime->mutex);
        return SHADOWSPILL_RUNTIME_OK;
    }
    atomic_store_explicit(&runtime->closing, 1U, memory_order_release);
    while (shadowspill_failure_status(runtime) == SHADOWSPILL_RUNTIME_OK &&
           (runtime->queued_actions != 0U ||
            runtime->pending_retirements != 0U)) {
        pthread_cond_wait(&runtime->condition, &runtime->mutex);
    }
    pthread_mutex_unlock(&runtime->mutex);

    int synchronization_failed = 0;
    if (runtime->h2d_stream_created && runtime->backend.synchronize_stream(
            runtime->backend.context, runtime->h2d_stream
        ) != 0) {
        synchronization_failed = 1;
    }
    if (runtime->d2h_stream_created && runtime->backend.synchronize_stream(
            runtime->backend.context, runtime->d2h_stream
        ) != 0) {
        synchronization_failed = 1;
    }
    pthread_mutex_lock(&runtime->mutex);
    if (synchronization_failed) {
        shadowspill_latch_failure_locked(
            runtime,
            SHADOWSPILL_RUNTIME_BACKEND_FAILURE,
            SHADOWSPILL_RUNTIME_NO_ID,
            SHADOWSPILL_RUNTIME_NO_ID,
            0U
        );
    }
    atomic_store_explicit(&runtime->worker_stop, 1U, memory_order_release);
    pthread_cond_broadcast(&runtime->condition);
    pthread_mutex_unlock(&runtime->mutex);
    if (runtime->progress_started) {
        (void)pthread_join(runtime->progress_thread, NULL);
        runtime->progress_started = 0;
    }
    pthread_mutex_lock(&runtime->mutex);
    ShadowSpillRuntimeStatus status = shadowspill_failure_status(runtime);
    atomic_store_explicit(&runtime->closed, 1U, memory_order_release);
    pthread_mutex_unlock(&runtime->mutex);
    release_resources(runtime);
    return status;
}

void shadowspill_runtime_destroy_legacy(ShadowSpillRuntime *runtime) {
    if (runtime == NULL) {
        return;
    }
    (void)shadowspill_runtime_close_legacy(runtime);
    pthread_cond_destroy(&runtime->condition);
    pthread_mutex_destroy(&runtime->mutex);
    pthread_cond_destroy(&runtime->allocation_pool.capacity_changed);
    pthread_mutex_destroy(&runtime->allocation_pool.lock);
    runtime->allocation_pool_initialized = 0U;
    pthread_mutex_destroy(&runtime->failure_lock);
    free(runtime);
}

ShadowSpillRuntimeStatus shadowspill_runtime_statistics(
    ShadowSpillRuntime *runtime,
    ShadowSpillRuntimeStatistics *statistics
) {
    if (runtime == NULL || statistics == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&runtime->mutex);
    pthread_mutex_lock(&runtime->allocation_pool.lock);
    *statistics = (ShadowSpillRuntimeStatistics){
        .slab_bytes = runtime->device_ranges.capacity,
        .requested_allocated_bytes = runtime->requested_allocated_bytes,
        .peak_requested_allocated_bytes =
            runtime->peak_requested_allocated_bytes,
        .allocated_bytes = runtime->device_ranges.allocated,
        .free_bytes = shadowspill_range_free_bytes(&runtime->device_ranges),
        .largest_free_range_bytes =
            shadowspill_range_largest_free(&runtime->device_ranges),
        .external_fragmentation_bytes =
            shadowspill_range_free_bytes(&runtime->device_ranges) -
            shadowspill_range_largest_free(&runtime->device_ranges),
        .peak_allocated_bytes = runtime->device_ranges.peak_allocated,
        .host_arena_bytes = runtime->host_ranges.capacity,
        .host_allocated_bytes = runtime->host_ranges.allocated,
        .host_peak_allocated_bytes = runtime->host_ranges.peak_allocated,
        .live_allocations = runtime->live_allocations,
        .blocked_allocators = runtime->blocked_allocators,
        .pending_retirements = runtime->pending_retirements,
        .registered_objects = runtime->registered_objects,
        .queued_actions = runtime->queued_actions,
        .transfers_to_device = runtime->transfers_to_device,
        .transfers_to_host = runtime->transfers_to_host,
        .bytes_to_device = runtime->bytes_to_device,
        .bytes_to_host = runtime->bytes_to_host,
        .wait_events_inserted = runtime->wait_events_inserted,
        .allocation_events = runtime->allocation_event_count,
        .allocation_event_capacity = runtime->allocation_event_capacity,
        .allocation_event_overflow =
            (uint64_t)runtime->allocation_event_overflow,
    };
    pthread_mutex_unlock(&runtime->allocation_pool.lock);
    pthread_mutex_unlock(&runtime->mutex);
    return SHADOWSPILL_RUNTIME_OK;
}
