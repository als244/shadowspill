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
    ShadowSpillMemoryLease *allocation = runtime->execution_leases;
    while (allocation != NULL) {
        ShadowSpillMemoryLease *next = allocation->next;
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
    runtime->execution_leases = NULL;
}

static void destroy_objects(ShadowSpillRuntime *runtime) {
    for (ShadowSpillObject *object = runtime->objects.owned_head;
         object != NULL; object = object->ownership_next) {
        if (object->readiness_event != NULL) {
            (void)shadowspill_event_lease_release(
                runtime, object->readiness_event
            );
            object->readiness_event = NULL;
            object->has_readiness_event = 0U;
        }
    }
    shadowspill_execution_table_destroy(&runtime->execution);
    shadowspill_object_table_destroy(&runtime->objects);
}

static void destroy_actions(ShadowSpillRuntime *runtime) {
    ShadowSpillQueuedAction *action = runtime->actions.head;
    while (action != NULL) {
        ShadowSpillQueuedAction *next = action->next;
        if (action->destination_priority_declared) {
            ShadowSpillMemoryPool *pool =
                action->kind == SHADOWSPILL_RUNTIME_PREFETCH
                ? shadowspill_execution_pool(runtime)
                : shadowspill_spill_pool(runtime);
            shadowspill_memory_pool_relinquish_transfer(pool);
            action->destination_priority_declared = 0U;
        }
        if (action->has_completion_event) {
            (void)shadowspill_event_lease_release(
                runtime, action->completion_event
            );
        }
        if (action->destination_lease != NULL) {
            ShadowSpillMemoryLease *lease = action->destination_lease;
            ShadowSpillMemoryPool *pool = lease->pool;
            pthread_mutex_lock(&pool->lock);
            if (pool->pool_id == runtime->execution_pool_id) {
                lease->release_task_id = action->task_id;
                shadowspill_release_execution_lease_locked(runtime, lease);
            } else {
                (void)shadowspill_memory_pool_release_lease_locked(lease);
                free(lease);
            }
            pthread_mutex_unlock(&pool->lock);
            action->destination_lease = NULL;
        }
        shadowspill_release_task_fence_locked(runtime, action->fence);
        if (!action->admitted) {
            shadowspill_object_release(action->object);
            free(action);
        } else {
            action->active = 0U;
            action->previous = NULL;
            action->next = NULL;
            action->lane_previous = NULL;
            action->lane_next = NULL;
        }
        action = next;
    }
    runtime->actions.head = NULL;
    runtime->actions.tail = NULL;
    atomic_store_explicit(&runtime->actions.count, 0U, memory_order_release);
}

static void release_resources(ShadowSpillRuntime *runtime) {
    if (runtime->completions_initialized) {
        shadowspill_completion_tracker_destroy(
            runtime, &runtime->completions
        );
        runtime->completions_initialized = 0U;
    }
    shadowspill_retirement_queue_destroy(runtime, &runtime->retirements);
    destroy_actions(runtime);
    destroy_allocations(runtime);
    destroy_objects(runtime);
    free(runtime->execution_leases_by_id);
    free(runtime->execution_leases_by_pointer);
    free(runtime->reusable_execution_leases_by_size);
    runtime->execution_leases_by_id = NULL;
    runtime->execution_leases_by_pointer = NULL;
    runtime->reusable_execution_leases_by_size = NULL;
    free(runtime->allocation_events);
    runtime->allocation_events = NULL;
    runtime->allocation_event_count = 0U;
    runtime->allocation_event_capacity = 0U;
    free(runtime->trace_events);
    runtime->trace_events = NULL;
    runtime->trace_event_count = 0U;
    runtime->trace_event_capacity = 0U;
    if (runtime->fetch_stream_created) {
        (void)runtime->backend.destroy_stream(
            runtime->backend.context, runtime->fetch_stream
        );
        runtime->fetch_stream_created = 0;
    }
    if (runtime->evict_stream_created) {
        (void)runtime->backend.destroy_stream(
            runtime->backend.context, runtime->evict_stream
        );
        runtime->evict_stream_created = 0;
    }
    ShadowSpillMemoryPool *execution_pool = shadowspill_execution_pool(runtime);
    ShadowSpillMemoryPool *spill_pool = shadowspill_spill_pool(runtime);
    void *device_base = execution_pool == NULL ? NULL : execution_pool->base;
    void *host_base = spill_pool == NULL ? NULL : spill_pool->base;
    shadowspill_memory_pool_destroy(execution_pool);
    shadowspill_memory_pool_destroy(spill_pool);
    if (device_base != NULL) {
        (void)runtime->backend.free_device(
            runtime->backend.context, device_base
        );
    }
    if (host_base != NULL) {
        (void)runtime->backend.free_host(
            runtime->backend.context, host_base
        );
    }
    free(runtime->pools);
    runtime->pools = NULL;
    runtime->pool_count = 0U;
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
    runtime->pools = calloc(
        SHADOWSPILL_INITIAL_POOL_COUNT, sizeof(*runtime->pools)
    );
    if (runtime->pools == NULL) {
        free(runtime);
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    runtime->pool_count = SHADOWSPILL_INITIAL_POOL_COUNT;
    runtime->execution_pool_id = SHADOWSPILL_EXECUTION_POOL_ID;
    runtime->spill_pool_id = SHADOWSPILL_SPILL_POOL_ID;
    runtime->backend = config->backend;
    runtime->worker_poll_nanoseconds = config->worker_poll_nanoseconds;
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
    atomic_init(&runtime->registered_objects, 0U);
    atomic_init(&runtime->transfers_to_device, 0U);
    atomic_init(&runtime->transfers_to_host, 0U);
    atomic_init(&runtime->bytes_to_device, 0U);
    atomic_init(&runtime->bytes_to_host, 0U);
    atomic_init(&runtime->wait_events_inserted, 0U);
    atomic_init(&runtime->actions.count, 0U);
    atomic_init(&runtime->execution_free_bytes_snapshot, 0U);
    atomic_init(&runtime->execution_largest_free_snapshot, 0U);
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
    const uint64_t execution_index_bucket_count = 4096U;
    runtime->execution_leases_by_id = calloc(
        (size_t)runtime->allocation_index_bucket_count,
        sizeof(*runtime->execution_leases_by_id)
    );
    runtime->execution_leases_by_pointer = calloc(
        (size_t)runtime->allocation_index_bucket_count,
        sizeof(*runtime->execution_leases_by_pointer)
    );
    runtime->reusable_execution_leases_by_size = calloc(
        (size_t)runtime->reusable_index_bucket_count,
        sizeof(*runtime->reusable_execution_leases_by_size)
    );
    if (runtime->execution_leases_by_id == NULL ||
        runtime->execution_leases_by_pointer == NULL ||
        runtime->reusable_execution_leases_by_size == NULL ||
        shadowspill_object_table_initialize(
            &runtime->objects, object_index_bucket_count
        ) != 0 || shadowspill_execution_table_initialize(
            &runtime->execution, execution_index_bucket_count
        ) != 0 || shadowspill_completion_tracker_initialize(
            &runtime->completions
        ) != 0) {
        free(runtime->execution_leases_by_id);
        free(runtime->execution_leases_by_pointer);
        free(runtime->reusable_execution_leases_by_size);
        shadowspill_execution_table_destroy(&runtime->execution);
        shadowspill_object_table_destroy(&runtime->objects);
        free(runtime);
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    runtime->completions_initialized = 1U;
    if (shadowspill_retirement_queue_initialize(
            &runtime->retirements
        ) != 0) {
        release_resources(runtime);
        free(runtime);
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    if (pthread_mutex_init(&runtime->actions.lock, NULL) != 0) {
        release_resources(runtime);
        free(runtime);
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    runtime->actions.lock_initialized = 1U;
    if (pthread_mutex_init(&runtime->failure_lock, NULL) != 0) {
        pthread_mutex_destroy(&runtime->actions.lock);
        runtime->actions.lock_initialized = 0U;
        release_resources(runtime);
        free(runtime);
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    if (pthread_mutex_init(&runtime->mutex, NULL) != 0) {
        pthread_mutex_destroy(&runtime->failure_lock);
        pthread_mutex_destroy(&runtime->actions.lock);
        runtime->actions.lock_initialized = 0U;
        release_resources(runtime);
        free(runtime);
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    if (pthread_cond_init(&runtime->condition, NULL) != 0) {
        pthread_mutex_destroy(&runtime->mutex);
        pthread_mutex_destroy(&runtime->failure_lock);
        pthread_mutex_destroy(&runtime->actions.lock);
        runtime->actions.lock_initialized = 0U;
        release_resources(runtime);
        free(runtime);
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    if (shadowspill_idle_wakeup_initialize(
            &runtime->idle_wakeup
        ) != 0) {
        pthread_cond_destroy(&runtime->condition);
        pthread_mutex_destroy(&runtime->mutex);
        pthread_mutex_destroy(&runtime->failure_lock);
        pthread_mutex_destroy(&runtime->actions.lock);
        runtime->actions.lock_initialized = 0U;
        release_resources(runtime);
        free(runtime);
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    ShadowSpillRuntimeStatus status = SHADOWSPILL_RUNTIME_BACKEND_FAILURE;
    if (shadowspill_transfer_lane_initialize(&runtime->fetch_lane) != 0) {
        status = SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
        goto fail;
    }
    if (shadowspill_transfer_lane_initialize(&runtime->evict_lane) != 0) {
        status = SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
        goto fail;
    }
    void *device_base = NULL;
    void *host_base = NULL;
    if (runtime->backend.allocate_device(
            runtime->backend.context,
            config->device_slab_bytes,
            &device_base
        ) != 0) {
        goto fail;
    }
    if (config->host_arena_bytes != 0U && runtime->backend.allocate_host(
            runtime->backend.context,
            config->host_arena_bytes,
            &host_base
        ) != 0) {
        (void)runtime->backend.free_device(
            runtime->backend.context, device_base
        );
        goto fail;
    }
    if (shadowspill_memory_pool_initialize(
            shadowspill_execution_pool(runtime),
            runtime->execution_pool_id,
            device_base,
            config->device_slab_bytes,
            config->minimum_alignment
        ) != 0) {
        (void)runtime->backend.free_device(
            runtime->backend.context, device_base
        );
        if (host_base != NULL) {
            (void)runtime->backend.free_host(
                runtime->backend.context, host_base
            );
        }
        status = SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
        goto fail;
    }
    if (shadowspill_memory_pool_initialize(
            shadowspill_spill_pool(runtime),
            runtime->spill_pool_id,
            host_base,
            config->host_arena_bytes,
            1U
        ) != 0) {
        shadowspill_memory_pool_destroy(shadowspill_execution_pool(runtime));
        (void)runtime->backend.free_device(
            runtime->backend.context, device_base
        );
        if (host_base != NULL) {
            (void)runtime->backend.free_host(
                runtime->backend.context, host_base
            );
        }
        status = SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
        goto fail;
    }
    shadowspill_publish_execution_geometry_locked(runtime);
    if (runtime->backend.create_stream(
            runtime->backend.context,
            SHADOWSPILL_TRANSFER_TO_DEVICE,
            &runtime->fetch_stream
        ) != 0) {
        goto fail;
    }
    runtime->fetch_stream_created = 1;
    if (runtime->backend.create_stream(
            runtime->backend.context,
            SHADOWSPILL_TRANSFER_TO_HOST,
            &runtime->evict_stream
        ) != 0) {
        goto fail;
    }
    runtime->evict_stream_created = 1;
    if (pthread_create(
            &runtime->worker_thread, NULL, shadowspill_worker_main, runtime
        ) != 0) {
        status = SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
        goto fail;
    }
    runtime->worker_started = 1;
    *output = runtime;
    return SHADOWSPILL_RUNTIME_OK;

fail:
    release_resources(runtime);
    shadowspill_idle_wakeup_destroy(&runtime->idle_wakeup);
    pthread_cond_destroy(&runtime->condition);
    pthread_mutex_destroy(&runtime->mutex);
    pthread_mutex_destroy(&runtime->failure_lock);
    shadowspill_transfer_lane_destroy(&runtime->evict_lane);
    shadowspill_transfer_lane_destroy(&runtime->fetch_lane);
    pthread_mutex_destroy(&runtime->actions.lock);
    runtime->actions.lock_initialized = 0U;
    free(runtime);
    return status;
}

ShadowSpillRuntimeStatus shadowspill_runtime_wait_idle_legacy(
    ShadowSpillRuntime *runtime
) {
    if (runtime == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    ShadowSpillIdleWakeup *wakeup = &runtime->idle_wakeup;
    pthread_mutex_lock(&wakeup->lock);
    while (atomic_load_explicit(&runtime->closed, memory_order_acquire) == 0U &&
           shadowspill_failure_status(runtime) == SHADOWSPILL_RUNTIME_OK &&
           (atomic_load_explicit(
                &runtime->actions.count, memory_order_acquire
            ) != 0U ||
            runtime->pending_retirements != 0U)) {
        pthread_cond_wait(&wakeup->condition, &wakeup->lock);
    }
    ShadowSpillRuntimeStatus status = shadowspill_failure_status(runtime);
    pthread_mutex_unlock(&wakeup->lock);
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
    uint64_t current_bytes = shadowspill_spill_pool(runtime)->ranges.capacity;
    if (status != SHADOWSPILL_RUNTIME_OK) {
        goto done;
    }
    if (atomic_load_explicit(&runtime->closing, memory_order_acquire) != 0U ||
        atomic_load_explicit(
            &runtime->actions.count, memory_order_acquire
        ) != 0U ||
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
        memcpy(replacement, shadowspill_spill_pool(runtime)->base, (size_t)current_bytes);
    }
    ShadowSpillRangeAllocator ranges = {0};
    if (shadowspill_range_clone_extended(
            &shadowspill_spill_pool(runtime)->ranges,
            host_arena_bytes,
            &ranges
        ) != 0) {
        (void)runtime->backend.free_host(runtime->backend.context, replacement);
        status = SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
        goto done;
    }
    if (shadowspill_spill_pool(runtime)->base != NULL && runtime->backend.free_host(
            runtime->backend.context, shadowspill_spill_pool(runtime)->base
        ) != 0) {
        shadowspill_range_destroy(&ranges);
        (void)runtime->backend.free_host(runtime->backend.context, replacement);
        status = SHADOWSPILL_RUNTIME_BACKEND_FAILURE;
        goto done;
    }
    shadowspill_memory_pool_rebase_locked(
        shadowspill_spill_pool(runtime), replacement
    );
    shadowspill_range_destroy(&shadowspill_spill_pool(runtime)->ranges);
    shadowspill_spill_pool(runtime)->ranges = ranges;

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
    pthread_mutex_unlock(&runtime->mutex);
    ShadowSpillIdleWakeup *wakeup = &runtime->idle_wakeup;
    pthread_mutex_lock(&wakeup->lock);
    while (shadowspill_failure_status(runtime) == SHADOWSPILL_RUNTIME_OK &&
           (atomic_load_explicit(
                &runtime->actions.count, memory_order_acquire
            ) != 0U ||
            runtime->pending_retirements != 0U)) {
        pthread_cond_wait(&wakeup->condition, &wakeup->lock);
    }
    pthread_mutex_unlock(&wakeup->lock);

    int synchronization_failed = 0;
    if (runtime->fetch_stream_created && runtime->backend.synchronize_stream(
            runtime->backend.context, runtime->fetch_stream
        ) != 0) {
        synchronization_failed = 1;
    }
    if (runtime->evict_stream_created && runtime->backend.synchronize_stream(
            runtime->backend.context, runtime->evict_stream
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
    shadowspill_idle_notify(runtime);
    if (runtime->worker_started) {
        (void)pthread_join(runtime->worker_thread, NULL);
        runtime->worker_started = 0;
    }
    pthread_mutex_lock(&runtime->mutex);
    ShadowSpillRuntimeStatus status = shadowspill_failure_status(runtime);
    atomic_store_explicit(&runtime->closed, 1U, memory_order_release);
    pthread_mutex_unlock(&runtime->mutex);
    shadowspill_idle_notify(runtime);
    release_resources(runtime);
    return status;
}

void shadowspill_runtime_destroy_legacy(ShadowSpillRuntime *runtime) {
    if (runtime == NULL) {
        return;
    }
    (void)shadowspill_runtime_close_legacy(runtime);
    shadowspill_idle_wakeup_destroy(&runtime->idle_wakeup);
    pthread_cond_destroy(&runtime->condition);
    pthread_mutex_destroy(&runtime->mutex);
    pthread_mutex_destroy(&runtime->failure_lock);
    shadowspill_transfer_lane_destroy(&runtime->evict_lane);
    shadowspill_transfer_lane_destroy(&runtime->fetch_lane);
    pthread_mutex_destroy(&runtime->actions.lock);
    runtime->actions.lock_initialized = 0U;
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
    pthread_mutex_lock(&shadowspill_execution_pool(runtime)->lock);
    pthread_mutex_lock(&shadowspill_spill_pool(runtime)->lock);
    uint64_t retirement_records_fenced = 0U;
    uint64_t retirement_records_evented = 0U;
    uint64_t retirement_records_preparing = 0U;
    uint64_t retirement_records_unfenced = 0U;
    for (const ShadowSpillMemoryLease *allocation =
             runtime->active_execution_leases;
         allocation != NULL; allocation = allocation->active_next) {
        if (!allocation->logical_freed || allocation->pointer == NULL) {
            continue;
        }
        if (allocation->retirement_fence != NULL) {
            ++retirement_records_fenced;
        } else if (allocation->retirement_events != NULL) {
            ++retirement_records_evented;
        } else if (allocation->retirement_preparing) {
            ++retirement_records_preparing;
        } else {
            ++retirement_records_unfenced;
        }
    }
    *statistics = (ShadowSpillRuntimeStatistics){
        .slab_bytes = shadowspill_execution_pool(runtime)->ranges.capacity,
        .requested_allocated_bytes = runtime->requested_allocated_bytes,
        .peak_requested_allocated_bytes =
            runtime->peak_requested_allocated_bytes,
        .allocated_bytes = shadowspill_execution_pool(runtime)->ranges.allocated,
        .free_bytes =
            shadowspill_memory_pool_free_bytes_locked(shadowspill_execution_pool(runtime)),
        .largest_free_range_bytes =
            shadowspill_memory_pool_largest_free_locked(
                shadowspill_execution_pool(runtime)
            ),
        .external_fragmentation_bytes =
            shadowspill_memory_pool_free_bytes_locked(shadowspill_execution_pool(runtime)) -
            shadowspill_memory_pool_largest_free_locked(
                shadowspill_execution_pool(runtime)
            ),
        .peak_allocated_bytes = shadowspill_execution_pool(runtime)->ranges.peak_allocated,
        .host_arena_bytes = shadowspill_spill_pool(runtime)->ranges.capacity,
        .host_allocated_bytes = shadowspill_spill_pool(runtime)->ranges.allocated,
        .host_peak_allocated_bytes = shadowspill_spill_pool(runtime)->ranges.peak_allocated,
        .live_allocations = runtime->live_allocations,
        .blocked_allocators = runtime->blocked_allocators,
        .pending_retirements = runtime->pending_retirements,
        .retirement_records_fenced = retirement_records_fenced,
        .retirement_records_evented = retirement_records_evented,
        .retirement_records_preparing = retirement_records_preparing,
        .retirement_records_unfenced = retirement_records_unfenced,
        .registered_objects = atomic_load_explicit(
            &runtime->registered_objects, memory_order_acquire
        ),
        .queued_actions = atomic_load_explicit(
            &runtime->actions.count, memory_order_acquire
        ),
        .transfers_to_device = atomic_load_explicit(
            &runtime->transfers_to_device, memory_order_acquire
        ),
        .transfers_to_host = atomic_load_explicit(
            &runtime->transfers_to_host, memory_order_acquire
        ),
        .bytes_to_device = atomic_load_explicit(
            &runtime->bytes_to_device, memory_order_acquire
        ),
        .bytes_to_host = atomic_load_explicit(
            &runtime->bytes_to_host, memory_order_acquire
        ),
        .wait_events_inserted = atomic_load_explicit(
            &runtime->wait_events_inserted, memory_order_acquire
        ),
        .allocation_events = runtime->allocation_event_count,
        .allocation_event_capacity = runtime->allocation_event_capacity,
        .allocation_event_overflow =
            (uint64_t)runtime->allocation_event_overflow,
    };
    pthread_mutex_unlock(&shadowspill_spill_pool(runtime)->lock);
    pthread_mutex_unlock(&shadowspill_execution_pool(runtime)->lock);
    pthread_mutex_unlock(&runtime->mutex);
    return SHADOWSPILL_RUNTIME_OK;
}
