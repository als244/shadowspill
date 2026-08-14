#include "internal.h"

#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int shadowspill_backend_is_valid(const ShadowSpillBackend *backend) {
    return backend != NULL &&
        backend->abi_version == SHADOWSPILL_BACKEND_ABI_VERSION &&
        backend->allocate_execution != NULL && backend->free_execution != NULL &&
        backend->allocate_spill != NULL && backend->free_spill != NULL &&
        backend->create_stream != NULL && backend->destroy_stream != NULL &&
        backend->create_event != NULL && backend->destroy_event != NULL &&
        backend->record_event != NULL && backend->query_event != NULL &&
        backend->wait_event != NULL && backend->copy_async != NULL &&
        backend->synchronize_stream != NULL;
}

int shadowspill_memory_pool_backend_is_valid(
    const ShadowSpillMemoryPoolBackend *backend
) {
    return backend != NULL &&
        backend->abi_version == SHADOWSPILL_MEMORY_POOL_BACKEND_ABI_VERSION &&
        backend->allocate_arena != NULL && backend->close != NULL;
}

int shadowspill_transfer_route_is_valid(
    const ShadowSpillTransferRoute *route
) {
    return route != NULL &&
        route->abi_version == SHADOWSPILL_TRANSFER_ROUTE_ABI_VERSION &&
        route->source_pool_id != route->destination_pool_id &&
        route->create_lane != NULL && route->destroy_lane != NULL &&
        route->copy_async != NULL && route->synchronize_lane != NULL;
}

static int legacy_allocate_execution(
    void *context, uint64_t bytes, void **base
) {
    ShadowSpillBackend *backend = context;
    return backend->allocate_execution(backend->context, bytes, base);
}

static int legacy_close_execution_pool(void *context, void *base) {
    ShadowSpillBackend *backend = context;
    return backend->free_execution(backend->context, base);
}

static int legacy_allocate_spill(void *context, uint64_t bytes, void **base) {
    ShadowSpillBackend *backend = context;
    return backend->allocate_spill(backend->context, bytes, base);
}

static int legacy_close_spill_pool(void *context, void *base) {
    ShadowSpillBackend *backend = context;
    return backend->free_spill(backend->context, base);
}

static int legacy_create_fetch_lane(
    void *context, ShadowSpillBackendStream *lane
) {
    ShadowSpillBackend *backend = context;
    return backend->create_stream(
        backend->context, SHADOWSPILL_TRANSFER_FETCH, lane
    );
}

static int legacy_create_evict_lane(
    void *context, ShadowSpillBackendStream *lane
) {
    ShadowSpillBackend *backend = context;
    return backend->create_stream(
        backend->context, SHADOWSPILL_TRANSFER_EVICT, lane
    );
}

static int legacy_destroy_lane(
    void *context, ShadowSpillBackendStream lane
) {
    ShadowSpillBackend *backend = context;
    return backend->destroy_stream(backend->context, lane);
}

static int legacy_fetch_async(
    void *context,
    void *destination,
    const void *source,
    uint64_t bytes,
    ShadowSpillBackendStream lane
) {
    ShadowSpillBackend *backend = context;
    return backend->copy_async(
        backend->context,
        destination,
        source,
        bytes,
        SHADOWSPILL_TRANSFER_FETCH,
        lane
    );
}

static int legacy_evict_async(
    void *context,
    void *destination,
    const void *source,
    uint64_t bytes,
    ShadowSpillBackendStream lane
) {
    ShadowSpillBackend *backend = context;
    return backend->copy_async(
        backend->context,
        destination,
        source,
        bytes,
        SHADOWSPILL_TRANSFER_EVICT,
        lane
    );
}

static int legacy_synchronize_lane(
    void *context, ShadowSpillBackendStream lane
) {
    ShadowSpillBackend *backend = context;
    return backend->synchronize_stream(backend->context, lane);
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
        if (action->has_completion_event) {
            (void)shadowspill_event_lease_release(
                runtime, action->completion_event
            );
        }
        if (action->dependency_event != NULL) {
            (void)shadowspill_event_lease_release(
                runtime, action->dependency_event
            );
            action->dependency_event = NULL;
        }
        if (action->destination_lease != NULL) {
            ShadowSpillMemoryLease *lease = action->destination_lease;
            ShadowSpillMemoryPool *pool = lease->pool;
            if (pool == NULL) {
                action->destination_lease = NULL;
            } else {
                pthread_mutex_lock(&pool->lock);
                if (pool->pool_id == runtime->execution_pool_id) {
                    shadowspill_cancel_execution_reservation_locked(
                        runtime, lease
                    );
                } else {
                    (void)shadowspill_memory_pool_cancel_reservation_locked(
                        lease
                    );
                    free(lease);
                }
                pthread_mutex_unlock(&pool->lock);
                action->destination_lease = NULL;
            }
        }
        shadowspill_release_task_fence_locked(runtime, action->fence);
        if (!action->admitted) {
            shadowspill_object_release(action->object);
            if (action->owns_trace_label) {
                free((void *)action->trace_label);
            }
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
        (void)runtime->fetch_route.destroy_lane(
            runtime->fetch_route.context, runtime->fetch_stream
        );
        runtime->fetch_stream_created = 0;
    }
    if (runtime->evict_stream_created) {
        (void)runtime->evict_route.destroy_lane(
            runtime->evict_route.context, runtime->evict_stream
        );
        runtime->evict_stream_created = 0;
    }
    for (uint32_t pool_id = 0U; pool_id < runtime->pool_count; ++pool_id) {
        shadowspill_memory_pool_close(&runtime->pools[pool_id]);
    }
    free(runtime->pools);
    runtime->pools = NULL;
    runtime->pool_count = 0U;
    shadowspill_transfer_profiles_destroy(runtime);
}

ShadowSpillRuntimeStatus shadowspill_runtime_create_legacy(
    const ShadowSpillRuntimeConfig *config,
    ShadowSpillRuntime **output
) {
    if (config == NULL || output == NULL ||
        config->abi_version != SHADOWSPILL_RUNTIME_ABI_VERSION ||
        config->execution_pool_bytes == 0U || config->minimum_alignment == 0U ||
        !shadowspill_backend_is_valid(&config->backend) ||
        !shadowspill_profiler_is_valid(&config->profiler)) {
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
    runtime->profiler = config->profiler;
    const ShadowSpillMemoryPoolBackend execution_backend = {
        .abi_version = SHADOWSPILL_MEMORY_POOL_BACKEND_ABI_VERSION,
        .context = &runtime->backend,
        .allocate_arena = legacy_allocate_execution,
        .close = legacy_close_execution_pool,
    };
    const ShadowSpillMemoryPoolBackend spill_backend = {
        .abi_version = SHADOWSPILL_MEMORY_POOL_BACKEND_ABI_VERSION,
        .context = &runtime->backend,
        .allocate_arena = legacy_allocate_spill,
        .close = legacy_close_spill_pool,
    };
    runtime->fetch_route = (ShadowSpillTransferRoute){
        .abi_version = SHADOWSPILL_TRANSFER_ROUTE_ABI_VERSION,
        .source_pool_id = runtime->spill_pool_id,
        .destination_pool_id = runtime->execution_pool_id,
        .context = &runtime->backend,
        .create_lane = legacy_create_fetch_lane,
        .destroy_lane = legacy_destroy_lane,
        .copy_async = legacy_fetch_async,
        .synchronize_lane = legacy_synchronize_lane,
    };
    runtime->evict_route = (ShadowSpillTransferRoute){
        .abi_version = SHADOWSPILL_TRANSFER_ROUTE_ABI_VERSION,
        .source_pool_id = runtime->execution_pool_id,
        .destination_pool_id = runtime->spill_pool_id,
        .context = &runtime->backend,
        .create_lane = legacy_create_evict_lane,
        .destroy_lane = legacy_destroy_lane,
        .copy_async = legacy_evict_async,
        .synchronize_lane = legacy_synchronize_lane,
    };
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
    atomic_init(&runtime->fetch_transfers, 0U);
    atomic_init(&runtime->evict_transfers, 0U);
    atomic_init(&runtime->bytes_fetched, 0U);
    atomic_init(&runtime->bytes_evicted, 0U);
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
    if (shadowspill_transfer_profiles_initialize(runtime) != 0) {
        release_resources(runtime);
        free(runtime);
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
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
    if (shadowspill_memory_pool_initialize(
            shadowspill_execution_pool(runtime),
            runtime->execution_pool_id,
            &execution_backend,
            config->execution_pool_bytes,
            config->minimum_alignment
        ) != 0) {
        status = SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
        goto fail;
    }
    if (shadowspill_memory_pool_initialize(
            shadowspill_spill_pool(runtime),
            runtime->spill_pool_id,
            &spill_backend,
            config->spill_pool_bytes,
            1U
        ) != 0) {
        shadowspill_memory_pool_close(shadowspill_execution_pool(runtime));
        status = SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
        goto fail;
    }
    shadowspill_publish_execution_geometry_locked(runtime);
    if (runtime->fetch_route.create_lane(
            runtime->fetch_route.context,
            &runtime->fetch_stream
        ) != 0) {
        goto fail;
    }
    runtime->fetch_stream_created = 1;
    shadowspill_profiler_name_stream(
        &runtime->profiler, runtime->fetch_stream, "shadowspill_fetch"
    );
    if (runtime->evict_route.create_lane(
            runtime->evict_route.context,
            &runtime->evict_stream
        ) != 0) {
        goto fail;
    }
    runtime->evict_stream_created = 1;
    shadowspill_profiler_name_stream(
        &runtime->profiler, runtime->evict_stream, "shadowspill_evict"
    );
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

ShadowSpillRuntimeStatus shadowspill_runtime_resize_spill_pool_legacy(
    ShadowSpillRuntime *runtime,
    uint64_t spill_pool_bytes
) {
    if (runtime == NULL || spill_pool_bytes > SIZE_MAX) {
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
    if (spill_pool_bytes < current_bytes) {
        status = SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
        goto done;
    }
    if (spill_pool_bytes == current_bytes) {
        goto done;
    }

    ShadowSpillMemoryPool *spill_pool = shadowspill_spill_pool(runtime);
    void *replacement = NULL;
    if (spill_pool->backend.allocate_arena(
            spill_pool->backend.context, spill_pool_bytes, &replacement
        ) != 0) {
        status = SHADOWSPILL_RUNTIME_BACKEND_FAILURE;
        goto done;
    }
    if (current_bytes != 0U) {
        memcpy(replacement, spill_pool->base, (size_t)current_bytes);
    }
    ShadowSpillRangeAllocator ranges = {0};
    if (shadowspill_range_clone_extended(
            &spill_pool->ranges,
            spill_pool_bytes,
            &ranges
        ) != 0) {
        (void)spill_pool->backend.close(
            spill_pool->backend.context, replacement
        );
        status = SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
        goto done;
    }
    if (spill_pool->base != NULL && spill_pool->backend.close(
            spill_pool->backend.context, spill_pool->base
        ) != 0) {
        shadowspill_range_destroy(&ranges);
        (void)spill_pool->backend.close(
            spill_pool->backend.context, replacement
        );
        status = SHADOWSPILL_RUNTIME_BACKEND_FAILURE;
        goto done;
    }
    shadowspill_memory_pool_rebase_locked(
        spill_pool, replacement
    );
    shadowspill_range_destroy(&spill_pool->ranges);
    spill_pool->ranges = ranges;

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
    if (runtime->fetch_stream_created && runtime->fetch_route.synchronize_lane(
            runtime->fetch_route.context, runtime->fetch_stream
        ) != 0) {
        synchronization_failed = 1;
    }
    if (runtime->evict_stream_created && runtime->evict_route.synchronize_lane(
            runtime->evict_route.context, runtime->evict_stream
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
        .execution_pool_bytes = shadowspill_execution_pool(runtime)->ranges.capacity,
        .requested_allocated_bytes = runtime->requested_allocated_bytes,
        .peak_requested_allocated_bytes =
            runtime->peak_requested_allocated_bytes,
        .allocated_bytes = shadowspill_execution_pool(runtime)->ranges.allocated,
        .free_bytes =
            shadowspill_memory_pool_free_bytes_locked(shadowspill_execution_pool(runtime)),
        .free_prefix_bytes = shadowspill_memory_pool_free_prefix_locked(
            shadowspill_execution_pool(runtime)
        ),
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
        .spill_pool_bytes = shadowspill_spill_pool(runtime)->ranges.capacity,
        .spill_allocated_bytes = shadowspill_spill_pool(runtime)->ranges.allocated,
        .spill_peak_allocated_bytes = shadowspill_spill_pool(runtime)->ranges.peak_allocated,
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
        .fetch_transfers = atomic_load_explicit(
            &runtime->fetch_transfers, memory_order_acquire
        ),
        .evict_transfers = atomic_load_explicit(
            &runtime->evict_transfers, memory_order_acquire
        ),
        .bytes_fetched = atomic_load_explicit(
            &runtime->bytes_fetched, memory_order_acquire
        ),
        .bytes_evicted = atomic_load_explicit(
            &runtime->bytes_evicted, memory_order_acquire
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
