#include "internal.h"

#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

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

int shadowspill_synchronization_backend_is_valid(
    const ShadowSpillSynchronizationBackend *backend
) {
    return backend != NULL &&
        backend->abi_version ==
            SHADOWSPILL_SYNCHRONIZATION_BACKEND_ABI_VERSION &&
        backend->create_event != NULL && backend->destroy_event != NULL &&
        backend->record_event != NULL && backend->query_event != NULL &&
        backend->wait_event != NULL;
}

static void destroy_allocations(ShadowSpillRuntime *runtime) {
    for (uint32_t pool_id = 0U; pool_id < runtime->pool_count; ++pool_id) {
        ShadowSpillMemoryPool *pool = &runtime->pools[pool_id];
        ShadowSpillMemoryLease *allocation = pool->owned_leases;
        while (allocation != NULL) {
            ShadowSpillMemoryLease *next = allocation->ownership_next;
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
            if (allocation->retirement_event != NULL) {
                (void)shadowspill_event_lease_release(
                    runtime, allocation->retirement_event
                );
                allocation->retirement_event = NULL;
            }
            free(allocation);
            allocation = next;
        }
        pool->owned_leases = NULL;
        pool->active_leases = NULL;
    }
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
    shadowspill_object_table_destroy(&runtime->objects);
}

static void destroy_actions(ShadowSpillRuntime *runtime) {
    ShadowSpillQueuedAction *action = runtime->actions.head;
    while (action != NULL) {
        ShadowSpillQueuedAction *next = action->next;
        pthread_mutex_lock(&action->object->lock);
        (void)shadowspill_object_remove_action_locked(
            action->object, action
        );
        pthread_mutex_unlock(&action->object->lock);
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
                if (action->kind == SHADOWSPILL_RUNTIME_PREFETCH) {
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
        (void)shadowspill_event_lease_release(
            runtime, action->trigger_event
        );
        action->trigger_event = NULL;
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
            action->object_previous = NULL;
            action->object_next = NULL;
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
    shadowspill_plan_destroy_all(runtime);
    destroy_objects(runtime);
    free(runtime->allocation_events);
    runtime->allocation_events = NULL;
    runtime->allocation_event_count = 0U;
    runtime->allocation_event_capacity = 0U;
    free(runtime->trace_events);
    runtime->trace_events = NULL;
    runtime->trace_event_count = 0U;
    runtime->trace_event_capacity = 0U;
    for (uint32_t route_id = runtime->route_count; route_id != 0U;) {
        ShadowSpillRouteState *route = &runtime->routes[--route_id];
        if (route->lane_created) {
            (void)route->route.destroy_lane(
                route->route.context, route->lane
            );
            route->lane_created = 0U;
        }
        shadowspill_transfer_lane_destroy(&route->transfers);
    }
    free(runtime->routes);
    runtime->routes = NULL;
    runtime->route_count = 0U;
    for (uint32_t pool_id = 0U; pool_id < runtime->pool_count; ++pool_id) {
        shadowspill_memory_pool_close(&runtime->pools[pool_id]);
    }
    free(runtime->pools);
    runtime->pools = NULL;
    runtime->pool_count = 0U;
    shadowspill_transfer_profiles_destroy(runtime);
}

static int runtime_config_is_valid(const ShadowSpillRuntimeConfig *config) {
    if (config == NULL ||
        config->abi_version != SHADOWSPILL_RUNTIME_ABI_VERSION ||
        config->pools == NULL || config->pool_count == 0U ||
        (config->routes == NULL && config->route_count != 0U) ||
        !shadowspill_synchronization_backend_is_valid(
            &config->synchronization
        ) || !shadowspill_profiler_is_valid(&config->profiler)) {
        return 0;
    }
    for (uint32_t pool_id = 0U; pool_id < config->pool_count; ++pool_id) {
        const ShadowSpillMemoryPoolDescription *pool = &config->pools[pool_id];
        if (pool->pool_id != pool_id || pool->minimum_alignment == 0U ||
            !shadowspill_memory_pool_backend_is_valid(&pool->backend)) {
            return 0;
        }
    }
    for (uint32_t route_id = 0U; route_id < config->route_count; ++route_id) {
        const ShadowSpillTransferRouteDescription *route =
            &config->routes[route_id];
        if (route->route_id != route_id || route->name == NULL ||
            !shadowspill_transfer_route_is_valid(&route->route) ||
            route->route.source_pool_id >= config->pool_count ||
            route->route.destination_pool_id >= config->pool_count) {
            return 0;
        }
        for (uint32_t previous = 0U; previous < route_id; ++previous) {
            const ShadowSpillTransferRoute *candidate =
                &config->routes[previous].route;
            if (candidate->source_pool_id == route->route.source_pool_id &&
                candidate->destination_pool_id ==
                    route->route.destination_pool_id) {
                return 0;
            }
        }
    }
    return 1;
}

ShadowSpillRuntimeStatus shadowspill_runtime_create(
    const ShadowSpillRuntimeConfig *config,
    ShadowSpillRuntime **output
) {
    if (output == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    *output = NULL;
    if (!runtime_config_is_valid(config)) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    ShadowSpillRuntime *runtime = calloc(1U, sizeof(*runtime));
    if (runtime == NULL) {
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    runtime->pools = calloc(config->pool_count, sizeof(*runtime->pools));
    runtime->routes = config->route_count == 0U
        ? NULL : calloc(config->route_count, sizeof(*runtime->routes));
    if (runtime->pools == NULL ||
        (config->route_count != 0U && runtime->routes == NULL)) {
        free(runtime->routes);
        free(runtime->pools);
        free(runtime);
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    runtime->pool_count = config->pool_count;
    runtime->route_count = config->route_count;
    runtime->synchronization = config->synchronization;
    runtime->profiler = config->profiler;
    for (uint32_t route_id = 0U; route_id < config->route_count; ++route_id) {
        runtime->routes[route_id].route = config->routes[route_id].route;
    }
    runtime->worker_poll_nanoseconds = config->worker_poll_nanoseconds;
    atomic_init(&runtime->next_allocation_id, 1U);
    atomic_init(&runtime->next_generation, 1U);
    atomic_init(&runtime->next_event_generation, 1U);
    runtime->failure.object_id = SHADOWSPILL_RUNTIME_NO_ID;
    runtime->failure.allocation_id = SHADOWSPILL_RUNTIME_NO_ID;
    runtime->failure.pool_id = UINT32_MAX;
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
    atomic_init(&runtime->worker_submission, NULL);
    atomic_init(&runtime->next_worker_submission_sequence, 0U);
    atomic_init(&runtime->allocation_event_count, 0U);
    atomic_init(&runtime->next_allocation_event_sequence, 0U);
    atomic_init(&runtime->allocation_telemetry_active, 0U);
    atomic_init(&runtime->allocation_event_overflow, 0U);
    atomic_init(&runtime->trace_event_count, 0U);
    atomic_init(&runtime->next_trace_event_sequence, 0U);
    atomic_init(&runtime->trace_prepared, 0U);
    atomic_init(&runtime->trace_active, 0U);
    atomic_init(&runtime->trace_event_overflow, 0U);
    const uint64_t object_index_bucket_count = 16384U;
    if (pthread_mutex_init(&runtime->plans_lock, NULL) != 0) {
        free(runtime->routes);
        free(runtime->pools);
        free(runtime);
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    runtime->plans_lock_initialized = 1U;
    if (shadowspill_object_table_initialize(
            &runtime->objects, object_index_bucket_count
        ) != 0 || shadowspill_completion_tracker_initialize(
            &runtime->completions
        ) != 0) {
        shadowspill_object_table_destroy(&runtime->objects);
        pthread_mutex_destroy(&runtime->plans_lock);
        runtime->plans_lock_initialized = 0U;
        free(runtime->routes);
        free(runtime->pools);
        free(runtime);
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    runtime->completions_initialized = 1U;
    if (shadowspill_transfer_profiles_initialize(runtime) != 0) {
        release_resources(runtime);
        pthread_mutex_destroy(&runtime->plans_lock);
        free(runtime);
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    if (shadowspill_retirement_queue_initialize(
            &runtime->retirements
        ) != 0) {
        release_resources(runtime);
        pthread_mutex_destroy(&runtime->plans_lock);
        free(runtime);
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    if (pthread_mutex_init(&runtime->actions.lock, NULL) != 0) {
        release_resources(runtime);
        pthread_mutex_destroy(&runtime->plans_lock);
        free(runtime);
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    runtime->actions.lock_initialized = 1U;
    if (pthread_mutex_init(&runtime->failure_lock, NULL) != 0) {
        pthread_mutex_destroy(&runtime->actions.lock);
        runtime->actions.lock_initialized = 0U;
        release_resources(runtime);
        pthread_mutex_destroy(&runtime->plans_lock);
        free(runtime);
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    if (pthread_mutex_init(&runtime->mutex, NULL) != 0) {
        pthread_mutex_destroy(&runtime->failure_lock);
        pthread_mutex_destroy(&runtime->actions.lock);
        runtime->actions.lock_initialized = 0U;
        release_resources(runtime);
        pthread_mutex_destroy(&runtime->plans_lock);
        free(runtime);
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    if (pthread_cond_init(&runtime->condition, NULL) != 0) {
        pthread_mutex_destroy(&runtime->mutex);
        pthread_mutex_destroy(&runtime->failure_lock);
        pthread_mutex_destroy(&runtime->actions.lock);
        runtime->actions.lock_initialized = 0U;
        release_resources(runtime);
        pthread_mutex_destroy(&runtime->plans_lock);
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
        pthread_mutex_destroy(&runtime->plans_lock);
        free(runtime);
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    ShadowSpillRuntimeStatus status = SHADOWSPILL_RUNTIME_BACKEND_FAILURE;
    for (uint32_t pool_id = 0U; pool_id < runtime->pool_count; ++pool_id) {
        const ShadowSpillMemoryPoolDescription *pool = &config->pools[pool_id];
        if (shadowspill_memory_pool_initialize(
                &runtime->pools[pool_id],
                pool_id,
                &pool->backend,
                pool->capacity_bytes,
                pool->minimum_alignment
            ) != 0) {
            status = SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
            goto fail;
        }
    }
    for (uint32_t pool_id = 0U; pool_id < runtime->pool_count; ++pool_id) {
        shadowspill_publish_pool_geometry_locked(&runtime->pools[pool_id]);
    }
    for (uint32_t route_id = 0U; route_id < runtime->route_count; ++route_id) {
        ShadowSpillRouteState *route = &runtime->routes[route_id];
        if (shadowspill_transfer_lane_initialize(&route->transfers) != 0) {
            status = SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
            goto fail;
        }
        if (route->route.create_lane(
                route->route.context, &route->lane
            ) != 0) {
            goto fail;
        }
        route->lane_created = 1U;
        shadowspill_profiler_name_stream(
            &runtime->profiler, route->lane, config->routes[route_id].name
        );
    }
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
    pthread_mutex_destroy(&runtime->actions.lock);
    runtime->actions.lock_initialized = 0U;
    pthread_mutex_destroy(&runtime->plans_lock);
    runtime->plans_lock_initialized = 0U;
    free(runtime);
    return status;
}

ShadowSpillRuntimeStatus shadowspill_runtime_wait_idle(
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

ShadowSpillRuntimeStatus shadowspill_memory_pool_grow(
    ShadowSpillRuntime *runtime,
    uint32_t pool_id,
    uint64_t capacity_bytes
) {
    ShadowSpillMemoryPool *pool = shadowspill_runtime_pool(runtime, pool_id);
    if (pool == NULL || capacity_bytes > SIZE_MAX) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    ShadowSpillRuntimeStatus status = shadowspill_runtime_wait_idle(runtime);
    if (status != SHADOWSPILL_RUNTIME_OK) {
        return status;
    }
    pthread_mutex_lock(&runtime->mutex);
    status = shadowspill_current_status_locked(runtime);
    uint64_t current_bytes = pool->ranges.capacity;
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
    if (capacity_bytes < current_bytes) {
        status = SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
        goto done;
    }
    if (capacity_bytes == current_bytes) {
        goto done;
    }

    void *replacement = NULL;
    if (pool->backend.allocate_arena(
            pool->backend.context, capacity_bytes, &replacement
        ) != 0) {
        status = SHADOWSPILL_RUNTIME_BACKEND_FAILURE;
        goto done;
    }
    if (current_bytes != 0U) {
        memcpy(replacement, pool->base, (size_t)current_bytes);
    }
    ShadowSpillRangeAllocator ranges = {0};
    if (shadowspill_range_clone_extended(
            &pool->ranges,
            capacity_bytes,
            &ranges
        ) != 0) {
        (void)pool->backend.close(
            pool->backend.context, replacement
        );
        status = SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
        goto done;
    }
    if (pool->base != NULL && pool->backend.close(
            pool->backend.context, pool->base
        ) != 0) {
        shadowspill_range_destroy(&ranges);
        (void)pool->backend.close(
            pool->backend.context, replacement
        );
        status = SHADOWSPILL_RUNTIME_BACKEND_FAILURE;
        goto done;
    }
    shadowspill_memory_pool_rebase_locked(
        pool, replacement
    );
    shadowspill_range_destroy(&pool->ranges);
    pool->ranges = ranges;
    shadowspill_publish_pool_geometry_locked(pool);

done:
    pthread_mutex_unlock(&runtime->mutex);
    return status;
}

ShadowSpillRuntimeStatus shadowspill_runtime_close(
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
    for (uint32_t route_id = 0U; route_id < runtime->route_count; ++route_id) {
        ShadowSpillRouteState *route = &runtime->routes[route_id];
        if (route->lane_created && route->route.synchronize_lane(
                route->route.context, route->lane
            ) != 0) {
            synchronization_failed = 1;
        }
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

void shadowspill_runtime_destroy(ShadowSpillRuntime *runtime) {
    if (runtime == NULL) {
        return;
    }
    /*
     * A failed allocator callback can leave this dispatch thread inside a
     * task scope.  Clear its thread-local reference before closing and
     * freeing the runtime so a later runtime cannot inherit stale scope state.
     */
    shadowspill_abort_current_task(runtime);
    (void)shadowspill_runtime_close(runtime);
    shadowspill_idle_wakeup_destroy(&runtime->idle_wakeup);
    pthread_cond_destroy(&runtime->condition);
    pthread_mutex_destroy(&runtime->mutex);
    pthread_mutex_destroy(&runtime->failure_lock);
    pthread_mutex_destroy(&runtime->actions.lock);
    runtime->actions.lock_initialized = 0U;
    if (runtime->plans_lock_initialized) {
        pthread_mutex_destroy(&runtime->plans_lock);
        runtime->plans_lock_initialized = 0U;
    }
    free(runtime);
}

ShadowSpillRuntimeStatus shadowspill_runtime_statistics(
    ShadowSpillRuntime *runtime,
    ShadowSpillRuntimeStatistics *statistics
) {
    if (runtime == NULL || statistics == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    ShadowSpillMemoryPool *execution_pool = shadowspill_runtime_pool(
        runtime, SHADOWSPILL_EXECUTION_POOL_ID
    );
    ShadowSpillMemoryPool *spill_pool = shadowspill_runtime_pool(
        runtime, SHADOWSPILL_SPILL_POOL_ID
    );
    if (execution_pool == NULL || spill_pool == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    pthread_mutex_lock(&runtime->mutex);
    pthread_mutex_lock(&execution_pool->lock);
    pthread_mutex_lock(&spill_pool->lock);
    uint64_t retirement_records_fenced = 0U;
    uint64_t retirement_records_evented = 0U;
    uint64_t retirement_records_preparing = 0U;
    uint64_t retirement_records_unfenced = 0U;
    for (const ShadowSpillMemoryLease *allocation =
             execution_pool->active_leases;
         allocation != NULL; allocation = allocation->active_next) {
        if (!allocation->logical_freed || allocation->pointer == NULL) {
            continue;
        }
        if (allocation->retirement_event != NULL) {
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
        .execution_pool_bytes = execution_pool->ranges.capacity,
        .requested_allocated_bytes = execution_pool->requested_allocated_bytes,
        .peak_requested_allocated_bytes =
            execution_pool->peak_requested_allocated_bytes,
        .allocated_bytes = execution_pool->ranges.allocated,
        .free_bytes =
            shadowspill_memory_pool_free_bytes_locked(execution_pool),
        .free_prefix_bytes = shadowspill_memory_pool_free_prefix_locked(
            execution_pool
        ),
        .largest_free_range_bytes =
            shadowspill_memory_pool_largest_free_locked(execution_pool),
        .external_fragmentation_bytes =
            shadowspill_memory_pool_free_bytes_locked(execution_pool) -
            shadowspill_memory_pool_largest_free_locked(execution_pool),
        .peak_allocated_bytes = execution_pool->ranges.peak_allocated,
        .spill_pool_bytes = spill_pool->ranges.capacity,
        .spill_allocated_bytes = spill_pool->ranges.allocated,
        .spill_peak_allocated_bytes = spill_pool->ranges.peak_allocated,
        .live_allocations = execution_pool->live_allocations,
        .blocked_allocators = execution_pool->blocked_allocators,
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
    pthread_mutex_unlock(&spill_pool->lock);
    pthread_mutex_unlock(&execution_pool->lock);
    pthread_mutex_unlock(&runtime->mutex);
    return SHADOWSPILL_RUNTIME_OK;
}
