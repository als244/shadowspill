/* Opening a runtime: what a configuration must say, and what is reserved. */
#include "internal.h"

#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int shadowspill_backend_is_valid(const ShadowSpillBackend *backend) {
    return backend != NULL &&
        backend->abi_version == SHADOWSPILL_BACKEND_ABI_VERSION &&
        backend->state != NULL && backend->allocate_device != NULL &&
        backend->free_device != NULL && backend->register_host_memory != NULL &&
        backend->unregister_host_memory != NULL &&
        backend->allocate_signals != NULL && backend->free_signals != NULL &&
        backend->wait_value != NULL &&
        backend->write_value != NULL &&
        backend->create_stream != NULL && backend->destroy_stream != NULL &&
        backend->synchronize_stream != NULL && backend->resolve_stream != NULL &&
        backend->copy_host_to_device != NULL &&
        backend->copy_device_to_host != NULL &&
        backend->copy_device_to_device != NULL &&
        backend->create_event != NULL && backend->destroy_event != NULL &&
        backend->record_event != NULL && backend->query_event != NULL &&
        backend->wait_event != NULL && backend->synchronize_event != NULL &&
        backend->elapsed_nanoseconds != NULL &&
        backend->capabilities != NULL && backend->physical_memory != NULL &&
        backend->statistics != NULL;
}

static int runtime_config_is_valid(const ShadowSpillRuntimeConfig *config) {
    if (config == NULL ||
        config->abi_version != SHADOWSPILL_ABI_VERSION ||
        config->pools == NULL || config->pool_count == 0U ||
        (config->routes == NULL && config->route_count != 0U) ||
        !shadowspill_backend_is_valid(config->backend)) {
        return 0;
    }
    for (uint32_t pool_id = 0U; pool_id < config->pool_count; ++pool_id) {
        const ShadowSpillMemoryPoolDescription *pool = &config->pools[pool_id];
        if (pool->pool_id != pool_id || pool->minimum_alignment == 0U) {
            return 0;
        }
        /* Nothing here judges the kind. Whether one is servable is whether an
           entry claims it, which only the lookup knows, so create answers that
           where the table exists rather than keeping a second opinion that
           would have to be widened for every kind a library adds. */
    }
    for (uint32_t route_id = 0U; route_id < config->route_count; ++route_id) {
        const ShadowSpillTransferRouteDescription *route =
            &config->routes[route_id];
        if (route->route_id != route_id || route->name == NULL ||
            route->source_pool_id >= config->pool_count ||
            route->destination_pool_id >= config->pool_count ||
            route->source_pool_id == route->destination_pool_id ||
            config->pools[route->source_pool_id].kind ==
                config->pools[route->destination_pool_id].kind) {
            return 0;
        }
        for (uint32_t previous = 0U; previous < route_id; ++previous) {
            const ShadowSpillTransferRouteDescription *candidate =
                &config->routes[previous];
            if (candidate->source_pool_id == route->source_pool_id &&
                candidate->destination_pool_id == route->destination_pool_id) {
                return 0;
            }
        }
    }
    return 1;
}

/*
 * Everything a runtime holds is released by asking what it reached: the
 * resources through `release_resources` and the primitives through
 * `release_primitives`, each skipping what was never created. One way to give
 * up, in place of an unwind at each of the nine places creation can fail.
 */
static ShadowSpillStatus abandon_runtime(
    ShadowSpillRuntime *runtime,
    ShadowSpillStatus status
) {
    shadowspill_runtime_release_resources(runtime);
    shadowspill_runtime_release_primitives(runtime);
    free(runtime);
    return status;
}

/* What the configuration says the runtime is, before anything is created. */
static void describe_runtime(
    ShadowSpillRuntime *runtime,
    const ShadowSpillRuntimeConfig *config
) {
    for (uint32_t route_id = 0U; route_id < config->route_count; ++route_id) {
        const ShadowSpillTransferRouteDescription *route = &config->routes[route_id];
        runtime->routes[route_id].source_pool_id = route->source_pool_id;
        runtime->routes[route_id].destination_pool_id = route->destination_pool_id;
    }
    runtime->worker_poll_nanoseconds = config->worker_poll_nanoseconds;
    runtime->background_transfer_window_bytes =
        config->background_transfer_window_bytes;
    atomic_init(&runtime->next_plan_id, 1U);
    atomic_init(&runtime->next_allocation_id, 1U);
    atomic_init(&runtime->next_generation, 1U);
    atomic_init(&runtime->next_event_generation, 1U);
    runtime->failure.object_id = SHADOWSPILL_RUNTIME_NO_ID;
    runtime->failure.allocation_id = SHADOWSPILL_RUNTIME_NO_ID;
    runtime->failure.pool_id = UINT32_MAX;
    atomic_init(&runtime->closing, 0U);
    atomic_init(&runtime->closed, 0U);
    atomic_init(&runtime->worker_stop, 0U);
    atomic_init(&runtime->failure_status, SHADOWSPILL_STATUS_OK);
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
}

/* The pools the runtime allocates from, and the lanes it transfers over. */
static ShadowSpillStatus open_pools_and_routes(
    ShadowSpillRuntime *runtime,
    const ShadowSpillRuntimeConfig *config
) {
    ShadowSpillStatus status = SHADOWSPILL_STATUS_BACKEND_FAILURE;
    for (uint32_t pool_id = 0U; pool_id < runtime->pool_count; ++pool_id) {
        const ShadowSpillMemoryPoolDescription *pool = &config->pools[pool_id];
        const ShadowSpillPoolMemoryDescription *memory =
            shadowspill_pool_memory_for_kind(&runtime->pool_memory, pool->kind);
        if (memory == NULL) {
            return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
        }
        /* The description's configuration is the pool's, not the kind's: two
           Remote pools on different machines share an entry and differ here. */
        ShadowSpillPoolMemoryDescription resolved = *memory;
        if (pool->configuration != NULL) {
            resolved.configuration = pool->configuration;
        }
        if (shadowspill_memory_pool_initialize(
                &runtime->pools[pool_id],
                pool_id,
                &resolved,
                pool->kind,
                pool->capacity_bytes,
                pool->minimum_alignment
            ) != 0) {
            status = SHADOWSPILL_STATUS_INTERNAL_FAILURE;
            return status;
        }
    }
    for (uint32_t pool_id = 0U; pool_id < runtime->pool_count; ++pool_id) {
        shadowspill_publish_pool_geometry_locked(&runtime->pools[pool_id]);
    }
    for (uint32_t route_id = 0U; route_id < runtime->route_count; ++route_id) {
        ShadowSpillRouteState *route = &runtime->routes[route_id];
        if (shadowspill_transfer_queue_initialize(&route->queue) != 0) {
            status = SHADOWSPILL_STATUS_INTERNAL_FAILURE;
            return status;
        }
        route->queue.background_window_bytes =
            runtime->background_transfer_window_bytes;
        if (runtime->backend.create_stream(
                runtime->backend.state, &route->stream
            ) != 0) {
            return SHADOWSPILL_STATUS_BACKEND_FAILURE;
        }
        route->stream_created = 1U;
        shadowspill_profiler_name_stream(
            &runtime->backend, route->stream, config->routes[route_id].name
        );
        /*
         * The lane comes from the two pools' kinds, the way the copy direction
         * used to. The built-in lane is granted this route's stream, which is
         * where its copies and its completion event both belong; a lane that
         * works elsewhere takes a stream of its own at create and leaves this
         * one to the runtime.
         */
        const ShadowSpillLaneDescription *description = shadowspill_lane_for_kinds(
            &runtime->lanes,
            runtime->pools[route->source_pool_id].kind,
            runtime->pools[route->destination_pool_id].kind
        );
        if (description == NULL) {
            return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
        }
        if (description->create(
                runtime,
                &runtime->backend,
                route->stream,
                description->configuration,
                &route->lane
            ) != 0) {
            return SHADOWSPILL_STATUS_BACKEND_FAILURE;
        }
        route->operations = description->operations;
    }
    return SHADOWSPILL_STATUS_OK;
}

ShadowSpillStatus shadowspill_runtime_create(
    const ShadowSpillRuntimeConfig *config,
    ShadowSpillRuntime **output
) {
    if (output == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    *output = NULL;
    if (!runtime_config_is_valid(config)) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    ShadowSpillRuntime *runtime = calloc(1U, sizeof(*runtime));
    if (runtime == NULL) {
        return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    }
    runtime->pools = calloc(config->pool_count, sizeof(*runtime->pools));
    runtime->routes = config->route_count == 0U
        ? NULL : calloc(config->route_count, sizeof(*runtime->routes));
    if (runtime->pools == NULL ||
        (config->route_count != 0U && runtime->routes == NULL)) {
        return abandon_runtime(runtime, SHADOWSPILL_STATUS_INTERNAL_FAILURE);
    }
    runtime->pool_count = config->pool_count;
    runtime->route_count = config->route_count;
    runtime->backend = *config->backend;
    describe_runtime(runtime, config);
    /* Before any route resolves one. The built-ins are seeded here, so a
       registered lane is found by the same lookup and a pair claimed twice
       fails now rather than at the first transfer. */
    if (shadowspill_lane_table_initialize(
            &runtime->lanes, runtime, config->lanes, config->lane_count
        ) != 0 || shadowspill_pool_memory_table_initialize(
            &runtime->pool_memory,
            &runtime->backend,
            config->pool_memory,
            config->pool_memory_count
        ) != 0) {
        return abandon_runtime(runtime, SHADOWSPILL_STATUS_INVALID_ARGUMENT);
    }
    const uint64_t object_index_bucket_count = 16384U;
    if (pthread_mutex_init(&runtime->plans_lock, NULL) != 0) {
        return abandon_runtime(runtime, SHADOWSPILL_STATUS_INTERNAL_FAILURE);
    }
    runtime->plans_lock_initialized = 1U;
    if (shadowspill_plan_registry_initialize(runtime) != 0) {
        return abandon_runtime(runtime, SHADOWSPILL_STATUS_INTERNAL_FAILURE);
    }
    if (shadowspill_event_pool_initialize(&runtime->events, 0U) != 0 ||
        shadowspill_event_pool_initialize(&runtime->timing_events, 1U) != 0 ||
        shadowspill_object_table_initialize(
            &runtime->objects, object_index_bucket_count
        ) != 0 || shadowspill_completion_tracker_initialize(
            &runtime->completions
        ) != 0) {
        return abandon_runtime(runtime, SHADOWSPILL_STATUS_INTERNAL_FAILURE);
    }
    runtime->completions_initialized = 1U;
    if (shadowspill_transfer_profiles_initialize(runtime) != 0 ||
        shadowspill_retirement_queue_initialize(&runtime->retirements) != 0) {
        return abandon_runtime(runtime, SHADOWSPILL_STATUS_INTERNAL_FAILURE);
    }
    if (pthread_mutex_init(&runtime->actions.lock, NULL) != 0) {
        return abandon_runtime(runtime, SHADOWSPILL_STATUS_INTERNAL_FAILURE);
    }
    runtime->actions.lock_initialized = 1U;
    if (pthread_mutex_init(&runtime->failure_lock, NULL) != 0) {
        return abandon_runtime(runtime, SHADOWSPILL_STATUS_INTERNAL_FAILURE);
    }
    runtime->failure_lock_initialized = 1U;
    if (pthread_mutex_init(&runtime->mutex, NULL) != 0) {
        return abandon_runtime(runtime, SHADOWSPILL_STATUS_INTERNAL_FAILURE);
    }
    runtime->mutex_initialized = 1U;
    if (shadowspill_idle_wakeup_initialize(&runtime->idle_wakeup) != 0) {
        return abandon_runtime(runtime, SHADOWSPILL_STATUS_INTERNAL_FAILURE);
    }
    runtime->idle_wakeup_initialized = 1U;
    const ShadowSpillStatus opened = open_pools_and_routes(runtime, config);
    if (opened != SHADOWSPILL_STATUS_OK) {
        return abandon_runtime(runtime, opened);
    }
    if (pthread_create(
            &runtime->worker_thread, NULL, shadowspill_worker_main, runtime
        ) != 0) {
        return abandon_runtime(runtime, SHADOWSPILL_STATUS_INTERNAL_FAILURE);
    }
    runtime->worker_started = 1;
    *output = runtime;
    return SHADOWSPILL_STATUS_OK;
}

ShadowSpillStatus shadowspill_runtime_reserve_event_leases(
    ShadowSpillRuntime *runtime,
    uint64_t minimum_free_leases
) {
    if (runtime == NULL || minimum_free_leases == 0U) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    if (atomic_load_explicit(&runtime->closing, memory_order_acquire) != 0U) {
        return SHADOWSPILL_STATUS_CLOSED;
    }
    ShadowSpillStatus status = shadowspill_runtime_wait_idle(runtime);
    if (status != SHADOWSPILL_STATUS_OK) {
        return status;
    }
    return shadowspill_event_pool_reserve(
        runtime, &runtime->events, minimum_free_leases
    );
}

ShadowSpillStatus shadowspill_runtime_reserve_retirement_records(
    ShadowSpillRuntime *runtime,
    uint64_t minimum_free_records
) {
    if (runtime == NULL || minimum_free_records == 0U) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    if (atomic_load_explicit(&runtime->closing, memory_order_acquire) != 0U) {
        return SHADOWSPILL_STATUS_CLOSED;
    }
    ShadowSpillStatus status = shadowspill_runtime_wait_idle(runtime);
    if (status != SHADOWSPILL_STATUS_OK) {
        return status;
    }
    return shadowspill_retirement_queue_reserve(
        &runtime->retirements, minimum_free_records
    );
}

ShadowSpillStatus shadowspill_runtime_reserve_memory_lease_records(
    ShadowSpillRuntime *runtime,
    uint32_t pool_id,
    uint64_t minimum_free_records
) {
    ShadowSpillMemoryPool *pool = shadowspill_runtime_pool(runtime, pool_id);
    if (pool == NULL || minimum_free_records == 0U) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    if (atomic_load_explicit(&runtime->closing, memory_order_acquire) != 0U) {
        return SHADOWSPILL_STATUS_CLOSED;
    }
    ShadowSpillStatus status = shadowspill_runtime_wait_idle(runtime);
    if (status != SHADOWSPILL_STATUS_OK) {
        return status;
    }
    return shadowspill_memory_pool_reserve_lease_records(
        pool, minimum_free_records
    );
}

ShadowSpillStatus shadowspill_runtime_next_plan_id(
    ShadowSpillRuntime *runtime,
    uint64_t *plan_id
) {
    if (runtime == NULL || plan_id == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    *plan_id = 0U;
    ShadowSpillStatus status = shadowspill_current_status_locked(runtime);
    if (status != SHADOWSPILL_STATUS_OK) {
        return status;
    }
    /* Monotonic and never reissued, so an id identifies one plan for the life
     * of the runtime even after that plan is gone. */
    *plan_id = atomic_fetch_add_explicit(
        &runtime->next_plan_id, 1U, memory_order_relaxed
    );
    return SHADOWSPILL_STATUS_OK;
}

ShadowSpillStatus shadowspill_runtime_wait_idle(
    ShadowSpillRuntime *runtime
) {
    if (runtime == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    ShadowSpillIdleWakeup *wakeup = &runtime->idle_wakeup;
    pthread_mutex_lock(&wakeup->lock);
    while (atomic_load_explicit(&runtime->closed, memory_order_acquire) == 0U &&
           shadowspill_failure_status(runtime) == SHADOWSPILL_STATUS_OK &&
           (atomic_load_explicit(
                &runtime->actions.count, memory_order_acquire
            ) != 0U ||
            runtime->pending_retirements != 0U)) {
        pthread_cond_wait(&wakeup->condition, &wakeup->lock);
    }
    ShadowSpillStatus status = shadowspill_failure_status(runtime);
    pthread_mutex_unlock(&wakeup->lock);
    return status;
}
