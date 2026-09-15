/* Closing one: draining the worker, then releasing what is left. */
#include "internal.h"

#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static void destroy_allocations(ShadowSpillRuntime *runtime) {
    for (uint32_t pool_id = 0U; pool_id < runtime->pool_count; ++pool_id) {
        ShadowSpillMemoryPool *pool = &runtime->pools[pool_id];
        ShadowSpillMemoryLease *allocation = pool->owned_leases;
        while (allocation != NULL) {
            ShadowSpillMemoryLease *next = allocation->ownership_next;
            ShadowSpillLeaseUseRecord *use = allocation->uses;
            while (use != NULL) {
                if (use->event != NULL) {
                    (void)shadowspill_event_lease_release(
                        runtime, use->event
                    );
                    use->event = NULL;
                }
                use = use->next;
            }
            allocation->uses = NULL;
            allocation->retirement_requirements = NULL;
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
        ShadowSpillPlan *plan_owner = action->plan_owner;
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
                if (action->kind == SHADOWSPILL_RUNTIME_FETCH) {
                    shadowspill_cancel_reservation_locked(
                        runtime, lease
                    );
                } else {
                    (void)shadowspill_memory_pool_cancel_reservation_locked(
                        lease
                    );
                    shadowspill_memory_pool_try_recycle_lease_record_locked(
                        lease
                    );
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
        if (plan_owner != NULL) {
            (void)atomic_fetch_sub_explicit(
                &plan_owner->pending_actions,
                1U,
                memory_order_release
            );
        }
        action = next;
    }
    runtime->actions.head = NULL;
    runtime->actions.tail = NULL;
    atomic_store_explicit(&runtime->actions.count, 0U, memory_order_release);
}

/*
 * The primitives a runtime holds that have no safe destroy before they are
 * created. Each is guarded by the flag creation sets, so this is the same
 * teardown whether the runtime was fully built or failed partway through it.
 */
void shadowspill_runtime_release_primitives(ShadowSpillRuntime *runtime) {
    if (runtime->idle_wakeup_initialized) {
        shadowspill_idle_wakeup_destroy(&runtime->idle_wakeup);
        runtime->idle_wakeup_initialized = 0U;
    }
    if (runtime->mutex_initialized) {
        pthread_mutex_destroy(&runtime->mutex);
        runtime->mutex_initialized = 0U;
    }
    if (runtime->failure_lock_initialized) {
        pthread_mutex_destroy(&runtime->failure_lock);
        runtime->failure_lock_initialized = 0U;
    }
    if (runtime->actions.lock_initialized) {
        pthread_mutex_destroy(&runtime->actions.lock);
        runtime->actions.lock_initialized = 0U;
    }
    if (runtime->plans_lock_initialized) {
        pthread_mutex_destroy(&runtime->plans_lock);
        runtime->plans_lock_initialized = 0U;
    }
}

void shadowspill_runtime_release_resources(ShadowSpillRuntime *runtime) {
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
    shadowspill_plan_registry_destroy(runtime);
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
            (void)runtime->backend.destroy_stream(
                runtime->backend.state, route->lane
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
    shadowspill_event_pool_destroy(runtime, &runtime->events);
    shadowspill_event_pool_destroy(runtime, &runtime->timing_events);
}

static ShadowSpillStatus runtime_close_internal(
    ShadowSpillRuntime *runtime,
    int wait_for_outstanding_work
) {
    if (runtime == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&runtime->mutex);
    if (atomic_load_explicit(&runtime->closed, memory_order_acquire) != 0U) {
        pthread_mutex_unlock(&runtime->mutex);
        return SHADOWSPILL_STATUS_OK;
    }
    atomic_store_explicit(&runtime->closing, 1U, memory_order_release);
    pthread_mutex_unlock(&runtime->mutex);
    ShadowSpillIdleWakeup *wakeup = &runtime->idle_wakeup;
    if (wait_for_outstanding_work) {
        pthread_mutex_lock(&wakeup->lock);
        while (shadowspill_failure_status(runtime) == SHADOWSPILL_STATUS_OK &&
               (atomic_load_explicit(
                    &runtime->actions.count, memory_order_acquire
                ) != 0U ||
                runtime->pending_retirements != 0U)) {
            pthread_cond_wait(&wakeup->condition, &wakeup->lock);
        }
        pthread_mutex_unlock(&wakeup->lock);
    }

    int synchronization_failed = 0;
    /*
     * Synchronizing a lane waits on the device. A close that is not waiting
     * for outstanding work is not in a position to wait on hardware either:
     * the work it would wait for is the work it just declined to finish.
     */
    for (uint32_t route_id = 0U;
         wait_for_outstanding_work && route_id < runtime->route_count;
         ++route_id) {
        ShadowSpillRouteState *route = &runtime->routes[route_id];
        if (route->lane_created && runtime->backend.synchronize_stream(
                runtime->backend.state, route->lane
            ) != 0) {
            synchronization_failed = 1;
        }
    }
    pthread_mutex_lock(&runtime->mutex);
    if (synchronization_failed) {
        shadowspill_latch_failure_locked(
            runtime,
            SHADOWSPILL_STATUS_BACKEND_FAILURE,
            SHADOWSPILL_FAILURE_REASON_BACKEND_CALL_REJECTED,
            SHADOWSPILL_RUNTIME_NO_ID,
            SHADOWSPILL_RUNTIME_NO_ID,
            0U
        );
    }
    atomic_store_explicit(&runtime->worker_stop, 1U, memory_order_release);
    pthread_mutex_unlock(&runtime->mutex);
    shadowspill_idle_notify(runtime);
    if (runtime->worker_started) {
        (void)pthread_join(runtime->worker_thread, NULL);
        runtime->worker_started = 0;
    }
    pthread_mutex_lock(&runtime->mutex);
    ShadowSpillStatus status = shadowspill_failure_status(runtime);
    atomic_store_explicit(&runtime->closed, 1U, memory_order_release);
    pthread_mutex_unlock(&runtime->mutex);
    shadowspill_idle_notify(runtime);
    shadowspill_runtime_release_resources(runtime);
    return status;
}

ShadowSpillStatus shadowspill_runtime_close(
    ShadowSpillRuntime *runtime
) {
    return runtime_close_internal(runtime, 1);
}

ShadowSpillStatus shadowspill_runtime_abandon(
    ShadowSpillRuntime *runtime,
    uint64_t *outstanding_actions,
    uint64_t *outstanding_retirements
) {
    if (runtime == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    if (outstanding_actions != NULL) {
        *outstanding_actions = atomic_load_explicit(
            &runtime->actions.count, memory_order_acquire
        );
    }
    if (outstanding_retirements != NULL) {
        *outstanding_retirements = runtime->pending_retirements;
    }
    atomic_store_explicit(&runtime->abandoned, 1U, memory_order_release);
    return runtime_close_internal(runtime, 0);
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
    shadowspill_runtime_release_primitives(runtime);
    free(runtime);
}
