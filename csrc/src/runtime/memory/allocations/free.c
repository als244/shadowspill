/* The free callback: what a retirement waits on before it lands. */
#include "internal.h"

ShadowSpillStatus shadowspill_publish_task_retirement_event(
    ShadowSpillRuntime *runtime,
    uint64_t task_id,
    ShadowSpillBackendStream stream
) {
    ShadowSpillMemoryPool *pool = shadowspill_current_allocation_pool(runtime);
    if (pool == NULL) {
        return SHADOWSPILL_STATUS_INVALID_STATE;
    }
    uint64_t count = 0U;
    shadowspill_memory_pool_lock_foreground(pool);
    for (ShadowSpillMemoryLease *allocation =
             shadowspill_current_task_retirements(runtime);
         allocation != NULL;
         allocation = allocation->task_retirement_next) {
        if (allocation->logical_freed && allocation->pointer != NULL &&
            allocation->release_task_id == task_id &&
            allocation->retirement_requirements == NULL &&
            allocation->retirement_event == NULL) {
            ++count;
        }
    }
    shadowspill_memory_pool_unlock_foreground(pool);
    if (count == 0U) {
        return SHADOWSPILL_STATUS_OK;
    }
    ShadowSpillEventLease *task_completion_event = NULL;
    const ShadowSpillStatus event_status =
        shadowspill_event_lease_create_locked(runtime, &task_completion_event);
    if (event_status != SHADOWSPILL_STATUS_OK ||
        runtime->backend.record_event(
            runtime->backend.state,
            task_completion_event->event,
            stream
        ) != 0 || shadowspill_completion_submit(
            runtime,
            stream,
            task_completion_event,
            SHADOWSPILL_RUNTIME_NO_ID,
            SHADOWSPILL_RUNTIME_NO_ID
        ) != SHADOWSPILL_STATUS_OK) {
        if (task_completion_event != NULL) {
            (void)shadowspill_event_lease_release(
                runtime, task_completion_event
            );
        }
        return SHADOWSPILL_STATUS_BACKEND_FAILURE;
    }
    ShadowSpillStatus status = SHADOWSPILL_STATUS_OK;
    shadowspill_memory_pool_lock_foreground(pool);
    for (ShadowSpillMemoryLease *allocation =
             shadowspill_current_task_retirements(runtime);
         allocation != NULL;
         allocation = allocation->task_retirement_next) {
        if (!allocation->logical_freed || allocation->pointer == NULL ||
            allocation->release_task_id != task_id ||
            allocation->retirement_requirements != NULL ||
            allocation->retirement_event != NULL) {
            continue;
        }
        if (shadowspill_memory_pool_publish_retirement_dependency_locked(
                allocation, task_completion_event
            ) != 0) {
            status = SHADOWSPILL_STATUS_INVALID_STATE;
            break;
        }
        allocation->retirement_event = task_completion_event;
        shadowspill_event_lease_retain(task_completion_event);
        const ShadowSpillStatus enqueue_status =
            shadowspill_retirement_enqueue_locked(runtime, allocation);
        if (enqueue_status != SHADOWSPILL_STATUS_OK) {
            status = enqueue_status;
            break;
        }
    }
    shadowspill_memory_pool_unlock_foreground(pool);
    (void)shadowspill_event_lease_release(runtime, task_completion_event);
    if (status != SHADOWSPILL_STATUS_OK) {
        return status;
    }
    return SHADOWSPILL_STATUS_OK;
}

static ShadowSpillStatus release_requirement_events(
    ShadowSpillRuntime *runtime,
    ShadowSpillLeaseUseRecord *requirements
) {
    ShadowSpillStatus status = SHADOWSPILL_STATUS_OK;
    for (ShadowSpillLeaseUseRecord *requirement = requirements;
         requirement != NULL; requirement = requirement->next) {
        if (requirement->event != NULL) {
            if (shadowspill_event_lease_release(
                    runtime, requirement->event
                ) != 0) {
                status = SHADOWSPILL_STATUS_BACKEND_FAILURE;
                continue;
            }
            requirement->event = NULL;
        }
    }
    return status;
}

static ShadowSpillStatus record_retirement_requirements(
    ShadowSpillRuntime *runtime,
    ShadowSpillLeaseUseRecord *requirements,
    uint64_t allocation_id
) {
    for (ShadowSpillLeaseUseRecord *requirement = requirements;
         requirement != NULL; requirement = requirement->next) {
        ShadowSpillStatus status =
            shadowspill_event_lease_create_locked(
                runtime, &requirement->event
            );
        if (status == SHADOWSPILL_STATUS_OK &&
            runtime->backend.record_event(
                runtime->backend.state,
                requirement->event->event,
                requirement->stream
            ) != 0) {
            status = SHADOWSPILL_STATUS_BACKEND_FAILURE;
        }
        if (status == SHADOWSPILL_STATUS_OK) {
            status = shadowspill_completion_submit(
                runtime,
                requirement->stream,
                requirement->event,
                SHADOWSPILL_RUNTIME_NO_ID,
                allocation_id
            );
        }
        if (status != SHADOWSPILL_STATUS_OK) {
            const ShadowSpillStatus release_status =
                release_requirement_events(runtime, requirements);
            return release_status == SHADOWSPILL_STATUS_OK
                ? status : release_status;
        }
    }
    return SHADOWSPILL_STATUS_OK;
}

ShadowSpillStatus shadowspill_memory_pool_free(
    ShadowSpillRuntime *runtime,
    uint32_t pool_id,
    uint64_t allocation_id,
    ShadowSpillBackendStream stream
) {
    ShadowSpillMemoryPool *pool = shadowspill_runtime_pool(runtime, pool_id);
    if (pool == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    shadowspill_memory_pool_lock_foreground(pool);
    ShadowSpillMemoryLease *allocation = shadowspill_find_lease(
        pool, allocation_id
    );
    ShadowSpillStatus status = SHADOWSPILL_STATUS_OK;
    if (allocation == NULL) {
        status = SHADOWSPILL_STATUS_INVALID_STATE;
        goto done;
    }
    if (allocation->logical_freed) {
        if (allocation->ever_plan_owned && !allocation->framework_free_seen) {
            allocation->framework_free_seen = 1;
            shadowspill_allocations_unindex_allocation_pointer_locked(pool, allocation);
            shadowspill_allocations_unindex_allocation_id_locked(pool, allocation);
            shadowspill_memory_pool_try_recycle_lease_record_locked(allocation);
            goto done;
        }
        status = SHADOWSPILL_STATUS_INVALID_STATE;
        goto done;
    }
    if (allocation->plan_owned) {
        if (allocation->framework_free_seen) {
            status = SHADOWSPILL_STATUS_INVALID_STATE;
            goto done;
        }
        allocation->framework_free_seen = 1;
        goto done;
    }
    if (allocation->ever_plan_owned) {
        if (allocation->framework_free_seen) {
            status = SHADOWSPILL_STATUS_INVALID_STATE;
            goto done;
        }
        allocation->framework_free_seen = 1;
    }
    if (shadowspill_allocations_append_lease_use_locked(allocation, stream) != 0) {
        status = SHADOWSPILL_STATUS_INTERNAL_FAILURE;
        goto done;
    }
    const uint64_t task_id = shadowspill_current_task_id(runtime);
    const int task_local_same_stream =
        task_id != SHADOWSPILL_RUNTIME_NO_ID &&
        allocation->uses != NULL &&
        allocation->uses->next == NULL &&
        shadowspill_allocations_stream_equal(allocation->uses->stream, stream);
    if (task_local_same_stream) {
        status = shadowspill_release_task_allocation(
            runtime,
            allocation->origin_task_id,
            allocation->origin_task_invocation,
            allocation->origin_task_allocation_ordinal,
            allocation->origin_task_allocation_is_scratch,
            allocation->requested_bytes,
            allocation->charged_bytes,
            allocation->alignment_bytes
        );
        if (status != SHADOWSPILL_STATUS_OK) {
            goto done;
        }
        allocation->release_task_id = task_id;
        allocation->logical_freed = 1;
        if (shadowspill_memory_pool_begin_retirement_locked(
                allocation, NULL, 1
            ) != 0) {
            status = SHADOWSPILL_STATUS_INVALID_STATE;
            goto done;
        }
        shadowspill_allocations_index_reusable_locked(pool, allocation);
        if (shadowspill_track_task_retirement(runtime, allocation) != 0) {
            status = SHADOWSPILL_STATUS_INVALID_STATE;
            goto done;
        }
        shadowspill_append_allocation_event_locked(
            runtime,
            allocation,
            SHADOWSPILL_ALLOCATION_LOGICAL_FREED,
            SHADOWSPILL_ALLOCATION_ANONYMOUS
        );
        (void)atomic_fetch_add_explicit(
            &runtime->pending_retirements, 1U, memory_order_release
        );
        (void)atomic_fetch_add_explicit(
            &pool->pending_retirements, 1U, memory_order_release
        );
        if (shadowspill_failure_status(runtime) != SHADOWSPILL_STATUS_OK) {
            status = shadowspill_failure_status(runtime);
        }
        goto done;
    }
    const uint64_t generation = allocation->generation;
    status = shadowspill_release_task_allocation(
        runtime,
        allocation->origin_task_id,
        allocation->origin_task_invocation,
        allocation->origin_task_allocation_ordinal,
        allocation->origin_task_allocation_is_scratch,
        allocation->requested_bytes,
        allocation->charged_bytes,
        allocation->alignment_bytes
    );
    if (status != SHADOWSPILL_STATUS_OK) {
        goto done;
    }
    allocation->release_task_id = task_id;
    allocation->logical_freed = 1;
    if (shadowspill_memory_pool_begin_retirement_locked(
            allocation, NULL, 0
        ) != 0) {
        status = SHADOWSPILL_STATUS_INVALID_STATE;
        goto done;
    }
    allocation->retirement_preparing = 1U;
    shadowspill_append_allocation_event_locked(
        runtime,
        allocation,
        SHADOWSPILL_ALLOCATION_LOGICAL_FREED,
        SHADOWSPILL_ALLOCATION_ANONYMOUS
    );
    (void)atomic_fetch_add_explicit(
        &runtime->pending_retirements, 1U, memory_order_release
    );
    (void)atomic_fetch_add_explicit(
        &pool->pending_retirements, 1U, memory_order_release
    );
    ShadowSpillLeaseUseRecord *requirements = allocation->uses;
    shadowspill_memory_pool_unlock_foreground(pool);

    status = record_retirement_requirements(
        runtime, requirements, allocation_id
    );

    shadowspill_memory_pool_lock_foreground(pool);
    allocation = shadowspill_find_lease(pool, allocation_id);
    if (allocation == NULL || allocation->generation != generation ||
        !allocation->retirement_preparing) {
        shadowspill_memory_pool_unlock_foreground(pool);
        const ShadowSpillStatus release_status =
            release_requirement_events(runtime, requirements);
        shadowspill_memory_pool_lock_foreground(pool);
        if (status == SHADOWSPILL_STATUS_OK) {
            status = release_status == SHADOWSPILL_STATUS_OK
                ? SHADOWSPILL_STATUS_INVALID_STATE : release_status;
        }
    } else {
        allocation->retirement_requirements =
            status == SHADOWSPILL_STATUS_OK ? requirements : NULL;
        allocation->retirement_preparing = 0U;
        if (status == SHADOWSPILL_STATUS_OK) {
            shadowspill_allocations_index_reusable_locked(pool, allocation);
            status = shadowspill_retirement_enqueue_locked(
                runtime, allocation
            );
        }
    }
    if (status != SHADOWSPILL_STATUS_OK) {
        shadowspill_latch_failure_locked(
            runtime,
            status,
            SHADOWSPILL_FAILURE_REASON_RETIREMENT_ENQUEUE_REJECTED,
            SHADOWSPILL_RUNTIME_NO_ID,
            allocation_id,
            0U
        );
    }
    if (shadowspill_failure_status(runtime) != SHADOWSPILL_STATUS_OK) {
        status = shadowspill_failure_status(runtime);
    }

done:
    shadowspill_memory_pool_unlock_foreground(pool);
    return status;
}

void shadowspill_finalize_aborted_task_retirements(
    ShadowSpillRuntime *runtime,
    uint64_t task_id
) {
    if (runtime == NULL || task_id == SHADOWSPILL_RUNTIME_NO_ID) {
        return;
    }
    ShadowSpillMemoryPool *pool = shadowspill_current_allocation_pool(runtime);
    if (pool == NULL) {
        return;
    }
    for (;;) {
        ShadowSpillMemoryLease *allocation = NULL;
        ShadowSpillLeaseUseRecord *requirements = NULL;
        uint64_t allocation_id = SHADOWSPILL_RUNTIME_NO_ID;
        uint64_t generation = 0U;

        shadowspill_memory_pool_lock_foreground(pool);
        for (allocation = shadowspill_current_task_retirements(runtime);
             allocation != NULL;
             allocation = allocation->task_retirement_next) {
            if (!allocation->logical_freed || allocation->pointer == NULL ||
                allocation->release_task_id != task_id ||
                allocation->retirement_requirements != NULL ||
                allocation->retirement_event != NULL ||
                allocation->retirement_preparing) {
                continue;
            }
            allocation->retirement_preparing = 1U;
            allocation->causal_dependency_expected = 0U;
            requirements = allocation->uses;
            allocation_id = allocation->allocation_id;
            generation = allocation->generation;
            shadowspill_memory_lease_retain(allocation);
            break;
        }
        shadowspill_memory_pool_unlock_foreground(pool);
        if (allocation == NULL) {
            return;
        }

        ShadowSpillStatus status = record_retirement_requirements(
            runtime, requirements, allocation_id
        );
        shadowspill_memory_pool_lock_foreground(pool);
        const int unchanged = allocation->pool == pool &&
            allocation->generation == generation &&
            allocation->retirement_preparing;
        if (unchanged) {
            allocation->retirement_preparing = 0U;
            if (status == SHADOWSPILL_STATUS_OK) {
                allocation->retirement_requirements = requirements;
                status = shadowspill_retirement_enqueue_locked(
                    runtime, allocation
                );
            }
        }
        shadowspill_memory_pool_unlock_foreground(pool);
        if (!unchanged && status == SHADOWSPILL_STATUS_OK) {
            status = release_requirement_events(runtime, requirements);
            if (status == SHADOWSPILL_STATUS_OK) {
                status = SHADOWSPILL_STATUS_INVALID_STATE;
            }
        }
        shadowspill_memory_lease_release(allocation);
        if (status != SHADOWSPILL_STATUS_OK) {
            shadowspill_latch_task_failure(
                runtime,
                status,
                SHADOWSPILL_FAILURE_REASON_LEASE_RELEASE_REJECTED,
                task_id,
                SHADOWSPILL_RUNTIME_NO_ID,
                allocation_id,
                0U
            );
            return;
        }
    }
}
