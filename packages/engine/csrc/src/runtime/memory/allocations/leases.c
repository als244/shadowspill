/* Creating one lease over pool memory, and releasing it. */
#include "internal.h"

static void publish_lease_record_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryLease *record,
    ShadowSpillAllocationCategory category
) {
    ShadowSpillMemoryPool *pool = record->pool;
    shadowspill_allocations_activate_allocation_locked(pool, record);
    shadowspill_allocations_index_allocation_id_locked(pool, record);
    shadowspill_allocations_index_allocation_pointer_locked(pool, record);
    pool->requested_allocated_bytes += record->requested_bytes;
    if (pool->requested_allocated_bytes >
        pool->peak_requested_allocated_bytes) {
        pool->peak_requested_allocated_bytes =
            pool->requested_allocated_bytes;
    }
    ++pool->live_allocations;
    shadowspill_append_allocation_event_locked(
        runtime, record, SHADOWSPILL_ALLOCATION_CREATED, category
    );
}

static ShadowSpillStatus own_and_publish_lease_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryLease *created,
    int plan_owned,
    ShadowSpillMemoryLease **record
) {
    ShadowSpillMemoryPool *pool = created->pool;
    created->plan_owned = plan_owned;
    publish_lease_record_locked(
        runtime,
        created,
        plan_owned ? SHADOWSPILL_ALLOCATION_PLANNED_OBJECT
                   : SHADOWSPILL_ALLOCATION_ANONYMOUS
    );
    const ShadowSpillStatus status = shadowspill_failure_status(runtime);
    if (status == SHADOWSPILL_STATUS_OK) {
        *record = created;
        return status;
    }
    shadowspill_allocations_unindex_allocation_pointer_locked(pool, created);
    shadowspill_allocations_unindex_allocation_id_locked(pool, created);
    shadowspill_allocations_deactivate_allocation_locked(created);
    pool->requested_allocated_bytes -= created->requested_bytes;
    --pool->live_allocations;
    (void)shadowspill_memory_pool_release_lease_locked(created);
    shadowspill_publish_pool_geometry_locked(pool);
    shadowspill_memory_pool_try_recycle_lease_record_locked(created);
    return status;
}

static ShadowSpillStatus create_lease_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryPool *pool,
    uint64_t bytes,
    uint64_t alignment,
    int plan_owned,
    ShadowSpillMemoryPlacement placement,
    ShadowSpillAllocationOrigin origin,
    ShadowSpillMemoryLease **record
) {
    if (record == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    *record = NULL;
    if (pool == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    if (alignment < pool->minimum_alignment) {
        alignment = pool->minimum_alignment;
    }
    ShadowSpillMemoryLease *created =
        shadowspill_memory_pool_acquire_lease_record_locked(
        runtime, pool, origin
    );
    if (created == NULL) {
        return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    }
    const int reserve_status = shadowspill_memory_pool_reserve_lease_locked(
        pool,
        created,
        bytes,
        alignment,
        placement
    );
    if (reserve_status != 0) {
        shadowspill_memory_pool_try_recycle_lease_record_locked(created);
        return reserve_status > 0
            ? SHADOWSPILL_STATUS_OUT_OF_MEMORY
            : SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    }
    shadowspill_publish_pool_geometry_locked(pool);
    return own_and_publish_lease_locked(
        runtime, created, plan_owned, record
    );
}

ShadowSpillStatus shadowspill_create_fixed_execution_lease_locked(
    ShadowSpillPlan *plan,
    const ShadowSpillFixedPlacementDescription *placement,
    int plan_owned,
    ShadowSpillAllocationOrigin origin,
    ShadowSpillMemoryLease **record
) {
    if (plan == NULL || placement == NULL || record == NULL ||
        (placement->kind != SHADOWSPILL_FIXED_TASK_ALLOCATION &&
         placement->kind != SHADOWSPILL_FIXED_ACTION_DESTINATION)) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    ShadowSpillRuntime *runtime = plan->runtime;
    *record = NULL;
    ShadowSpillMemoryPool *pool = plan->execution_pool;
    ShadowSpillMemoryLease *created =
        shadowspill_memory_pool_acquire_lease_record_locked(
        runtime, pool, origin
    );
    if (created == NULL) {
        return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    }
    ShadowSpillStatus status =
        shadowspill_fixed_layout_adopt_execution_lease_locked(
            plan,
            created,
            placement->offset,
            placement->bytes,
            placement->alignment_bytes
        );
    if (status != SHADOWSPILL_STATUS_OK) {
        shadowspill_memory_pool_try_recycle_lease_record_locked(created);
        return status;
    }
    return own_and_publish_lease_locked(
        runtime, created, plan_owned, record
    );
}

ShadowSpillStatus shadowspill_create_successor_lease_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryPool *pool,
    uint64_t bytes,
    uint64_t alignment,
    ShadowSpillAllocationOrigin origin,
    ShadowSpillMemoryLease **record
) {
    if (runtime == NULL || pool == NULL || record == NULL || bytes == 0U) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    *record = NULL;
    ShadowSpillMemoryLease *created =
        shadowspill_memory_pool_acquire_lease_record_locked(
        runtime, pool, origin
    );
    if (created == NULL) {
        return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    }
    const int reserve_status =
        shadowspill_memory_pool_reserve_causal_successor_locked(
            pool,
            created,
            bytes,
            alignment
        );
    if (reserve_status != 0) {
        shadowspill_memory_pool_try_recycle_lease_record_locked(created);
        return reserve_status > 0
            ? SHADOWSPILL_STATUS_OUT_OF_MEMORY
            : SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    }
    /* The predecessor is now promised to this transfer reservation. */
    shadowspill_allocations_unindex_reusable_locked(pool, created->causal_predecessor);
    created->plan_owned = 1;
    created->ever_plan_owned = 1;
    *record = created;
    return SHADOWSPILL_STATUS_OK;
}

static void publish_successor_lease_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryLease *successor
) {
    if (successor->active_previous_link != NULL) {
        return;
    }
    publish_lease_record_locked(
        runtime,
        successor,
        SHADOWSPILL_ALLOCATION_PLANNED_OBJECT
    );
    shadowspill_publish_pool_geometry_locked(successor->pool);
}

int shadowspill_acquire_reserved_lease_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryLease *successor,
    ShadowSpillEventLease **dependency_event
) {
    const int status = shadowspill_memory_pool_acquire_reserved_lease_locked(
        successor, dependency_event
    );
    if (status == 0) {
        publish_successor_lease_locked(runtime, successor);
    }
    return status;
}

void shadowspill_cancel_reservation_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryLease *lease
) {
    if (runtime == NULL || lease == NULL) {
        return;
    }
    if (lease->state == SHADOWSPILL_LEASE_SUCCESSOR_RESERVED) {
        ShadowSpillMemoryLease *predecessor = lease->causal_predecessor;
        if (shadowspill_memory_pool_cancel_reservation_locked(lease) != 0) {
            shadowspill_latch_failure_locked(
                runtime,
                SHADOWSPILL_STATUS_INTERNAL_FAILURE,
                SHADOWSPILL_FAILURE_REASON_RESERVATION_CANCEL_REJECTED,
                SHADOWSPILL_RUNTIME_NO_ID,
                lease->allocation_id,
                lease->requested_bytes
            );
        } else if (predecessor != NULL && predecessor->logical_freed &&
                   predecessor->pointer != NULL &&
                   !predecessor->retirement_preparing) {
            shadowspill_allocations_index_reusable_locked(
                predecessor->pool, predecessor
            );
        }
        if (lease->state == SHADOWSPILL_LEASE_FREE) {
            lease->plan_owned = 0;
            lease->framework_free_seen = 1;
            shadowspill_memory_pool_try_recycle_lease_record_locked(lease);
        }
        return;
    }
    shadowspill_release_lease_locked(runtime, lease);
}

ShadowSpillStatus shadowspill_create_lease_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryPool *pool,
    uint64_t bytes,
    uint64_t alignment,
    int plan_owned,
    ShadowSpillMemoryPlacement placement,
    ShadowSpillAllocationOrigin origin,
    ShadowSpillMemoryLease **record
) {
    return create_lease_locked(
        runtime,
        pool,
        bytes,
        alignment,
        plan_owned,
        placement,
        origin,
        record
    );
}

void shadowspill_release_lease_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryLease *allocation
) {
    if (allocation->pointer == NULL) {
        return;
    }
    ShadowSpillMemoryLease *causal_successor = allocation->causal_successor;
    ShadowSpillMemoryPool *pool = allocation->pool;
    shadowspill_allocations_unindex_reusable_locked(pool, allocation);
    shadowspill_append_allocation_event_locked(
        runtime,
        allocation,
        SHADOWSPILL_ALLOCATION_RELEASED,
        allocation->plan_owned ? SHADOWSPILL_ALLOCATION_PLANNED_OBJECT
                               : SHADOWSPILL_ALLOCATION_ANONYMOUS
    );
    const uint64_t requested_bytes = allocation->requested_bytes;
    const uint64_t charged_bytes = allocation->charged_bytes;
    const int retain_framework_lookup = allocation->ever_plan_owned &&
        !allocation->framework_free_seen;
    if (!retain_framework_lookup) {
        shadowspill_allocations_unindex_allocation_pointer_locked(pool, allocation);
        shadowspill_allocations_unindex_allocation_id_locked(pool, allocation);
    }
    if (shadowspill_memory_pool_release_lease_locked(allocation) != 0) {
        shadowspill_latch_failure_locked(
            runtime,
            SHADOWSPILL_STATUS_INTERNAL_FAILURE,
            SHADOWSPILL_FAILURE_REASON_LEASE_RELEASE_REJECTED,
            SHADOWSPILL_RUNTIME_NO_ID,
            allocation->allocation_id,
            charged_bytes
        );
        return;
    }
    if (shadowspill_memory_pool_release_use_records_locked(
            pool, allocation->uses
        ) != 0) {
        shadowspill_latch_failure_locked(
            runtime,
            SHADOWSPILL_STATUS_INVALID_STATE,
            SHADOWSPILL_FAILURE_REASON_USE_RECORD_RETURN_REJECTED,
            SHADOWSPILL_RUNTIME_NO_ID,
            allocation->allocation_id,
            0U
        );
        return;
    }
    allocation->uses = NULL;
    shadowspill_publish_pool_geometry_locked(pool);
    shadowspill_allocations_deactivate_allocation_locked(allocation);
    allocation->logical_freed = 1;
    allocation->plan_owned = 0;
    allocation->bound_object = NULL;
    pool->requested_allocated_bytes -= requested_bytes;
    if (pool->live_allocations != 0U) {
        --pool->live_allocations;
    }
    if (causal_successor != NULL &&
        causal_successor->state == SHADOWSPILL_LEASE_RESERVED) {
        publish_successor_lease_locked(runtime, causal_successor);
    }
    shadowspill_memory_pool_try_recycle_lease_record_locked(allocation);
}
