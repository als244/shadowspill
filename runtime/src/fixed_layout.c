#include "internal.h"

#include <stdint.h>

/*
 * Own one contiguous execution-pool slice for an admitted physical layout.
 * Individual fixed leases borrow subranges of this slice; only this owner
 * changes the production range allocator's physical accounting.
 */
ShadowSpillRuntimeStatus shadowspill_fixed_layout_reserve_slice(
    ShadowSpillRuntime *runtime,
    uint64_t bytes
) {
    if (runtime == NULL || bytes == 0U) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    ShadowSpillMemoryPool *pool = shadowspill_execution_pool(runtime);
    shadowspill_memory_pool_lock_reservation(pool);
    ShadowSpillRuntimeStatus status = SHADOWSPILL_RUNTIME_OK;
    if (runtime->fixed_layout.active ||
        runtime->active_execution_leases != NULL) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
    } else {
        uint64_t offset = 0U;
        const int reserve_status = shadowspill_memory_pool_reserve_locked(
            pool,
            bytes,
            pool->minimum_alignment,
            SHADOWSPILL_MEMORY_BEST_FIT_LOW,
            &offset
        );
        if (reserve_status != 0) {
            status = reserve_status > 0
                ? SHADOWSPILL_RUNTIME_OUT_OF_MEMORY
                : SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
        } else {
            runtime->fixed_layout = (ShadowSpillFixedLayoutState){
                .slice_offset = offset,
                .slice_bytes = bytes,
                .active = 1U,
            };
            shadowspill_publish_execution_geometry_locked(runtime);
        }
    }
    shadowspill_memory_pool_unlock_reservation(pool);
    shadowspill_memory_pool_relinquish_reservation(pool);
    return status;
}

ShadowSpillRuntimeStatus shadowspill_fixed_layout_clear(
    ShadowSpillRuntime *runtime
) {
    if (runtime == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    ShadowSpillMemoryPool *pool = shadowspill_execution_pool(runtime);
    shadowspill_memory_pool_lock_reservation(pool);
    ShadowSpillRuntimeStatus status = SHADOWSPILL_RUNTIME_OK;
    if (!runtime->fixed_layout.active) {
        /* Clearing an absent plan slice is intentionally idempotent. */
    } else if (runtime->active_execution_leases != NULL ||
               pool->leases != NULL) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
    } else if (shadowspill_memory_pool_release_locked(
                   pool,
                   runtime->fixed_layout.slice_offset,
                   runtime->fixed_layout.slice_bytes
               ) != 0) {
        status = SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    } else {
        runtime->fixed_layout = (ShadowSpillFixedLayoutState){0};
        shadowspill_publish_execution_geometry_locked(runtime);
    }
    shadowspill_memory_pool_unlock_reservation(pool);
    shadowspill_memory_pool_relinquish_reservation(pool);
    return status;
}

ShadowSpillRuntimeStatus shadowspill_fixed_layout_adopt_execution_lease_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryLease *lease,
    uint64_t relative_offset,
    uint64_t bytes,
    uint64_t alignment
) {
    if (runtime == NULL || lease == NULL || !runtime->fixed_layout.active ||
        bytes == 0U || relative_offset > runtime->fixed_layout.slice_bytes ||
        bytes > runtime->fixed_layout.slice_bytes - relative_offset) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    ShadowSpillMemoryPool *pool = shadowspill_execution_pool(runtime);
    if (alignment < pool->minimum_alignment) {
        alignment = pool->minimum_alignment;
    }
    const uint64_t absolute_offset =
        runtime->fixed_layout.slice_offset + relative_offset;
    if (absolute_offset % alignment != 0U ||
        shadowspill_memory_pool_adopt_borrowed_lease_locked(
            pool, lease, bytes, alignment, absolute_offset
        ) != 0) {
        return SHADOWSPILL_RUNTIME_PLAN_VIOLATION;
    }
    return SHADOWSPILL_RUNTIME_OK;
}
