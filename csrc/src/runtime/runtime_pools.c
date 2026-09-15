/* Growing a pool, and reading what one holds. */
#include "internal.h"

#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

ShadowSpillStatus shadowspill_memory_pool_grow(
    ShadowSpillRuntime *runtime,
    uint32_t pool_id,
    uint64_t capacity_bytes
) {
    ShadowSpillMemoryPool *pool = shadowspill_runtime_pool(runtime, pool_id);
    if (pool == NULL || capacity_bytes > SIZE_MAX) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    ShadowSpillStatus status = shadowspill_runtime_wait_idle(runtime);
    if (status != SHADOWSPILL_STATUS_OK) {
        return status;
    }
    pthread_mutex_lock(&runtime->mutex);
    status = shadowspill_current_status_locked(runtime);
    uint64_t current_bytes = pool->ranges.capacity;
    if (status != SHADOWSPILL_STATUS_OK) {
        goto done;
    }
    if (atomic_load_explicit(&runtime->closing, memory_order_acquire) != 0U ||
        atomic_load_explicit(
            &runtime->actions.count, memory_order_acquire
        ) != 0U ||
        runtime->pending_retirements != 0U) {
        status = SHADOWSPILL_STATUS_INVALID_STATE;
        goto done;
    }
    if (capacity_bytes < current_bytes) {
        status = SHADOWSPILL_STATUS_INVALID_ARGUMENT;
        goto done;
    }
    if (capacity_bytes == current_bytes) {
        goto done;
    }

    void *replacement = NULL;
    if (shadowspill_memory_pool_arena_allocate(
            pool, capacity_bytes, &replacement
        ) != 0) {
        status = SHADOWSPILL_STATUS_BACKEND_FAILURE;
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
        (void)shadowspill_memory_pool_arena_release(pool, replacement, capacity_bytes);
        status = SHADOWSPILL_STATUS_INTERNAL_FAILURE;
        goto done;
    }
    if (pool->base != NULL && shadowspill_memory_pool_arena_release(
            pool, pool->base, pool->arena_bytes
        ) != 0) {
        shadowspill_range_destroy(&ranges);
        (void)shadowspill_memory_pool_arena_release(pool, replacement, capacity_bytes);
        status = SHADOWSPILL_STATUS_BACKEND_FAILURE;
        goto done;
    }
    shadowspill_memory_pool_rebase_locked(
        pool, replacement
    );
    pool->arena_bytes = capacity_bytes;
    shadowspill_range_destroy(&pool->ranges);
    pool->ranges = ranges;
    shadowspill_publish_pool_geometry_locked(pool);

done:
    pthread_mutex_unlock(&runtime->mutex);
    return status;
}

ShadowSpillStatus shadowspill_memory_pool_live_allocations(
    ShadowSpillRuntime *runtime,
    uint32_t pool_id,
    ShadowSpillLiveAllocation *out,
    uint64_t capacity,
    uint64_t *count
) {
    if (runtime == NULL || count == NULL || (out == NULL && capacity != 0U)) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    ShadowSpillMemoryPool *pool = shadowspill_runtime_pool(runtime, pool_id);
    if (pool == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    /* Every entry is copied out under the pool's lock, so a caller reads one
       consistent moment rather than a list mutating under it. The count is
       always the whole list: a caller given a short buffer learns the size it
       needs instead of a silently truncated answer. */
    uint64_t live = 0U;
    pthread_mutex_lock(&pool->lock);
    for (ShadowSpillMemoryLease *lease = pool->active_leases; lease != NULL;
         lease = lease->active_next) {
        if (live < capacity) {
            out[live] = (ShadowSpillLiveAllocation){
                .allocation_id = lease->allocation_id,
                .offset = lease->offset,
                .charged_bytes = lease->charged_bytes,
                .requested_bytes = lease->requested_bytes,
                .origin_plan_id = lease->origin_plan_id,
                .origin_task_id = lease->origin_task_id,
                .origin_task_invocation = lease->origin_task_invocation,
                .origin_task_allocation_ordinal =
                    lease->origin_task_allocation_ordinal,
                .object_id = lease->bound_object == NULL
                    ? SHADOWSPILL_RUNTIME_NO_ID
                    : lease->bound_object->object_id,
                .references = atomic_load_explicit(
                    &lease->references, memory_order_acquire
                ),
                .scratch = (uint8_t)(
                    lease->origin_task_allocation_is_scratch != 0U
                ),
                .plan_owned = (uint8_t)(lease->plan_owned != 0),
                .ever_plan_owned = (uint8_t)(lease->ever_plan_owned != 0),
                .logical_freed = (uint8_t)(lease->logical_freed != 0),
                .framework_free_seen =
                    (uint8_t)(lease->framework_free_seen != 0),
            };
        }
        ++live;
    }
    pthread_mutex_unlock(&pool->lock);
    *count = live;
    return SHADOWSPILL_STATUS_OK;
}

ShadowSpillStatus shadowspill_memory_pool_statistics(
    ShadowSpillRuntime *runtime,
    uint32_t pool_id,
    ShadowSpillMemoryPoolStatistics *statistics
) {
    ShadowSpillMemoryPool *pool = shadowspill_runtime_pool(runtime, pool_id);
    if (pool == NULL || statistics == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&pool->lock);
    const uint64_t free_bytes =
        shadowspill_memory_pool_free_bytes_locked(pool);
    const uint64_t largest_free =
        shadowspill_memory_pool_largest_free_locked(pool);
    *statistics = (ShadowSpillMemoryPoolStatistics){
        .pool_id = pool_id,
        .kind = pool->kind,
        .capacity_bytes = pool->ranges.capacity,
        .requested_allocated_bytes = pool->requested_allocated_bytes,
        .peak_requested_allocated_bytes = pool->peak_requested_allocated_bytes,
        .allocated_bytes = pool->ranges.allocated,
        .peak_allocated_bytes = pool->ranges.peak_allocated,
        .free_bytes = free_bytes,
        .free_prefix_bytes =
            shadowspill_memory_pool_free_prefix_locked(pool),
        .largest_free_range_bytes = largest_free,
        .external_fragmentation_bytes = free_bytes - largest_free,
        .live_allocations = pool->live_allocations,
        .blocked_allocators = pool->blocked_allocators,
        .memory_lease_record_capacity = pool->lease_record_capacity,
        .memory_lease_record_in_use = pool->lease_record_in_use,
        .memory_lease_record_peak_in_use = pool->lease_record_peak_in_use,
        .memory_lease_record_growth_rejections =
            pool->lease_record_growth_rejections,
        .lease_use_record_capacity = pool->use_record_capacity,
        .lease_use_record_in_use = pool->use_record_in_use,
        .lease_use_record_peak_in_use = pool->use_record_peak_in_use,
        .lease_use_record_growth_rejections =
            pool->use_record_growth_rejections,
    };
    pthread_mutex_unlock(&pool->lock);
    return SHADOWSPILL_STATUS_OK;
}
