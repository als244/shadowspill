/* Reading what a pool holds. */
#include "internal.h"

#include <pthread.h>
#include <stddef.h>
#include <stdint.h>

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
