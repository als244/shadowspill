/* Everything the runtime reports about itself, in one record. */
#include "internal.h"

#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

ShadowSpillStatus shadowspill_runtime_statistics(
    ShadowSpillRuntime *runtime,
    ShadowSpillRuntimeStatistics *statistics
) {
    if (runtime == NULL || statistics == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    if (runtime->pools == NULL || runtime->pool_count == 0U) {
        return SHADOWSPILL_STATUS_INVALID_STATE;
    }
    pthread_mutex_lock(&runtime->mutex);
    /* Every pool, not two the runtime picked: which pools carry which role is a
     * plan's choice, and a runtime may own more than two. */
    for (uint32_t pool_id = 0U; pool_id < runtime->pool_count; ++pool_id) {
        pthread_mutex_lock(&runtime->pools[pool_id].lock);
    }
    pthread_mutex_lock(&runtime->events.lock);
    pthread_mutex_lock(&runtime->retirements.lock);
    uint64_t retirement_records_fenced = 0U;
    uint64_t retirement_records_evented = 0U;
    uint64_t retirement_records_preparing = 0U;
    uint64_t retirement_records_unfenced = 0U;
    uint64_t caller_owned_allocations = 0U;
    for (uint32_t pool_id = 0U; pool_id < runtime->pool_count; ++pool_id) {
        for (const ShadowSpillMemoryLease *allocation =
                 runtime->pools[pool_id].active_leases;
             allocation != NULL; allocation = allocation->active_next) {
            if (allocation->pointer != NULL && allocation->ever_plan_owned &&
                !allocation->plan_owned && !allocation->framework_free_seen) {
                ++caller_owned_allocations;
            }
            if (!allocation->logical_freed || allocation->pointer == NULL) {
                continue;
            }
            if (allocation->retirement_event != NULL) {
                ++retirement_records_fenced;
            } else if (allocation->retirement_requirements != NULL) {
                ++retirement_records_evented;
            } else if (allocation->retirement_preparing) {
                ++retirement_records_preparing;
            } else {
                ++retirement_records_unfenced;
            }
        }
    }
    *statistics = (ShadowSpillRuntimeStatistics){
        .pool_count = runtime->pool_count,
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
        .event_lease_capacity = runtime->events.capacity,
        .event_lease_in_use = runtime->events.in_use,
        .event_lease_peak_in_use = runtime->events.peak_in_use,
        .event_lease_growth_rejections = runtime->events.growth_rejections,
        .event_lease_driver_creates = runtime->events.driver_creates,
        .event_lease_sealed = runtime->events.sealed,
        .timing_event_capacity = runtime->timing_events.capacity,
        .timing_event_in_use = runtime->timing_events.in_use,
        .timing_event_peak_in_use = runtime->timing_events.peak_in_use,
        .timing_event_driver_creates = runtime->timing_events.driver_creates,
        .retirement_record_capacity = runtime->retirements.capacity,
        .retirement_record_in_use = runtime->retirements.in_use,
        .retirement_record_peak_in_use = runtime->retirements.peak_in_use,
        .retirement_record_growth_rejections =
            runtime->retirements.growth_rejections,
        .caller_owned_allocations = caller_owned_allocations,
    };
    pthread_mutex_unlock(&runtime->retirements.lock);
    pthread_mutex_unlock(&runtime->events.lock);
    for (uint32_t pool_id = runtime->pool_count; pool_id != 0U;) {
        pthread_mutex_unlock(&runtime->pools[--pool_id].lock);
    }
    pthread_mutex_unlock(&runtime->mutex);
    return SHADOWSPILL_STATUS_OK;
}
