/* The lease records a pool owns, taken and given back. */
#include "internal.h"

static void initialize_memory_lease_record(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryPool *pool,
    ShadowSpillMemoryLease *record,
    ShadowSpillAllocationOrigin origin
) {
    ShadowSpillMemoryLease *ownership_next = record->ownership_next;
    memset(record, 0, sizeof(*record));
    record->metadata_owner = pool;
    record->ownership_next = ownership_next;
    record->metadata_in_use = 1U;
    record->allocation_id = atomic_fetch_add_explicit(
        &runtime->next_allocation_id, 1U, memory_order_relaxed
    );
    atomic_init(&record->references, 1U);
    record->generation = atomic_fetch_add_explicit(
        &runtime->next_generation, 1U, memory_order_relaxed
    );
    record->origin_plan_id = origin.plan_id;
    record->origin_task_id = origin.task_id;
    record->origin_task_allocation_sequence = SHADOWSPILL_RUNTIME_NO_ID;
    record->origin_task_allocation_ordinal = SHADOWSPILL_RUNTIME_NO_ID;
    record->origin_task_allocation_is_scratch = 0U;
    record->release_task_id = SHADOWSPILL_RUNTIME_NO_ID;
}

ShadowSpillMemoryLease *shadowspill_memory_pool_acquire_lease_record_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryPool *pool,
    ShadowSpillAllocationOrigin origin
) {
    ShadowSpillMemoryLease *record = pool->free_lease_records;
    if (record != NULL) {
        pool->free_lease_records = record->free_record_next;
        record->free_record_next = NULL;
        --pool->lease_record_available;
    } else {
        /*
         * A pool's bytes are sealed and its metadata is not, and the two are
         * different kinds of resource.  How many bytes a plan needs is what
         * the plan says.  How many lease records it needs is not: a record is
         * taken per live lease and one more each time a free range is split
         * to fit a request, so the peak follows the order allocations and
         * releases happen in and how far the dispatcher runs ahead of the
         * device.  Neither is visible when the reserve is computed, and a
         * count that cannot be derived cannot be a correctness bound.
         *
         * So the reserve is a warm start rather than a limit.  It is sized to
         * cover the steady state, growth happens only at a new high-water
         * mark and converges within the first invocations, and a pool that
         * outgrows it says so through its capacity rather than refusing a
         * request the memory was there to serve.
         */
        record = calloc(1U, sizeof(*record));
        if (record == NULL) {
            ++pool->lease_record_growth_rejections;
            return NULL;
        }
        record->metadata_owner = pool;
        record->ownership_next = pool->owned_leases;
        pool->owned_leases = record;
        ++pool->lease_record_capacity;
    }
    ++pool->lease_record_in_use;
    if (pool->lease_record_in_use > pool->lease_record_peak_in_use) {
        pool->lease_record_peak_in_use = pool->lease_record_in_use;
    }
    initialize_memory_lease_record(runtime, pool, record, origin);
    return record;
}

static int memory_lease_record_is_recyclable(
    const ShadowSpillMemoryLease *record
) {
    return record != NULL && record->metadata_owner != NULL &&
        record->metadata_in_use && record->pool == NULL &&
        record->state == SHADOWSPILL_LEASE_FREE &&
        record->pointer == NULL && record->active_previous_link == NULL &&
        !record->in_id_index && !record->in_pointer_index &&
        !record->in_reusable_index && !record->task_retirement_linked &&
        record->uses == NULL && record->retirement_requirements == NULL &&
        record->retirement_event == NULL && record->bound_object == NULL &&
        record->causal_predecessor == NULL &&
        record->causal_successor == NULL && record->pool_next == NULL &&
        record->pool_previous_link == NULL &&
        (!record->ever_plan_owned || record->framework_free_seen) &&
        atomic_load_explicit(&record->references, memory_order_acquire) == 1U;
}

void shadowspill_memory_pool_try_recycle_lease_record_locked(
    ShadowSpillMemoryLease *record
) {
    if (!memory_lease_record_is_recyclable(record)) {
        return;
    }
    ShadowSpillMemoryPool *owner = record->metadata_owner;
    ShadowSpillMemoryLease *ownership_next = record->ownership_next;
    memset(record, 0, sizeof(*record));
    record->metadata_owner = owner;
    record->ownership_next = ownership_next;
    atomic_init(&record->references, 1U);
    record->free_record_next = owner->free_lease_records;
    owner->free_lease_records = record;
    ++owner->lease_record_available;
    if (owner->lease_record_in_use != 0U) {
        --owner->lease_record_in_use;
    }
}

void shadowspill_memory_lease_retain(ShadowSpillMemoryLease *lease) {
    if (lease != NULL) {
        (void)atomic_fetch_add_explicit(
            &lease->references, 1U, memory_order_relaxed
        );
    }
}

void shadowspill_memory_lease_release(ShadowSpillMemoryLease *lease) {
    if (lease == NULL || lease->metadata_owner == NULL) {
        return;
    }
    uint32_t references = atomic_load_explicit(
        &lease->references, memory_order_acquire
    );
    while (references > 1U && !atomic_compare_exchange_weak_explicit(
               &lease->references,
               &references,
               references - 1U,
               memory_order_acq_rel,
               memory_order_acquire
           )) {
    }
    if (references != 2U) {
        return;
    }
    ShadowSpillMemoryPool *owner = lease->metadata_owner;
    pthread_mutex_lock(&owner->lock);
    shadowspill_memory_pool_try_recycle_lease_record_locked(lease);
    pthread_mutex_unlock(&owner->lock);
}
