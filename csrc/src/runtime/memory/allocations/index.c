/* Finding an allocation: by id, by pointer, or among the reusable. */
#include "internal.h"

int shadowspill_allocations_stream_equal(
    ShadowSpillBackendStream left,
    ShadowSpillBackendStream right
) {
    return memcmp(&left, &right, sizeof(left)) == 0;
}

void shadowspill_publish_pool_geometry_locked(ShadowSpillMemoryPool *pool) {
    atomic_store_explicit(
        &pool->free_bytes_snapshot,
        shadowspill_memory_pool_free_bytes_locked(pool),
        memory_order_release
    );
    atomic_store_explicit(
        &pool->largest_free_bytes_snapshot,
        shadowspill_memory_pool_largest_free_locked(pool),
        memory_order_release
    );
}

uint64_t shadowspill_allocations_mix_index(uint64_t value, uint64_t bucket_count) {
    value ^= value >> 33U;
    value *= UINT64_C(0xff51afd7ed558ccd);
    value ^= value >> 33U;
    return value % bucket_count;
}

static void *allocation_lookup_pointer(
    const ShadowSpillMemoryLease *allocation
) {
    return allocation->pointer != NULL
        ? allocation->pointer
        : allocation->retired_pointer;
}

void shadowspill_allocations_index_allocation_id_locked(
    ShadowSpillMemoryPool *pool,
    ShadowSpillMemoryLease *allocation
) {
    if (allocation->in_id_index) {
        return;
    }
    const uint64_t bucket = shadowspill_allocations_mix_index(
        allocation->allocation_id, pool->allocation_index_bucket_count
    );
    allocation->id_index_next = pool->leases_by_id[bucket];
    pool->leases_by_id[bucket] = allocation;
    allocation->in_id_index = 1U;
}

void shadowspill_allocations_unindex_allocation_id_locked(
    ShadowSpillMemoryPool *pool,
    ShadowSpillMemoryLease *allocation
) {
    if (!allocation->in_id_index) {
        return;
    }
    const uint64_t bucket = shadowspill_allocations_mix_index(
        allocation->allocation_id, pool->allocation_index_bucket_count
    );
    ShadowSpillMemoryLease **link = &pool->leases_by_id[bucket];
    while (*link != NULL && *link != allocation) {
        link = &(*link)->id_index_next;
    }
    if (*link == allocation) {
        *link = allocation->id_index_next;
    }
    allocation->id_index_next = NULL;
    allocation->in_id_index = 0U;
}

void shadowspill_allocations_index_allocation_pointer_locked(
    ShadowSpillMemoryPool *pool,
    ShadowSpillMemoryLease *allocation
) {
    if (allocation->in_pointer_index) {
        return;
    }
    const uint64_t address =
        (uint64_t)(uintptr_t)allocation_lookup_pointer(allocation);
    const uint64_t bucket = shadowspill_allocations_mix_index(
        address, pool->allocation_index_bucket_count
    );
    allocation->pointer_index_next = pool->leases_by_pointer[bucket];
    pool->leases_by_pointer[bucket] = allocation;
    allocation->in_pointer_index = 1U;
}

void shadowspill_allocations_unindex_allocation_pointer_locked(
    ShadowSpillMemoryPool *pool,
    ShadowSpillMemoryLease *allocation
) {
    if (!allocation->in_pointer_index) {
        return;
    }
    const uint64_t address =
        (uint64_t)(uintptr_t)allocation_lookup_pointer(allocation);
    const uint64_t bucket = shadowspill_allocations_mix_index(
        address, pool->allocation_index_bucket_count
    );
    ShadowSpillMemoryLease **link = &pool->leases_by_pointer[bucket];
    while (*link != NULL && *link != allocation) {
        link = &(*link)->pointer_index_next;
    }
    if (*link == allocation) {
        *link = allocation->pointer_index_next;
    }
    allocation->pointer_index_next = NULL;
    allocation->in_pointer_index = 0U;
}

void shadowspill_allocations_index_reusable_locked(
    ShadowSpillMemoryPool *pool,
    ShadowSpillMemoryLease *allocation
) {
    if (allocation->in_reusable_index) {
        return;
    }
    const uint64_t bucket = shadowspill_allocations_mix_index(
        allocation->charged_bytes, pool->reusable_index_bucket_count
    );
    allocation->reusable_index_next = pool->reusable_leases_by_size[bucket];
    pool->reusable_leases_by_size[bucket] = allocation;
    allocation->in_reusable_index = 1U;
}

void shadowspill_allocations_unindex_reusable_locked(
    ShadowSpillMemoryPool *pool,
    ShadowSpillMemoryLease *allocation
) {
    if (!allocation->in_reusable_index) {
        return;
    }
    const uint64_t bucket = shadowspill_allocations_mix_index(
        allocation->charged_bytes, pool->reusable_index_bucket_count
    );
    ShadowSpillMemoryLease **link = &pool->reusable_leases_by_size[bucket];
    while (*link != NULL && *link != allocation) {
        link = &(*link)->reusable_index_next;
    }
    if (*link == allocation) {
        *link = allocation->reusable_index_next;
    }
    allocation->reusable_index_next = NULL;
    allocation->in_reusable_index = 0U;
}

void shadowspill_allocations_activate_allocation_locked(
    ShadowSpillMemoryPool *pool,
    ShadowSpillMemoryLease *allocation
) {
    allocation->active_next = pool->active_leases;
    allocation->active_previous_link = &pool->active_leases;
    if (allocation->active_next != NULL) {
        allocation->active_next->active_previous_link = &allocation->active_next;
    }
    pool->active_leases = allocation;
}

void shadowspill_allocations_deactivate_allocation_locked(
    ShadowSpillMemoryLease *allocation
) {
    if (allocation->active_previous_link == NULL) {
        return;
    }
    *allocation->active_previous_link = allocation->active_next;
    if (allocation->active_next != NULL) {
        allocation->active_next->active_previous_link =
            allocation->active_previous_link;
    }
    allocation->active_next = NULL;
    allocation->active_previous_link = NULL;
}

ShadowSpillMemoryLease *shadowspill_find_lease(
    ShadowSpillMemoryPool *pool,
    uint64_t allocation_id
) {
    const uint64_t bucket = shadowspill_allocations_mix_index(
        allocation_id, pool->allocation_index_bucket_count
    );
    for (ShadowSpillMemoryLease *record = pool->leases_by_id[bucket];
         record != NULL; record = record->id_index_next) {
        if (record->allocation_id == allocation_id) {
            return record;
        }
    }
    return NULL;
}

ShadowSpillMemoryLease *shadowspill_find_lease_by_pointer(
    ShadowSpillMemoryPool *pool,
    const void *pointer
) {
    const uint64_t bucket = shadowspill_allocations_mix_index(
        (uint64_t)(uintptr_t)pointer,
        pool->allocation_index_bucket_count
    );
    for (ShadowSpillMemoryLease *record = pool->leases_by_pointer[bucket];
         record != NULL; record = record->pointer_index_next) {
        if (record->pointer == pointer && !record->logical_freed) {
            return record;
        }
        if (record->logical_freed && record->ever_plan_owned &&
            !record->framework_free_seen &&
            record->retired_pointer == pointer) {
            return record;
        }
    }
    return NULL;
}
