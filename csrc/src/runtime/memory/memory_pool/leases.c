/* The release frontier: reserve, adopt, retire, release. */
#include "internal.h"

int shadowspill_memory_pool_reserve_locked(
    ShadowSpillMemoryPool *pool,
    uint64_t bytes,
    uint64_t alignment,
    ShadowSpillMemoryPlacement placement,
    uint64_t *offset
) {
    if (pool == NULL || !pool->initialized || offset == NULL || bytes == 0U) {
        return -1;
    }
    if (alignment < pool->minimum_alignment) {
        alignment = pool->minimum_alignment;
    }
    switch (placement) {
        case SHADOWSPILL_MEMORY_FIRST_FIT:
            return shadowspill_range_allocate(
                &pool->ranges, bytes, alignment, offset
            );
        case SHADOWSPILL_MEMORY_BEST_FIT_LOW:
            return shadowspill_range_allocate_best_fit_low(
                &pool->ranges, bytes, alignment, offset
            );
        case SHADOWSPILL_MEMORY_BEST_FIT_HIGH:
            return shadowspill_range_allocate_best_fit_high(
                &pool->ranges, bytes, alignment, offset
            );
    }
    return -1;
}

int shadowspill_memory_pool_release_locked(
    ShadowSpillMemoryPool *pool,
    uint64_t offset,
    uint64_t bytes
) {
    if (pool == NULL || !pool->initialized) {
        return -1;
    }
    const int status = shadowspill_range_free(&pool->ranges, offset, bytes);
    if (status == 0) {
        (void)atomic_fetch_add_explicit(
            &pool->capacity_epoch, 1U, memory_order_release
        );
    }
    return status;
}

int shadowspill_memory_pool_reserve_lease_locked(
    ShadowSpillMemoryPool *pool,
    ShadowSpillMemoryLease *lease,
    uint64_t bytes,
    uint64_t alignment,
    ShadowSpillMemoryPlacement placement
) {
    if (pool == NULL || lease == NULL || lease->state != SHADOWSPILL_LEASE_FREE) {
        return -1;
    }
    const uint64_t charged = bytes == 0U ? 1U : bytes;
    if (alignment < pool->minimum_alignment) {
        alignment = pool->minimum_alignment;
    }
    uint64_t offset = 0U;
    const int status = shadowspill_memory_pool_reserve_locked(
        pool, charged, alignment, placement, &offset
    );
    if (status != 0) {
        return status;
    }
    const int adopt_status = shadowspill_memory_pool_adopt_lease_locked(
        pool, lease, bytes, alignment, offset
    );
    if (adopt_status != 0) {
        (void)shadowspill_memory_pool_release_locked(pool, offset, charged);
    }
    return adopt_status;
}

static int adopt_lease_locked(
    ShadowSpillMemoryPool *pool,
    ShadowSpillMemoryLease *lease,
    uint64_t bytes,
    uint64_t alignment,
    uint64_t offset,
    uint8_t owns_pool_range
) {
    const uint64_t charged = bytes == 0U ? 1U : bytes;
    if (pool == NULL || lease == NULL || alignment == 0U ||
        lease->pool != NULL ||
        lease->pool_next != NULL || lease->pool_previous_link != NULL ||
        offset > pool->ranges.capacity ||
        charged > pool->ranges.capacity - offset) {
        return -1;
    }
    lease->pool = pool;
    lease->state = SHADOWSPILL_LEASE_IN_USE;
    lease->requested_bytes = bytes;
    lease->charged_bytes = charged;
    lease->alignment_bytes = alignment;
    lease->offset = offset;
    lease->request_sequence = pool->next_request_sequence++;
    lease->pointer = shadowspill_memory_pool_pointer(pool, offset);
    lease->retired_pointer = NULL;
    lease->causal_predecessor = NULL;
    lease->causal_successor = NULL;
    lease->causal_predecessor_generation = 0U;
    lease->causal_event = NULL;
    lease->causal_dependency_expected = 0U;
    lease->owns_pool_range = owns_pool_range;
    lease->pool_next = pool->range_leases;
    lease->pool_previous_link = &pool->range_leases;
    if (lease->pool_next != NULL) {
        lease->pool_next->pool_previous_link = &lease->pool_next;
    }
    pool->range_leases = lease;
    return 0;
}

int shadowspill_memory_pool_adopt_lease_locked(
    ShadowSpillMemoryPool *pool,
    ShadowSpillMemoryLease *lease,
    uint64_t bytes,
    uint64_t alignment,
    uint64_t offset
) {
    return adopt_lease_locked(pool, lease, bytes, alignment, offset, 1U);
}

int shadowspill_memory_pool_adopt_borrowed_lease_locked(
    ShadowSpillMemoryPool *pool,
    ShadowSpillMemoryLease *lease,
    uint64_t bytes,
    uint64_t alignment,
    uint64_t offset
) {
    return adopt_lease_locked(pool, lease, bytes, alignment, offset, 0U);
}

int shadowspill_memory_pool_mark_reserved_locked(
    ShadowSpillMemoryLease *lease
) {
    if (lease == NULL || lease->pool == NULL ||
        lease->state != SHADOWSPILL_LEASE_IN_USE) {
        return -1;
    }
    lease->state = SHADOWSPILL_LEASE_RESERVED;
    lease->pool->reserved_bytes += lease->charged_bytes;
    return 0;
}

int shadowspill_memory_pool_begin_retirement_locked(
    ShadowSpillMemoryLease *lease,
    ShadowSpillEventLease *dependency_event,
    int dependency_expected
) {
    if (lease == NULL || lease->pool == NULL ||
        lease->state != SHADOWSPILL_LEASE_IN_USE ||
        lease->causal_successor != NULL ||
        (dependency_event != NULL && dependency_expected != 0)) {
        return -1;
    }
    lease->state = SHADOWSPILL_LEASE_RETIRE_PENDING;
    lease->release_sequence = lease->pool->next_release_sequence++;
    lease->causal_event = dependency_event;
    lease->causal_dependency_expected = dependency_expected != 0 ? 1U : 0U;
    return 0;
}

int shadowspill_memory_pool_publish_retirement_dependency_locked(
    ShadowSpillMemoryLease *lease,
    ShadowSpillEventLease *dependency_event
) {
    if (lease == NULL || dependency_event == NULL || lease->pool == NULL ||
        lease->state != SHADOWSPILL_LEASE_RETIRE_PENDING ||
        lease->causal_event != NULL ||
        lease->causal_dependency_expected == 0U) {
        return -1;
    }
    lease->causal_event = dependency_event;
    lease->causal_dependency_expected = 0U;
    return 0;
}

int shadowspill_memory_pool_cancel_retirement_locked(
    ShadowSpillMemoryLease *lease
) {
    if (lease == NULL || lease->pool == NULL ||
        lease->state != SHADOWSPILL_LEASE_RETIRE_PENDING ||
        lease->causal_successor != NULL) {
        return -1;
    }
    lease->state = SHADOWSPILL_LEASE_IN_USE;
    lease->causal_event = NULL;
    lease->causal_dependency_expected = 0U;
    return 0;
}

int shadowspill_memory_pool_acquire_reserved_lease_locked(
    ShadowSpillMemoryLease *lease,
    ShadowSpillEventLease **dependency_event
) {
    if (dependency_event == NULL) {
        return -1;
    }
    *dependency_event = NULL;
    if (lease == NULL || lease->pool == NULL) {
        return -1;
    }
    if (lease->state == SHADOWSPILL_LEASE_SUCCESSOR_RESERVED) {
        return shadowspill_pool_handoff_causal_range_locked(
            lease, SHADOWSPILL_LEASE_IN_USE, dependency_event
        );
    }
    if (lease->state != SHADOWSPILL_LEASE_RESERVED ||
        lease->pool->reserved_bytes < lease->charged_bytes) {
        return -1;
    }
    lease->pool->reserved_bytes -= lease->charged_bytes;
    lease->state = SHADOWSPILL_LEASE_IN_USE;
    return 0;
}

int shadowspill_memory_pool_cancel_reservation_locked(
    ShadowSpillMemoryLease *lease
) {
    if (lease == NULL || lease->pool == NULL) {
        return -1;
    }
    if (lease->state != SHADOWSPILL_LEASE_SUCCESSOR_RESERVED) {
        return shadowspill_memory_pool_release_lease_locked(lease);
    }
    ShadowSpillMemoryPool *pool = lease->pool;
    ShadowSpillMemoryLease *predecessor = lease->causal_predecessor;
    if (predecessor == NULL || predecessor->causal_successor != lease ||
        predecessor->generation != lease->causal_predecessor_generation ||
        pool->reserved_bytes < lease->charged_bytes) {
        return -1;
    }
    predecessor->causal_successor = NULL;
    lease->causal_predecessor = NULL;
    lease->causal_predecessor_generation = 0U;
    pool->reserved_bytes -= lease->charged_bytes;
    lease->pool = NULL;
    lease->state = SHADOWSPILL_LEASE_FREE;
    lease->requested_bytes = 0U;
    lease->charged_bytes = 0U;
    lease->alignment_bytes = 0U;
    lease->offset = 0U;
    lease->pointer = NULL;
    lease->causal_event = NULL;
    lease->causal_dependency_expected = 0U;
    lease->owns_pool_range = 0U;
    return 0;
}

int shadowspill_memory_pool_release_lease_locked(
    ShadowSpillMemoryLease *lease
) {
    if (lease != NULL &&
        lease->state == SHADOWSPILL_LEASE_PREDECESSOR_TRANSFERRED &&
        lease->pool == NULL) {
        lease->state = SHADOWSPILL_LEASE_FREE;
        lease->pointer = NULL;
        lease->requested_bytes = 0U;
        lease->charged_bytes = 0U;
        lease->alignment_bytes = 0U;
        lease->offset = 0U;
        lease->causal_event = NULL;
        lease->causal_dependency_expected = 0U;
        lease->owns_pool_range = 0U;
        return 0;
    }
    if (lease == NULL || lease->pool == NULL ||
        lease->state == SHADOWSPILL_LEASE_FREE ||
        lease->pool_previous_link == NULL ||
        *lease->pool_previous_link != lease) {
        return -1;
    }
    ShadowSpillMemoryPool *pool = lease->pool;
    if (lease->causal_successor != NULL) {
        ShadowSpillMemoryLease *successor = lease->causal_successor;
        const int handoff_status = shadowspill_pool_handoff_causal_range_locked(
            successor,
            SHADOWSPILL_LEASE_RESERVED,
            NULL
        );
        if (handoff_status != 0) {
            return handoff_status;
        }
        lease->state = SHADOWSPILL_LEASE_FREE;
        lease->pointer = NULL;
        lease->requested_bytes = 0U;
        lease->charged_bytes = 0U;
        lease->alignment_bytes = 0U;
        lease->offset = 0U;
        lease->causal_event = NULL;
        lease->causal_dependency_expected = 0U;
        lease->owns_pool_range = 0U;
        return 0;
    }
    if (lease->state == SHADOWSPILL_LEASE_SUCCESSOR_RESERVED ||
        lease->state == SHADOWSPILL_LEASE_PREDECESSOR_TRANSFERRED) {
        return -1;
    }
    if (lease->state == SHADOWSPILL_LEASE_RESERVED) {
        if (pool->reserved_bytes < lease->charged_bytes) {
            return -1;
        }
        pool->reserved_bytes -= lease->charged_bytes;
    }
    lease->release_sequence = pool->next_release_sequence++;
    const int status = lease->owns_pool_range
        ? shadowspill_memory_pool_release_locked(
              pool, lease->offset, lease->charged_bytes
          )
        : 0;
    if (status == 0) {
        lease->retired_pointer = lease->pointer;
        *lease->pool_previous_link = lease->pool_next;
        if (lease->pool_next != NULL) {
            lease->pool_next->pool_previous_link = lease->pool_previous_link;
        }
        lease->pool_next = NULL;
        lease->pool_previous_link = NULL;
        lease->pool = NULL;
        lease->state = SHADOWSPILL_LEASE_FREE;
        lease->requested_bytes = 0U;
        lease->charged_bytes = 0U;
        lease->alignment_bytes = 0U;
        lease->offset = 0U;
        lease->pointer = NULL;
        lease->causal_event = NULL;
        lease->causal_dependency_expected = 0U;
        lease->owns_pool_range = 0U;
    }
    return status;
}
