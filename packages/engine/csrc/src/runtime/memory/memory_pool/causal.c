/* Promising a successor the memory a predecessor has yet to release. */
#include "internal.h"

static int lease_is_causally_releasable(
    const ShadowSpillMemoryLease *lease
) {
    return lease->state == SHADOWSPILL_LEASE_RETIRE_PENDING;
}

static int lease_can_publish_causal_dependency(
    const ShadowSpillMemoryLease *lease
) {
    if (!lease_is_causally_releasable(lease)) {
        return 0;
    }
    /*
     * MemoryPool does not know why the owner is releasing this range.  It
     * only needs proof that one dependency either exists now or is guaranteed
     * to be published by the owner.  Multi-event ordinary retirements leave
     * both fields clear and therefore cannot be represented by one causal
     * successor.
     */
    return lease->causal_event != NULL ||
        lease->causal_dependency_expected != 0U;
}

static int causal_candidate_precedes(
    const ShadowSpillMemoryLease *candidate,
    const ShadowSpillMemoryLease *selected
) {
    if (selected == NULL) {
        return 1;
    }
    const int candidate_dependency_published =
        candidate->causal_event != NULL;
    const int selected_dependency_published = selected->causal_event != NULL;
    if (candidate_dependency_published != selected_dependency_published) {
        return candidate_dependency_published > selected_dependency_published;
    }
    if (candidate->charged_bytes != selected->charged_bytes) {
        return candidate->charged_bytes < selected->charged_bytes;
    }
    if (candidate->release_sequence != selected->release_sequence) {
        return candidate->release_sequence < selected->release_sequence;
    }
    return candidate->offset < selected->offset;
}

static int release_sequence_compare(
    const void *left_value,
    const void *right_value
) {
    const ShadowSpillMemoryLease *left =
        *(ShadowSpillMemoryLease *const *)left_value;
    const ShadowSpillMemoryLease *right =
        *(ShadowSpillMemoryLease *const *)right_value;
    if (left->release_sequence != right->release_sequence) {
        return left->release_sequence < right->release_sequence ? -1 : 1;
    }
    if (left->offset != right->offset) {
        return left->offset < right->offset ? -1 : 1;
    }
    return 0;
}

int shadowspill_memory_pool_reserve_causal_successor_locked(
    ShadowSpillMemoryPool *pool,
    ShadowSpillMemoryLease *successor,
    uint64_t bytes,
    uint64_t alignment
) {
    if (pool == NULL || successor == NULL || successor->pool != NULL ||
        successor->state != SHADOWSPILL_LEASE_FREE || bytes == 0U) {
        return -1;
    }
    if (alignment < pool->minimum_alignment) {
        alignment = pool->minimum_alignment;
    }
    ShadowSpillMemoryLease *selected = NULL;
    for (ShadowSpillMemoryLease *candidate = pool->range_leases;
         candidate != NULL; candidate = candidate->pool_next) {
        if (!candidate->owns_pool_range ||
            !lease_can_publish_causal_dependency(candidate) ||
            candidate->causal_successor != NULL ||
            candidate->charged_bytes < bytes ||
            candidate->offset % alignment != 0U) {
            continue;
        }
        if (causal_candidate_precedes(candidate, selected)) {
            selected = candidate;
        }
    }
    if (selected == NULL) {
        return 1;
    }

    /*
     * The successor claims the predecessor's complete charged extent. This
     * keeps every byte unavailable until the predecessor dependency is
     * satisfied; a smaller logical request merely carries internal slack.
     * Splitting a still-live predecessor would make the unclaimed fragment
     * reusable too early.
     */
    successor->pool = pool;
    successor->state = SHADOWSPILL_LEASE_SUCCESSOR_RESERVED;
    successor->requested_bytes = bytes;
    successor->charged_bytes = selected->charged_bytes;
    successor->alignment_bytes = alignment;
    successor->offset = selected->offset;
    successor->request_sequence = pool->next_request_sequence++;
    successor->pointer = selected->pointer;
    successor->retired_pointer = NULL;
    successor->causal_predecessor = selected;
    successor->causal_predecessor_generation = selected->generation;
    selected->causal_successor = successor;
    successor->causal_dependency_expected = 0U;
    successor->owns_pool_range = 1U;
    pool->reserved_bytes += successor->charged_bytes;
    return 0;
}

int shadowspill_memory_pool_can_reserve_after_releases_locked(
    const ShadowSpillMemoryPool *pool,
    uint64_t bytes,
    uint64_t alignment
) {
    if (pool == NULL || !pool->initialized || bytes == 0U) {
        return -1;
    }
    uint64_t candidate_count = 0U;
    for (const ShadowSpillMemoryLease *lease = pool->range_leases;
         lease != NULL; lease = lease->pool_next) {
        if (lease->owns_pool_range &&
            lease_can_publish_causal_dependency(lease) &&
            lease->causal_successor == NULL) {
            ++candidate_count;
        }
    }
    if (candidate_count == 0U) {
        return 0;
    }
    if (pool->release_frontier_workspace == NULL ||
        pool->release_range_workspace == NULL ||
        candidate_count > pool->release_frontier_capacity) {
        return -1;
    }
    uint64_t frontier_count = 0U;
    return shadowspill_memory_pool_find_release_frontier_locked(
        pool,
        bytes,
        alignment,
        pool->release_frontier_workspace,
        pool->release_frontier_capacity,
        &frontier_count
    );
}

int shadowspill_memory_pool_find_release_frontier_locked(
    const ShadowSpillMemoryPool *pool,
    uint64_t bytes,
    uint64_t alignment,
    ShadowSpillMemoryLease **frontier,
    uint64_t frontier_capacity,
    uint64_t *frontier_count
) {
    if (pool == NULL || !pool->initialized || bytes == 0U) {
        return -1;
    }
    if (frontier_count == NULL ||
        (frontier_capacity != 0U && frontier == NULL)) {
        return -1;
    }
    *frontier_count = 0U;
    if (alignment < pool->minimum_alignment) {
        alignment = pool->minimum_alignment;
    }
    if (pool->release_range_workspace == NULL ||
        pool->release_range_capacity == 0U) {
        return -1;
    }
    ShadowSpillRangeAllocator future = {0};
    if (shadowspill_range_clone_extended_with_nodes(
            &pool->ranges,
            pool->ranges.capacity,
            &future,
            pool->release_range_workspace,
            pool->release_range_capacity
        ) != 0) {
        return -1;
    }
    uint64_t candidate_count = 0U;
    for (ShadowSpillMemoryLease *lease = pool->range_leases;
         lease != NULL; lease = lease->pool_next) {
        if (!lease->owns_pool_range ||
            !lease_can_publish_causal_dependency(lease) ||
            lease->causal_successor != NULL) {
            continue;
        }
        if (candidate_count >= frontier_capacity) {
            shadowspill_range_destroy(&future);
            return -1;
        }
        frontier[candidate_count++] = lease;
    }
    if (candidate_count > 1U) {
        qsort(
            frontier,
            (size_t)candidate_count,
            sizeof(*frontier),
            release_sequence_compare
        );
    }
    int reserve_status = 1;
    for (uint64_t index = 0U; index < candidate_count; ++index) {
        ShadowSpillMemoryLease *lease = frontier[index];
        if (shadowspill_range_free(
                &future, lease->offset, lease->charged_bytes
            ) != 0) {
            shadowspill_range_destroy(&future);
            return -1;
        }
        uint64_t ignored_offset = 0U;
        reserve_status = shadowspill_range_allocate_best_fit_low(
            &future, bytes, alignment, &ignored_offset
        );
        if (reserve_status <= 0) {
            *frontier_count = index + 1U;
            break;
        }
    }
    shadowspill_range_destroy(&future);
    return reserve_status == 0 ? 1 : reserve_status > 0 ? 0 : -1;
}

int shadowspill_pool_handoff_causal_range_locked(
    ShadowSpillMemoryLease *successor,
    ShadowSpillMemoryLeaseState successor_state,
    ShadowSpillEventLease **dependency_event
) {
    ShadowSpillMemoryLease *predecessor = successor == NULL
        ? NULL
        : successor->causal_predecessor;
    ShadowSpillMemoryPool *pool = successor == NULL ? NULL : successor->pool;
    if (predecessor == NULL || pool == NULL ||
        successor->state != SHADOWSPILL_LEASE_SUCCESSOR_RESERVED ||
        predecessor->pool != pool ||
        predecessor->generation != successor->causal_predecessor_generation ||
        predecessor->causal_successor != successor ||
        predecessor->pool_previous_link == NULL ||
        *predecessor->pool_previous_link != predecessor ||
        successor->pool_previous_link != NULL || successor->pool_next != NULL) {
        return -1;
    }
    if (successor_state == SHADOWSPILL_LEASE_IN_USE) {
        if (predecessor->causal_event == NULL || dependency_event == NULL) {
            return 1;
        }
        if (pool->reserved_bytes < successor->charged_bytes) {
            return -1;
        }
        shadowspill_event_lease_retain(predecessor->causal_event);
        *dependency_event = predecessor->causal_event;
    }

    successor->pool_previous_link = predecessor->pool_previous_link;
    successor->pool_next = predecessor->pool_next;
    *successor->pool_previous_link = successor;
    if (successor->pool_next != NULL) {
        successor->pool_next->pool_previous_link = &successor->pool_next;
    }
    predecessor->pool_previous_link = NULL;
    predecessor->pool_next = NULL;
    predecessor->pool = NULL;
    predecessor->state = SHADOWSPILL_LEASE_PREDECESSOR_TRANSFERRED;
    predecessor->owns_pool_range = 0U;
    predecessor->retired_pointer = predecessor->pointer;
    predecessor->causal_successor = NULL;
    predecessor->causal_event = NULL;
    predecessor->causal_dependency_expected = 0U;
    successor->causal_predecessor = NULL;
    successor->causal_predecessor_generation = 0U;
    successor->state = successor_state;
    if (successor_state == SHADOWSPILL_LEASE_IN_USE) {
        pool->reserved_bytes -= successor->charged_bytes;
    }
    (void)atomic_fetch_add_explicit(
        &pool->capacity_epoch, 1U, memory_order_release
    );
    return 0;
}
