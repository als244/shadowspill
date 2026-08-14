#include "internal.h"

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>

/*
 * This canary links the generic MemoryPool implementation directly. These
 * two dependencies are intentionally tiny because the test does not create a
 * backend arena or submit an event; it exercises only pool-owned transitions.
 */
int shadowspill_memory_pool_backend_is_valid(
    const ShadowSpillMemoryPoolBackend *backend
) {
    (void)backend;
    return 1;
}

void shadowspill_event_lease_retain(ShadowSpillEventLease *lease) {
    if (lease != NULL) {
        (void)atomic_fetch_add_explicit(
            &lease->references, 1U, memory_order_relaxed
        );
    }
}

static int initialize_pool(ShadowSpillMemoryPool *pool) {
    *pool = (ShadowSpillMemoryPool){
        .minimum_alignment = 1U,
        .initialized = 1U,
        .next_request_sequence = 1U,
        .next_release_sequence = 1U,
    };
    return pthread_mutex_init(&pool->lock, NULL) != 0 ||
        pthread_cond_init(&pool->capacity_changed, NULL) != 0 ||
        shadowspill_range_initialize(&pool->ranges, 128U) != 0;
}

static void destroy_pool(ShadowSpillMemoryPool *pool) {
    shadowspill_range_destroy(&pool->ranges);
    pthread_cond_destroy(&pool->capacity_changed);
    pthread_mutex_destroy(&pool->lock);
}

static void initialize_event(ShadowSpillEventLease *event) {
    *event = (ShadowSpillEventLease){.generation = 17U};
    atomic_init(&event->references, 1U);
    atomic_init(&event->backend_complete, 0U);
}

static int reserve_predecessor(
    ShadowSpillMemoryPool *pool,
    ShadowSpillMemoryLease *predecessor,
    ShadowSpillEventLease *event
) {
    *predecessor = (ShadowSpillMemoryLease){.generation = 11U};
    if (shadowspill_memory_pool_reserve_lease_locked(
            pool,
            predecessor,
            96U,
            1U,
            SHADOWSPILL_MEMORY_BEST_FIT_LOW
        ) != 0) {
        return -1;
    }
    return shadowspill_memory_pool_begin_retirement_locked(
        predecessor, event, 0
    );
}

static int completion_without_successor_frees_and_coalesces(void) {
    ShadowSpillMemoryPool pool = {0};
    ShadowSpillMemoryLease predecessor = {0};
    ShadowSpillEventLease event = {0};
    if (initialize_pool(&pool) != 0) {
        return -1;
    }
    initialize_event(&event);
    int failed = reserve_predecessor(&pool, &predecessor, &event) != 0;
    atomic_store_explicit(&event.backend_complete, 1U, memory_order_release);
    failed = failed || shadowspill_memory_pool_free_bytes_locked(&pool) != 32U;
    failed = failed || shadowspill_memory_pool_release_lease_locked(
            &predecessor
        ) != 0;
    failed = failed || predecessor.state != SHADOWSPILL_LEASE_FREE ||
        pool.leases != NULL ||
        shadowspill_memory_pool_free_bytes_locked(&pool) != 128U ||
        shadowspill_memory_pool_largest_free_locked(&pool) != 128U;
    destroy_pool(&pool);
    return failed ? -1 : 0;
}

static int completion_hands_range_directly_to_reserved_successor(void) {
    ShadowSpillMemoryPool pool = {0};
    ShadowSpillMemoryLease predecessor = {0};
    ShadowSpillMemoryLease successor = {0};
    ShadowSpillEventLease event = {0};
    if (initialize_pool(&pool) != 0) {
        return -1;
    }
    initialize_event(&event);
    int failed = reserve_predecessor(&pool, &predecessor, &event) != 0 ||
        shadowspill_memory_pool_reserve_causal_successor_locked(
            &pool, &successor, 64U, 1U
        ) != 0;

    /* Backend completion alone never publishes capacity to the free list. */
    atomic_store_explicit(&event.backend_complete, 1U, memory_order_release);
    failed = failed || predecessor.state != SHADOWSPILL_LEASE_RETIRE_PENDING ||
        successor.state != SHADOWSPILL_LEASE_SUCCESSOR_RESERVED ||
        shadowspill_memory_pool_free_bytes_locked(&pool) != 32U ||
        pool.ranges.allocated != 96U;

    /* The pool commit hands off ownership without an observable FREE state. */
    failed = failed || shadowspill_memory_pool_release_lease_locked(
            &predecessor
        ) != 0;
    failed = failed || predecessor.state != SHADOWSPILL_LEASE_FREE ||
        successor.state != SHADOWSPILL_LEASE_RESERVED ||
        successor.pool != &pool || pool.leases != &successor ||
        successor.offset != 0U || successor.charged_bytes != 96U ||
        pool.reserved_bytes != 96U ||
        shadowspill_memory_pool_free_bytes_locked(&pool) != 32U ||
        pool.ranges.allocated != 96U;

    ShadowSpillEventLease *dependency = NULL;
    failed = failed || shadowspill_memory_pool_acquire_reserved_lease_locked(
            &successor, &dependency
        ) != 0;
    failed = failed || dependency != NULL ||
        successor.state != SHADOWSPILL_LEASE_IN_USE ||
        pool.reserved_bytes != 0U;
    failed = failed || shadowspill_memory_pool_release_lease_locked(
            &successor
        ) != 0;
    failed = failed || shadowspill_memory_pool_free_bytes_locked(&pool) != 128U;
    destroy_pool(&pool);
    return failed ? -1 : 0;
}

static int acquisition_hands_range_to_successor_with_dependency(void) {
    ShadowSpillMemoryPool pool = {0};
    ShadowSpillMemoryLease predecessor = {0};
    ShadowSpillMemoryLease successor = {0};
    ShadowSpillEventLease event = {0};
    ShadowSpillEventLease *dependency = NULL;
    if (initialize_pool(&pool) != 0) {
        return -1;
    }
    initialize_event(&event);
    int failed = reserve_predecessor(&pool, &predecessor, &event) != 0 ||
        shadowspill_memory_pool_reserve_causal_successor_locked(
            &pool, &successor, 64U, 1U
        ) != 0 ||
        shadowspill_memory_pool_acquire_reserved_lease_locked(
            &successor, &dependency
        ) != 0;
    failed = failed || dependency != &event ||
        atomic_load_explicit(&event.references, memory_order_acquire) != 2U ||
        predecessor.state != SHADOWSPILL_LEASE_PREDECESSOR_TRANSFERRED ||
        predecessor.pool != NULL ||
        successor.state != SHADOWSPILL_LEASE_IN_USE ||
        successor.pool != &pool || pool.leases != &successor ||
        successor.offset != 0U || successor.charged_bytes != 96U ||
        pool.reserved_bytes != 0U ||
        shadowspill_memory_pool_free_bytes_locked(&pool) != 32U;

    /* Later predecessor completion retires metadata, not the reused range. */
    atomic_store_explicit(&event.backend_complete, 1U, memory_order_release);
    failed = failed || shadowspill_memory_pool_release_lease_locked(
            &predecessor
        ) != 0;
    failed = failed || predecessor.state != SHADOWSPILL_LEASE_FREE ||
        shadowspill_memory_pool_free_bytes_locked(&pool) != 32U ||
        pool.leases != &successor;

    failed = failed || shadowspill_memory_pool_release_lease_locked(
            &successor
        ) != 0;
    failed = failed || shadowspill_memory_pool_free_bytes_locked(&pool) != 128U;
    (void)atomic_fetch_sub_explicit(
        &dependency->references, 1U, memory_order_release
    );
    destroy_pool(&pool);
    return failed ? -1 : 0;
}

static int eventless_generic_retirement_is_not_a_causal_candidate(void) {
    ShadowSpillMemoryPool pool = {0};
    ShadowSpillMemoryLease predecessor = {0};
    ShadowSpillMemoryLease successor = {0};
    if (initialize_pool(&pool) != 0) {
        return -1;
    }
    int failed = shadowspill_memory_pool_reserve_lease_locked(
            &pool,
            &predecessor,
            96U,
            1U,
            SHADOWSPILL_MEMORY_BEST_FIT_LOW
        ) != 0;
    failed = failed || shadowspill_memory_pool_begin_retirement_locked(
            &predecessor, NULL, 0
        ) != 0;
    const int reserve_status =
        shadowspill_memory_pool_reserve_causal_successor_locked(
            &pool, &successor, 64U, 1U
        );
    failed = failed || reserve_status != 1 ||
        predecessor.causal_successor != NULL || successor.pool != NULL ||
        successor.state != SHADOWSPILL_LEASE_FREE;
    failed = failed || shadowspill_memory_pool_release_lease_locked(
            &predecessor
        ) != 0;
    destroy_pool(&pool);
    return failed ? -1 : 0;
}

static int promised_dependency_is_published_before_acquisition(void) {
    ShadowSpillMemoryPool pool = {0};
    ShadowSpillMemoryLease predecessor = {0};
    ShadowSpillMemoryLease successor = {0};
    ShadowSpillEventLease event = {0};
    ShadowSpillEventLease *dependency = NULL;
    if (initialize_pool(&pool) != 0) {
        return -1;
    }
    int failed = shadowspill_memory_pool_reserve_lease_locked(
            &pool,
            &predecessor,
            96U,
            1U,
            SHADOWSPILL_MEMORY_BEST_FIT_LOW
        ) != 0;
    predecessor.generation = 23U;
    failed = failed || shadowspill_memory_pool_begin_retirement_locked(
            &predecessor, NULL, 1
        ) != 0;
    failed = failed || shadowspill_memory_pool_reserve_causal_successor_locked(
            &pool, &successor, 64U, 1U
        ) != 0;

    /* A future dependency permits reservation, not premature acquisition. */
    failed = failed || shadowspill_memory_pool_acquire_reserved_lease_locked(
            &successor, &dependency
        ) != 1;
    failed = failed || dependency != NULL ||
        successor.state != SHADOWSPILL_LEASE_SUCCESSOR_RESERVED;

    initialize_event(&event);
    failed = failed ||
        shadowspill_memory_pool_publish_retirement_dependency_locked(
            &predecessor, &event
        ) != 0;
    failed = failed || shadowspill_memory_pool_acquire_reserved_lease_locked(
            &successor, &dependency
        ) != 0;
    failed = failed || dependency != &event ||
        successor.state != SHADOWSPILL_LEASE_IN_USE ||
        predecessor.state != SHADOWSPILL_LEASE_PREDECESSOR_TRANSFERRED;
    failed = failed || shadowspill_memory_pool_release_lease_locked(
            &predecessor
        ) != 0;
    failed = failed || shadowspill_memory_pool_release_lease_locked(
            &successor
        ) != 0;
    (void)atomic_fetch_sub_explicit(
        &dependency->references, 1U, memory_order_release
    );
    destroy_pool(&pool);
    return failed ? -1 : 0;
}

int main(void) {
    if (completion_without_successor_frees_and_coalesces() != 0) {
        fprintf(stderr, "completion-to-free transition failed\n");
        return 1;
    }
    if (completion_hands_range_directly_to_reserved_successor() != 0) {
        fprintf(stderr, "completion-to-successor handoff failed\n");
        return 1;
    }
    if (acquisition_hands_range_to_successor_with_dependency() != 0) {
        fprintf(stderr, "acquisition-to-successor handoff failed\n");
        return 1;
    }
    if (eventless_generic_retirement_is_not_a_causal_candidate() != 0) {
        fprintf(stderr, "eventless retirement became a causal candidate\n");
        return 1;
    }
    if (promised_dependency_is_published_before_acquisition() != 0) {
        fprintf(stderr, "promised dependency acquisition failed\n");
        return 1;
    }
    return 0;
}
