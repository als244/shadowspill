/* Which lock covers what, and who may take it. */
#include "internal.h"

void shadowspill_memory_pool_lock_foreground(ShadowSpillMemoryPool *pool) {
    (void)atomic_fetch_add_explicit(
        &pool->foreground_waiters, 1U, memory_order_relaxed
    );
    while (atomic_load_explicit(
               &pool->reservation_waiters, memory_order_acquire
           ) != 0U || pthread_mutex_trylock(&pool->lock) != 0) {
        shadowspill_pool_cpu_relax();
    }
    (void)atomic_fetch_sub_explicit(
        &pool->foreground_waiters, 1U, memory_order_relaxed
    );
}

void shadowspill_memory_pool_unlock_foreground(ShadowSpillMemoryPool *pool) {
    pthread_mutex_unlock(&pool->lock);
}

void shadowspill_memory_pool_declare_reservation(ShadowSpillMemoryPool *pool) {
    (void)atomic_fetch_add_explicit(
        &pool->reservation_waiters, 1U, memory_order_release
    );
}

void shadowspill_memory_pool_relinquish_reservation(
    ShadowSpillMemoryPool *pool
) {
    (void)atomic_fetch_sub_explicit(
        &pool->reservation_waiters, 1U, memory_order_release
    );
}

void shadowspill_memory_pool_lock_reservation(ShadowSpillMemoryPool *pool) {
    shadowspill_memory_pool_declare_reservation(pool);
    while (pthread_mutex_trylock(&pool->lock) != 0) {
        shadowspill_pool_cpu_relax();
    }
}

int shadowspill_memory_pool_try_lock_reservation(ShadowSpillMemoryPool *pool) {
    return pthread_mutex_trylock(&pool->lock) == 0;
}

void shadowspill_memory_pool_unlock_reservation(ShadowSpillMemoryPool *pool) {
    pthread_mutex_unlock(&pool->lock);
}

int shadowspill_memory_pool_try_lock_reclamation(ShadowSpillMemoryPool *pool) {
    const int reservation_waiting = atomic_load_explicit(
        &pool->reservation_waiters, memory_order_acquire
    ) != 0U;
    /*
     * A destination reservation may depend on a completed retirement from
     * the same causal prefix. Let that reclamation satisfy the reservation
     * before either yields to foreground allocation. Without this ordering,
     * each worker operation can wait for the other indefinitely.
     */
    if ((!reservation_waiting && atomic_load_explicit(
             &pool->foreground_waiters, memory_order_relaxed
         ) != 0U) || pthread_mutex_trylock(&pool->lock) != 0) {
        return 0;
    }
    if (!reservation_waiting && atomic_load_explicit(
            &pool->foreground_waiters, memory_order_relaxed
        ) != 0U) {
        pthread_mutex_unlock(&pool->lock);
        return 0;
    }
    return 1;
}

void shadowspill_memory_pool_unlock_reclamation(ShadowSpillMemoryPool *pool) {
    pthread_mutex_unlock(&pool->lock);
}
