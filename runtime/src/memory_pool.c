#include "internal.h"

#include <stddef.h>
#include <stdint.h>

ShadowSpillMemoryPool *shadowspill_runtime_pool(
    ShadowSpillRuntime *runtime,
    uint32_t pool_id
) {
    if (runtime == NULL || runtime->pools == NULL ||
        pool_id >= runtime->pool_count) {
        return NULL;
    }
    return &runtime->pools[pool_id];
}

const ShadowSpillMemoryPool *shadowspill_runtime_pool_const(
    const ShadowSpillRuntime *runtime,
    uint32_t pool_id
) {
    if (runtime == NULL || runtime->pools == NULL ||
        pool_id >= runtime->pool_count) {
        return NULL;
    }
    return &runtime->pools[pool_id];
}

ShadowSpillMemoryPool *shadowspill_execution_pool(ShadowSpillRuntime *runtime) {
    return runtime == NULL
        ? NULL
        : shadowspill_runtime_pool(runtime, runtime->execution_pool_id);
}

const ShadowSpillMemoryPool *shadowspill_execution_pool_const(
    const ShadowSpillRuntime *runtime
) {
    return runtime == NULL
        ? NULL
        : shadowspill_runtime_pool_const(runtime, runtime->execution_pool_id);
}

ShadowSpillMemoryPool *shadowspill_spill_pool(ShadowSpillRuntime *runtime) {
    return runtime == NULL
        ? NULL
        : shadowspill_runtime_pool(runtime, runtime->spill_pool_id);
}

ShadowSpillObjectLocation *shadowspill_object_location(
    ShadowSpillObject *object,
    uint32_t pool_id
) {
    if (object == NULL || object->locations == NULL ||
        pool_id >= object->location_count) {
        return NULL;
    }
    return &object->locations[pool_id];
}

ShadowSpillObjectLocation *shadowspill_execution_location(
    ShadowSpillRuntime *runtime,
    ShadowSpillObject *object
) {
    return runtime == NULL
        ? NULL
        : shadowspill_object_location(object, runtime->execution_pool_id);
}

ShadowSpillObjectLocation *shadowspill_spill_location(
    ShadowSpillRuntime *runtime,
    ShadowSpillObject *object
) {
    return runtime == NULL
        ? NULL
        : shadowspill_object_location(object, runtime->spill_pool_id);
}

static void cpu_relax(void) {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    __asm__ volatile("yield");
#else
    atomic_signal_fence(memory_order_seq_cst);
#endif
}

int shadowspill_memory_pool_initialize(
    ShadowSpillMemoryPool *pool,
    uint32_t pool_id,
    const ShadowSpillMemoryPoolBackend *backend,
    uint64_t capacity,
    uint64_t minimum_alignment
) {
    if (pool == NULL || minimum_alignment == 0U ||
        !shadowspill_memory_pool_backend_is_valid(backend)) {
        return -1;
    }
    void *base = NULL;
    if (capacity != 0U && backend->allocate_arena(
            backend->context, capacity, &base
        ) != 0) {
        return -1;
    }
    if (pthread_mutex_init(&pool->lock, NULL) != 0) {
        if (base != NULL) {
            (void)backend->free_arena(backend->context, base);
        }
        return -1;
    }
    if (pthread_cond_init(&pool->capacity_changed, NULL) != 0) {
        pthread_mutex_destroy(&pool->lock);
        if (base != NULL) {
            (void)backend->free_arena(backend->context, base);
        }
        return -1;
    }
    if (shadowspill_range_initialize(&pool->ranges, capacity) != 0) {
        pthread_cond_destroy(&pool->capacity_changed);
        pthread_mutex_destroy(&pool->lock);
        if (base != NULL) {
            (void)backend->free_arena(backend->context, base);
        }
        return -1;
    }
    pool->backend = *backend;
    pool->base = base;
    pool->pool_id = pool_id;
    pool->minimum_alignment = minimum_alignment;
    atomic_init(&pool->foreground_waiters, 0U);
    atomic_init(&pool->transfer_waiters, 0U);
    pool->initialized = 1U;
    return 0;
}

void shadowspill_memory_pool_destroy(ShadowSpillMemoryPool *pool) {
    if (pool == NULL || !pool->initialized) {
        return;
    }
    shadowspill_range_destroy(&pool->ranges);
    if (pool->base != NULL) {
        (void)pool->backend.free_arena(pool->backend.context, pool->base);
    }
    pthread_cond_destroy(&pool->capacity_changed);
    pthread_mutex_destroy(&pool->lock);
    *pool = (ShadowSpillMemoryPool){0};
}

void shadowspill_memory_pool_lock_foreground(ShadowSpillMemoryPool *pool) {
    (void)atomic_fetch_add_explicit(
        &pool->foreground_waiters, 1U, memory_order_relaxed
    );
    while (atomic_load_explicit(
               &pool->transfer_waiters, memory_order_acquire
           ) != 0U || pthread_mutex_trylock(&pool->lock) != 0) {
        cpu_relax();
    }
    (void)atomic_fetch_sub_explicit(
        &pool->foreground_waiters, 1U, memory_order_relaxed
    );
}

void shadowspill_memory_pool_unlock_foreground(ShadowSpillMemoryPool *pool) {
    pthread_mutex_unlock(&pool->lock);
}

void shadowspill_memory_pool_declare_transfer(ShadowSpillMemoryPool *pool) {
    (void)atomic_fetch_add_explicit(
        &pool->transfer_waiters, 1U, memory_order_release
    );
}

void shadowspill_memory_pool_relinquish_transfer(ShadowSpillMemoryPool *pool) {
    (void)atomic_fetch_sub_explicit(
        &pool->transfer_waiters, 1U, memory_order_release
    );
}

int shadowspill_memory_pool_try_lock_transfer(ShadowSpillMemoryPool *pool) {
    return pthread_mutex_trylock(&pool->lock) == 0;
}

void shadowspill_memory_pool_unlock_transfer(ShadowSpillMemoryPool *pool) {
    pthread_mutex_unlock(&pool->lock);
}

int shadowspill_memory_pool_try_lock_reclamation(ShadowSpillMemoryPool *pool) {
    const int transfer_waiting = atomic_load_explicit(
        &pool->transfer_waiters, memory_order_acquire
    ) != 0U;
    /*
     * A destination reservation may depend on a completed retirement from
     * the same causal prefix. Let that reclamation satisfy the reservation
     * before either yields to foreground allocation. Without this ordering,
     * each worker operation can wait for the other indefinitely.
     */
    if ((!transfer_waiting && atomic_load_explicit(
             &pool->foreground_waiters, memory_order_relaxed
         ) != 0U) || pthread_mutex_trylock(&pool->lock) != 0) {
        return 0;
    }
    if (!transfer_waiting && atomic_load_explicit(
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
        pthread_cond_broadcast(&pool->capacity_changed);
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
    uint64_t offset = 0U;
    const int status = shadowspill_memory_pool_reserve_locked(
        pool, charged, alignment, placement, &offset
    );
    if (status != 0) {
        return status;
    }
    const int adopt_status = shadowspill_memory_pool_adopt_lease_locked(
        pool, lease, bytes, offset
    );
    if (adopt_status != 0) {
        (void)shadowspill_memory_pool_release_locked(pool, offset, charged);
    }
    return adopt_status;
}

int shadowspill_memory_pool_adopt_lease_locked(
    ShadowSpillMemoryPool *pool,
    ShadowSpillMemoryLease *lease,
    uint64_t bytes,
    uint64_t offset
) {
    const uint64_t charged = bytes == 0U ? 1U : bytes;
    if (pool == NULL || lease == NULL || lease->pool != NULL ||
        lease->pool_next != NULL || lease->pool_previous_link != NULL ||
        offset > pool->ranges.capacity ||
        charged > pool->ranges.capacity - offset) {
        return -1;
    }
    lease->pool = pool;
    lease->state = SHADOWSPILL_LEASE_RESERVED;
    lease->requested_bytes = bytes;
    lease->charged_bytes = charged;
    lease->offset = offset;
    lease->pointer = shadowspill_memory_pool_pointer(pool, offset);
    lease->retired_pointer = NULL;
    lease->pool_next = pool->leases;
    lease->pool_previous_link = &pool->leases;
    if (lease->pool_next != NULL) {
        lease->pool_next->pool_previous_link = &lease->pool_next;
    }
    pool->leases = lease;
    return 0;
}

int shadowspill_memory_pool_release_lease_locked(
    ShadowSpillMemoryLease *lease
) {
    if (lease == NULL || lease->pool == NULL ||
        lease->state == SHADOWSPILL_LEASE_FREE ||
        lease->pool_previous_link == NULL ||
        *lease->pool_previous_link != lease) {
        return -1;
    }
    ShadowSpillMemoryPool *pool = lease->pool;
    const int status = shadowspill_memory_pool_release_locked(
        pool, lease->offset, lease->charged_bytes
    );
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
        lease->offset = 0U;
        lease->pointer = NULL;
    }
    return status;
}

void shadowspill_memory_pool_rebase_locked(
    ShadowSpillMemoryPool *pool,
    void *new_base
) {
    if (pool == NULL) {
        return;
    }
    pool->base = new_base;
    for (ShadowSpillMemoryLease *lease = pool->leases; lease != NULL;
         lease = lease->pool_next) {
        lease->pointer = shadowspill_memory_pool_pointer(pool, lease->offset);
    }
}

uint64_t shadowspill_memory_pool_free_bytes_locked(
    const ShadowSpillMemoryPool *pool
) {
    return shadowspill_range_free_bytes(&pool->ranges);
}

uint64_t shadowspill_memory_pool_largest_free_locked(
    const ShadowSpillMemoryPool *pool
) {
    return shadowspill_range_largest_free(&pool->ranges);
}

void *shadowspill_memory_pool_pointer(
    const ShadowSpillMemoryPool *pool,
    uint64_t offset
) {
    if (pool == NULL || pool->base == NULL || offset > pool->ranges.capacity) {
        return NULL;
    }
    return (void *)((unsigned char *)pool->base + offset);
}
