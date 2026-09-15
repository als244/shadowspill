/* The mapping a pool is made of, and where a location resolves to. */
#include "internal.h"

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

ShadowSpillObjectLocation *shadowspill_plan_execution_location(
    const ShadowSpillPlan *plan,
    ShadowSpillObject *object
) {
    return plan == NULL || plan->execution_pool == NULL
        ? NULL
        : shadowspill_object_location(object, plan->execution_pool->pool_id);
}

ShadowSpillObjectLocation *shadowspill_plan_spill_location(
    const ShadowSpillPlan *plan,
    ShadowSpillObject *object
) {
    return plan == NULL || plan->spill_pool == NULL
        ? NULL
        : shadowspill_object_location(object, plan->spill_pool->pool_id);
}

void shadowspill_pool_cpu_relax(void) {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    __asm__ volatile("yield");
#else
    atomic_signal_fence(memory_order_seq_cst);
#endif
}

/* A pinned-host arena is an anonymous private mapping, so the pool owns
 * page-aligned memory the C allocator never touches, and the backend pins it
 * in place. Device arenas are the backend's to allocate. */
static int arena_allocate(
    const ShadowSpillBackend *backend, uint8_t kind, uint64_t capacity, void **base
) {
    if (kind == SHADOWSPILL_POOL_DEVICE) {
        return backend->allocate_device(backend->state, capacity, base);
    }
    if (capacity == 0U || capacity > SIZE_MAX) {
        return -1;
    }
    void *host = mmap(
        NULL, (size_t)capacity, PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0
    );
    if (host == MAP_FAILED) {
        return -1;
    }
    if (backend->register_host_memory(backend->state, host, capacity) != 0) {
        (void)munmap(host, (size_t)capacity);
        return -1;
    }
    *base = host;
    return 0;
}

static int arena_release(
    const ShadowSpillBackend *backend, uint8_t kind, void *base, uint64_t capacity
) {
    if (kind == SHADOWSPILL_POOL_DEVICE) {
        return backend->free_device(backend->state, base, capacity);
    }
    const int status = backend->unregister_host_memory(backend->state, base, capacity);
    (void)munmap(base, (size_t)capacity);
    return status;
}

int shadowspill_memory_pool_arena_allocate(
    const ShadowSpillMemoryPool *pool, uint64_t bytes, void **base
) {
    return arena_allocate(pool->backend, pool->kind, bytes, base);
}

int shadowspill_memory_pool_arena_release(
    const ShadowSpillMemoryPool *pool, void *base, uint64_t capacity
) {
    return arena_release(pool->backend, pool->kind, base, capacity);
}

int shadowspill_memory_pool_initialize(
    ShadowSpillMemoryPool *pool,
    uint32_t pool_id,
    const ShadowSpillBackend *backend,
    uint8_t kind,
    uint64_t capacity,
    uint64_t minimum_alignment
) {
    if (pool == NULL || minimum_alignment == 0U || backend == NULL ||
        kind > SHADOWSPILL_POOL_PINNED_HOST) {
        return -1;
    }
    void *base = NULL;
    if (capacity != 0U && arena_allocate(backend, kind, capacity, &base) != 0) {
        return -1;
    }
    if (pthread_mutex_init(&pool->lock, NULL) != 0) {
        if (base != NULL) {
            (void)arena_release(backend, kind, base, capacity);
        }
        return -1;
    }
    if (shadowspill_range_initialize(&pool->ranges, capacity) != 0) {
        pthread_mutex_destroy(&pool->lock);
        if (base != NULL) {
            (void)arena_release(backend, kind, base, capacity);
        }
        return -1;
    }
    pool->allocation_index_bucket_count = 65536U;
    pool->reusable_index_bucket_count = 8192U;
    pool->leases_by_id = calloc(
        (size_t)pool->allocation_index_bucket_count,
        sizeof(*pool->leases_by_id)
    );
    pool->leases_by_pointer = calloc(
        (size_t)pool->allocation_index_bucket_count,
        sizeof(*pool->leases_by_pointer)
    );
    pool->reusable_leases_by_size = calloc(
        (size_t)pool->reusable_index_bucket_count,
        sizeof(*pool->reusable_leases_by_size)
    );
    if (pool->leases_by_id == NULL || pool->leases_by_pointer == NULL ||
        pool->reusable_leases_by_size == NULL) {
        free(pool->reusable_leases_by_size);
        free(pool->leases_by_pointer);
        free(pool->leases_by_id);
        shadowspill_range_destroy(&pool->ranges);
        pthread_mutex_destroy(&pool->lock);
        if (base != NULL) {
            (void)arena_release(backend, kind, base, capacity);
        }
        *pool = (ShadowSpillMemoryPool){0};
        return -1;
    }
    pool->backend = backend;
    pool->kind = kind;
    pool->arena_bytes = capacity;
    pool->base = base;
    pool->pool_id = pool_id;
    pool->minimum_alignment = minimum_alignment;
    pool->next_request_sequence = 1U;
    pool->next_release_sequence = 1U;
    atomic_init(&pool->foreground_waiters, 0U);
    atomic_init(&pool->reservation_waiters, 0U);
    atomic_init(&pool->capacity_epoch, 0U);
    atomic_init(&pool->pending_retirements, 0U);
    atomic_init(&pool->pending_capacity_actions, 0U);
    atomic_init(&pool->free_bytes_snapshot, capacity);
    atomic_init(&pool->largest_free_bytes_snapshot, capacity);
    pool->initialized = 1U;
    return 0;
}

void shadowspill_memory_pool_close(ShadowSpillMemoryPool *pool) {
    if (pool == NULL || !pool->initialized) {
        return;
    }
    shadowspill_range_destroy(&pool->ranges);
    free(pool->release_range_workspace);
    free(pool->release_frontier_workspace);
    free(pool->reusable_leases_by_size);
    free(pool->leases_by_pointer);
    free(pool->leases_by_id);
    ShadowSpillLeaseUseRecord *use = pool->owned_use_records;
    while (use != NULL) {
        ShadowSpillLeaseUseRecord *next = use->ownership_next;
        free(use);
        use = next;
    }
    if (pool->base != NULL) {
        (void)arena_release(pool->backend, pool->kind, pool->base, pool->arena_bytes);
    }
    pthread_mutex_destroy(&pool->lock);
    *pool = (ShadowSpillMemoryPool){0};
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

void shadowspill_memory_pool_rebase_locked(
    ShadowSpillMemoryPool *pool,
    void *new_base
) {
    if (pool == NULL) {
        return;
    }
    pool->base = new_base;
    for (ShadowSpillMemoryLease *lease = pool->range_leases; lease != NULL;
         lease = lease->pool_next) {
        lease->pointer = shadowspill_memory_pool_pointer(pool, lease->offset);
    }
}

uint64_t shadowspill_memory_pool_free_bytes_locked(
    const ShadowSpillMemoryPool *pool
) {
    return shadowspill_range_free_bytes(&pool->ranges);
}

uint64_t shadowspill_memory_pool_free_prefix_locked(
    const ShadowSpillMemoryPool *pool
) {
    return shadowspill_range_free_prefix(&pool->ranges);
}

uint64_t shadowspill_memory_pool_largest_free_locked(
    const ShadowSpillMemoryPool *pool
) {
    return shadowspill_range_largest_free(&pool->ranges);
}
