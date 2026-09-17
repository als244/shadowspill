/* How a pool's memory is acquired and released, and where a location
   resolves to. */
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

/*
 * The two kinds the runtime implements itself, one entry each. There is no
 * switch on kind any more: an entry is found by kind and its own pair is
 * called, which is the same way a kind a library registers is reached.
 *
 * Both are configured with the backend and hand it back as their state, since
 * that is all either needs to release what it took.
 */

static int device_acquire(
    void *configuration, uint64_t capacity, void **base, void **state
) {
    const ShadowSpillBackend *backend = configuration;
    if (backend == NULL) {
        return -1;
    }
    if (backend->allocate_device(backend->state, capacity, base) != 0) {
        return -1;
    }
    *state = configuration;
    return 0;
}

static int device_release(void *state, void *base, uint64_t capacity) {
    const ShadowSpillBackend *backend = state;
    return backend->free_device(backend->state, base, capacity);
}

/* An anonymous private mapping, so the pool owns page-aligned memory the C
 * allocator never touches, and the backend pins it in place. */
static int pinned_host_acquire(
    void *configuration, uint64_t capacity, void **base, void **state
) {
    const ShadowSpillBackend *backend = configuration;
    if (backend == NULL || capacity == 0U || capacity > SIZE_MAX) {
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
    *state = configuration;
    return 0;
}

static int pinned_host_release(void *state, void *base, uint64_t capacity) {
    const ShadowSpillBackend *backend = state;
    const int status =
        backend->unregister_host_memory(backend->state, base, capacity);
    (void)munmap(base, (size_t)capacity);
    return status;
}

/*
 * Filled in while the runtime builds its lookup, so the descriptions point at
 * a backend that outlives them.
 */
void shadowspill_builtin_pool_memory_describe(
    const ShadowSpillBackend *backend,
    ShadowSpillPoolMemoryDescription descriptions[2]
) {
    descriptions[0] = (ShadowSpillPoolMemoryDescription){
        .kind = SHADOWSPILL_POOL_DEVICE,
        .acquire = device_acquire,
        .release = device_release,
        .configuration = (void *)(uintptr_t)backend,
    };
    descriptions[1] = (ShadowSpillPoolMemoryDescription){
        .kind = SHADOWSPILL_POOL_PINNED_HOST,
        .acquire = pinned_host_acquire,
        .release = pinned_host_release,
        .configuration = (void *)(uintptr_t)backend,
    };
}

int shadowspill_memory_pool_initialize(
    ShadowSpillMemoryPool *pool,
    uint32_t pool_id,
    const ShadowSpillPoolMemoryDescription *memory,
    uint8_t kind,
    uint64_t capacity,
    uint64_t minimum_alignment
) {
    if (pool == NULL || minimum_alignment == 0U || memory == NULL) {
        return -1;
    }
    void *base = NULL;
    void *state = NULL;
    if (capacity != 0U && memory->acquire(
            memory->configuration, capacity, &base, &state
        ) != 0) {
        return -1;
    }
    if (pthread_mutex_init(&pool->lock, NULL) != 0) {
        if (base != NULL) {
            (void)memory->release(state, base, capacity);
        }
        return -1;
    }
    if (shadowspill_range_initialize(&pool->ranges, capacity) != 0) {
        pthread_mutex_destroy(&pool->lock);
        if (base != NULL) {
            (void)memory->release(state, base, capacity);
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
            (void)memory->release(state, base, capacity);
        }
        *pool = (ShadowSpillMemoryPool){0};
        return -1;
    }
    pool->memory = *memory;
    /*
     * `configuration` is borrowed for create: nothing may read it once
     * `acquire` has returned, and the caller that built it is free to drop it
     * the moment the runtime exists -- a Python caller does. Keeping the
     * pointer would leave a dangling one that works right up until someone
     * finds a use for it, so drop it rather than write down a rule.
     */
    pool->memory.configuration = NULL;
    pool->memory_state = state;
    pool->kind = kind;
    pool->memory_bytes = capacity;
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
    /* A pool that never acquired has nothing to give back and no entry to give
       it back through -- the admission replay's pool is offsets only. Checking
       both says so, rather than relying on the two always agreeing. */
    if (pool->base != NULL && pool->memory.release != NULL) {
        (void)pool->memory.release(
            pool->memory_state, pool->base, pool->memory_bytes
        );
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
