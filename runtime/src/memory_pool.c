#include "internal.h"

#include <stddef.h>
#include <stdint.h>

int shadowspill_memory_pool_initialize(
    ShadowSpillMemoryPool *pool,
    ShadowSpillMemoryKind kind,
    void *base,
    uint64_t capacity,
    uint64_t minimum_alignment
) {
    if (pool == NULL || minimum_alignment == 0U ||
        (capacity != 0U && base == NULL)) {
        return -1;
    }
    if (pthread_mutex_init(&pool->lock, NULL) != 0) {
        return -1;
    }
    if (pthread_cond_init(&pool->capacity_changed, NULL) != 0) {
        pthread_mutex_destroy(&pool->lock);
        return -1;
    }
    if (shadowspill_range_initialize(&pool->ranges, capacity) != 0) {
        pthread_cond_destroy(&pool->capacity_changed);
        pthread_mutex_destroy(&pool->lock);
        return -1;
    }
    pool->base = base;
    pool->minimum_alignment = minimum_alignment;
    pool->kind = kind;
    pool->initialized = 1U;
    return 0;
}

void shadowspill_memory_pool_destroy(ShadowSpillMemoryPool *pool) {
    if (pool == NULL || !pool->initialized) {
        return;
    }
    shadowspill_range_destroy(&pool->ranges);
    pthread_cond_destroy(&pool->capacity_changed);
    pthread_mutex_destroy(&pool->lock);
    *pool = (ShadowSpillMemoryPool){0};
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
