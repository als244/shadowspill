#include "internal.h"

#include <stdlib.h>

int shadowspill_event_pool_initialize(ShadowSpillEventPool *pool) {
    if (pool == NULL) {
        return -1;
    }
    *pool = (ShadowSpillEventPool){0};
    if (pthread_mutex_init(&pool->lock, NULL) != 0) {
        return -1;
    }
    pool->initialized = 1U;
    return 0;
}

void shadowspill_event_pool_destroy(
    ShadowSpillRuntime *runtime,
    ShadowSpillEventPool *pool
) {
    if (runtime == NULL || pool == NULL || !pool->initialized) {
        return;
    }
    ShadowSpillEventPoolBlock *block = pool->blocks;
    while (block != NULL) {
        ShadowSpillEventPoolBlock *next = block->next;
        for (uint64_t index = 0U; index < block->count; ++index) {
            ShadowSpillEventLease *lease = &block->leases[index];
            if (atomic_load_explicit(
                    &lease->references, memory_order_acquire
                ) != 0U) {
                (void)runtime->synchronization.destroy_event(
                    runtime->synchronization.context, lease->event
                );
            }
        }
        free(block->leases);
        free(block);
        block = next;
    }
    pthread_mutex_destroy(&pool->lock);
    *pool = (ShadowSpillEventPool){0};
}

ShadowSpillRuntimeStatus shadowspill_event_pool_reserve(
    ShadowSpillEventPool *pool,
    uint64_t minimum_free_leases
) {
    if (pool == NULL || !pool->initialized || minimum_free_leases == 0U) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&pool->lock);
    if (pool->available >= minimum_free_leases) {
        pool->sealed = 1U;
        pthread_mutex_unlock(&pool->lock);
        return SHADOWSPILL_RUNTIME_OK;
    }
    const uint64_t additional = minimum_free_leases - pool->available;
    pthread_mutex_unlock(&pool->lock);

    ShadowSpillEventPoolBlock *block = calloc(1U, sizeof(*block));
    ShadowSpillEventLease *leases = additional > SIZE_MAX / sizeof(*leases)
        ? NULL : calloc((size_t)additional, sizeof(*leases));
    if (block == NULL || leases == NULL) {
        free(leases);
        free(block);
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    block->leases = leases;
    block->count = additional;

    pthread_mutex_lock(&pool->lock);
    block->next = pool->blocks;
    pool->blocks = block;
    for (uint64_t index = 0U; index < additional; ++index) {
        leases[index].pool_owned = 1U;
        leases[index].free_next = pool->free_head;
        pool->free_head = &leases[index];
    }
    pool->capacity += additional;
    pool->available += additional;
    pool->sealed = 1U;
    pthread_mutex_unlock(&pool->lock);
    return SHADOWSPILL_RUNTIME_OK;
}

static ShadowSpillEventLease *acquire_event_record(
    ShadowSpillRuntime *runtime
) {
    ShadowSpillEventPool *pool = &runtime->events;
    pthread_mutex_lock(&pool->lock);
    ShadowSpillEventLease *lease = pool->free_head;
    const uint8_t sealed = pool->sealed;
    if (lease != NULL) {
        pool->free_head = lease->free_next;
        lease->free_next = NULL;
        --pool->available;
        ++pool->in_use;
        if (pool->in_use > pool->peak_in_use) {
            pool->peak_in_use = pool->in_use;
        }
    } else if (sealed) {
        ++pool->growth_rejections;
    }
    pthread_mutex_unlock(&pool->lock);
    if (lease != NULL || sealed) {
        return lease;
    }
    return calloc(1U, sizeof(*lease));
}

static void release_event_record(
    ShadowSpillRuntime *runtime,
    ShadowSpillEventLease *lease
) {
    if (!lease->pool_owned) {
        free(lease);
        return;
    }
    ShadowSpillEventPool *pool = &runtime->events;
    pthread_mutex_lock(&pool->lock);
    lease->free_next = pool->free_head;
    pool->free_head = lease;
    ++pool->available;
    if (pool->in_use != 0U) {
        --pool->in_use;
    }
    pthread_mutex_unlock(&pool->lock);
}

ShadowSpillRuntimeStatus shadowspill_event_lease_create_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillEventLease **output
) {
    if (runtime == NULL || output == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    *output = NULL;
    ShadowSpillEventLease *lease = acquire_event_record(runtime);
    if (lease == NULL) {
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    if (runtime->synchronization.create_event(
            runtime->synchronization.context, &lease->event
        ) != 0) {
        release_event_record(runtime, lease);
        return SHADOWSPILL_RUNTIME_BACKEND_FAILURE;
    }
    lease->generation = atomic_fetch_add_explicit(
        &runtime->next_event_generation, 1U, memory_order_relaxed
    );
    if (lease->generation == 0U) {
        lease->generation = atomic_fetch_add_explicit(
            &runtime->next_event_generation, 1U, memory_order_relaxed
        );
    }
    lease->completion_object_id = SHADOWSPILL_RUNTIME_NO_ID;
    lease->completion_allocation_id = SHADOWSPILL_RUNTIME_NO_ID;
    lease->completion_next = NULL;
    lease->completion_linked = 0U;
    atomic_init(&lease->references, 1U);
    atomic_init(&lease->backend_complete, 0U);
    *output = lease;
    return SHADOWSPILL_RUNTIME_OK;
}

void shadowspill_event_lease_retain(ShadowSpillEventLease *lease) {
    if (lease != NULL) {
        (void)atomic_fetch_add_explicit(
            &lease->references, 1U, memory_order_relaxed
        );
    }
}

int shadowspill_event_lease_is_complete(const ShadowSpillEventLease *lease) {
    return lease != NULL && atomic_load_explicit(
        &lease->backend_complete, memory_order_acquire
    ) != 0U;
}

int shadowspill_event_lease_release(
    ShadowSpillRuntime *runtime,
    ShadowSpillEventLease *lease
) {
    if (lease == NULL || atomic_fetch_sub_explicit(
            &lease->references, 1U, memory_order_acq_rel
        ) != 1U) {
        return 0;
    }
    const int status = runtime->synchronization.destroy_event(
        runtime->synchronization.context, lease->event
    );
    if (status != 0 && lease->pool_owned) {
        atomic_store_explicit(&lease->references, 1U, memory_order_release);
        return status;
    }
    lease->event = (ShadowSpillBackendEvent){0};
    lease->completion_object_id = SHADOWSPILL_RUNTIME_NO_ID;
    lease->completion_allocation_id = SHADOWSPILL_RUNTIME_NO_ID;
    lease->completion_next = NULL;
    lease->completion_linked = 0U;
    atomic_store_explicit(&lease->backend_complete, 0U, memory_order_relaxed);
    release_event_record(runtime, lease);
    return status;
}

int shadowspill_event_lease_query(
    ShadowSpillRuntime *runtime,
    ShadowSpillEventLease *lease,
    int *complete
) {
    if (runtime == NULL || lease == NULL || complete == NULL) {
        return -1;
    }
    if (atomic_load_explicit(
            &lease->backend_complete, memory_order_acquire
        ) != 0U) {
        *complete = 1;
        return 0;
    }
    if (runtime->synchronization.query_event(
            runtime->synchronization.context, lease->event, complete
        ) != 0) {
        return -1;
    }
    if (*complete) {
        atomic_store_explicit(
            &lease->backend_complete, 1U, memory_order_release
        );
    }
    return 0;
}
