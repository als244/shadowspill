#include "internal.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

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
    return shadowspill_runtime_pool(runtime, SHADOWSPILL_EXECUTION_POOL_ID);
}

const ShadowSpillMemoryPool *shadowspill_execution_pool_const(
    const ShadowSpillRuntime *runtime
) {
    return runtime == NULL
        ? NULL
        : shadowspill_runtime_pool_const(
              runtime, SHADOWSPILL_EXECUTION_POOL_ID
          );
}

ShadowSpillMemoryPool *shadowspill_spill_pool(ShadowSpillRuntime *runtime) {
    return runtime == NULL
        ? NULL
        : shadowspill_runtime_pool(runtime, SHADOWSPILL_SPILL_POOL_ID);
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
        : shadowspill_object_location(object, SHADOWSPILL_EXECUTION_POOL_ID);
}

ShadowSpillObjectLocation *shadowspill_spill_location(
    ShadowSpillRuntime *runtime,
    ShadowSpillObject *object
) {
    return runtime == NULL
        ? NULL
        : shadowspill_object_location(object, SHADOWSPILL_SPILL_POOL_ID);
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
            (void)backend->close(backend->context, base);
        }
        return -1;
    }
    if (shadowspill_range_initialize(&pool->ranges, capacity) != 0) {
        pthread_mutex_destroy(&pool->lock);
        if (base != NULL) {
            (void)backend->close(backend->context, base);
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
            (void)backend->close(backend->context, base);
        }
        *pool = (ShadowSpillMemoryPool){0};
        return -1;
    }
    pool->backend = *backend;
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
    free(pool->reusable_leases_by_size);
    free(pool->leases_by_pointer);
    free(pool->leases_by_id);
    if (pool->base != NULL) {
        (void)pool->backend.close(pool->backend.context, pool->base);
    }
    pthread_mutex_destroy(&pool->lock);
    *pool = (ShadowSpillMemoryPool){0};
}

ShadowSpillRuntimeStatus shadowspill_memory_pool_reserve_lease_records(
    ShadowSpillMemoryPool *pool,
    uint64_t minimum_free_records
) {
    if (pool == NULL || !pool->initialized || minimum_free_records == 0U) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&pool->lock);
    if (pool->lease_record_available >= minimum_free_records) {
        pool->lease_records_sealed = 1U;
        pthread_mutex_unlock(&pool->lock);
        return SHADOWSPILL_RUNTIME_OK;
    }
    const uint64_t additional =
        minimum_free_records - pool->lease_record_available;
    pthread_mutex_unlock(&pool->lock);

    ShadowSpillMemoryLease *created = NULL;
    uint64_t created_count = 0U;
    while (created_count < additional) {
        ShadowSpillMemoryLease *record = calloc(1U, sizeof(*record));
        if (record == NULL) {
            while (created != NULL) {
                ShadowSpillMemoryLease *next = created->free_record_next;
                free(created);
                created = next;
            }
            return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
        }
        record->free_record_next = created;
        created = record;
        ++created_count;
    }

    pthread_mutex_lock(&pool->lock);
    while (created != NULL) {
        ShadowSpillMemoryLease *next = created->free_record_next;
        created->metadata_owner = pool;
        atomic_init(&created->references, 1U);
        created->ownership_next = pool->owned_leases;
        pool->owned_leases = created;
        created->free_record_next = pool->free_lease_records;
        pool->free_lease_records = created;
        created = next;
    }
    pool->lease_record_capacity += created_count;
    pool->lease_record_available += created_count;
    pool->lease_records_sealed = 1U;
    pthread_mutex_unlock(&pool->lock);
    return SHADOWSPILL_RUNTIME_OK;
}

void shadowspill_memory_pool_lock_foreground(ShadowSpillMemoryPool *pool) {
    (void)atomic_fetch_add_explicit(
        &pool->foreground_waiters, 1U, memory_order_relaxed
    );
    while (atomic_load_explicit(
               &pool->reservation_waiters, memory_order_acquire
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
        cpu_relax();
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
    ShadowSpillMemoryLease **frontier = candidate_count == 0U
        ? NULL
        : calloc((size_t)candidate_count, sizeof(*frontier));
    if (candidate_count != 0U && frontier == NULL) {
        return -1;
    }
    uint64_t frontier_count = 0U;
    const int status = shadowspill_memory_pool_find_release_frontier_locked(
        pool,
        bytes,
        alignment,
        frontier,
        candidate_count,
        &frontier_count
    );
    free(frontier);
    return status;
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
    ShadowSpillRangeAllocator future = {0};
    if (shadowspill_range_clone_extended(
            &pool->ranges, pool->ranges.capacity, &future
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

static int handoff_causal_range_locked(
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
        return handoff_causal_range_locked(
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
        const int handoff_status = handoff_causal_range_locked(
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

void *shadowspill_memory_pool_pointer(
    const ShadowSpillMemoryPool *pool,
    uint64_t offset
) {
    if (pool == NULL || pool->base == NULL || offset > pool->ranges.capacity) {
        return NULL;
    }
    return (void *)((unsigned char *)pool->base + offset);
}
