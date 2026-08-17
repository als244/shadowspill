#include "internal.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int stream_equal(
    ShadowSpillBackendStream left,
    ShadowSpillBackendStream right
) {
    return memcmp(&left, &right, sizeof(left)) == 0;
}

void shadowspill_publish_pool_geometry_locked(ShadowSpillMemoryPool *pool) {
    atomic_store_explicit(
        &pool->free_bytes_snapshot,
        shadowspill_memory_pool_free_bytes_locked(pool),
        memory_order_release
    );
    atomic_store_explicit(
        &pool->largest_free_bytes_snapshot,
        shadowspill_memory_pool_largest_free_locked(pool),
        memory_order_release
    );
}

static uint64_t mix_index(uint64_t value, uint64_t bucket_count) {
    value ^= value >> 33U;
    value *= UINT64_C(0xff51afd7ed558ccd);
    value ^= value >> 33U;
    return value % bucket_count;
}

static void *allocation_lookup_pointer(
    const ShadowSpillMemoryLease *allocation
) {
    return allocation->pointer != NULL
        ? allocation->pointer
        : allocation->retired_pointer;
}

static void index_allocation_id_locked(
    ShadowSpillMemoryPool *pool,
    ShadowSpillMemoryLease *allocation
) {
    if (allocation->in_id_index) {
        return;
    }
    const uint64_t bucket = mix_index(
        allocation->allocation_id, pool->allocation_index_bucket_count
    );
    allocation->id_index_next = pool->leases_by_id[bucket];
    pool->leases_by_id[bucket] = allocation;
    allocation->in_id_index = 1U;
}

static void unindex_allocation_id_locked(
    ShadowSpillMemoryPool *pool,
    ShadowSpillMemoryLease *allocation
) {
    if (!allocation->in_id_index) {
        return;
    }
    const uint64_t bucket = mix_index(
        allocation->allocation_id, pool->allocation_index_bucket_count
    );
    ShadowSpillMemoryLease **link = &pool->leases_by_id[bucket];
    while (*link != NULL && *link != allocation) {
        link = &(*link)->id_index_next;
    }
    if (*link == allocation) {
        *link = allocation->id_index_next;
    }
    allocation->id_index_next = NULL;
    allocation->in_id_index = 0U;
}

static void index_allocation_pointer_locked(
    ShadowSpillMemoryPool *pool,
    ShadowSpillMemoryLease *allocation
) {
    if (allocation->in_pointer_index) {
        return;
    }
    const uint64_t address =
        (uint64_t)(uintptr_t)allocation_lookup_pointer(allocation);
    const uint64_t bucket = mix_index(
        address, pool->allocation_index_bucket_count
    );
    allocation->pointer_index_next = pool->leases_by_pointer[bucket];
    pool->leases_by_pointer[bucket] = allocation;
    allocation->in_pointer_index = 1U;
}

static void unindex_allocation_pointer_locked(
    ShadowSpillMemoryPool *pool,
    ShadowSpillMemoryLease *allocation
) {
    if (!allocation->in_pointer_index) {
        return;
    }
    const uint64_t address =
        (uint64_t)(uintptr_t)allocation_lookup_pointer(allocation);
    const uint64_t bucket = mix_index(
        address, pool->allocation_index_bucket_count
    );
    ShadowSpillMemoryLease **link = &pool->leases_by_pointer[bucket];
    while (*link != NULL && *link != allocation) {
        link = &(*link)->pointer_index_next;
    }
    if (*link == allocation) {
        *link = allocation->pointer_index_next;
    }
    allocation->pointer_index_next = NULL;
    allocation->in_pointer_index = 0U;
}

static void index_reusable_locked(
    ShadowSpillMemoryPool *pool,
    ShadowSpillMemoryLease *allocation
) {
    if (allocation->in_reusable_index) {
        return;
    }
    const uint64_t bucket = mix_index(
        allocation->charged_bytes, pool->reusable_index_bucket_count
    );
    allocation->reusable_index_next = pool->reusable_leases_by_size[bucket];
    pool->reusable_leases_by_size[bucket] = allocation;
    allocation->in_reusable_index = 1U;
}

static void unindex_reusable_locked(
    ShadowSpillMemoryPool *pool,
    ShadowSpillMemoryLease *allocation
) {
    if (!allocation->in_reusable_index) {
        return;
    }
    const uint64_t bucket = mix_index(
        allocation->charged_bytes, pool->reusable_index_bucket_count
    );
    ShadowSpillMemoryLease **link = &pool->reusable_leases_by_size[bucket];
    while (*link != NULL && *link != allocation) {
        link = &(*link)->reusable_index_next;
    }
    if (*link == allocation) {
        *link = allocation->reusable_index_next;
    }
    allocation->reusable_index_next = NULL;
    allocation->in_reusable_index = 0U;
}

static void activate_allocation_locked(
    ShadowSpillMemoryPool *pool,
    ShadowSpillMemoryLease *allocation
) {
    allocation->active_next = pool->active_leases;
    allocation->active_previous_link = &pool->active_leases;
    if (allocation->active_next != NULL) {
        allocation->active_next->active_previous_link = &allocation->active_next;
    }
    pool->active_leases = allocation;
}

static void deactivate_allocation_locked(
    ShadowSpillMemoryLease *allocation
) {
    if (allocation->active_previous_link == NULL) {
        return;
    }
    *allocation->active_previous_link = allocation->active_next;
    if (allocation->active_next != NULL) {
        allocation->active_next->active_previous_link =
            allocation->active_previous_link;
    }
    allocation->active_next = NULL;
    allocation->active_previous_link = NULL;
}

static void free_stream_records(ShadowSpillStreamRecord *streams) {
    while (streams != NULL) {
        ShadowSpillStreamRecord *next = streams->next;
        free(streams);
        streams = next;
    }
}

ShadowSpillMemoryLease *shadowspill_find_execution_lease(
    ShadowSpillMemoryPool *pool,
    uint64_t allocation_id
) {
    const uint64_t bucket = mix_index(
        allocation_id, pool->allocation_index_bucket_count
    );
    for (ShadowSpillMemoryLease *record = pool->leases_by_id[bucket];
         record != NULL; record = record->id_index_next) {
        if (record->allocation_id == allocation_id) {
            return record;
        }
    }
    return NULL;
}

ShadowSpillMemoryLease *shadowspill_find_execution_lease_by_pointer(
    ShadowSpillMemoryPool *pool,
    const void *pointer
) {
    const uint64_t bucket = mix_index(
        (uint64_t)(uintptr_t)pointer,
        pool->allocation_index_bucket_count
    );
    for (ShadowSpillMemoryLease *record = pool->leases_by_pointer[bucket];
         record != NULL; record = record->pointer_index_next) {
        if (record->pointer == pointer && !record->logical_freed) {
            return record;
        }
        if (record->logical_freed && record->ever_plan_owned &&
            !record->framework_free_seen &&
            record->retired_pointer == pointer) {
            return record;
        }
    }
    return NULL;
}

static int has_release_source(const ShadowSpillMemoryPool *pool) {
    return atomic_load_explicit(
        &pool->pending_retirements, memory_order_acquire
    ) != 0U || atomic_load_explicit(
        &pool->pending_capacity_actions, memory_order_acquire
    ) != 0U;
}

ShadowSpillRuntimeStatus shadowspill_publish_task_retirement_event(
    ShadowSpillRuntime *runtime,
    uint64_t task_id,
    ShadowSpillBackendStream stream
) {
    ShadowSpillMemoryPool *pool = shadowspill_current_allocation_pool(runtime);
    if (pool == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    uint64_t count = 0U;
    shadowspill_memory_pool_lock_foreground(pool);
    for (ShadowSpillMemoryLease *allocation =
             shadowspill_current_task_retirements(runtime);
         allocation != NULL;
         allocation = allocation->task_retirement_next) {
        if (allocation->logical_freed && allocation->pointer != NULL &&
            allocation->release_task_id == task_id &&
            allocation->retirement_events == NULL &&
            allocation->retirement_event == NULL) {
            ++count;
        }
    }
    shadowspill_memory_pool_unlock_foreground(pool);
    if (count == 0U) {
        return SHADOWSPILL_RUNTIME_OK;
    }
    ShadowSpillEventLease *task_completion_event = NULL;
    const ShadowSpillRuntimeStatus event_status =
        shadowspill_event_lease_create_locked(runtime, &task_completion_event);
    if (event_status != SHADOWSPILL_RUNTIME_OK ||
        runtime->synchronization.record_event(
            runtime->synchronization.context,
            task_completion_event->event,
            stream
        ) != 0 || shadowspill_completion_submit(
            runtime,
            stream,
            task_completion_event,
            SHADOWSPILL_RUNTIME_NO_ID,
            SHADOWSPILL_RUNTIME_NO_ID
        ) != SHADOWSPILL_RUNTIME_OK) {
        if (task_completion_event != NULL) {
            (void)shadowspill_event_lease_release(
                runtime, task_completion_event
            );
        }
        return SHADOWSPILL_RUNTIME_BACKEND_FAILURE;
    }
    ShadowSpillRuntimeStatus status = SHADOWSPILL_RUNTIME_OK;
    shadowspill_memory_pool_lock_foreground(pool);
    for (ShadowSpillMemoryLease *allocation =
             shadowspill_current_task_retirements(runtime);
         allocation != NULL;
         allocation = allocation->task_retirement_next) {
        if (!allocation->logical_freed || allocation->pointer == NULL ||
            allocation->release_task_id != task_id ||
            allocation->retirement_events != NULL ||
            allocation->retirement_event != NULL) {
            continue;
        }
        if (shadowspill_memory_pool_publish_retirement_dependency_locked(
                allocation, task_completion_event
            ) != 0) {
            status = SHADOWSPILL_RUNTIME_INVALID_STATE;
            break;
        }
        allocation->retirement_event = task_completion_event;
        shadowspill_event_lease_retain(task_completion_event);
        const ShadowSpillRuntimeStatus enqueue_status =
            shadowspill_retirement_enqueue_locked(runtime, allocation);
        if (enqueue_status != SHADOWSPILL_RUNTIME_OK) {
            status = enqueue_status;
            break;
        }
    }
    shadowspill_memory_pool_unlock_foreground(pool);
    (void)shadowspill_event_lease_release(runtime, task_completion_event);
    if (status != SHADOWSPILL_RUNTIME_OK) {
        return status;
    }
    return SHADOWSPILL_RUNTIME_OK;
}

static void initialize_memory_lease_record(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryPool *pool,
    ShadowSpillMemoryLease *record,
    uint64_t origin_task_id
) {
    ShadowSpillMemoryLease *ownership_next = record->ownership_next;
    memset(record, 0, sizeof(*record));
    record->metadata_owner = pool;
    record->ownership_next = ownership_next;
    record->metadata_in_use = 1U;
    record->allocation_id = atomic_fetch_add_explicit(
        &runtime->next_allocation_id, 1U, memory_order_relaxed
    );
    atomic_init(&record->references, 1U);
    record->generation = atomic_fetch_add_explicit(
        &runtime->next_generation, 1U, memory_order_relaxed
    );
    record->origin_task_id = origin_task_id;
    record->origin_task_allocation_sequence = SHADOWSPILL_RUNTIME_NO_ID;
    record->origin_task_allocation_ordinal = SHADOWSPILL_RUNTIME_NO_ID;
    record->origin_task_allocation_is_scratch = 0U;
    record->release_task_id = SHADOWSPILL_RUNTIME_NO_ID;
}

ShadowSpillMemoryLease *shadowspill_memory_pool_acquire_lease_record_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryPool *pool,
    uint64_t origin_task_id
) {
    ShadowSpillMemoryLease *record = pool->free_lease_records;
    if (record != NULL) {
        pool->free_lease_records = record->free_record_next;
        record->free_record_next = NULL;
        --pool->lease_record_available;
    } else if (pool->lease_records_sealed) {
        ++pool->lease_record_growth_rejections;
        return NULL;
    } else {
        record = calloc(1U, sizeof(*record));
        if (record == NULL) {
            return NULL;
        }
        record->metadata_owner = pool;
        record->ownership_next = pool->owned_leases;
        pool->owned_leases = record;
        ++pool->lease_record_capacity;
    }
    ++pool->lease_record_in_use;
    if (pool->lease_record_in_use > pool->lease_record_peak_in_use) {
        pool->lease_record_peak_in_use = pool->lease_record_in_use;
    }
    initialize_memory_lease_record(runtime, pool, record, origin_task_id);
    return record;
}

static int memory_lease_record_is_recyclable(
    const ShadowSpillMemoryLease *record
) {
    return record != NULL && record->metadata_owner != NULL &&
        record->metadata_in_use && record->pool == NULL &&
        record->state == SHADOWSPILL_LEASE_FREE &&
        record->pointer == NULL && record->active_previous_link == NULL &&
        !record->in_id_index && !record->in_pointer_index &&
        !record->in_reusable_index && !record->task_retirement_linked &&
        record->streams == NULL && record->retirement_events == NULL &&
        record->retirement_event == NULL && record->bound_object == NULL &&
        record->causal_predecessor == NULL &&
        record->causal_successor == NULL && record->pool_next == NULL &&
        record->pool_previous_link == NULL &&
        (!record->ever_plan_owned || record->framework_free_seen) &&
        atomic_load_explicit(&record->references, memory_order_acquire) == 1U;
}

void shadowspill_memory_pool_try_recycle_lease_record_locked(
    ShadowSpillMemoryLease *record
) {
    if (!memory_lease_record_is_recyclable(record)) {
        return;
    }
    ShadowSpillMemoryPool *owner = record->metadata_owner;
    ShadowSpillMemoryLease *ownership_next = record->ownership_next;
    memset(record, 0, sizeof(*record));
    record->metadata_owner = owner;
    record->ownership_next = ownership_next;
    atomic_init(&record->references, 1U);
    record->free_record_next = owner->free_lease_records;
    owner->free_lease_records = record;
    ++owner->lease_record_available;
    if (owner->lease_record_in_use != 0U) {
        --owner->lease_record_in_use;
    }
}

void shadowspill_memory_lease_retain(ShadowSpillMemoryLease *lease) {
    if (lease != NULL) {
        (void)atomic_fetch_add_explicit(
            &lease->references, 1U, memory_order_relaxed
        );
    }
}

void shadowspill_memory_lease_release(ShadowSpillMemoryLease *lease) {
    if (lease == NULL || lease->metadata_owner == NULL) {
        return;
    }
    uint32_t references = atomic_load_explicit(
        &lease->references, memory_order_acquire
    );
    while (references > 1U && !atomic_compare_exchange_weak_explicit(
               &lease->references,
               &references,
               references - 1U,
               memory_order_acq_rel,
               memory_order_acquire
           )) {
    }
    if (references != 2U) {
        return;
    }
    ShadowSpillMemoryPool *owner = lease->metadata_owner;
    pthread_mutex_lock(&owner->lock);
    shadowspill_memory_pool_try_recycle_lease_record_locked(lease);
    pthread_mutex_unlock(&owner->lock);
}

static void publish_execution_record_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryLease *record,
    ShadowSpillAllocationCategory category
) {
    ShadowSpillMemoryPool *pool = record->pool;
    activate_allocation_locked(pool, record);
    index_allocation_id_locked(pool, record);
    index_allocation_pointer_locked(pool, record);
    pool->requested_allocated_bytes += record->requested_bytes;
    if (pool->requested_allocated_bytes >
        pool->peak_requested_allocated_bytes) {
        pool->peak_requested_allocated_bytes =
            pool->requested_allocated_bytes;
    }
    ++pool->live_allocations;
    shadowspill_append_allocation_event_locked(
        runtime, record, SHADOWSPILL_ALLOCATION_CREATED, category
    );
}

static ShadowSpillRuntimeStatus own_and_publish_execution_lease_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryLease *created,
    int plan_owned,
    ShadowSpillMemoryLease **record
) {
    ShadowSpillMemoryPool *pool = created->pool;
    created->plan_owned = plan_owned;
    publish_execution_record_locked(
        runtime,
        created,
        plan_owned ? SHADOWSPILL_ALLOCATION_PLANNED_OBJECT
                   : SHADOWSPILL_ALLOCATION_ANONYMOUS
    );
    const ShadowSpillRuntimeStatus status = shadowspill_failure_status(runtime);
    if (status == SHADOWSPILL_RUNTIME_OK) {
        *record = created;
        return status;
    }
    unindex_allocation_pointer_locked(pool, created);
    unindex_allocation_id_locked(pool, created);
    deactivate_allocation_locked(created);
    pool->requested_allocated_bytes -= created->requested_bytes;
    --pool->live_allocations;
    (void)shadowspill_memory_pool_release_lease_locked(created);
    shadowspill_publish_pool_geometry_locked(pool);
    shadowspill_memory_pool_try_recycle_lease_record_locked(created);
    return status;
}

static ShadowSpillRuntimeStatus create_execution_lease_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryPool *pool,
    uint64_t bytes,
    uint64_t alignment,
    int plan_owned,
    ShadowSpillMemoryPlacement placement,
    uint64_t origin_task_id,
    ShadowSpillMemoryLease **record
) {
    if (record == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    *record = NULL;
    if (pool == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    if (alignment < pool->minimum_alignment) {
        alignment = pool->minimum_alignment;
    }
    ShadowSpillMemoryLease *created =
        shadowspill_memory_pool_acquire_lease_record_locked(
        runtime, pool, origin_task_id
    );
    if (created == NULL) {
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    const int reserve_status = shadowspill_memory_pool_reserve_lease_locked(
        pool,
        created,
        bytes,
        alignment,
        placement
    );
    if (reserve_status != 0) {
        shadowspill_memory_pool_try_recycle_lease_record_locked(created);
        return reserve_status > 0
            ? SHADOWSPILL_RUNTIME_OUT_OF_MEMORY
            : SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    shadowspill_publish_pool_geometry_locked(pool);
    return own_and_publish_execution_lease_locked(
        runtime, created, plan_owned, record
    );
}

ShadowSpillRuntimeStatus shadowspill_create_fixed_execution_lease_locked(
    ShadowSpillPlan *plan,
    const ShadowSpillFixedPlacementDescription *placement,
    int plan_owned,
    uint64_t origin_task_id,
    ShadowSpillMemoryLease **record
) {
    if (plan == NULL || placement == NULL || record == NULL ||
        (placement->kind != SHADOWSPILL_FIXED_TASK_ALLOCATION &&
         placement->kind != SHADOWSPILL_FIXED_ACTION_DESTINATION)) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    ShadowSpillRuntime *runtime = plan->runtime;
    *record = NULL;
    ShadowSpillMemoryPool *pool = plan->execution_pool;
    ShadowSpillMemoryLease *created =
        shadowspill_memory_pool_acquire_lease_record_locked(
        runtime, pool, origin_task_id
    );
    if (created == NULL) {
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    ShadowSpillRuntimeStatus status =
        shadowspill_fixed_layout_adopt_execution_lease_locked(
            plan,
            created,
            placement->offset,
            placement->bytes,
            placement->alignment_bytes
        );
    if (status != SHADOWSPILL_RUNTIME_OK) {
        shadowspill_memory_pool_try_recycle_lease_record_locked(created);
        return status;
    }
    return own_and_publish_execution_lease_locked(
        runtime, created, plan_owned, record
    );
}

ShadowSpillRuntimeStatus shadowspill_create_execution_successor_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryPool *pool,
    uint64_t bytes,
    uint64_t alignment,
    uint64_t origin_task_id,
    ShadowSpillMemoryLease **record
) {
    if (runtime == NULL || pool == NULL || record == NULL || bytes == 0U) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    *record = NULL;
    ShadowSpillMemoryLease *created =
        shadowspill_memory_pool_acquire_lease_record_locked(
        runtime, pool, origin_task_id
    );
    if (created == NULL) {
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    const int reserve_status =
        shadowspill_memory_pool_reserve_causal_successor_locked(
            pool,
            created,
            bytes,
            alignment
        );
    if (reserve_status != 0) {
        shadowspill_memory_pool_try_recycle_lease_record_locked(created);
        return reserve_status > 0
            ? SHADOWSPILL_RUNTIME_OUT_OF_MEMORY
            : SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    /* The predecessor is now promised to this transfer reservation. */
    unindex_reusable_locked(pool, created->causal_predecessor);
    created->plan_owned = 1;
    created->ever_plan_owned = 1;
    *record = created;
    return SHADOWSPILL_RUNTIME_OK;
}

static void publish_execution_successor_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryLease *successor
) {
    if (successor->active_previous_link != NULL) {
        return;
    }
    publish_execution_record_locked(
        runtime,
        successor,
        SHADOWSPILL_ALLOCATION_PLANNED_OBJECT
    );
    shadowspill_publish_pool_geometry_locked(successor->pool);
}

int shadowspill_acquire_reserved_execution_lease_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryLease *successor,
    ShadowSpillEventLease **dependency_event
) {
    const int status = shadowspill_memory_pool_acquire_reserved_lease_locked(
        successor, dependency_event
    );
    if (status == 0) {
        publish_execution_successor_locked(runtime, successor);
    }
    return status;
}

void shadowspill_cancel_execution_reservation_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryLease *lease
) {
    if (runtime == NULL || lease == NULL) {
        return;
    }
    if (lease->state == SHADOWSPILL_LEASE_SUCCESSOR_RESERVED) {
        ShadowSpillMemoryLease *predecessor = lease->causal_predecessor;
        if (shadowspill_memory_pool_cancel_reservation_locked(lease) != 0) {
            shadowspill_latch_failure_locked(
                runtime,
                SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE,
                SHADOWSPILL_RUNTIME_NO_ID,
                lease->allocation_id,
                lease->requested_bytes
            );
        } else if (predecessor != NULL && predecessor->logical_freed &&
                   predecessor->pointer != NULL &&
                   !predecessor->retirement_preparing) {
            index_reusable_locked(predecessor->pool, predecessor);
        }
        if (lease->state == SHADOWSPILL_LEASE_FREE) {
            lease->plan_owned = 0;
            lease->framework_free_seen = 1;
            shadowspill_memory_pool_try_recycle_lease_record_locked(lease);
        }
        return;
    }
    shadowspill_release_execution_lease_locked(runtime, lease);
}

ShadowSpillRuntimeStatus shadowspill_create_execution_lease_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryPool *pool,
    uint64_t bytes,
    uint64_t alignment,
    int plan_owned,
    ShadowSpillMemoryPlacement placement,
    uint64_t origin_task_id,
    ShadowSpillMemoryLease **record
) {
    return create_execution_lease_locked(
        runtime,
        pool,
        bytes,
        alignment,
        plan_owned,
        placement,
        origin_task_id,
        record
    );
}

static ShadowSpillRuntimeStatus reuse_pending_allocation_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryPool *pool,
    uint64_t bytes,
    uint64_t alignment,
    ShadowSpillBackendStream stream,
    uint64_t origin_task_id,
    int exact_task_local_only,
    ShadowSpillMemoryLease **record
) {
    const uint64_t required = bytes == 0U ? 1U : bytes;
    if (alignment < pool->minimum_alignment) {
        alignment = pool->minimum_alignment;
    }
    const uint64_t reusable_bucket = mix_index(
        required, pool->reusable_index_bucket_count
    );
    ShadowSpillMemoryLease *selected = NULL;
    for (ShadowSpillMemoryLease *candidate = exact_task_local_only
             ? pool->reusable_leases_by_size[reusable_bucket]
             : pool->active_leases;
         candidate != NULL;
         candidate = exact_task_local_only
             ? candidate->reusable_index_next
             : candidate->active_next) {
        if (!candidate->owns_pool_range || !candidate->logical_freed ||
            candidate->pointer == NULL ||
            candidate->retirement_preparing || candidate->ever_plan_owned ||
            candidate->causal_successor != NULL ||
            candidate->charged_bytes < required ||
            (exact_task_local_only &&
             candidate->charged_bytes != required) ||
            candidate->offset % alignment != 0U) {
            continue;
        }
        const int task_local = candidate->retirement_events == NULL &&
            candidate->retirement_event == NULL &&
            candidate->release_task_id == origin_task_id &&
            origin_task_id != SHADOWSPILL_RUNTIME_NO_ID;
        if (exact_task_local_only && !task_local) {
            continue;
        }
        if (!task_local && candidate->retirement_events == NULL &&
            candidate->retirement_event == NULL) {
            continue;
        }
        int stream_compatible = 1;
        for (ShadowSpillStreamRecord *used = candidate->streams;
             used != NULL; used = used->next) {
            if (!stream_equal(used->stream, stream)) {
                stream_compatible = 0;
                break;
            }
        }
        if (!stream_compatible) {
            continue;
        }
        if (selected == NULL ||
            candidate->charged_bytes < selected->charged_bytes ||
            (candidate->charged_bytes == selected->charged_bytes &&
             (candidate->release_sequence < selected->release_sequence ||
              (candidate->release_sequence == selected->release_sequence &&
               candidate->offset < selected->offset)))) {
            selected = candidate;
        }
    }
    if (selected == NULL) {
        *record = NULL;
        return SHADOWSPILL_RUNTIME_OK;
    }
    ShadowSpillMemoryLease *split = NULL;
    if (selected->charged_bytes > required) {
        split = shadowspill_memory_pool_acquire_lease_record_locked(
            runtime, pool, origin_task_id
        );
        if (split == NULL) {
            return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
        }
        /*
         * The split range is already charged to ``selected``.  Adopt it into
         * the pool's lease registry before changing either range owner so the
         * new lease receives the same intrusive-list ownership invariants as
         * every ordinarily allocated lease.  Adoption does not reserve the
         * range a second time.
         */
        if (shadowspill_memory_pool_adopt_lease_locked(
                pool,
                split,
                bytes,
                alignment,
                selected->offset
            ) != 0) {
            shadowspill_memory_pool_try_recycle_lease_record_locked(split);
            return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
        }
    }
    /*
     * Every recorded use was on ``stream`` (checked above). Execution-stream order
     * already retires the old use before later work that consumes the reused
     * range, so adding a wait for an event recorded on the same stream is both
     * redundant and an unnecessary driver call in the allocator hot path.
     */
    unindex_reusable_locked(pool, selected);
    if (split != NULL) {
        unindex_allocation_pointer_locked(pool, selected);
        pool->requested_allocated_bytes -= selected->requested_bytes;
        selected->requested_bytes = 0U;
        selected->charged_bytes -= required;
        selected->offset += required;
        selected->pointer =
            shadowspill_memory_pool_pointer(
                pool, selected->offset
            );
        index_allocation_pointer_locked(pool, selected);
        index_reusable_locked(pool, selected);

        split->state = SHADOWSPILL_LEASE_IN_USE;
        activate_allocation_locked(pool, split);
        index_allocation_id_locked(pool, split);
        index_allocation_pointer_locked(pool, split);
        pool->requested_allocated_bytes += bytes;
        if (pool->requested_allocated_bytes >
            pool->peak_requested_allocated_bytes) {
            pool->peak_requested_allocated_bytes =
                pool->requested_allocated_bytes;
        }
        ++pool->live_allocations;
        shadowspill_append_allocation_event_locked(
            runtime,
            split,
            SHADOWSPILL_ALLOCATION_CREATED,
            SHADOWSPILL_ALLOCATION_ANONYMOUS
        );
        *record = split;
        return shadowspill_failure_status(runtime);
    }
    selected->retirement_events = NULL;
    if (atomic_fetch_sub_explicit(
            &runtime->pending_retirements, 1U, memory_order_release
        ) == 1U) {
        shadowspill_idle_notify(runtime);
    }
    (void)atomic_fetch_sub_explicit(
        &pool->pending_retirements, 1U, memory_order_release
    );
    if (selected->retirement_event != NULL) {
        selected->retirement_event = NULL;
    }
    pool->requested_allocated_bytes -= selected->requested_bytes;
    free_stream_records(selected->streams);
    selected->streams = NULL;
    unindex_allocation_id_locked(pool, selected);
    selected->allocation_id = atomic_fetch_add_explicit(
        &runtime->next_allocation_id, 1U, memory_order_relaxed
    );
    index_allocation_id_locked(pool, selected);
    selected->generation = atomic_fetch_add_explicit(
        &runtime->next_generation, 1U, memory_order_relaxed
    );
    selected->requested_bytes = bytes;
    selected->alignment_bytes = alignment;
    selected->origin_task_id = origin_task_id;
    selected->release_task_id = SHADOWSPILL_RUNTIME_NO_ID;
    selected->bound_object = NULL;
    selected->logical_freed = 0;
    selected->state = SHADOWSPILL_LEASE_IN_USE;
    selected->causal_event = NULL;
    selected->causal_dependency_expected = 0U;
    selected->framework_free_seen = 0;
    selected->plan_owned = 0;
    selected->ever_plan_owned = 0;
    pool->requested_allocated_bytes += bytes;
    if (pool->requested_allocated_bytes >
        pool->peak_requested_allocated_bytes) {
        pool->peak_requested_allocated_bytes =
            pool->requested_allocated_bytes;
    }
    shadowspill_append_allocation_event_locked(
        runtime,
        selected,
        SHADOWSPILL_ALLOCATION_CREATED,
        SHADOWSPILL_ALLOCATION_ANONYMOUS
    );
    *record = selected;
    return shadowspill_failure_status(runtime);
}

void shadowspill_release_execution_lease_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryLease *allocation
) {
    if (allocation->pointer == NULL) {
        return;
    }
    ShadowSpillMemoryLease *causal_successor = allocation->causal_successor;
    ShadowSpillMemoryPool *pool = allocation->pool;
    unindex_reusable_locked(pool, allocation);
    shadowspill_append_allocation_event_locked(
        runtime,
        allocation,
        SHADOWSPILL_ALLOCATION_RELEASED,
        allocation->plan_owned ? SHADOWSPILL_ALLOCATION_PLANNED_OBJECT
                               : SHADOWSPILL_ALLOCATION_ANONYMOUS
    );
    const uint64_t requested_bytes = allocation->requested_bytes;
    const uint64_t charged_bytes = allocation->charged_bytes;
    const int retain_framework_lookup = allocation->ever_plan_owned &&
        !allocation->framework_free_seen;
    if (!retain_framework_lookup) {
        unindex_allocation_pointer_locked(pool, allocation);
        unindex_allocation_id_locked(pool, allocation);
    }
    if (shadowspill_memory_pool_release_lease_locked(allocation) != 0) {
        shadowspill_latch_failure_locked(
            runtime,
            SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE,
            SHADOWSPILL_RUNTIME_NO_ID,
            allocation->allocation_id,
            charged_bytes
        );
        return;
    }
    free_stream_records(allocation->streams);
    allocation->streams = NULL;
    shadowspill_publish_pool_geometry_locked(pool);
    deactivate_allocation_locked(allocation);
    allocation->logical_freed = 1;
    allocation->plan_owned = 0;
    allocation->bound_object = NULL;
    pool->requested_allocated_bytes -= requested_bytes;
    if (pool->live_allocations != 0U) {
        --pool->live_allocations;
    }
    if (causal_successor != NULL &&
        causal_successor->state == SHADOWSPILL_LEASE_RESERVED) {
        publish_execution_successor_locked(runtime, causal_successor);
    }
    shadowspill_memory_pool_try_recycle_lease_record_locked(allocation);
}

ShadowSpillRuntimeStatus shadowspill_memory_pool_allocate(
    ShadowSpillRuntime *runtime,
    uint32_t pool_id,
    uint64_t bytes,
    uint64_t alignment,
    ShadowSpillBackendStream stream,
    ShadowSpillAllocation *allocation
) {
    ShadowSpillMemoryPool *pool = shadowspill_runtime_pool(runtime, pool_id);
    if (pool == NULL || allocation == NULL || alignment == 0U) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    ShadowSpillMemoryPool *scope_pool =
        shadowspill_current_allocation_pool(runtime);
    if (scope_pool != NULL && scope_pool != pool) {
        return SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    if (alignment < pool->minimum_alignment) {
        alignment = pool->minimum_alignment;
    }
    const uint64_t charged_bytes = bytes == 0U ? 0U : bytes;
    ShadowSpillRuntimeStatus status = bytes == 0U
        ? SHADOWSPILL_RUNTIME_OK
        : shadowspill_validate_task_allocation(
            runtime, bytes, charged_bytes, alignment
        );
    if (status != SHADOWSPILL_RUNTIME_OK) {
        return status;
    }
    const uint64_t task_id = shadowspill_current_task_id(runtime);
    const uint64_t allocation_ordinal =
        shadowspill_current_task_core_allocation_ordinal(runtime);
    const int allocation_is_scratch =
        shadowspill_current_task_allocation_is_scratch(runtime);
    const uint64_t task_invocation =
        shadowspill_current_task_invocation(runtime);
    ShadowSpillPlan *plan = shadowspill_current_plan(runtime);
    if (plan != NULL && plan->execution_pool != pool) {
        return SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    /*
     * Keep large, short-lived framework values at the high end of an
     * unsealed pool while small provider state grows from the low end.  A
     * provider cache retained by an isolated profiling task then occupies a
     * compact prefix instead of pinning a tiny range behind a multi-gigabyte
     * representative input.  Fixed-layout allocations bypass this policy.
     */
    const ShadowSpillMemoryPlacement dynamic_placement =
        bytes >= (UINT64_C(64) << 20U)
        ? SHADOWSPILL_MEMORY_BEST_FIT_HIGH
        : SHADOWSPILL_MEMORY_BEST_FIT_LOW;
    const ShadowSpillFixedPlacementDescription *fixed_placement =
        shadowspill_fixed_layout_find_placement(
            plan,
            SHADOWSPILL_FIXED_TASK_ALLOCATION,
            task_id,
            allocation_ordinal,
            SHADOWSPILL_RUNTIME_NO_ID
        );
    if (fixed_placement != NULL) {
        status = shadowspill_fixed_layout_wait_for_dependencies(
            plan,
            SHADOWSPILL_FIXED_TASK_ALLOCATION,
            task_id,
            allocation_ordinal,
            task_invocation,
            stream
        );
        if (status != SHADOWSPILL_RUNTIME_OK) {
            shadowspill_latch_task_failure(
                runtime,
                status,
                task_id,
                SHADOWSPILL_RUNTIME_NO_ID,
                SHADOWSPILL_RUNTIME_NO_ID,
                bytes
            );
            return status;
        }
    }
    shadowspill_memory_pool_lock_foreground(pool);
    status = shadowspill_current_status_locked(runtime);
    while (status == SHADOWSPILL_RUNTIME_OK) {
        ShadowSpillMemoryLease *record = NULL;
        if (fixed_placement != NULL) {
            status = shadowspill_create_fixed_execution_lease_locked(
                plan,
                fixed_placement,
                0,
                task_id,
                &record
            );
        }
        /*
         * A logically freed exact-size lease from this task is immediately
         * reusable on the same stream: stream order already places the new
         * consumer after the prior use.  Recycle it before consuming another
         * slab range so allocation-heavy compiled tasks retain caching-
         * allocator behavior without fixed offsets or backend events.
         */
        if (fixed_placement == NULL) {
            status = reuse_pending_allocation_locked(
                runtime, pool, bytes, alignment, stream, task_id, 1, &record
            );
        }
        if (fixed_placement == NULL &&
            status == SHADOWSPILL_RUNTIME_OK && record == NULL) {
            status = shadowspill_create_execution_lease_locked(
                runtime,
                pool,
                bytes,
                alignment,
                0,
                dynamic_placement,
                task_id,
                &record
            );
        }
        if (fixed_placement == NULL &&
            status == SHADOWSPILL_RUNTIME_OUT_OF_MEMORY) {
            status = reuse_pending_allocation_locked(
                runtime,
                pool,
                bytes,
                alignment,
                stream,
                task_id,
                0,
                &record
            );
            if (status == SHADOWSPILL_RUNTIME_OK && record == NULL) {
                status = SHADOWSPILL_RUNTIME_OUT_OF_MEMORY;
            }
        }
        if (status == SHADOWSPILL_RUNTIME_OK) {
            *allocation = (ShadowSpillAllocation){
                .pool_id = pool_id,
                .allocation_id = record->allocation_id,
                .generation = record->generation,
                .requested_bytes = record->requested_bytes,
                .charged_bytes = record->charged_bytes,
                .pointer = record->pointer,
            };
            record->origin_task_allocation_sequence =
                shadowspill_commit_task_allocation(
                    runtime, record->requested_bytes, record->charged_bytes
                );
            record->origin_task_allocation_ordinal = allocation_ordinal;
            record->origin_task_allocation_is_scratch =
                allocation_is_scratch ? 1U : 0U;
            record->origin_task_invocation = task_invocation;
            break;
        }
        if (status != SHADOWSPILL_RUNTIME_OUT_OF_MEMORY) {
            break;
        }
        if (!has_release_source(pool)) {
            shadowspill_latch_pool_failure_locked(
                runtime,
                pool,
                SHADOWSPILL_RUNTIME_NO_PROGRESS,
                SHADOWSPILL_RUNTIME_NO_ID,
                SHADOWSPILL_RUNTIME_NO_ID,
                bytes
            );
            status = SHADOWSPILL_RUNTIME_NO_PROGRESS;
            break;
        }
        const uint64_t capacity_epoch = atomic_load_explicit(
            &pool->capacity_epoch, memory_order_acquire
        );
        shadowspill_append_trace_event_locked(
            runtime,
            SHADOWSPILL_TRACE_ALLOCATION_WAIT_BEGIN,
            task_id,
            SHADOWSPILL_RUNTIME_NO_ID,
            SHADOWSPILL_RUNTIME_NO_ID,
            bytes,
            shadowspill_memory_pool_free_bytes_locked(pool),
            shadowspill_memory_pool_largest_free_locked(pool)
        );
        ++pool->blocked_allocators;
        shadowspill_memory_pool_unlock_foreground(pool);
        status = SHADOWSPILL_RUNTIME_OK;
        if (task_id != SHADOWSPILL_RUNTIME_NO_ID) {
            status = shadowspill_publish_task_retirement_event(
                runtime, task_id, stream
            );
            if (status != SHADOWSPILL_RUNTIME_OK) {
                shadowspill_latch_failure_locked(
                    runtime,
                    status,
                    SHADOWSPILL_RUNTIME_NO_ID,
                    SHADOWSPILL_RUNTIME_NO_ID,
                    bytes
                );
            }
        }
        while (status == SHADOWSPILL_RUNTIME_OK && atomic_load_explicit(
                   &pool->capacity_epoch, memory_order_acquire
               ) == capacity_epoch) {
            status = shadowspill_failure_status(runtime);
            if (status == SHADOWSPILL_RUNTIME_OK && atomic_load_explicit(
                    &runtime->worker_stop, memory_order_acquire
                ) != 0U) {
                status = SHADOWSPILL_RUNTIME_CLOSED;
            }
            shadowspill_cpu_relax();
        }
        shadowspill_memory_pool_lock_foreground(pool);
        --pool->blocked_allocators;
        shadowspill_append_trace_event_locked(
            runtime,
            SHADOWSPILL_TRACE_ALLOCATION_WAIT_END,
            task_id,
            SHADOWSPILL_RUNTIME_NO_ID,
            SHADOWSPILL_RUNTIME_NO_ID,
            bytes,
            shadowspill_memory_pool_free_bytes_locked(pool),
            shadowspill_memory_pool_largest_free_locked(pool)
        );
        if (status == SHADOWSPILL_RUNTIME_OK) {
            status = shadowspill_current_status_locked(runtime);
        }
    }
    shadowspill_memory_pool_unlock_foreground(pool);
    return status;
}

ShadowSpillRuntimeStatus shadowspill_memory_pool_allocation_for_pointer(
    ShadowSpillRuntime *runtime,
    uint32_t pool_id,
    const void *pointer,
    ShadowSpillAllocation *allocation
) {
    ShadowSpillMemoryPool *pool = shadowspill_runtime_pool(runtime, pool_id);
    if (pool == NULL || pointer == NULL || allocation == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    shadowspill_memory_pool_lock_foreground(pool);
    ShadowSpillMemoryLease *record =
        shadowspill_find_execution_lease_by_pointer(pool, pointer);
    if (record == NULL) {
        shadowspill_memory_pool_unlock_foreground(pool);
        return SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    *allocation = (ShadowSpillAllocation){
        .pool_id = pool_id,
        .allocation_id = record->allocation_id,
        .generation = record->generation,
        .requested_bytes = record->requested_bytes,
        .charged_bytes = record->charged_bytes,
        .pointer = record->pointer,
    };
    shadowspill_memory_pool_unlock_foreground(pool);
    return SHADOWSPILL_RUNTIME_OK;
}

static int append_stream(
    ShadowSpillMemoryLease *allocation,
    ShadowSpillBackendStream stream
) {
    for (ShadowSpillStreamRecord *item = allocation->streams; item != NULL;
         item = item->next) {
        if (stream_equal(item->stream, stream)) {
            return 0;
        }
    }
    ShadowSpillStreamRecord *created = calloc(1U, sizeof(*created));
    if (created == NULL) {
        return -1;
    }
    created->stream = stream;
    created->next = allocation->streams;
    allocation->streams = created;
    return 0;
}

ShadowSpillRuntimeStatus shadowspill_memory_pool_record_stream(
    ShadowSpillRuntime *runtime,
    uint32_t pool_id,
    uint64_t allocation_id,
    ShadowSpillBackendStream stream
) {
    ShadowSpillMemoryPool *pool = shadowspill_runtime_pool(runtime, pool_id);
    if (pool == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    shadowspill_memory_pool_lock_foreground(pool);
    ShadowSpillMemoryLease *allocation = shadowspill_find_execution_lease(
        pool, allocation_id
    );
    ShadowSpillRuntimeStatus status = SHADOWSPILL_RUNTIME_OK;
    if (allocation == NULL || allocation->logical_freed) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
    } else if (append_stream(allocation, stream) != 0) {
        status = SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    shadowspill_memory_pool_unlock_foreground(pool);
    return status;
}

static void destroy_event_list(
    ShadowSpillRuntime *runtime,
    ShadowSpillEventRecord *events
) {
    while (events != NULL) {
        ShadowSpillEventRecord *next = events->next;
        (void)shadowspill_event_lease_release(runtime, events->event);
        free(events);
        events = next;
    }
}

typedef struct ShadowSpillStreamSnapshot {
    ShadowSpillBackendStream *streams;
    uint64_t count;
} ShadowSpillStreamSnapshot;

static int snapshot_streams_locked(
    const ShadowSpillMemoryLease *allocation,
    ShadowSpillStreamSnapshot *snapshot
) {
    uint64_t count = 0U;
    for (const ShadowSpillStreamRecord *item = allocation->streams;
         item != NULL; item = item->next) {
        ++count;
    }
    ShadowSpillBackendStream *streams = count == 0U
        ? NULL
        : calloc((size_t)count, sizeof(*streams));
    if (count != 0U && streams == NULL) {
        return -1;
    }
    uint64_t index = 0U;
    for (const ShadowSpillStreamRecord *item = allocation->streams;
         item != NULL; item = item->next) {
        streams[index++] = item->stream;
    }
    *snapshot = (ShadowSpillStreamSnapshot){
        .streams = streams,
        .count = count,
    };
    return 0;
}

static ShadowSpillRuntimeStatus record_retirement_events(
    ShadowSpillRuntime *runtime,
    const ShadowSpillStreamSnapshot *snapshot,
    uint64_t allocation_id,
    ShadowSpillEventRecord **events
) {
    *events = NULL;
    for (uint64_t index = 0U; index < snapshot->count; ++index) {
        ShadowSpillEventRecord *event = calloc(1U, sizeof(*event));
        const ShadowSpillRuntimeStatus event_status = event == NULL
            ? SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE
            : shadowspill_event_lease_create_locked(runtime, &event->event);
        if (event_status != SHADOWSPILL_RUNTIME_OK ||
            runtime->synchronization.record_event(
                runtime->synchronization.context,
                event->event->event,
                snapshot->streams[index]
            ) != 0 || shadowspill_completion_submit(
                runtime,
                snapshot->streams[index],
                event->event,
                SHADOWSPILL_RUNTIME_NO_ID,
                allocation_id
            ) != SHADOWSPILL_RUNTIME_OK) {
            if (event != NULL && event->event != NULL) {
                (void)shadowspill_event_lease_release(runtime, event->event);
            }
            free(event);
            destroy_event_list(runtime, *events);
            *events = NULL;
            return event_status == SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE
                ? event_status
                : SHADOWSPILL_RUNTIME_BACKEND_FAILURE;
        }
        event->next = *events;
        *events = event;
    }
    return SHADOWSPILL_RUNTIME_OK;
}

ShadowSpillRuntimeStatus shadowspill_memory_pool_free(
    ShadowSpillRuntime *runtime,
    uint32_t pool_id,
    uint64_t allocation_id,
    ShadowSpillBackendStream stream
) {
    ShadowSpillMemoryPool *pool = shadowspill_runtime_pool(runtime, pool_id);
    if (pool == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    shadowspill_memory_pool_lock_foreground(pool);
    ShadowSpillMemoryLease *allocation = shadowspill_find_execution_lease(
        pool, allocation_id
    );
    ShadowSpillRuntimeStatus status = SHADOWSPILL_RUNTIME_OK;
    if (allocation == NULL) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
        goto done;
    }
    if (allocation->logical_freed) {
        if (allocation->ever_plan_owned && !allocation->framework_free_seen) {
            allocation->framework_free_seen = 1;
            unindex_allocation_pointer_locked(pool, allocation);
            unindex_allocation_id_locked(pool, allocation);
            shadowspill_memory_pool_try_recycle_lease_record_locked(allocation);
            goto done;
        }
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
        goto done;
    }
    if (allocation->plan_owned) {
        if (allocation->framework_free_seen) {
            status = SHADOWSPILL_RUNTIME_INVALID_STATE;
            goto done;
        }
        allocation->framework_free_seen = 1;
        goto done;
    }
    if (allocation->ever_plan_owned) {
        if (allocation->framework_free_seen) {
            status = SHADOWSPILL_RUNTIME_INVALID_STATE;
            goto done;
        }
        allocation->framework_free_seen = 1;
    }
    if (append_stream(allocation, stream) != 0) {
        status = SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
        goto done;
    }
    const uint64_t task_id = shadowspill_current_task_id(runtime);
    const int task_local_same_stream =
        task_id != SHADOWSPILL_RUNTIME_NO_ID &&
        allocation->streams != NULL &&
        allocation->streams->next == NULL &&
        stream_equal(allocation->streams->stream, stream);
    if (task_local_same_stream) {
        status = shadowspill_release_task_allocation(
            runtime,
            allocation->origin_task_id,
            allocation->origin_task_invocation,
            allocation->origin_task_allocation_ordinal,
            allocation->origin_task_allocation_is_scratch,
            allocation->requested_bytes,
            allocation->charged_bytes,
            allocation->alignment_bytes
        );
        if (status != SHADOWSPILL_RUNTIME_OK) {
            goto done;
        }
        allocation->release_task_id = task_id;
        allocation->logical_freed = 1;
        if (shadowspill_memory_pool_begin_retirement_locked(
                allocation, NULL, 1
            ) != 0) {
            status = SHADOWSPILL_RUNTIME_INVALID_STATE;
            goto done;
        }
        index_reusable_locked(pool, allocation);
        if (shadowspill_track_task_retirement(runtime, allocation) != 0) {
            status = SHADOWSPILL_RUNTIME_INVALID_STATE;
            goto done;
        }
        shadowspill_append_allocation_event_locked(
            runtime,
            allocation,
            SHADOWSPILL_ALLOCATION_LOGICAL_FREED,
            SHADOWSPILL_ALLOCATION_ANONYMOUS
        );
        (void)atomic_fetch_add_explicit(
            &runtime->pending_retirements, 1U, memory_order_release
        );
        (void)atomic_fetch_add_explicit(
            &pool->pending_retirements, 1U, memory_order_release
        );
        if (shadowspill_failure_status(runtime) != SHADOWSPILL_RUNTIME_OK) {
            status = shadowspill_failure_status(runtime);
        }
        goto done;
    }
    ShadowSpillStreamSnapshot snapshot = {0};
    if (snapshot_streams_locked(allocation, &snapshot) != 0) {
        status = SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
        goto done;
    }
    const uint64_t generation = allocation->generation;
    status = shadowspill_release_task_allocation(
        runtime,
        allocation->origin_task_id,
        allocation->origin_task_invocation,
        allocation->origin_task_allocation_ordinal,
        allocation->origin_task_allocation_is_scratch,
        allocation->requested_bytes,
        allocation->charged_bytes,
        allocation->alignment_bytes
    );
    if (status != SHADOWSPILL_RUNTIME_OK) {
        free(snapshot.streams);
        goto done;
    }
    allocation->release_task_id = task_id;
    allocation->logical_freed = 1;
    if (shadowspill_memory_pool_begin_retirement_locked(
            allocation, NULL, 0
        ) != 0) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
        goto done;
    }
    allocation->retirement_preparing = 1U;
    shadowspill_append_allocation_event_locked(
        runtime,
        allocation,
        SHADOWSPILL_ALLOCATION_LOGICAL_FREED,
        SHADOWSPILL_ALLOCATION_ANONYMOUS
    );
    (void)atomic_fetch_add_explicit(
        &runtime->pending_retirements, 1U, memory_order_release
    );
    (void)atomic_fetch_add_explicit(
        &pool->pending_retirements, 1U, memory_order_release
    );
    shadowspill_memory_pool_unlock_foreground(pool);

    ShadowSpillEventRecord *events = NULL;
    status = record_retirement_events(
        runtime, &snapshot, allocation_id, &events
    );
    free(snapshot.streams);

    shadowspill_memory_pool_lock_foreground(pool);
    allocation = shadowspill_find_execution_lease(pool, allocation_id);
    if (allocation == NULL || allocation->generation != generation ||
        !allocation->retirement_preparing) {
        destroy_event_list(runtime, events);
        if (status == SHADOWSPILL_RUNTIME_OK) {
            status = SHADOWSPILL_RUNTIME_INVALID_STATE;
        }
    } else {
        allocation->retirement_events = events;
        allocation->retirement_preparing = 0U;
        if (status == SHADOWSPILL_RUNTIME_OK) {
            index_reusable_locked(pool, allocation);
            status = shadowspill_retirement_enqueue_locked(
                runtime, allocation
            );
        }
    }
    if (status != SHADOWSPILL_RUNTIME_OK) {
        shadowspill_latch_failure_locked(
            runtime, status, SHADOWSPILL_RUNTIME_NO_ID, allocation_id, 0U
        );
    }
    if (shadowspill_failure_status(runtime) != SHADOWSPILL_RUNTIME_OK) {
        status = shadowspill_failure_status(runtime);
    }

done:
    shadowspill_memory_pool_unlock_foreground(pool);
    return status;
}

void shadowspill_finalize_aborted_task_retirements(
    ShadowSpillRuntime *runtime,
    uint64_t task_id
) {
    if (runtime == NULL || task_id == SHADOWSPILL_RUNTIME_NO_ID) {
        return;
    }
    ShadowSpillMemoryPool *pool = shadowspill_current_allocation_pool(runtime);
    if (pool == NULL) {
        return;
    }
    shadowspill_memory_pool_lock_foreground(pool);
    for (ShadowSpillMemoryLease *allocation =
             shadowspill_current_task_retirements(runtime);
         allocation != NULL;
         allocation = allocation->task_retirement_next) {
        if (!allocation->logical_freed || allocation->pointer == NULL ||
            allocation->release_task_id != task_id ||
            allocation->retirement_events != NULL ||
            allocation->retirement_event != NULL) {
            continue;
        }
        ShadowSpillEventRecord *events = NULL;
        int event_failure = 0;
        for (ShadowSpillStreamRecord *item = allocation->streams;
             item != NULL; item = item->next) {
            ShadowSpillEventRecord *event = calloc(1U, sizeof(*event));
            const ShadowSpillRuntimeStatus event_status = event == NULL
                ? SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE
                : shadowspill_event_lease_create_locked(runtime, &event->event);
            if (event_status != SHADOWSPILL_RUNTIME_OK ||
                runtime->synchronization.record_event(
                    runtime->synchronization.context,
                    event->event->event,
                    item->stream
                ) != 0 || shadowspill_completion_submit(
                    runtime,
                    item->stream,
                    event->event,
                    SHADOWSPILL_RUNTIME_NO_ID,
                    allocation->allocation_id
                ) != SHADOWSPILL_RUNTIME_OK) {
                if (event != NULL && event->event != NULL) {
                    (void)shadowspill_event_lease_release(runtime, event->event);
                }
                free(event);
                destroy_event_list(runtime, events);
                shadowspill_latch_failure_locked(
                    runtime,
                    SHADOWSPILL_RUNTIME_BACKEND_FAILURE,
                    SHADOWSPILL_RUNTIME_NO_ID,
                    allocation->allocation_id,
                    0U
                );
                event_failure = 1;
                break;
            }
            event->next = events;
            events = event;
        }
        if (event_failure) {
            break;
        }
        allocation->retirement_events = events;
        const ShadowSpillRuntimeStatus enqueue_status =
            shadowspill_retirement_enqueue_locked(runtime, allocation);
        if (enqueue_status != SHADOWSPILL_RUNTIME_OK) {
            shadowspill_latch_failure_locked(
                runtime,
                enqueue_status,
                SHADOWSPILL_RUNTIME_NO_ID,
                allocation->allocation_id,
                0U
            );
            break;
        }
    }
    shadowspill_memory_pool_unlock_foreground(pool);
}
