/* Giving a pending allocation to the next caller that fits it. */
#include "internal.h"

ShadowSpillStatus shadowspill_allocations_reuse_pending_allocation_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryPool *pool,
    uint64_t bytes,
    uint64_t alignment,
    ShadowSpillBackendStream stream,
    ShadowSpillAllocationOrigin origin,
    int exact_task_local_only,
    ShadowSpillMemoryLease **record
) {
    const uint64_t required = bytes == 0U ? 1U : bytes;
    if (alignment < pool->minimum_alignment) {
        alignment = pool->minimum_alignment;
    }
    const uint64_t reusable_bucket = shadowspill_allocations_mix_index(
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
        const int task_local = candidate->retirement_requirements == NULL &&
            candidate->retirement_event == NULL &&
            candidate->release_task_id == origin.task_id &&
            origin.task_id != SHADOWSPILL_RUNTIME_NO_ID;
        if (exact_task_local_only && !task_local) {
            continue;
        }
        if (!task_local && candidate->retirement_requirements == NULL &&
            candidate->retirement_event == NULL) {
            continue;
        }
        int stream_compatible = 1;
        for (ShadowSpillLeaseUseRecord *used = candidate->uses;
             used != NULL; used = used->next) {
            if (!shadowspill_allocations_stream_equal(used->stream, stream)) {
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
        return SHADOWSPILL_STATUS_OK;
    }
    ShadowSpillMemoryLease *split = NULL;
    if (selected->charged_bytes > required) {
        split = shadowspill_memory_pool_acquire_lease_record_locked(
            runtime, pool, origin
        );
        if (split == NULL) {
            return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
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
            return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
        }
    }
    /*
     * Every recorded use was on ``stream`` (checked above). Execution-stream order
     * already retires the old use before later work that consumes the reused
     * range, so adding a wait for an event recorded on the same stream is both
     * redundant and an unnecessary driver call in the allocator hot path.
     */
    shadowspill_allocations_unindex_reusable_locked(pool, selected);
    if (split != NULL) {
        shadowspill_allocations_unindex_allocation_pointer_locked(pool, selected);
        pool->requested_allocated_bytes -= selected->requested_bytes;
        selected->requested_bytes = 0U;
        selected->charged_bytes -= required;
        selected->offset += required;
        selected->pointer =
            shadowspill_memory_pool_pointer(
                pool, selected->offset
            );
        shadowspill_allocations_index_allocation_pointer_locked(pool, selected);
        shadowspill_allocations_index_reusable_locked(pool, selected);

        split->state = SHADOWSPILL_LEASE_IN_USE;
        shadowspill_allocations_activate_allocation_locked(pool, split);
        shadowspill_allocations_index_allocation_id_locked(pool, split);
        shadowspill_allocations_index_allocation_pointer_locked(pool, split);
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
    ShadowSpillLeaseUseRecord *old_uses = selected->uses;
    const int requirements_owned_by_queue =
        selected->retirement_requirements != NULL;
    selected->retirement_requirements = NULL;
    selected->uses = NULL;
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
    if (!requirements_owned_by_queue &&
        shadowspill_memory_pool_release_use_records_locked(
            pool, old_uses
        ) != 0) {
        return SHADOWSPILL_STATUS_INVALID_STATE;
    }
    shadowspill_allocations_unindex_allocation_id_locked(pool, selected);
    selected->allocation_id = atomic_fetch_add_explicit(
        &runtime->next_allocation_id, 1U, memory_order_relaxed
    );
    shadowspill_allocations_index_allocation_id_locked(pool, selected);
    selected->generation = atomic_fetch_add_explicit(
        &runtime->next_generation, 1U, memory_order_relaxed
    );
    selected->requested_bytes = bytes;
    selected->alignment_bytes = alignment;
    selected->origin_plan_id = origin.plan_id;
    selected->origin_task_id = origin.task_id;
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
