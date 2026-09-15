#ifndef SHADOWSPILL_ALLOCATIONS_INTERNAL_H
#define SHADOWSPILL_ALLOCATIONS_INTERNAL_H

/*
 * What a pool's allocations are, and what happens to one.
 *
 * An allocation is reached through the index, owned through a lease record,
 * created or reused as a lease, and given back through free. One file per
 * stage of that life, in that order. Every function whose name ends in
 * ``_locked`` is called with the pool's lock held and returns with it held.
 */

#include "../../internal.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int shadowspill_allocations_stream_equal(
    ShadowSpillBackendStream left,
    ShadowSpillBackendStream right
);

void shadowspill_publish_pool_geometry_locked(ShadowSpillMemoryPool *pool);

uint64_t shadowspill_allocations_mix_index(uint64_t value, uint64_t bucket_count);

void shadowspill_allocations_index_allocation_id_locked(
    ShadowSpillMemoryPool *pool,
    ShadowSpillMemoryLease *allocation
);

void shadowspill_allocations_unindex_allocation_id_locked(
    ShadowSpillMemoryPool *pool,
    ShadowSpillMemoryLease *allocation
);

void shadowspill_allocations_index_allocation_pointer_locked(
    ShadowSpillMemoryPool *pool,
    ShadowSpillMemoryLease *allocation
);

void shadowspill_allocations_unindex_allocation_pointer_locked(
    ShadowSpillMemoryPool *pool,
    ShadowSpillMemoryLease *allocation
);

void shadowspill_allocations_index_reusable_locked(
    ShadowSpillMemoryPool *pool,
    ShadowSpillMemoryLease *allocation
);

void shadowspill_allocations_unindex_reusable_locked(
    ShadowSpillMemoryPool *pool,
    ShadowSpillMemoryLease *allocation
);

void shadowspill_allocations_activate_allocation_locked(
    ShadowSpillMemoryPool *pool,
    ShadowSpillMemoryLease *allocation
);

void shadowspill_allocations_deactivate_allocation_locked(
    ShadowSpillMemoryLease *allocation
);

ShadowSpillMemoryLease *shadowspill_find_lease(
    ShadowSpillMemoryPool *pool,
    uint64_t allocation_id
);

ShadowSpillMemoryLease *shadowspill_find_lease_by_pointer(
    ShadowSpillMemoryPool *pool,
    const void *pointer
);

ShadowSpillMemoryLease *shadowspill_memory_pool_acquire_lease_record_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryPool *pool,
    ShadowSpillAllocationOrigin origin
);

void shadowspill_memory_pool_try_recycle_lease_record_locked(
    ShadowSpillMemoryLease *record
);

void shadowspill_memory_lease_retain(ShadowSpillMemoryLease *lease);

void shadowspill_memory_lease_release(ShadowSpillMemoryLease *lease);

ShadowSpillStatus shadowspill_create_fixed_execution_lease_locked(
    ShadowSpillPlan *plan,
    const ShadowSpillFixedPlacementDescription *placement,
    int plan_owned,
    ShadowSpillAllocationOrigin origin,
    ShadowSpillMemoryLease **record
);

ShadowSpillStatus shadowspill_create_lease_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryPool *pool,
    uint64_t bytes,
    uint64_t alignment,
    int plan_owned,
    ShadowSpillMemoryPlacement placement,
    ShadowSpillAllocationOrigin origin,
    ShadowSpillMemoryLease **record
);

ShadowSpillStatus shadowspill_allocations_reuse_pending_allocation_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryPool *pool,
    uint64_t bytes,
    uint64_t alignment,
    ShadowSpillBackendStream stream,
    ShadowSpillAllocationOrigin origin,
    int exact_task_local_only,
    ShadowSpillMemoryLease **record
);

int shadowspill_allocations_append_lease_use_locked(
    ShadowSpillMemoryLease *allocation,
    ShadowSpillBackendStream stream
);

ShadowSpillStatus shadowspill_publish_task_retirement_event(
    ShadowSpillRuntime *runtime,
    uint64_t task_id,
    ShadowSpillBackendStream stream
);

#endif  /* SHADOWSPILL_ALLOCATIONS_INTERNAL_H */
