#ifndef SHADOWSPILL_INTERNAL_EVENT_POOL_H
#define SHADOWSPILL_INTERNAL_EVENT_POOL_H

#include <shadowspill/runtime.h>

typedef struct ShadowSpillEventLease ShadowSpillEventLease;
typedef struct ShadowSpillEventPool ShadowSpillEventPool;

int shadowspill_event_pool_initialize(ShadowSpillEventPool *pool);
void shadowspill_event_pool_destroy(
    ShadowSpillRuntime *runtime,
    ShadowSpillEventPool *pool
);
ShadowSpillRuntimeStatus shadowspill_event_pool_reserve(
    ShadowSpillEventPool *pool,
    uint64_t minimum_free_leases
);

ShadowSpillRuntimeStatus shadowspill_event_lease_create_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillEventLease **lease
);
void shadowspill_event_lease_retain(ShadowSpillEventLease *lease);
int shadowspill_event_lease_is_complete(const ShadowSpillEventLease *lease);
int shadowspill_event_lease_release(
    ShadowSpillRuntime *runtime,
    ShadowSpillEventLease *lease
);
int shadowspill_event_lease_query(
    ShadowSpillRuntime *runtime,
    ShadowSpillEventLease *lease,
    int *complete
);

#endif
