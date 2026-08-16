#ifndef SHADOWSPILL_INTERNAL_EVENT_POOL_H
#define SHADOWSPILL_INTERNAL_EVENT_POOL_H

#include <shadowspill/runtime.h>

typedef struct ShadowSpillEventLease ShadowSpillEventLease;

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
