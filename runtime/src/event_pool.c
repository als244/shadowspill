#include "internal.h"

#include <stdlib.h>

ShadowSpillRuntimeStatus shadowspill_event_lease_create_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillEventLease **output
) {
    if (runtime == NULL || output == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    *output = NULL;
    ShadowSpillEventLease *lease = calloc(1U, sizeof(*lease));
    if (lease == NULL) {
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    if (runtime->backend.create_event(
            runtime->backend.context, &lease->event
        ) != 0) {
        free(lease);
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

int shadowspill_event_lease_release(
    ShadowSpillRuntime *runtime,
    ShadowSpillEventLease *lease
) {
    if (lease == NULL || atomic_fetch_sub_explicit(
            &lease->references, 1U, memory_order_acq_rel
        ) != 1U) {
        return 0;
    }
    const int status = runtime->backend.destroy_event(
        runtime->backend.context, lease->event
    );
    free(lease);
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
    if (runtime->backend.query_event(
            runtime->backend.context, lease->event, complete
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
