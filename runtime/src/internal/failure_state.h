#ifndef SHADOWSPILL_INTERNAL_FAILURE_STATE_H
#define SHADOWSPILL_INTERNAL_FAILURE_STATE_H

#include <shadowspill/runtime.h>

/* Thread-safe first-cause publication; callers need not hold another lock. */
void shadowspill_latch_failure_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillRuntimeStatus status,
    uint64_t object_id,
    uint64_t allocation_id,
    uint64_t requested_bytes
);

void shadowspill_latch_placement_failure(
    ShadowSpillRuntime *runtime,
    uint64_t requested_bytes,
    uint64_t allocation_ordinal,
    uint64_t expected_allocation_ordinal,
    uint64_t expected_requested_bytes
);

ShadowSpillRuntimeStatus shadowspill_current_status_locked(
    ShadowSpillRuntime *runtime
);

ShadowSpillRuntimeStatus shadowspill_failure_status(
    const ShadowSpillRuntime *runtime
);

#endif
