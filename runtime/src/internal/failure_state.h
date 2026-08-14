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

/* Publish a failure whose causal task is carried by asynchronous work. */
void shadowspill_latch_task_failure(
    ShadowSpillRuntime *runtime,
    ShadowSpillRuntimeStatus status,
    uint64_t task_id,
    uint64_t object_id,
    uint64_t allocation_id,
    uint64_t requested_bytes
);

void shadowspill_latch_task_envelope_failure(
    ShadowSpillRuntime *runtime,
    uint64_t requested_bytes,
    uint64_t charged_bytes,
    uint64_t live_requested_bytes,
    uint64_t live_charged_bytes,
    uint64_t live_requested_limit_bytes,
    uint64_t live_charged_limit_bytes,
    uint64_t maximum_requested_allocation_bytes,
    uint64_t maximum_charged_allocation_bytes
);

ShadowSpillRuntimeStatus shadowspill_current_status_locked(
    ShadowSpillRuntime *runtime
);

ShadowSpillRuntimeStatus shadowspill_failure_status(
    const ShadowSpillRuntime *runtime
);

#endif
