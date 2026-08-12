#ifndef SHADOWSPILL_INTERNAL_FAILURE_STATE_H
#define SHADOWSPILL_INTERNAL_FAILURE_STATE_H

#include <shadowspill/runtime.h>

void shadowspill_latch_failure_locked(
    ShadowSpillRuntime *runtime,
    ShadowSpillRuntimeStatus status,
    uint64_t object_id,
    uint64_t allocation_id,
    uint64_t requested_bytes
);

ShadowSpillRuntimeStatus shadowspill_current_status_locked(
    ShadowSpillRuntime *runtime
);

#endif
