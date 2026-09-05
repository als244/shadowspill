#include "internal.h"

#include <stdio.h>

ShadowSpillStatus shadowspill_pytorch_allocation_scope_begin(
    uint64_t scope_id
) {
    if (shadowspill_pytorch_task_range_active()) {
        return SHADOWSPILL_STATUS_INVALID_STATE;
    }
    const uint64_t profiling_base = UINT64_C(1) << 62U;
    char range_name[192];
    (void)snprintf(
        range_name,
        sizeof(range_name),
        "shadowspill.pytorch.profiling.allocation_scope_%06llu",
        (unsigned long long)(
            scope_id >= profiling_base ? scope_id - profiling_base : scope_id
        )
    );
    shadowspill_pytorch_task_range_begin(range_name, NULL);
    ShadowSpillRuntime *runtime = shadowspill_pytorch_runtime();
    const ShadowSpillStatus status = runtime == NULL
        ? SHADOWSPILL_STATUS_CLOSED
        : shadowspill_allocation_scope_begin(
              runtime, shadowspill_pytorch_allocator_pool_id(), scope_id
          );
    if (status != SHADOWSPILL_STATUS_OK) {
        shadowspill_pytorch_task_range_end();
    }
    return status;
}

ShadowSpillStatus shadowspill_pytorch_allocation_scope_end(
    uint64_t scope_id,
    uintptr_t compute_stream_address
) {
    ShadowSpillRuntime *runtime = shadowspill_pytorch_runtime();
    const ShadowSpillStatus status = runtime == NULL
        ? SHADOWSPILL_STATUS_CLOSED
        : shadowspill_allocation_scope_end(
              runtime,
              scope_id,
              shadowspill_pytorch_stream(compute_stream_address)
          );
    shadowspill_pytorch_task_range_end();
    return status;
}

void shadowspill_pytorch_allocation_scope_abort(void) {
    ShadowSpillRuntime *runtime = shadowspill_pytorch_runtime();
    shadowspill_allocation_scope_abort(runtime);
    shadowspill_pytorch_task_range_end();
}
