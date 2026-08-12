#ifndef SHADOWSPILL_INTERNAL_TASK_BOUNDARIES_H
#define SHADOWSPILL_INTERNAL_TASK_BOUNDARIES_H

#include <shadowspill/runtime.h>

ShadowSpillRuntimeStatus shadowspill_before_task_legacy(
    ShadowSpillRuntime *runtime,
    uint64_t task_id,
    ShadowSpillBackendStream compute_stream,
    const uint64_t *input_object_ids,
    uint32_t input_count,
    ShadowSpillObjectBinding *bindings,
    uint32_t binding_capacity
);

ShadowSpillRuntimeStatus shadowspill_after_task_legacy(
    ShadowSpillRuntime *runtime,
    uint64_t task_id,
    ShadowSpillBackendStream compute_stream,
    const ShadowSpillObjectUpdate *updates,
    uint32_t update_count,
    const ShadowSpillRuntimeAction *actions,
    uint32_t action_count
);

#endif
