#include "internal.h"
#include "internal/task_boundaries.h"

ShadowSpillRuntimeStatus shadowspill_before_task(
    ShadowSpillRuntime *runtime,
    uint64_t task_id,
    ShadowSpillBackendStream compute_stream,
    const uint64_t *input_object_ids,
    uint32_t input_count,
    ShadowSpillObjectBinding *bindings,
    uint32_t binding_capacity
) {
    return shadowspill_before_task_legacy(
        runtime,
        task_id,
        compute_stream,
        input_object_ids,
        input_count,
        bindings,
        binding_capacity
    );
}

ShadowSpillRuntimeStatus shadowspill_after_task(
    ShadowSpillRuntime *runtime,
    uint64_t task_id,
    ShadowSpillBackendStream compute_stream,
    const ShadowSpillObjectUpdate *updates,
    uint32_t update_count,
    const ShadowSpillRuntimeAction *actions,
    uint32_t action_count
) {
    return shadowspill_after_task_legacy(
        runtime,
        task_id,
        compute_stream,
        updates,
        update_count,
        actions,
        action_count
    );
}
