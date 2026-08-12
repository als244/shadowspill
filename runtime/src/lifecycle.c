#include "internal.h"
#include "internal/lifecycle.h"

ShadowSpillRuntimeStatus shadowspill_runtime_create(
    const ShadowSpillRuntimeConfig *config,
    ShadowSpillRuntime **runtime
) {
    return shadowspill_runtime_create_legacy(config, runtime);
}

ShadowSpillRuntimeStatus shadowspill_runtime_wait_idle(
    ShadowSpillRuntime *runtime
) {
    return shadowspill_runtime_wait_idle_legacy(runtime);
}

ShadowSpillRuntimeStatus shadowspill_runtime_resize_spill_pool(
    ShadowSpillRuntime *runtime,
    uint64_t spill_pool_bytes
) {
    return shadowspill_runtime_resize_spill_pool_legacy(
        runtime, spill_pool_bytes
    );
}

ShadowSpillRuntimeStatus shadowspill_runtime_close(ShadowSpillRuntime *runtime) {
    return shadowspill_runtime_close_legacy(runtime);
}

void shadowspill_runtime_destroy(ShadowSpillRuntime *runtime) {
    shadowspill_runtime_destroy_legacy(runtime);
}
