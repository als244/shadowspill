#ifndef SHADOWSPILL_INTERNAL_LIFECYCLE_H
#define SHADOWSPILL_INTERNAL_LIFECYCLE_H

#include <shadowspill/runtime.h>

ShadowSpillRuntimeStatus shadowspill_runtime_create_legacy(
    const ShadowSpillRuntimeConfig *config,
    ShadowSpillRuntime **runtime
);
ShadowSpillRuntimeStatus shadowspill_runtime_wait_idle_legacy(
    ShadowSpillRuntime *runtime
);
ShadowSpillRuntimeStatus shadowspill_runtime_resize_host_arena_legacy(
    ShadowSpillRuntime *runtime,
    uint64_t host_arena_bytes
);
ShadowSpillRuntimeStatus shadowspill_runtime_close_legacy(
    ShadowSpillRuntime *runtime
);
void shadowspill_runtime_destroy_legacy(ShadowSpillRuntime *runtime);

#endif
