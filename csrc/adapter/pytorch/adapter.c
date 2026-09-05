#include "internal.h"

ShadowSpillPytorchAdapterState adapter = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .device_ordinal = -1,
    .published_device_ordinal = -1,
    .published_allocator_pool_id = UINT32_MAX,
};

ShadowSpillStatus shadowspill_pytorch_runtime_handle(
    uintptr_t *runtime_handle
) {
    if (runtime_handle == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    ShadowSpillRuntime *runtime = shadowspill_pytorch_runtime();
    *runtime_handle = (uintptr_t)runtime;
    return runtime == NULL ? SHADOWSPILL_STATUS_CLOSED : SHADOWSPILL_STATUS_OK;
}

ShadowSpillStatus shadowspill_pytorch_adapter_capabilities(
    ShadowSpillPytorchAdapterCapabilities *capabilities
) {
    if (capabilities == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    *capabilities = (ShadowSpillPytorchAdapterCapabilities){
        .abi_version = SHADOWSPILL_PYTORCH_ADAPTER_ABI_VERSION,
        .runtime_abi_version = SHADOWSPILL_ABI_VERSION,
        .backend_abi_version = SHADOWSPILL_BACKEND_ABI_VERSION,
        .slab_memory_strategy = 1U,
        .record_stream_callback = 1U,
#ifdef SHADOWSPILL_PYTORCH_STORAGE_ADAPTER
        .storage_rebinding = 1U,
#else
        .storage_rebinding = 0U,
#endif
        .debug_task_dispatch_timing = 1U,
        .runtime_trace = 1U,
    };
    return SHADOWSPILL_STATUS_OK;
}
