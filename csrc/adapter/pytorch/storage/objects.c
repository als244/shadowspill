#include "internal.h"

#include "../internal.h"

ShadowSpillStatus shadowspill_pytorch_validate_object_binding(
    uint32_t pool_id,
    uint64_t object_id,
    uint64_t address,
    uint64_t size_bytes
) {
    if (address == 0U && size_bytes != 0U) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    ShadowSpillRuntime *runtime = shadowspill_pytorch_runtime();
    if (runtime == NULL) {
        return SHADOWSPILL_STATUS_CLOSED;
    }
    ShadowSpillObjectLocationSnapshot snapshot = {0};
    ShadowSpillStatus status = shadowspill_object_location_snapshot(
        runtime, object_id, pool_id, &snapshot
    );
    if (status != SHADOWSPILL_STATUS_OK) {
        return status;
    }
    return snapshot.size_bytes == size_bytes && snapshot.has_lease &&
            snapshot.current &&
            snapshot.pointer == (void *)(uintptr_t)address
        ? SHADOWSPILL_STATUS_OK
        : SHADOWSPILL_STATUS_INVALID_STATE;
}

ShadowSpillStatus shadowspill_pytorch_acquire_objects_handle(
    uintptr_t acquisition_handle,
    uintptr_t consumer_stream_address,
    ShadowSpillObjectBinding *bindings,
    uint32_t binding_capacity
) {
    ShadowSpillRuntime *runtime = shadowspill_pytorch_runtime();
    if (runtime == NULL) {
        return SHADOWSPILL_STATUS_CLOSED;
    }
    return acquisition_handle == 0U
        ? SHADOWSPILL_STATUS_INVALID_ARGUMENT
        : shadowspill_acquire_objects_handle(
            runtime,
            (const ShadowSpillObjectAcquisitionHandle *)acquisition_handle,
            shadowspill_pytorch_stream(consumer_stream_address),
            bindings,
            binding_capacity
        );
}

ShadowSpillStatus
shadowspill_pytorch_transfer_acquired_object_to_caller(
    uintptr_t acquisition_handle,
    uint32_t object_ordinal,
    uintptr_t consumer_stream,
    uint64_t expected_address,
    uint64_t expected_generation,
    uint64_t expected_allocation_id,
    ShadowSpillAllocation *allocation
) {
    ShadowSpillRuntime *runtime = shadowspill_pytorch_runtime();
    return runtime == NULL
        ? SHADOWSPILL_STATUS_CLOSED
        : shadowspill_transfer_acquired_object_to_caller(
              runtime,
              (const ShadowSpillObjectAcquisitionHandle *)acquisition_handle,
              object_ordinal,
              shadowspill_pytorch_stream(consumer_stream),
              (const void *)(uintptr_t)expected_address,
              expected_generation,
              expected_allocation_id,
              allocation
          );
}

ShadowSpillStatus shadowspill_pytorch_release_caller_allocation(
    uint64_t allocation_id,
    uintptr_t stream
) {
    ShadowSpillRuntime *runtime = shadowspill_pytorch_runtime();
    return runtime == NULL
        ? SHADOWSPILL_STATUS_CLOSED
        : shadowspill_memory_pool_free(
              runtime,
              shadowspill_pytorch_allocator_pool_id(),
              allocation_id,
              shadowspill_pytorch_stream(stream)
          );
}
