#include "internal.h"

#include "../internal.h"

ShadowSpillStatus shadowspill_pytorch_register_object(
    uint32_t pool_id,
    uint64_t object_id,
    uint64_t size_bytes,
    uint8_t retain_spill_copy,
    uint64_t source_address
) {
    if (retain_spill_copy > 1U ||
        (size_bytes != 0U && source_address == 0U)) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    if (runtime == NULL) {
        return SHADOWSPILL_STATUS_CLOSED;
    }
    const ShadowSpillObjectDescription description = {
        .object_id = object_id,
        .size_bytes = size_bytes,
        .retain_spill_copy = retain_spill_copy,
        .initial_pool_id = pool_id,
        .initially_resident = 1U,
    };
    ShadowSpillStatus status = shadowspill_register_object(
        runtime, &description
    );
    if (status != SHADOWSPILL_STATUS_OK) {
        return status;
    }
    return shadowspill_write_object(
        runtime,
        object_id,
        pool_id,
        (const void *)(uintptr_t)source_address,
        size_bytes
    );
}

ShadowSpillStatus shadowspill_pytorch_register_placeholder_object(
    uint64_t object_id,
    uint64_t size_bytes,
    uint8_t retain_spill_copy
) {
    if (retain_spill_copy > 1U) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    if (runtime == NULL) {
        return SHADOWSPILL_STATUS_CLOSED;
    }
    const ShadowSpillObjectDescription description = {
        .object_id = object_id,
        .size_bytes = size_bytes,
        .retain_spill_copy = retain_spill_copy,
        .initially_resident = 0U,
    };
    return shadowspill_register_object(runtime, &description);
}

ShadowSpillStatus shadowspill_pytorch_validate_object_binding(
    uint32_t pool_id,
    uint64_t object_id,
    uint64_t address,
    uint64_t size_bytes
) {
    if (address == 0U && size_bytes != 0U) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
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
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    if (runtime == NULL) {
        return SHADOWSPILL_STATUS_CLOSED;
    }
    return acquisition_handle == 0U
        ? SHADOWSPILL_STATUS_INVALID_ARGUMENT
        : shadowspill_acquire_objects_handle(
            runtime,
            (const ShadowSpillObjectAcquisitionHandle *)acquisition_handle,
            adapter_stream(consumer_stream_address),
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
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    return runtime == NULL
        ? SHADOWSPILL_STATUS_CLOSED
        : shadowspill_transfer_acquired_object_to_caller(
              runtime,
              (const ShadowSpillObjectAcquisitionHandle *)acquisition_handle,
              object_ordinal,
              adapter_stream(consumer_stream),
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
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    return runtime == NULL
        ? SHADOWSPILL_STATUS_CLOSED
        : shadowspill_memory_pool_free(
              runtime,
              bound_allocator_pool_id(),
              allocation_id,
              adapter_stream(stream)
          );
}

ShadowSpillStatus
shadowspill_pytorch_validate_task_replacement_binding(
    uintptr_t task_handle,
    uint32_t publication_ordinal,
    uint64_t retired_address,
    uint64_t successor_address
) {
    if (task_handle == 0U || retired_address == 0U ||
        successor_address == 0U) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = bound_runtime(&device_ordinal);
    (void)device_ordinal;
    return runtime == NULL
        ? SHADOWSPILL_STATUS_CLOSED
        : shadowspill_task_validate_replacement_binding(
              runtime,
              (const ShadowSpillTaskHandle *)task_handle,
              publication_ordinal,
              (const void *)(uintptr_t)retired_address,
              (const void *)(uintptr_t)successor_address
          );
}
