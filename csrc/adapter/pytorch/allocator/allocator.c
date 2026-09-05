#include "internal.h"

#include "../internal.h"
#include "../failure/internal.h"

#include <pthread.h>
#include <stdatomic.h>

static ShadowSpillRuntime *acquire_allocator_callback_runtime(
    int32_t *device_ordinal
) {
    if (atomic_load_explicit(
            &adapter.shutdown_started, memory_order_acquire
        ) != 0U) {
        *device_ordinal = shadowspill_pytorch_device_ordinal();
        return NULL;
    }
    (void)atomic_fetch_add_explicit(
        &adapter.active_allocator_callbacks, 1U, memory_order_acq_rel
    );
    if (atomic_load_explicit(
            &adapter.shutdown_started, memory_order_acquire
        ) != 0U) {
        (void)atomic_fetch_sub_explicit(
            &adapter.active_allocator_callbacks, 1U, memory_order_release
        );
        *device_ordinal = shadowspill_pytorch_device_ordinal();
        return NULL;
    }
    *device_ordinal = shadowspill_pytorch_device_ordinal();
    ShadowSpillRuntime *runtime = shadowspill_pytorch_runtime();
    if (runtime == NULL) {
        (void)atomic_fetch_sub_explicit(
            &adapter.active_allocator_callbacks, 1U, memory_order_release
        );
    }
    return runtime;
}

static void release_allocator_callback_runtime(void) {
    (void)atomic_fetch_sub_explicit(
        &adapter.active_allocator_callbacks, 1U, memory_order_release
    );
}

ShadowSpillStatus shadowspill_pytorch_allocation_for_pointer(
    uint64_t address,
    ShadowSpillAllocation *allocation
) {
    if (address == 0U || allocation == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    ShadowSpillRuntime *runtime = shadowspill_pytorch_runtime();
    return runtime == NULL
        ? SHADOWSPILL_STATUS_CLOSED
        : shadowspill_memory_pool_allocation_for_pointer(
              runtime,
              shadowspill_pytorch_allocator_pool_id(),
              (const void *)(uintptr_t)address,
              allocation
          );
}

void *shadowspill_pytorch_backend_malloc_impl(
    ptrdiff_t bytes,
    int32_t device_ordinal,
    void *stream
) {
    const ShadowSpillProfilerRange range =
        shadowspill_pytorch_profile_range_begin("shadowspill.runtime.allocate");
    pthread_mutex_lock(&adapter.mutex);
    ++adapter.allocation_callbacks;
    if (bytes == 0) {
        ++adapter.zero_size_allocation_callbacks;
    }
    pthread_mutex_unlock(&adapter.mutex);
    int32_t expected_device;
    ShadowSpillRuntime *runtime = acquire_allocator_callback_runtime(
        &expected_device
    );
    if (bytes == 0 && runtime == NULL) {
        shadowspill_pytorch_profile_range_end(range);
        return NULL;
    }
    if (bytes == 0 && runtime != NULL && device_ordinal == expected_device) {
        shadowspill_pytorch_profile_range_end(range);
        release_allocator_callback_runtime();
        return NULL;
    }
    if (runtime == NULL || bytes < 0 || device_ordinal != expected_device) {
        shadowspill_pytorch_latch_failure(
            runtime == NULL ? SHADOWSPILL_STATUS_CLOSED
                            : SHADOWSPILL_STATUS_INVALID_ARGUMENT,
            device_ordinal,
            NULL,
            bytes < 0 ? 0U : (uint64_t)bytes
        );
        shadowspill_pytorch_profile_range_end(range);
        if (runtime != NULL) {
            release_allocator_callback_runtime();
        }
        return NULL;
    }
    ShadowSpillAllocation allocation = {0};
    ShadowSpillStatus status = shadowspill_memory_pool_allocate(
        runtime,
        shadowspill_pytorch_allocator_pool_id(),
        (uint64_t)bytes,
        256U,
        shadowspill_pytorch_stream((uintptr_t)stream),
        &allocation
    );
    if (status != SHADOWSPILL_STATUS_OK) {
        shadowspill_pytorch_latch_failure(status, device_ordinal, NULL, (uint64_t)bytes);
        shadowspill_pytorch_profile_range_end(range);
        release_allocator_callback_runtime();
        return NULL;
    }
    shadowspill_pytorch_profile_range_end(range);
    release_allocator_callback_runtime();
    return allocation.pointer;
}

void shadowspill_pytorch_backend_free(
    void *address,
    size_t bytes,
    int32_t device_ordinal,
    void *stream
) {
    pthread_mutex_lock(&adapter.mutex);
    ++adapter.free_callbacks;
    pthread_mutex_unlock(&adapter.mutex);
    if (address == NULL) {
        return;
    }
    int32_t expected_device;
    ShadowSpillRuntime *runtime = acquire_allocator_callback_runtime(
        &expected_device
    );
    if (runtime == NULL) {
        return;
    }
    if (device_ordinal != expected_device) {
        shadowspill_pytorch_latch_failure(
            SHADOWSPILL_STATUS_INVALID_ARGUMENT,
            device_ordinal,
            address,
            (uint64_t)bytes
        );
        release_allocator_callback_runtime();
        return;
    }
    ShadowSpillAllocation allocation = {0};
    ShadowSpillStatus status =
        shadowspill_memory_pool_allocation_for_pointer(
            runtime, shadowspill_pytorch_allocator_pool_id(), address, &allocation
    );
    if (status != SHADOWSPILL_STATUS_OK) {
        pthread_mutex_lock(&adapter.mutex);
        ++adapter.pointer_lookup_failures;
        pthread_mutex_unlock(&adapter.mutex);
        shadowspill_pytorch_latch_failure(status, device_ordinal, address, (uint64_t)bytes);
        release_allocator_callback_runtime();
        return;
    }
    status = shadowspill_memory_pool_free(
        runtime,
        shadowspill_pytorch_allocator_pool_id(),
        allocation.allocation_id,
        shadowspill_pytorch_stream((uintptr_t)stream)
    );
    if (status != SHADOWSPILL_STATUS_OK) {
        shadowspill_pytorch_latch_failure(status, device_ordinal, address, (uint64_t)bytes);
    }
    release_allocator_callback_runtime();
}

void shadowspill_pytorch_backend_record_stream(void *address, void *stream) {
    pthread_mutex_lock(&adapter.mutex);
    ++adapter.record_stream_callbacks;
    pthread_mutex_unlock(&adapter.mutex);
    if (address == NULL) {
        return;
    }
    int32_t device_ordinal;
    ShadowSpillRuntime *runtime = acquire_allocator_callback_runtime(
        &device_ordinal
    );
    if (runtime == NULL) {
        return;
    }
    ShadowSpillAllocation allocation = {0};
    ShadowSpillStatus status =
        shadowspill_memory_pool_allocation_for_pointer(
            runtime, shadowspill_pytorch_allocator_pool_id(), address, &allocation
    );
    if (status != SHADOWSPILL_STATUS_OK) {
        pthread_mutex_lock(&adapter.mutex);
        ++adapter.pointer_lookup_failures;
        pthread_mutex_unlock(&adapter.mutex);
        shadowspill_pytorch_latch_failure(status, device_ordinal, address, 0U);
        release_allocator_callback_runtime();
        return;
    }
    status = shadowspill_memory_pool_record_stream(
        runtime,
        shadowspill_pytorch_allocator_pool_id(),
        allocation.allocation_id,
        shadowspill_pytorch_stream((uintptr_t)stream)
    );
    if (status != SHADOWSPILL_STATUS_OK) {
        shadowspill_pytorch_latch_failure(status, device_ordinal, address, 0U);
    }
    release_allocator_callback_runtime();
}
