#ifndef SHADOWSPILL_PYTORCH_INTERNAL_H
#define SHADOWSPILL_PYTORCH_INTERNAL_H

/*
 * The adapter's state: one process-global instance, because PyTorch's
 * pluggable allocator calls malloc(size, device, stream) with no pointer of
 * the caller's to carry it in, so every entry point must find it by name.
 * Library-private: nothing outside libshadowspill_pytorch sees it, and the
 * C++ wrapper does not include this header (it holds C11 atomics).
 */

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>

#include <shadowspill/backend.h>
#include <shadowspill/pytorch_adapter.h>

/* A backend as loaded from its shared object: the table, and what releasing
   it takes. Nothing here is valid after unload. */
typedef struct ShadowSpillPytorchLoadedBackend {
    void *library;
    ShadowSpillBackend table;
    ShadowSpillBackendDestroy destroy;
} ShadowSpillPytorchLoadedBackend;

typedef struct ShadowSpillPytorchAdapterState {
    pthread_mutex_t mutex;
    ShadowSpillPytorchLoadedBackend backend;
    ShadowSpillRuntime *runtime;
    int32_t device_ordinal;
    uint32_t allocator_pool_id;
    uint64_t allocation_callbacks;
    uint64_t zero_size_allocation_callbacks;
    uint64_t free_callbacks;
    uint64_t record_stream_callbacks;
    uint64_t pointer_lookup_failures;
    uint64_t callback_failures;
    uint64_t physical_checks;
    uint64_t peak_process_physical_bytes;
    uint64_t observed_external_high_water_bytes;
    uint64_t physical_budget_sealed;
    _Atomic uint8_t profiler_annotations_enabled;
    _Atomic uint8_t shutdown_started;
    _Atomic uint64_t active_allocator_callbacks;
    _Atomic(ShadowSpillRuntime *) published_runtime;
    _Atomic int32_t published_device_ordinal;
    _Atomic uint32_t published_allocator_pool_id;
    char failure_task_label[SHADOWSPILL_RUNTIME_TRACE_LABEL_MAX_BYTES + 1U];
    ShadowSpillPytorchPhysicalAdmission admission;
    /*
     * `failure` is the first failure, which is what stopped the runtime and
     * what a caller asking "is this runtime usable" wants. `recent` is the
     * last one, which is what a caller asking "why did my call fail" wants.
     * They are different questions: a failure that was recovered from must
     * not describe a later one, and a first failure must not be overwritten
     * by the calls it subsequently causes to fail.
     */
    ShadowSpillPytorchAdapterFailure failure;
    ShadowSpillPytorchAdapterFailure recent;
    uint8_t recent_valid;
    uint8_t bootstrapped;
    uint8_t closed;
} ShadowSpillPytorchAdapterState;

extern ShadowSpillPytorchAdapterState adapter;

/* The bound runtime, or NULL before bootstrap and after close. Lock-free:
   bootstrap publishes it last and close clears it first, so a reader that
   sees it sees every field written before it. */
static inline ShadowSpillRuntime *shadowspill_pytorch_runtime(void) {
    return atomic_load_explicit(
        &adapter.published_runtime, memory_order_acquire
    );
}

/* The device the runtime was bound to; -1 before bootstrap. */
static inline int32_t shadowspill_pytorch_device_ordinal(void) {
    return atomic_load_explicit(
        &adapter.published_device_ordinal, memory_order_relaxed
    );
}

/* The pool PyTorch's allocations are served from. */
static inline uint32_t shadowspill_pytorch_allocator_pool_id(void) {
    return atomic_load_explicit(
        &adapter.published_allocator_pool_id, memory_order_relaxed
    );
}

/* The framework's stream handle as the token the backend's table accepts.
   The table is written once at bootstrap and cleared only after the runtime
   it served is gone, so reading it here needs no lock. */
static inline ShadowSpillBackendStream shadowspill_pytorch_stream(
    uint64_t framework_stream_handle
) {
    return adapter.backend.table.wrap_stream(
        adapter.backend.table.state, framework_stream_handle
    );
}

#endif
