#include "internal.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>

static inline void adapter_cpu_relax(void) {
#if defined(__x86_64__) || defined(__i386__)
    __asm__ __volatile__("pause" ::: "memory");
#elif defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__("yield" ::: "memory");
#else
    atomic_signal_fence(memory_order_seq_cst);
#endif
}

/* Refuse new callbacks and wait for those in flight to leave the runtime.
   Hands back the runtime to close, or NULL with the status to return: OK
   when already closed, INVALID_STATE while another close is under way,
   CLOSED when nothing was ever bound. */
static ShadowSpillStatus begin_shutdown(ShadowSpillRuntime **runtime) {
    ShadowSpillStatus status = SHADOWSPILL_STATUS_OK;
    *runtime = NULL;
    pthread_mutex_lock(&adapter.mutex);
    if (adapter.closed) {
        status = SHADOWSPILL_STATUS_OK;
    } else if (atomic_load_explicit(
                   &adapter.shutdown_started, memory_order_acquire
               ) != 0U) {
        status = SHADOWSPILL_STATUS_INVALID_STATE;
    } else if (adapter.runtime == NULL) {
        status = SHADOWSPILL_STATUS_CLOSED;
    } else {
        *runtime = adapter.runtime;
        atomic_store_explicit(
            &adapter.shutdown_started, 1U, memory_order_release
        );
    }
    pthread_mutex_unlock(&adapter.mutex);
    if (*runtime == NULL) {
        return status;
    }
    while (atomic_load_explicit(
               &adapter.active_allocator_callbacks, memory_order_acquire
           ) != 0U) {
        adapter_cpu_relax();
    }
    return SHADOWSPILL_STATUS_OK;
}

/* A refused close lets callbacks run again. */
static void resume(void) {
    atomic_store_explicit(
        &adapter.shutdown_started, 0U, memory_order_release
    );
}

static ShadowSpillStatus caller_allocations_released(
    ShadowSpillRuntime *runtime
) {
    ShadowSpillRuntimeStatistics statistics = {0};
    const ShadowSpillStatus status =
        shadowspill_runtime_statistics(runtime, &statistics);
    if (status != SHADOWSPILL_STATUS_OK) {
        return status;
    }
    return statistics.caller_owned_allocations == 0U
        ? SHADOWSPILL_STATUS_OK
        : SHADOWSPILL_STATUS_INVALID_STATE;
}

/* Take the runtime out of every published field, under the lock, and hand
   its backend to the caller to release once the runtime is gone. */
static ShadowSpillStatus unpublish(
    ShadowSpillRuntime *runtime,
    ShadowSpillPytorchLoadedBackend *backend
) {
    pthread_mutex_lock(&adapter.mutex);
    if (adapter.runtime != runtime) {
        pthread_mutex_unlock(&adapter.mutex);
        return SHADOWSPILL_STATUS_INVALID_STATE;
    }
    atomic_store_explicit(
        &adapter.published_runtime, NULL, memory_order_release
    );
    atomic_store_explicit(
        &adapter.published_allocator_pool_id, UINT32_MAX, memory_order_relaxed
    );
    *backend = adapter.backend;
    adapter.runtime = NULL;
    adapter.backend = (ShadowSpillPytorchLoadedBackend){0};
    atomic_store_explicit(
        &adapter.profiler_annotations_enabled, 0U, memory_order_release
    );
    adapter.closed = 1U;
    pthread_mutex_unlock(&adapter.mutex);
    return SHADOWSPILL_STATUS_OK;
}

/*
 * runtime_destroy stops and joins the worker before releasing anything it
 * can observe. Keep the backend alive until all lanes, events, pinned
 * registrations, and pool arenas have been explicitly closed.
 */
static ShadowSpillStatus release(
    ShadowSpillRuntime *runtime,
    ShadowSpillPytorchLoadedBackend *backend,
    int wait_for_outstanding_work,
    uint64_t *outstanding_actions,
    uint64_t *outstanding_retirements
) {
    const ShadowSpillStatus status =
        wait_for_outstanding_work
            ? shadowspill_runtime_close(runtime)
            : shadowspill_runtime_abandon(
                  runtime, outstanding_actions, outstanding_retirements
              );
    shadowspill_runtime_destroy(runtime);
    shadowspill_pytorch_backend_unload(backend);
    return status;
}

static ShadowSpillStatus close_adapter_runtime(
    int require_no_caller_allocations,
    int wait_for_outstanding_work,
    uint64_t *outstanding_actions,
    uint64_t *outstanding_retirements
) {
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillStatus status = begin_shutdown(&runtime);
    if (runtime == NULL) {
        return status;
    }
    if (require_no_caller_allocations) {
        status = caller_allocations_released(runtime);
        if (status != SHADOWSPILL_STATUS_OK) {
            resume();
            return status;
        }
    }
    ShadowSpillPytorchLoadedBackend backend = {0};
    status = unpublish(runtime, &backend);
    if (status != SHADOWSPILL_STATUS_OK) {
        resume();
        return status;
    }
    return release(
        runtime,
        &backend,
        wait_for_outstanding_work,
        outstanding_actions,
        outstanding_retirements
    );
}

void shadowspill_pytorch_process_exit(int status, void *argument) {
    (void)argument;
    /*
     * The process is going away, so this never waits. Outstanding transfers
     * cannot complete once exit handlers are running, and everything a drain
     * protects is reclaimed at process exit regardless; waiting here is how a
     * third party calling exit() turns into a process that never dies.
     *
     * A nonzero status with nothing latched means something outside
     * ShadowSpill ended the process. Say so: it is the only evidence such a
     * caller leaves behind.
     */
    uint64_t actions = 0U;
    uint64_t retirements = 0U;
    const ShadowSpillStatus latched = close_adapter_runtime(
        0, 0, &actions, &retirements
    );
    if (status == 0 && actions == 0U && retirements == 0U) {
        return;
    }
    (void)fprintf(
        stderr,
        "ShadowSpill: the process is exiting with status %d and "
        "%llu action(s) and %llu retirement(s) outstanding; "
        "ShadowSpill did not wait for them. Latched failure: %s.\n",
        status,
        (unsigned long long)actions,
        (unsigned long long)retirements,
        latched == SHADOWSPILL_STATUS_OK
            ? "none, so the exit came from outside ShadowSpill"
            : shadowspill_status_string(latched)
    );
    (void)fflush(stderr);
}

ShadowSpillStatus shadowspill_pytorch_allocator_close(void) {
    return close_adapter_runtime(1, 1, NULL, NULL);
}
