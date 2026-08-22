#ifndef SHADOWSPILL_RUNTIME_INTERNAL_H
#define SHADOWSPILL_RUNTIME_INTERNAL_H

#include <pthread.h>
#include <stddef.h>
#include <stdatomic.h>
#include <stdint.h>

#include <shadowspill/runtime.h>

#include "failure_state.h"
#include "sync/internal.h"
#include "memory/internal.h"
#include "objects/internal.h"
#include "transfers/internal.h"
#include "tasks/internal.h"
#include "plan/internal.h"
#include "telemetry/internal.h"

struct ShadowSpillRuntime {
    /* Cold lifecycle and the still-unmigrated action-list owner. */
    pthread_mutex_t mutex;
    pthread_mutex_t failure_lock;
    ShadowSpillIdleWakeup idle_wakeup;
    pthread_t worker_thread;
    int worker_started;
    _Atomic uint8_t closing;
    _Atomic uint8_t closed;
    _Atomic uint8_t worker_stop;
    _Atomic uint32_t failure_status;
    uint64_t worker_poll_nanoseconds;

    ShadowSpillSynchronizationBackend synchronization;
    ShadowSpillProfiler profiler;
    ShadowSpillRouteState *routes;
    uint32_t route_count;

    pthread_rwlock_t transfer_profiles_lock;
    ShadowSpillTransferProfile *transfer_profiles;
    uint32_t transfer_profile_count;
    uint64_t transfer_profile_generation;
    uint8_t transfer_profiles_initialized;

    ShadowSpillMemoryPool *pools;
    uint32_t pool_count;
    ShadowSpillObjectTable objects;
    pthread_mutex_t plans_lock;
    ShadowSpillPlan *plans;
    uint8_t plans_lock_initialized;
    ShadowSpillEventPool events;
    ShadowSpillCompletionTracker completions;
    uint8_t completions_initialized;
    ShadowSpillRetirementQueue retirements;
    ShadowSpillActionQueue actions;
    _Atomic(ShadowSpillTaskRecord *) worker_submission;
    _Atomic uint64_t next_worker_submission_sequence;

    _Atomic uint64_t next_allocation_id;
    _Atomic uint64_t next_generation;
    _Atomic uint64_t next_event_generation;
    _Atomic uint64_t pending_retirements;
    _Atomic uint64_t pending_capacity_actions;
    _Atomic uint64_t registered_objects;
    _Atomic uint64_t fetch_transfers;
    _Atomic uint64_t evict_transfers;
    _Atomic uint64_t bytes_fetched;
    _Atomic uint64_t bytes_evicted;
    _Atomic uint64_t wait_events_inserted;
    uint64_t event_query_epoch;
    ShadowSpillAllocationEvent *allocation_events;
    _Atomic uint64_t allocation_event_count;
    uint64_t allocation_event_capacity;
    _Atomic uint64_t next_allocation_event_sequence;
    _Atomic uint8_t allocation_telemetry_active;
    _Atomic uint8_t allocation_event_overflow;
    ShadowSpillTraceEvent *trace_events;
    _Atomic uint64_t trace_event_count;
    uint64_t trace_event_capacity;
    _Atomic uint64_t next_trace_event_sequence;
    uint64_t trace_step_id;
    uint64_t trace_begin_timestamp_ns;
    uint64_t trace_end_timestamp_ns;
    uint64_t trace_allocation_event_capacity;
    _Atomic uint8_t trace_prepared;
    _Atomic uint8_t trace_active;
    _Atomic uint8_t trace_event_overflow;
    ShadowSpillRuntimeFailure failure;
};

static inline void shadowspill_cpu_relax(void) {
#if defined(__x86_64__) || defined(__i386__)
    __asm__ volatile("pause" ::: "memory");
#elif defined(__aarch64__) || defined(__arm__)
    __asm__ volatile("yield" ::: "memory");
#else
    atomic_signal_fence(memory_order_seq_cst);
#endif
}

int shadowspill_transfer_route_is_valid(
    const ShadowSpillTransferRoute *route
);

#define shadowspill_append_trace_event_locked(runtime_, ...)                  \
    do {                                                                       \
        ShadowSpillRuntime *const shadowspill_trace_runtime__ = (runtime_);    \
        if (atomic_load_explicit(                                              \
                &shadowspill_trace_runtime__->trace_active,                    \
                memory_order_acquire                                           \
            ) != 0U) {                                                         \
            shadowspill_trace_append_enabled(                                 \
                shadowspill_trace_runtime__, __VA_ARGS__                       \
            );                                                                 \
        }                                                                      \
    } while (0)
int shadowspill_memory_pool_backend_is_valid(
    const ShadowSpillMemoryPoolBackend *backend
);

int shadowspill_synchronization_backend_is_valid(
    const ShadowSpillSynchronizationBackend *backend
);

void *shadowspill_worker_main(void *pointer);

void shadowspill_notify_worker(ShadowSpillRuntime *runtime);

#endif
