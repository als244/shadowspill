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

/*
 * Plan ids are a dense counter from one, so the slot array is indexed by id
 * directly. A hash table over these keys would hash the identity function and
 * chain buckets that hold one entry each; this is that table with the indirection
 * removed. `claimed` is what separates an id never created with from one whose
 * plan has since been freed, since both have a NULL plan.
 */
typedef struct ShadowSpillPlanSlot {
    ShadowSpillPlan *plan;
    uint8_t claimed;
} ShadowSpillPlanSlot;

typedef struct ShadowSpillPlanRegistry {
    ShadowSpillPlanSlot *slots;
    uint64_t capacity;
    pthread_mutex_t lock;
    uint8_t lock_initialized;
} ShadowSpillPlanRegistry;

struct ShadowSpillRuntime {
    /* Cold lifecycle and the still-unmigrated action-list owner. */
    pthread_mutex_t mutex;
    pthread_mutex_t failure_lock;
    ShadowSpillIdleWakeup idle_wakeup;
    /* One flag per primitive that has no safe destroy before it is created, so
     * a runtime that failed partway through creation is torn down by asking
     * what it reached rather than by unwinding at each place it can fail. */
    uint8_t mutex_initialized;
    uint8_t failure_lock_initialized;
    uint8_t idle_wakeup_initialized;
    pthread_t worker_thread;
    int worker_started;
    _Atomic uint8_t closing;
    /*
     * Set when the process is exiting and this runtime is being abandoned.
     * Cleanup that could block -- draining queued work, synchronizing a lane,
     * finalizing an aborted task's retirements behind a pool lock the worker
     * may hold -- is skipped while it is set, because none of it can complete
     * once exit handlers are running and all of it is reclaimed at _exit.
     */
    _Atomic uint8_t abandoned;
    _Atomic uint8_t closed;
    _Atomic uint8_t worker_stop;
    _Atomic uint32_t failure_status;
    uint64_t worker_poll_nanoseconds;
    uint64_t background_transfer_window_bytes;

    ShadowSpillBackend backend;
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
    /* Every plan id ever created with, and the plan it names. Its own lock, so
     * asking what a lease's plan id means does not wait behind plan creation or
     * teardown. `plan` goes NULL when the plan record is freed while the slot
     * stays claimed: a lease can outlive its plan, and its id must still answer
     * rather than come back as some later plan's. */
    ShadowSpillPlanRegistry plans_by_id;
    uint8_t plans_lock_initialized;
    ShadowSpillEventPool events;
    ShadowSpillEventPool timing_events;
    ShadowSpillCompletionTracker completions;
    uint8_t completions_initialized;
    ShadowSpillRetirementQueue retirements;
    ShadowSpillActionQueue actions;
    _Atomic(ShadowSpillTaskRecord *) worker_submission;
    _Atomic uint64_t next_worker_submission_sequence;

    _Atomic uint64_t next_plan_id;
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
    uint64_t trace_began_at_ns;
    uint64_t trace_ended_at_ns;
    uint64_t trace_allocation_event_capacity;
    _Atomic uint8_t trace_prepared;
    _Atomic uint8_t trace_active;
    _Atomic uint8_t trace_event_overflow;
    /* The caller's timing event that transfer intervals are measured from;
     * meaningful only while a trace is active. */
    /* Whether the backend's profiler is emitting ranges. One writer, the
       annotations setter; read on every range a caller opens. */
    atomic_uchar profiler_annotations_enabled;
    ShadowSpillBackendEvent trace_origin_event;
    uint8_t trace_origin_present;
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

/* The TRANSFER_COMPLETED append: the same gate, plus the stream interval. */
#define shadowspill_append_stamped_trace_event_locked(runtime_, ...)          \
    do {                                                                       \
        ShadowSpillRuntime *const shadowspill_trace_runtime__ = (runtime_);    \
        if (atomic_load_explicit(                                              \
                &shadowspill_trace_runtime__->trace_active,                    \
                memory_order_acquire                                           \
            ) != 0U) {                                                         \
            shadowspill_trace_append_stamped_enabled(                         \
                shadowspill_trace_runtime__, __VA_ARGS__                       \
            );                                                                 \
        }                                                                      \
    } while (0)

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
void *shadowspill_worker_main(void *pointer);

void shadowspill_notify_worker(ShadowSpillRuntime *runtime);

/* Shared between the runtime's own files; see runtime_open.c for the order. */
void shadowspill_runtime_release_resources(ShadowSpillRuntime *runtime);
void shadowspill_runtime_release_primitives(ShadowSpillRuntime *runtime);

#endif
