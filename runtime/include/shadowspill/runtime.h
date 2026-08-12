#ifndef SHADOWSPILL_RUNTIME_H
#define SHADOWSPILL_RUNTIME_H

#include <stdint.h>

#include <shadowspill/backend.h>
#include <shadowspill/profiler.h>

#if defined(_WIN32)
#define SHADOWSPILL_RUNTIME_API __declspec(dllexport)
#else
#define SHADOWSPILL_RUNTIME_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SHADOWSPILL_RUNTIME_ABI_VERSION 17U
#define SHADOWSPILL_TRACE_ABI_VERSION 1U
#define SHADOWSPILL_TRANSFER_PROFILE_ABI_VERSION 1U
#define SHADOWSPILL_RUNTIME_TRACE_LABEL_MAX_BYTES 1024U
#define SHADOWSPILL_RUNTIME_NO_ID UINT64_MAX

typedef struct ShadowSpillRuntime ShadowSpillRuntime;
typedef struct ShadowSpillExecutionRecord ShadowSpillExecutionHandle;

/*
 * Runtime instances are thread-safe. Returned pointers are accelerator
 * addresses and must never be dereferenced by host code.
 */

typedef enum ShadowSpillRuntimeStatus {
    SHADOWSPILL_RUNTIME_OK = 0,
    SHADOWSPILL_RUNTIME_INVALID_ARGUMENT = 1,
    SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE = 2,
    SHADOWSPILL_RUNTIME_OUT_OF_MEMORY = 3,
    SHADOWSPILL_RUNTIME_NO_PROGRESS = 4,
    SHADOWSPILL_RUNTIME_INVALID_STATE = 5,
    SHADOWSPILL_RUNTIME_PLAN_VIOLATION = 6,
    SHADOWSPILL_RUNTIME_BACKEND_FAILURE = 7,
    SHADOWSPILL_RUNTIME_WORKER_FAILURE = 8,
    SHADOWSPILL_RUNTIME_CLOSED = 9,
} ShadowSpillRuntimeStatus;

typedef enum ShadowSpillObjectResidency {
    SHADOWSPILL_OBJECT_SPILL_ONLY = 0,
    SHADOWSPILL_OBJECT_EXECUTION_READY = 1,
    SHADOWSPILL_OBJECT_PREFETCHING = 2,
    SHADOWSPILL_OBJECT_OFFLOADING = 3,
    SHADOWSPILL_OBJECT_RELEASED = 4,
} ShadowSpillObjectResidency;

typedef enum ShadowSpillRuntimeActionKind {
    SHADOWSPILL_RUNTIME_RELEASE = 0,
    SHADOWSPILL_RUNTIME_OFFLOAD = 1,
    SHADOWSPILL_RUNTIME_PREFETCH = 2,
} ShadowSpillRuntimeActionKind;

typedef enum ShadowSpillAllocationEventKind {
    SHADOWSPILL_ALLOCATION_CREATED = 0,
    SHADOWSPILL_ALLOCATION_RELEASED = 1,
    SHADOWSPILL_ALLOCATION_PROMOTED = 2,
    SHADOWSPILL_ALLOCATION_LOGICAL_FREED = 3,
} ShadowSpillAllocationEventKind;

typedef enum ShadowSpillAllocationCategory {
    SHADOWSPILL_ALLOCATION_ANONYMOUS = 0,
    SHADOWSPILL_ALLOCATION_PLANNED_OBJECT = 1,
    SHADOWSPILL_ALLOCATION_CALLER_OWNED = 2,
} ShadowSpillAllocationCategory;

typedef enum ShadowSpillTraceEventKind {
    SHADOWSPILL_TRACE_SESSION_BEGIN = 0,
    SHADOWSPILL_TRACE_SESSION_END = 1,
    SHADOWSPILL_TRACE_BEFORE_TASK = 2,
    SHADOWSPILL_TRACE_AFTER_TASK = 3,
    SHADOWSPILL_TRACE_READINESS_WAIT = 4,
    SHADOWSPILL_TRACE_ACTION_QUEUED = 5,
    SHADOWSPILL_TRACE_DESTINATION_RESERVED = 6,
    SHADOWSPILL_TRACE_TRANSFER_DISPATCHED = 7,
    SHADOWSPILL_TRACE_TRANSFER_COMPLETED = 8,
    SHADOWSPILL_TRACE_ALLOCATION_WAIT_BEGIN = 9,
    SHADOWSPILL_TRACE_ALLOCATION_WAIT_END = 10,
    SHADOWSPILL_TRACE_RETIREMENT_COMPLETED = 11,
    SHADOWSPILL_TRACE_FAILURE_LATCHED = 12,
} ShadowSpillTraceEventKind;

typedef struct ShadowSpillRuntimeConfig {
    uint32_t abi_version;
    uint64_t execution_pool_bytes;
    uint64_t spill_pool_bytes;
    uint64_t minimum_alignment;
    uint64_t worker_poll_nanoseconds;
    ShadowSpillBackend backend;
    ShadowSpillProfiler profiler;
} ShadowSpillRuntimeConfig;

typedef struct ShadowSpillAllocation {
    uint64_t allocation_id;
    uint64_t generation;
    uint64_t requested_bytes;
    uint64_t charged_bytes;
    void *pointer;
} ShadowSpillAllocation;

typedef struct ShadowSpillObjectDescription {
    uint64_t object_id;
    uint64_t size_bytes;
    uint64_t initial_version;
    uint8_t retain_spill_copy;
    uint8_t initially_spill_resident;
} ShadowSpillObjectDescription;

typedef struct ShadowSpillObjectBinding {
    uint64_t object_id;
    uint64_t generation;
    uint64_t allocation_id;
    uint64_t authoritative_version;
    void *pointer;
} ShadowSpillObjectBinding;

typedef struct ShadowSpillObjectUpdate {
    uint64_t object_id;
    uint64_t version_delta;
} ShadowSpillObjectUpdate;

typedef struct ShadowSpillRuntimeAction {
    uint64_t object_id;
    uint8_t kind;
    /*
     * Optional, borrowed semantic profiler label. Admission copies the string,
     * so the caller only needs to keep it alive for the duration of the call.
     * NULL selects a deterministic object/task-ID fallback.
     */
    const char *trace_label;
} ShadowSpillRuntimeAction;

typedef struct ShadowSpillExecutionDescription {
    uint64_t task_id;
    const uint64_t *input_object_ids;
    uint32_t input_count;
    const ShadowSpillObjectUpdate *updates;
    uint32_t update_count;
    const ShadowSpillRuntimeAction *actions;
    uint32_t action_count;
} ShadowSpillExecutionDescription;

typedef struct ShadowSpillAllocationEvent {
    uint64_t sequence;
    uint64_t task_id;
    uint64_t allocation_id;
    uint64_t generation;
    uint64_t requested_bytes;
    uint64_t charged_bytes;
    uint64_t slab_offset;
    uint8_t kind;
    uint8_t category;
} ShadowSpillAllocationEvent;

typedef struct ShadowSpillTraceConfig {
    uint32_t abi_version;
    uint64_t event_capacity;
    uint64_t allocation_event_capacity;
} ShadowSpillTraceConfig;

typedef struct ShadowSpillTransferRouteKey {
    uint32_t source_pool_id;
    uint32_t destination_pool_id;
} ShadowSpillTransferRouteKey;

typedef enum ShadowSpillTransferProfileProvenance {
    SHADOWSPILL_TRANSFER_PROFILE_INITIALIZATION = 0,
    SHADOWSPILL_TRANSFER_PROFILE_RECALIBRATION = 1,
} ShadowSpillTransferProfileProvenance;

typedef struct ShadowSpillTransferCalibrationConfig {
    uint32_t abi_version;
    uint64_t small_copy_bytes;
    uint64_t large_copy_bytes;
    uint32_t warmup_copies;
    uint32_t measured_copies;
    uint8_t provenance;
} ShadowSpillTransferCalibrationConfig;

/*
 * One cell in the dense row-major pool-to-pool transfer matrix. Identity
 * cells are available with zero latency and do not require a physical copy.
 * ``generation`` changes atomically whenever any selected route is
 * recalibrated.
 */
typedef struct ShadowSpillTransferProfile {
    uint32_t abi_version;
    uint32_t source_pool_id;
    uint32_t destination_pool_id;
    uint64_t generation;
    uint64_t latency_nanoseconds;
    uint64_t bandwidth_bytes_per_second;
    uint64_t calibrated_timestamp_nanoseconds;
    uint64_t small_copy_bytes;
    uint64_t large_copy_bytes;
    uint32_t measured_copies;
    uint8_t available;
    uint8_t calibrated;
    uint8_t provenance;
} ShadowSpillTransferProfile;

/*
 * One host-clock observation emitted by the neutral runtime. ``detail_0`` and
 * ``detail_1`` have event-specific meanings documented in runtime.md. IDs use
 * SHADOWSPILL_RUNTIME_NO_ID when they do not apply.
 */
typedef struct ShadowSpillTraceEvent {
    uint64_t sequence;
    uint64_t timestamp_ns;
    uint64_t step_id;
    uint64_t task_id;
    uint64_t object_id;
    uint64_t allocation_id;
    uint64_t bytes;
    uint64_t detail_0;
    uint64_t detail_1;
    uint8_t kind;
} ShadowSpillTraceEvent;

typedef struct ShadowSpillTraceSummary {
    uint32_t abi_version;
    uint64_t step_id;
    uint64_t event_count;
    uint64_t allocation_event_count;
    uint64_t event_capacity;
    uint64_t allocation_event_capacity;
    uint64_t begin_timestamp_ns;
    uint64_t end_timestamp_ns;
    uint8_t active;
    uint8_t event_overflow;
    uint8_t allocation_event_overflow;
} ShadowSpillTraceSummary;

typedef struct ShadowSpillRuntimeStatistics {
    uint64_t execution_pool_bytes;
    uint64_t requested_allocated_bytes;
    uint64_t peak_requested_allocated_bytes;
    uint64_t allocated_bytes;
    uint64_t free_bytes;
    uint64_t largest_free_range_bytes;
    uint64_t external_fragmentation_bytes;
    uint64_t peak_allocated_bytes;
    uint64_t spill_pool_bytes;
    uint64_t spill_allocated_bytes;
    uint64_t spill_peak_allocated_bytes;
    uint64_t live_allocations;
    uint64_t blocked_allocators;
    uint64_t pending_retirements;
    uint64_t retirement_records_fenced;
    uint64_t retirement_records_evented;
    uint64_t retirement_records_preparing;
    uint64_t retirement_records_unfenced;
    uint64_t registered_objects;
    uint64_t queued_actions;
    uint64_t fetch_transfers;
    uint64_t evict_transfers;
    uint64_t bytes_fetched;
    uint64_t bytes_evicted;
    uint64_t wait_events_inserted;
    uint64_t allocation_events;
    uint64_t allocation_event_capacity;
    uint64_t allocation_event_overflow;
} ShadowSpillRuntimeStatistics;

typedef struct ShadowSpillRuntimeFailure {
    uint32_t status;
    uint64_t object_id;
    uint64_t allocation_id;
    uint64_t requested_bytes;
    uint64_t free_bytes;
    uint64_t largest_free_range_bytes;
} ShadowSpillRuntimeFailure;

typedef struct ShadowSpillObjectSnapshot {
    uint64_t object_id;
    uint64_t size_bytes;
    uint64_t generation;
    uint64_t allocation_id;
    uint64_t authoritative_version;
    uint64_t execution_version;
    uint64_t spill_version;
    uint8_t residency;
    uint8_t spill_current;
    uint8_t has_spill_lease;
    void *execution_pointer;
    uint64_t retired_generation;
    void *retired_execution_pointer;
} ShadowSpillObjectSnapshot;

/*
 * Creates one runtime, execution and spill pools, directed transfer lanes,
 * and worker thread. The configuration and backend table are copied; the
 * backend context is borrowed and must outlive the runtime. On failure, output
 * is set to NULL and all successfully created resources are reclaimed.
 */
SHADOWSPILL_RUNTIME_API ShadowSpillRuntimeStatus shadowspill_runtime_create(
    const ShadowSpillRuntimeConfig *config,
    ShadowSpillRuntime **runtime
);

/*
 * Rejects new work, drains queued work, synchronizes both transfer streams,
 * joins the worker, and releases owned resources. This call is explicitly
 * synchronizing and idempotent. It returns the first latched failure.
 */
SHADOWSPILL_RUNTIME_API ShadowSpillRuntimeStatus shadowspill_runtime_close(
    ShadowSpillRuntime *runtime
);

/*
 * Recalibrates every configured directed route when ``routes`` is NULL and
 * ``route_count`` is zero, or only the supplied route keys otherwise. The
 * runtime must be locally idle. This function deliberately performs no
 * inter-process coordination, allowing callers to invoke it concurrently in
 * independent processes after establishing their own barriers. A successful
 * call atomically publishes one new matrix generation.
 */
SHADOWSPILL_RUNTIME_API ShadowSpillRuntimeStatus
shadowspill_runtime_calibrate_transfer_capabilities(
    ShadowSpillRuntime *runtime,
    const ShadowSpillTransferCalibrationConfig *config,
    const ShadowSpillTransferRouteKey *routes,
    uint32_t route_count
);

/*
 * Copies the complete row-major N-by-N profile matrix. ``capacity`` must be at
 * least N*N. The caller receives a lock-consistent generation and count.
 */
SHADOWSPILL_RUNTIME_API ShadowSpillRuntimeStatus
shadowspill_runtime_transfer_profiles(
    ShadowSpillRuntime *runtime,
    ShadowSpillTransferProfile *profiles,
    uint32_t capacity,
    uint32_t *count,
    uint64_t *generation
);

/* Calls close if needed and releases the runtime record. NULL is accepted. */
SHADOWSPILL_RUNTIME_API void shadowspill_runtime_destroy(
    ShadowSpillRuntime *runtime
);

/*
 * Synchronously leases an aligned range from the existing slab; it never grows
 * physical storage. The returned pointer remains valid until logical free and
 * all recorded streams retire it. This call may block only when already
 * pending work can make a suitable range available. Otherwise it returns and
 * latches NO_PROGRESS.
 */
SHADOWSPILL_RUNTIME_API ShadowSpillRuntimeStatus shadowspill_allocate(
    ShadowSpillRuntime *runtime,
    uint64_t bytes,
    uint64_t alignment,
    ShadowSpillBackendStream stream,
    ShadowSpillAllocation *allocation
);

/*
 * Resolves an exact live slab address to its allocation identity and current
 * generation. This read-only lookup exists for framework allocator callbacks
 * whose free/record-stream protocols carry an address rather than an ID.
 */
SHADOWSPILL_RUNTIME_API ShadowSpillRuntimeStatus
shadowspill_allocation_for_pointer(
    ShadowSpillRuntime *runtime,
    const void *pointer,
    ShadowSpillAllocation *allocation
);

/*
 * Performs logical free immediately. A later allocation on the sole recorded
 * stream may reuse the whole pending block by adding its retirement event as a
 * stream dependency. Global and background-transfer reuse waits for every
 * recorded stream to retire. Plan-owned allocations ignore framework logical
 * free until a plan action releases or offloads the owning object.
 */
SHADOWSPILL_RUNTIME_API ShadowSpillRuntimeStatus shadowspill_free(
    ShadowSpillRuntime *runtime,
    uint64_t allocation_id,
    ShadowSpillBackendStream stream
);

/* Adds a borrowed stream token to an allocation's retirement set. */
SHADOWSPILL_RUNTIME_API ShadowSpillRuntimeStatus shadowspill_record_stream(
    ShadowSpillRuntime *runtime,
    uint64_t allocation_id,
    ShadowSpillBackendStream stream
);

/*
 * Registers one logical alias group. The description is borrowed for this
 * call. Requested initial spill storage is leased from the configured spill pool.
 */
SHADOWSPILL_RUNTIME_API ShadowSpillRuntimeStatus shadowspill_register_object(
    ShadowSpillRuntime *runtime,
    const ShadowSpillObjectDescription *description
);

/*
 * Removes a SPILL_ONLY or RELEASED object with no live allocation or queued
 * action, reclaiming retained spill storage. Intended for deterministic plan
 * teardown after final writeback.
 */
SHADOWSPILL_RUNTIME_API ShadowSpillRuntimeStatus shadowspill_unregister_object(
    ShadowSpillRuntime *runtime,
    uint64_t object_id
);

/*
 * Copies one exact object payload into its existing spill lease before execution
 * materialization. The object must be SPILL_ONLY with no execution allocation.
 * Source is borrowed for the call and may be NULL only for a zero-size object.
 */
SHADOWSPILL_RUNTIME_API ShadowSpillRuntimeStatus shadowspill_write_spill_object(
    ShadowSpillRuntime *runtime,
    uint64_t object_id,
    const void *source,
    uint64_t bytes
);

/*
 * Copies one exact, current SPILL_ONLY payload into caller-owned memory. This
 * function does not wait for transfers; callers first use wait_idle or an
 * equivalent lifecycle boundary. Destination may be NULL only for zero size.
 */
SHADOWSPILL_RUNTIME_API ShadowSpillRuntimeStatus shadowspill_read_spill_object(
    ShadowSpillRuntime *runtime,
    uint64_t object_id,
    void *destination,
    uint64_t bytes
);

/*
 * Promotes an ordinary live allocation to plan ownership. The allocation must
 * cover the object and may be bound only once. Object identity survives later
 * address and generation changes.
 */
SHADOWSPILL_RUNTIME_API ShadowSpillRuntimeStatus shadowspill_bind_object(
    ShadowSpillRuntime *runtime,
    uint64_t object_id,
    uint64_t allocation_id
);

/*
 * Replaces an EXECUTION_READY object's current lease with one fresh ordinary
 * allocation created by the active task. The prior lease is retired behind
 * that task's completion fence; no payload copy is performed. The returned
 * binding names the new canonical generation. This operation is valid only
 * inside a task scope and is used for functional mutation outputs.
 */
SHADOWSPILL_RUNTIME_API ShadowSpillRuntimeStatus
shadowspill_replace_object_allocation(
    ShadowSpillRuntime *runtime,
    uint64_t object_id,
    uint64_t allocation_id,
    ShadowSpillObjectBinding *binding
);

/*
 * Removes one EXECUTION_READY object while leaving its allocation live under
 * ordinary caller ownership. No queued action may reference the object. The
 * framework's eventual logical free and recorded streams govern range reuse.
 */
SHADOWSPILL_RUNTIME_API ShadowSpillRuntimeStatus
shadowspill_transfer_object_to_caller(
    ShadowSpillRuntime *runtime,
    uint64_t object_id,
    ShadowSpillAllocation *allocation
);

/*
 * Acquires all input generations and returns one binding per input position.
 * Duplicate object IDs share a binding and one readiness wait. Every distinct
 * in-flight FETCH inserts a wait on compute_stream without host synchronization.
 * Input arrays are borrowed for the call. Returned pointers remain valid for
 * the reported generation until its annotated release/offload completes.
 */
SHADOWSPILL_RUNTIME_API ShadowSpillRuntimeStatus shadowspill_before_task(
    ShadowSpillRuntime *runtime,
    uint64_t task_id,
    ShadowSpillBackendStream compute_stream,
    const uint64_t *input_object_ids,
    uint32_t input_count,
    ShadowSpillObjectBinding *bindings,
    uint32_t binding_capacity
);

/*
 * Applies declared device-version updates, records one completion event on the
 * borrowed compute stream, and copies the ordered action list into runtime
 * ownership. It never reorders or substitutes supplied actions. Arrays are
 * borrowed only for the call; completion is asynchronous.
 */
SHADOWSPILL_RUNTIME_API ShadowSpillRuntimeStatus shadowspill_after_task(
    ShadowSpillRuntime *runtime,
    uint64_t task_id,
    ShadowSpillBackendStream compute_stream,
    const ShadowSpillObjectUpdate *updates,
    uint32_t update_count,
    const ShadowSpillRuntimeAction *actions,
    uint32_t action_count
);

/*
 * Resolves one immutable execution task during plan adoption. Input, mutation,
 * and action arrays are copied and every referenced object is retained. An
 * identical duplicate is idempotent; a conflicting task identity is rejected.
 */
SHADOWSPILL_RUNTIME_API ShadowSpillRuntimeStatus
shadowspill_admit_execution(
    ShadowSpillRuntime *runtime,
    const ShadowSpillExecutionDescription *description
);

/*
 * Releases every immutable execution record admitted for the completed plan.
 * The runtime must be idle, no task boundary may be active, and every logical
 * plan object must already be unregistered. Ordinary caller-owned allocations
 * are unaffected. This explicitly synchronizing lifecycle boundary permits a
 * later plan to reuse the same dense task and object identities.
 */
SHADOWSPILL_RUNTIME_API ShadowSpillRuntimeStatus
shadowspill_clear_execution_plan(ShadowSpillRuntime *runtime);

/*
 * Resolves one stable, immutable execution handle on the cold path. The handle
 * is borrowed from runtime and remains valid until its execution plan is
 * cleared or that runtime is destroyed. It must only be passed back to the
 * runtime that produced it.
 */
SHADOWSPILL_RUNTIME_API ShadowSpillRuntimeStatus
shadowspill_resolve_execution(
    ShadowSpillRuntime *runtime,
    uint64_t task_id,
    const ShadowSpillExecutionHandle **handle
);

/* Execute an admitted boundary without resupplying or decoding its topology. */
SHADOWSPILL_RUNTIME_API ShadowSpillRuntimeStatus
shadowspill_before_execution(
    ShadowSpillRuntime *runtime,
    uint64_t task_id,
    ShadowSpillBackendStream compute_stream,
    ShadowSpillObjectBinding *bindings,
    uint32_t binding_capacity
);

SHADOWSPILL_RUNTIME_API ShadowSpillRuntimeStatus
shadowspill_after_execution(
    ShadowSpillRuntime *runtime,
    uint64_t task_id,
    ShadowSpillBackendStream compute_stream
);

/* Hot-path equivalents that bypass execution-table lookup and locking. */
SHADOWSPILL_RUNTIME_API ShadowSpillRuntimeStatus
shadowspill_before_execution_handle(
    ShadowSpillRuntime *runtime,
    const ShadowSpillExecutionHandle *handle,
    ShadowSpillBackendStream compute_stream,
    ShadowSpillObjectBinding *bindings,
    uint32_t binding_capacity
);

SHADOWSPILL_RUNTIME_API ShadowSpillRuntimeStatus
shadowspill_after_execution_handle(
    ShadowSpillRuntime *runtime,
    const ShadowSpillExecutionHandle *handle,
    ShadowSpillBackendStream compute_stream
);

/*
 * Clears the calling thread's active task scope after frontend execution
 * aborts before after_task. This does not cancel already submitted device work.
 */
SHADOWSPILL_RUNTIME_API void shadowspill_abort_task(
    ShadowSpillRuntime *runtime
);

/*
 * Starts one bounded allocation-lifetime capture. Storage is allocated before
 * capture begins, so allocator callbacks only append fixed-size records. A
 * full buffer latches ALLOCATION_FAILURE rather than silently losing evidence.
 */
SHADOWSPILL_RUNTIME_API ShadowSpillRuntimeStatus
shadowspill_allocation_telemetry_start(
    ShadowSpillRuntime *runtime,
    uint64_t capacity
);

/* Stops capture. Previously recorded events remain readable until restart. */
SHADOWSPILL_RUNTIME_API ShadowSpillRuntimeStatus
shadowspill_allocation_telemetry_stop(ShadowSpillRuntime *runtime);

/*
 * Copies the complete ordered event stream. Pass events=NULL and capacity=0
 * to query count. Caller owns the destination and no runtime pointer escapes.
 */
SHADOWSPILL_RUNTIME_API ShadowSpillRuntimeStatus
shadowspill_allocation_telemetry_read(
    ShadowSpillRuntime *runtime,
    ShadowSpillAllocationEvent *events,
    uint64_t capacity,
    uint64_t *count
);

/*
 * Planning-only allocation of reusable trace buffers. Calling this does not
 * enable tracing. Growth is rejected while a trace or allocation-profile
 * session is active; no trace buffer grows from a runtime hot path.
 */
SHADOWSPILL_RUNTIME_API ShadowSpillRuntimeStatus shadowspill_trace_prepare(
    ShadowSpillRuntime *runtime,
    const ShadowSpillTraceConfig *config
);

/* Begins one prepared trace and its allocation-lifetime capture. */
SHADOWSPILL_RUNTIME_API ShadowSpillRuntimeStatus shadowspill_trace_begin(
    ShadowSpillRuntime *runtime,
    uint64_t step_id
);

/*
 * Stops appending without synchronizing a stream or worker. Callers establish
 * their required completion boundary before ending the trace.
 */
SHADOWSPILL_RUNTIME_API ShadowSpillRuntimeStatus shadowspill_trace_end(
    ShadowSpillRuntime *runtime
);

/*
 * Copies one stopped trace into caller-owned arrays. NULL arrays with zero
 * capacities query the required counts through summary. No runtime pointer or
 * backend handle escapes.
 */
SHADOWSPILL_RUNTIME_API ShadowSpillRuntimeStatus shadowspill_trace_read(
    ShadowSpillRuntime *runtime,
    ShadowSpillTraceSummary *summary,
    ShadowSpillTraceEvent *events,
    uint64_t event_capacity,
    ShadowSpillAllocationEvent *allocation_events,
    uint64_t allocation_event_capacity
);

/* Explicitly synchronizing test/checkpoint helper; returns first failure. */
SHADOWSPILL_RUNTIME_API ShadowSpillRuntimeStatus shadowspill_runtime_wait_idle(
    ShadowSpillRuntime *runtime
);

/*
 * Planning-only growth of the one pinned-host arena. The runtime must be idle;
 * existing object offsets and payloads are preserved. The backend host pointer
 * is CPU-addressable by contract. Shrinkage is rejected. This operation owns
 * both arenas briefly, so callers must include that transient in host-budget
 * admission. It is forbidden after frontend physical admission is sealed.
 */
SHADOWSPILL_RUNTIME_API ShadowSpillRuntimeStatus
shadowspill_runtime_resize_spill_pool(
    ShadowSpillRuntime *runtime,
    uint64_t spill_pool_bytes
);

/* Copies a lock-consistent telemetry snapshot into caller-owned storage. */
SHADOWSPILL_RUNTIME_API ShadowSpillRuntimeStatus shadowspill_runtime_statistics(
    ShadowSpillRuntime *runtime,
    ShadowSpillRuntimeStatistics *statistics
);

/* Copies the immutable first-failure snapshot; status is OK before failure. */
SHADOWSPILL_RUNTIME_API ShadowSpillRuntimeStatus shadowspill_runtime_failure(
    ShadowSpillRuntime *runtime,
    ShadowSpillRuntimeFailure *failure
);

/* Copies one object's current logical state into caller-owned storage. */
SHADOWSPILL_RUNTIME_API ShadowSpillRuntimeStatus shadowspill_object_snapshot(
    ShadowSpillRuntime *runtime,
    uint64_t object_id,
    ShadowSpillObjectSnapshot *snapshot
);

SHADOWSPILL_RUNTIME_API uint32_t shadowspill_runtime_abi_version(void);

SHADOWSPILL_RUNTIME_API const char *shadowspill_runtime_status_string(
    ShadowSpillRuntimeStatus status
);

#ifdef __cplusplus
}
#endif

#endif
