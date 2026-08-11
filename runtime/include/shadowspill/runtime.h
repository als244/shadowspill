#ifndef SHADOWSPILL_RUNTIME_H
#define SHADOWSPILL_RUNTIME_H

#include <stdint.h>

#include <shadowspill/backend.h>

#if defined(_WIN32)
#define SHADOWSPILL_RUNTIME_API __declspec(dllexport)
#else
#define SHADOWSPILL_RUNTIME_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SHADOWSPILL_RUNTIME_ABI_VERSION 1U
#define SHADOWSPILL_RUNTIME_NO_ID UINT64_MAX

typedef struct ShadowSpillRuntime ShadowSpillRuntime;

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
    SHADOWSPILL_OBJECT_HOST_ONLY = 0,
    SHADOWSPILL_OBJECT_DEVICE_READY = 1,
    SHADOWSPILL_OBJECT_PREFETCHING = 2,
    SHADOWSPILL_OBJECT_OFFLOADING = 3,
    SHADOWSPILL_OBJECT_RELEASED = 4,
} ShadowSpillObjectResidency;

typedef enum ShadowSpillRuntimeActionKind {
    SHADOWSPILL_RUNTIME_RELEASE = 0,
    SHADOWSPILL_RUNTIME_OFFLOAD = 1,
    SHADOWSPILL_RUNTIME_PREFETCH = 2,
} ShadowSpillRuntimeActionKind;

typedef struct ShadowSpillRuntimeConfig {
    uint32_t abi_version;
    uint64_t device_slab_bytes;
    uint64_t host_arena_bytes;
    uint64_t minimum_alignment;
    uint64_t progress_poll_nanoseconds;
    ShadowSpillBackend backend;
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
    uint8_t retain_host_backing;
    uint8_t initially_host_resident;
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
} ShadowSpillRuntimeAction;

typedef struct ShadowSpillRuntimeStatistics {
    uint64_t slab_bytes;
    uint64_t allocated_bytes;
    uint64_t free_bytes;
    uint64_t largest_free_range_bytes;
    uint64_t external_fragmentation_bytes;
    uint64_t peak_allocated_bytes;
    uint64_t host_arena_bytes;
    uint64_t host_allocated_bytes;
    uint64_t host_peak_allocated_bytes;
    uint64_t live_allocations;
    uint64_t blocked_allocators;
    uint64_t pending_retirements;
    uint64_t registered_objects;
    uint64_t queued_actions;
    uint64_t transfers_to_device;
    uint64_t transfers_to_host;
    uint64_t bytes_to_device;
    uint64_t bytes_to_host;
    uint64_t wait_events_inserted;
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
    uint64_t device_version;
    uint64_t host_version;
    uint8_t residency;
    uint8_t host_current;
    uint8_t has_host_range;
    void *device_pointer;
} ShadowSpillObjectSnapshot;

/*
 * Creates one runtime, physical device slab, host arena, transfer-stream pair,
 * and progress thread. The configuration and backend table are copied; the
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
 * Performs logical free immediately. Physical reuse waits for events recorded
 * on the supplied stream and every stream previously passed to record_stream.
 * Plan-owned allocations ignore framework logical free until a plan action
 * releases or offloads the owning object.
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
 * call. Requested initial host backing is leased from the bounded host arena.
 */
SHADOWSPILL_RUNTIME_API ShadowSpillRuntimeStatus shadowspill_register_object(
    ShadowSpillRuntime *runtime,
    const ShadowSpillObjectDescription *description
);

/*
 * Copies one exact object payload into its existing host backing before device
 * materialization. The object must be HOST_ONLY with no device allocation.
 * Source is borrowed for the call and may be NULL only for a zero-size object.
 */
SHADOWSPILL_RUNTIME_API ShadowSpillRuntimeStatus shadowspill_write_host_object(
    ShadowSpillRuntime *runtime,
    uint64_t object_id,
    const void *source,
    uint64_t bytes
);

/*
 * Copies one exact, current HOST_ONLY payload into caller-owned memory. This
 * function does not wait for transfers; callers first use wait_idle or an
 * equivalent lifecycle boundary. Destination may be NULL only for zero size.
 */
SHADOWSPILL_RUNTIME_API ShadowSpillRuntimeStatus shadowspill_read_host_object(
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
 * Acquires all input generations and returns one binding per input position.
 * Duplicate object IDs share a binding and one readiness wait. Every distinct
 * in-flight H2D inserts a wait on compute_stream without host synchronization.
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

/* Explicitly synchronizing test/checkpoint helper; returns first failure. */
SHADOWSPILL_RUNTIME_API ShadowSpillRuntimeStatus shadowspill_runtime_wait_idle(
    ShadowSpillRuntime *runtime
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
