#ifndef SHADOWSPILL_RUNTIME_H
#define SHADOWSPILL_RUNTIME_H

#include <stdint.h>

#include <shadowspill/shadowspill.h>

#include <shadowspill/backend.h>
#include <shadowspill/profiler.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SHADOWSPILL_RUNTIME_TRACE_LABEL_MAX_BYTES 1024U
#define SHADOWSPILL_RUNTIME_NO_ID UINT64_MAX

typedef struct ShadowSpillRuntime ShadowSpillRuntime;
typedef struct ShadowSpillPlan ShadowSpillPlan;
typedef struct ShadowSpillTaskRecord ShadowSpillTaskHandle;
typedef struct ShadowSpillTaskRecord ShadowSpillActionBatchHandle;
typedef struct ShadowSpillObjectAcquisitionRecord
    ShadowSpillObjectAcquisitionHandle;
typedef struct ShadowSpillObjectHandle ShadowSpillObjectHandle;

/*
 * Runtime instances are thread-safe. Returned pointers are accelerator
 * addresses and must never be dereferenced by host code.
 */

/* Execution names for the shared statuses; see <shadowspill/status.h>. */

/*
 * Why an operation failed, where the status alone does not say.
 *
 * The status is the coarse class a caller acts on; the reason names the
 * specific condition, so a report can explain itself. Several of these sit
 * under one status on purpose - a lease that cannot be released and a process
 * allocator that refuses a record are both internal failures, and a caller
 * treats them
 * alike, but a reader must be able to tell them apart.
 */
typedef enum ShadowSpillFailureReason {
    SHADOWSPILL_FAILURE_REASON_UNSPECIFIED = 0,
    /* The process allocator refused memory for an internal record. This is
     * anonymous memory, and is neither the device pool nor the spill pool. */
    SHADOWSPILL_FAILURE_REASON_PROCESS_ALLOCATION_REFUSED = 1,
    /* A sealed bookkeeping table had no free record. The reserve was sized
     * too small for what this workload allocates; the pool has bytes. */
    SHADOWSPILL_FAILURE_REASON_RECORD_CAPACITY_EXHAUSTED = 2,
    /* A lease could not be released: it was not linked to the pool it names,
     * was already free, or was mid-handoff. */
    SHADOWSPILL_FAILURE_REASON_LEASE_RELEASE_REJECTED = 3,
    /* A successor's claim on a predecessor's range could not be cancelled. */
    SHADOWSPILL_FAILURE_REASON_RESERVATION_CANCEL_REJECTED = 4,
    /* Freed bytes could not be returned to the range allocator. */
    SHADOWSPILL_FAILURE_REASON_RANGE_RETURN_REJECTED = 5,
    /* No range large enough, and nothing left to release for one. */
    SHADOWSPILL_FAILURE_REASON_POOL_EXHAUSTED = 6,
} ShadowSpillFailureReason;

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

typedef struct ShadowSpillMemoryPoolDescription {
    uint32_t pool_id;
    uint64_t capacity_bytes;
    uint64_t minimum_alignment;
    ShadowSpillMemoryPoolBackend backend;
} ShadowSpillMemoryPoolDescription;

typedef struct ShadowSpillTransferRouteDescription {
    uint32_t route_id;
    const char *name;
    ShadowSpillTransferRoute route;
} ShadowSpillTransferRouteDescription;

typedef struct ShadowSpillRuntimeConfig {
    uint32_t abi_version;
    const ShadowSpillMemoryPoolDescription *pools;
    uint32_t pool_count;
    const ShadowSpillTransferRouteDescription *routes;
    uint32_t route_count;
    uint64_t worker_poll_nanoseconds;
    ShadowSpillSynchronizationBackend synchronization;
    ShadowSpillProfiler profiler;
} ShadowSpillRuntimeConfig;

/*
 * Immutable pool and route roles selected by one admitted callable. Multiple
 * plans may share a runtime topology and runtime-owned logical objects.
 */
typedef struct ShadowSpillPlanDescription {
    uint32_t execution_pool_id;
    uint32_t spill_pool_id;
    uint32_t fetch_route_id;
    uint32_t evict_route_id;
} ShadowSpillPlanDescription;

typedef enum ShadowSpillObjectConsistency {
    SHADOWSPILL_OBJECT_CAUSAL = 0,
    SHADOWSPILL_OBJECT_UNORDERED = 1,
} ShadowSpillObjectConsistency;

/*
 * Acquire and release one retained runtime-global object handle. The handle
 * contains no pool role or framework metadata and remains valid across object
 * generation and residency changes.
 */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_object_handle_acquire(
    ShadowSpillRuntime *runtime,
    uint64_t runtime_object_id,
    ShadowSpillObjectHandle **output
);

SHADOWSPILL_API ShadowSpillStatus
shadowspill_object_handle_release(
    ShadowSpillObjectHandle *handle
);

/*
 * Release one completed residency generation without destroying its logical
 * object.  This is used by bounded producer slots after every external owner
 * of the prior value has released its handle.  Plan bindings remain valid and
 * a later task may publish a new generation into the same logical object.
 */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_object_release_generation(
    const ShadowSpillObjectHandle *handle,
    uint64_t expected_generation
);

/*
 * Bind one Program-local identity to a retained runtime object handle. Equal
 * plan-local IDs in different plans have no relationship unless both bindings
 * use handles for the same runtime object.
 */
SHADOWSPILL_API ShadowSpillStatus shadowspill_plan_bind_object(
    ShadowSpillPlan *plan,
    uint64_t plan_object_id,
    const ShadowSpillObjectHandle *object,
    uint8_t consistency
);

typedef struct ShadowSpillAllocation {
    uint32_t pool_id;
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
    uint32_t initial_pool_id;
    uint8_t retain_spill_copy;
    uint8_t initially_resident;
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

typedef enum ShadowSpillTaskAllocationOperation {
    SHADOWSPILL_TASK_ALLOCATION_ALLOCATE = 0,
    SHADOWSPILL_TASK_ALLOCATION_FREE = 1,
} ShadowSpillTaskAllocationOperation;

/*
 * Pointer-free runtime projection of one compiled-task allocator operation.
 * Output and mutation ownership remains in the framework storage contract;
 * the neutral runtime validates only callback order and geometry.
 */
typedef struct ShadowSpillTaskAllocationContractStep {
    uint64_t allocation_ordinal;
    uint64_t requested_bytes;
    uint64_t charged_bytes;
    uint64_t alignment_bytes;
    uint8_t operation;
    /*
     * Required allocations publish framework-visible output or mutation
     * storage. Anonymous/provider operations are optional core observations:
     * runtime insertions use bounded dynamic scratch and omissions are
     * reconciled in order.
     */
    uint8_t required;
} ShadowSpillTaskAllocationContractStep;

typedef enum ShadowSpillTaskPublicationKind {
    /* Publish the first/current execution-pool lease for a logical object. */
    SHADOWSPILL_TASK_PUBLICATION_BIND = 0,
    /* Replace the logical object's prior lease without changing identity. */
    SHADOWSPILL_TASK_PUBLICATION_REPLACE = 1,
} ShadowSpillTaskPublicationKind;

/*
 * Cold-path description of one framework-visible task allocation. The
 * plan-local object identity is resolved to a retained object pointer during
 * task admission; repeated publication uses only the task handle and ordinal.
 */
typedef struct ShadowSpillTaskPublicationDescription {
    uint64_t object_id;
    uint8_t kind;
} ShadowSpillTaskPublicationDescription;

typedef enum ShadowSpillFixedPlacementKind {
    SHADOWSPILL_FIXED_INITIAL_OBJECT = 0,
    SHADOWSPILL_FIXED_TASK_ALLOCATION = 1,
    SHADOWSPILL_FIXED_ACTION_DESTINATION = 2,
    SHADOWSPILL_DYNAMIC_TASK_ALLOCATION = 3,
    SHADOWSPILL_DYNAMIC_ACTION_DESTINATION = 4,
} ShadowSpillFixedPlacementKind;

/*
 * One allocation policy inside an admitted physical layout. ``ordinal`` is a
 * task-local allocator ordinal or task-local action ordinal. Initial objects
 * use ``object_id`` and set task_id/ordinal to SHADOWSPILL_RUNTIME_NO_ID.
 * Dynamic task allocations and action destinations set ``offset`` to
 * SHADOWSPILL_RUNTIME_NO_ID; every fixed kind names a subrange of the
 * plan-owned execution-pool slice.
 */
typedef struct ShadowSpillFixedPlacementDescription {
    uint64_t task_id;
    uint64_t ordinal;
    uint64_t object_id;
    uint64_t offset;
    uint64_t bytes;
    uint64_t alignment_bytes;
    uint8_t kind;
} ShadowSpillFixedPlacementDescription;

/*
 * One cross-lane address-reuse proof. The predecessor must be an admitted
 * eviction action. The successor names a fixed task allocation or fixed fetch
 * destination using the same task-local identity as its placement.
 */
typedef struct ShadowSpillFixedDependencyDescription {
    uint64_t predecessor_task_id;
    uint64_t predecessor_action_ordinal;
    uint64_t successor_task_id;
    uint64_t successor_ordinal;
    uint8_t successor_kind;
} ShadowSpillFixedDependencyDescription;

typedef struct ShadowSpillFixedLayoutDescription {
    uint32_t abi_version;
    uint64_t slice_bytes;
    const ShadowSpillFixedPlacementDescription *placements;
    uint64_t placement_count;
    const ShadowSpillFixedDependencyDescription *dependencies;
    uint64_t dependency_count;
} ShadowSpillFixedLayoutDescription;

typedef struct ShadowSpillTaskDescription {
    uint64_t task_id;
    /*
     * Optional borrowed semantic profiler label. Admission copies it into the
     * immutable task handle, so repeated execution performs no ID lookup.
     */
    const char *trace_label;
    const uint64_t *input_object_ids;
    uint32_t input_count;
    const ShadowSpillObjectUpdate *updates;
    uint32_t update_count;
    const ShadowSpillTaskPublicationDescription *publications;
    uint32_t publication_count;
    const ShadowSpillRuntimeAction *actions;
    uint32_t action_count;
    const ShadowSpillTaskAllocationContractStep *allocation_contract_steps;
    uint32_t allocation_contract_step_count;
    uint8_t enforce_allocation_contract;
    /*
     * Conservative task-local allocator envelope. Zero leaves a field
     * unbounded for a caller without a task profile. These bounds constrain
     * behavior, never addresses or allocation order.
     */
    uint64_t maximum_requested_allocation_bytes;
    uint64_t maximum_charged_allocation_bytes;
    uint64_t live_requested_allocation_limit_bytes;
    uint64_t live_charged_allocation_limit_bytes;
    /*
     * Bounded dynamic storage for allocator operations absent from the fixed
     * core contract. Zero preserves strict exact-contract behavior.
     */
    uint64_t dynamic_scratch_maximum_allocation_bytes;
    uint64_t dynamic_scratch_live_limit_bytes;
} ShadowSpillTaskDescription;

typedef struct ShadowSpillAllocationEvent {
    uint64_t sequence;
    uint32_t pool_id;
    uint64_t task_id;
    uint64_t allocation_id;
    uint64_t generation;
    uint64_t requested_bytes;
    uint64_t charged_bytes;
    uint64_t alignment_bytes;
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

typedef enum ShadowSpillTransferCalibrationMode {
    SHADOWSPILL_TRANSFER_CALIBRATION_IDENTITY = 0,
    SHADOWSPILL_TRANSFER_CALIBRATION_SOLO = 1,
    SHADOWSPILL_TRANSFER_CALIBRATION_BIDIRECTIONAL = 2,
} ShadowSpillTransferCalibrationMode;

typedef struct ShadowSpillTransferCalibrationConfig {
    uint32_t abi_version;
    uint64_t small_copy_bytes;
    uint64_t large_copy_bytes;
    uint32_t warmup_copies;
    uint32_t measured_copies;
    uint8_t provenance;
} ShadowSpillTransferCalibrationConfig;

/*
 * One cell in the complete row-major pool-to-pool transfer matrix. Identity
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
    /* Effective sustained rate consumed by planning and simulation. */
    uint64_t bandwidth_bytes_per_second;
    /* Independently measured rate with no reverse-route traffic. */
    uint64_t solo_bandwidth_bytes_per_second;
    /* Directional rate while the reverse route is simultaneously saturated. */
    uint64_t concurrent_bandwidth_bytes_per_second;
    uint64_t solo_measurement_nanoseconds;
    uint64_t concurrent_measurement_nanoseconds;
    uint64_t calibrated_timestamp_nanoseconds;
    uint64_t small_copy_bytes;
    uint64_t large_copy_bytes;
    uint32_t measured_copies;
    uint8_t available;
    uint8_t calibrated;
    uint8_t provenance;
    uint8_t calibration_mode;
    uint8_t concurrent_route_count;
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
    uint64_t free_prefix_bytes;
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
    uint64_t event_lease_capacity;
    uint64_t event_lease_in_use;
    uint64_t event_lease_peak_in_use;
    uint64_t event_lease_growth_rejections;
    uint64_t retirement_record_capacity;
    uint64_t retirement_record_in_use;
    uint64_t retirement_record_peak_in_use;
    uint64_t retirement_record_growth_rejections;
    uint64_t memory_lease_record_capacity;
    uint64_t memory_lease_record_in_use;
    uint64_t memory_lease_record_peak_in_use;
    uint64_t memory_lease_record_growth_rejections;
    uint64_t lease_use_record_capacity;
    uint64_t lease_use_record_in_use;
    uint64_t lease_use_record_peak_in_use;
    uint64_t lease_use_record_growth_rejections;
    /* Framework-owned plan outputs that still reference pool storage. */
    uint64_t caller_owned_allocations;
} ShadowSpillRuntimeStatistics;

typedef struct ShadowSpillRuntimeFailure {
    uint32_t status;
    /* ShadowSpillFailureReason; UNSPECIFIED where the status says it all. */
    uint32_t reason;
    uint32_t pool_id;
    uint64_t task_id;
    uint64_t object_id;
    uint64_t allocation_id;
    uint64_t requested_bytes;
    uint64_t free_bytes;
    uint64_t largest_free_range_bytes;
    uint64_t task_live_requested_bytes;
    uint64_t task_live_charged_bytes;
    uint64_t task_live_requested_limit_bytes;
    uint64_t task_live_charged_limit_bytes;
    uint64_t task_maximum_requested_allocation_bytes;
    uint64_t task_maximum_charged_allocation_bytes;
    uint64_t task_allocation_operation_index;
    uint64_t task_allocation_expected_ordinal;
    uint64_t task_allocation_actual_ordinal;
    uint64_t task_allocation_expected_requested_bytes;
    uint64_t task_allocation_actual_requested_bytes;
    uint64_t task_allocation_expected_charged_bytes;
    uint64_t task_allocation_actual_charged_bytes;
    uint64_t task_allocation_expected_alignment_bytes;
    uint64_t task_allocation_actual_alignment_bytes;
    uint8_t task_allocation_expected_operation;
    uint8_t task_allocation_actual_operation;
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
    void *spill_pointer;
    uint64_t retired_generation;
    void *retired_execution_pointer;
} ShadowSpillObjectSnapshot;

typedef struct ShadowSpillObjectLocationSnapshot {
    uint64_t object_id;
    uint64_t size_bytes;
    uint64_t authoritative_version;
    uint64_t version;
    uint64_t allocation_id;
    uint64_t generation;
    uint32_t pool_id;
    uint8_t current;
    uint8_t has_lease;
    void *pointer;
} ShadowSpillObjectLocationSnapshot;

/*
 * Creates one runtime from explicit pool and directed-route registries, a
 * synchronization backend, profiler, and worker. Registry entries are copied;
 * backend problems are borrowed and must outlive the runtime. Pool and route
 * IDs must equal their contiguous registry indices. On failure, output is set to
 * NULL and successfully created resources are reclaimed in reverse order.
 */
SHADOWSPILL_API ShadowSpillStatus shadowspill_runtime_create(
    const ShadowSpillRuntimeConfig *config,
    ShadowSpillRuntime **runtime
);

/*
 * Cold-path capacity reservation for neutral event records. Repeated calls may
 * grow the owner for additional admitted callables at an idle boundary. After
 * the first call, steady execution never falls back to process allocation when
 * the pool is full.
 */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_runtime_reserve_event_leases(
    ShadowSpillRuntime *runtime,
    uint64_t minimum_free_leases
);

/*
 * Cold-path capacity reservation for immutable retirement queue records.
 * Once reserved, queue publication fails closed instead of allocating process
 * memory when the inventory is exhausted.
 */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_runtime_reserve_retirement_records(
    ShadowSpillRuntime *runtime,
    uint64_t minimum_free_records
);

/*
 * Cold-path capacity reservation for one pool's reusable MemoryLease records.
 * The first call seals hot acquisition: later exhaustion fails closed instead
 * of allocating process-heap metadata from an allocator callback.
 */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_runtime_reserve_memory_lease_records(
    ShadowSpillRuntime *runtime,
    uint32_t pool_id,
    uint64_t minimum_free_records
);

SHADOWSPILL_API ShadowSpillStatus shadowspill_plan_create(
    ShadowSpillRuntime *runtime,
    const ShadowSpillPlanDescription *description,
    ShadowSpillPlan **plan
);

SHADOWSPILL_API ShadowSpillStatus shadowspill_plan_close(
    ShadowSpillPlan *plan
);

SHADOWSPILL_API void shadowspill_plan_destroy(ShadowSpillPlan *plan);

/*
 * Rejects new work, drains queued work, synchronizes both transfer streams,
 * joins the worker, and releases owned resources. This call is explicitly
 * synchronizing and idempotent. It returns the first latched failure.
 */
SHADOWSPILL_API ShadowSpillStatus shadowspill_runtime_close(
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
SHADOWSPILL_API ShadowSpillStatus
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
SHADOWSPILL_API ShadowSpillStatus
shadowspill_runtime_transfer_profiles(
    ShadowSpillRuntime *runtime,
    ShadowSpillTransferProfile *profiles,
    uint32_t capacity,
    uint32_t *count,
    uint64_t *generation
);

/* Calls close if needed and releases the runtime record. NULL is accepted. */
SHADOWSPILL_API void shadowspill_runtime_destroy(
    ShadowSpillRuntime *runtime
);

/*
 * Synchronously leases an aligned range from the existing slab; it never grows
 * physical storage. The returned pointer remains valid until logical free and
 * all recorded streams retire it. This call may block only when already
 * pending work can make a suitable range available. Otherwise it returns and
 * latches NO_PROGRESS.
 */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_memory_pool_allocate(
    ShadowSpillRuntime *runtime,
    uint32_t pool_id,
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
SHADOWSPILL_API ShadowSpillStatus
shadowspill_memory_pool_allocation_for_pointer(
    ShadowSpillRuntime *runtime,
    uint32_t pool_id,
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
SHADOWSPILL_API ShadowSpillStatus shadowspill_memory_pool_free(
    ShadowSpillRuntime *runtime,
    uint32_t pool_id,
    uint64_t allocation_id,
    ShadowSpillBackendStream stream
);

/* Adds a borrowed stream token to an allocation's retirement set. */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_memory_pool_record_stream(
    ShadowSpillRuntime *runtime,
    uint32_t pool_id,
    uint64_t allocation_id,
    ShadowSpillBackendStream stream
);

/*
 * Registers one logical alias group. The description is borrowed for this
 * call. Requested initial spill storage is leased from the configured spill pool.
 */
SHADOWSPILL_API ShadowSpillStatus shadowspill_register_object(
    ShadowSpillRuntime *runtime,
    const ShadowSpillObjectDescription *description
);

/*
 * Removes a SPILL_ONLY or RELEASED object with no live allocation or queued
 * action, reclaiming retained spill storage. Intended for deterministic plan
 * teardown after final writeback.
 */
SHADOWSPILL_API ShadowSpillStatus shadowspill_unregister_object(
    ShadowSpillRuntime *runtime,
    uint64_t object_id
);

/*
 * Changes the public identity of one idle SPILL_ONLY or RELEASED object
 * without moving any pool lease or payload. This is used by framework
 * adapters to transfer a preloaded generic lease into and out of a resolved
 * execution plan. No task record or queued action may reference the
 * object while it is rekeyed.
 */
SHADOWSPILL_API ShadowSpillStatus shadowspill_rekey_object(
    ShadowSpillRuntime *runtime,
    uint64_t object_id,
    uint64_t replacement_object_id
);

/*
 * Copies one exact object payload into its existing lease in pool_id. The
 * location must be the object's current authoritative generation. Source is
 * borrowed for the call and may be NULL only for a zero-size object.
 */
SHADOWSPILL_API ShadowSpillStatus shadowspill_write_object(
    ShadowSpillRuntime *runtime,
    uint64_t object_id,
    uint32_t pool_id,
    const void *source,
    uint64_t bytes
);

/*
 * Copies one exact, current object payload from pool_id into caller-owned
 * memory. This function does not wait for transfers; callers first use
 * wait_idle or an equivalent lifecycle boundary. Destination may be NULL only
 * for zero size.
 */
SHADOWSPILL_API ShadowSpillStatus shadowspill_read_object(
    ShadowSpillRuntime *runtime,
    uint64_t object_id,
    uint32_t pool_id,
    void *destination,
    uint64_t bytes
);

/* Admit one immutable task and return its direct repeated-path handle. */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_plan_admit_task(
    ShadowSpillPlan *plan,
    const ShadowSpillTaskDescription *description,
    const ShadowSpillTaskHandle **handle
);

/* Borrow immutable identity already resolved by task admission. */
SHADOWSPILL_API uint64_t shadowspill_task_id(
    const ShadowSpillTaskHandle *handle
);

SHADOWSPILL_API const char *shadowspill_task_trace_label(
    const ShadowSpillTaskHandle *handle
);

/* Cold-path initial publication through one plan-local object binding. */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_plan_publish_initial_allocation(
    ShadowSpillPlan *plan,
    uint64_t plan_object_id,
    const void *pointer,
    ShadowSpillObjectBinding *binding
);

/*
 * Publish one framework allocation through a predecoded task-owned record.
 * The logical object is stable; REPLACE changes only its physical lease and
 * generation. This call is valid only inside the matching active task scope.
 */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_task_publish_allocation(
    ShadowSpillRuntime *runtime,
    const ShadowSpillTaskHandle *handle,
    uint32_t publication_ordinal,
    const void *pointer,
    ShadowSpillObjectBinding *binding
);

/* Validate a current or just-retired view through the same direct record. */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_task_validate_replacement_binding(
    ShadowSpillRuntime *runtime,
    const ShadowSpillTaskHandle *handle,
    uint32_t publication_ordinal,
    const void *retired_pointer,
    const void *successor_pointer
);

SHADOWSPILL_API ShadowSpillStatus
shadowspill_plan_clear_tasks(ShadowSpillPlan *plan);

/*
 * Actively wait until only this plan has no claimed task scope, queued action,
 * or task-owned retirement. Work admitted by other plans does not participate.
 */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_plan_wait_idle(ShadowSpillPlan *plan);

/*
 * Copies and validates one immutable physical-layout certificate and reserves
 * its single parent slice. Task and action identities are resolved when
 * shadowspill_plan_seal_fixed_layout() is called after task admission.
 */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_plan_admit_fixed_layout(
    ShadowSpillPlan *plan,
    const ShadowSpillFixedLayoutDescription *description
);

SHADOWSPILL_API ShadowSpillStatus
shadowspill_plan_seal_fixed_layout(ShadowSpillPlan *plan);

/*
 * Admit one immutable ordered object set for non-execution acquisition, such
 * as returning public outputs to a frontend. Duplicate identities are
 * expanded from one retained snapshot and one readiness wait. The borrowed
 * handle remains valid until the plan is cleared or destroyed.
 */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_plan_admit_object_acquisition(
    ShadowSpillPlan *plan,
    const uint64_t *object_ids,
    uint32_t object_count,
    const ShadowSpillObjectAcquisitionHandle **handle
);

/* Hand one acquired ordinal to caller ownership through its direct object. */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_transfer_acquired_object_to_caller(
    ShadowSpillRuntime *runtime,
    const ShadowSpillObjectAcquisitionHandle *handle,
    uint32_t object_ordinal,
    ShadowSpillBackendStream consumer_stream,
    const void *expected_pointer,
    uint64_t expected_generation,
    uint64_t expected_allocation_id,
    ShadowSpillAllocation *allocation
);

/* Admit an immutable action-only trigger batch without creating a task. */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_plan_admit_action_batch(
    ShadowSpillPlan *plan,
    uint64_t batch_id,
    const ShadowSpillRuntimeAction *actions,
    uint32_t action_count,
    const ShadowSpillActionBatchHandle **handle
);

/* Publish an admitted batch and wait only for worker submission acknowledgement. */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_submit_action_batch_handle(
    ShadowSpillRuntime *runtime,
    const ShadowSpillActionBatchHandle *handle,
    ShadowSpillBackendStream trigger_stream
);

/*
 * Snapshot an admitted object set and insert any published readiness-event
 * waits on consumer_stream. This does not open an allocation or task scope.
 */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_acquire_objects_handle(
    ShadowSpillRuntime *runtime,
    const ShadowSpillObjectAcquisitionHandle *handle,
    ShadowSpillBackendStream consumer_stream,
    ShadowSpillObjectBinding *bindings,
    uint32_t binding_capacity
);

/* Repeated hot path over the task handle returned by plan admission. */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_before_task_handle(
    ShadowSpillRuntime *runtime,
    const ShadowSpillTaskHandle *handle,
    ShadowSpillBackendStream compute_stream,
    const ShadowSpillObjectBinding **bindings,
    uint32_t *binding_count
);

SHADOWSPILL_API ShadowSpillStatus
shadowspill_after_task_handle(
    ShadowSpillRuntime *runtime,
    const ShadowSpillTaskHandle *handle,
    ShadowSpillBackendStream compute_stream
);

/*
 * Clears the calling thread's active task scope after frontend execution
 * aborts before after_task. This does not cancel already submitted device work.
 */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_abort_task_handle(
    ShadowSpillRuntime *runtime,
    const ShadowSpillTaskHandle *handle
);

/*
 * Attribute allocator activity to one non-execution scope. This is used by
 * structural profiling and other isolated measurements that need causal
 * retirement fences without pretending to execute an admitted task. The end
 * call records a completion event only when the scope retired allocations.
 */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_allocation_scope_begin(
    ShadowSpillRuntime *runtime,
    uint32_t pool_id,
    uint64_t scope_id
);

SHADOWSPILL_API ShadowSpillStatus
shadowspill_allocation_scope_end(
    ShadowSpillRuntime *runtime,
    uint64_t scope_id,
    ShadowSpillBackendStream stream
);

SHADOWSPILL_API void shadowspill_allocation_scope_abort(
    ShadowSpillRuntime *runtime
);

/*
 * Starts one bounded allocation-lifetime capture. Storage is allocated before
 * capture begins, so allocator callbacks only append fixed-size records. A
 * full buffer latches ALLOCATION_FAILURE rather than silently losing evidence.
 */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_allocation_telemetry_start(
    ShadowSpillRuntime *runtime,
    uint64_t capacity
);

/* Stops capture. Previously recorded events remain readable until restart. */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_allocation_telemetry_stop(ShadowSpillRuntime *runtime);

/*
 * Copies the complete ordered event stream. Pass events=NULL and capacity=0
 * to query count. Caller owns the destination and no runtime pointer escapes.
 */
SHADOWSPILL_API ShadowSpillStatus
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
SHADOWSPILL_API ShadowSpillStatus shadowspill_trace_prepare(
    ShadowSpillRuntime *runtime,
    const ShadowSpillTraceConfig *config
);

/* Begins one prepared trace and its allocation-lifetime capture. */
SHADOWSPILL_API ShadowSpillStatus shadowspill_trace_begin(
    ShadowSpillRuntime *runtime,
    uint64_t step_id
);

/*
 * Stops appending without synchronizing a stream or worker. Callers establish
 * their required completion boundary before ending the trace.
 */
SHADOWSPILL_API ShadowSpillStatus shadowspill_trace_end(
    ShadowSpillRuntime *runtime
);

/*
 * Copies one stopped trace into caller-owned arrays. NULL arrays with zero
 * capacities query the required counts through summary. No runtime pointer or
 * backend handle escapes.
 */
SHADOWSPILL_API ShadowSpillStatus shadowspill_trace_read(
    ShadowSpillRuntime *runtime,
    ShadowSpillTraceSummary *summary,
    ShadowSpillTraceEvent *events,
    uint64_t event_capacity,
    ShadowSpillAllocationEvent *allocation_events,
    uint64_t allocation_event_capacity
);

/* Explicitly synchronizing test/checkpoint helper; returns first failure. */
SHADOWSPILL_API ShadowSpillStatus shadowspill_runtime_wait_idle(
    ShadowSpillRuntime *runtime
);

/*
 * Clears a latched NO_PROGRESS allocation failure after every external
 * producer stream has been synchronized and the failed allocator caller has
 * returned. This exists only for deterministic fault teardown: it allows the
 * worker to drain already-owned actions so objects and pool leases can be
 * reclaimed. Every other failure remains permanently latched.
 */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_runtime_recover_no_progress(ShadowSpillRuntime *runtime);

/*
 * Planning-only growth of the one pinned-host arena. The runtime must be idle;
 * existing object offsets and payloads are preserved. The backend host pointer
 * is CPU-addressable by contract. Shrinkage is rejected. This operation owns
 * both arenas briefly, so callers must include that transient in host-budget
 * admission. It is forbidden after frontend physical admission is sealed.
 */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_memory_pool_grow(
    ShadowSpillRuntime *runtime,
    uint32_t pool_id,
    uint64_t capacity_bytes
);

/* Copies a lock-consistent telemetry snapshot into caller-owned storage. */
SHADOWSPILL_API ShadowSpillStatus shadowspill_runtime_statistics(
    ShadowSpillRuntime *runtime,
    ShadowSpillRuntimeStatistics *statistics
);

/* Copies the immutable first-failure snapshot; status is OK before failure. */
SHADOWSPILL_API ShadowSpillStatus shadowspill_runtime_failure(
    ShadowSpillRuntime *runtime,
    ShadowSpillRuntimeFailure *failure
);

/* Copies one object's current logical state into caller-owned storage. */
SHADOWSPILL_API ShadowSpillStatus shadowspill_object_snapshot(
    ShadowSpillRuntime *runtime,
    uint64_t object_id,
    ShadowSpillObjectSnapshot *snapshot
);

/* Copies one object's current location in an explicitly selected pool. */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_object_location_snapshot(
    ShadowSpillRuntime *runtime,
    uint64_t object_id,
    uint32_t pool_id,
    ShadowSpillObjectLocationSnapshot *snapshot
);

/* One sentence naming the condition behind a status. */
SHADOWSPILL_API const char *shadowspill_failure_reason_string(
    ShadowSpillFailureReason reason
);

#ifdef __cplusplus
}
#endif

#endif
