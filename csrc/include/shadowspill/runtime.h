#ifndef SHADOWSPILL_RUNTIME_H
#define SHADOWSPILL_RUNTIME_H

#include <stdint.h>

#include <shadowspill/shadowspill.h>

#include <shadowspill/backend.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SHADOWSPILL_RUNTIME_TRACE_LABEL_MAX_BYTES 1024U
#define SHADOWSPILL_RUNTIME_NO_ID UINT64_MAX

/*
 * The scope a runtime-owned object's storage is attributed to. No task makes it
 * and no plan owns it -- the runtime does, when an object is registered -- and
 * any number of plans may then bind that object, so it outlives all of them.
 *
 * Distinct from SHADOWSPILL_RUNTIME_NO_ID, which means no scope was open at all:
 * a provider taking its own workspace between tasks. Both belong to no plan, but
 * only this one backs something the program named, and the two want telling
 * apart when reading what a pool holds.
 */
#define SHADOWSPILL_RUNTIME_OBJECT_SCOPE_ID (UINT64_MAX - UINT64_C(1))

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

/* Execution statuses are in the shared vocabulary; see <shadowspill/status.h>. */

/* ------------------------------------------------------------------------
 * Vocabulary
 *
 * Statuses, reasons, and the enumerations every description below
 * uses. Nothing here allocates or holds state.
 */

/*
 * Why an operation failed, where the status alone does not say.
 *
 * The status is the coarse class a caller acts on; the reason names the
 * specific condition, so a report can explain itself. Several of these sit
 * under one status on purpose - a lease that cannot be released and a process
 * allocator that refuses a record are both internal failures, and a caller
 * treats them alike, but a reader must be able to tell them apart.
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
    /* A backend event could not be released back to its pool. */
    SHADOWSPILL_FAILURE_REASON_EVENT_RELEASE_REJECTED = 7,
    /* Stream-use records could not be returned to the pool that lent them. */
    SHADOWSPILL_FAILURE_REASON_USE_RECORD_RETURN_REJECTED = 8,
    /* The backend refused a stream or event operation. */
    SHADOWSPILL_FAILURE_REASON_BACKEND_CALL_REJECTED = 9,
    /* An object was not in the residency, version or lease state the plan
     * requires at this point in the step. */
    SHADOWSPILL_FAILURE_REASON_OBJECT_STATE_REJECTED = 10,
    /* A task's retirement event could not be published, so its allocations
     * have no completion source to retire against. */
    SHADOWSPILL_FAILURE_REASON_RETIREMENT_PUBLICATION_REJECTED = 11,
    /* A retirement could not be queued for the worker to finish. */
    SHADOWSPILL_FAILURE_REASON_RETIREMENT_ENQUEUE_REJECTED = 12,
    /* A task boundary did not complete; the status says which step failed. */
    SHADOWSPILL_FAILURE_REASON_TASK_BOUNDARY_REJECTED = 13,
    /* A planned task allocation could not be placed at its fixed offset. */
    SHADOWSPILL_FAILURE_REASON_TASK_ALLOCATION_REJECTED = 14,
} ShadowSpillFailureReason;

typedef enum ShadowSpillObjectResidency {
    SHADOWSPILL_OBJECT_SPILL_ONLY = 0,
    SHADOWSPILL_OBJECT_EXECUTION_READY = 1,
    SHADOWSPILL_OBJECT_FETCHING = 2,
    SHADOWSPILL_OBJECT_EVICTING = 3,
    SHADOWSPILL_OBJECT_RELEASED = 4,
} ShadowSpillObjectResidency;

/*
 * What a plan asks of one object at a task boundary. A release drops the
 * execution copy; an evict copies it to the spill pool and then drops it; a
 * fetch copies the spill copy into the execution pool; a write-back copies
 * the execution copy to the spill pool and keeps it, so the spill copy is
 * current again and a later release costs nothing. A write-back whose spill
 * copy is already current completes without a copy. A release behind a
 * pending write-back of its object frees the execution copy once that copy
 * has landed.
 */
typedef enum ShadowSpillRuntimeActionKind {
    SHADOWSPILL_RUNTIME_RELEASE = 0,
    SHADOWSPILL_RUNTIME_EVICT = 1,
    SHADOWSPILL_RUNTIME_FETCH = 2,
    SHADOWSPILL_RUNTIME_WRITE_BACK = 3,
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

/* ------------------------------------------------------------------------
 * Descriptions
 *
 * What a caller fills in to admit something: pools, routes, a
 * runtime, a plan, an object, a task, a fixed layout. Every one is
 * borrowed for the call that reads it.
 */

/* Where a pool's arena lives: device memory the backend allocates, or host
   memory the pool allocates and the backend registers. */
typedef enum ShadowSpillPoolKind {
    SHADOWSPILL_POOL_DEVICE = 0,
    SHADOWSPILL_POOL_PINNED_HOST = 1
} ShadowSpillPoolKind;

typedef struct ShadowSpillMemoryPoolDescription {
    uint32_t pool_id;
    uint8_t kind;
    uint64_t capacity_bytes;
    uint64_t minimum_alignment;
} ShadowSpillMemoryPoolDescription;

/* A directed copy path between a pinned-host pool and the device pool. The
   runtime derives the copy direction from the two pools' kinds and owns the
   lane it dispatches on. */
typedef struct ShadowSpillTransferRouteDescription {
    uint32_t route_id;
    const char *name;
    uint32_t source_pool_id;
    uint32_t destination_pool_id;
} ShadowSpillTransferRouteDescription;

typedef struct ShadowSpillRuntimeConfig {
    uint32_t abi_version;
    /* The backend every pool, route, and event is built on; copied at create. */
    const ShadowSpillBackend *backend;
    const ShadowSpillMemoryPoolDescription *pools;
    uint32_t pool_count;
    const ShadowSpillTransferRouteDescription *routes;
    uint32_t route_count;
    uint64_t worker_poll_nanoseconds;
    /* How far a lane may run ahead with background transfers: copies the
       plan did not schedule (an opening restore, a reconciliation) are
       dispatched only while the lane holds fewer than this many of their
       bytes in flight, so a transfer the plan did schedule never waits
       behind more than this. Zero removes the bound. A single background
       copy larger than the window runs alone. */
    uint64_t background_transfer_window_bytes;
} ShadowSpillRuntimeConfig;

/*
 * Immutable pool and route roles selected by one admitted plan. Multiple plans
 * may share a runtime topology and runtime-owned logical objects, which is why
 * each carries an id of its own.
 */
typedef struct ShadowSpillPlanDescription {
    /*
     * This plan's identity, chosen by the caller and never interpreted here. It
     * is carried onto every lease the plan's tasks make, so a pool shared by
     * several plans can still say which one a range belongs to -- task ids
     * cannot, being plan-local and therefore shared between plans.
     *
     * The caller names this same id on every allocation scope it opens for the
     * plan, so one number spans the whole period the plan owns, the profiling
     * that precedes its tasks included.
     *
     * Zero is not a valid plan id: it is what a lease made outside any plan
     * reports, so no plan may hold it.
     */
    uint64_t plan_id;
    uint32_t execution_pool_id;
    uint32_t spill_pool_id;
    uint32_t fetch_route_id;
    uint32_t evict_route_id;
} ShadowSpillPlanDescription;

typedef enum ShadowSpillObjectConsistency {
    SHADOWSPILL_OBJECT_CAUSAL = 0,
    SHADOWSPILL_OBJECT_UNORDERED = 1,
} ShadowSpillObjectConsistency;

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
     * storage. Anonymous/provider operations are optional invariant observations:
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
     * invariant contract. Zero preserves strict exact-contract behavior.
     */
    uint64_t dynamic_scratch_maximum_allocation_bytes;
    uint64_t dynamic_scratch_live_limit_bytes;
} ShadowSpillTaskDescription;

/* ------------------------------------------------------------------------
 * Diagnostic records
 *
 * What the runtime hands back when asked: allocation and trace
 * events, transfer profiles, statistics, failures, snapshots.
 * Caller-owned buffers, filled in place.
 */

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

/* A stream timestamp the trace could not measure. */
#define SHADOWSPILL_TRACE_NO_STREAM_TIME UINT64_MAX

/*
 * One observation emitted by the neutral runtime. ``timestamp_ns`` is the
 * host clock at the moment the runtime recorded it. ``detail_0`` and
 * ``detail_1`` have event-specific meanings documented in runtime.md. IDs use
 * SHADOWSPILL_RUNTIME_NO_ID when they do not apply.
 *
 * A TRANSFER_COMPLETED event also carries the transfer's interval on the
 * device: ``lane_started_at_ns`` and ``lane_finished_at_ns`` are measured on the
 * transfer lane from the origin event the trace was begun with, so they sit
 * on the same timeline as any other event measured from that origin. Both
 * hold SHADOWSPILL_TRACE_NO_STREAM_TIME when the trace has no origin or the
 * backend could not measure the interval; every other kind carries that
 * value always.
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
    uint64_t lane_started_at_ns;
    uint64_t lane_finished_at_ns;
    uint8_t kind;
} ShadowSpillTraceEvent;

typedef struct ShadowSpillTraceSummary {
    uint32_t abi_version;
    uint64_t step_id;
    uint64_t event_count;
    uint64_t allocation_event_count;
    uint64_t event_capacity;
    uint64_t allocation_event_capacity;
    uint64_t began_at_ns;
    uint64_t ended_at_ns;
    uint8_t active;
    uint8_t event_overflow;
    uint8_t allocation_event_overflow;
} ShadowSpillTraceSummary;

/*
 * What one pool holds, asked of that pool. A runtime may own any number of
 * pools, and which of them a given plan uses as its execution and spill pools is
 * the plan's choice, so a pool's own numbers are read per pool rather than
 * flattened into named fields for two of them.
 *
 * `free_bytes` less `largest_free_range_bytes` is `external_fragmentation_bytes`,
 * the number a contiguous-range refusal turns on: the free total says a request
 * should fit, the largest range says whether it does.
 */
typedef struct ShadowSpillMemoryPoolStatistics {
    uint32_t pool_id;
    /* The pool kind from its description: device or pinned host. */
    uint8_t kind;
    uint64_t capacity_bytes;
    uint64_t requested_allocated_bytes;
    uint64_t peak_requested_allocated_bytes;
    uint64_t allocated_bytes;
    uint64_t peak_allocated_bytes;
    uint64_t free_bytes;
    uint64_t free_prefix_bytes;
    uint64_t largest_free_range_bytes;
    uint64_t external_fragmentation_bytes;
    uint64_t live_allocations;
    uint64_t blocked_allocators;
    uint64_t memory_lease_record_capacity;
    uint64_t memory_lease_record_in_use;
    uint64_t memory_lease_record_peak_in_use;
    uint64_t memory_lease_record_growth_rejections;
    uint64_t lease_use_record_capacity;
    uint64_t lease_use_record_in_use;
    uint64_t lease_use_record_peak_in_use;
    uint64_t lease_use_record_growth_rejections;
} ShadowSpillMemoryPoolStatistics;

/*
 * What the runtime holds that no pool does: the work in flight, the records it
 * owns, and how many pools there are to ask about. Anything a pool knows about
 * itself is in `ShadowSpillMemoryPoolStatistics`.
 */
typedef struct ShadowSpillRuntimeStatistics {
    /* Pools are `pool_id` 0 through `pool_count - 1`. */
    uint32_t pool_count;
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
    /* Backend events created for leases: every one after sealing is a
       steady-state driver call the plan did not reserve for. */
    uint64_t event_lease_driver_creates;
    uint64_t event_lease_sealed;
    uint64_t timing_event_capacity;
    uint64_t timing_event_in_use;
    uint64_t timing_event_peak_in_use;
    uint64_t timing_event_driver_creates;
    uint64_t retirement_record_capacity;
    uint64_t retirement_record_in_use;
    uint64_t retirement_record_peak_in_use;
    uint64_t retirement_record_growth_rejections;
    /* Framework-owned plan outputs that still reference pool storage. */
    uint64_t caller_owned_allocations;
} ShadowSpillRuntimeStatistics;

/*
 * One live allocation, as `shadowspill_memory_pool_live_allocations` reports
 * it. Every field here is already recorded on the lease; this is the shape a
 * caller reads them in.
 *
 * The three flags are what separate an allocation that is merely alive from one
 * that has outlived the scope which made it. `scratch` was requested as task
 * workspace rather than as a planned object; `plan_owned` is the converse, a
 * range the plan placed. `logical_freed` means the frontend has already given
 * it up and only retirement is outstanding, so it is not a survivor at all.
 */
typedef struct ShadowSpillLiveAllocation {
    uint64_t allocation_id;
    /* Byte offset into the pool's arena. Position, not size, is what explains
     * a contiguous-range refusal, so this is the field to sort on. */
    uint64_t offset;
    uint64_t charged_bytes;
    uint64_t requested_bytes;
    /* The plan whose scope made it, as named in `ShadowSpillPlanDescription`
     * or at `shadowspill_allocation_scope_begin`. Zero when none was named.
     * A pool outlives any one plan, so this is what separates a range left
     * behind by an earlier plan from one the current plan made. */
    uint64_t origin_plan_id;
    /* The scope that made it: SHADOWSPILL_RUNTIME_NO_ID when no scope was open,
     * or SHADOWSPILL_RUNTIME_OBJECT_SCOPE_ID for a runtime-owned object's
     * storage, which belongs to the runtime rather than to any plan. */
    uint64_t origin_task_id;
    uint64_t origin_task_invocation;
    uint64_t origin_task_allocation_ordinal;
    /* The object this range is bound to, or SHADOWSPILL_RUNTIME_NO_ID when it
     * is bound to none. The runtime does not know what an object is *for* --
     * a frontend that has the program can resolve the id to a role, which is
     * why the id travels rather than a classification. */
    uint64_t object_id;
    uint32_t references;
    uint8_t scratch;
    uint8_t plan_owned;
    /* Whether the plan ever owned it, which stays set after ownership moves on.
     * `ever_plan_owned` without `plan_owned` is a range promoted out to a named
     * owner -- an output the caller holds now -- and no longer the plan's to
     * release. */
    uint8_t ever_plan_owned;
    uint8_t logical_freed;
    uint8_t framework_free_seen;
} ShadowSpillLiveAllocation;

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

/* ------------------------------------------------------------------------
 * Runtime and plan lifecycle
 *
 * Create a runtime, reserve the records it must never allocate on a
 * hot path, open and close plans, and tear it all down.
 */

/* Whether a backend table carries the contract version and every required
   entry (the profiler entries may be NULL). */
SHADOWSPILL_API int shadowspill_backend_is_valid(const ShadowSpillBackend *backend);

/*
 * Creates one runtime from explicit pool and directed-route registries over
 * one backend. Registry entries and the backend table are copied; the provider
 * state the table names is borrowed and must outlive the runtime. Pool and
 * route IDs must equal their contiguous registry indices. On failure, output
 * is set to NULL and whatever was created is reclaimed.
 */
SHADOWSPILL_API ShadowSpillStatus shadowspill_runtime_create(
    const ShadowSpillRuntimeConfig *config,
    ShadowSpillRuntime **runtime
);

/*
 * Cold-path capacity reservation for neutral event records. Repeated calls
 * grow the pool for additional admitted plans, each waiting for an idle
 * boundary first. After the first call, steady execution never falls back to
 * process allocation when the pool is full.
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

/*
 * What became of one plan id. A lease can outlive the plan that made it, so the
 * id on a lease is answerable after the plan itself is gone -- CLOSED is how a
 * range left behind by earlier work is told from one the current plan made.
 */
typedef enum ShadowSpillPlanState {
    /* No plan was ever created with this id. */
    SHADOWSPILL_PLAN_STATE_UNKNOWN = 0,
    /* Created and open: it may still admit work. */
    SHADOWSPILL_PLAN_STATE_LIVE = 1,
    /* Closed, record still present: it admits no further work. */
    SHADOWSPILL_PLAN_STATE_CLOSED = 2,
    /* Closed and the record freed. The id stays claimed, because a lease may
     * still carry it. */
    SHADOWSPILL_PLAN_STATE_DESTROYED = 3
} ShadowSpillPlanState;

SHADOWSPILL_API ShadowSpillStatus shadowspill_runtime_plan_state(
    ShadowSpillRuntime *runtime,
    uint64_t plan_id,
    ShadowSpillPlanState *state
);

/*
 * The plan one id names, or NULL when no record holds it -- the id was never
 * created with, or its plan has been destroyed. Valid only while the caller
 * knows the plan is live, since a concurrent destroy would leave it dangling;
 * ask `shadowspill_runtime_plan_state` to learn what became of an id without
 * touching the record.
 */
SHADOWSPILL_API ShadowSpillPlan *shadowspill_runtime_plan(
    ShadowSpillRuntime *runtime,
    uint64_t plan_id
);

/*
 * Take the next plan id for this runtime. This only ever counts up, so an id is
 * unique for the life of the runtime and is not reissued when the plan holding
 * it is destroyed. A lease is therefore never readable as belonging to a later
 * plan that happens to have been given the same number.
 *
 * Taken before `shadowspill_plan_create` rather than returned by it, so the
 * caller can name the id on the allocation scopes it opens first.
 *
 * An id may be used for one plan only. Plan creation enforces that outright:
 * an id this did not issue is refused, and so is one that has already been
 * created with, whether that plan is still live or has since been closed.
 */
SHADOWSPILL_API ShadowSpillStatus shadowspill_runtime_next_plan_id(
    ShadowSpillRuntime *runtime,
    uint64_t *plan_id
);

/*
 * `description->plan_id` must be an id from `shadowspill_runtime_next_plan_id`
 * that no live plan on this runtime already holds; anything else is
 * INVALID_ARGUMENT.
 */
SHADOWSPILL_API ShadowSpillStatus shadowspill_plan_create(
    ShadowSpillRuntime *runtime,
    const ShadowSpillPlanDescription *description,
    ShadowSpillPlan **plan
);

/*
 * Take back every range this plan's own scopes allocated, whatever still points
 * at it, and write how many were reclaimed.
 *
 * A plan owes the pool every range its tasks and its profiling probes made, and
 * this is how the runtime collects when nothing else has. It works in leases and
 * bytes, which is what the runtime owns; it does not consult the framework, and
 * cannot, so a caller that may still read an object backed by one of these ranges
 * must drop it first. That is what makes this the forcing path rather than the
 * ordinary one, where a lease goes when the framework's own reference count
 * reaches zero.
 *
 * Ranges carrying no plan -- a provider taking its own workspace between tasks --
 * belong to no plan and are left alone.
 *
 * A lease the framework has not freed yet keeps its pointer and id indexed, so the
 * free that eventually arrives still resolves instead of faulting.
 */
SHADOWSPILL_API ShadowSpillStatus shadowspill_plan_reclaim_scoped_leases(
    ShadowSpillPlan *plan,
    uint64_t *reclaimed
);

SHADOWSPILL_API ShadowSpillStatus shadowspill_plan_close(
    ShadowSpillPlan *plan
);

SHADOWSPILL_API void shadowspill_plan_destroy(ShadowSpillPlan *plan);

/*
 * Rejects new work, drains queued work, synchronizes every route's lane, joins
 * the worker, and releases owned resources. This call is explicitly
 * synchronizing and idempotent. It returns the first latched failure.
 */
SHADOWSPILL_API ShadowSpillStatus shadowspill_runtime_close(
    ShadowSpillRuntime *runtime
);

/*
 * Closes without waiting for anything: no drain, no stream synchronization.
 * For a process that is already going away, where waiting is both pointless
 * and unbounded. Outstanding work cannot complete once the process is
 * exiting, and everything the drain protects -- device allocations, pinned
 * registrations, the context itself -- is reclaimed at process exit anyway,
 * so this stops the worker, releases what it owns, and returns.
 *
 * The counts a caller passes are filled in before anything is stopped, so a
 * caller can report what was still outstanding. Either may be NULL.
 */
SHADOWSPILL_API ShadowSpillStatus shadowspill_runtime_abandon(
    ShadowSpillRuntime *runtime,
    uint64_t *outstanding_actions,
    uint64_t *outstanding_retirements
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

/* ------------------------------------------------------------------------
 * Pools and allocation
 *
 * Serving one allocation, freeing it, and naming the stream that
 * used it. Called from whichever thread is dispatching.
 */

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
 * free until a plan action releases or evicts the owning object.
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

/* ------------------------------------------------------------------------
 * Objects
 *
 * Registering the logical values a plan names, reading and writing
 * them, and holding handles onto them across generations.
 */

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
 * Bind one plan-local identity to a retained runtime object handle. Equal
 * plan-local IDs in different plans have no relationship unless both bindings
 * use handles for the same runtime object.
 */
SHADOWSPILL_API ShadowSpillStatus shadowspill_plan_bind_object(
    ShadowSpillPlan *plan,
    uint64_t plan_object_id,
    const ShadowSpillObjectHandle *object,
    uint8_t consistency
);

/*
 * Registers one logical object. The description is borrowed for this call. An
 * initially resident object is leased storage in the pool `initial_pool_id`
 * names; one that is not resident holds no lease until a task publishes into
 * it.
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

/* ------------------------------------------------------------------------
 * Admitting a plan
 *
 * Tasks, initial allocations, the fixed layout, object acquisitions
 * and action batches. All of it before the first step runs.
 */

/* Admit one immutable task and return its direct repeated-path handle. */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_plan_admit_task(
    ShadowSpillPlan *plan,
    const ShadowSpillTaskDescription *description,
    const ShadowSpillTaskHandle **handle
);

/* The id a plan was created with, the inverse of `shadowspill_runtime_plan`. */
SHADOWSPILL_API uint64_t shadowspill_plan_id(
    const ShadowSpillPlan *plan
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
 * Actively wait until this plan has no claimed task scope, queued action or
 * task-owned retirement. Work admitted by other plans does not participate.
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

/* ------------------------------------------------------------------------
 * Task boundaries
 *
 * The two calls every planned task runs between, plus the abort that
 * closes a scope the caller cannot close normally. See
 * docs/architecture/task-boundaries.md.
 */

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

/*
 * Resolves every planned allocation's range-reuse dependency for one task,
 * so the wait belongs to the task boundary rather than to the allocation
 * inside the task that first needs the range.
 */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_wait_task_allocations_handle(
    ShadowSpillRuntime *runtime,
    const ShadowSpillTaskHandle *handle,
    ShadowSpillBackendStream compute_stream
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

/* ------------------------------------------------------------------------
 * Allocation scopes
 *
 * Attributing allocations made outside any task - profiling probes,
 * warmup - to a scope that can be closed and audited.
 */

/*
 * Attribute allocator activity to one non-execution scope. This is used by
 * structural profiling and other isolated measurements that need causal
 * retirement fences without pretending to execute an admitted task. The end
 * call records a completion event only when the scope retired allocations.
 *
 * `plan_id` names the plan the measurement is for. A scope deliberately runs
 * outside any task, so there is no task here for the runtime to read a plan
 * from, and the caller names it instead. Without it an allocation made here
 * would be attributable to nothing -- which is the case these ids remove, since
 * a profiling probe's retained workspace can outlive the scope that made it.
 */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_allocation_scope_begin(
    ShadowSpillRuntime *runtime,
    uint32_t pool_id,
    uint64_t plan_id,
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

/* ------------------------------------------------------------------------
 * Telemetry and tracing
 *
 * Two bounded rings, both off by default and both diagnostic: a step
 * never depends on either, and a full ring stops recording rather
 * than stopping the runtime.
 */

/*
 * Starts one bounded allocation-lifetime capture. Storage is allocated before
 * capture begins, so allocator callbacks only append fixed-size records. A
 * full buffer stops recording and counts what it could not keep, so the gap
 * is visible in the read-back rather than silently lost; the step itself is
 * never affected.
 */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_allocation_telemetry_start(
    ShadowSpillRuntime *runtime,
    uint64_t capacity
);

/* Stops capture. Events already recorded stay readable until the next start. */
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

/*
 * Begins one prepared trace and its allocation-lifetime capture.
 *
 * ``origin_event`` is a caller-owned timing event already recorded on the
 * caller's compute stream; transfer intervals in the trace are measured from
 * it, so they share the caller's timeline. It must outlive the trace and is
 * never destroyed here. A zero token records no stream intervals.
 */
SHADOWSPILL_API ShadowSpillStatus shadowspill_trace_begin(
    ShadowSpillRuntime *runtime,
    uint64_t step_id,
    ShadowSpillBackendEvent origin_event
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

/* ------------------------------------------------------------------------
 * Waiting, recovery and inspection
 *
 * Draining outstanding work, recovering from a no-progress stall,
 * growing a pool, and asking what happened.
 */

/* Blocks until no action is queued and no retirement is pending, then returns
   the first latched failure. */
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
 * Planning-only growth of the pool `pool_id` names. The runtime must be idle
 * and hold no in-flight actions or pending retirements; existing object
 * offsets and payloads are preserved, and shrinkage is rejected. The old and
 * new arenas are both held while the payload is copied across, so a caller
 * must include that transient in its own budget.
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

/* What the pool `pool_id` names holds. Unknown pool ids are INVALID_ARGUMENT. */
SHADOWSPILL_API ShadowSpillStatus shadowspill_memory_pool_statistics(
    ShadowSpillRuntime *runtime,
    uint32_t pool_id,
    ShadowSpillMemoryPoolStatistics *statistics
);

/*
 * Copies one entry per live allocation of the pool `pool_id` names into
 * caller-owned storage, in no particular order, and always writes the total
 * live count to `count` so a caller can size its buffer and call again.
 *
 * Statistics answer how many allocations are live; this answers which. A
 * fixed layout refused for want of a contiguous range is not explained by a
 * count, because a few bytes in the wrong place cost the largest free range
 * while leaving the free total almost untouched -- so the offsets are the
 * diagnosis.
 *
 * Writes min(capacity, live) entries and returns OK even when the buffer was
 * too small; `*count` greater than `capacity` is how truncation is reported.
 * `out` may be NULL when `capacity` is zero, which asks only for the count.
 */
SHADOWSPILL_API ShadowSpillStatus shadowspill_memory_pool_live_allocations(
    ShadowSpillRuntime *runtime,
    uint32_t pool_id,
    ShadowSpillLiveAllocation *out,
    uint64_t capacity,
    uint64_t *count
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

/* One sentence for any reason; see shadowspill_status_string() for the status. */
SHADOWSPILL_API const char *shadowspill_failure_reason_string(
    ShadowSpillFailureReason reason
);

#ifdef __cplusplus
}
#endif

#endif
