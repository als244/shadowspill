/* The records the runtime hands back: statistics, failures, snapshots. */

#ifndef SHADOWSPILL_RUNTIME_DIAGNOSTICS_H
#define SHADOWSPILL_RUNTIME_DIAGNOSTICS_H

#include <stdint.h>

#include <shadowspill/shadowspill.h>
#include <shadowspill/backend.h>
#include <shadowspill/runtime/vocabulary.h>
#include <shadowspill/runtime/descriptions.h>

#ifdef __cplusplus
extern "C" {
#endif

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
 * A TRANSFER_COMPLETED event also carries what the transfer did on its lane.
 * ``lane_issued_at_ns``, ``lane_started_at_ns`` and ``lane_finished_at_ns`` are
 * measured from the origin event the trace was begun with, so they sit on the
 * same timeline as any other event measured from that origin.
 *
 * ``lane_issued_at_ns`` is when the runtime handed the transfer to the lane,
 * before any dependency it was given had cleared, and ``lane_started_at_ns`` is
 * when its bytes began moving. **The gap between them is the wait** -- without
 * it a transfer held behind an event reads as a slow one.
 *
 * Each holds SHADOWSPILL_TRACE_NO_STREAM_TIME where the trace has no origin,
 * the lane does not report it, or the backend could not measure it; every other
 * event kind carries that value in all three always.
 *
 * ``timestamp_ns`` is on the host clock and these are on the origin's, and
 * ``ShadowSpillTraceSummary.origin_host_ns`` is what relates them.
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
    uint64_t lane_issued_at_ns;
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
    /* The host clock where the origin event was recorded, which is what puts
       an event's `timestamp_ns` and its `lane_*_ns` on one axis: a lane instant
       plus this is the host instant it happened at. Zero when the trace was
       begun with no origin, and then every lane instant is NO_STREAM_TIME. */
    uint64_t origin_host_ns;
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
    /* Byte offset into the pool's memory. Position, not size, is what explains
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

/*
 * One admitted plan's fixed layout, as `shadowspill_memory_pool_plan_slices`
 * reports it. `offset` and `bytes` are where the plan's own layout lies.
 * `slab_plan_id` is the plan that reserved the range it lies in -- the plan
 * itself, unless its layout was admitted into another's -- and `slab_bytes`
 * that range's size.
 */
typedef struct ShadowSpillPlanSlice {
    uint64_t plan_id;
    uint64_t offset;
    uint64_t bytes;
    uint64_t slab_plan_id;
    uint64_t slab_bytes;
} ShadowSpillPlanSlice;

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

#ifdef __cplusplus
}
#endif

#endif
