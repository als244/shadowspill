/* What a caller describes to the runtime: pools, routes, objects, tasks. */

#ifndef SHADOWSPILL_RUNTIME_DESCRIPTIONS_H
#define SHADOWSPILL_RUNTIME_DESCRIPTIONS_H

#include <stdint.h>

#include <shadowspill/shadowspill.h>
#include <shadowspill/backend.h>
#include <shadowspill/runtime/vocabulary.h>

#ifdef __cplusplus
extern "C" {
#endif

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

#ifdef __cplusplus
}
#endif

#endif
