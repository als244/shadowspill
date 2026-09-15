/* Statuses, reasons, and the enumerations every description uses. */

#ifndef SHADOWSPILL_RUNTIME_VOCABULARY_H
#define SHADOWSPILL_RUNTIME_VOCABULARY_H

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

#ifdef __cplusplus
}
#endif

#endif
