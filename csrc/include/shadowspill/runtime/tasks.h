/* Task boundaries, and the allocation scopes outside them. */

#ifndef SHADOWSPILL_RUNTIME_TASKS_H
#define SHADOWSPILL_RUNTIME_TASKS_H

#include <stdint.h>

#include <shadowspill/shadowspill.h>
#include <shadowspill/backend.h>
#include <shadowspill/runtime/vocabulary.h>
#include <shadowspill/runtime/descriptions.h>

#ifdef __cplusplus
extern "C" {
#endif

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
    uint64_t trigger_stream_handle
);

/*
 * Snapshot an admitted object set and insert any published readiness-event
 * waits on consumer_stream. This does not open an allocation or task scope.
 */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_acquire_objects_handle(
    ShadowSpillRuntime *runtime,
    const ShadowSpillObjectAcquisitionHandle *handle,
    uint64_t consumer_stream_handle,
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

#ifdef __cplusplus
}
#endif

#endif
