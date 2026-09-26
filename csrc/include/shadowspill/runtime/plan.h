/* Admitting a plan: what it needs resident, and what it may move. */

#ifndef SHADOWSPILL_RUNTIME_PLAN_H
#define SHADOWSPILL_RUNTIME_PLAN_H

#include <stdint.h>

#include <shadowspill/shadowspill.h>
#include <shadowspill/backend.h>
#include <shadowspill/runtime/vocabulary.h>
#include <shadowspill/runtime/descriptions.h>
#include <shadowspill/runtime/diagnostics.h>

#ifdef __cplusplus
extern "C" {
#endif

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

/*
 * Admit a layout into the slice `host` holds instead of reserving one: the
 * layout is placed at that slice's offset and must fit inside it. Plans whose
 * layouts share a slice never run at once -- a call begins only once every
 * plan placing into the slice has drained, and only if nothing is live in it
 * (shadowspill_plan_require_empty_layout()) -- so the bytes serve each in turn
 * and are held once. A host that shares a slice itself is followed to the
 * plan that reserved it. That plan's layout cannot be cleared while another
 * is admitted into its slice, so it is cleared last.
 */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_plan_admit_fixed_layout_in(
    ShadowSpillPlan *plan,
    const ShadowSpillFixedLayoutDescription *description,
    ShadowSpillPlan *host
);

SHADOWSPILL_API ShadowSpillStatus
shadowspill_plan_seal_fixed_layout(ShadowSpillPlan *plan);

/*
 * Refuse unless nothing is live inside this plan's fixed layout. A frontend
 * asks between calls, once the work of every plan placing into these bytes
 * has drained: everything a call places is gone by then, the writeback that
 * finishes after the call returns included, so a survivor is something the
 * next call would overwrite. The first one found is latched as a plan
 * violation naming the task that made it, its allocation and its size. A
 * plan admitted without a layout passes.
 */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_plan_require_empty_layout(ShadowSpillPlan *plan);

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

#ifdef __cplusplus
}
#endif

#endif
