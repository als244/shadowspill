/* Registering an object, and acquiring it for a task. */

#ifndef SHADOWSPILL_RUNTIME_OBJECTS_H
#define SHADOWSPILL_RUNTIME_OBJECTS_H

#include <stdint.h>

#include <shadowspill/shadowspill.h>
#include <shadowspill/backend.h>
#include <shadowspill/runtime/vocabulary.h>
#include <shadowspill/runtime/descriptions.h>

#ifdef __cplusplus
extern "C" {
#endif

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

#ifdef __cplusplus
}
#endif

#endif
