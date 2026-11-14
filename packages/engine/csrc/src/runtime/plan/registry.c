/*
 * Which plan a lease belongs to, answerable after the plan is gone.
 *
 * One slot per plan id ever created with, holding the plan record while it
 * exists. A lease records the id of the plan whose scope made it, and a closing
 * plan is expected to release every one of those, so a live lease naming a
 * closed or destroyed plan is a defect. The id has to stay meaningful after the
 * record is freed for that defect to be visible at all, and because an object
 * shared between plans is meant to outlive any one of them.
 *
 * Plan ids are a dense counter from one, so the slots are indexed by id and the
 * lookup needs no hashing: this is a hash table whose hash is the identity and
 * whose buckets hold one entry each, with the indirection removed.
 *
 * The registry has its own lock and never takes another while holding it, so
 * asking what an id means does not wait behind plan creation or teardown.
 */

#include "../internal.h"

#include <stdlib.h>
#include <string.h>

/*
 * Claim one id for a plan. Zero on success, 1 when some plan already used it,
 * -1 when the registry could not grow. Takes the registry's own lock, which is
 * never held while any other runtime lock is taken.
 */
int shadowspill_plan_registry_claim(
    ShadowSpillRuntime *runtime,
    uint64_t plan_id,
    ShadowSpillPlan *plan
) {
    if (plan_id == 0U || !runtime->plans_by_id.lock_initialized) {
        return 1;
    }
    pthread_mutex_lock(&runtime->plans_by_id.lock);
    int result = 0;
    if (plan_id > runtime->plans_by_id.capacity) {
        /* Ids count up from one, so the highest id is the capacity needed.
         * Grown in blocks so plan creation does not reallocate every time. */
        const uint64_t grown = (plan_id + 63U) & ~UINT64_C(63);
        ShadowSpillPlanSlot *slots = realloc(
            runtime->plans_by_id.slots, (size_t)grown * sizeof(*slots)
        );
        if (slots == NULL) {
            result = -1;
        } else {
            memset(
                slots + runtime->plans_by_id.capacity,
                0,
                (size_t)(grown - runtime->plans_by_id.capacity) * sizeof(*slots)
            );
            runtime->plans_by_id.slots = slots;
            runtime->plans_by_id.capacity = grown;
        }
    }
    if (result == 0) {
        ShadowSpillPlanSlot *slot = &runtime->plans_by_id.slots[plan_id - 1U];
        if (slot->claimed != 0U) {
            result = 1;
        } else {
            slot->claimed = 1U;
            slot->plan = plan;
        }
    }
    pthread_mutex_unlock(&runtime->plans_by_id.lock);
    return result;
}

/*
 * Forget the record while keeping the id claimed. Only the plan the slot names
 * may clear it: a creation refused for a duplicate id is torn down holding that
 * id, and must not take the slot from the plan that owns it.
 */
void shadowspill_plan_registry_release(
    ShadowSpillRuntime *runtime,
    uint64_t plan_id,
    const ShadowSpillPlan *plan
) {
    if (runtime == NULL || !runtime->plans_by_id.lock_initialized ||
        plan_id == 0U || plan_id > runtime->plans_by_id.capacity) {
        return;
    }
    pthread_mutex_lock(&runtime->plans_by_id.lock);
    ShadowSpillPlanSlot *slot = &runtime->plans_by_id.slots[plan_id - 1U];
    if (slot->plan == plan) {
        slot->plan = NULL;
    }
    pthread_mutex_unlock(&runtime->plans_by_id.lock);
}

int shadowspill_plan_registry_initialize(ShadowSpillRuntime *runtime) {
    if (pthread_mutex_init(&runtime->plans_by_id.lock, NULL) != 0) {
        return -1;
    }
    runtime->plans_by_id.lock_initialized = 1U;
    return 0;
}

void shadowspill_plan_registry_destroy(ShadowSpillRuntime *runtime) {
    free(runtime->plans_by_id.slots);
    runtime->plans_by_id.slots = NULL;
    runtime->plans_by_id.capacity = 0U;
    if (runtime->plans_by_id.lock_initialized) {
        pthread_mutex_destroy(&runtime->plans_by_id.lock);
        runtime->plans_by_id.lock_initialized = 0U;
    }
}

ShadowSpillPlan *shadowspill_runtime_plan(
    ShadowSpillRuntime *runtime,
    uint64_t plan_id
) {
    if (runtime == NULL || !runtime->plans_by_id.lock_initialized ||
        plan_id == 0U || plan_id > runtime->plans_by_id.capacity) {
        return NULL;
    }
    pthread_mutex_lock(&runtime->plans_by_id.lock);
    ShadowSpillPlan *plan = runtime->plans_by_id.slots[plan_id - 1U].plan;
    pthread_mutex_unlock(&runtime->plans_by_id.lock);
    return plan;
}

ShadowSpillStatus shadowspill_runtime_plan_state(
    ShadowSpillRuntime *runtime,
    uint64_t plan_id,
    ShadowSpillPlanState *state
) {
    if (runtime == NULL || state == NULL || plan_id == 0U ||
        !runtime->plans_by_id.lock_initialized) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&runtime->plans_by_id.lock);
    const ShadowSpillPlanSlot slot = plan_id <= runtime->plans_by_id.capacity
        ? runtime->plans_by_id.slots[plan_id - 1U]
        : (ShadowSpillPlanSlot){NULL, 0U};
    pthread_mutex_unlock(&runtime->plans_by_id.lock);
    if (slot.claimed == 0U) {
        *state = SHADOWSPILL_PLAN_STATE_UNKNOWN;
    } else if (slot.plan == NULL) {
        *state = SHADOWSPILL_PLAN_STATE_DESTROYED;
    } else if (atomic_load_explicit(
                   &slot.plan->closing, memory_order_acquire
               ) != 0U) {
        *state = SHADOWSPILL_PLAN_STATE_CLOSED;
    } else {
        *state = SHADOWSPILL_PLAN_STATE_LIVE;
    }
    return SHADOWSPILL_STATUS_OK;
}
