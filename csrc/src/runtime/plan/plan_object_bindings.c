#include "internal.h"

#include <stdlib.h>
#include <string.h>

static uint64_t binding_bucket(
    const ShadowSpillPlanObjectTable *table,
    uint64_t plan_object_id
) {
    plan_object_id ^= plan_object_id >> 33U;
    plan_object_id *= UINT64_C(0xff51afd7ed558ccd);
    plan_object_id ^= plan_object_id >> 33U;
    return plan_object_id % table->bucket_count;
}

int shadowspill_plan_object_table_initialize(
    ShadowSpillPlanObjectTable *table,
    uint64_t bucket_count
) {
    if (table == NULL || bucket_count == 0U ||
        pthread_rwlock_init(&table->lock, NULL) != 0) {
        return -1;
    }
    table->lock_initialized = 1U;
    table->by_id = calloc((size_t)bucket_count, sizeof(*table->by_id));
    if (table->by_id == NULL) {
        pthread_rwlock_destroy(&table->lock);
        table->lock_initialized = 0U;
        return -1;
    }
    table->bucket_count = bucket_count;
    return 0;
}

void shadowspill_plan_object_table_clear(ShadowSpillPlanObjectTable *table) {
    if (table == NULL || !table->lock_initialized) {
        return;
    }
    pthread_rwlock_wrlock(&table->lock);
    ShadowSpillPlanObjectBinding *binding = table->owned_head;
    table->owned_head = NULL;
    memset(
        table->by_id,
        0,
        (size_t)table->bucket_count * sizeof(*table->by_id)
    );
    pthread_rwlock_unlock(&table->lock);
    while (binding != NULL) {
        ShadowSpillPlanObjectBinding *next = binding->ownership_next;
        (void)shadowspill_object_owner_release(binding->object);
        free(binding);
        binding = next;
    }
}

void shadowspill_plan_object_table_destroy(ShadowSpillPlanObjectTable *table) {
    if (table == NULL || !table->lock_initialized) {
        return;
    }
    shadowspill_plan_object_table_clear(table);
    free(table->by_id);
    table->by_id = NULL;
    table->owned_head = NULL;
    table->bucket_count = 0U;
    pthread_rwlock_destroy(&table->lock);
    table->lock_initialized = 0U;
}

ShadowSpillStatus shadowspill_plan_bind_object(
    ShadowSpillPlan *plan,
    uint64_t plan_object_id,
    const ShadowSpillObjectHandle *object_handle,
    uint8_t consistency
) {
    if (plan == NULL || plan_object_id == SHADOWSPILL_RUNTIME_NO_ID ||
        object_handle == NULL || object_handle->runtime != plan->runtime ||
        object_handle->object == NULL ||
        consistency > SHADOWSPILL_OBJECT_UNORDERED ||
        atomic_load_explicit(&plan->closing, memory_order_acquire) != 0U) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    ShadowSpillObject *object = object_handle->object;
    if (atomic_load_explicit(&object->detached, memory_order_acquire) != 0U) {
        return SHADOWSPILL_STATUS_INVALID_STATE;
    }
    ShadowSpillStatus status = shadowspill_object_owner_retain(object);
    if (status != SHADOWSPILL_STATUS_OK) {
        return status;
    }
    ShadowSpillPlanObjectBinding *created = calloc(1U, sizeof(*created));
    if (created == NULL) {
        (void)shadowspill_object_owner_release(object);
        return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    }
    created->plan_object_id = plan_object_id;
    created->object = object;
    created->consistency = consistency;
    ShadowSpillPlanObjectTable *table = &plan->object_bindings;
    const uint64_t bucket = binding_bucket(table, plan_object_id);
    pthread_rwlock_wrlock(&table->lock);
    for (ShadowSpillPlanObjectBinding *binding = table->by_id[bucket];
         binding != NULL; binding = binding->hash_next) {
        if (binding->plan_object_id != plan_object_id) {
            continue;
        }
        const int matches = binding->object == object &&
            binding->consistency == consistency;
        pthread_rwlock_unlock(&table->lock);
        (void)shadowspill_object_owner_release(object);
        free(created);
        return matches
            ? SHADOWSPILL_STATUS_OK : SHADOWSPILL_STATUS_INVALID_STATE;
    }
    created->hash_next = table->by_id[bucket];
    table->by_id[bucket] = created;
    created->ownership_next = table->owned_head;
    table->owned_head = created;
    pthread_rwlock_unlock(&table->lock);
    return SHADOWSPILL_STATUS_OK;
}

ShadowSpillObject *shadowspill_plan_object_acquire(
    ShadowSpillPlan *plan,
    uint64_t plan_object_id,
    uint8_t *consistency
) {
    if (plan == NULL || !plan->object_bindings_initialized) {
        return NULL;
    }
    ShadowSpillPlanObjectTable *table = &plan->object_bindings;
    const uint64_t bucket = binding_bucket(table, plan_object_id);
    pthread_rwlock_rdlock(&table->lock);
    ShadowSpillObject *object = NULL;
    for (ShadowSpillPlanObjectBinding *binding = table->by_id[bucket];
         binding != NULL; binding = binding->hash_next) {
        if (binding->plan_object_id == plan_object_id) {
            object = binding->object;
            shadowspill_object_retain(object);
            if (consistency != NULL) {
                *consistency = binding->consistency;
            }
            break;
        }
    }
    pthread_rwlock_unlock(&table->lock);
    return object;
}
