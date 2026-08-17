#include "internal.h"

#include <stdlib.h>

static int description_is_valid(
    const ShadowSpillRuntime *runtime,
    const ShadowSpillPlanDescription *description
) {
    if (runtime == NULL || description == NULL ||
        description->execution_pool_id >= runtime->pool_count ||
        description->spill_pool_id >= runtime->pool_count ||
        description->execution_pool_id == description->spill_pool_id ||
        description->fetch_route_id >= runtime->route_count ||
        description->evict_route_id >= runtime->route_count) {
        return 0;
    }
    const ShadowSpillTransferRoute *fetch =
        &runtime->routes[description->fetch_route_id].route;
    const ShadowSpillTransferRoute *evict =
        &runtime->routes[description->evict_route_id].route;
    return fetch->source_pool_id == description->spill_pool_id &&
        fetch->destination_pool_id == description->execution_pool_id &&
        evict->source_pool_id == description->execution_pool_id &&
        evict->destination_pool_id == description->spill_pool_id;
}

static void unlink_plan(ShadowSpillPlan *plan) {
    ShadowSpillRuntime *runtime = plan->runtime;
    pthread_mutex_lock(&runtime->plans_lock);
    if (plan->ownership_previous_link != NULL) {
        *plan->ownership_previous_link = plan->ownership_next;
        if (plan->ownership_next != NULL) {
            plan->ownership_next->ownership_previous_link =
                plan->ownership_previous_link;
        }
        plan->ownership_next = NULL;
        plan->ownership_previous_link = NULL;
    }
    pthread_mutex_unlock(&runtime->plans_lock);
}

static void destroy_plan_record(ShadowSpillPlan *plan) {
    if (plan == NULL) {
        return;
    }
    shadowspill_fixed_layout_destroy(plan);
    shadowspill_object_acquisitions_clear(plan);
    if (plan->tasks_initialized) {
        shadowspill_task_table_destroy(&plan->tasks);
        plan->tasks_initialized = 0U;
    }
    if (plan->object_bindings_initialized) {
        shadowspill_plan_object_table_destroy(&plan->object_bindings);
        plan->object_bindings_initialized = 0U;
    }
    if (plan->lifecycle_lock_initialized) {
        pthread_mutex_destroy(&plan->lifecycle_lock);
        plan->lifecycle_lock_initialized = 0U;
    }
    free(plan);
}

ShadowSpillRuntimeStatus shadowspill_plan_create(
    ShadowSpillRuntime *runtime,
    const ShadowSpillPlanDescription *description,
    ShadowSpillPlan **output
) {
    if (output == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    *output = NULL;
    if (!description_is_valid(runtime, description) ||
        atomic_load_explicit(&runtime->closing, memory_order_acquire) != 0U) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    ShadowSpillRuntimeStatus status = shadowspill_current_status_locked(runtime);
    if (status != SHADOWSPILL_RUNTIME_OK) {
        return status;
    }
    ShadowSpillPlan *plan = calloc(1U, sizeof(*plan));
    if (plan == NULL) {
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    plan->runtime = runtime;
    plan->execution_pool =
        &runtime->pools[description->execution_pool_id];
    plan->spill_pool = &runtime->pools[description->spill_pool_id];
    plan->fetch_route = &runtime->routes[description->fetch_route_id];
    plan->evict_route = &runtime->routes[description->evict_route_id];
    atomic_init(&plan->active_invocations, 0U);
    atomic_init(&plan->closing, 0U);
    atomic_init(&plan->closed, 0U);
    if (pthread_mutex_init(&plan->lifecycle_lock, NULL) != 0) {
        free(plan);
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    plan->lifecycle_lock_initialized = 1U;
    if (shadowspill_plan_object_table_initialize(
            &plan->object_bindings, 4096U
        ) != 0) {
        destroy_plan_record(plan);
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    plan->object_bindings_initialized = 1U;
    if (shadowspill_task_table_initialize(&plan->tasks, 4096U) != 0) {
        destroy_plan_record(plan);
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    plan->tasks_initialized = 1U;

    pthread_mutex_lock(&runtime->plans_lock);
    if (atomic_load_explicit(&runtime->closing, memory_order_acquire) != 0U) {
        pthread_mutex_unlock(&runtime->plans_lock);
        destroy_plan_record(plan);
        return SHADOWSPILL_RUNTIME_CLOSED;
    }
    plan->ownership_next = runtime->plans;
    plan->ownership_previous_link = &runtime->plans;
    if (runtime->plans != NULL) {
        runtime->plans->ownership_previous_link = &plan->ownership_next;
    }
    runtime->plans = plan;
    pthread_mutex_unlock(&runtime->plans_lock);
    *output = plan;
    return SHADOWSPILL_RUNTIME_OK;
}

ShadowSpillRuntimeStatus shadowspill_plan_close(ShadowSpillPlan *plan) {
    if (plan == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&plan->lifecycle_lock);
    if (atomic_load_explicit(&plan->closed, memory_order_acquire) != 0U) {
        pthread_mutex_unlock(&plan->lifecycle_lock);
        return SHADOWSPILL_RUNTIME_OK;
    }
    atomic_store_explicit(&plan->closing, 1U, memory_order_release);
    if (atomic_load_explicit(
            &plan->active_invocations, memory_order_acquire
        ) != 0U) {
        pthread_mutex_unlock(&plan->lifecycle_lock);
        return SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    ShadowSpillRuntimeStatus status = shadowspill_plan_clear_tasks(plan);
    if (status == SHADOWSPILL_RUNTIME_OK) {
        unlink_plan(plan);
        atomic_store_explicit(&plan->closed, 1U, memory_order_release);
    } else {
        atomic_store_explicit(&plan->closing, 0U, memory_order_release);
    }
    pthread_mutex_unlock(&plan->lifecycle_lock);
    return status;
}

void shadowspill_plan_destroy(ShadowSpillPlan *plan) {
    if (plan == NULL) {
        return;
    }
    (void)shadowspill_plan_close(plan);
    if (plan->ownership_previous_link != NULL) {
        unlink_plan(plan);
    }
    destroy_plan_record(plan);
}

void shadowspill_plan_destroy_all(ShadowSpillRuntime *runtime) {
    if (runtime == NULL || !runtime->plans_lock_initialized) {
        return;
    }
    pthread_mutex_lock(&runtime->plans_lock);
    ShadowSpillPlan *plan = runtime->plans;
    runtime->plans = NULL;
    pthread_mutex_unlock(&runtime->plans_lock);
    while (plan != NULL) {
        ShadowSpillPlan *next = plan->ownership_next;
        plan->ownership_next = NULL;
        plan->ownership_previous_link = NULL;
        destroy_plan_record(plan);
        plan = next;
    }
}
