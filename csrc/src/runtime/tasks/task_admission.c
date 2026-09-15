/* Admitting a task, or a batch of actions, into a plan. */
#include "../internal.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static ShadowSpillStatus admit_record(
    ShadowSpillPlan *plan,
    const ShadowSpillTaskDescription *description,
    uint8_t boundary_kind,
    const ShadowSpillTaskRecord **result
) {
    if (plan == NULL || description == NULL ||
        (description->trace_label != NULL &&
         strnlen(
             description->trace_label,
             SHADOWSPILL_RUNTIME_TRACE_LABEL_MAX_BYTES + 1U
         ) > SHADOWSPILL_RUNTIME_TRACE_LABEL_MAX_BYTES) ||
        (description->input_count != 0U &&
         description->input_object_ids == NULL) ||
        (description->update_count != 0U && description->updates == NULL) ||
        (description->publication_count != 0U &&
         description->publications == NULL) ||
        (description->action_count != 0U && description->actions == NULL) ||
        (description->allocation_contract_step_count != 0U &&
         description->allocation_contract_steps == NULL) ||
        !shadowspill_task_valid_allocation_contract(description) ||
        (description->maximum_requested_allocation_bytes != 0U &&
         description->live_requested_allocation_limit_bytes != 0U &&
         description->maximum_requested_allocation_bytes >
             description->live_requested_allocation_limit_bytes) ||
        (description->maximum_charged_allocation_bytes != 0U &&
         description->live_charged_allocation_limit_bytes != 0U &&
         description->maximum_charged_allocation_bytes >
             description->live_charged_allocation_limit_bytes) ||
        (description->dynamic_scratch_maximum_allocation_bytes != 0U &&
         description->dynamic_scratch_live_limit_bytes != 0U &&
         description->dynamic_scratch_maximum_allocation_bytes >
             description->dynamic_scratch_live_limit_bytes)) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    ShadowSpillRuntime *runtime = plan->runtime;
    ShadowSpillStatus status = shadowspill_current_status_locked(runtime);
    if (status != SHADOWSPILL_STATUS_OK) {
        return status;
    }
    for (uint32_t index = 0U; index < description->action_count; ++index) {
        for (uint32_t previous = 0U; previous < index; ++previous) {
            if (description->actions[previous].object_id ==
                description->actions[index].object_id) {
                return SHADOWSPILL_STATUS_PLAN_VIOLATION;
            }
        }
    }
    ShadowSpillTaskRecord *created = shadowspill_task_create_record(
        plan, description, boundary_kind
    );
    if (created == NULL) {
        return SHADOWSPILL_STATUS_INVALID_STATE;
    }
    ShadowSpillTaskTable *table = &plan->tasks;
    pthread_rwlock_wrlock(&table->lock);
    ShadowSpillTaskRecord *existing = shadowspill_task_find_unlocked(
        table, description->task_id
    );
    if (existing != NULL) {
        const int matches = shadowspill_task_same_description(
            existing, description, boundary_kind
        );
        pthread_rwlock_unlock(&table->lock);
        shadowspill_task_destroy_record(created);
        if (matches && result != NULL) {
            *result = existing;
        }
        return matches
            ? SHADOWSPILL_STATUS_OK
            : SHADOWSPILL_STATUS_INVALID_STATE;
    }
    const uint64_t bucket = shadowspill_task_bucket(table, created->task_id);
    created->hash_next = table->by_id[bucket];
    table->by_id[bucket] = created;
    created->ownership_next = table->owned_head;
    table->owned_head = created;
    pthread_rwlock_unlock(&table->lock);
    if (result != NULL) {
        *result = created;
    }
    return SHADOWSPILL_STATUS_OK;
}

ShadowSpillStatus shadowspill_plan_admit_task(
    ShadowSpillPlan *plan,
    const ShadowSpillTaskDescription *description,
    const ShadowSpillTaskHandle **handle
) {
    if (handle == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    *handle = NULL;
    const ShadowSpillTaskRecord *record = NULL;
    const ShadowSpillStatus status = admit_record(
        plan,
        description,
        SHADOWSPILL_BOUNDARY_TASK,
        &record
    );
    if (status == SHADOWSPILL_STATUS_OK) {
        *handle = record;
    }
    return status;
}

uint64_t shadowspill_task_id(const ShadowSpillTaskHandle *handle) {
    const ShadowSpillTaskRecord *record = handle;
    return record == NULL ? SHADOWSPILL_RUNTIME_NO_ID : record->task_id;
}

const char *shadowspill_task_trace_label(const ShadowSpillTaskHandle *handle) {
    const ShadowSpillTaskRecord *record = handle;
    return record == NULL ? NULL : record->trace_label;
}

ShadowSpillStatus shadowspill_plan_admit_action_batch(
    ShadowSpillPlan *plan,
    uint64_t batch_id,
    const ShadowSpillRuntimeAction *actions,
    uint32_t action_count,
    const ShadowSpillActionBatchHandle **handle
) {
    if (handle == NULL || (action_count != 0U && actions == NULL)) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    *handle = NULL;
    const ShadowSpillTaskDescription description = {
        .task_id = batch_id,
        .actions = actions,
        .action_count = action_count,
    };
    const ShadowSpillTaskRecord *record = NULL;
    const ShadowSpillStatus status = admit_record(
        plan,
        &description,
        SHADOWSPILL_BOUNDARY_ACTION_BATCH,
        &record
    );
    if (status == SHADOWSPILL_STATUS_OK) {
        *handle = record;
    }
    return status;
}

ShadowSpillStatus shadowspill_plan_clear_tasks(
    ShadowSpillPlan *plan
) {
    if (plan == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    ShadowSpillStatus status = shadowspill_plan_wait_idle(plan);
    if (status != SHADOWSPILL_STATUS_OK) {
        return status;
    }
    if (atomic_load_explicit(
            &plan->active_task_scopes, memory_order_acquire
        ) != 0U ||
        atomic_load_explicit(
            &plan->pending_actions, memory_order_acquire
        ) != 0U ||
        atomic_load_explicit(
            &plan->pending_retirements, memory_order_acquire
        ) != 0U) {
        return SHADOWSPILL_STATUS_INVALID_STATE;
    }
    status = shadowspill_fixed_layout_clear(plan);
    if (status != SHADOWSPILL_STATUS_OK) {
        return status;
    }
    shadowspill_object_acquisitions_clear(plan);
    shadowspill_task_table_clear(&plan->tasks);
    shadowspill_plan_object_table_clear(&plan->object_bindings);
    return SHADOWSPILL_STATUS_OK;
}
