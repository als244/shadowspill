#include "internal.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *shadowspill_copy_action_trace_label(
    const ShadowSpillRuntimeAction *action,
    uint64_t task_id,
    uint64_t size_bytes
) {
    if (action->trace_label != NULL) {
        const size_t length = strnlen(
            action->trace_label,
            SHADOWSPILL_RUNTIME_TRACE_LABEL_MAX_BYTES + 1U
        );
        if (length > SHADOWSPILL_RUNTIME_TRACE_LABEL_MAX_BYTES) {
            return NULL;
        }
        char *copy = malloc(length + 1U);
        if (copy == NULL) {
            return NULL;
        }
        memcpy(copy, action->trace_label, length + 1U);
        return copy;
    }
    const char *operation = action->kind == SHADOWSPILL_RUNTIME_PREFETCH
        ? "fetch"
        : action->kind == SHADOWSPILL_RUNTIME_OFFLOAD ? "evict" : "release";
    char fallback[256];
    const int written = snprintf(
        fallback,
        sizeof(fallback),
        "shadowspill.runtime.transfer.%s.object_%llu.bytes_%llu.trigger_task_%llu",
        operation,
        (unsigned long long)action->object_id,
        (unsigned long long)size_bytes,
        (unsigned long long)task_id
    );
    if (written < 0 || (size_t)written >= sizeof(fallback)) {
        return NULL;
    }
    return strdup(fallback);
}

static uint64_t task_bucket(
    const ShadowSpillTaskTable *table,
    uint64_t task_id
) {
    task_id ^= task_id >> 33U;
    task_id *= UINT64_C(0xff51afd7ed558ccd);
    task_id ^= task_id >> 33U;
    return task_id % table->bucket_count;
}

static ShadowSpillTaskRecord *find_unlocked(
    const ShadowSpillTaskTable *table,
    uint64_t task_id
) {
    if (table->by_id == NULL || table->bucket_count == 0U) {
        return NULL;
    }
    const uint64_t bucket = task_bucket(table, task_id);
    for (ShadowSpillTaskRecord *record = table->by_id[bucket];
         record != NULL; record = record->hash_next) {
        if (record->task_id == task_id) {
            return record;
        }
    }
    return NULL;
}

static void destroy_record(ShadowSpillTaskRecord *record) {
    if (record == NULL) {
        return;
    }
    for (uint32_t index = 0U; index < record->input_count; ++index) {
        shadowspill_object_release(record->inputs[index]);
    }
    for (uint32_t index = 0U; index < record->update_count; ++index) {
        shadowspill_object_release(record->updates[index].object);
    }
    for (uint32_t index = 0U; index < record->action_count; ++index) {
        shadowspill_object_release(record->actions[index].object);
        free(record->actions[index].trace_label);
    }
    free(record->inputs);
    free(record->input_plan_object_ids);
    free(record->input_consistency);
    free(record->unique_inputs);
    free(record->input_unique_indices);
    free(record->unique_first_positions);
    free(record->updates);
    free(record->actions);
    free(record->queued_actions);
    free(record->allocation_contract_steps);
    free(record);
}

int shadowspill_task_table_initialize(
    ShadowSpillTaskTable *table,
    uint64_t bucket_count
) {
    if (table == NULL || bucket_count == 0U || bucket_count > SIZE_MAX) {
        return -1;
    }
    if (pthread_rwlock_init(&table->lock, NULL) != 0) {
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

void shadowspill_task_table_destroy(ShadowSpillTaskTable *table) {
    if (table == NULL) {
        return;
    }
    ShadowSpillTaskRecord *record = table->owned_head;
    while (record != NULL) {
        ShadowSpillTaskRecord *next = record->ownership_next;
        destroy_record(record);
        record = next;
    }
    free(table->by_id);
    if (table->lock_initialized) {
        pthread_rwlock_destroy(&table->lock);
    }
    *table = (ShadowSpillTaskTable){0};
}

void shadowspill_task_table_clear(ShadowSpillTaskTable *table) {
    if (table == NULL || !table->lock_initialized) {
        return;
    }
    pthread_rwlock_wrlock(&table->lock);
    ShadowSpillTaskRecord *record = table->owned_head;
    table->owned_head = NULL;
    memset(
        table->by_id,
        0,
        (size_t)table->bucket_count * sizeof(*table->by_id)
    );
    pthread_rwlock_unlock(&table->lock);

    while (record != NULL) {
        ShadowSpillTaskRecord *next = record->ownership_next;
        destroy_record(record);
        record = next;
    }
}

ShadowSpillTaskRecord *shadowspill_task_table_acquire(
    ShadowSpillTaskTable *table,
    uint64_t task_id
) {
    if (table == NULL || !table->lock_initialized) {
        return NULL;
    }
    pthread_rwlock_rdlock(&table->lock);
    ShadowSpillTaskRecord *record = find_unlocked(table, task_id);
    pthread_rwlock_unlock(&table->lock);
    return record;
}

static int same_description(
    const ShadowSpillTaskRecord *record,
    const ShadowSpillTaskDescription *description,
    uint8_t boundary_kind
) {
    if (record->boundary_kind != boundary_kind ||
        record->input_count != description->input_count ||
        record->update_count != description->update_count ||
        record->action_count != description->action_count ||
        record->allocation_contract_step_count !=
            description->allocation_contract_step_count ||
        record->enforce_allocation_contract !=
            description->enforce_allocation_contract ||
        record->maximum_requested_allocation_bytes !=
            description->maximum_requested_allocation_bytes ||
        record->maximum_charged_allocation_bytes !=
            description->maximum_charged_allocation_bytes ||
        record->live_requested_allocation_limit_bytes !=
            description->live_requested_allocation_limit_bytes ||
        record->live_charged_allocation_limit_bytes !=
            description->live_charged_allocation_limit_bytes ||
        record->dynamic_scratch_maximum_allocation_bytes !=
            description->dynamic_scratch_maximum_allocation_bytes ||
        record->dynamic_scratch_live_limit_bytes !=
            description->dynamic_scratch_live_limit_bytes) {
        return 0;
    }
    for (uint32_t index = 0U; index < record->input_count; ++index) {
        if (record->input_plan_object_ids[index] !=
            description->input_object_ids[index]) {
            return 0;
        }
    }
    for (uint32_t index = 0U; index < record->update_count; ++index) {
        if (record->updates[index].plan_object_id !=
                description->updates[index].object_id ||
            record->updates[index].version_delta !=
                description->updates[index].version_delta) {
            return 0;
        }
    }
    for (uint32_t index = 0U; index < record->action_count; ++index) {
        if (record->actions[index].plan_object_id !=
                description->actions[index].object_id ||
            record->actions[index].kind !=
                description->actions[index].kind ||
            (description->actions[index].trace_label != NULL &&
             strcmp(
                 record->actions[index].trace_label,
                 description->actions[index].trace_label
             ) != 0)) {
            return 0;
        }
    }
    for (uint32_t index = 0U;
         index < record->allocation_contract_step_count; ++index) {
        const ShadowSpillTaskAllocationContractStep *left =
            &record->allocation_contract_steps[index];
        const ShadowSpillTaskAllocationContractStep *right =
            &description->allocation_contract_steps[index];
        if (left->allocation_ordinal != right->allocation_ordinal ||
            left->requested_bytes != right->requested_bytes ||
            left->charged_bytes != right->charged_bytes ||
            left->alignment_bytes != right->alignment_bytes ||
            left->operation != right->operation ||
            left->required != right->required) {
            return 0;
        }
    }
    return 1;
}

static ShadowSpillTaskRecord *create_record(
    ShadowSpillPlan *plan,
    const ShadowSpillTaskDescription *description,
    uint8_t boundary_kind
) {
    ShadowSpillTaskRecord *record = calloc(1U, sizeof(*record));
    if (record == NULL) {
        return NULL;
    }
    record->plan_owner = plan;
    record->task_id = description->task_id;
    record->boundary_kind = boundary_kind;
    atomic_init(&record->invocation_count, 0U);
    atomic_init(&record->submission_sequence, 0U);
    atomic_init(&record->submission_invocation, 0U);
    atomic_init(&record->acknowledgement_sequence, 0U);
    record->input_count = description->input_count;
    record->update_count = description->update_count;
    record->action_count = description->action_count;
    record->allocation_contract_step_count =
        description->allocation_contract_step_count;
    record->enforce_allocation_contract = description->enforce_allocation_contract;
    record->maximum_requested_allocation_bytes =
        description->maximum_requested_allocation_bytes;
    record->maximum_charged_allocation_bytes =
        description->maximum_charged_allocation_bytes;
    record->live_requested_allocation_limit_bytes =
        description->live_requested_allocation_limit_bytes;
    record->live_charged_allocation_limit_bytes =
        description->live_charged_allocation_limit_bytes;
    record->dynamic_scratch_maximum_allocation_bytes =
        description->dynamic_scratch_maximum_allocation_bytes;
    record->dynamic_scratch_live_limit_bytes =
        description->dynamic_scratch_live_limit_bytes;
    if (record->input_count != 0U) {
        record->inputs = calloc(record->input_count, sizeof(*record->inputs));
        record->input_plan_object_ids = calloc(
            record->input_count, sizeof(*record->input_plan_object_ids)
        );
        record->input_consistency = calloc(
            record->input_count, sizeof(*record->input_consistency)
        );
        record->unique_inputs = calloc(
            record->input_count, sizeof(*record->unique_inputs)
        );
        record->input_unique_indices = calloc(
            record->input_count, sizeof(*record->input_unique_indices)
        );
        record->unique_first_positions = calloc(
            record->input_count, sizeof(*record->unique_first_positions)
        );
    }
    if (record->update_count != 0U) {
        record->updates = calloc(record->update_count, sizeof(*record->updates));
    }
    if (record->action_count != 0U) {
        record->actions = calloc(record->action_count, sizeof(*record->actions));
        record->queued_actions = calloc(
            record->action_count, sizeof(*record->queued_actions)
        );
    }
    if (record->allocation_contract_step_count != 0U) {
        record->allocation_contract_steps = calloc(
            record->allocation_contract_step_count,
            sizeof(*record->allocation_contract_steps)
        );
    }
    if ((record->input_count != 0U &&
         (record->inputs == NULL || record->input_plan_object_ids == NULL ||
          record->input_consistency == NULL || record->unique_inputs == NULL ||
          record->input_unique_indices == NULL ||
          record->unique_first_positions == NULL)) ||
        (record->update_count != 0U && record->updates == NULL) ||
        (record->action_count != 0U &&
         (record->actions == NULL || record->queued_actions == NULL)) ||
        (record->allocation_contract_step_count != 0U &&
         record->allocation_contract_steps == NULL)) {
        destroy_record(record);
        return NULL;
    }
    if (record->allocation_contract_step_count != 0U) {
        memcpy(
            record->allocation_contract_steps,
            description->allocation_contract_steps,
            record->allocation_contract_step_count *
                sizeof(*record->allocation_contract_steps)
        );
        for (uint32_t index = 0U;
             index < record->allocation_contract_step_count;
             ++index) {
            const ShadowSpillTaskAllocationContractStep *step =
                &record->allocation_contract_steps[index];
            if (step->operation == SHADOWSPILL_TASK_ALLOCATION_ALLOCATE) {
                record->allocation_contract_allocation_count =
                    (uint32_t)(step->allocation_ordinal + 1U);
            }
        }
    }
    for (uint32_t index = 0U; index < record->input_count; ++index) {
        uint8_t consistency = SHADOWSPILL_OBJECT_CAUSAL;
        ShadowSpillObject *object = shadowspill_plan_object_acquire(
            plan, description->input_object_ids[index], &consistency
        );
        if (object == NULL) {
            destroy_record(record);
            return NULL;
        }
        record->inputs[index] = object;
        record->input_plan_object_ids[index] =
            description->input_object_ids[index];
        record->input_consistency[index] = consistency;
        uint32_t unique_index = record->unique_input_count;
        for (uint32_t previous = 0U;
             previous < record->unique_input_count; ++previous) {
            if (record->unique_inputs[previous] == object) {
                unique_index = previous;
                break;
            }
        }
        if (unique_index == record->unique_input_count) {
            record->unique_inputs[unique_index] = object;
            record->unique_first_positions[unique_index] = index;
            ++record->unique_input_count;
        }
        record->input_unique_indices[index] = unique_index;
    }
    for (uint32_t index = 0U; index < record->update_count; ++index) {
        ShadowSpillObject *object = shadowspill_plan_object_acquire(
            plan, description->updates[index].object_id, NULL
        );
        if (object == NULL) {
            destroy_record(record);
            return NULL;
        }
        record->updates[index] = (ShadowSpillTaskUpdate){
            .object = object,
            .plan_object_id = description->updates[index].object_id,
            .version_delta = description->updates[index].version_delta,
        };
    }
    for (uint32_t index = 0U; index < record->action_count; ++index) {
        if (description->actions[index].kind > SHADOWSPILL_RUNTIME_PREFETCH) {
            destroy_record(record);
            return NULL;
        }
        for (uint32_t previous = 0U; previous < index; ++previous) {
            if (description->actions[previous].object_id ==
                description->actions[index].object_id) {
                destroy_record(record);
                return NULL;
            }
        }
        ShadowSpillObject *object = shadowspill_plan_object_acquire(
            plan, description->actions[index].object_id, NULL
        );
        if (object == NULL) {
            destroy_record(record);
            return NULL;
        }
        char *trace_label = shadowspill_copy_action_trace_label(
            &description->actions[index], record->task_id, object->size_bytes
        );
        if (trace_label == NULL) {
            shadowspill_object_release(object);
            destroy_record(record);
            return NULL;
        }
        record->actions[index] = (ShadowSpillTaskAction){
            .object = object,
            .plan_object_id = description->actions[index].object_id,
            .kind = description->actions[index].kind,
            .trace_label = trace_label,
        };
        record->queued_actions[index] = (ShadowSpillQueuedAction){
            .task_id = record->task_id,
            .plan_object_id = description->actions[index].object_id,
            .action_ordinal = index,
            .kind = description->actions[index].kind,
            .object = object,
            .plan_owner = plan,
            .route = description->actions[index].kind ==
                    SHADOWSPILL_RUNTIME_RELEASE
                ? NULL
                : description->actions[index].kind ==
                        SHADOWSPILL_RUNTIME_PREFETCH
                    ? plan->fetch_route
                    : plan->evict_route,
            .trace_label = trace_label,
            .admitted = 1U,
        };
    }
    return record;
}

static int valid_allocation_contract(
    const ShadowSpillTaskDescription *description
) {
    if (!description->enforce_allocation_contract) {
        return description->allocation_contract_step_count == 0U;
    }
    const uint32_t count = description->allocation_contract_step_count;
    const ShadowSpillTaskAllocationContractStep **allocations = count == 0U
        ? NULL
        : calloc(count, sizeof(*allocations));
    if (count != 0U && allocations == NULL) {
        return 0;
    }
    uint64_t next_ordinal = 0U;
    int valid = 1;
    for (uint32_t index = 0U; index < count && valid; ++index) {
        const ShadowSpillTaskAllocationContractStep *step =
            &description->allocation_contract_steps[index];
        if (step->charged_bytes == 0U || step->alignment_bytes == 0U ||
            step->requested_bytes > step->charged_bytes) {
            valid = 0;
            break;
        }
        if (step->operation == SHADOWSPILL_TASK_ALLOCATION_ALLOCATE) {
            if (step->allocation_ordinal != next_ordinal) {
                valid = 0;
                break;
            }
            allocations[next_ordinal++] = step;
            continue;
        }
        if (step->operation != SHADOWSPILL_TASK_ALLOCATION_FREE ||
            step->required ||
            step->allocation_ordinal >= next_ordinal ||
            allocations[step->allocation_ordinal] == NULL) {
            valid = 0;
            break;
        }
        const ShadowSpillTaskAllocationContractStep *allocation =
            allocations[step->allocation_ordinal];
        if (allocation->requested_bytes != step->requested_bytes ||
            allocation->charged_bytes != step->charged_bytes ||
            allocation->alignment_bytes != step->alignment_bytes) {
            valid = 0;
            break;
        }
        allocations[step->allocation_ordinal] = NULL;
    }
    free(allocations);
    return valid;
}

static ShadowSpillRuntimeStatus admit_record(
    ShadowSpillPlan *plan,
    const ShadowSpillTaskDescription *description,
    uint8_t boundary_kind,
    const ShadowSpillTaskRecord **result
) {
    if (plan == NULL || description == NULL ||
        (description->input_count != 0U &&
         description->input_object_ids == NULL) ||
        (description->update_count != 0U && description->updates == NULL) ||
        (description->action_count != 0U && description->actions == NULL) ||
        (description->allocation_contract_step_count != 0U &&
         description->allocation_contract_steps == NULL) ||
        !valid_allocation_contract(description) ||
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
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    ShadowSpillRuntime *runtime = plan->runtime;
    ShadowSpillRuntimeStatus status = shadowspill_current_status_locked(runtime);
    if (status != SHADOWSPILL_RUNTIME_OK) {
        return status;
    }
    for (uint32_t index = 0U; index < description->action_count; ++index) {
        for (uint32_t previous = 0U; previous < index; ++previous) {
            if (description->actions[previous].object_id ==
                description->actions[index].object_id) {
                return SHADOWSPILL_RUNTIME_PLAN_VIOLATION;
            }
        }
    }
    ShadowSpillTaskRecord *created = create_record(
        plan, description, boundary_kind
    );
    if (created == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    ShadowSpillTaskTable *table = &plan->tasks;
    pthread_rwlock_wrlock(&table->lock);
    ShadowSpillTaskRecord *existing = find_unlocked(
        table, description->task_id
    );
    if (existing != NULL) {
        const int matches = same_description(
            existing, description, boundary_kind
        );
        pthread_rwlock_unlock(&table->lock);
        destroy_record(created);
        if (matches && result != NULL) {
            *result = existing;
        }
        return matches
            ? SHADOWSPILL_RUNTIME_OK
            : SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    const uint64_t bucket = task_bucket(table, created->task_id);
    created->hash_next = table->by_id[bucket];
    table->by_id[bucket] = created;
    created->ownership_next = table->owned_head;
    table->owned_head = created;
    pthread_rwlock_unlock(&table->lock);
    if (result != NULL) {
        *result = created;
    }
    return SHADOWSPILL_RUNTIME_OK;
}

ShadowSpillRuntimeStatus shadowspill_plan_admit_task(
    ShadowSpillPlan *plan,
    const ShadowSpillTaskDescription *description,
    const ShadowSpillTaskHandle **handle
) {
    if (handle == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    *handle = NULL;
    const ShadowSpillTaskRecord *record = NULL;
    const ShadowSpillRuntimeStatus status = admit_record(
        plan,
        description,
        SHADOWSPILL_BOUNDARY_TASK,
        &record
    );
    if (status == SHADOWSPILL_RUNTIME_OK) {
        *handle = record;
    }
    return status;
}

ShadowSpillRuntimeStatus shadowspill_plan_admit_action_batch(
    ShadowSpillPlan *plan,
    uint64_t batch_id,
    const ShadowSpillRuntimeAction *actions,
    uint32_t action_count,
    const ShadowSpillActionBatchHandle **handle
) {
    if (handle == NULL || (action_count != 0U && actions == NULL)) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    *handle = NULL;
    const ShadowSpillTaskDescription description = {
        .task_id = batch_id,
        .actions = actions,
        .action_count = action_count,
    };
    const ShadowSpillTaskRecord *record = NULL;
    const ShadowSpillRuntimeStatus status = admit_record(
        plan,
        &description,
        SHADOWSPILL_BOUNDARY_ACTION_BATCH,
        &record
    );
    if (status == SHADOWSPILL_RUNTIME_OK) {
        *handle = record;
    }
    return status;
}

ShadowSpillRuntimeStatus shadowspill_plan_clear_tasks(
    ShadowSpillPlan *plan
) {
    if (plan == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    ShadowSpillRuntime *runtime = plan->runtime;
    ShadowSpillRuntimeStatus status = shadowspill_runtime_wait_idle(runtime);
    if (status != SHADOWSPILL_RUNTIME_OK) {
        return status;
    }
    if (atomic_load_explicit(
            &runtime->actions.count, memory_order_acquire
        ) != 0U ||
        atomic_load_explicit(
            &runtime->pending_retirements, memory_order_acquire
        ) != 0U) {
        return SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    status = shadowspill_fixed_layout_clear(plan);
    if (status != SHADOWSPILL_RUNTIME_OK) {
        return status;
    }
    shadowspill_object_acquisitions_clear(plan);
    shadowspill_task_table_clear(&plan->tasks);
    shadowspill_plan_object_table_clear(&plan->object_bindings);
    return SHADOWSPILL_RUNTIME_OK;
}

ShadowSpillRuntimeStatus shadowspill_before_task_handle(
    ShadowSpillRuntime *runtime,
    const ShadowSpillTaskHandle *handle,
    ShadowSpillBackendStream compute_stream,
    ShadowSpillObjectBinding *bindings,
    uint32_t binding_capacity
) {
    const ShadowSpillTaskRecord *record = handle;
    if (runtime == NULL || record == NULL ||
        record->plan_owner == NULL || record->plan_owner->runtime != runtime ||
        record->boundary_kind != SHADOWSPILL_BOUNDARY_TASK) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    if ((record->input_count != 0U && bindings == NULL) ||
        binding_capacity < record->input_count) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }

    shadowspill_append_trace_event_locked(
        runtime,
        SHADOWSPILL_TRACE_BEFORE_TASK,
        record->task_id,
        SHADOWSPILL_RUNTIME_NO_ID,
        SHADOWSPILL_RUNTIME_NO_ID,
        0U,
        record->input_count,
        atomic_load_explicit(&runtime->actions.count, memory_order_acquire)
    );
    ShadowSpillRuntimeStatus status = shadowspill_acquire_object_bindings(
        runtime,
        record->task_id,
        record->unique_inputs,
        record->unique_input_count,
        record->input_unique_indices,
        record->unique_first_positions,
        record->input_count,
        compute_stream,
        bindings,
        binding_capacity
    );
    if (status == SHADOWSPILL_RUNTIME_OK &&
        shadowspill_enter_task_scope(runtime, record) != 0) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    return status;
}

ShadowSpillRuntimeStatus shadowspill_after_task_handle(
    ShadowSpillRuntime *runtime,
    const ShadowSpillTaskHandle *handle,
    ShadowSpillBackendStream compute_stream
) {
    const ShadowSpillTaskRecord *record = handle;
    if (runtime == NULL || record == NULL ||
        record->plan_owner == NULL || record->plan_owner->runtime != runtime ||
        record->boundary_kind != SHADOWSPILL_BOUNDARY_TASK) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    return shadowspill_after_task_record(runtime, record, compute_stream);
}

ShadowSpillRuntimeStatus shadowspill_submit_action_batch_handle(
    ShadowSpillRuntime *runtime,
    const ShadowSpillActionBatchHandle *handle,
    ShadowSpillBackendStream trigger_stream
) {
    const ShadowSpillTaskRecord *record = handle;
    if (runtime == NULL || record == NULL || record->plan_owner == NULL ||
        record->plan_owner->runtime != runtime ||
        record->boundary_kind != SHADOWSPILL_BOUNDARY_ACTION_BATCH) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    if (shadowspill_enter_task_scope(runtime, record) != 0) {
        return SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    return shadowspill_after_task_record(
        runtime, record, trigger_stream
    );
}
