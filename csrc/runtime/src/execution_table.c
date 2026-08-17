#include "internal.h"
#include "internal/task_boundaries.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void wait_for_object_state_change(ShadowSpillObject *object) {
    struct timespec deadline = {0};
    if (clock_gettime(CLOCK_REALTIME, &deadline) != 0) {
        pthread_cond_wait(&object->state_changed, &object->lock);
        return;
    }
    const uint64_t nanoseconds =
        (uint64_t)deadline.tv_nsec + UINT64_C(1000000);
    deadline.tv_sec += (time_t)(nanoseconds / UINT64_C(1000000000));
    deadline.tv_nsec = (long)(nanoseconds % UINT64_C(1000000000));
    (void)pthread_cond_timedwait(
        &object->state_changed, &object->lock, &deadline
    );
}

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

static uint64_t execution_bucket(
    const ShadowSpillExecutionTable *table,
    uint64_t task_id
) {
    task_id ^= task_id >> 33U;
    task_id *= UINT64_C(0xff51afd7ed558ccd);
    task_id ^= task_id >> 33U;
    return task_id % table->bucket_count;
}

static ShadowSpillExecutionRecord *find_unlocked(
    const ShadowSpillExecutionTable *table,
    uint64_t task_id
) {
    if (table->by_id == NULL || table->bucket_count == 0U) {
        return NULL;
    }
    const uint64_t bucket = execution_bucket(table, task_id);
    for (ShadowSpillExecutionRecord *record = table->by_id[bucket];
         record != NULL; record = record->hash_next) {
        if (record->task_id == task_id) {
            return record;
        }
    }
    return NULL;
}

static void destroy_record(ShadowSpillExecutionRecord *record) {
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
    free(record->unique_inputs);
    free(record->input_unique_indices);
    free(record->unique_first_positions);
    free(record->updates);
    free(record->actions);
    free(record->queued_actions);
    free(record->allocation_abi_steps);
    free(record);
}

int shadowspill_execution_table_initialize(
    ShadowSpillExecutionTable *table,
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

void shadowspill_execution_table_destroy(ShadowSpillExecutionTable *table) {
    if (table == NULL) {
        return;
    }
    ShadowSpillExecutionRecord *record = table->owned_head;
    while (record != NULL) {
        ShadowSpillExecutionRecord *next = record->ownership_next;
        destroy_record(record);
        record = next;
    }
    free(table->by_id);
    if (table->lock_initialized) {
        pthread_rwlock_destroy(&table->lock);
    }
    *table = (ShadowSpillExecutionTable){0};
}

void shadowspill_execution_table_clear(ShadowSpillExecutionTable *table) {
    if (table == NULL || !table->lock_initialized) {
        return;
    }
    pthread_rwlock_wrlock(&table->lock);
    ShadowSpillExecutionRecord *record = table->owned_head;
    table->owned_head = NULL;
    memset(
        table->by_id,
        0,
        (size_t)table->bucket_count * sizeof(*table->by_id)
    );
    pthread_rwlock_unlock(&table->lock);

    while (record != NULL) {
        ShadowSpillExecutionRecord *next = record->ownership_next;
        destroy_record(record);
        record = next;
    }
}

ShadowSpillExecutionRecord *shadowspill_execution_table_acquire(
    ShadowSpillExecutionTable *table,
    uint64_t task_id
) {
    if (table == NULL || !table->lock_initialized) {
        return NULL;
    }
    pthread_rwlock_rdlock(&table->lock);
    ShadowSpillExecutionRecord *record = find_unlocked(table, task_id);
    pthread_rwlock_unlock(&table->lock);
    return record;
}

static int same_description(
    const ShadowSpillExecutionRecord *record,
    const ShadowSpillExecutionDescription *description
) {
    if (record->input_count != description->input_count ||
        record->update_count != description->update_count ||
        record->action_count != description->action_count ||
        record->allocation_abi_step_count !=
            description->allocation_abi_step_count ||
        record->enforce_allocation_abi !=
            description->enforce_allocation_abi ||
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
        if (record->inputs[index]->object_id !=
            description->input_object_ids[index]) {
            return 0;
        }
    }
    for (uint32_t index = 0U; index < record->update_count; ++index) {
        if (record->updates[index].object->object_id !=
                description->updates[index].object_id ||
            record->updates[index].version_delta !=
                description->updates[index].version_delta) {
            return 0;
        }
    }
    for (uint32_t index = 0U; index < record->action_count; ++index) {
        if (record->actions[index].object->object_id !=
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
         index < record->allocation_abi_step_count; ++index) {
        const ShadowSpillTaskAllocationABIStep *left =
            &record->allocation_abi_steps[index];
        const ShadowSpillTaskAllocationABIStep *right =
            &description->allocation_abi_steps[index];
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

static ShadowSpillExecutionRecord *create_record(
    ShadowSpillPlan *plan,
    const ShadowSpillExecutionDescription *description
) {
    ShadowSpillRuntime *runtime = plan->runtime;
    ShadowSpillExecutionRecord *record = calloc(1U, sizeof(*record));
    if (record == NULL) {
        return NULL;
    }
    record->plan_owner = plan;
    record->task_id = description->task_id;
    atomic_init(&record->invocation_count, 0U);
    record->input_count = description->input_count;
    record->update_count = description->update_count;
    record->action_count = description->action_count;
    record->allocation_abi_step_count =
        description->allocation_abi_step_count;
    record->enforce_allocation_abi = description->enforce_allocation_abi;
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
    if (record->allocation_abi_step_count != 0U) {
        record->allocation_abi_steps = calloc(
            record->allocation_abi_step_count,
            sizeof(*record->allocation_abi_steps)
        );
    }
    if ((record->input_count != 0U &&
         (record->inputs == NULL || record->unique_inputs == NULL ||
          record->input_unique_indices == NULL ||
          record->unique_first_positions == NULL)) ||
        (record->update_count != 0U && record->updates == NULL) ||
        (record->action_count != 0U &&
         (record->actions == NULL || record->queued_actions == NULL)) ||
        (record->allocation_abi_step_count != 0U &&
         record->allocation_abi_steps == NULL)) {
        destroy_record(record);
        return NULL;
    }
    if (record->allocation_abi_step_count != 0U) {
        memcpy(
            record->allocation_abi_steps,
            description->allocation_abi_steps,
            record->allocation_abi_step_count *
                sizeof(*record->allocation_abi_steps)
        );
        for (uint32_t index = 0U;
             index < record->allocation_abi_step_count;
             ++index) {
            const ShadowSpillTaskAllocationABIStep *step =
                &record->allocation_abi_steps[index];
            if (step->operation == SHADOWSPILL_TASK_ALLOCATION_ALLOCATE) {
                record->allocation_abi_allocation_count =
                    (uint32_t)(step->allocation_ordinal + 1U);
            }
        }
    }
    for (uint32_t index = 0U; index < record->input_count; ++index) {
        ShadowSpillObject *object = shadowspill_object_table_acquire(
            &runtime->objects, description->input_object_ids[index]
        );
        if (object == NULL) {
            destroy_record(record);
            return NULL;
        }
        record->inputs[index] = object;
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
        ShadowSpillObject *object = shadowspill_object_table_acquire(
            &runtime->objects, description->updates[index].object_id
        );
        if (object == NULL) {
            destroy_record(record);
            return NULL;
        }
        record->updates[index] = (ShadowSpillExecutionUpdate){
            .object = object,
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
        ShadowSpillObject *object = shadowspill_object_table_acquire(
            &runtime->objects, description->actions[index].object_id
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
        record->actions[index] = (ShadowSpillExecutionAction){
            .object = object,
            .kind = description->actions[index].kind,
            .trace_label = trace_label,
        };
        record->queued_actions[index] = (ShadowSpillQueuedAction){
            .task_id = record->task_id,
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

static int valid_allocation_abi(
    const ShadowSpillExecutionDescription *description
) {
    if (!description->enforce_allocation_abi) {
        return description->allocation_abi_step_count == 0U;
    }
    const uint32_t count = description->allocation_abi_step_count;
    const ShadowSpillTaskAllocationABIStep **allocations = count == 0U
        ? NULL
        : calloc(count, sizeof(*allocations));
    if (count != 0U && allocations == NULL) {
        return 0;
    }
    uint64_t next_ordinal = 0U;
    int valid = 1;
    for (uint32_t index = 0U; index < count && valid; ++index) {
        const ShadowSpillTaskAllocationABIStep *step =
            &description->allocation_abi_steps[index];
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
        const ShadowSpillTaskAllocationABIStep *allocation =
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

ShadowSpillRuntimeStatus shadowspill_plan_admit_execution(
    ShadowSpillPlan *plan,
    const ShadowSpillExecutionDescription *description
) {
    if (plan == NULL || description == NULL ||
        (description->input_count != 0U &&
         description->input_object_ids == NULL) ||
        (description->update_count != 0U && description->updates == NULL) ||
        (description->action_count != 0U && description->actions == NULL) ||
        (description->allocation_abi_step_count != 0U &&
         description->allocation_abi_steps == NULL) ||
        !valid_allocation_abi(description) ||
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
    ShadowSpillExecutionRecord *created = create_record(plan, description);
    if (created == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    ShadowSpillExecutionTable *table = &plan->execution;
    pthread_rwlock_wrlock(&table->lock);
    ShadowSpillExecutionRecord *existing = find_unlocked(
        table, description->task_id
    );
    if (existing != NULL) {
        const int matches = same_description(existing, description);
        pthread_rwlock_unlock(&table->lock);
        destroy_record(created);
        return matches
            ? SHADOWSPILL_RUNTIME_OK
            : SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    const uint64_t bucket = execution_bucket(table, created->task_id);
    created->hash_next = table->by_id[bucket];
    table->by_id[bucket] = created;
    created->ownership_next = table->owned_head;
    table->owned_head = created;
    pthread_rwlock_unlock(&table->lock);
    return SHADOWSPILL_RUNTIME_OK;
}

ShadowSpillRuntimeStatus shadowspill_plan_clear_execution(
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
    shadowspill_execution_table_clear(&plan->execution);
    return SHADOWSPILL_RUNTIME_OK;
}

ShadowSpillRuntimeStatus shadowspill_admit_execution(
    ShadowSpillRuntime *runtime,
    const ShadowSpillExecutionDescription *description
) {
    return runtime == NULL || runtime->default_plan == NULL
        ? SHADOWSPILL_RUNTIME_INVALID_ARGUMENT
        : shadowspill_plan_admit_execution(runtime->default_plan, description);
}

ShadowSpillRuntimeStatus shadowspill_clear_execution_plan(
    ShadowSpillRuntime *runtime
) {
    return runtime == NULL || runtime->default_plan == NULL
        ? SHADOWSPILL_RUNTIME_INVALID_ARGUMENT
        : shadowspill_plan_clear_execution(runtime->default_plan);
}

ShadowSpillRuntimeStatus shadowspill_before_execution(
    ShadowSpillRuntime *runtime,
    uint64_t task_id,
    ShadowSpillBackendStream compute_stream,
    ShadowSpillObjectBinding *bindings,
    uint32_t binding_capacity
) {
    if (runtime == NULL || runtime->default_plan == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    ShadowSpillExecutionRecord *record = shadowspill_execution_table_acquire(
        &runtime->default_plan->execution, task_id
    );
    if (record == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    return shadowspill_before_execution_handle(
        runtime, record, compute_stream, bindings, binding_capacity
    );
}

ShadowSpillRuntimeStatus shadowspill_resolve_execution(
    ShadowSpillRuntime *runtime,
    uint64_t task_id,
    const ShadowSpillExecutionHandle **handle
) {
    if (runtime == NULL || runtime->default_plan == NULL || handle == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    return shadowspill_plan_resolve_execution(
        runtime->default_plan, task_id, handle
    );
}

ShadowSpillRuntimeStatus shadowspill_plan_resolve_execution(
    ShadowSpillPlan *plan,
    uint64_t task_id,
    const ShadowSpillExecutionHandle **handle
) {
    if (plan == NULL || handle == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    ShadowSpillExecutionRecord *record = shadowspill_execution_table_acquire(
        &plan->execution, task_id
    );
    if (record == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    *handle = record;
    return SHADOWSPILL_RUNTIME_OK;
}

ShadowSpillRuntimeStatus shadowspill_before_execution_handle(
    ShadowSpillRuntime *runtime,
    const ShadowSpillExecutionHandle *handle,
    ShadowSpillBackendStream compute_stream,
    ShadowSpillObjectBinding *bindings,
    uint32_t binding_capacity
) {
    const ShadowSpillExecutionRecord *record = handle;
    if (runtime == NULL || record == NULL ||
        record->plan_owner == NULL || record->plan_owner->runtime != runtime) {
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
    ShadowSpillRuntimeStatus status = shadowspill_current_status_locked(runtime);
    for (uint32_t index = 0U;
         status == SHADOWSPILL_RUNTIME_OK &&
             index < record->unique_input_count;
         ++index) {
        ShadowSpillObject *object = record->unique_inputs[index];
        pthread_mutex_lock(&object->lock);
        while (status == SHADOWSPILL_RUNTIME_OK &&
               shadowspill_object_has_unpublished_fetch_locked(object)) {
            shadowspill_append_trace_event_locked(
                runtime,
                SHADOWSPILL_TRACE_READINESS_WAIT,
                record->task_id,
                object->object_id,
                object->allocation_id,
                object->size_bytes,
                0U,
                atomic_load_explicit(
                    &runtime->actions.count, memory_order_acquire
                )
            );
            /*
             * Normal readiness publication wakes this condition directly.
             * The bounded wait additionally guarantees that a failure latched
             * for another object is observed even when no further transition
             * can signal this particular object.
             */
            wait_for_object_state_change(object);
            status = shadowspill_current_status_locked(runtime);
        }
        ShadowSpillMemoryLease *lease = shadowspill_execution_location(runtime, object)->lease;
        if (status != SHADOWSPILL_RUNTIME_OK) {
            pthread_mutex_unlock(&object->lock);
            break;
        }
        if ((object->residency != SHADOWSPILL_OBJECT_EXECUTION_READY &&
             object->residency != SHADOWSPILL_OBJECT_PREFETCHING) ||
            lease == NULL || lease->pointer == NULL ||
            lease->allocation_id != object->allocation_id ||
            lease->generation != object->generation ||
            shadowspill_execution_location(runtime, object)->version != object->authoritative_version) {
            status = SHADOWSPILL_RUNTIME_PLAN_VIOLATION;
            const uint64_t allocation_id = object->allocation_id;
            const uint64_t size_bytes = object->size_bytes;
            pthread_mutex_unlock(&object->lock);
            shadowspill_latch_failure_locked(
                runtime,
                status,
                object->object_id,
                allocation_id,
                size_bytes
            );
            break;
        }
        ShadowSpillEventLease *readiness_event = NULL;
        if (object->residency == SHADOWSPILL_OBJECT_PREFETCHING) {
            if (!object->has_readiness_event) {
                status = SHADOWSPILL_RUNTIME_BACKEND_FAILURE;
                const uint64_t allocation_id = object->allocation_id;
                const uint64_t size_bytes = object->size_bytes;
                pthread_mutex_unlock(&object->lock);
                shadowspill_latch_failure_locked(
                    runtime,
                    status,
                    object->object_id,
                    allocation_id,
                    size_bytes
                );
                break;
            }
            readiness_event = object->readiness_event;
            shadowspill_event_lease_retain(readiness_event);
        }
        const ShadowSpillObjectBinding snapshot = {
            .object_id = object->object_id,
            .generation = object->generation,
            .allocation_id = object->allocation_id,
            .authoritative_version = object->authoritative_version,
            .pointer = lease->pointer,
        };
        bindings[record->unique_first_positions[index]] = snapshot;
        pthread_mutex_unlock(&object->lock);
        if (readiness_event != NULL) {
            if (runtime->synchronization.wait_event(
                    runtime->synchronization.context,
                    compute_stream,
                    readiness_event->event
                ) != 0) {
                (void)shadowspill_event_lease_release(
                    runtime, readiness_event
                );
                status = SHADOWSPILL_RUNTIME_BACKEND_FAILURE;
                shadowspill_latch_failure_locked(
                    runtime,
                    status,
                    snapshot.object_id,
                    snapshot.allocation_id,
                    object->size_bytes
                );
                break;
            }
            const uint64_t wait_count = atomic_fetch_add_explicit(
                &runtime->wait_events_inserted, 1U, memory_order_acq_rel
            ) + 1U;
            shadowspill_append_trace_event_locked(
                runtime,
                SHADOWSPILL_TRACE_READINESS_WAIT,
                record->task_id,
                snapshot.object_id,
                snapshot.allocation_id,
                object->size_bytes,
                1U,
                wait_count
            );
            (void)shadowspill_event_lease_release(runtime, readiness_event);
        }
    }
    for (uint32_t position = 0U;
         status == SHADOWSPILL_RUNTIME_OK && position < record->input_count;
         ++position) {
        const uint32_t first_position = record->unique_first_positions[
            record->input_unique_indices[position]
        ];
        bindings[position] = bindings[first_position];
    }
    if (status == SHADOWSPILL_RUNTIME_OK &&
        shadowspill_enter_execution_scope(runtime, record) != 0) {
        status = SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    return status;
}

ShadowSpillRuntimeStatus shadowspill_after_execution(
    ShadowSpillRuntime *runtime,
    uint64_t task_id,
    ShadowSpillBackendStream compute_stream
) {
    if (runtime == NULL || runtime->default_plan == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    ShadowSpillExecutionRecord *record = shadowspill_execution_table_acquire(
        &runtime->default_plan->execution, task_id
    );
    if (record == NULL) {
        return SHADOWSPILL_RUNTIME_INVALID_STATE;
    }
    return shadowspill_after_execution_handle(runtime, record, compute_stream);
}

ShadowSpillRuntimeStatus shadowspill_after_execution_handle(
    ShadowSpillRuntime *runtime,
    const ShadowSpillExecutionHandle *handle,
    ShadowSpillBackendStream compute_stream
) {
    const ShadowSpillExecutionRecord *record = handle;
    if (runtime == NULL || record == NULL ||
        record->plan_owner == NULL || record->plan_owner->runtime != runtime) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    return shadowspill_after_execution_record(runtime, record, compute_stream);
}
