/* One description, turned into the record the runtime keeps. */
#include "../internal.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int shadowspill_task_same_description(
    const ShadowSpillTaskRecord *record,
    const ShadowSpillTaskDescription *description,
    uint8_t boundary_kind
) {
    const int same_trace_label =
        (record->trace_label == NULL && description->trace_label == NULL) ||
        (record->trace_label != NULL && description->trace_label != NULL &&
         strcmp(record->trace_label, description->trace_label) == 0);
    if (!same_trace_label || record->boundary_kind != boundary_kind ||
        record->input_count != description->input_count ||
        record->update_count != description->update_count ||
        record->publication_count != description->publication_count ||
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
    for (uint32_t index = 0U; index < record->publication_count; ++index) {
        if (record->publications[index].plan_object_id !=
                description->publications[index].object_id ||
            record->publications[index].kind !=
                description->publications[index].kind) {
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

/*
 * A record is made of six things, and each is taken in one step below. Every
 * step leaves what it took on the record and answers 0 or -1, so the entry
 * unwinds once -- `destroy_record` releases whatever any step had reached --
 * rather than repeating the unwind at each of the places a step can fail.
 */

static int copy_task_scalars(
    ShadowSpillTaskRecord *record,
    const ShadowSpillTaskDescription *description,
    uint8_t boundary_kind
) {
    if (description->trace_label != NULL) {
        const size_t label_bytes = strnlen(
            description->trace_label,
            SHADOWSPILL_RUNTIME_TRACE_LABEL_MAX_BYTES + 1U
        );
        if (label_bytes > SHADOWSPILL_RUNTIME_TRACE_LABEL_MAX_BYTES) {
            return -1;
        }
        record->trace_label = strdup(description->trace_label);
        if (record->trace_label == NULL) {
            return -1;
        }
    }
    record->boundary_kind = boundary_kind;
    atomic_init(&record->invocation_count, 0U);
    atomic_init(&record->submission_sequence, 0U);
    atomic_init(&record->submission_invocation, 0U);
    atomic_init(&record->acknowledgement_sequence, 0U);
    atomic_init(&record->invocation_active, 0U);
    record->input_count = description->input_count;
    record->update_count = description->update_count;
    record->publication_count = description->publication_count;
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
    return 0;
}

static int allocate_task_arrays(ShadowSpillTaskRecord *record) {
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
        record->input_bindings = calloc(
            record->input_count, sizeof(*record->input_bindings)
        );
    }
    if (record->update_count != 0U) {
        record->updates = calloc(record->update_count, sizeof(*record->updates));
    }
    if (record->publication_count != 0U) {
        record->publications = calloc(
            record->publication_count, sizeof(*record->publications)
        );
    }
    if (record->action_count != 0U) {
        record->actions = calloc(record->action_count, sizeof(*record->actions));
        record->queued_actions = calloc(
            record->action_count, sizeof(*record->queued_actions)
        );
        record->release_bindings = calloc(
            record->action_count, sizeof(*record->release_bindings)
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
          record->unique_first_positions == NULL ||
          record->input_bindings == NULL)) ||
        (record->update_count != 0U && record->updates == NULL) ||
        (record->publication_count != 0U && record->publications == NULL) ||
        (record->action_count != 0U &&
         (record->actions == NULL || record->queued_actions == NULL ||
          record->release_bindings == NULL)) ||
        (record->allocation_contract_step_count != 0U &&
         record->allocation_contract_steps == NULL)) {
        return -1;
    }
    return 0;
}

static int copy_allocation_contract(
    ShadowSpillTaskRecord *record,
    const ShadowSpillTaskDescription *description
) {
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
    if (record->allocation_contract_allocation_count != 0U) {
        record->allocation_contract_states = calloc(
            record->allocation_contract_allocation_count,
            sizeof(*record->allocation_contract_states)
        );
        if (record->allocation_contract_states == NULL) {
            return -1;
        }
    }
    return 0;
}

static int bind_task_inputs(
    ShadowSpillTaskRecord *record,
    ShadowSpillPlan *plan,
    const ShadowSpillTaskDescription *description
) {
    for (uint32_t index = 0U; index < record->input_count; ++index) {
        uint8_t consistency = SHADOWSPILL_OBJECT_CAUSAL;
        ShadowSpillObject *object = shadowspill_plan_object_acquire(
            plan, description->input_object_ids[index], &consistency
        );
        if (object == NULL) {
            return -1;
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
    return 0;
}

static int bind_task_updates(
    ShadowSpillTaskRecord *record,
    ShadowSpillPlan *plan,
    const ShadowSpillTaskDescription *description
) {
    for (uint32_t index = 0U; index < record->update_count; ++index) {
        ShadowSpillObject *object = shadowspill_plan_object_acquire(
            plan, description->updates[index].object_id, NULL
        );
        if (object == NULL) {
            return -1;
        }
        record->updates[index] = (ShadowSpillTaskUpdate){
            .object = object,
            .plan_object_id = description->updates[index].object_id,
            .version_delta = description->updates[index].version_delta,
        };
    }
    return 0;
}

static int bind_task_publications(
    ShadowSpillTaskRecord *record,
    ShadowSpillPlan *plan,
    const ShadowSpillTaskDescription *description
) {
    for (uint32_t index = 0U; index < record->publication_count; ++index) {
        const ShadowSpillTaskPublicationDescription *publication =
            &description->publications[index];
        if (publication->kind > SHADOWSPILL_TASK_PUBLICATION_REPLACE) {
            return -1;
        }
        for (uint32_t previous = 0U; previous < index; ++previous) {
            if (description->publications[previous].object_id ==
                publication->object_id) {
                return -1;
            }
        }
        ShadowSpillObject *object = shadowspill_plan_object_acquire(
            plan, publication->object_id, NULL
        );
        if (object == NULL) {
            return -1;
        }
        record->publications[index] = (ShadowSpillTaskPublication){
            .object = object,
            .plan_object_id = publication->object_id,
            .kind = publication->kind,
        };
    }
    return 0;
}

static int bind_task_actions(
    ShadowSpillTaskRecord *record,
    ShadowSpillPlan *plan,
    const ShadowSpillTaskDescription *description
) {
    for (uint32_t index = 0U; index < record->action_count; ++index) {
        if (description->actions[index].kind >
            SHADOWSPILL_RUNTIME_WRITE_BACK) {
            return -1;
        }
        for (uint32_t previous = 0U; previous < index; ++previous) {
            if (description->actions[previous].object_id ==
                description->actions[index].object_id) {
                return -1;
            }
        }
        ShadowSpillObject *object = shadowspill_plan_object_acquire(
            plan, description->actions[index].object_id, NULL
        );
        if (object == NULL) {
            return -1;
        }
        char *trace_label = shadowspill_copy_action_trace_label(
            &description->actions[index], record->task_id, object->size_bytes
        );
        if (trace_label == NULL) {
            shadowspill_object_release(object);
            return -1;
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
                        SHADOWSPILL_RUNTIME_FETCH
                    ? plan->fetch_route
                    : plan->evict_route,
            .trace_label = trace_label,
            .admitted = 1U,
            .background = record->boundary_kind ==
                    SHADOWSPILL_BOUNDARY_ACTION_BATCH
                ? 1U
                : 0U,
        };
        if (description->actions[index].kind ==
            SHADOWSPILL_RUNTIME_RELEASE) {
            record->release_bindings[record->release_binding_count++] =
                (ShadowSpillTaskReleaseBinding){
                    .object = object,
                    .action = &record->queued_actions[index],
                };
        }
    }
    return 0;
}

ShadowSpillTaskRecord *shadowspill_task_create_record(
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
    if (copy_task_scalars(record, description, boundary_kind) != 0 ||
        allocate_task_arrays(record) != 0 ||
        copy_allocation_contract(record, description) != 0 ||
        bind_task_inputs(record, plan, description) != 0 ||
        bind_task_updates(record, plan, description) != 0 ||
        bind_task_publications(record, plan, description) != 0 ||
        bind_task_actions(record, plan, description) != 0) {
        shadowspill_task_destroy_record(record);
        return NULL;
    }
    if (record->release_binding_count != 0U) {
        qsort(
            record->release_bindings,
            record->release_binding_count,
            sizeof(*record->release_bindings),
            shadowspill_task_compare_release_bindings
        );
    }
    return record;
}

int shadowspill_task_valid_allocation_contract(
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
