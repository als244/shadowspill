#include "../internal.h"

#include <stdlib.h>

static void destroy_acquisition(
    ShadowSpillObjectAcquisitionRecord *record
) {
    if (record == NULL) {
        return;
    }
    for (uint32_t index = 0U; index < record->object_count; ++index) {
        shadowspill_object_release(record->objects[index]);
    }
    free(record->objects);
    free(record->unique_objects);
    free(record->object_unique_indices);
    free(record->unique_first_positions);
    free(record);
}

void shadowspill_object_acquisitions_clear(ShadowSpillPlan *plan) {
    if (plan == NULL) {
        return;
    }
    ShadowSpillObjectAcquisitionRecord *record = plan->object_acquisitions;
    plan->object_acquisitions = NULL;
    while (record != NULL) {
        ShadowSpillObjectAcquisitionRecord *next = record->ownership_next;
        destroy_acquisition(record);
        record = next;
    }
}

ShadowSpillRuntimeStatus shadowspill_plan_admit_object_acquisition(
    ShadowSpillPlan *plan,
    const uint64_t *object_ids,
    uint32_t object_count,
    const ShadowSpillObjectAcquisitionHandle **handle
) {
    if (plan == NULL || handle == NULL ||
        (object_count != 0U && object_ids == NULL)) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    *handle = NULL;
    pthread_mutex_lock(&plan->lifecycle_lock);
    if (atomic_load_explicit(&plan->closing, memory_order_acquire) != 0U ||
        atomic_load_explicit(&plan->closed, memory_order_acquire) != 0U) {
        pthread_mutex_unlock(&plan->lifecycle_lock);
        return SHADOWSPILL_RUNTIME_CLOSED;
    }
    ShadowSpillObjectAcquisitionRecord *record = calloc(1U, sizeof(*record));
    if (record == NULL) {
        pthread_mutex_unlock(&plan->lifecycle_lock);
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    record->plan_owner = plan;
    record->object_count = object_count;
    if (object_count != 0U) {
        record->objects = calloc(object_count, sizeof(*record->objects));
        record->unique_objects = calloc(
            object_count, sizeof(*record->unique_objects)
        );
        record->object_unique_indices = calloc(
            object_count, sizeof(*record->object_unique_indices)
        );
        record->unique_first_positions = calloc(
            object_count, sizeof(*record->unique_first_positions)
        );
    }
    if (object_count != 0U &&
        (record->objects == NULL || record->unique_objects == NULL ||
         record->object_unique_indices == NULL ||
         record->unique_first_positions == NULL)) {
        destroy_acquisition(record);
        pthread_mutex_unlock(&plan->lifecycle_lock);
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    for (uint32_t index = 0U; index < object_count; ++index) {
        ShadowSpillObject *object = shadowspill_plan_object_acquire(
            plan, object_ids[index], NULL
        );
        if (object == NULL) {
            record->object_count = index;
            destroy_acquisition(record);
            pthread_mutex_unlock(&plan->lifecycle_lock);
            return SHADOWSPILL_RUNTIME_INVALID_STATE;
        }
        record->objects[index] = object;
        uint32_t unique_index = record->unique_object_count;
        for (uint32_t prior = 0U;
             prior < record->unique_object_count; ++prior) {
            if (record->unique_objects[prior] == object) {
                unique_index = prior;
                break;
            }
        }
        if (unique_index == record->unique_object_count) {
            record->unique_objects[unique_index] = object;
            record->unique_first_positions[unique_index] = index;
            ++record->unique_object_count;
        }
        record->object_unique_indices[index] = unique_index;
    }
    record->ownership_next = plan->object_acquisitions;
    plan->object_acquisitions = record;
    *handle = record;
    pthread_mutex_unlock(&plan->lifecycle_lock);
    return SHADOWSPILL_RUNTIME_OK;
}

ShadowSpillRuntimeStatus shadowspill_acquire_object_bindings(
    ShadowSpillRuntime *runtime,
    const ShadowSpillPlan *plan,
    uint64_t trace_task_id,
    ShadowSpillObject *const *unique_objects,
    uint32_t unique_object_count,
    const uint32_t *object_unique_indices,
    const uint32_t *unique_first_positions,
    uint32_t object_count,
    ShadowSpillBackendStream consumer_stream,
    ShadowSpillObjectBinding *bindings,
    uint32_t binding_capacity
) {
    if (runtime == NULL || plan == NULL || plan->runtime != runtime ||
        (object_count != 0U &&
         (unique_objects == NULL || object_unique_indices == NULL ||
          unique_first_positions == NULL || bindings == NULL)) ||
        binding_capacity < object_count) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    ShadowSpillRuntimeStatus status = shadowspill_current_status_locked(runtime);
    for (uint32_t index = 0U;
         status == SHADOWSPILL_RUNTIME_OK && index < unique_object_count;
         ++index) {
        ShadowSpillObject *object = unique_objects[index];
        pthread_mutex_lock(&object->lock);
        const uint64_t object_id = object->object_id;
        const uint64_t allocation_id = object->allocation_id;
        const uint64_t size_bytes = object->size_bytes;
        if (shadowspill_object_has_unpublished_fetch_locked(object)) {
            pthread_mutex_unlock(&object->lock);
            shadowspill_append_trace_event_locked(
                runtime,
                SHADOWSPILL_TRACE_READINESS_WAIT,
                trace_task_id,
                object_id,
                allocation_id,
                size_bytes,
                0U,
                atomic_load_explicit(
                    &runtime->actions.count, memory_order_acquire
                )
            );
            shadowspill_latch_failure_locked(
                runtime,
                SHADOWSPILL_RUNTIME_INVALID_STATE,
                object_id,
                allocation_id,
                size_bytes
            );
            return SHADOWSPILL_RUNTIME_INVALID_STATE;
        }
        ShadowSpillMemoryLease *lease =
            shadowspill_plan_execution_location(plan, object)->lease;
        if ((object->residency != SHADOWSPILL_OBJECT_EXECUTION_READY &&
             object->residency != SHADOWSPILL_OBJECT_PREFETCHING) ||
            lease == NULL || lease->pointer == NULL ||
            lease->allocation_id != allocation_id ||
            lease->generation != object->generation ||
            shadowspill_plan_execution_location(plan, object)->version !=
                object->authoritative_version) {
            pthread_mutex_unlock(&object->lock);
            shadowspill_latch_failure_locked(
                runtime,
                SHADOWSPILL_RUNTIME_PLAN_VIOLATION,
                object_id,
                allocation_id,
                size_bytes
            );
            return SHADOWSPILL_RUNTIME_PLAN_VIOLATION;
        }
        ShadowSpillEventLease *readiness_event = NULL;
        if (object->residency == SHADOWSPILL_OBJECT_PREFETCHING) {
            if (!object->has_readiness_event) {
                pthread_mutex_unlock(&object->lock);
                shadowspill_latch_failure_locked(
                    runtime,
                    SHADOWSPILL_RUNTIME_BACKEND_FAILURE,
                    object_id,
                    allocation_id,
                    size_bytes
                );
                return SHADOWSPILL_RUNTIME_BACKEND_FAILURE;
            }
            readiness_event = object->readiness_event;
            shadowspill_event_lease_retain(readiness_event);
        }
        const ShadowSpillObjectBinding snapshot = {
            .object_id = object_id,
            .generation = object->generation,
            .allocation_id = allocation_id,
            .authoritative_version = object->authoritative_version,
            .pointer = lease->pointer,
        };
        bindings[unique_first_positions[index]] = snapshot;
        pthread_mutex_unlock(&object->lock);

        if (readiness_event != NULL) {
            if (runtime->synchronization.wait_event(
                    runtime->synchronization.context,
                    consumer_stream,
                    readiness_event->event
                ) != 0) {
                (void)shadowspill_event_lease_release(
                    runtime, readiness_event
                );
                shadowspill_latch_failure_locked(
                    runtime,
                    SHADOWSPILL_RUNTIME_BACKEND_FAILURE,
                    object_id,
                    allocation_id,
                    size_bytes
                );
                return SHADOWSPILL_RUNTIME_BACKEND_FAILURE;
            }
            const uint64_t wait_count = atomic_fetch_add_explicit(
                &runtime->wait_events_inserted, 1U, memory_order_acq_rel
            ) + 1U;
            shadowspill_append_trace_event_locked(
                runtime,
                SHADOWSPILL_TRACE_READINESS_WAIT,
                trace_task_id,
                object_id,
                allocation_id,
                size_bytes,
                1U,
                wait_count
            );
            (void)shadowspill_event_lease_release(runtime, readiness_event);
        }
    }
    for (uint32_t position = 0U;
         status == SHADOWSPILL_RUNTIME_OK && position < object_count;
         ++position) {
        const uint32_t first_position = unique_first_positions[
            object_unique_indices[position]
        ];
        bindings[position] = bindings[first_position];
    }
    return status;
}

ShadowSpillRuntimeStatus shadowspill_acquire_objects_handle(
    ShadowSpillRuntime *runtime,
    const ShadowSpillObjectAcquisitionHandle *handle,
    ShadowSpillBackendStream consumer_stream,
    ShadowSpillObjectBinding *bindings,
    uint32_t binding_capacity
) {
    const ShadowSpillObjectAcquisitionRecord *record = handle;
    if (runtime == NULL || record == NULL || record->plan_owner == NULL ||
        record->plan_owner->runtime != runtime) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    return shadowspill_acquire_object_bindings(
        runtime,
        record->plan_owner,
        SHADOWSPILL_RUNTIME_NO_ID,
        record->unique_objects,
        record->unique_object_count,
        record->object_unique_indices,
        record->unique_first_positions,
        record->object_count,
        consumer_stream,
        bindings,
        binding_capacity
    );
}

ShadowSpillRuntimeStatus shadowspill_transfer_acquired_object_to_caller(
    ShadowSpillRuntime *runtime,
    const ShadowSpillObjectAcquisitionHandle *handle,
    uint32_t object_ordinal,
    ShadowSpillBackendStream consumer_stream,
    const void *expected_pointer,
    uint64_t expected_generation,
    uint64_t expected_allocation_id,
    ShadowSpillAllocation *allocation
) {
    const ShadowSpillObjectAcquisitionRecord *record = handle;
    if (runtime == NULL || record == NULL || record->plan_owner == NULL ||
        record->plan_owner->runtime != runtime || allocation == NULL ||
        object_ordinal >= record->object_count || expected_pointer == NULL ||
        expected_allocation_id == SHADOWSPILL_RUNTIME_NO_ID) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    return shadowspill_object_transfer_to_caller(
        runtime,
        record->plan_owner->execution_pool,
        record->plan_owner->spill_pool,
        record->objects[object_ordinal],
        consumer_stream,
        expected_pointer,
        expected_generation,
        expected_allocation_id,
        1U,
        allocation
    );
}
