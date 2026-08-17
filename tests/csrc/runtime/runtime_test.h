#ifndef SHADOWSPILL_RUNTIME_TEST_H
#define SHADOWSPILL_RUNTIME_TEST_H

#include <stdint.h>

#include <shadowspill/backend_mock.h>
#include <shadowspill/runtime.h>

enum {
    SHADOWSPILL_TEST_MAX_RUNTIMES = 16,
    SHADOWSPILL_TEST_MAX_TASKS = 512,
};

typedef struct ShadowSpillTestTask {
    uint64_t task_id;
    const ShadowSpillTaskHandle *handle;
} ShadowSpillTestTask;

typedef struct ShadowSpillTestRuntime {
    ShadowSpillRuntime *runtime;
    ShadowSpillPlan *plan;
    ShadowSpillTestTask tasks[SHADOWSPILL_TEST_MAX_TASKS];
    uint32_t task_count;
} ShadowSpillTestRuntime;

static ShadowSpillTestRuntime
    shadowspill_test_runtimes[SHADOWSPILL_TEST_MAX_RUNTIMES];

static inline ShadowSpillTestRuntime *shadowspill_test_runtime_record(
    ShadowSpillRuntime *runtime,
    int create
) {
    ShadowSpillTestRuntime *empty = NULL;
    for (uint32_t index = 0U; index < SHADOWSPILL_TEST_MAX_RUNTIMES; ++index) {
        ShadowSpillTestRuntime *record = &shadowspill_test_runtimes[index];
        if (record->runtime == runtime) {
            return record;
        }
        if (empty == NULL && record->runtime == NULL) {
            empty = record;
        }
    }
    if (!create || empty == NULL || runtime == NULL) {
        return NULL;
    }
    ShadowSpillPlan *plan = NULL;
    if (shadowspill_plan_create_for_pools(runtime, 0U, 1U, &plan) !=
            SHADOWSPILL_RUNTIME_OK) {
        return NULL;
    }
    empty->runtime = runtime;
    empty->plan = plan;
    empty->task_count = 0U;
    return empty;
}

static inline ShadowSpillRuntimeStatus shadowspill_test_bind_object(
    ShadowSpillTestRuntime *record,
    uint64_t object_id
) {
    ShadowSpillObjectHandle *object = NULL;
    ShadowSpillRuntimeStatus status = shadowspill_object_handle_acquire(
        record->runtime, object_id, &object
    );
    if (status == SHADOWSPILL_RUNTIME_OK) {
        status = shadowspill_plan_bind_object(
            record->plan, object_id, object, SHADOWSPILL_OBJECT_CAUSAL
        );
    }
    const ShadowSpillRuntimeStatus release_status =
        shadowspill_object_handle_release(object);
    return status == SHADOWSPILL_RUNTIME_OK ? release_status : status;
}

static inline ShadowSpillRuntimeStatus shadowspill_test_bind_task_objects(
    ShadowSpillTestRuntime *record,
    const ShadowSpillTaskDescription *description
) {
    for (uint32_t index = 0U; index < description->input_count; ++index) {
        const ShadowSpillRuntimeStatus status = shadowspill_test_bind_object(
            record, description->input_object_ids[index]
        );
        if (status != SHADOWSPILL_RUNTIME_OK) {
            return status;
        }
    }
    for (uint32_t index = 0U; index < description->update_count; ++index) {
        const ShadowSpillRuntimeStatus status = shadowspill_test_bind_object(
            record, description->updates[index].object_id
        );
        if (status != SHADOWSPILL_RUNTIME_OK) {
            return status;
        }
    }
    for (uint32_t index = 0U; index < description->action_count; ++index) {
        const ShadowSpillRuntimeStatus status = shadowspill_test_bind_object(
            record, description->actions[index].object_id
        );
        if (status != SHADOWSPILL_RUNTIME_OK) {
            return status;
        }
    }
    return SHADOWSPILL_RUNTIME_OK;
}

static inline ShadowSpillRuntimeStatus shadowspill_test_admit_task(
    ShadowSpillRuntime *runtime,
    const ShadowSpillTaskDescription *description
) {
    ShadowSpillTestRuntime *record = shadowspill_test_runtime_record(runtime, 1);
    if (record == NULL || description == NULL ||
        record->task_count == SHADOWSPILL_TEST_MAX_TASKS) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    ShadowSpillRuntimeStatus status = shadowspill_test_bind_task_objects(
        record, description
    );
    if (status != SHADOWSPILL_RUNTIME_OK) {
        return status;
    }
    const ShadowSpillTaskHandle *handle = NULL;
    status = shadowspill_plan_admit_task(record->plan, description, &handle);
    if (status != SHADOWSPILL_RUNTIME_OK) {
        return status;
    }
    for (uint32_t index = 0U; index < record->task_count; ++index) {
        if (record->tasks[index].task_id == description->task_id) {
            record->tasks[index].handle = handle;
            return SHADOWSPILL_RUNTIME_OK;
        }
    }
    record->tasks[record->task_count++] = (ShadowSpillTestTask){
        .task_id = description->task_id,
        .handle = handle,
    };
    return SHADOWSPILL_RUNTIME_OK;
}

static inline const ShadowSpillTaskHandle *shadowspill_test_task_handle(
    ShadowSpillRuntime *runtime,
    uint64_t task_id
) {
    const ShadowSpillTestRuntime *record =
        shadowspill_test_runtime_record(runtime, 0);
    if (record == NULL) {
        return NULL;
    }
    for (uint32_t index = 0U; index < record->task_count; ++index) {
        if (record->tasks[index].task_id == task_id) {
            return record->tasks[index].handle;
        }
    }
    return NULL;
}

static inline ShadowSpillRuntimeStatus shadowspill_test_before_task(
    ShadowSpillRuntime *runtime,
    uint64_t task_id,
    ShadowSpillBackendStream stream,
    ShadowSpillObjectBinding *bindings,
    uint32_t binding_capacity
) {
    const ShadowSpillTaskHandle *handle = shadowspill_test_task_handle(
        runtime, task_id
    );
    return handle == NULL
        ? SHADOWSPILL_RUNTIME_INVALID_STATE
        : shadowspill_before_task_handle(
              runtime, handle, stream, bindings, binding_capacity
          );
}

static inline ShadowSpillRuntimeStatus shadowspill_test_after_task(
    ShadowSpillRuntime *runtime,
    uint64_t task_id,
    ShadowSpillBackendStream stream
) {
    const ShadowSpillTaskHandle *handle = shadowspill_test_task_handle(
        runtime, task_id
    );
    return handle == NULL
        ? SHADOWSPILL_RUNTIME_INVALID_STATE
        : shadowspill_after_task_handle(runtime, handle, stream);
}

static inline ShadowSpillRuntimeStatus shadowspill_test_submit_actions(
    ShadowSpillRuntime *runtime,
    uint64_t batch_id,
    ShadowSpillBackendStream stream,
    const ShadowSpillRuntimeAction *actions,
    uint32_t action_count
) {
    ShadowSpillTestRuntime *record = shadowspill_test_runtime_record(runtime, 1);
    if (record == NULL || (action_count != 0U && actions == NULL)) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    for (uint32_t index = 0U; index < action_count; ++index) {
        const ShadowSpillRuntimeStatus status = shadowspill_test_bind_object(
            record, actions[index].object_id
        );
        if (status != SHADOWSPILL_RUNTIME_OK) {
            return status;
        }
    }
    const ShadowSpillActionBatchHandle *handle = NULL;
    ShadowSpillRuntimeStatus status = shadowspill_plan_admit_action_batch(
        record->plan, batch_id, actions, action_count, &handle
    );
    return status == SHADOWSPILL_RUNTIME_OK
        ? shadowspill_submit_action_batch_handle(runtime, handle, stream)
        : status;
}

static inline ShadowSpillRuntimeStatus shadowspill_test_admit_fixed_layout(
    ShadowSpillRuntime *runtime,
    const ShadowSpillFixedLayoutDescription *description
) {
    ShadowSpillTestRuntime *record = shadowspill_test_runtime_record(runtime, 1);
    return record == NULL
        ? SHADOWSPILL_RUNTIME_INVALID_STATE
        : shadowspill_plan_admit_fixed_layout(record->plan, description);
}

static inline ShadowSpillRuntimeStatus shadowspill_test_seal_fixed_layout(
    ShadowSpillRuntime *runtime
) {
    ShadowSpillTestRuntime *record = shadowspill_test_runtime_record(runtime, 0);
    return record == NULL
        ? SHADOWSPILL_RUNTIME_INVALID_STATE
        : shadowspill_plan_seal_fixed_layout(record->plan);
}

static inline ShadowSpillRuntimeStatus shadowspill_test_clear_plan(
    ShadowSpillRuntime *runtime
) {
    ShadowSpillTestRuntime *record = shadowspill_test_runtime_record(runtime, 0);
    const ShadowSpillRuntimeStatus status = record == NULL
        ? SHADOWSPILL_RUNTIME_INVALID_STATE
        : shadowspill_plan_clear_tasks(record->plan);
    if (status == SHADOWSPILL_RUNTIME_OK && record != NULL) {
        record->task_count = 0U;
    }
    return status;
}

static inline void shadowspill_test_destroy_runtime(ShadowSpillRuntime *runtime) {
    ShadowSpillTestRuntime *record = shadowspill_test_runtime_record(runtime, 0);
    if (record != NULL) {
        record->runtime = NULL;
        record->plan = NULL;
        record->task_count = 0U;
    }
    shadowspill_runtime_destroy(runtime);
}

static inline ShadowSpillRuntimeStatus shadowspill_test_create_runtime(
    ShadowSpillMockBackend *backend,
    uint64_t execution_pool_bytes,
    uint64_t spill_pool_bytes,
    uint64_t minimum_alignment,
    uint64_t worker_poll_nanoseconds,
    ShadowSpillRuntime **runtime
) {
    ShadowSpillMockRuntimeTopology topology;
    shadowspill_mock_runtime_topology(
        backend,
        execution_pool_bytes,
        spill_pool_bytes,
        minimum_alignment,
        worker_poll_nanoseconds,
        &topology
    );
    return shadowspill_runtime_create(&topology.runtime, runtime);
}

#endif
