#include "../internal.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct ShadowSpillTaskScope {
    ShadowSpillRuntime *runtime;
    ShadowSpillMemoryPool *allocation_pool;
    uint64_t task_id;
    const ShadowSpillTaskRecord *task;
    uint64_t invocation;
    uint64_t operation_index;
    uint64_t allocation_sequence;
    uint64_t pending_invariant_ordinal;
    uint8_t pending_is_scratch;
    uint8_t pending_allocation;
    uint8_t *allocation_states;
    uint32_t allocation_state_count;
    uint64_t live_requested_bytes;
    uint64_t live_charged_bytes;
    uint64_t peak_requested_bytes;
    uint64_t peak_charged_bytes;
    uint64_t allocation_count;
    uint64_t free_count;
    uint64_t scratch_live_requested_bytes;
    uint64_t scratch_live_charged_bytes;
    uint64_t scratch_peak_charged_bytes;
    uint64_t scratch_allocation_count;
    ShadowSpillMemoryLease *retirement_head;
    ShadowSpillMemoryLease *retirement_tail;
} ShadowSpillTaskScope;

enum {
    SHADOWSPILL_TASK_ALLOCATION_UNSEEN = 0U,
    SHADOWSPILL_TASK_ALLOCATION_MATCHED = 1U,
    SHADOWSPILL_TASK_ALLOCATION_OMITTED = 2U,
    SHADOWSPILL_TASK_ALLOCATION_OMIT_CANDIDATE = 3U,
};

static _Thread_local ShadowSpillTaskScope task_scope = {
    .runtime = NULL,
    .allocation_pool = NULL,
    .task_id = SHADOWSPILL_RUNTIME_NO_ID,
    .task = NULL,
};

uint64_t shadowspill_current_task_id(ShadowSpillRuntime *runtime) {
    return task_scope.runtime == runtime
        ? task_scope.task_id
        : SHADOWSPILL_RUNTIME_NO_ID;
}

uint64_t shadowspill_current_task_allocation_ordinal(
    ShadowSpillRuntime *runtime
) {
    return task_scope.runtime == runtime && task_scope.task != NULL
        ? task_scope.allocation_sequence
        : SHADOWSPILL_RUNTIME_NO_ID;
}

uint64_t shadowspill_current_task_invariant_allocation_ordinal(
    ShadowSpillRuntime *runtime
) {
    return task_scope.runtime == runtime && task_scope.task != NULL &&
            task_scope.pending_allocation && !task_scope.pending_is_scratch
        ? task_scope.pending_invariant_ordinal
        : SHADOWSPILL_RUNTIME_NO_ID;
}

int shadowspill_current_task_allocation_is_scratch(
    ShadowSpillRuntime *runtime
) {
    return task_scope.runtime == runtime && task_scope.task != NULL &&
        task_scope.pending_allocation && task_scope.pending_is_scratch;
}

uint64_t shadowspill_current_task_invocation(ShadowSpillRuntime *runtime) {
    return task_scope.runtime == runtime && task_scope.task != NULL
        ? task_scope.invocation
        : 0U;
}

ShadowSpillPlan *shadowspill_current_plan(ShadowSpillRuntime *runtime) {
    return task_scope.runtime == runtime && task_scope.task != NULL
        ? task_scope.task->plan_owner
        : NULL;
}

ShadowSpillMemoryPool *shadowspill_current_allocation_pool(
    ShadowSpillRuntime *runtime
) {
    return task_scope.runtime == runtime ? task_scope.allocation_pool : NULL;
}

int shadowspill_track_task_retirement(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryLease *lease
) {
    if (task_scope.runtime != runtime || lease == NULL) {
        return -1;
    }
    if (lease->task_retirement_linked) {
        return 0;
    }
    lease->task_retirement_next = NULL;
    lease->task_retirement_linked = 1U;
    if (task_scope.retirement_tail == NULL) {
        task_scope.retirement_head = lease;
    } else {
        task_scope.retirement_tail->task_retirement_next = lease;
    }
    task_scope.retirement_tail = lease;
    return 0;
}

ShadowSpillMemoryLease *shadowspill_current_task_retirements(
    ShadowSpillRuntime *runtime
) {
    return task_scope.runtime == runtime
        ? task_scope.retirement_head : NULL;
}

int shadowspill_enter_allocation_scope(
    ShadowSpillRuntime *runtime,
    ShadowSpillMemoryPool *pool,
    uint64_t task_id
) {
    if (runtime == NULL || pool == NULL || task_scope.runtime != NULL ||
        task_id == SHADOWSPILL_RUNTIME_NO_ID) {
        return -1;
    }
    task_scope.runtime = runtime;
    task_scope.allocation_pool = pool;
    task_scope.task_id = task_id;
    task_scope.task = NULL;
    task_scope.invocation = 0U;
    task_scope.operation_index = 0U;
    task_scope.allocation_sequence = 0U;
    task_scope.pending_invariant_ordinal = SHADOWSPILL_RUNTIME_NO_ID;
    task_scope.pending_is_scratch = 0U;
    task_scope.pending_allocation = 0U;
    task_scope.allocation_states = NULL;
    task_scope.allocation_state_count = 0U;
    task_scope.live_requested_bytes = 0U;
    task_scope.live_charged_bytes = 0U;
    task_scope.peak_requested_bytes = 0U;
    task_scope.peak_charged_bytes = 0U;
    task_scope.allocation_count = 0U;
    task_scope.free_count = 0U;
    task_scope.scratch_live_requested_bytes = 0U;
    task_scope.scratch_live_charged_bytes = 0U;
    task_scope.scratch_peak_charged_bytes = 0U;
    task_scope.scratch_allocation_count = 0U;
    task_scope.retirement_head = NULL;
    task_scope.retirement_tail = NULL;
    return 0;
}

static void prepare_allocation_matcher(
    const ShadowSpillTaskRecord *record
) {
    const uint32_t count = record->allocation_contract_allocation_count;
    task_scope.allocation_states = record->allocation_contract_states;
    if (count != 0U) {
        memset(task_scope.allocation_states, 0, count);
    }
    task_scope.allocation_state_count = count;
}

static const ShadowSpillTaskAllocationContractStep *expected_allocation_step(void) {
    if (task_scope.task == NULL ||
        !task_scope.task->enforce_allocation_contract ||
        task_scope.operation_index >=
            task_scope.task->allocation_contract_step_count) {
        return NULL;
    }
    return &task_scope.task->allocation_contract_steps[
        task_scope.operation_index
    ];
}

static ShadowSpillStatus latch_allocation_contract_mismatch(
    ShadowSpillRuntime *runtime,
    uint8_t actual_operation,
    uint64_t actual_ordinal,
    uint64_t requested_bytes,
    uint64_t charged_bytes,
    uint64_t alignment_bytes
) {
    const ShadowSpillTaskAllocationContractStep *expected =
        expected_allocation_step();
    const ShadowSpillTaskAllocationMismatch mismatch = {
        .operation_index = task_scope.operation_index,
        .expected_ordinal = expected == NULL
            ? SHADOWSPILL_RUNTIME_NO_ID : expected->allocation_ordinal,
        .actual_ordinal = actual_ordinal,
        .expected_requested_bytes = expected == NULL
            ? 0U : expected->requested_bytes,
        .actual_requested_bytes = requested_bytes,
        .expected_charged_bytes = expected == NULL
            ? 0U : expected->charged_bytes,
        .actual_charged_bytes = charged_bytes,
        .expected_alignment_bytes = expected == NULL
            ? 0U : expected->alignment_bytes,
        .actual_alignment_bytes = alignment_bytes,
        .expected_operation = expected == NULL
            ? UINT8_MAX : expected->operation,
        .actual_operation = actual_operation,
    };
    shadowspill_latch_task_allocation_contract_failure(runtime, &mismatch);
    return SHADOWSPILL_STATUS_TASK_ALLOCATION_CONTRACT_MISMATCH;
}

int shadowspill_claim_task_invocation(
    const ShadowSpillTaskRecord *record
) {
    if (record == NULL || record->plan_owner == NULL) {
        return -1;
    }
    ShadowSpillPlan *plan = record->plan_owner;
    if (atomic_load_explicit(&plan->closing, memory_order_acquire) != 0U ||
        atomic_load_explicit(&plan->closed, memory_order_acquire) != 0U) {
        return -1;
    }
    (void)atomic_fetch_add_explicit(
        &plan->active_task_scopes, 1U, memory_order_acq_rel
    );
    if (atomic_load_explicit(&plan->closing, memory_order_acquire) != 0U ||
        atomic_load_explicit(&plan->closed, memory_order_acquire) != 0U) {
        (void)atomic_fetch_sub_explicit(
            &plan->active_task_scopes, 1U, memory_order_release
        );
        return -1;
    }
    ShadowSpillTaskRecord *mutable_record = (ShadowSpillTaskRecord *)record;
    uint8_t expected_active = 0U;
    if (atomic_compare_exchange_strong_explicit(
        &mutable_record->invocation_active,
        &expected_active,
        1U,
        memory_order_acq_rel,
        memory_order_acquire
    )) {
        return 0;
    }
    (void)atomic_fetch_sub_explicit(
        &plan->active_task_scopes, 1U, memory_order_release
    );
    return -1;
}

void shadowspill_release_task_invocation(
    const ShadowSpillTaskRecord *record
) {
    if (record == NULL || record->plan_owner == NULL) {
        return;
    }
    if (atomic_exchange_explicit(
            &((ShadowSpillTaskRecord *)record)->invocation_active,
            0U,
            memory_order_acq_rel
        ) != 0U) {
        (void)atomic_fetch_sub_explicit(
            &record->plan_owner->active_task_scopes,
            1U,
            memory_order_release
        );
    }
}

int shadowspill_enter_claimed_task_scope(
    ShadowSpillRuntime *runtime,
    const ShadowSpillTaskRecord *record
) {
    if (record == NULL || record->plan_owner == NULL ||
        record->plan_owner->runtime != runtime ||
        atomic_load_explicit(
            &record->invocation_active, memory_order_acquire
        ) == 0U ||
        shadowspill_enter_allocation_scope(
            runtime, record->plan_owner->execution_pool, record->task_id
        ) != 0) {
        return -1;
    }
    ShadowSpillTaskRecord *mutable_record = (ShadowSpillTaskRecord *)record;
    task_scope.task = record;
    prepare_allocation_matcher(record);
    task_scope.invocation = atomic_fetch_add_explicit(
        &mutable_record->invocation_count,
        1U,
        memory_order_acq_rel
    ) + 1U;
    return 0;
}

int shadowspill_enter_task_scope(
    ShadowSpillRuntime *runtime,
    const ShadowSpillTaskRecord *record
) {
    if (shadowspill_claim_task_invocation(record) != 0) {
        return -1;
    }
    if (shadowspill_enter_claimed_task_scope(runtime, record) != 0) {
        shadowspill_release_task_invocation(record);
        return -1;
    }
    return 0;
}

static int allocation_geometry_matches(
    const ShadowSpillTaskAllocationContractStep *step,
    uint64_t requested_bytes,
    uint64_t charged_bytes,
    uint64_t alignment_bytes
) {
    return step != NULL &&
        step->operation == SHADOWSPILL_TASK_ALLOCATION_ALLOCATE &&
        step->requested_bytes == requested_bytes &&
        step->charged_bytes == charged_bytes &&
        step->alignment_bytes == alignment_bytes;
}

static void skip_omitted_free_operations(void) {
    for (;;) {
        const ShadowSpillTaskAllocationContractStep *step =
            expected_allocation_step();
        if (step == NULL ||
            step->operation != SHADOWSPILL_TASK_ALLOCATION_FREE ||
            step->allocation_ordinal >= task_scope.allocation_state_count ||
            task_scope.allocation_states[step->allocation_ordinal] !=
                SHADOWSPILL_TASK_ALLOCATION_OMITTED) {
            return;
        }
        ++task_scope.operation_index;
    }
}

static void classify_task_allocation(
    uint64_t requested_bytes,
    uint64_t charged_bytes,
    uint64_t alignment_bytes
) {
    task_scope.pending_allocation = 1U;
    task_scope.pending_is_scratch = 1U;
    task_scope.pending_invariant_ordinal = SHADOWSPILL_RUNTIME_NO_ID;
    skip_omitted_free_operations();
    const uint64_t start = task_scope.operation_index;
    uint64_t scan = start;
    while (scan < task_scope.task->allocation_contract_step_count) {
        const ShadowSpillTaskAllocationContractStep *step =
            &task_scope.task->allocation_contract_steps[scan];
        if (step->operation == SHADOWSPILL_TASK_ALLOCATION_ALLOCATE) {
            if (allocation_geometry_matches(
                    step, requested_bytes, charged_bytes, alignment_bytes
                )) {
                for (uint64_t index = start; index < scan; ++index) {
                    const ShadowSpillTaskAllocationContractStep *skipped =
                        &task_scope.task->allocation_contract_steps[index];
                    if (skipped->operation ==
                            SHADOWSPILL_TASK_ALLOCATION_ALLOCATE &&
                        skipped->allocation_ordinal <
                            task_scope.allocation_state_count &&
                        task_scope.allocation_states[
                            skipped->allocation_ordinal
                        ] == SHADOWSPILL_TASK_ALLOCATION_OMIT_CANDIDATE) {
                        task_scope.allocation_states[
                            skipped->allocation_ordinal
                        ] = SHADOWSPILL_TASK_ALLOCATION_OMITTED;
                    }
                }
                task_scope.operation_index = scan;
                task_scope.pending_is_scratch = 0U;
                task_scope.pending_invariant_ordinal = step->allocation_ordinal;
                return;
            }
            if (step->required) {
                break;
            }
            if (step->allocation_ordinal < task_scope.allocation_state_count) {
                task_scope.allocation_states[step->allocation_ordinal] =
                    SHADOWSPILL_TASK_ALLOCATION_OMIT_CANDIDATE;
            }
            ++scan;
            continue;
        }
        if (step->allocation_ordinal >= task_scope.allocation_state_count ||
            (task_scope.allocation_states[step->allocation_ordinal] !=
                 SHADOWSPILL_TASK_ALLOCATION_OMITTED &&
             task_scope.allocation_states[step->allocation_ordinal] !=
                 SHADOWSPILL_TASK_ALLOCATION_OMIT_CANDIDATE)) {
            break;
        }
        ++scan;
    }
    for (uint64_t index = start; index < scan; ++index) {
        const ShadowSpillTaskAllocationContractStep *candidate =
            &task_scope.task->allocation_contract_steps[index];
        if (candidate->operation == SHADOWSPILL_TASK_ALLOCATION_ALLOCATE &&
            candidate->allocation_ordinal < task_scope.allocation_state_count &&
            task_scope.allocation_states[candidate->allocation_ordinal] ==
                SHADOWSPILL_TASK_ALLOCATION_OMIT_CANDIDATE) {
            task_scope.allocation_states[candidate->allocation_ordinal] =
                SHADOWSPILL_TASK_ALLOCATION_UNSEEN;
        }
    }
}

ShadowSpillStatus shadowspill_validate_task_allocation(
    ShadowSpillRuntime *runtime,
    uint64_t requested_bytes,
    uint64_t charged_bytes,
    uint64_t alignment_bytes
) {
    if (runtime == NULL || requested_bytes == 0U || charged_bytes == 0U ||
        alignment_bytes == 0U) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    if (task_scope.runtime != runtime || task_scope.task == NULL) {
        return SHADOWSPILL_STATUS_OK;
    }
    const ShadowSpillTaskRecord *record = task_scope.task;
    const uint64_t projected_requested =
        task_scope.live_requested_bytes + requested_bytes;
    const uint64_t projected_charged =
        task_scope.live_charged_bytes + charged_bytes;
    const int request_exceeded =
        (record->maximum_requested_allocation_bytes != 0U &&
         requested_bytes > record->maximum_requested_allocation_bytes) ||
        (record->maximum_charged_allocation_bytes != 0U &&
         charged_bytes > record->maximum_charged_allocation_bytes);
    const int live_exceeded =
        (record->live_requested_allocation_limit_bytes != 0U &&
         projected_requested > record->live_requested_allocation_limit_bytes) ||
        (record->live_charged_allocation_limit_bytes != 0U &&
         projected_charged > record->live_charged_allocation_limit_bytes);
    if (request_exceeded || live_exceeded) {
        shadowspill_latch_task_envelope_failure(
            runtime,
            requested_bytes,
            charged_bytes,
            projected_requested,
            projected_charged,
            record->live_requested_allocation_limit_bytes,
            record->live_charged_allocation_limit_bytes,
            record->maximum_requested_allocation_bytes,
            record->maximum_charged_allocation_bytes
        );
        return SHADOWSPILL_STATUS_TASK_ALLOCATION_ENVELOPE_EXCEEDED;
    }
    if (!record->enforce_allocation_contract) {
        return SHADOWSPILL_STATUS_OK;
    }
    if (task_scope.pending_allocation) {
        return SHADOWSPILL_STATUS_INVALID_STATE;
    }
    classify_task_allocation(requested_bytes, charged_bytes, alignment_bytes);
    if (task_scope.pending_is_scratch &&
        (record->dynamic_scratch_live_limit_bytes == 0U ||
         (record->dynamic_scratch_maximum_allocation_bytes != 0U &&
          charged_bytes >
              record->dynamic_scratch_maximum_allocation_bytes) ||
         task_scope.scratch_live_charged_bytes + charged_bytes >
             record->dynamic_scratch_live_limit_bytes)) {
        task_scope.pending_allocation = 0U;
        return latch_allocation_contract_mismatch(
            runtime,
            SHADOWSPILL_TASK_ALLOCATION_ALLOCATE,
            task_scope.allocation_sequence,
            requested_bytes,
            charged_bytes,
            alignment_bytes
        );
    }
    return SHADOWSPILL_STATUS_OK;
}

uint64_t shadowspill_commit_task_allocation(
    ShadowSpillRuntime *runtime,
    uint64_t requested_bytes,
    uint64_t charged_bytes
) {
    if (task_scope.runtime != runtime || task_scope.task == NULL) {
        return SHADOWSPILL_RUNTIME_NO_ID;
    }
    const uint64_t allocation_sequence = task_scope.allocation_sequence++;
    if (task_scope.pending_allocation && task_scope.pending_is_scratch) {
        task_scope.scratch_live_requested_bytes += requested_bytes;
        task_scope.scratch_live_charged_bytes += charged_bytes;
        if (task_scope.scratch_live_charged_bytes >
            task_scope.scratch_peak_charged_bytes) {
            task_scope.scratch_peak_charged_bytes =
                task_scope.scratch_live_charged_bytes;
        }
        ++task_scope.scratch_allocation_count;
    } else if (task_scope.pending_allocation) {
        if (task_scope.pending_invariant_ordinal <
            task_scope.allocation_state_count) {
            task_scope.allocation_states[task_scope.pending_invariant_ordinal] =
                SHADOWSPILL_TASK_ALLOCATION_MATCHED;
        }
        ++task_scope.operation_index;
    }
    task_scope.live_requested_bytes += requested_bytes;
    task_scope.live_charged_bytes += charged_bytes;
    if (task_scope.live_requested_bytes > task_scope.peak_requested_bytes) {
        task_scope.peak_requested_bytes = task_scope.live_requested_bytes;
    }
    if (task_scope.live_charged_bytes > task_scope.peak_charged_bytes) {
        task_scope.peak_charged_bytes = task_scope.live_charged_bytes;
    }
    ++task_scope.allocation_count;
    task_scope.pending_allocation = 0U;
    task_scope.pending_is_scratch = 0U;
    task_scope.pending_invariant_ordinal = SHADOWSPILL_RUNTIME_NO_ID;
    return allocation_sequence;
}

ShadowSpillStatus shadowspill_release_task_allocation(
    ShadowSpillRuntime *runtime,
    uint64_t origin_task_id,
    uint64_t origin_task_invocation,
    uint64_t allocation_ordinal,
    int allocation_is_scratch,
    uint64_t requested_bytes,
    uint64_t charged_bytes,
    uint64_t alignment_bytes
) {
    if (task_scope.runtime != runtime || task_scope.task == NULL ||
        task_scope.task_id != origin_task_id ||
        task_scope.invocation != origin_task_invocation) {
        return SHADOWSPILL_STATUS_OK;
    }
    if (task_scope.task->enforce_allocation_contract &&
        !allocation_is_scratch) {
        skip_omitted_free_operations();
        const ShadowSpillTaskAllocationContractStep *expected =
            expected_allocation_step();
        if (expected == NULL ||
            expected->operation != SHADOWSPILL_TASK_ALLOCATION_FREE ||
            expected->allocation_ordinal != allocation_ordinal ||
            expected->requested_bytes != requested_bytes ||
            expected->charged_bytes != charged_bytes ||
            expected->alignment_bytes != alignment_bytes) {
            return latch_allocation_contract_mismatch(
                runtime,
                SHADOWSPILL_TASK_ALLOCATION_FREE,
                allocation_ordinal,
                requested_bytes,
                charged_bytes,
                alignment_bytes
            );
        }
        ++task_scope.operation_index;
    }
    if (allocation_is_scratch) {
        if (requested_bytes <= task_scope.scratch_live_requested_bytes) {
            task_scope.scratch_live_requested_bytes -= requested_bytes;
        } else {
            task_scope.scratch_live_requested_bytes = 0U;
        }
        if (charged_bytes <= task_scope.scratch_live_charged_bytes) {
            task_scope.scratch_live_charged_bytes -= charged_bytes;
        } else {
            task_scope.scratch_live_charged_bytes = 0U;
        }
    }
    if (requested_bytes <= task_scope.live_requested_bytes) {
        task_scope.live_requested_bytes -= requested_bytes;
    } else {
        task_scope.live_requested_bytes = 0U;
    }
    if (charged_bytes <= task_scope.live_charged_bytes) {
        task_scope.live_charged_bytes -= charged_bytes;
    } else {
        task_scope.live_charged_bytes = 0U;
    }
    ++task_scope.free_count;
    return SHADOWSPILL_STATUS_OK;
}

ShadowSpillStatus shadowspill_validate_task_allocation_complete(
    ShadowSpillRuntime *runtime
) {
    if (task_scope.runtime != runtime || task_scope.task == NULL ||
        !task_scope.task->enforce_allocation_contract) {
        return SHADOWSPILL_STATUS_OK;
    }
    for (;;) {
        skip_omitted_free_operations();
        const ShadowSpillTaskAllocationContractStep *expected =
            expected_allocation_step();
        if (expected == NULL) {
            return SHADOWSPILL_STATUS_OK;
        }
        if (expected->operation != SHADOWSPILL_TASK_ALLOCATION_ALLOCATE ||
            expected->required) {
            break;
        }
        if (expected->allocation_ordinal < task_scope.allocation_state_count) {
            task_scope.allocation_states[expected->allocation_ordinal] =
                SHADOWSPILL_TASK_ALLOCATION_OMITTED;
        }
        ++task_scope.operation_index;
    }
    if (task_scope.operation_index ==
        task_scope.task->allocation_contract_step_count) {
        return SHADOWSPILL_STATUS_OK;
    }
    return latch_allocation_contract_mismatch(
        runtime,
        UINT8_MAX,
        SHADOWSPILL_RUNTIME_NO_ID,
        0U,
        0U,
        0U
    );
}

void shadowspill_leave_task_scope(ShadowSpillRuntime *runtime) {
    if (task_scope.runtime == runtime) {
        ShadowSpillTaskRecord *record =
            (ShadowSpillTaskRecord *)task_scope.task;
        ShadowSpillMemoryLease *retirement = task_scope.retirement_head;
        while (retirement != NULL) {
            ShadowSpillMemoryLease *next =
                retirement->task_retirement_next;
            ShadowSpillMemoryPool *owner = retirement->metadata_owner;
            if (owner != NULL) {
                pthread_mutex_lock(&owner->lock);
            }
            retirement->task_retirement_next = NULL;
            retirement->task_retirement_linked = 0U;
            shadowspill_memory_pool_try_recycle_lease_record_locked(retirement);
            if (owner != NULL) {
                pthread_mutex_unlock(&owner->lock);
            }
            retirement = next;
        }
        task_scope.runtime = NULL;
        task_scope.allocation_pool = NULL;
        task_scope.task_id = SHADOWSPILL_RUNTIME_NO_ID;
        task_scope.task = NULL;
        task_scope.invocation = 0U;
        task_scope.operation_index = 0U;
        task_scope.allocation_sequence = 0U;
        task_scope.pending_invariant_ordinal = SHADOWSPILL_RUNTIME_NO_ID;
        task_scope.pending_is_scratch = 0U;
        task_scope.pending_allocation = 0U;
        task_scope.allocation_states = NULL;
        task_scope.allocation_state_count = 0U;
        task_scope.live_requested_bytes = 0U;
        task_scope.live_charged_bytes = 0U;
        task_scope.peak_requested_bytes = 0U;
        task_scope.peak_charged_bytes = 0U;
        task_scope.allocation_count = 0U;
        task_scope.free_count = 0U;
        task_scope.scratch_live_requested_bytes = 0U;
        task_scope.scratch_live_charged_bytes = 0U;
        task_scope.scratch_peak_charged_bytes = 0U;
        task_scope.scratch_allocation_count = 0U;
        task_scope.retirement_head = NULL;
        task_scope.retirement_tail = NULL;
        shadowspill_release_task_invocation(record);
    }
}


void shadowspill_abort_current_task(ShadowSpillRuntime *runtime) {
    if (runtime == NULL) {
        return;
    }
    shadowspill_finalize_aborted_task_retirements(
        runtime, shadowspill_current_task_id(runtime)
    );
    shadowspill_task_clear_pending_handoffs(task_scope.task);
    shadowspill_leave_task_scope(runtime);
}

ShadowSpillStatus shadowspill_abort_task_handle(
    ShadowSpillRuntime *runtime,
    const ShadowSpillTaskHandle *handle
) {
    const ShadowSpillTaskRecord *record = handle;
    if (runtime == NULL || record == NULL || record->plan_owner == NULL ||
        record->plan_owner->runtime != runtime ||
        record->boundary_kind != SHADOWSPILL_BOUNDARY_TASK) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    if (shadowspill_current_plan(runtime) != record->plan_owner ||
        shadowspill_current_task_id(runtime) != record->task_id) {
        return SHADOWSPILL_STATUS_INVALID_STATE;
    }
    shadowspill_abort_current_task(runtime);
    return SHADOWSPILL_STATUS_OK;
}
