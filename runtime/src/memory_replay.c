#include "internal.h"

#include <shadowspill/memory_replay.h>

#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct ReplayState {
    ShadowSpillMemoryPool pool;
    ShadowSpillMemoryLease *leases;
    ShadowSpillEventLease *events;
    uint64_t *expected_dependency_ids;
    uint64_t peak_reserved_bytes;
    uint64_t peak_fragmentation_bytes;
    uint64_t digest;
} ReplayState;

static uint64_t digest_u64(uint64_t digest, uint64_t value) {
    for (uint32_t byte = 0U; byte < 8U; ++byte) {
        digest ^= value & 0xffU;
        digest *= UINT64_C(1099511628211);
        value >>= 8U;
    }
    return digest;
}

static void initialize_result(ShadowSpillMemoryReplayResult *result) {
    ShadowSpillMemoryReplayDecision *decisions = result->decisions;
    const uint64_t decision_capacity = result->decision_capacity;
    ShadowSpillMemoryReuseDependency *dependencies = result->dependencies;
    const uint64_t dependency_capacity = result->dependency_capacity;
    *result = (ShadowSpillMemoryReplayResult){
        .status = SHADOWSPILL_MEMORY_REPLAY_INVALID_ARGUMENT,
        .error_operation_index = SHADOWSPILL_MEMORY_REPLAY_NO_ID,
        .error_lease_id = SHADOWSPILL_MEMORY_REPLAY_NO_ID,
        .decisions = decisions,
        .decision_capacity = decision_capacity,
        .dependencies = dependencies,
        .dependency_capacity = dependency_capacity,
    };
}

static int initialize_state(
    const ShadowSpillMemoryReplayProgram *program,
    ReplayState *state
) {
    state->leases = program->lease_count == 0U
        ? NULL
        : calloc((size_t)program->lease_count, sizeof(*state->leases));
    state->events = program->dependency_count == 0U
        ? NULL
        : calloc((size_t)program->dependency_count, sizeof(*state->events));
    state->expected_dependency_ids = program->lease_count == 0U
        ? NULL
        : malloc((size_t)program->lease_count * sizeof(*state->expected_dependency_ids));
    if ((program->lease_count != 0U && state->leases == NULL) ||
        (program->lease_count != 0U && state->expected_dependency_ids == NULL) ||
        (program->dependency_count != 0U && state->events == NULL) ||
        pthread_mutex_init(&state->pool.lock, NULL) != 0) {
        free(state->expected_dependency_ids);
        free(state->events);
        free(state->leases);
        return -1;
    }
    if (pthread_cond_init(&state->pool.capacity_changed, NULL) != 0) {
        pthread_mutex_destroy(&state->pool.lock);
        free(state->expected_dependency_ids);
        free(state->events);
        free(state->leases);
        return -1;
    }
    if (shadowspill_range_initialize(
            &state->pool.ranges, program->capacity_bytes
        ) != 0) {
        pthread_cond_destroy(&state->pool.capacity_changed);
        pthread_mutex_destroy(&state->pool.lock);
        free(state->expected_dependency_ids);
        free(state->events);
        free(state->leases);
        return -1;
    }
    state->pool.minimum_alignment = program->minimum_alignment;
    state->pool.next_request_sequence = 1U;
    state->pool.next_release_sequence = 1U;
    state->pool.initialized = 1U;
    atomic_init(&state->pool.foreground_waiters, 0U);
    atomic_init(&state->pool.reservation_waiters, 0U);
    for (uint64_t index = 0U; index < program->lease_count; ++index) {
        state->leases[index].generation = index + 1U;
        atomic_init(&state->leases[index].references, 1U);
        state->expected_dependency_ids[index] =
            SHADOWSPILL_MEMORY_REPLAY_NO_ID;
    }
    for (uint64_t index = 0U; index < program->dependency_count; ++index) {
        state->events[index].generation = index + 1U;
        atomic_init(&state->events[index].references, 1U);
        atomic_init(&state->events[index].backend_complete, 0U);
    }
    state->digest = UINT64_C(1469598103934665603);
    return 0;
}

static void destroy_state(ReplayState *state) {
    shadowspill_range_destroy(&state->pool.ranges);
    pthread_cond_destroy(&state->pool.capacity_changed);
    pthread_mutex_destroy(&state->pool.lock);
    free(state->events);
    free(state->leases);
    free(state->expected_dependency_ids);
}

static int valid_program(
    const ShadowSpillMemoryReplayProgram *program,
    const ShadowSpillMemoryReplayResult *result
) {
    return program != NULL && result != NULL &&
        program->abi_version == SHADOWSPILL_MEMORY_REPLAY_ABI_VERSION &&
        program->minimum_alignment != 0U &&
        (program->operation_count == 0U || program->operations != NULL) &&
        result->decision_capacity >= program->operation_count &&
        (program->operation_count == 0U || result->decisions != NULL) &&
        (result->dependency_capacity == 0U || result->dependencies != NULL);
}

static ShadowSpillEventLease *dependency_event(
    const ShadowSpillMemoryReplayProgram *program,
    ReplayState *state,
    uint64_t dependency_id
) {
    return dependency_id == SHADOWSPILL_MEMORY_REPLAY_NO_ID
        ? NULL
        : dependency_id < program->dependency_count
            ? &state->events[dependency_id]
            : NULL;
}

static uint64_t event_id(
    const ShadowSpillMemoryReplayProgram *program,
    const ReplayState *state,
    const ShadowSpillEventLease *event
) {
    if (event == NULL || state->events == NULL ||
        event < state->events || event >= state->events + program->dependency_count) {
        return SHADOWSPILL_MEMORY_REPLAY_NO_ID;
    }
    return (uint64_t)(event - state->events);
}

static uint64_t lease_id(
    const ShadowSpillMemoryReplayProgram *program,
    const ReplayState *state,
    const ShadowSpillMemoryLease *lease
) {
    if (lease == NULL || state->leases == NULL || lease < state->leases ||
        lease >= state->leases + program->lease_count) {
        return SHADOWSPILL_MEMORY_REPLAY_NO_ID;
    }
    return (uint64_t)(lease - state->leases);
}

static void update_peaks(ReplayState *state) {
    const uint64_t free_bytes = shadowspill_memory_pool_free_bytes_locked(
        &state->pool
    );
    const uint64_t largest = shadowspill_memory_pool_largest_free_locked(
        &state->pool
    );
    const uint64_t fragmentation = free_bytes - largest;
    if (state->pool.reserved_bytes > state->peak_reserved_bytes) {
        state->peak_reserved_bytes = state->pool.reserved_bytes;
    }
    if (fragmentation > state->peak_fragmentation_bytes) {
        state->peak_fragmentation_bytes = fragmentation;
    }
}

static int append_dependency(
    ShadowSpillMemoryReplayResult *result,
    uint64_t predecessor,
    uint64_t successor,
    uint64_t dependency,
    uint64_t operation_index
) {
    if (result->dependency_result_count >= result->dependency_capacity) {
        return -1;
    }
    result->dependencies[result->dependency_result_count++] =
        (ShadowSpillMemoryReuseDependency){
            .predecessor_lease_id = predecessor,
            .successor_lease_id = successor,
            .dependency_id = dependency,
            .consumer_operation_index = operation_index,
        };
    return 0;
}

static ShadowSpillMemoryReplayStatus apply_operation(
    const ShadowSpillMemoryReplayProgram *program,
    ReplayState *state,
    ShadowSpillMemoryReplayResult *result,
    uint64_t operation_index
) {
    const ShadowSpillMemoryReplayOperation *operation =
        &program->operations[operation_index];
    if (operation->lease_id >= program->lease_count ||
        (operation_index != 0U && operation->sequence <
            program->operations[operation_index - 1U].sequence)) {
        return SHADOWSPILL_MEMORY_REPLAY_INVALID_SCRIPT;
    }
    ShadowSpillMemoryLease *lease = &state->leases[operation->lease_id];
    const uint64_t allocated_before = state->pool.ranges.allocated;
    const uint64_t prior_offset = lease->offset;
    const uint64_t prior_requested_bytes = lease->requested_bytes;
    const uint64_t prior_charged_bytes = lease->charged_bytes;
    uint64_t predecessor = SHADOWSPILL_MEMORY_REPLAY_NO_ID;
    uint64_t dependency = SHADOWSPILL_MEMORY_REPLAY_NO_ID;
    int status = -1;
    switch ((ShadowSpillMemoryReplayOperationKind)operation->kind) {
        case SHADOWSPILL_MEMORY_REPLAY_ACQUIRE:
            if (operation->bytes == 0U || operation->alignment == 0U) {
                return SHADOWSPILL_MEMORY_REPLAY_INVALID_SCRIPT;
            }
            status = shadowspill_memory_pool_reserve_lease_locked(
                &state->pool,
                lease,
                operation->bytes,
                operation->alignment,
                SHADOWSPILL_MEMORY_BEST_FIT_LOW
            );
            break;
        case SHADOWSPILL_MEMORY_REPLAY_BEGIN_RETIREMENT: {
            ShadowSpillEventLease *event = dependency_event(
                program, state, operation->dependency_id
            );
            if (event == NULL) {
                return SHADOWSPILL_MEMORY_REPLAY_INVALID_SCRIPT;
            }
            if (operation->dependency_expected != 0U) {
                state->expected_dependency_ids[operation->lease_id] =
                    operation->dependency_id;
                event = NULL;
            }
            status = shadowspill_memory_pool_begin_retirement_locked(
                lease, event, operation->dependency_expected != 0U
            );
            break;
        }
        case SHADOWSPILL_MEMORY_REPLAY_PUBLISH_DEPENDENCY: {
            ShadowSpillEventLease *event = dependency_event(
                program, state, operation->dependency_id
            );
            if (event == NULL) {
                return SHADOWSPILL_MEMORY_REPLAY_INVALID_SCRIPT;
            }
            status = shadowspill_memory_pool_publish_retirement_dependency_locked(
                lease, event
            );
            if (status == 0) {
                state->expected_dependency_ids[operation->lease_id] =
                    SHADOWSPILL_MEMORY_REPLAY_NO_ID;
            }
            break;
        }
        case SHADOWSPILL_MEMORY_REPLAY_RESERVE:
            if (operation->bytes == 0U || operation->alignment == 0U) {
                return SHADOWSPILL_MEMORY_REPLAY_INVALID_SCRIPT;
            }
            status = shadowspill_memory_pool_reserve_lease_locked(
                &state->pool,
                lease,
                operation->bytes,
                operation->alignment,
                SHADOWSPILL_MEMORY_BEST_FIT_LOW
            );
            if (status == 1) {
                status = shadowspill_memory_pool_reserve_causal_successor_locked(
                    &state->pool,
                    lease,
                    operation->bytes,
                    operation->alignment
                );
            }
            if (status == 0 &&
                lease->state != SHADOWSPILL_LEASE_SUCCESSOR_RESERVED) {
                status = shadowspill_memory_pool_mark_reserved_locked(lease);
            }
            predecessor = lease_id(
                program, state, lease->causal_predecessor
            );
            break;
        case SHADOWSPILL_MEMORY_REPLAY_ACQUIRE_RESERVED: {
            ShadowSpillMemoryLease *predecessor_lease =
                lease->causal_predecessor;
            predecessor = lease_id(program, state, predecessor_lease);
            if (predecessor != SHADOWSPILL_MEMORY_REPLAY_NO_ID &&
                predecessor_lease->causal_event == NULL &&
                predecessor_lease->causal_dependency_expected != 0U) {
                const uint64_t expected =
                    state->expected_dependency_ids[predecessor];
                ShadowSpillEventLease *event = dependency_event(
                    program, state, expected
                );
                if (event == NULL ||
                    shadowspill_memory_pool_publish_retirement_dependency_locked(
                        predecessor_lease, event
                    ) != 0) {
                    return SHADOWSPILL_MEMORY_REPLAY_INVALID_SCRIPT;
                }
                state->expected_dependency_ids[predecessor] =
                    SHADOWSPILL_MEMORY_REPLAY_NO_ID;
            }
            ShadowSpillEventLease *event = NULL;
            status = shadowspill_memory_pool_acquire_reserved_lease_locked(
                lease, &event
            );
            dependency = event_id(program, state, event);
            if (status == 0 && event != NULL) {
                if (append_dependency(
                        result,
                        predecessor,
                        operation->lease_id,
                        dependency,
                        operation_index
                    ) != 0) {
                    (void)atomic_fetch_sub_explicit(
                        &event->references, 1U, memory_order_release
                    );
                    return SHADOWSPILL_MEMORY_REPLAY_ALLOCATION_FAILURE;
                }
                (void)atomic_fetch_sub_explicit(
                    &event->references, 1U, memory_order_release
                );
            }
            break;
        }
        case SHADOWSPILL_MEMORY_REPLAY_COMPLETE_RETIREMENT: {
            ShadowSpillEventLease *event = dependency_event(
                program, state, operation->dependency_id
            );
            if (event == NULL) {
                return SHADOWSPILL_MEMORY_REPLAY_INVALID_SCRIPT;
            }
            atomic_store_explicit(
                &event->backend_complete, 1U, memory_order_release
            );
            status = shadowspill_memory_pool_release_lease_locked(lease);
            break;
        }
        case SHADOWSPILL_MEMORY_REPLAY_RELEASE:
            status = shadowspill_memory_pool_release_lease_locked(lease);
            break;
        default:
            return SHADOWSPILL_MEMORY_REPLAY_INVALID_SCRIPT;
    }
    if (status == 1) {
        return SHADOWSPILL_MEMORY_REPLAY_INFEASIBLE;
    }
    if (status != 0) {
        return SHADOWSPILL_MEMORY_REPLAY_INVALID_SCRIPT;
    }
    const uint64_t allocated_after = state->pool.ranges.allocated;
    const int64_t delta = allocated_after >= allocated_before
        ? (int64_t)(allocated_after - allocated_before)
        : -(int64_t)(allocated_before - allocated_after);
    const int lease_still_described = lease->pool != NULL ||
        lease->state == SHADOWSPILL_LEASE_PREDECESSOR_TRANSFERRED;
    ShadowSpillMemoryReplayDecision *decision =
        &result->decisions[result->decision_count++];
    *decision = (ShadowSpillMemoryReplayDecision){
        .operation_index = operation_index,
        .sequence = operation->sequence,
        .lease_id = operation->lease_id,
        .predecessor_lease_id = predecessor,
        .dependency_id = dependency,
        .offset = lease_still_described ? lease->offset : prior_offset,
        .requested_bytes = lease_still_described
            ? lease->requested_bytes : prior_requested_bytes,
        .charged_bytes = lease_still_described
            ? lease->charged_bytes : prior_charged_bytes,
        .physical_bytes_delta = delta,
        .resulting_state = (uint8_t)lease->state,
    };
    state->digest = digest_u64(state->digest, operation->sequence);
    state->digest = digest_u64(state->digest, operation->lease_id);
    state->digest = digest_u64(state->digest, operation->kind);
    state->digest = digest_u64(state->digest, predecessor);
    state->digest = digest_u64(state->digest, dependency);
    state->digest = digest_u64(state->digest, decision->offset);
    state->digest = digest_u64(state->digest, decision->charged_bytes);
    state->digest = digest_u64(state->digest, (uint64_t)delta);
    state->digest = digest_u64(state->digest, decision->resulting_state);
    update_peaks(state);
    return SHADOWSPILL_MEMORY_REPLAY_OK;
}

uint32_t shadowspill_memory_replay_abi_version(void) {
    return SHADOWSPILL_MEMORY_REPLAY_ABI_VERSION;
}

ShadowSpillMemoryReplayStatus shadowspill_memory_replay_run(
    const ShadowSpillMemoryReplayProgram *program,
    ShadowSpillMemoryReplayResult *result
) {
    if (result == NULL) {
        return SHADOWSPILL_MEMORY_REPLAY_INVALID_ARGUMENT;
    }
    initialize_result(result);
    if (!valid_program(program, result)) {
        return SHADOWSPILL_MEMORY_REPLAY_INVALID_ARGUMENT;
    }
    ReplayState state = {0};
    if (initialize_state(program, &state) != 0) {
        result->status = SHADOWSPILL_MEMORY_REPLAY_ALLOCATION_FAILURE;
        return SHADOWSPILL_MEMORY_REPLAY_ALLOCATION_FAILURE;
    }
    ShadowSpillMemoryReplayStatus status = SHADOWSPILL_MEMORY_REPLAY_OK;
    for (uint64_t index = 0U; index < program->operation_count; ++index) {
        status = apply_operation(program, &state, result, index);
        if (status != SHADOWSPILL_MEMORY_REPLAY_OK) {
            const ShadowSpillMemoryReplayOperation *operation =
                &program->operations[index];
            result->error_operation_index = index;
            result->error_lease_id = operation->lease_id;
            result->error_requested_bytes = operation->bytes;
            result->error_free_bytes = shadowspill_memory_pool_free_bytes_locked(
                &state.pool
            );
            result->error_largest_free_range_bytes =
                shadowspill_memory_pool_largest_free_locked(&state.pool);
            break;
        }
    }
    result->status = (uint32_t)status;
    result->peak_allocated_bytes = state.pool.ranges.peak_allocated;
    result->peak_reserved_bytes = state.peak_reserved_bytes;
    result->peak_fragmentation_bytes = state.peak_fragmentation_bytes;
    result->final_allocated_bytes = state.pool.ranges.allocated;
    result->final_reserved_bytes = state.pool.reserved_bytes;
    result->final_largest_free_range_bytes =
        shadowspill_memory_pool_largest_free_locked(&state.pool);
    result->decision_digest = state.digest;
    destroy_state(&state);
    return status;
}

const char *shadowspill_memory_replay_status_string(
    ShadowSpillMemoryReplayStatus status
) {
    switch (status) {
        case SHADOWSPILL_MEMORY_REPLAY_OK:
            return "ok";
        case SHADOWSPILL_MEMORY_REPLAY_INVALID_ARGUMENT:
            return "invalid argument";
        case SHADOWSPILL_MEMORY_REPLAY_ALLOCATION_FAILURE:
            return "allocation failure";
        case SHADOWSPILL_MEMORY_REPLAY_INFEASIBLE:
            return "infeasible";
        case SHADOWSPILL_MEMORY_REPLAY_INVALID_SCRIPT:
            return "invalid replay script";
    }
    return "unknown memory replay status";
}
