#include "internal.h"

#include <shadowspill/admission_replay.h>

#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct ReplayState {
    ShadowSpillMemoryPool pool;
    ShadowSpillMemoryLease *leases;
    ShadowSpillEventLease *events;
    uint64_t *expected_dependency_ids;
    ShadowSpillMemoryLease **release_frontier;
    ShadowSpillMemoryLease **blocking_order;
    uint8_t *retirement_completed_early;
    uint64_t peak_reserved_bytes;
    uint64_t peak_fragmentation_bytes;
    uint64_t digest;
} ReplayState;

struct ShadowSpillAdmissionReplayWorkspace {
    ReplayState state;
    ShadowSpillRange *range_nodes;
    ShadowSpillRange *release_range_nodes;
    uint64_t range_node_capacity;
    uint64_t release_range_node_capacity;
    uint64_t lease_capacity;
    uint64_t dependency_capacity;
    uint8_t synchronization_initialized;
};

static uint64_t digest_u64(uint64_t digest, uint64_t value) {
    for (uint32_t byte = 0U; byte < 8U; ++byte) {
        digest ^= value & 0xffU;
        digest *= UINT64_C(1099511628211);
        value >>= 8U;
    }
    return digest;
}

static ShadowSpillMemoryPlacement replay_placement(
    const ShadowSpillAdmissionReplayProgram *program,
    uint64_t bytes
) {
    return program->large_request_threshold_bytes != 0U &&
            bytes >= program->large_request_threshold_bytes
        ? SHADOWSPILL_MEMORY_BEST_FIT_HIGH
        : SHADOWSPILL_MEMORY_BEST_FIT_LOW;
}

static void initialize_result(ShadowSpillAdmissionReplayResult *result) {
    ShadowSpillAdmissionReplayDecision *decisions = result->decisions;
    const uint64_t decision_capacity = result->decision_capacity;
    ShadowSpillAdmissionReuseDependency *dependencies = result->dependencies;
    const uint64_t dependency_capacity = result->dependency_capacity;
    ShadowSpillAdmissionReplayLiveLease *live_leases = result->live_leases;
    const uint64_t live_lease_capacity = result->live_lease_capacity;
    *result = (ShadowSpillAdmissionReplayResult){
        .status = SHADOWSPILL_ADMISSION_REPLAY_INVALID_ARGUMENT,
        .error_operation_index = SHADOWSPILL_ADMISSION_REPLAY_NO_ID,
        .error_lease_id = SHADOWSPILL_ADMISSION_REPLAY_NO_ID,
        .decisions = decisions,
        .decision_capacity = decision_capacity,
        .dependencies = dependencies,
        .dependency_capacity = dependency_capacity,
        .live_leases = live_leases,
        .live_lease_capacity = live_lease_capacity,
    };
}

static int reset_state(
    const ShadowSpillAdmissionReplayProgram *program,
    ShadowSpillAdmissionReplayWorkspace *workspace
) {
    ReplayState *state = &workspace->state;
    if (program->lease_count > workspace->lease_capacity ||
        program->dependency_count > workspace->dependency_capacity) {
        return -1;
    }
    memset(
        state->leases,
        0,
        (size_t)workspace->lease_capacity * sizeof(*state->leases)
    );
    memset(
        state->events,
        0,
        (size_t)workspace->dependency_capacity * sizeof(*state->events)
    );
    memset(
        state->expected_dependency_ids,
        0xff,
        (size_t)workspace->lease_capacity *
            sizeof(*state->expected_dependency_ids)
    );
    memset(
        state->retirement_completed_early,
        0,
        (size_t)workspace->lease_capacity *
            sizeof(*state->retirement_completed_early)
    );
    if (shadowspill_range_initialize_with_nodes(
            &state->pool.ranges,
            program->capacity_bytes,
            workspace->range_nodes,
            workspace->range_node_capacity
        ) != 0) {
        return -1;
    }
    state->pool.range_leases = NULL;
    state->pool.backend = (ShadowSpillMemoryPoolBackend){0};
    state->pool.base = NULL;
    state->pool.pool_id = 0U;
    state->pool.minimum_alignment = program->minimum_alignment;
    state->pool.next_request_sequence = 1U;
    state->pool.next_release_sequence = 1U;
    state->pool.reserved_bytes = 0U;
    state->pool.release_frontier_workspace = state->release_frontier;
    state->pool.release_frontier_capacity = workspace->lease_capacity;
    state->pool.release_range_workspace = workspace->release_range_nodes;
    state->pool.release_range_capacity =
        workspace->release_range_node_capacity;
    state->pool.initialized = 1U;
    atomic_store_explicit(
        &state->pool.foreground_waiters, 0U, memory_order_relaxed
    );
    atomic_store_explicit(
        &state->pool.reservation_waiters, 0U, memory_order_relaxed
    );
    for (uint64_t index = 0U; index < program->lease_count; ++index) {
        state->leases[index].generation = index + 1U;
        atomic_init(&state->leases[index].references, 1U);
        state->expected_dependency_ids[index] =
            SHADOWSPILL_ADMISSION_REPLAY_NO_ID;
    }
    for (uint64_t index = 0U; index < program->dependency_count; ++index) {
        state->events[index].generation = index + 1U;
        atomic_init(&state->events[index].references, 1U);
        atomic_init(&state->events[index].backend_complete, 0U);
    }
    state->peak_reserved_bytes = 0U;
    state->peak_fragmentation_bytes = 0U;
    state->digest = UINT64_C(1469598103934665603);
    return 0;
}

static int valid_program(
    const ShadowSpillAdmissionReplayProgram *program,
    const ShadowSpillAdmissionReplayResult *result
) {
    return program != NULL && result != NULL &&
        program->abi_version == SHADOWSPILL_ABI_VERSION &&
        program->minimum_alignment != 0U &&
        (program->operation_count == 0U || program->operations != NULL) &&
        result->decision_capacity >= program->operation_count &&
        (program->operation_count == 0U || result->decisions != NULL) &&
        (result->dependency_capacity == 0U || result->dependencies != NULL) &&
        (result->live_lease_capacity == 0U || result->live_leases != NULL);
}

static ShadowSpillEventLease *dependency_event(
    const ShadowSpillAdmissionReplayProgram *program,
    ReplayState *state,
    uint64_t dependency_id
) {
    return dependency_id == SHADOWSPILL_ADMISSION_REPLAY_NO_ID
        ? NULL
        : dependency_id < program->dependency_count
            ? &state->events[dependency_id]
            : NULL;
}

static uint64_t event_id(
    const ShadowSpillAdmissionReplayProgram *program,
    const ReplayState *state,
    const ShadowSpillEventLease *event
) {
    if (event == NULL || state->events == NULL ||
        event < state->events || event >= state->events + program->dependency_count) {
        return SHADOWSPILL_ADMISSION_REPLAY_NO_ID;
    }
    return (uint64_t)(event - state->events);
}

static uint64_t lease_id(
    const ShadowSpillAdmissionReplayProgram *program,
    const ReplayState *state,
    const ShadowSpillMemoryLease *lease
) {
    if (lease == NULL || state->leases == NULL || lease < state->leases ||
        lease >= state->leases + program->lease_count) {
        return SHADOWSPILL_ADMISSION_REPLAY_NO_ID;
    }
    return (uint64_t)(lease - state->leases);
}

static int lease_offset_compare(const void *left_value, const void *right_value) {
    const ShadowSpillMemoryLease *left =
        *(ShadowSpillMemoryLease *const *)left_value;
    const ShadowSpillMemoryLease *right =
        *(ShadowSpillMemoryLease *const *)right_value;
    if (left->offset != right->offset) {
        return left->offset < right->offset ? -1 : 1;
    }
    if (left->charged_bytes != right->charged_bytes) {
        return left->charged_bytes < right->charged_bytes ? -1 : 1;
    }
    return left < right ? -1 : left != right;
}

static int collect_failure_live_leases(
    const ShadowSpillAdmissionReplayProgram *program,
    ReplayState *state,
    ShadowSpillAdmissionReplayResult *result
) {
    uint64_t lease_count = 0U;
    for (ShadowSpillMemoryLease *lease = state->pool.range_leases;
         lease != NULL; lease = lease->pool_next) {
        state->blocking_order[lease_count++] = lease;
    }
    qsort(
        state->blocking_order,
        (size_t)lease_count,
        sizeof(*state->blocking_order),
        lease_offset_compare
    );
    result->live_lease_count = lease_count;
    if (result->live_leases == NULL || result->live_lease_capacity == 0U) {
        return 0;
    }
    if (lease_count > result->live_lease_capacity) {
        return -1;
    }
    for (uint64_t index = 0U; index < lease_count; ++index) {
        const ShadowSpillMemoryLease *lease = state->blocking_order[index];
        result->live_leases[index] = (ShadowSpillAdmissionReplayLiveLease){
            .lease_id = lease_id(program, state, lease),
            .offset = lease->offset,
            .requested_bytes = lease->requested_bytes,
            .charged_bytes = lease->charged_bytes,
            .state = (uint8_t)lease->state,
        };
    }
    return 0;
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
    ShadowSpillAdmissionReplayResult *result,
    uint64_t predecessor,
    uint64_t successor,
    uint64_t dependency,
    uint64_t operation_index
) {
    if (result->dependency_result_count >= result->dependency_capacity) {
        return -1;
    }
    result->dependencies[result->dependency_result_count++] =
        (ShadowSpillAdmissionReuseDependency){
            .predecessor_lease_id = predecessor,
            .successor_lease_id = successor,
            .dependency_id = dependency,
            .consumer_operation_index = operation_index,
        };
    return 0;
}

static ShadowSpillEventLease *publish_retirement_dependency(
    const ShadowSpillAdmissionReplayProgram *program,
    ReplayState *state,
    uint64_t predecessor,
    ShadowSpillMemoryLease *lease
) {
    if (lease->causal_event != NULL) {
        return lease->causal_event;
    }
    if (lease->causal_dependency_expected == 0U ||
        predecessor >= program->lease_count) {
        return NULL;
    }
    const uint64_t expected = state->expected_dependency_ids[predecessor];
    ShadowSpillEventLease *event = dependency_event(
        program, state, expected
    );
    if (event == NULL ||
        shadowspill_memory_pool_publish_retirement_dependency_locked(
            lease, event
        ) != 0) {
        return NULL;
    }
    state->expected_dependency_ids[predecessor] =
        SHADOWSPILL_ADMISSION_REPLAY_NO_ID;
    return event;
}

static int reserve_after_release_frontier(
    const ShadowSpillAdmissionReplayProgram *program,
    ReplayState *state,
    ShadowSpillAdmissionReplayResult *result,
    uint64_t operation_index,
    ShadowSpillMemoryLease *successor,
    uint64_t bytes,
    uint64_t alignment
) {
    uint64_t frontier_count = 0U;
    const int found = shadowspill_memory_pool_find_release_frontier_locked(
        &state->pool,
        bytes,
        alignment,
        state->release_frontier,
        program->lease_count,
        &frontier_count
    );
    if (found <= 0) {
        return found < 0 ? -1 : 1;
    }
    for (uint64_t index = 0U; index < frontier_count; ++index) {
        ShadowSpillMemoryLease *predecessor_lease =
            state->release_frontier[index];
        const uint64_t predecessor = lease_id(
            program, state, predecessor_lease
        );
        ShadowSpillEventLease *event = publish_retirement_dependency(
            program, state, predecessor, predecessor_lease
        );
        const uint64_t dependency = event_id(program, state, event);
        if (predecessor == SHADOWSPILL_ADMISSION_REPLAY_NO_ID ||
            dependency == SHADOWSPILL_ADMISSION_REPLAY_NO_ID ||
            append_dependency(
                result,
                predecessor,
                lease_id(program, state, successor),
                dependency,
                operation_index
            ) != 0) {
            return -1;
        }
        atomic_store_explicit(
            &event->backend_complete, 1U, memory_order_release
        );
        if (shadowspill_memory_pool_release_lease_locked(
                predecessor_lease
            ) != 0) {
            return -1;
        }
        state->retirement_completed_early[predecessor] = 1U;
    }
    return shadowspill_memory_pool_reserve_lease_locked(
        &state->pool,
        successor,
        bytes,
        alignment,
        replay_placement(program, bytes)
    );
}

static ShadowSpillAdmissionReplayStatus apply_operation(
    const ShadowSpillAdmissionReplayProgram *program,
    ReplayState *state,
    ShadowSpillAdmissionReplayResult *result,
    uint64_t operation_index
) {
    const ShadowSpillAdmissionReplayOperation *operation =
        &program->operations[operation_index];
    if (operation->lease_id >= program->lease_count ||
        (operation_index != 0U && operation->sequence <
            program->operations[operation_index - 1U].sequence)) {
        return SHADOWSPILL_ADMISSION_REPLAY_INVALID_OPERATIONS;
    }
    ShadowSpillMemoryLease *lease = &state->leases[operation->lease_id];
    const uint64_t allocated_before = state->pool.ranges.allocated;
    const uint64_t prior_offset = lease->offset;
    const uint64_t prior_requested_bytes = lease->requested_bytes;
    const uint64_t prior_charged_bytes = lease->charged_bytes;
    uint64_t predecessor = SHADOWSPILL_ADMISSION_REPLAY_NO_ID;
    uint64_t dependency = SHADOWSPILL_ADMISSION_REPLAY_NO_ID;
    int status = -1;
    switch ((ShadowSpillAdmissionReplayOperationKind)operation->kind) {
        case SHADOWSPILL_ADMISSION_REPLAY_ACQUIRE:
            if (operation->bytes == 0U || operation->alignment == 0U) {
                return SHADOWSPILL_ADMISSION_REPLAY_INVALID_OPERATIONS;
            }
            status = shadowspill_memory_pool_reserve_lease_locked(
                &state->pool,
                lease,
                operation->bytes,
                operation->alignment,
                replay_placement(program, operation->bytes)
            );
            if (status == 1) {
                status = reserve_after_release_frontier(
                    program,
                    state,
                    result,
                    operation_index,
                    lease,
                    operation->bytes,
                    operation->alignment
                );
            }
            break;
        case SHADOWSPILL_ADMISSION_REPLAY_BEGIN_RETIREMENT: {
            ShadowSpillEventLease *event = dependency_event(
                program, state, operation->dependency_id
            );
            if (event == NULL) {
                return SHADOWSPILL_ADMISSION_REPLAY_INVALID_OPERATIONS;
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
        case SHADOWSPILL_ADMISSION_REPLAY_PUBLISH_DEPENDENCY: {
            ShadowSpillEventLease *event = dependency_event(
                program, state, operation->dependency_id
            );
            if (event == NULL) {
                return SHADOWSPILL_ADMISSION_REPLAY_INVALID_OPERATIONS;
            }
            status = shadowspill_memory_pool_publish_retirement_dependency_locked(
                lease, event
            );
            if (status == 0) {
                state->expected_dependency_ids[operation->lease_id] =
                    SHADOWSPILL_ADMISSION_REPLAY_NO_ID;
            }
            break;
        }
        case SHADOWSPILL_ADMISSION_REPLAY_RESERVE:
            if (operation->bytes == 0U || operation->alignment == 0U) {
                return SHADOWSPILL_ADMISSION_REPLAY_INVALID_OPERATIONS;
            }
            status = shadowspill_memory_pool_reserve_lease_locked(
                &state->pool,
                lease,
                operation->bytes,
                operation->alignment,
                replay_placement(program, operation->bytes)
            );
            if (status == 1) {
                status = shadowspill_memory_pool_reserve_causal_successor_locked(
                    &state->pool,
                    lease,
                    operation->bytes,
                    operation->alignment
                );
            }
            if (status == 1) {
                status = reserve_after_release_frontier(
                    program,
                    state,
                    result,
                    operation_index,
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
        case SHADOWSPILL_ADMISSION_REPLAY_ACQUIRE_RESERVED: {
            ShadowSpillMemoryLease *predecessor_lease =
                lease->causal_predecessor;
            predecessor = lease_id(program, state, predecessor_lease);
            if (predecessor != SHADOWSPILL_ADMISSION_REPLAY_NO_ID &&
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
                    return SHADOWSPILL_ADMISSION_REPLAY_INVALID_OPERATIONS;
                }
                state->expected_dependency_ids[predecessor] =
                    SHADOWSPILL_ADMISSION_REPLAY_NO_ID;
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
                    return SHADOWSPILL_ADMISSION_REPLAY_ALLOCATION_FAILURE;
                }
                (void)atomic_fetch_sub_explicit(
                    &event->references, 1U, memory_order_release
                );
            }
            break;
        }
        case SHADOWSPILL_ADMISSION_REPLAY_COMPLETE_RETIREMENT: {
            ShadowSpillEventLease *event = dependency_event(
                program, state, operation->dependency_id
            );
            if (event == NULL) {
                return SHADOWSPILL_ADMISSION_REPLAY_INVALID_OPERATIONS;
            }
            atomic_store_explicit(
                &event->backend_complete, 1U, memory_order_release
            );
            if (state->retirement_completed_early[operation->lease_id] != 0U) {
                state->retirement_completed_early[operation->lease_id] = 0U;
                status = 0;
            } else {
                status = shadowspill_memory_pool_release_lease_locked(lease);
            }
            break;
        }
        case SHADOWSPILL_ADMISSION_REPLAY_RELEASE:
            status = shadowspill_memory_pool_release_lease_locked(lease);
            break;
        default:
            return SHADOWSPILL_ADMISSION_REPLAY_INVALID_OPERATIONS;
    }
    if (status == 1) {
        return SHADOWSPILL_ADMISSION_REPLAY_INFEASIBLE;
    }
    if (status != 0) {
        return SHADOWSPILL_ADMISSION_REPLAY_INVALID_OPERATIONS;
    }
    const uint64_t allocated_after = state->pool.ranges.allocated;
    const int64_t delta = allocated_after >= allocated_before
        ? (int64_t)(allocated_after - allocated_before)
        : -(int64_t)(allocated_before - allocated_after);
    const int lease_still_described = lease->pool != NULL ||
        lease->state == SHADOWSPILL_LEASE_PREDECESSOR_TRANSFERRED;
    ShadowSpillAdmissionReplayDecision *decision =
        &result->decisions[result->decision_count++];
    *decision = (ShadowSpillAdmissionReplayDecision){
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
    return SHADOWSPILL_ADMISSION_REPLAY_OK;
}

ShadowSpillAdmissionReplayStatus shadowspill_admission_replay_workspace_create(
    uint64_t lease_capacity,
    uint64_t dependency_capacity,
    ShadowSpillAdmissionReplayWorkspace **workspace
) {
    if (workspace == NULL || lease_capacity > SIZE_MAX ||
        dependency_capacity > SIZE_MAX ||
        lease_capacity > (UINT64_MAX - 2U) / 2U) {
        return SHADOWSPILL_ADMISSION_REPLAY_INVALID_ARGUMENT;
    }
    *workspace = NULL;
    ShadowSpillAdmissionReplayWorkspace *created = calloc(1U, sizeof(*created));
    if (created == NULL) {
        return SHADOWSPILL_ADMISSION_REPLAY_ALLOCATION_FAILURE;
    }
    created->lease_capacity = lease_capacity;
    created->dependency_capacity = dependency_capacity;
    created->range_node_capacity = lease_capacity + 1U;
    created->release_range_node_capacity = 2U * lease_capacity + 2U;
    created->state.leases = calloc(
        lease_capacity == 0U ? 1U : (size_t)lease_capacity,
        sizeof(*created->state.leases)
    );
    created->state.events = calloc(
        dependency_capacity == 0U ? 1U : (size_t)dependency_capacity,
        sizeof(*created->state.events)
    );
    created->state.expected_dependency_ids = malloc(
        (lease_capacity == 0U ? 1U : (size_t)lease_capacity) *
        sizeof(*created->state.expected_dependency_ids)
    );
    created->state.release_frontier = calloc(
        lease_capacity == 0U ? 1U : (size_t)lease_capacity,
        sizeof(*created->state.release_frontier)
    );
    created->state.blocking_order = calloc(
        lease_capacity == 0U ? 1U : (size_t)lease_capacity,
        sizeof(*created->state.blocking_order)
    );
    created->state.retirement_completed_early = calloc(
        lease_capacity == 0U ? 1U : (size_t)lease_capacity,
        sizeof(*created->state.retirement_completed_early)
    );
    created->range_nodes = calloc(
        (size_t)created->range_node_capacity,
        sizeof(*created->range_nodes)
    );
    created->release_range_nodes = calloc(
        (size_t)created->release_range_node_capacity,
        sizeof(*created->release_range_nodes)
    );
    if (created->state.leases == NULL || created->state.events == NULL ||
        created->state.expected_dependency_ids == NULL ||
        created->state.release_frontier == NULL ||
        created->state.blocking_order == NULL ||
        created->state.retirement_completed_early == NULL ||
        created->range_nodes == NULL || created->release_range_nodes == NULL ||
        pthread_mutex_init(&created->state.pool.lock, NULL) != 0) {
        shadowspill_admission_replay_workspace_destroy(created);
        return SHADOWSPILL_ADMISSION_REPLAY_ALLOCATION_FAILURE;
    }
    created->synchronization_initialized = 1U;
    atomic_init(&created->state.pool.foreground_waiters, 0U);
    atomic_init(&created->state.pool.reservation_waiters, 0U);
    atomic_init(&created->state.pool.capacity_epoch, 0U);
    *workspace = created;
    return SHADOWSPILL_ADMISSION_REPLAY_OK;
}

void shadowspill_admission_replay_workspace_destroy(
    ShadowSpillAdmissionReplayWorkspace *workspace
) {
    if (workspace == NULL) {
        return;
    }
    if (workspace->synchronization_initialized != 0U) {
        pthread_mutex_destroy(&workspace->state.pool.lock);
    }
    free(workspace->release_range_nodes);
    free(workspace->range_nodes);
    free(workspace->state.events);
    free(workspace->state.leases);
    free(workspace->state.expected_dependency_ids);
    free(workspace->state.release_frontier);
    free(workspace->state.blocking_order);
    free(workspace->state.retirement_completed_early);
    free(workspace);
}

ShadowSpillAdmissionReplayStatus shadowspill_admission_replay_run_reusing(
    const ShadowSpillAdmissionReplayProgram *program,
    ShadowSpillAdmissionReplayResult *result,
    ShadowSpillAdmissionReplayWorkspace *workspace
) {
    if (result == NULL) {
        return SHADOWSPILL_ADMISSION_REPLAY_INVALID_ARGUMENT;
    }
    initialize_result(result);
    if (!valid_program(program, result) || workspace == NULL ||
        program->lease_count > workspace->lease_capacity ||
        program->dependency_count > workspace->dependency_capacity) {
        return SHADOWSPILL_ADMISSION_REPLAY_INVALID_ARGUMENT;
    }
    if (reset_state(program, workspace) != 0) {
        result->status = SHADOWSPILL_ADMISSION_REPLAY_ALLOCATION_FAILURE;
        return SHADOWSPILL_ADMISSION_REPLAY_ALLOCATION_FAILURE;
    }
    ReplayState *state = &workspace->state;
    ShadowSpillAdmissionReplayStatus status = SHADOWSPILL_ADMISSION_REPLAY_OK;
    for (uint64_t index = 0U; index < program->operation_count; ++index) {
        status = apply_operation(program, state, result, index);
        if (status != SHADOWSPILL_ADMISSION_REPLAY_OK) {
            const ShadowSpillAdmissionReplayOperation *operation =
                &program->operations[index];
            result->error_operation_index = index;
            result->error_lease_id = operation->lease_id;
            result->error_requested_bytes = operation->bytes;
            result->error_free_bytes = shadowspill_memory_pool_free_bytes_locked(
                &state->pool
            );
            result->error_largest_free_range_bytes =
                shadowspill_memory_pool_largest_free_locked(&state->pool);
            if (status == SHADOWSPILL_ADMISSION_REPLAY_INFEASIBLE &&
                collect_failure_live_leases(program, state, result) != 0) {
                status = SHADOWSPILL_ADMISSION_REPLAY_ALLOCATION_FAILURE;
            }
            break;
        }
    }
    result->status = (uint32_t)status;
    result->peak_allocated_bytes = state->pool.ranges.peak_allocated;
    result->peak_reserved_bytes = state->peak_reserved_bytes;
    result->peak_fragmentation_bytes = state->peak_fragmentation_bytes;
    result->final_allocated_bytes = state->pool.ranges.allocated;
    result->final_reserved_bytes = state->pool.reserved_bytes;
    result->final_largest_free_range_bytes =
        shadowspill_memory_pool_largest_free_locked(&state->pool);
    result->decision_digest = state->digest;
    return status;
}

ShadowSpillAdmissionReplayStatus shadowspill_admission_replay_run(
    const ShadowSpillAdmissionReplayProgram *program,
    ShadowSpillAdmissionReplayResult *result
) {
    if (result == NULL) {
        return SHADOWSPILL_ADMISSION_REPLAY_INVALID_ARGUMENT;
    }
    initialize_result(result);
    if (!valid_program(program, result)) {
        return SHADOWSPILL_ADMISSION_REPLAY_INVALID_ARGUMENT;
    }
    ShadowSpillAdmissionReplayWorkspace *workspace = NULL;
    ShadowSpillAdmissionReplayStatus status =
        shadowspill_admission_replay_workspace_create(
            program->lease_count,
            program->dependency_count,
            &workspace
        );
    if (status == SHADOWSPILL_ADMISSION_REPLAY_OK) {
        status = shadowspill_admission_replay_run_reusing(
            program, result, workspace
        );
    } else {
        result->status = (uint32_t)status;
    }
    shadowspill_admission_replay_workspace_destroy(workspace);
    return status;
}
