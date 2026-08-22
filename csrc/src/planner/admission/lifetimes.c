/*
 * Turning one schedule's operations into the lifetimes a layout places.
 *
 * The operation walk already says which lease each operation creates and
 * retires, and why; the simulator says when each task and transfer ran.
 * Joining them is all this does, and the join is one pass over leases rather
 * than over operations: a lease's lifetime is decided by exactly two
 * operations, and most operations decide nothing.
 *
 * The rules it follows - where an operation sits, why a lease exists, and the
 * transitions that emit no operation at all - are specified in
 * docs/architecture/admission-leases.md.
 */

#include <stdlib.h>
#include <string.h>

#include "internal.h"

#define TASK_ALLOCATION_ALLOCATE 0U
#define NO_INDEX SHADOWSPILL_PLANNER_NO_INDEX
#define NO_LEASE SHADOWSPILL_ADMISSION_NO_LEASE
#define NO_OPERATION SHADOWSPILL_ADMISSION_NO_OPERATION

/* --------------------------------------------------------------- lookups */

/*
 * The three index tables the join needs, none of which the caller supplies.
 *
 * `task_interval` and `transfer_of_action` turn a task or action index into
 * its simulated interval. `latest_step` and `lease_of_slot` are what make a
 * reallocated slot readable: a task may reallocate an earlier ordinal's
 * slot, which gives the same lease a new identity and emits no operation, so
 * a lease's identity comes from the last step that allocated its slot rather
 * than from the operation that created it.
 */
typedef struct {
    const ShadowSpillTaskInterval **task_interval;
    const ShadowSpillTransferInterval **transfer_of_action;
    uint32_t *latest_step;
    uint64_t *lease_of_slot;
    uint32_t *step_of_lease;
} LeaseIndex;

static void lease_index_destroy(LeaseIndex *index)
{
    free(index->task_interval);
    free(index->transfer_of_action);
    free(index->latest_step);
    free(index->lease_of_slot);
    free(index->step_of_lease);
    memset(index, 0, sizeof(*index));
}

/* Tasks and transfers report their own index, so both invert by scatter. */
static int index_intervals(
    const ShadowSpillLeaseLifetimeProblem *problem, LeaseIndex *index
)
{
    const uint32_t tasks = problem->admission->task_count;
    const uint32_t actions = problem->schedule->action_count;
    index->task_interval = calloc(tasks ? tasks : 1U, sizeof(*index->task_interval));
    index->transfer_of_action =
        calloc(actions ? actions : 1U, sizeof(*index->transfer_of_action));
    if (index->task_interval == NULL || index->transfer_of_action == NULL) {
        return -1;
    }
    for (uint32_t item = 0U; item < problem->task_interval_count; ++item) {
        const ShadowSpillTaskInterval *interval = &problem->task_intervals[item];
        if (interval->task < tasks) {
            index->task_interval[interval->task] = interval;
        }
    }

    /* A transfer names its sequence within its direction, not its action. Both
     * directions number densely from zero, so invert by scatter and then walk
     * the schedule once to recover the pairing. */
    const uint32_t transfers = problem->transfer_interval_count;
    const ShadowSpillTransferInterval **by_sequence =
        calloc(transfers ? transfers * 2U : 2U, sizeof(*by_sequence));
    if (by_sequence == NULL) {
        return -1;
    }
    for (uint32_t item = 0U; item < transfers; ++item) {
        const ShadowSpillTransferInterval *interval =
            &problem->transfer_intervals[item];
        if (interval->direction < 2U && interval->sequence < transfers) {
            by_sequence[interval->direction * transfers + interval->sequence] =
                interval;
        }
    }

    int status = 0;
    uint32_t sequence[2] = {0U, 0U};
    for (uint32_t action = 0U; action < actions; ++action) {
        const uint8_t kind = problem->schedule->action_kinds[action];
        if (kind == SHADOWSPILL_MEMORY_RELEASE) {
            continue;
        }
        const uint8_t direction = kind == SHADOWSPILL_MEMORY_OFFLOAD
            ? SHADOWSPILL_TRANSFER_EVICT
            : SHADOWSPILL_TRANSFER_FETCH;
        const uint32_t wanted = sequence[direction]++;
        if (wanted >= transfers) {
            status = -1;
            break;
        }
        index->transfer_of_action[action] =
            by_sequence[direction * transfers + wanted];
        if (index->transfer_of_action[action] == NULL) {
            status = -1;
            break;
        }
    }
    free(by_sequence);
    return status;
}

/* Which allocation step owns each lease, and which lease each slot holds. */
static int index_allocation_steps(
    const ShadowSpillLeaseLifetimeProblem *problem, LeaseIndex *index
)
{
    const ShadowSpillAdmissionOperations *operations = problem->operations;
    const ShadowSpillAdmissionTopology *topology = problem->admission;
    const uint32_t slots = topology->allocation_slot_count;
    const uint64_t leases = operations->lease_count;
    index->latest_step = malloc((slots ? slots : 1U) * sizeof(*index->latest_step));
    index->lease_of_slot =
        malloc((slots ? slots : 1U) * sizeof(*index->lease_of_slot));
    index->step_of_lease =
        malloc((leases ? leases : 1U) * sizeof(*index->step_of_lease));
    if (index->latest_step == NULL || index->lease_of_slot == NULL ||
        index->step_of_lease == NULL) {
        return -1;
    }
    for (uint32_t slot = 0U; slot < slots; ++slot) {
        index->latest_step[slot] = NO_INDEX;
        index->lease_of_slot[slot] = NO_LEASE;
    }

    const uint32_t steps = topology->task_allocation_offsets[topology->task_count];
    for (uint32_t step = 0U; step < steps; ++step) {
        if (topology->task_allocation_kinds[step] == TASK_ALLOCATION_ALLOCATE) {
            index->latest_step[topology->task_allocation_slots[step]] = step;
        }
    }

    for (uint64_t lease = 0U; lease < leases; ++lease) {
        const uint64_t start = operations->lease_starts[lease];
        const uint32_t step = operations->allocation_offsets[start];
        index->step_of_lease[lease] = step;
        if (step != NO_INDEX) {
            index->lease_of_slot[topology->task_allocation_slots[step]] = lease;
        }
    }
    return 0;
}

/* -------------------------------------------------------------- identity */

/* The task an allocation step belongs to. Steps are flattened per task, so
 * the owning task is the range containing the step. */
static uint32_t task_of_step(
    const ShadowSpillAdmissionTopology *topology, uint32_t step
)
{
    uint32_t low = 0U;
    uint32_t high = topology->task_count;
    while (low + 1U < high) {
        const uint32_t middle = low + (high - low) / 2U;
        if (topology->task_allocation_offsets[middle] <= step) {
            low = middle;
        } else {
            high = middle;
        }
    }
    return low;
}

/* Anonymous scratch, a fresh output, or the generation one supersedes. */
static uint8_t allocation_purpose(
    const ShadowSpillAdmissionTopology *topology, uint32_t task, uint32_t alias
)
{
    if (alias == NO_INDEX) {
        return SHADOWSPILL_ADMISSION_PURPOSE_TASK_WORKSPACE;
    }
    for (uint32_t item = topology->replacement_offsets[task];
         item < topology->replacement_offsets[task + 1U]; ++item) {
        if (topology->replacement_aliases[item] == alias) {
            return SHADOWSPILL_ADMISSION_PURPOSE_MUTATION_REPLACEMENT;
        }
    }
    return SHADOWSPILL_ADMISSION_PURPOSE_TASK_OUTPUT;
}

/*
 * Why a lease exists and what it belongs to.
 *
 * A task allocation takes its identity from the last step that allocated its
 * slot, because a reallocation emits no operation of its own. Everything else
 * takes it from the operation that created the lease, read through the
 * boundary that operation sits on: an action boundary names an action and its
 * triggering task, and initial residency names neither.
 */
static void lease_identity(
    const ShadowSpillLeaseLifetimeProblem *problem,
    const LeaseIndex *index,
    uint64_t lease,
    ShadowSpillLeaseIdentity *identity
)
{
    const ShadowSpillAdmissionOperations *operations = problem->operations;
    const ShadowSpillAdmissionTopology *topology = problem->admission;
    const uint64_t start = operations->lease_starts[lease];
    const uint32_t own_step = index->step_of_lease[lease];

    identity->lease_id = lease;
    identity->causal_start = start;
    identity->task = NO_INDEX;
    identity->alias = NO_INDEX;
    identity->action = NO_INDEX;

    if (own_step != NO_INDEX) {
        const uint32_t slot = topology->task_allocation_slots[own_step];
        const uint32_t step = index->latest_step[slot];
        const uint32_t task = task_of_step(topology, step);
        identity->task = task;
        identity->alias = topology->task_allocation_aliases[step];
        identity->purpose = allocation_purpose(topology, task, identity->alias);
        return;
    }

    identity->purpose = operations->purposes[start];
    const uint32_t at = operations->indices[start];
    switch (operations->boundaries[start]) {
    case SHADOWSPILL_ADMISSION_BOUNDARY_ACTION_TRIGGER:
    case SHADOWSPILL_ADMISSION_BOUNDARY_ACTION_COMPLETION:
        identity->task = problem->schedule->action_trigger_tasks[at];
        identity->action = at;
        break;
    case SHADOWSPILL_ADMISSION_BOUNDARY_INITIAL:
        /* Initial residency names neither a task nor an action, and the
         * operation's index carries no meaning. */
        break;
    default:
        identity->task = at;
        break;
    }
    if (lease < operations->lease_capacity) {
        identity->alias = operations->lease_aliases[lease];
    }
}

/* ---------------------------------------------------------------- timing */

/* Initial residency is live from zero; a fetch destination from the moment
 * its triggering task ends; everything else from when its task starts. */
static int predicted_start(
    const LeaseIndex *index,
    const ShadowSpillLeaseIdentity *identity,
    uint64_t *when
)
{
    if (identity->purpose == SHADOWSPILL_ADMISSION_PURPOSE_INITIAL_OBJECT) {
        *when = 0U;
        return 0;
    }
    if (identity->task == NO_INDEX) {
        return -1;
    }
    const ShadowSpillTaskInterval *task = index->task_interval[identity->task];
    if (task == NULL) {
        return -1;
    }
    *when = identity->purpose == SHADOWSPILL_ADMISSION_PURPOSE_FETCH_DESTINATION
        ? task->end_ns
        : task->start_ns;
    return 0;
}

/* An eviction frees its address once the copy lands; anything else frees it
 * when the task it is retired at ends. */
static int predicted_end(
    const ShadowSpillLeaseLifetimeProblem *problem,
    const LeaseIndex *index,
    uint64_t retire,
    uint64_t *when
)
{
    const ShadowSpillAdmissionOperations *operations = problem->operations;
    const uint8_t purpose = operations->purposes[retire];
    const uint8_t boundary = operations->boundaries[retire];
    const uint32_t at = operations->indices[retire];
    const int at_action =
        boundary == SHADOWSPILL_ADMISSION_BOUNDARY_ACTION_TRIGGER ||
        boundary == SHADOWSPILL_ADMISSION_BOUNDARY_ACTION_COMPLETION;

    if (purpose == SHADOWSPILL_ADMISSION_PURPOSE_EVICTION) {
        if (!at_action || index->transfer_of_action[at] == NULL) {
            return -1;
        }
        *when = index->transfer_of_action[at]->end_ns;
        return 0;
    }
    uint32_t task;
    if (at_action) {
        task = problem->schedule->action_trigger_tasks[at];
    } else if (boundary == SHADOWSPILL_ADMISSION_BOUNDARY_INITIAL) {
        return -1;
    } else {
        task = at;
    }
    if (task >= problem->admission->task_count ||
        index->task_interval[task] == NULL) {
        return -1;
    }
    *when = index->task_interval[task]->end_ns;
    return 0;
}

/* ------------------------------------------------------------- partition */

/*
 * Move the caller-owned dynamic leases to the end, preserving lease order in
 * both parts. There are only ever a handful, so the fixed entries shift left
 * over them in one pass and the dynamic ones are appended from scratch.
 */
static int partition_dynamic(
    ShadowSpillLeaseLifetimeResult *result, const uint8_t *dynamic, uint64_t count
)
{
    uint64_t dynamic_count = 0U;
    for (uint64_t item = 0U; item < count; ++item) {
        dynamic_count += dynamic[item] != 0U;
    }
    result->fixed_count = count - dynamic_count;
    if (dynamic_count == 0U) {
        return 0;
    }

    ShadowSpillLeaseLifetime *held_lifetimes =
        malloc(dynamic_count * sizeof(*held_lifetimes));
    ShadowSpillLeaseIdentity *held_identities =
        malloc(dynamic_count * sizeof(*held_identities));
    if (held_lifetimes == NULL || held_identities == NULL) {
        free(held_lifetimes);
        free(held_identities);
        return -1;
    }

    uint64_t kept = 0U;
    uint64_t held = 0U;
    for (uint64_t item = 0U; item < count; ++item) {
        if (dynamic[item] != 0U) {
            held_lifetimes[held] = result->lifetimes[item];
            held_identities[held] = result->identities[item];
            ++held;
        } else {
            result->lifetimes[kept] = result->lifetimes[item];
            result->identities[kept] = result->identities[item];
            ++kept;
        }
    }
    memcpy(&result->lifetimes[kept], held_lifetimes,
           dynamic_count * sizeof(*held_lifetimes));
    memcpy(&result->identities[kept], held_identities,
           dynamic_count * sizeof(*held_identities));
    free(held_lifetimes);
    free(held_identities);
    return 0;
}

/* ----------------------------------------------------------------- entry */

static ShadowSpillPlannerStatus build_lifetimes(
    const ShadowSpillLeaseLifetimeProblem *problem,
    ShadowSpillLeaseLifetimeResult *result,
    LeaseIndex *index,
    uint8_t *dynamic
)
{
    const ShadowSpillAdmissionOperations *operations = problem->operations;
    const ShadowSpillAdmissionTopology *topology = problem->admission;
    const uint64_t count = operations->lease_count;
    const uint64_t terminal_time = problem->makespan_ns + 1U;
    const uint64_t terminal_boundary = operations->operation_count + 1U;

    for (uint32_t alias = 0U; alias < topology->alias_count; ++alias) {
        result->alias_leases[alias] = NO_LEASE;
    }

    for (uint64_t lease = 0U; lease < count; ++lease) {
        ShadowSpillLeaseIdentity *identity = &result->identities[lease];
        lease_identity(problem, index, lease, identity);

        const uint64_t retire = operations->lease_retires[lease];
        uint64_t ends;
        if (retire == NO_OPERATION) {
            ends = terminal_time;
            identity->causal_end = terminal_boundary;
            /* A lease that outlives the step is the alias's final generation,
             * which is what a caller-owned handoff ends up naming. */
            if (identity->alias != NO_INDEX) {
                result->alias_leases[identity->alias] = lease;
            }
        } else {
            if (predicted_end(problem, index, retire, &ends) != 0) {
                return SHADOWSPILL_PLANNER_INVALID_ARGUMENT;
            }
            identity->causal_end = retire;
        }

        uint64_t begins;
        if (predicted_start(index, identity, &begins) != 0 || ends < begins) {
            return SHADOWSPILL_PLANNER_INVALID_ARGUMENT;
        }
        const uint64_t start = identity->causal_start;
        result->lifetimes[lease] = (ShadowSpillLeaseLifetime){
            .bytes = operations->bytes[start],
            .alignment = operations->alignments[start],
            .start_ns = begins,
            .end_ns = ends,
        };
        dynamic[lease] = 0U;
    }

    /* A handoff moves a live lease to the destination alias without
     * allocating, so the destination owns the final generation. */
    const uint32_t handoffs = topology->handoff_offsets[topology->task_count];
    for (uint32_t item = 0U; item < handoffs; ++item) {
        const uint32_t source = topology->handoff_source_aliases[item];
        const uint32_t destination = topology->handoff_destination_aliases[item];
        if (result->alias_leases[source] == NO_LEASE) {
            continue;
        }
        result->alias_leases[destination] = result->alias_leases[source];
        result->alias_leases[source] = NO_LEASE;
    }

    for (uint32_t item = 0U; item < problem->dynamic_alias_count; ++item) {
        const uint32_t alias = problem->dynamic_aliases[item];
        if (alias >= topology->alias_count ||
            result->alias_leases[alias] == NO_LEASE) {
            return SHADOWSPILL_PLANNER_INVALID_ARGUMENT;
        }
        dynamic[result->alias_leases[alias]] = 1U;
    }

    const uint32_t steps = topology->task_allocation_offsets[topology->task_count];
    for (uint32_t step = 0U; step < steps; ++step) {
        result->allocation_step_leases[step] =
            topology->task_allocation_kinds[step] == TASK_ALLOCATION_ALLOCATE
            ? index->lease_of_slot[topology->task_allocation_slots[step]]
            : NO_LEASE;
    }
    result->lifetime_count = count;
    return SHADOWSPILL_PLANNER_OK;
}

ShadowSpillPlannerStatus shadowspill_build_lease_lifetimes(
    const ShadowSpillLeaseLifetimeProblem *problem,
    ShadowSpillLeaseLifetimeResult *result
)
{
    if (problem == NULL || result == NULL ||
        problem->abi_version != SHADOWSPILL_PLANNER_ABI_VERSION ||
        problem->operations == NULL || problem->admission == NULL ||
        problem->schedule == NULL || result->lifetimes == NULL ||
        result->identities == NULL || result->allocation_step_leases == NULL ||
        result->alias_leases == NULL) {
        return SHADOWSPILL_PLANNER_INVALID_ARGUMENT;
    }
    if (problem->dynamic_alias_count > 0U && problem->dynamic_aliases == NULL) {
        return SHADOWSPILL_PLANNER_INVALID_ARGUMENT;
    }
    result->lifetime_count = 0U;
    result->fixed_count = 0U;

    const uint64_t count = problem->operations->lease_count;
    LeaseIndex index;
    memset(&index, 0, sizeof(index));
    uint8_t *dynamic = malloc(count ? count : 1U);
    ShadowSpillPlannerStatus status = SHADOWSPILL_PLANNER_ALLOCATION_FAILURE;
    if (dynamic == NULL || index_intervals(problem, &index) != 0 ||
        index_allocation_steps(problem, &index) != 0) {
        goto done;
    }
    status = build_lifetimes(problem, result, &index, dynamic);
    if (status != SHADOWSPILL_PLANNER_OK) {
        goto done;
    }
    status = partition_dynamic(result, dynamic, count) == 0
        ? SHADOWSPILL_PLANNER_OK
        : SHADOWSPILL_PLANNER_ALLOCATION_FAILURE;

done:
    free(dynamic);
    lease_index_destroy(&index);
    return status;
}
