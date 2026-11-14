/* Reuse dependencies of a fixed layout.
 *
 * Two leases that share an address are sound only if the second cannot begin
 * before the first has released it. This finds every such pair and the
 * completion the second must wait for, which is what a certificate hands the
 * runtime. Leases are taken in causal order; a segment tree over the distinct
 * range endpoints names the latest lease to have owned each address, so the
 * leases it names across a successor's range are its predecessors, and the
 * successor then becomes the owner. A predecessor must retire before the
 * successor is acquired, in operation order, and its retirement must publish a
 * completion dependency; a layout where either fails is refused.
 */

#include <shadowspill/planner.h>

#include "internal.h"

#include <stdlib.h>
#include <string.h>

#define EMPTY (-1)
#define MIXED (-2)
#define NONE (-3)

/* ------------------------------------------------------------ owner tree */

/* Per node, the one lease owning every address in the node's interval, EMPTY
 * where none does, or MIXED. `lazy` is an assignment not yet pushed to the
 * children. Owners are indices into the lease arrays. */
typedef struct {
    uint64_t *coordinates;
    uint32_t coordinate_count;
    uint32_t length;
    int32_t *owner;
    int32_t *lazy;
} OwnerTree;

static int compare_coordinate(const void *left, const void *right)
{
    const uint64_t first = *(const uint64_t *)left;
    const uint64_t second = *(const uint64_t *)right;
    return (first > second) - (first < second);
}

static uint32_t coordinate_rank(const OwnerTree *tree, uint64_t value)
{
    uint32_t low = 0U;
    uint32_t high = tree->coordinate_count;
    while (low < high) {
        const uint32_t middle = low + (high - low) / 2U;
        if (tree->coordinates[middle] < value) {
            low = middle + 1U;
        } else {
            high = middle;
        }
    }
    return low;
}

static void owner_tree_destroy(OwnerTree *tree)
{
    free(tree->coordinates);
    free(tree->owner);
    free(tree->lazy);
    memset(tree, 0, sizeof(*tree));
}

static int owner_tree_create(
    OwnerTree *tree,
    const ShadowSpillLeaseLifetime *lifetimes,
    const uint64_t *offsets,
    uint64_t count
)
{
    memset(tree, 0, sizeof(*tree));
    tree->coordinates = malloc((size_t)(count * 2U + 1U) * sizeof(*tree->coordinates));
    if (tree->coordinates == NULL) {
        return -1;
    }
    for (uint64_t index = 0U; index < count; ++index) {
        tree->coordinates[index * 2U] = offsets[index];
        tree->coordinates[index * 2U + 1U] = offsets[index] + lifetimes[index].bytes;
    }
    qsort(tree->coordinates, (size_t)count * 2U, sizeof(*tree->coordinates), compare_coordinate);
    uint32_t distinct = 0U;
    for (uint64_t index = 0U; index < count * 2U; ++index) {
        if (distinct == 0U || tree->coordinates[distinct - 1U] != tree->coordinates[index]) {
            tree->coordinates[distinct++] = tree->coordinates[index];
        }
    }
    tree->coordinate_count = distinct;
    tree->length = distinct > 1U ? distinct - 1U : 0U;
    const size_t nodes = 4U * (size_t)(tree->length == 0U ? 1U : tree->length);
    tree->owner = malloc(nodes * sizeof(*tree->owner));
    tree->lazy = malloc(nodes * sizeof(*tree->lazy));
    if (tree->owner == NULL || tree->lazy == NULL) {
        return -1;
    }
    for (size_t node = 0U; node < nodes; ++node) {
        tree->owner[node] = EMPTY;
        tree->lazy[node] = NONE;
    }
    return 0;
}

static void push(OwnerTree *tree, uint32_t node)
{
    const int32_t value = tree->lazy[node];
    if (value == NONE) {
        return;
    }
    tree->owner[node * 2U] = value;
    tree->lazy[node * 2U] = value;
    tree->owner[node * 2U + 1U] = value;
    tree->lazy[node * 2U + 1U] = value;
    tree->lazy[node] = NONE;
}

/* The distinct owners found across [start, end), each once. */
typedef struct {
    int32_t *items;
    uint32_t count;
    uint32_t capacity;
    uint32_t *stamps;
    uint32_t serial;
} Found;

static int found_add(Found *found, int32_t owner)
{
    if (found->stamps[owner] == found->serial) {
        return 0;
    }
    found->stamps[owner] = found->serial;
    if (found->count == found->capacity) {
        const uint32_t capacity = found->capacity == 0U ? 16U : found->capacity * 2U;
        int32_t *items = realloc(found->items, (size_t)capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        found->items = items;
        found->capacity = capacity;
    }
    found->items[found->count++] = owner;
    return 0;
}

static int query(
    OwnerTree *tree,
    uint32_t node,
    uint32_t left,
    uint32_t right,
    uint32_t start,
    uint32_t end,
    Found *found
)
{
    if (end <= left || right <= start) {
        return 0;
    }
    if (start <= left && right <= end && tree->owner[node] != MIXED) {
        if (tree->owner[node] >= 0 && found_add(found, tree->owner[node]) != 0) {
            return -1;
        }
        return 0;
    }
    if (right - left == 1U) {
        return 0;
    }
    push(tree, node);
    const uint32_t middle = left + (right - left) / 2U;
    if (query(tree, node * 2U, left, middle, start, end, found) != 0 ||
        query(tree, node * 2U + 1U, middle, right, start, end, found) != 0) {
        return -1;
    }
    return 0;
}

static void assign(
    OwnerTree *tree,
    uint32_t node,
    uint32_t left,
    uint32_t right,
    uint32_t start,
    uint32_t end,
    int32_t owner
)
{
    if (end <= left || right <= start) {
        return;
    }
    if ((start <= left && right <= end) || right - left == 1U) {
        tree->owner[node] = owner;
        tree->lazy[node] = owner;
        return;
    }
    push(tree, node);
    const uint32_t middle = left + (right - left) / 2U;
    assign(tree, node * 2U, left, middle, start, end, owner);
    assign(tree, node * 2U + 1U, middle, right, start, end, owner);
    const int32_t first = tree->owner[node * 2U];
    const int32_t second = tree->owner[node * 2U + 1U];
    tree->owner[node] = first == second ? first : MIXED;
}

/* ------------------------------------------------------------- recovery */

/* One reuse: the leases by index, for decoding, and by id, for ordering. */
typedef struct {
    uint64_t successor_id;
    uint64_t dependency_id;
    uint64_t predecessor_id;
    uint32_t predecessor;
    uint32_t successor;
} Pair;

typedef struct {
    const ShadowSpillLayoutReuseProblem *problem;
    Pair *pairs;
    uint64_t count;
    uint64_t capacity;
} Pairs;

static int pairs_add(Pairs *pairs, Pair pair)
{
    if (pairs->count == pairs->capacity) {
        const uint64_t capacity = pairs->capacity == 0U ? 1024U : pairs->capacity * 2U;
        Pair *items = realloc(pairs->pairs, (size_t)capacity * sizeof(*items));
        if (items == NULL) {
            return -1;
        }
        pairs->pairs = items;
        pairs->capacity = capacity;
    }
    pairs->pairs[pairs->count++] = pair;
    return 0;
}

/* Successors in causal order: the acquiring operation first, the lease id
 * breaking ties. */
typedef struct {
    uint64_t causal_start;
    uint64_t lease_id;
    uint32_t index;
} Visit;

static int compare_visit(const void *left, const void *right)
{
    const Visit *first = left;
    const Visit *second = right;
    if (first->causal_start != second->causal_start) {
        return first->causal_start < second->causal_start ? -1 : 1;
    }
    if (first->lease_id != second->lease_id) {
        return first->lease_id < second->lease_id ? -1 : 1;
    }
    return 0;
}

/* The record order: by successor, then the completion waited for, then the
 * predecessor. */
static int compare_pair(const void *left, const void *right)
{
    const Pair *first = left;
    const Pair *second = right;
    if (first->successor_id != second->successor_id) {
        return first->successor_id < second->successor_id ? -1 : 1;
    }
    if (first->dependency_id != second->dependency_id) {
        return first->dependency_id < second->dependency_id ? -1 : 1;
    }
    if (first->predecessor_id != second->predecessor_id) {
        return first->predecessor_id < second->predecessor_id ? -1 : 1;
    }
    return 0;
}

/* Where a lease's retirement sits and what it publishes. */
typedef struct {
    uint64_t dependency_id;
    uint32_t task;
    uint32_t action;
    uint8_t purpose;
} Retirement;

/* Decodes a predecessor's retirement, or names the refusal. */
static uint32_t retirement(
    const ShadowSpillLayoutReuseProblem *problem,
    uint64_t lease_id,
    Retirement *retired
)
{
    const ShadowSpillAdmissionOperations *operations = problem->operations;
    if (lease_id >= operations->lease_count) {
        return SHADOWSPILL_LAYOUT_REUSE_REFUSAL_NO_DEPENDENCY;
    }
    const uint64_t retire = operations->lease_retires[lease_id];
    if (retire == SHADOWSPILL_ADMISSION_NO_OPERATION ||
        retire >= operations->operation_count ||
        operations->dependency_ids[retire] == SHADOWSPILL_ADMISSION_NO_DEPENDENCY) {
        return SHADOWSPILL_LAYOUT_REUSE_REFUSAL_NO_DEPENDENCY;
    }
    const uint8_t boundary = operations->boundaries[retire];
    const uint32_t position = operations->indices[retire];
    retired->dependency_id = operations->dependency_ids[retire];
    retired->purpose = operations->purposes[retire];
    if (boundary == SHADOWSPILL_ADMISSION_BOUNDARY_ACTION_TRIGGER ||
        boundary == SHADOWSPILL_ADMISSION_BOUNDARY_ACTION_COMPLETION) {
        if (position >= problem->schedule->action_count) {
            return SHADOWSPILL_LAYOUT_REUSE_REFUSAL_PREDECESSOR_WITHOUT_TASK;
        }
        retired->task = problem->schedule->action_trigger_tasks[position];
        retired->action = position;
    } else if (boundary == SHADOWSPILL_ADMISSION_BOUNDARY_INITIAL) {
        return SHADOWSPILL_LAYOUT_REUSE_REFUSAL_PREDECESSOR_WITHOUT_TASK;
    } else {
        retired->task = position;
        retired->action = SHADOWSPILL_PLANNER_NO_INDEX;
    }
    return SHADOWSPILL_LAYOUT_REUSE_REFUSAL_NONE;
}

static int valid_problem(const ShadowSpillLayoutReuseProblem *problem)
{
    if (problem == NULL || problem->abi_version != SHADOWSPILL_ABI_VERSION ||
        problem->operations == NULL || problem->schedule == NULL) {
        return 0;
    }
    const ShadowSpillAdmissionOperations *operations = problem->operations;
    if (operations->lease_retires == NULL || operations->boundaries == NULL ||
        operations->indices == NULL || operations->purposes == NULL ||
        operations->dependency_ids == NULL ||
        (problem->schedule->action_count != 0U &&
         problem->schedule->action_trigger_tasks == NULL)) {
        return 0;
    }
    if (problem->lifetime_count != 0U &&
        (problem->lifetimes == NULL || problem->identities == NULL ||
         problem->offsets == NULL)) {
        return 0;
    }
    return problem->lifetime_count <= UINT32_MAX;
}

static int valid_result(const ShadowSpillLayoutReuseResult *result)
{
    if (result == NULL) {
        return 0;
    }
    if (result->capacity == 0U) {
        return 1;
    }
    return result->dependency_ids != NULL && result->predecessor_leases != NULL &&
        result->predecessor_purposes != NULL && result->predecessor_tasks != NULL &&
        result->predecessor_actions != NULL && result->successor_leases != NULL &&
        result->successor_tasks != NULL && result->successor_actions != NULL;
}

static void refuse(
    ShadowSpillLayoutReuseResult *result,
    uint32_t refusal,
    uint64_t predecessor,
    uint64_t successor
)
{
    result->refusal = refusal;
    result->refusal_predecessor = predecessor;
    result->refusal_successor = successor;
}

ShadowSpillStatus shadowspill_recover_layout_reuse(
    const ShadowSpillLayoutReuseProblem *problem,
    ShadowSpillLayoutReuseResult *result
)
{
    if (!valid_problem(problem) || !valid_result(result)) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    result->count = 0U;
    refuse(result, SHADOWSPILL_LAYOUT_REUSE_REFUSAL_NONE, 0U, 0U);
    const uint32_t count = (uint32_t)problem->lifetime_count;
    if (count == 0U) {
        return SHADOWSPILL_STATUS_OK;
    }
    const ShadowSpillLeaseIdentity *identities = problem->identities;

    ShadowSpillStatus status = SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    OwnerTree tree;
    memset(&tree, 0, sizeof(tree));
    Found found = {0};
    Pairs pairs = {.problem = problem};
    Visit *visits = malloc((size_t)count * sizeof(*visits));
    found.stamps = calloc(count, sizeof(*found.stamps));
    if (visits == NULL || found.stamps == NULL ||
        owner_tree_create(&tree, problem->lifetimes, problem->offsets, count) != 0) {
        goto done;
    }
    for (uint32_t index = 0U; index < count; ++index) {
        visits[index] = (Visit){
            .causal_start = identities[index].causal_start,
            .lease_id = identities[index].lease_id,
            .index = index,
        };
    }
    qsort(visits, count, sizeof(*visits), compare_visit);

    for (uint32_t position = 0U; position < count; ++position) {
        const uint32_t successor = visits[position].index;
        const uint64_t start = problem->offsets[successor];
        const uint64_t end = start + problem->lifetimes[successor].bytes;
        const uint32_t left = coordinate_rank(&tree, start);
        const uint32_t right = coordinate_rank(&tree, end);
        found.count = 0U;
        found.serial = position + 1U;
        if (query(&tree, 1U, 0U, tree.length, left, right, &found) != 0) {
            goto done;
        }
        assign(&tree, 1U, 0U, tree.length, left, right, (int32_t)successor);
        for (uint32_t item = 0U; item < found.count; ++item) {
            const uint32_t predecessor = (uint32_t)found.items[item];
            const uint64_t predecessor_id = identities[predecessor].lease_id;
            const uint64_t successor_id = identities[successor].lease_id;
            if (identities[predecessor].causal_end > identities[successor].causal_start) {
                refuse(result, SHADOWSPILL_LAYOUT_REUSE_REFUSAL_OVERLAP, predecessor_id, successor_id);
                status = SHADOWSPILL_STATUS_INVALID_OPERATIONS;
                goto done;
            }
            Retirement retired;
            const uint32_t refusal = retirement(problem, predecessor_id, &retired);
            if (refusal != SHADOWSPILL_LAYOUT_REUSE_REFUSAL_NONE) {
                refuse(result, refusal, predecessor_id, successor_id);
                status = SHADOWSPILL_STATUS_INVALID_OPERATIONS;
                goto done;
            }
            if (identities[successor].task == SHADOWSPILL_PLANNER_NO_INDEX &&
                identities[successor].action == SHADOWSPILL_PLANNER_NO_INDEX) {
                refuse(result, SHADOWSPILL_LAYOUT_REUSE_REFUSAL_SUCCESSOR_WITHOUT_CONSUMER, predecessor_id, successor_id);
                status = SHADOWSPILL_STATUS_INVALID_OPERATIONS;
                goto done;
            }
            if (pairs_add(&pairs, (Pair){
                    .successor_id = successor_id,
                    .dependency_id = retired.dependency_id,
                    .predecessor_id = predecessor_id,
                    .predecessor = predecessor,
                    .successor = successor,
                }) != 0) {
                goto done;
            }
        }
    }

    result->count = pairs.count;
    if (pairs.count > result->capacity) {
        /* The caller's arrays are too small; `count` says how many entries
         * a second call needs. */
        status = SHADOWSPILL_STATUS_INVALID_ARGUMENT;
        goto done;
    }
    qsort(pairs.pairs, (size_t)pairs.count, sizeof(*pairs.pairs), compare_pair);
    for (uint64_t index = 0U; index < pairs.count; ++index) {
        const Pair pair = pairs.pairs[index];
        Retirement retired;
        (void)retirement(problem, pair.predecessor_id, &retired);
        result->dependency_ids[index] = pair.dependency_id;
        result->predecessor_leases[index] = pair.predecessor_id;
        result->predecessor_purposes[index] = retired.purpose;
        result->predecessor_tasks[index] = retired.task;
        result->predecessor_actions[index] = retired.action;
        result->successor_leases[index] = pair.successor_id;
        result->successor_tasks[index] = identities[pair.successor].task;
        result->successor_actions[index] = identities[pair.successor].action;
    }
    status = SHADOWSPILL_STATUS_OK;

done:
    free(visits);
    free(found.items);
    free(found.stamps);
    free(pairs.pairs);
    owner_tree_destroy(&tree);
    return status;
}
