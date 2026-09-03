/* Deterministic fixed-offset placement for one execution-pool slice.
 *
 * Leases are placed largest first, longest-lived first among equals, and each
 * takes the lowest offset that clears every lease it overlaps in time. Two
 * leases whose lifetimes are disjoint may share an offset, which is what makes
 * the slice smaller than the sum of the leases.
 *
 * Finding that offset needs only the UNION of the address ranges the
 * overlapping leases occupy, never the leases themselves. A packed layout's
 * union collapses hard - measured at 10 to 17 disjoint address ranges where
 * 200 to 430 leases overlap - so the index below stores merged ranges rather
 * than leases.
 * That keeps a query proportional to how fragmented the layout is rather than
 * to how many leases are live.
 */

#include <shadowspill/planner.h>

#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------- occupancy */

/* Half-open address range [start, end). */
typedef struct {
    uint64_t start;
    uint64_t end;
} AddressRange;

/* The addresses already taken, as sorted disjoint ranges. Inserting a range
 * can only merge neighbours, never split them, so this stays as short as the
 * layout is contiguous. */
typedef struct {
    AddressRange *data;
    uint32_t count;
    uint32_t capacity;
} Occupancy;

static void occupancy_destroy(Occupancy *list)
{
    free(list->data);
    list->data = NULL;
    list->count = 0U;
    list->capacity = 0U;
}

static int occupancy_reserve(Occupancy *list, uint32_t needed)
{
    if (needed <= list->capacity) {
        return 0;
    }
    uint32_t capacity = list->capacity == 0U ? 4U : list->capacity;
    while (capacity < needed) {
        capacity *= 2U;
    }
    AddressRange *data = realloc(list->data, (size_t)capacity * sizeof(*data));
    if (data == NULL) {
        return -1;
    }
    list->data = data;
    list->capacity = capacity;
    return 0;
}

static int occupancy_append(Occupancy *list, AddressRange range)
{
    if (occupancy_reserve(list, list->count + 1U) != 0) {
        return -1;
    }
    list->data[list->count++] = range;
    return 0;
}

/* Adds [start, end) to the union. Ranges that touch are merged: a zero-width
 * gap can hold nothing, so merging them cannot change any later answer. */
static int occupancy_insert(Occupancy *list, uint64_t start, uint64_t end)
{
    uint32_t first = 0U;
    while (first < list->count && list->data[first].end < start) {
        ++first;
    }
    uint32_t last = first;
    while (last < list->count && list->data[last].start <= end) {
        ++last;
    }
    if (first == last) {
        if (occupancy_reserve(list, list->count + 1U) != 0) {
            return -1;
        }
        memmove(
            &list->data[first + 1U],
            &list->data[first],
            (size_t)(list->count - first) * sizeof(*list->data)
        );
        list->data[first] = (AddressRange){.start = start, .end = end};
        list->count += 1U;
        return 0;
    }
    const uint64_t merged_start = list->data[first].start < start
        ? list->data[first].start
        : start;
    const uint64_t merged_end = list->data[last - 1U].end > end
        ? list->data[last - 1U].end
        : end;
    list->data[first] = (AddressRange){.start = merged_start, .end = merged_end};
    memmove(
        &list->data[first + 1U],
        &list->data[last],
        (size_t)(list->count - last) * sizeof(*list->data)
    );
    list->count -= last - first - 1U;
    return 0;
}

/* ---------------------------------------------------------------- time axis */

/* The distinct endpoint times, sorted. A lease's lifetime becomes the rank
 * range [start_rank, end_rank), which is what the index below is keyed on. */
typedef struct {
    uint64_t *times;
    uint32_t time_count;
    uint32_t *start_rank;
    uint32_t *end_rank;
} TimeAxis;

static void time_axis_destroy(TimeAxis *axis)
{
    free(axis->times);
    free(axis->start_rank);
    free(axis->end_rank);
    memset(axis, 0, sizeof(*axis));
}

static int compare_time(const void *left, const void *right)
{
    const uint64_t first = *(const uint64_t *)left;
    const uint64_t second = *(const uint64_t *)right;
    return (first > second) - (first < second);
}

static uint32_t time_axis_rank(const TimeAxis *axis, uint64_t value)
{
    uint32_t low = 0U;
    uint32_t high = axis->time_count;
    while (low < high) {
        const uint32_t middle = low + (high - low) / 2U;
        if (axis->times[middle] < value) {
            low = middle + 1U;
        } else {
            high = middle;
        }
    }
    return low;
}

static int time_axis_build(
    const ShadowSpillPlacementProblem *problem,
    TimeAxis *axis
)
{
    const uint32_t count = problem->lifetime_count;
    axis->times = malloc((size_t)count * 2U * sizeof(*axis->times));
    axis->start_rank = malloc((size_t)count * sizeof(*axis->start_rank));
    axis->end_rank = malloc((size_t)count * sizeof(*axis->end_rank));
    if (axis->times == NULL || axis->start_rank == NULL ||
        axis->end_rank == NULL) {
        return -1;
    }
    for (uint32_t index = 0U; index < count; ++index) {
        axis->times[index * 2U] = problem->lifetimes[index].start_ns;
        axis->times[index * 2U + 1U] = problem->lifetimes[index].end_ns;
    }
    qsort(axis->times, (size_t)count * 2U, sizeof(*axis->times), compare_time);

    uint32_t distinct = 0U;
    for (uint32_t index = 0U; index < count * 2U; ++index) {
        if (distinct == 0U || axis->times[distinct - 1U] != axis->times[index]) {
            axis->times[distinct++] = axis->times[index];
        }
    }
    axis->time_count = distinct;

    /* Rank each endpoint once here rather than on every query below. */
    for (uint32_t index = 0U; index < count; ++index) {
        axis->start_rank[index] =
            time_axis_rank(axis, problem->lifetimes[index].start_ns);
        axis->end_rank[index] =
            time_axis_rank(axis, problem->lifetimes[index].end_ns);
    }
    return 0;
}

/* --------------------------------------------------------- occupancy index */

/* Segment tree over [0, length) holding, per node, the union of the address
 * ranges occupied by the leases recorded there. A lease is recorded on the
 * nodes that exactly cover its lifetime, so every lease reachable from a node
 * the query visits does overlap the query - which is what makes it sound to
 * merge them before the query ever runs. */
typedef struct {
    Occupancy *nodes;
    uint32_t node_count;
    uint32_t length;
} OccupancyIndex;

static void occupancy_index_destroy(OccupancyIndex *index)
{
    for (uint32_t node = 0U; node < index->node_count; ++node) {
        occupancy_destroy(&index->nodes[node]);
    }
    free(index->nodes);
    index->nodes = NULL;
    index->node_count = 0U;
}

static int occupancy_index_create(OccupancyIndex *index, uint32_t length)
{
    index->length = length == 0U ? 1U : length;
    index->node_count = index->length * 4U + 4U;
    index->nodes = calloc(index->node_count, sizeof(*index->nodes));
    return index->nodes == NULL ? -1 : 0;
}

static int occupancy_index_insert(
    OccupancyIndex *index,
    uint32_t node,
    uint32_t left,
    uint32_t right,
    uint32_t start,
    uint32_t end,
    uint64_t offset,
    uint64_t bytes
)
{
    if (start <= left && right <= end) {
        return occupancy_insert(&index->nodes[node], offset, offset + bytes);
    }
    const uint32_t middle = left + (right - left) / 2U;
    if (start < middle && occupancy_index_insert(
            index, node * 2U, left, middle, start, end, offset, bytes) != 0) {
        return -1;
    }
    if (middle < end && occupancy_index_insert(
            index, node * 2U + 1U, middle, right, start, end, offset, bytes
        ) != 0) {
        return -1;
    }
    return 0;
}

/* Gathers the ranges of every visited node. Ranges from different nodes may
 * overlap; the scan below folds them together, so they are not merged here. */
static int occupancy_index_collect(
    const OccupancyIndex *index,
    uint32_t node,
    uint32_t left,
    uint32_t right,
    uint32_t start,
    uint32_t end,
    Occupancy *found
)
{
    if (end <= left || right <= start) {
        return 0;
    }
    const Occupancy *bucket = &index->nodes[node];
    for (uint32_t item = 0U; item < bucket->count; ++item) {
        if (occupancy_append(found, bucket->data[item]) != 0) {
            return -1;
        }
    }
    if (right - left == 1U) {
        return 0;
    }
    const uint32_t middle = left + (right - left) / 2U;
    if (start < middle && occupancy_index_collect(
            index, node * 2U, left, middle, start, end, found) != 0) {
        return -1;
    }
    if (middle < end && occupancy_index_collect(
            index, node * 2U + 1U, middle, right, start, end, found) != 0) {
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------ placing order */

/* Largest first, then longest-lived, then earliest, then lowest index. The
 * last two keys only break ties; they exist so that equal records placed in
 * either order give the same answer. */
typedef struct {
    uint64_t bytes;
    uint64_t start_ns;
    uint32_t span;
    uint32_t lifetime;
} OrderKey;

static int compare_order(const void *left, const void *right)
{
    const OrderKey *first = left;
    const OrderKey *second = right;
    if (first->bytes != second->bytes) {
        return first->bytes > second->bytes ? -1 : 1;
    }
    if (first->span != second->span) {
        return first->span > second->span ? -1 : 1;
    }
    if (first->start_ns != second->start_ns) {
        return first->start_ns < second->start_ns ? -1 : 1;
    }
    if (first->lifetime != second->lifetime) {
        return first->lifetime < second->lifetime ? -1 : 1;
    }
    return 0;
}

static OrderKey *placing_order(
    const ShadowSpillPlacementProblem *problem,
    const TimeAxis *axis
)
{
    const uint32_t count = problem->lifetime_count;
    OrderKey *order = malloc((size_t)count * sizeof(*order));
    if (order == NULL) {
        return NULL;
    }
    for (uint32_t index = 0U; index < count; ++index) {
        order[index] = (OrderKey){
            .bytes = problem->lifetimes[index].bytes,
            .start_ns = problem->lifetimes[index].start_ns,
            .span = axis->end_rank[index] - axis->start_rank[index],
            .lifetime = index,
        };
    }
    qsort(order, count, sizeof(*order), compare_order);
    return order;
}

/* ------------------------------------------------------------- lowest fit */

static uint64_t align_up(uint64_t value, uint64_t alignment)
{
    if (alignment <= 1U) {
        return value;
    }
    return ((value + alignment - 1U) / alignment) * alignment;
}

/* Orders gathered ranges by address. A qsort comparator would cost more than
 * the comparison itself, which is a single load. */
static void sort_by_address(AddressRange *runs, uint32_t count, AddressRange *scratch)
{
    if (count < 2U) {
        return;
    }
    const uint32_t middle = count / 2U;
    sort_by_address(runs, middle, scratch);
    sort_by_address(runs + middle, count - middle, scratch);
    uint32_t left = 0U;
    uint32_t right = middle;
    uint32_t out = 0U;
    while (left < middle && right < count) {
        scratch[out++] = runs[left].start <= runs[right].start
            ? runs[left++]
            : runs[right++];
    }
    while (left < middle) {
        scratch[out++] = runs[left++];
    }
    while (right < count) {
        scratch[out++] = runs[right++];
    }
    memcpy(runs, scratch, (size_t)count * sizeof(*runs));
}

/* Lowest aligned offset clearing every gathered range, which must already be
 * ordered by address. Ranges ending at or below the cursor are skipped, so
 * overlapping ranges from different nodes fold together here. */
static uint64_t lowest_fit(
    const AddressRange *runs,
    uint32_t count,
    uint64_t needed,
    uint64_t alignment
)
{
    uint64_t cursor = 0U;
    for (uint32_t item = 0U; item < count; ++item) {
        if (runs[item].end <= cursor) {
            continue;
        }
        if (align_up(cursor, alignment) + needed <= runs[item].start) {
            break;
        }
        cursor = runs[item].end;
    }
    return align_up(cursor, alignment);
}

/* ------------------------------------------------------------------ entry */

typedef struct {
    TimeAxis axis;
    OccupancyIndex index;
    OrderKey *order;
    Occupancy found;
    Occupancy scratch;
} Placer;

static void placer_destroy(Placer *placer)
{
    time_axis_destroy(&placer->axis);
    occupancy_index_destroy(&placer->index);
    free(placer->order);
    occupancy_destroy(&placer->found);
    occupancy_destroy(&placer->scratch);
    memset(placer, 0, sizeof(*placer));
}

/* One region: every lease placed from offset zero, offsets in input order. */
static ShadowSpillStatus place_region(
    const ShadowSpillLeaseLifetime *lifetimes,
    uint32_t count,
    uint64_t *offsets,
    uint64_t *required_bytes
)
{
    *required_bytes = 0U;
    if (count == 0U) {
        return SHADOWSPILL_STATUS_OK;
    }
    const ShadowSpillPlacementProblem problem = {
        .abi_version = SHADOWSPILL_ABI_VERSION,
        .lifetime_count = count,
        .lifetimes = lifetimes,
    };
    Placer placer;
    memset(&placer, 0, sizeof(placer));
    ShadowSpillStatus status = SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    if (time_axis_build(&problem, &placer.axis) != 0 ||
        occupancy_index_create(&placer.index, placer.axis.time_count) != 0) {
        goto done;
    }
    placer.order = placing_order(&problem, &placer.axis);
    if (placer.order == NULL) {
        goto done;
    }

    const uint32_t length = placer.index.length;
    for (uint32_t position = 0U; position < count; ++position) {
        const uint32_t lifetime = placer.order[position].lifetime;
        const uint32_t start = placer.axis.start_rank[lifetime];
        const uint32_t end = placer.axis.end_rank[lifetime];
        const ShadowSpillLeaseLifetime record = lifetimes[lifetime];

        placer.found.count = 0U;
        if (occupancy_index_collect(
                &placer.index, 1U, 0U, length, start, end, &placer.found) != 0 ||
            occupancy_reserve(&placer.scratch, placer.found.count) != 0) {
            goto done;
        }
        sort_by_address(placer.found.data, placer.found.count, placer.scratch.data);

        const uint64_t offset = lowest_fit(
            placer.found.data, placer.found.count, record.bytes, record.alignment
        );
        offsets[lifetime] = offset;
        if (offset + record.bytes > *required_bytes) {
            *required_bytes = offset + record.bytes;
        }
        if (occupancy_index_insert(
                &placer.index, 1U, 0U, length, start, end, offset, record.bytes
            ) != 0) {
            goto done;
        }
    }
    status = SHADOWSPILL_STATUS_OK;

done:
    placer_destroy(&placer);
    return status;
}

/* The leases placement is asked for, copied out contiguously so the region
 * placer can run on them, with where each came from. */
static uint32_t gather(
    const ShadowSpillPlacementProblem *problem,
    ShadowSpillLeaseLifetime *records,
    uint32_t *sources
)
{
    uint32_t count = 0U;
    for (uint32_t index = 0U; index < problem->lifetime_count; ++index) {
        if (problem->excluded[index] == 0U) {
            records[count] = problem->lifetimes[index];
            sources[count++] = index;
        }
    }
    return count;
}

ShadowSpillStatus shadowspill_place_lifetimes(
    const ShadowSpillPlacementProblem *problem,
    ShadowSpillPlacementResult *result
)
{
    if (problem == NULL || result == NULL ||
        problem->abi_version != SHADOWSPILL_ABI_VERSION) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    result->required_bytes = 0U;
    const uint32_t count = problem->lifetime_count;
    if (count == 0U) {
        return SHADOWSPILL_STATUS_OK;
    }
    if (problem->lifetimes == NULL || result->offsets == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    if (problem->excluded == NULL) {
        return place_region(
            problem->lifetimes, count, result->offsets, &result->required_bytes
        );
    }

    /* Placement of the leases not left out, written back where they came
     * from; an excluded lease's offset is not touched. */
    ShadowSpillLeaseLifetime *records = malloc((size_t)count * sizeof(*records));
    uint32_t *sources = malloc((size_t)count * sizeof(*sources));
    uint64_t *offsets = malloc((size_t)count * sizeof(*offsets));
    ShadowSpillStatus status = SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    if (records == NULL || sources == NULL || offsets == NULL) {
        goto done;
    }
    const uint32_t placed = gather(problem, records, sources);
    status = place_region(records, placed, offsets, &result->required_bytes);
    if (status == SHADOWSPILL_STATUS_OK) {
        for (uint32_t index = 0U; index < placed; ++index) {
            result->offsets[sources[index]] = offsets[index];
        }
    }

done:
    free(records);
    free(sources);
    free(offsets);
    return status;
}
