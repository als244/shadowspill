/* Deterministic fixed-offset placement for one execution-pool slice.
 *
 * Leases are placed largest first, longest-lived first among equals, and each
 * takes the lowest offset that clears every lease it overlaps in time. Two
 * leases whose lifetimes are disjoint may share an offset, which is what makes
 * the slice smaller than the sum of the leases.
 *
 * Finding that offset needs only the UNION of the address ranges the
 * overlapping leases occupy, never the leases themselves. A packed layout's
 * union collapses to far fewer disjoint ranges than there are overlapping
 * leases, so the index below stores merged ranges rather than leases. That
 * keeps a query proportional to how fragmented the layout is rather than to
 * how many leases are live.
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

/* Adds [start, end) to the union. Ranges that touch are merged: a zero-width
 * gap can hold nothing, so merging them cannot change any later answer. The
 * ranges are disjoint and sorted, so their ends are sorted too, and the first
 * range that could touch the new one is found by bisection. */
static int occupancy_insert(Occupancy *list, uint64_t start, uint64_t end)
{
    uint32_t low = 0U;
    uint32_t high = list->count;
    while (low < high) {
        const uint32_t middle = low + (high - low) / 2U;
        if (list->data[middle].end < start) {
            low = middle + 1U;
        } else {
            high = middle;
        }
    }
    const uint32_t first = low;
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
    const ShadowSpillPlacementProblem *context,
    TimeAxis *axis
)
{
    const uint32_t count = context->lifetime_count;
    axis->times = malloc((size_t)count * 2U * sizeof(*axis->times));
    axis->start_rank = malloc((size_t)count * sizeof(*axis->start_rank));
    axis->end_rank = malloc((size_t)count * sizeof(*axis->end_rank));
    if (axis->times == NULL || axis->start_rank == NULL ||
        axis->end_rank == NULL) {
        return -1;
    }
    for (uint32_t index = 0U; index < count; ++index) {
        axis->times[index * 2U] = context->lifetimes[index].start_ns;
        axis->times[index * 2U + 1U] = context->lifetimes[index].end_ns;
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
            time_axis_rank(axis, context->lifetimes[index].start_ns);
        axis->end_rank[index] =
            time_axis_rank(axis, context->lifetimes[index].end_ns);
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
    /* Ranges recorded exactly at this node: those of the leases whose
     * lifetime this node is one canonical cover of. */
    Occupancy *nodes;
    /* The union of every range recorded at this node or anywhere below it.
     * A query whose lifetime contains a node's whole interval overlaps every
     * lease recorded in that subtree, so it reads this one list and never
     * descends: the gather costs O(log n) nodes rather than every node under
     * the lifetime. */
    Occupancy *subtree;
    uint32_t node_count;
    uint32_t length;
} OccupancyIndex;

static void occupancy_index_destroy(OccupancyIndex *index)
{
    for (uint32_t node = 0U; node < index->node_count; ++node) {
        occupancy_destroy(&index->nodes[node]);
        occupancy_destroy(&index->subtree[node]);
    }
    free(index->nodes);
    free(index->subtree);
    index->nodes = NULL;
    index->subtree = NULL;
    index->node_count = 0U;
}

static int occupancy_index_create(OccupancyIndex *index, uint32_t length)
{
    index->length = length == 0U ? 1U : length;
    index->node_count = index->length * 4U + 4U;
    index->nodes = calloc(index->node_count, sizeof(*index->nodes));
    index->subtree = calloc(index->node_count, sizeof(*index->subtree));
    return index->nodes == NULL || index->subtree == NULL ? -1 : 0;
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
    if (occupancy_insert(&index->subtree[node], offset, offset + bytes) != 0) {
        return -1;
    }
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

/* The lists a query reads, each sorted by address, read in place. */
typedef struct {
    const Occupancy **lists;
    uint32_t count;
    uint32_t capacity;
} Gathered;

static void gathered_destroy(Gathered *gathered)
{
    free(gathered->lists);
    memset(gathered, 0, sizeof(*gathered));
}

static int gathered_add(Gathered *gathered, const Occupancy *list)
{
    if (list->count == 0U) {
        return 0;
    }
    if (gathered->count == gathered->capacity) {
        const uint32_t capacity =
            gathered->capacity == 0U ? 64U : gathered->capacity * 2U;
        const Occupancy **lists =
            realloc(gathered->lists, (size_t)capacity * sizeof(*lists));
        if (lists == NULL) {
            return -1;
        }
        gathered->lists = lists;
        gathered->capacity = capacity;
    }
    gathered->lists[gathered->count++] = list;
    return 0;
}

/* Gathers the lists of every visited node: a node inside the query hands
 * over its subtree union and is not descended into, a node the query only
 * crosses hands over what is recorded at it and is. Nothing is copied: the
 * lists are read where they are, and ranges from different lists may
 * overlap, which the merge below folds together. */
static int occupancy_index_collect(
    const OccupancyIndex *index,
    uint32_t node,
    uint32_t left,
    uint32_t right,
    uint32_t start,
    uint32_t end,
    Gathered *found
)
{
    if (end <= left || right <= start) {
        return 0;
    }
    const int inside = start <= left && right <= end;
    if (gathered_add(found, inside ? &index->subtree[node] : &index->nodes[node])
            != 0) {
        return -1;
    }
    if (inside || right - left == 1U) {
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
    const ShadowSpillPlacementProblem *context,
    const TimeAxis *axis
)
{
    const uint32_t count = context->lifetime_count;
    OrderKey *order = malloc((size_t)count * sizeof(*order));
    if (order == NULL) {
        return NULL;
    }
    for (uint32_t index = 0U; index < count; ++index) {
        order[index] = (OrderKey){
            .bytes = context->lifetimes[index].bytes,
            .start_ns = context->lifetimes[index].start_ns,
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

/* One list's next unread range, keyed on its start for the merge. */
typedef struct {
    uint64_t start;
    uint32_t list;
    uint32_t item;
} Head;

static void head_sift_down(Head *heap, uint32_t count, uint32_t index)
{
    for (;;) {
        const uint32_t left = index * 2U + 1U;
        const uint32_t right = left + 1U;
        uint32_t smallest = index;
        if (left < count && heap[left].start < heap[smallest].start) {
            smallest = left;
        }
        if (right < count && heap[right].start < heap[smallest].start) {
            smallest = right;
        }
        if (smallest == index) {
            return;
        }
        const Head swap = heap[index];
        heap[index] = heap[smallest];
        heap[smallest] = swap;
        index = smallest;
    }
}

/* Lowest aligned offset clearing every gathered range. The lists are merged
 * by start through a heap of their heads and read only as far as the first
 * gap that fits. Ranges ending at or below the cursor are skipped, so
 * overlapping ranges from different lists fold together here; the order of
 * ranges sharing a start does not matter, since each raises the cursor to
 * its end and the cursor settles at the largest. `heap` holds one entry per
 * gathered list. */
static uint64_t lowest_fit(
    const Gathered *found,
    Head *heap,
    uint64_t needed,
    uint64_t alignment
)
{
    uint32_t count = 0U;
    for (uint32_t list = 0U; list < found->count; ++list) {
        heap[count++] = (Head){
            .start = found->lists[list]->data[0].start, .list = list, .item = 0U
        };
    }
    for (uint32_t index = count / 2U; index-- > 0U;) {
        head_sift_down(heap, count, index);
    }
    uint64_t cursor = 0U;
    while (count != 0U) {
        const Head head = heap[0];
        const Occupancy *list = found->lists[head.list];
        const AddressRange range = list->data[head.item];
        if (range.end > cursor) {
            if (align_up(cursor, alignment) + needed <= range.start) {
                break;
            }
            cursor = range.end;
        }
        if (head.item + 1U < list->count) {
            heap[0] = (Head){
                .start = list->data[head.item + 1U].start,
                .list = head.list,
                .item = head.item + 1U,
            };
        } else {
            heap[0] = heap[--count];
        }
        head_sift_down(heap, count, 0U);
    }
    return align_up(cursor, alignment);
}

/* ------------------------------------------------------------------ entry */

typedef struct {
    TimeAxis axis;
    OccupancyIndex index;
    OrderKey *order;
    Gathered found;
    Head *heap;
    uint32_t heap_capacity;
} Placer;

static void placer_destroy(Placer *placer)
{
    time_axis_destroy(&placer->axis);
    occupancy_index_destroy(&placer->index);
    free(placer->order);
    gathered_destroy(&placer->found);
    free(placer->heap);
    memset(placer, 0, sizeof(*placer));
}

/* The heap holds one head per gathered list. */
static int placer_reserve_heap(Placer *placer)
{
    if (placer->found.count <= placer->heap_capacity) {
        return 0;
    }
    Head *heap = realloc(
        placer->heap, (size_t)placer->found.capacity * sizeof(*heap)
    );
    if (heap == NULL) {
        return -1;
    }
    placer->heap = heap;
    placer->heap_capacity = placer->found.capacity;
    return 0;
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
    const ShadowSpillPlacementProblem context = {
        .abi_version = SHADOWSPILL_ABI_VERSION,
        .lifetime_count = count,
        .lifetimes = lifetimes,
    };
    Placer placer;
    memset(&placer, 0, sizeof(placer));
    ShadowSpillStatus status = SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    if (time_axis_build(&context, &placer.axis) != 0 ||
        occupancy_index_create(&placer.index, placer.axis.time_count) != 0) {
        goto done;
    }
    placer.order = placing_order(&context, &placer.axis);
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
            placer_reserve_heap(&placer) != 0) {
            goto done;
        }
        const uint64_t offset = lowest_fit(
            &placer.found, placer.heap, record.bytes, record.alignment
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
    const ShadowSpillPlacementProblem *context,
    ShadowSpillLeaseLifetime *records,
    uint32_t *sources
)
{
    uint32_t count = 0U;
    for (uint32_t index = 0U; index < context->lifetime_count; ++index) {
        if (context->excluded[index] == 0U) {
            records[count] = context->lifetimes[index];
            sources[count++] = index;
        }
    }
    return count;
}

ShadowSpillStatus shadowspill_place_lifetimes(
    const ShadowSpillPlacementProblem *context,
    ShadowSpillPlacementResult *result
)
{
    if (context == NULL || result == NULL ||
        context->abi_version != SHADOWSPILL_ABI_VERSION) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    result->required_bytes = 0U;
    const uint32_t count = context->lifetime_count;
    if (count == 0U) {
        return SHADOWSPILL_STATUS_OK;
    }
    if (context->lifetimes == NULL || result->offsets == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    if (context->excluded == NULL) {
        return place_region(
            context->lifetimes, count, result->offsets, &result->required_bytes
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
    const uint32_t placed = gather(context, records, sources);
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
