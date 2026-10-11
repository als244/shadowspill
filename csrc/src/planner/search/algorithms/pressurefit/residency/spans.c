/* What residency costs at a boundary, and over a span. */
#include "internal.h"

/* The charged interval of one span, or an empty one (end < start): one
 * boundary of fetch headroom before a span that does not begin with a
 * production, and not the span's last boundary when nothing accesses the
 * alias later and it may leave. */
static void charge_span(
    const ShadowSpillPressureFitResidencyProblem *problem,
    int fetch_headroom,
    uint32_t alias,
    uint32_t start,
    uint32_t end,
    int32_t *charged_start,
    int32_t *charged_end
) {
    *charged_start = (int32_t)start;
    if (fetch_headroom != 0 && start > 0U &&
        !shadowspill_produced_at(problem, alias, start)) {
        --*charged_start;
    }
    *charged_end = (int32_t)end;
    const int32_t end_boundary = (int32_t)end - 1;
    if (end_boundary >= 0 && problem->final_location[alias] != 0 &&
        !shadowspill_span_accessed_after(problem, alias, start, end, end_boundary)) {
        --*charged_end;
    }
}

void shadowspill_residency_span_charge(
    const ShadowSpillPressureFitResidencyProblem *problem,
    const ShadowSpillPressureFitResidencyOptions *options,
    uint32_t alias,
    uint32_t start,
    uint32_t end,
    int32_t *charged_start,
    int32_t *charged_end
) {
    charge_span(
        problem, options->fetch_headroom != 0U, alias, start, end, charged_start, charged_end
    );
}

void shadowspill_residency_span_around(
    const ShadowSpillResidency *residency,
    uint32_t alias,
    uint32_t inside,
    uint32_t *start,
    uint32_t *end
) {
    const uint32_t index = shadowspill_residency_span_index(residency, alias, inside);
    const ShadowSpillSpanList *list = &residency->aliases[alias];
    *start = list->bounds[2U * index];
    *end = list->bounds[2U * index + 1U];
}

/* Append [start, end] to an ascending list of disjoint intervals, merging
 * it into the last one when they overlap or touch. */
static uint32_t append_interval(
    uint32_t *intervals, uint32_t count, uint32_t start, uint32_t end
) {
    if (count != 0U && start <= intervals[2U * count - 1U] + 1U) {
        if (end > intervals[2U * count - 1U]) {
            intervals[2U * count - 1U] = end;
        }
        return count;
    }
    intervals[2U * count] = start;
    intervals[2U * count + 1U] = end;
    return count + 1U;
}

uint32_t shadowspill_residency_charged_intervals(
    const ShadowSpillPressureFitResidencyProblem *problem,
    int fetch_headroom,
    const ShadowSpillResidency *residency,
    uint32_t alias,
    uint32_t *intervals
) {
    const ShadowSpillSpanList *list = &residency->aliases[alias];
    /* The spans' charged intervals, ascending: two consecutive spans can
     * share at most their meeting boundary, so appending merges them. */
    uint32_t spans = 0U;
    for (uint32_t index = 0U; index < list->count; ++index) {
        int32_t charged_start = 0;
        int32_t charged_end = -1;
        charge_span(
            problem, fetch_headroom, alias,
            list->bounds[2U * index], list->bounds[2U * index + 1U],
            &charged_start, &charged_end
        );
        if (charged_end >= charged_start) {
            spans = append_interval(
                intervals, spans, (uint32_t)charged_start, (uint32_t)charged_end
            );
        }
    }
    /* Then the reserved boundaries, each a point, merged in from the back
     * so the list stays in place: the merged list is written from its last
     * slot downwards and never overtakes what it still has to read. */
    const uint32_t reserved_first = problem->reserved_offsets[alias];
    const uint32_t reserved_last = problem->reserved_offsets[alias + 1U];
    uint32_t reserved = reserved_last - reserved_first;
    if (reserved == 0U) {
        return spans;
    }
    uint32_t total = spans + reserved;
    uint32_t write = total;
    uint32_t span = spans;
    while (reserved != 0U) {
        const uint32_t point = problem->reserved_positions[reserved_first + reserved - 1U];
        --write;
        if (span != 0U && intervals[2U * (span - 1U)] > point) {
            --span;
            intervals[2U * write] = intervals[2U * span];
            intervals[2U * write + 1U] = intervals[2U * span + 1U];
        } else {
            --reserved;
            intervals[2U * write] = point;
            intervals[2U * write + 1U] = point;
        }
    }
    /* Ascending by start now; merge whatever overlaps or touches. */
    uint32_t merged = 0U;
    for (uint32_t item = 0U; item < total; ++item) {
        merged = append_interval(
            intervals, merged, intervals[2U * item], intervals[2U * item + 1U]
        );
    }
    return merged;
}

int shadowspill_residency_add_charges(
    const ShadowSpillPressureFitResidencyProblem *problem,
    int fetch_headroom,
    const ShadowSpillResidency *residency,
    uint64_t *pressure
) {
    const uint32_t count = problem->boundary_count;
    const size_t pressure_cells = (size_t)problem->device_count * count;
    /* Each interval is two updates of a difference array, and one prefix
     * sum per device turns them back into bytes per boundary. Arithmetic
     * is modulo 2^64 on the way and exact at the end, because every prefix
     * is a true, non-negative sum. */
    uint64_t *difference = calloc(pressure_cells == 0U ? 1U : pressure_cells, sizeof(*difference));
    uint32_t *intervals = malloc((size_t)count * 4U * sizeof(*intervals));
    if (difference == NULL || intervals == NULL) {
        free(difference);
        free(intervals);
        return -1;
    }
    for (uint32_t alias = 0U; alias < problem->alias_count; ++alias) {
        if (!shadowspill_alias_may_cut(problem, alias)) {
            continue;
        }
        const uint32_t charged = shadowspill_residency_charged_intervals(
            problem, fetch_headroom, residency, alias, intervals
        );
        uint64_t *row = difference + (size_t)problem->alias_device[alias] * count;
        const uint64_t size = problem->alias_size_bytes[alias];
        for (uint32_t item = 0U; item < charged; ++item) {
            row[intervals[2U * item]] += size;
            if (intervals[2U * item + 1U] + 1U < count) {
                row[intervals[2U * item + 1U] + 1U] -= size;
            }
        }
    }
    for (uint32_t device = 0U; device < problem->device_count; ++device) {
        uint64_t running = 0U;
        const uint64_t *row = difference + (size_t)device * count;
        uint64_t *target = pressure + (size_t)device * count;
        for (uint32_t boundary = 0U; boundary < count; ++boundary) {
            running += row[boundary];
            target[boundary] += running;
        }
    }
    free(difference);
    free(intervals);
    return 0;
}

/* Whether `boundary` lies in one of `count` ascending disjoint intervals. */
static int intervals_cover(const uint32_t *intervals, uint32_t count, uint32_t boundary) {
    uint32_t low = 0U;
    uint32_t high = count;
    while (low < high) {
        const uint32_t middle = low + (high - low) / 2U;
        if (intervals[2U * middle + 1U] < boundary) {
            low = middle + 1U;
        } else {
            high = middle;
        }
    }
    return low < count && intervals[2U * low] <= boundary;
}

int shadowspill_residency_pressure_at(
    const ShadowSpillPressureFitResidencyProblem *problem,
    const ShadowSpillPressureFitResidencyOptions *options,
    const ShadowSpillResidency *residency,
    uint32_t device,
    uint32_t boundary,
    uint64_t *pressure_bytes
) {
    if (problem == NULL || options == NULL || residency == NULL ||
        pressure_bytes == NULL || device >= problem->device_count ||
        boundary >= problem->boundary_count) {
        return -1;
    }
    uint32_t *intervals = malloc((size_t)problem->boundary_count * 4U * sizeof(*intervals));
    if (intervals == NULL) {
        return -1;
    }
    uint64_t pressure = 0U;
    for (uint32_t alias = 0U; alias < problem->alias_count; ++alias) {
        if (problem->alias_device[alias] != device ||
            !shadowspill_alias_may_cut(problem, alias)) {
            continue;
        }
        const uint32_t charged = shadowspill_residency_charged_intervals(
            problem, options->fetch_headroom != 0U, residency, alias, intervals
        );
        if (!intervals_cover(intervals, charged, boundary)) {
            continue;
        }
        if (pressure > UINT64_MAX - problem->alias_size_bytes[alias]) {
            free(intervals);
            return -1;
        }
        pressure += problem->alias_size_bytes[alias];
    }
    free(intervals);
    *pressure_bytes = pressure;
    return 0;
}
