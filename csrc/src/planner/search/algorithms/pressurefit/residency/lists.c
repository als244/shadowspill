/* A residency as span lists: creating, copying and editing them. */
#include "internal.h"

int shadowspill_residency_create(
    ShadowSpillResidency *residency, uint32_t alias_count, uint32_t boundary_count
) {
    residency->alias_count = alias_count;
    residency->boundary_count = boundary_count;
    residency->aliases = calloc(
        alias_count == 0U ? 1U : alias_count, sizeof(*residency->aliases)
    );
    return residency->aliases == NULL ? -1 : 0;
}

void shadowspill_residency_destroy(ShadowSpillResidency *residency) {
    if (residency->aliases != NULL) {
        for (uint32_t alias = 0U; alias < residency->alias_count; ++alias) {
            free(residency->aliases[alias].bounds);
        }
    }
    free(residency->aliases);
    memset(residency, 0, sizeof(*residency));
}

/* Room for `count` spans in one list, keeping what it holds. */
static int reserve(ShadowSpillSpanList *list, uint32_t count) {
    if (count <= list->capacity) {
        return 0;
    }
    uint32_t capacity = list->capacity == 0U ? 2U : list->capacity;
    while (capacity < count) {
        if (capacity > UINT32_MAX / 2U) {
            capacity = count;
            break;
        }
        capacity *= 2U;
    }
    uint32_t *bounds = realloc(list->bounds, (size_t)capacity * 2U * sizeof(*bounds));
    if (bounds == NULL) {
        return -1;
    }
    list->bounds = bounds;
    list->capacity = capacity;
    return 0;
}

int shadowspill_residency_copy(
    ShadowSpillResidency *destination, const ShadowSpillResidency *source
) {
    if (destination->alias_count != source->alias_count ||
        destination->boundary_count != source->boundary_count) {
        return -1;
    }
    for (uint32_t alias = 0U; alias < source->alias_count; ++alias) {
        const ShadowSpillSpanList *from = &source->aliases[alias];
        ShadowSpillSpanList *to = &destination->aliases[alias];
        if (reserve(to, from->count) != 0) {
            return -1;
        }
        memcpy(to->bounds, from->bounds, (size_t)from->count * 2U * sizeof(*to->bounds));
        to->count = from->count;
    }
    return 0;
}

int shadowspill_residency_append_span(
    ShadowSpillResidency *residency, uint32_t alias, uint32_t start, uint32_t end
) {
    ShadowSpillSpanList *list = &residency->aliases[alias];
    if (start > end || end >= residency->boundary_count ||
        (list->count != 0U && list->bounds[2U * list->count - 1U] >= start) ||
        reserve(list, list->count + 1U) != 0) {
        return -1;
    }
    list->bounds[2U * list->count] = start;
    list->bounds[2U * list->count + 1U] = end;
    ++list->count;
    return 0;
}

/* Make room for one more span at `position`, shifting the rest up. */
static int open_slot(ShadowSpillSpanList *list, uint32_t position) {
    if (reserve(list, list->count + 1U) != 0) {
        return -1;
    }
    memmove(
        &list->bounds[2U * (position + 1U)],
        &list->bounds[2U * position],
        (size_t)(list->count - position) * 2U * sizeof(*list->bounds)
    );
    ++list->count;
    return 0;
}

int shadowspill_residency_clear_cells(
    ShadowSpillResidency *residency, uint32_t alias, uint32_t first, uint32_t last
) {
    ShadowSpillSpanList *list = &residency->aliases[alias];
    /* The first span that reaches the range. */
    uint32_t index = 0U;
    while (index < list->count && list->bounds[2U * index + 1U] < first) {
        ++index;
    }
    while (index < list->count && list->bounds[2U * index] <= last) {
        const uint32_t start = list->bounds[2U * index];
        const uint32_t end = list->bounds[2U * index + 1U];
        if (start < first && end > last) {
            /* The range lies strictly inside: the span becomes two. */
            if (open_slot(list, index + 1U) != 0) {
                return -1;
            }
            list->bounds[2U * index + 1U] = first - 1U;
            list->bounds[2U * (index + 1U)] = last + 1U;
            list->bounds[2U * (index + 1U) + 1U] = end;
            return 0;
        }
        if (start < first) {
            /* Keeps its head. */
            list->bounds[2U * index + 1U] = first - 1U;
            ++index;
            continue;
        }
        if (end > last) {
            /* Keeps its tail, and nothing later can reach the range. */
            list->bounds[2U * index] = last + 1U;
            return 0;
        }
        /* Covered whole: drop it and look at what moved into its place. */
        memmove(
            &list->bounds[2U * index],
            &list->bounds[2U * (index + 1U)],
            (size_t)(list->count - index - 1U) * 2U * sizeof(*list->bounds)
        );
        --list->count;
    }
    return 0;
}

int shadowspill_residency_split_after(
    ShadowSpillResidency *residency, uint32_t alias, uint32_t boundary
) {
    const uint32_t index = shadowspill_residency_span_index(residency, alias, boundary);
    ShadowSpillSpanList *list = &residency->aliases[alias];
    if (index == UINT32_MAX || list->bounds[2U * index + 1U] == boundary) {
        return 0;
    }
    const uint32_t end = list->bounds[2U * index + 1U];
    if (open_slot(list, index + 1U) != 0) {
        return -1;
    }
    list->bounds[2U * index + 1U] = boundary;
    list->bounds[2U * (index + 1U)] = boundary + 1U;
    list->bounds[2U * (index + 1U) + 1U] = end;
    return 0;
}

int shadowspill_residency_next_span(
    const ShadowSpillResidency *residency,
    uint32_t alias,
    uint32_t *cursor,
    uint32_t *start,
    uint32_t *end
) {
    const ShadowSpillSpanList *list = &residency->aliases[alias];
    /* The first span ending at or after the cursor. */
    uint32_t low = 0U;
    uint32_t high = list->count;
    while (low < high) {
        const uint32_t middle = low + (high - low) / 2U;
        if (list->bounds[2U * middle + 1U] < *cursor) {
            low = middle + 1U;
        } else {
            high = middle;
        }
    }
    if (low == list->count) {
        *cursor = residency->boundary_count;
        return 0;
    }
    const uint32_t span_start = list->bounds[2U * low];
    *start = span_start > *cursor ? span_start : *cursor;
    *end = list->bounds[2U * low + 1U];
    *cursor = *end + 1U;
    return 1;
}

uint64_t shadowspill_residency_hash(
    const ShadowSpillResidency *residency, uint64_t hash, uint64_t prime
) {
    for (uint32_t alias = 0U; alias < residency->alias_count; ++alias) {
        const ShadowSpillSpanList *list = &residency->aliases[alias];
        hash = (hash ^ list->count) * prime;
        for (uint32_t index = 0U; index < 2U * list->count; ++index) {
            hash = (hash ^ list->bounds[index]) * prime;
        }
    }
    return hash;
}
