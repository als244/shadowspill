#include "internal.h"

#include <stdint.h>
#include <stdlib.h>

static int align_up(uint64_t value, uint64_t alignment, uint64_t *result) {
    uint64_t remainder = value % alignment;
    uint64_t addition = remainder == 0U ? 0U : alignment - remainder;
    if (addition > UINT64_MAX - value) {
        return -1;
    }
    *result = value + addition;
    return 0;
}

static uint64_t align_down(uint64_t value, uint64_t alignment) {
    return value - value % alignment;
}

static int allocate_from_range(
    ShadowSpillRangeAllocator *allocator,
    ShadowSpillRange *selected,
    ShadowSpillRange *previous,
    uint64_t aligned,
    uint64_t bytes,
    uint64_t *offset
) {
    uint64_t leading = aligned - selected->offset;
    uint64_t trailing = selected->bytes - leading - bytes;
    if (leading != 0U && trailing != 0U) {
        ShadowSpillRange *tail = calloc(1U, sizeof(*tail));
        if (tail == NULL) {
            return -1;
        }
        tail->offset = aligned + bytes;
        tail->bytes = trailing;
        tail->next = selected->next;
        selected->bytes = leading;
        selected->next = tail;
    } else if (leading != 0U) {
        selected->bytes = leading;
    } else if (trailing != 0U) {
        selected->offset = aligned + bytes;
        selected->bytes = trailing;
    } else if (previous == NULL) {
        allocator->free_ranges = selected->next;
        free(selected);
    } else {
        previous->next = selected->next;
        free(selected);
    }
    allocator->allocated += bytes;
    if (allocator->allocated > allocator->peak_allocated) {
        allocator->peak_allocated = allocator->allocated;
    }
    *offset = aligned;
    return 0;
}

int shadowspill_range_initialize(
    ShadowSpillRangeAllocator *allocator,
    uint64_t capacity
) {
    allocator->capacity = capacity;
    allocator->allocated = 0U;
    allocator->peak_allocated = 0U;
    allocator->free_ranges = NULL;
    if (capacity == 0U) {
        return 0;
    }
    allocator->free_ranges = calloc(1U, sizeof(*allocator->free_ranges));
    if (allocator->free_ranges == NULL) {
        return -1;
    }
    allocator->free_ranges->bytes = capacity;
    return 0;
}

int shadowspill_range_clone_extended(
    const ShadowSpillRangeAllocator *source,
    uint64_t capacity,
    ShadowSpillRangeAllocator *destination
) {
    if (source == NULL || destination == NULL || capacity < source->capacity) {
        return -1;
    }
    *destination = (ShadowSpillRangeAllocator){
        .capacity = capacity,
        .allocated = source->allocated,
        .peak_allocated = source->peak_allocated,
    };
    ShadowSpillRange **tail = &destination->free_ranges;
    for (const ShadowSpillRange *range = source->free_ranges; range != NULL;
         range = range->next) {
        ShadowSpillRange *copy = calloc(1U, sizeof(*copy));
        if (copy == NULL) {
            shadowspill_range_destroy(destination);
            return -1;
        }
        copy->offset = range->offset;
        copy->bytes = range->bytes;
        *tail = copy;
        tail = &copy->next;
    }
    if (capacity == source->capacity) {
        return 0;
    }
    uint64_t extension = capacity - source->capacity;
    ShadowSpillRange *last = destination->free_ranges;
    if (last != NULL) {
        while (last->next != NULL) {
            last = last->next;
        }
    }
    if (last != NULL && last->offset + last->bytes == source->capacity) {
        last->bytes += extension;
        return 0;
    }
    ShadowSpillRange *created = calloc(1U, sizeof(*created));
    if (created == NULL) {
        shadowspill_range_destroy(destination);
        return -1;
    }
    created->offset = source->capacity;
    created->bytes = extension;
    if (last == NULL) {
        destination->free_ranges = created;
    } else {
        last->next = created;
    }
    return 0;
}

void shadowspill_range_destroy(ShadowSpillRangeAllocator *allocator) {
    ShadowSpillRange *range = allocator->free_ranges;
    while (range != NULL) {
        ShadowSpillRange *next = range->next;
        free(range);
        range = next;
    }
    *allocator = (ShadowSpillRangeAllocator){0};
}

int shadowspill_range_allocate(
    ShadowSpillRangeAllocator *allocator,
    uint64_t bytes,
    uint64_t alignment,
    uint64_t *offset
) {
    ShadowSpillRange *previous = NULL;
    for (ShadowSpillRange *range = allocator->free_ranges; range != NULL;
         range = range->next) {
        uint64_t aligned;
        if (align_up(range->offset, alignment, &aligned) != 0 ||
            aligned < range->offset || aligned - range->offset > range->bytes) {
            previous = range;
            continue;
        }
        uint64_t leading = aligned - range->offset;
        if (bytes > range->bytes - leading) {
            previous = range;
            continue;
        }
        return allocate_from_range(
            allocator, range, previous, aligned, bytes, offset
        );
    }
    return 1;
}

int shadowspill_range_allocate_best_fit_low(
    ShadowSpillRangeAllocator *allocator,
    uint64_t bytes,
    uint64_t alignment,
    uint64_t *offset
) {
    ShadowSpillRange *selected = NULL;
    ShadowSpillRange *selected_previous = NULL;
    uint64_t selected_aligned = 0U;
    uint64_t selected_bytes = UINT64_MAX;
    ShadowSpillRange *previous = NULL;
    for (ShadowSpillRange *range = allocator->free_ranges; range != NULL;
         range = range->next) {
        uint64_t aligned;
        if (align_up(range->offset, alignment, &aligned) != 0 ||
            aligned < range->offset || aligned - range->offset > range->bytes) {
            previous = range;
            continue;
        }
        const uint64_t leading = aligned - range->offset;
        if (bytes > range->bytes - leading) {
            previous = range;
            continue;
        }
        if (range->bytes < selected_bytes ||
            (range->bytes == selected_bytes && aligned < selected_aligned)) {
            selected = range;
            selected_previous = previous;
            selected_aligned = aligned;
            selected_bytes = range->bytes;
        }
        previous = range;
    }
    if (selected == NULL) {
        return 1;
    }
    return allocate_from_range(
        allocator,
        selected,
        selected_previous,
        selected_aligned,
        bytes,
        offset
    );
}

int shadowspill_range_allocate_best_fit_high(
    ShadowSpillRangeAllocator *allocator,
    uint64_t bytes,
    uint64_t alignment,
    uint64_t *offset
) {
    ShadowSpillRange *selected = NULL;
    ShadowSpillRange *selected_previous = NULL;
    uint64_t selected_aligned = 0U;
    uint64_t selected_bytes = UINT64_MAX;
    ShadowSpillRange *previous = NULL;
    for (ShadowSpillRange *range = allocator->free_ranges; range != NULL;
         range = range->next) {
        uint64_t end = range->offset + range->bytes;
        if (bytes > range->bytes || end < range->offset) {
            previous = range;
            continue;
        }
        uint64_t aligned = align_down(end - bytes, alignment);
        if (aligned < range->offset) {
            previous = range;
            continue;
        }
        if (range->bytes < selected_bytes ||
            (range->bytes == selected_bytes && aligned > selected_aligned)) {
            selected = range;
            selected_previous = previous;
            selected_aligned = aligned;
            selected_bytes = range->bytes;
        }
        previous = range;
    }
    if (selected == NULL) {
        return 1;
    }
    return allocate_from_range(
        allocator,
        selected,
        selected_previous,
        selected_aligned,
        bytes,
        offset
    );
}

int shadowspill_range_free(
    ShadowSpillRangeAllocator *allocator,
    uint64_t offset,
    uint64_t bytes
) {
    if (bytes == 0U) {
        return 0;
    }
    ShadowSpillRange *node = calloc(1U, sizeof(*node));
    if (node == NULL) {
        return -1;
    }
    node->offset = offset;
    node->bytes = bytes;
    ShadowSpillRange *previous = NULL;
    ShadowSpillRange *current = allocator->free_ranges;
    while (current != NULL && current->offset < offset) {
        previous = current;
        current = current->next;
    }
    node->next = current;
    if (previous == NULL) {
        allocator->free_ranges = node;
    } else {
        previous->next = node;
    }
    if (node->next != NULL && node->offset + node->bytes == node->next->offset) {
        ShadowSpillRange *next = node->next;
        node->bytes += next->bytes;
        node->next = next->next;
        free(next);
    }
    if (previous != NULL && previous->offset + previous->bytes == node->offset) {
        previous->bytes += node->bytes;
        previous->next = node->next;
        free(node);
    }
    allocator->allocated -= bytes;
    return 0;
}

uint64_t shadowspill_range_free_bytes(
    const ShadowSpillRangeAllocator *allocator
) {
    return allocator->capacity - allocator->allocated;
}

uint64_t shadowspill_range_largest_free(
    const ShadowSpillRangeAllocator *allocator
) {
    uint64_t largest = 0U;
    for (const ShadowSpillRange *range = allocator->free_ranges; range != NULL;
         range = range->next) {
        if (range->bytes > largest) {
            largest = range->bytes;
        }
    }
    return largest;
}
