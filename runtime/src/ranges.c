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
    ShadowSpillRange *best = NULL;
    ShadowSpillRange *best_previous = NULL;
    uint64_t best_aligned = 0U;
    uint64_t best_waste = UINT64_MAX;
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
        uint64_t waste = range->bytes - leading - bytes;
        if (best == NULL || waste < best_waste ||
            (waste == best_waste && aligned < best_aligned)) {
            best = range;
            best_previous = previous;
            best_aligned = aligned;
            best_waste = waste;
        }
        previous = range;
    }
    if (best == NULL) {
        return 1;
    }

    uint64_t leading = best_aligned - best->offset;
    uint64_t trailing = best->bytes - leading - bytes;
    if (leading != 0U && trailing != 0U) {
        ShadowSpillRange *tail = calloc(1U, sizeof(*tail));
        if (tail == NULL) {
            return -1;
        }
        tail->offset = best_aligned + bytes;
        tail->bytes = trailing;
        tail->next = best->next;
        best->bytes = leading;
        best->next = tail;
    } else if (leading != 0U) {
        best->bytes = leading;
    } else if (trailing != 0U) {
        best->offset = best_aligned + bytes;
        best->bytes = trailing;
    } else if (best_previous == NULL) {
        allocator->free_ranges = best->next;
        free(best);
    } else {
        best_previous->next = best->next;
        free(best);
    }
    allocator->allocated += bytes;
    if (allocator->allocated > allocator->peak_allocated) {
        allocator->peak_allocated = allocator->allocated;
    }
    *offset = best_aligned;
    return 0;
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
