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

static ShadowSpillRange *acquire_node(ShadowSpillRangeAllocator *allocator) {
    ShadowSpillRange *node = allocator->available_nodes;
    if (node != NULL) {
        allocator->available_nodes = node->next;
        *node = (ShadowSpillRange){0};
        return node;
    }
    if (allocator->node_storage != NULL) {
        return NULL;
    }
    node = calloc(1U, sizeof(*node));
    if (node != NULL) {
        ++allocator->node_capacity;
    }
    return node;
}

static void release_node(
    ShadowSpillRangeAllocator *allocator,
    ShadowSpillRange *node
) {
    if (node == NULL) {
        return;
    }
    *node = (ShadowSpillRange){.next = allocator->available_nodes};
    allocator->available_nodes = node;
}

static void initialize_node_arena(
    ShadowSpillRangeAllocator *allocator,
    ShadowSpillRange *nodes,
    uint64_t node_capacity
) {
    allocator->node_storage = nodes;
    allocator->node_capacity = node_capacity;
    allocator->available_nodes = NULL;
    for (uint64_t index = node_capacity; index > 0U; --index) {
        ShadowSpillRange *node = &nodes[index - 1U];
        *node = (ShadowSpillRange){.next = allocator->available_nodes};
        allocator->available_nodes = node;
    }
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
        ShadowSpillRange *tail = acquire_node(allocator);
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
        release_node(allocator, selected);
    } else {
        previous->next = selected->next;
        release_node(allocator, selected);
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
    if (allocator == NULL) {
        return -1;
    }
    *allocator = (ShadowSpillRangeAllocator){0};
    allocator->capacity = capacity;
    if (capacity == 0U) {
        return 0;
    }
    allocator->free_ranges = acquire_node(allocator);
    if (allocator->free_ranges == NULL) {
        return -1;
    }
    allocator->free_ranges->bytes = capacity;
    return 0;
}

int shadowspill_range_initialize_with_nodes(
    ShadowSpillRangeAllocator *allocator,
    uint64_t capacity,
    ShadowSpillRange *nodes,
    uint64_t node_capacity
) {
    if (allocator == NULL || nodes == NULL || node_capacity == 0U) {
        return -1;
    }
    *allocator = (ShadowSpillRangeAllocator){0};
    initialize_node_arena(allocator, nodes, node_capacity);
    allocator->capacity = capacity;
    if (capacity == 0U) {
        return 0;
    }
    allocator->free_ranges = acquire_node(allocator);
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
    if (allocator == NULL) {
        return;
    }
    if (allocator->node_storage == NULL) {
        ShadowSpillRange *range = allocator->free_ranges;
        while (range != NULL) {
            ShadowSpillRange *next = range->next;
            free(range);
            range = next;
        }
        range = allocator->available_nodes;
        while (range != NULL) {
            ShadowSpillRange *next = range->next;
            free(range);
            range = next;
        }
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

int shadowspill_range_allocate_highest(
    ShadowSpillRangeAllocator *allocator,
    uint64_t bytes,
    uint64_t alignment,
    uint64_t minimum_offset,
    uint64_t *offset
) {
    ShadowSpillRange *selected = NULL;
    ShadowSpillRange *selected_previous = NULL;
    uint64_t selected_aligned = 0U;
    ShadowSpillRange *previous = NULL;
    for (ShadowSpillRange *range = allocator->free_ranges; range != NULL;
         range = range->next) {
        const uint64_t end = range->offset + range->bytes;
        const uint64_t start = range->offset > minimum_offset
            ? range->offset
            : minimum_offset;
        if (end < range->offset || start > end || bytes > end - start) {
            previous = range;
            continue;
        }
        const uint64_t aligned = align_down(end - bytes, alignment);
        if (aligned < start) {
            previous = range;
            continue;
        }
        if (selected == NULL || aligned > selected_aligned) {
            selected = range;
            selected_previous = previous;
            selected_aligned = aligned;
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

int shadowspill_range_allocate_at(
    ShadowSpillRangeAllocator *allocator,
    uint64_t offset,
    uint64_t bytes
) {
    if (allocator == NULL || bytes == 0U ||
        offset > allocator->capacity || bytes > allocator->capacity - offset) {
        return -1;
    }
    ShadowSpillRange *previous = NULL;
    for (ShadowSpillRange *range = allocator->free_ranges; range != NULL;
         range = range->next) {
        if (offset < range->offset) {
            return 1;
        }
        const uint64_t leading = offset - range->offset;
        if (leading <= range->bytes && bytes <= range->bytes - leading) {
            return allocate_from_range(
                allocator, range, previous, offset, bytes, &(uint64_t){0}
            );
        }
        previous = range;
    }
    return 1;
}

int shadowspill_range_free(
    ShadowSpillRangeAllocator *allocator,
    uint64_t offset,
    uint64_t bytes
) {
    if (bytes == 0U) {
        return 0;
    }
    ShadowSpillRange *node = acquire_node(allocator);
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
        release_node(allocator, next);
    }
    if (previous != NULL && previous->offset + previous->bytes == node->offset) {
        previous->bytes += node->bytes;
        previous->next = node->next;
        release_node(allocator, node);
    }
    allocator->allocated -= bytes;
    return 0;
}

uint64_t shadowspill_range_free_bytes(
    const ShadowSpillRangeAllocator *allocator
) {
    return allocator->capacity - allocator->allocated;
}

uint64_t shadowspill_range_free_prefix(
    const ShadowSpillRangeAllocator *allocator
) {
    const ShadowSpillRange *first = allocator->free_ranges;
    return first != NULL && first->offset == 0U ? first->bytes : 0U;
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
