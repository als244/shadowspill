/* The sparse per-alias lists the reduction reads. */
#include "internal.h"

/* Per alias: its anchors (boundary order, with the latest task at each) and
 * its reserved boundaries, derived from the dense cell arrays. */
int shadowspill_residency_sparse_lists_build(
    const uint8_t *anchors,
    const uint32_t *latest_access_task,
    const uint8_t *output_reservations,
    uint32_t alias_count,
    uint32_t boundary_count,
    ShadowSpillPressureFitResidencySparseLists *lists
) {
    const uint64_t cells = (uint64_t)alias_count * boundary_count;
    uint64_t anchor_total = 0U;
    uint64_t reserved_total = 0U;
    for (uint64_t cell = 0U; cell < cells; ++cell) {
        anchor_total += anchors[cell] != 0U;
        reserved_total += output_reservations[cell] != 0U;
    }
    lists->anchor_offsets = malloc(((size_t)alias_count + 1U) * sizeof(uint32_t));
    lists->anchor_positions = malloc((anchor_total == 0U ? 1U : (size_t)anchor_total) * sizeof(uint32_t));
    lists->anchor_tasks = malloc((anchor_total == 0U ? 1U : (size_t)anchor_total) * sizeof(uint32_t));
    lists->reserved_offsets = malloc(((size_t)alias_count + 1U) * sizeof(uint32_t));
    lists->reserved_positions = malloc((reserved_total == 0U ? 1U : (size_t)reserved_total) * sizeof(uint32_t));
    if (lists->anchor_offsets == NULL || lists->anchor_positions == NULL ||
        lists->anchor_tasks == NULL || lists->reserved_offsets == NULL ||
        lists->reserved_positions == NULL) {
        shadowspill_residency_sparse_lists_destroy(lists);
        return -1;
    }
    uint32_t anchor_written = 0U;
    uint32_t reserved_written = 0U;
    for (uint32_t alias = 0U; alias < alias_count; ++alias) {
        lists->anchor_offsets[alias] = anchor_written;
        lists->reserved_offsets[alias] = reserved_written;
        const uint64_t row = (uint64_t)alias * boundary_count;
        for (uint32_t boundary = 0U; boundary < boundary_count; ++boundary) {
            if (anchors[row + boundary] != 0U) {
                lists->anchor_positions[anchor_written] = boundary;
                lists->anchor_tasks[anchor_written] = latest_access_task[row + boundary];
                ++anchor_written;
            }
            if (output_reservations[row + boundary] != 0U) {
                lists->reserved_positions[reserved_written++] = boundary;
            }
        }
    }
    lists->anchor_offsets[alias_count] = anchor_written;
    lists->reserved_offsets[alias_count] = reserved_written;
    return 0;
}

void shadowspill_residency_sparse_lists_destroy(ShadowSpillPressureFitResidencySparseLists *lists) {
    free(lists->anchor_offsets);
    free(lists->anchor_positions);
    free(lists->anchor_tasks);
    free(lists->reserved_offsets);
    free(lists->reserved_positions);
    memset(lists, 0, sizeof(*lists));
}
