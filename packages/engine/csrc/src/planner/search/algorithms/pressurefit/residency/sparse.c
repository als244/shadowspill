/* The per-alias lists a prepared problem owns. */
#include "internal.h"

void shadowspill_residency_sparse_lists_destroy(ShadowSpillPressureFitResidencySparseLists *lists) {
    free(lists->anchor_offsets);
    free(lists->anchor_positions);
    free(lists->anchor_tasks);
    free(lists->anchor_earliest);
    free(lists->anchor_flags);
    free(lists->reserved_offsets);
    free(lists->reserved_positions);
    free(lists->write_offsets);
    free(lists->write_boundaries);
    memset(lists, 0, sizeof(*lists));
}
