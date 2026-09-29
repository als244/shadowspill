/* Seed residency from declared state and task access anchors. */
#include "internal.h"

void shadowspill_problem_build_anchor_seed(
    const ShadowSpillSimulationProgram *program,
    PreparedProblem *prepared
) {
    uint32_t boundary_count = program->task_count + 1U;
    for (uint32_t alias = 0U; alias < program->alias_count; ++alias) {
        uint32_t first = UINT32_MAX;
        uint32_t last = 0U;
        for (uint32_t position = 0U; position < boundary_count; ++position) {
            if (prepared->anchors[(size_t)alias * boundary_count + position] == 0U) {
                continue;
            }
            if (first == UINT32_MAX) {
                first = position;
            }
            last = position;
        }
        if (first == UINT32_MAX) {
            continue;
        }
        for (uint32_t position = first; position <= last; ++position) {
            prepared->seed_resident[
                (size_t)alias * boundary_count + position
            ] = 1U;
        }
    }
}
