/* Seed residency from declared state and task access anchors. */
#include "internal.h"

/* The seed is the hull of each alias's anchors: one span per accessed
 * alias, from its first anchor to its last. */
int shadowspill_problem_build_anchor_seed(
    const ShadowSpillSimulationProgram *program,
    PreparedProblem *prepared
) {
    if (shadowspill_residency_create(
            &prepared->seed, program->alias_count, program->task_count + 1U
        ) != 0) {
        return -1;
    }
    const ShadowSpillPressureFitResidencySparseLists *lists = &prepared->sparse;
    for (uint32_t alias = 0U; alias < program->alias_count; ++alias) {
        const uint32_t begin = lists->anchor_offsets[alias];
        const uint32_t end = lists->anchor_offsets[alias + 1U];
        if (begin != end &&
            shadowspill_residency_append_span(
                &prepared->seed,
                alias,
                lists->anchor_positions[begin],
                lists->anchor_positions[end - 1U]
            ) != 0) {
            return -1;
        }
    }
    return 0;
}
