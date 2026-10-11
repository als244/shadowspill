/* The facts a schedule is emitted against: the problem, validated once. */
#include "internal.h"

uint64_t shadowspill_schedule_cell(uint32_t alias, uint32_t count, uint32_t index) {
    return (uint64_t)alias * count + index;
}

int shadowspill_schedule_checked_cells(uint32_t left, uint32_t right, size_t *result) {
    if (left != 0U && (size_t)right > SIZE_MAX / left) {
        return -1;
    }
    *result = (size_t)left * right;
    return 0;
}

int shadowspill_schedule_facts_create(
    const ShadowSpillPressureFitProblem *problem,
    ShadowSpillScheduleFacts *facts
) {
    if (problem == NULL || facts == NULL || problem->residency == NULL ||
        problem->context.simulation == NULL ||
        problem->abi_version != SHADOWSPILL_ABI_VERSION ||
        problem->residency->abi_version != SHADOWSPILL_ABI_VERSION ||
        problem->context.simulation->abi_version != SHADOWSPILL_ABI_VERSION ||
        problem->residency->alias_count != problem->context.simulation->alias_count ||
        problem->residency->device_count != problem->context.simulation->device_count ||
        problem->residency->boundary_count !=
            problem->context.simulation->task_count + 1U) {
        return -1;
    }
    memset(facts, 0, sizeof(*facts));
    facts->problem = problem;
    facts->alias_count = problem->residency->alias_count;
    facts->task_count = problem->context.simulation->task_count;
    facts->boundary_count = problem->residency->boundary_count;
    facts->device_count = problem->residency->device_count;
    return 0;
}

void shadowspill_schedule_facts_destroy(ShadowSpillScheduleFacts *facts) {
    if (facts == NULL) {
        return;
    }
    memset(facts, 0, sizeof(*facts));
}
