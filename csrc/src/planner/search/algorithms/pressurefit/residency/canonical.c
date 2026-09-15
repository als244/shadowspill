/* One spelling per set of breaks, so two problems compare. */
#include "internal.h"

/* One row's canonical breaks: none on a cell that is not resident, and at
 * a span's last cell exactly when the alias is resident again later. This
 * never changes the span structure, which is why it can run once on the
 * seed and afterwards only on rows a cut touched. */
void shadowspill_residency_canonicalize_row(
    uint8_t *breaks,
    const uint8_t *resident,
    uint32_t alias,
    uint32_t boundary_count
) {
    const uint64_t row = (uint64_t)alias * boundary_count;
    const size_t packed_bytes = shadowspill_packed_cells(
        (uint64_t)(alias + 1U) * boundary_count
    );
    /* From the top of the row down: a cell keeps its break only while
     * resident; a span's last cell (resident, next not resident) takes a
     * break exactly when the alias is resident again later. */
    int later = 0;
    int next_resident = 0;
    uint64_t offset = row + boundary_count;
    while (offset > row) {
        const unsigned width = (unsigned)((offset - row) < 64U ? offset - row : 64U);
        offset -= width;
        const uint64_t present = shadowspill_cells_load(resident, packed_bytes, offset, width);
        const uint64_t broken = shadowspill_cells_load(breaks, packed_bytes, offset, width);
        uint64_t following = present >> 1U;
        if (next_resident) {
            following |= UINT64_C(1) << (width - 1U);
        }
        const uint64_t ends = present & ~following;
        uint64_t above = present >> 1U;
        above |= above >> 1U;
        above |= above >> 2U;
        above |= above >> 4U;
        above |= above >> 8U;
        above |= above >> 16U;
        above |= above >> 32U;
        if (later) {
            above = ~UINT64_C(0);
        }
        uint64_t updated = (broken & present & ~ends) | (ends & above);
        if (width < 64U) {
            updated &= (UINT64_C(1) << width) - 1U;
        }
        if (updated != broken) {
            shadowspill_cells_store(breaks, offset, width, updated);
        }
        later = later || present != 0U;
        next_resident = (int)(present & 1U);
    }
}

void shadowspill_canonicalize_breaks(
    uint8_t *breaks,
    const uint8_t *resident,
    uint32_t alias_count,
    uint32_t boundary_count
) {
    for (uint32_t alias = 0U; alias < alias_count; ++alias) {
        shadowspill_residency_canonicalize_row(breaks, resident, alias, boundary_count);
    }
}

int shadowspill_residency_valid_problem(
    const ShadowSpillPressureFitResidencyProblem *problem,
    const ShadowSpillPressureFitResidencyOptions *options,
    const ShadowSpillPressureFitResidencyResult *result
) {
    if (problem == NULL || options == NULL || result == NULL ||
        problem->abi_version != SHADOWSPILL_ABI_VERSION ||
        problem->boundary_count == 0U || problem->device_count == 0U) {
        return 0;
    }
    uint64_t cells = (uint64_t)problem->alias_count * problem->boundary_count;
    return problem->alias_size_bytes != NULL && problem->alias_device != NULL &&
        problem->alias_retain_spill_copy != NULL && problem->initial_location != NULL &&
        problem->final_location != NULL && problem->anchors != NULL &&
        problem->productions != NULL && problem->latest_access_task != NULL &&
        problem->output_reservations != NULL && problem->write_prefix != NULL &&
        problem->first_input_task != NULL && problem->fetch_runtime_ns != NULL &&
        problem->evict_runtime_ns != NULL && problem->task_ideal_end_ns != NULL &&
        problem->device_capacity_bytes != NULL &&
        problem->boundary_capacity_bytes != NULL &&
        problem->device_priority != NULL && options->seed_resident != NULL &&
        options->seed_breaks != NULL && options->extra_pressure_bytes != NULL &&
        result->resident != NULL && result->breaks != NULL &&
        result->resident_capacity >= cells && result->break_capacity >= cells;
}
