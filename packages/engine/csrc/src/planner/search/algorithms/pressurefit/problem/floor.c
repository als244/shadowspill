/* The residency no plan may go below, and what it costs. */
#include "internal.h"

ShadowSpillStatus shadowspill_problem_validate_required_floor(
    const ShadowSpillSimulationProgram *program,
    PreparedProblem *prepared
) {
    uint32_t boundary_count = program->task_count + 1U;
    const ShadowSpillPressureFitResidencySparseLists *lists = &prepared->sparse;
    const uint32_t anchor_total = lists->anchor_offsets[program->alias_count];
    /* Which anchors the floor charged, so a reservation at a charged anchor's
     * boundary is not charged again. */
    uint8_t *charged = calloc(anchor_total == 0U ? 1U : anchor_total, 1U);
    if (charged == NULL) {
        return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    }
    /* The aliases the reducer may not cut live in the resident slice, which
     * the capacity below already excludes, so they are not in the floor. */
    for (uint32_t alias = 0U; alias < program->alias_count; ++alias) {
        uint32_t device = program->alias_device[alias];
        uint64_t size = program->alias_size_bytes[alias];
        if (prepared->evict_eligible[alias] == 0U) {
            continue;
        }
        for (uint32_t anchor = lists->anchor_offsets[alias];
             anchor < lists->anchor_offsets[alias + 1U];
             ++anchor) {
            const uint32_t position = lists->anchor_positions[anchor];
            const uint32_t latest = lists->anchor_tasks[anchor];
            int contributes = position == 0U ||
                prepared->final_location[alias] == SHADOWSPILL_MEMORY_DEVICE ||
                (latest != UINT32_MAX && latest > position - 1U);
            if (!contributes) {
                continue;
            }
            charged[anchor] = 1U;
            size_t pressure = (size_t)device * boundary_count + position;
            if (shadowspill_problem_add_u64(prepared->required_bytes[pressure], size,
                        &prepared->required_bytes[pressure]) != 0) {
                free(charged);
                return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
            }
        }
    }
    for (uint32_t alias = 0U; alias < program->alias_count; ++alias) {
        uint32_t device = program->alias_device[alias];
        if (prepared->evict_eligible[alias] == 0U) {
            continue;
        }
        for (uint32_t reserved = lists->reserved_offsets[alias];
             reserved < lists->reserved_offsets[alias + 1U];
             ++reserved) {
            const uint32_t task = lists->reserved_positions[reserved];
            const uint32_t last = lists->anchor_offsets[alias + 1U];
            const uint32_t anchor = shadowspill_anchor_lower_bound(
                lists->anchor_positions, lists->anchor_offsets[alias], last, task
            );
            if (anchor < last && lists->anchor_positions[anchor] == task &&
                charged[anchor] != 0U) {
                continue;
            }
            size_t pressure = (size_t)device * boundary_count + task;
            if (shadowspill_problem_add_u64(
                    prepared->required_bytes[pressure],
                    program->alias_size_bytes[alias],
                    &prepared->required_bytes[pressure]
                ) != 0) {
                free(charged);
                return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
            }
        }
    }
    free(charged);
    for (uint32_t device = 0U; device < program->device_count; ++device) {
        for (uint32_t position = 0U; position < boundary_count; ++position) {
            uint64_t required = prepared->required_bytes[
                (size_t)device * boundary_count + position
            ];
            if (required > prepared->boundary_capacity_bytes[
                    (uint64_t)device * boundary_count + position
                ]) {
                prepared->failure_kind =
                    SHADOWSPILL_PRESSUREFIT_PREFLIGHT_REQUIRED_CAPACITY;
                prepared->error_device = device;
                prepared->error_boundary = (int32_t)position;
                prepared->failure_required_bytes = required;
                prepared->failure_capacity_bytes =
                    prepared->boundary_capacity_bytes[
                        (uint64_t)device * boundary_count + position
                    ];
                return SHADOWSPILL_STATUS_ANALYTIC_INFEASIBLE;
            }
        }
    }
    return SHADOWSPILL_STATUS_OK;
}

/*
 * The resident slice: a static home for every lease of an alias the reducer
 * may not cut.
 *
 * Below the threshold an object is not worth a round trip -- its copies are
 * latency-bound and its bytes hardly relieve a boundary -- so it stays
 * resident from first to last access. Such leases are small and long-lived,
 * the worst shape for a layout whose other leases come and go, and too small
 * for packing them, among the rest or on their own, to be worth what it
 * costs in holes. So each gets a home of its own in a slice at the end of the
 * fixed range: one lease per alias, plus one for every task that mutates it
 * in place, since that task holds both generations. The slice is their sum,
 * each rounded up to the alignment. The device capacity handed to the reducer
 * loses the slice, so the search never charges these aliases again; every
 * placement lays the slice out from the leases the plan actually has; and
 * the emitter fetches each alias at the trigger chosen here, as late as the
 * ideal timeline allows.
 */
ShadowSpillStatus shadowspill_problem_reserve_resident_slice(
    const ShadowSpillIndexedProblem *source,
    const ShadowSpillPressureFitOptions *options,
    PreparedProblem *prepared
) {
    const ShadowSpillSimulationProgram *program = source->context.simulation;
    const uint64_t threshold = options->minimum_object_bytes_evict_eligible;
    uint64_t alignment = 1U;
    if (source->context.placement != NULL && source->context.placement->minimum_alignment != 0U) {
        alignment = source->context.placement->minimum_alignment;
    } else if (source->context.admission != NULL &&
               source->context.admission->minimum_alignment != 0U) {
        alignment = source->context.admission->minimum_alignment;
    }
    const size_t aliases = program->alias_count == 0U ? 1U : program->alias_count;
    uint32_t *generations = malloc(aliases * sizeof(*generations));
    if (generations == NULL) {
        return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    }
    for (uint32_t alias = 0U; alias < program->alias_count; ++alias) {
        generations[alias] = 1U;
    }
    for (uint32_t index = 0U; index < program->mutation_count; ++index) {
        ++generations[program->mutation_aliases[index]];
    }
    ShadowSpillStatus status = SHADOWSPILL_STATUS_OK;
    for (uint32_t alias = 0U; alias < program->alias_count; ++alias) {
        const uint64_t size = program->alias_size_bytes[alias];
        if (threshold == 0U || size == 0U || size >= threshold) {
            continue;
        }
        const ShadowSpillPressureFitResidencySparseLists *lists = &prepared->sparse;
        const uint32_t anchor = lists->anchor_offsets[alias];
        if (anchor == lists->anchor_offsets[alias + 1U]) {
            continue; /* never accessed, so never given a lease */
        }
        const uint32_t first = lists->anchor_positions[anchor];
        if (program->alias_device[alias] >= program->device_count) {
            status = SHADOWSPILL_STATUS_INVALID_ARGUMENT;
            goto done;
        }
        prepared->evict_eligible[alias] = 0U;
        ++prepared->evict_ineligible_aliases;
        if (shadowspill_problem_add_u64(prepared->evict_ineligible_bytes, size,
                    &prepared->evict_ineligible_bytes) != 0) {
            status = SHADOWSPILL_STATUS_INVALID_ARGUMENT;
            goto done;
        }
        if (prepared->initial_location[alias] != SHADOWSPILL_MEMORY_DEVICE &&
            (lists->anchor_flags[anchor] & SHADOWSPILL_ANCHOR_PRODUCED) == 0U &&
            first > 0U) {
            /* Consumed before it is produced, so fetched: as late as the
             * ideal timeline allows, which is where the emitter puts it. */
            const uint64_t ideal_start = prepared->task_ideal_end_ns[first - 1U];
            const uint64_t transfer = prepared->fetch_runtime_ns[alias];
            const uint64_t target =
                ideal_start > transfer ? ideal_start - transfer : 0U;
            prepared->fixed_fetch_trigger[alias] = shadowspill_latest_safe_trigger(
                prepared->task_ideal_end_ns, 0U, first - 1U, target
            );
        }
        uint64_t *slice =
            &prepared->resident_slice_bytes[program->alias_device[alias]];
        for (uint32_t held = 0U; held < generations[alias]; ++held) {
            uint64_t home;
            if (shadowspill_resident_home(slice, size, alignment, &home) != 0) {
                status = SHADOWSPILL_STATUS_INVALID_ARGUMENT;
                goto done;
            }
        }
    }
    for (uint32_t device = 0U; device < program->device_count; ++device) {
        const uint64_t slice = prepared->resident_slice_bytes[device];
        if (slice > prepared->device_capacity_bytes[device]) {
            prepared->failure_kind = SHADOWSPILL_PRESSUREFIT_PREFLIGHT_RESIDENT_SLICE_CAPACITY;
            prepared->error_device = device;
            prepared->failure_required_bytes = slice;
            prepared->failure_capacity_bytes = prepared->device_capacity_bytes[device];
            status = SHADOWSPILL_STATUS_ANALYTIC_INFEASIBLE;
            goto done;
        }
        prepared->device_capacity_bytes[device] -= slice;
    }

done:
    free(generations);
    return status;
}
