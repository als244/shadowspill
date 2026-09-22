/* One residency turned into a plan: reduced, emitted, placed, simulated. */
#include "internal.h"

int shadowspill_candidate_simulate_schedule(
    const ShadowSpillPressureFitProblem *problem,
    const ShadowSpillIndexedSchedule *schedule,
    SimulationWorkspace *workspace,
    ShadowSpillCandidateAdmissionWorkspace *admission_workspace,
    ShadowSpillCapacityViolation *first_violation,
    ShadowSpillSimulationResult *result,
    ShadowSpillStatus *admission_status,
    ShadowSpillAdmissionReplayResult *admission_result
) {
    if (shadowspill_candidate_simulation_workspace_reserve_transfers(
            workspace,
            schedule->action_count
        ) != 0) {
        return -1;
    }
    ShadowSpillSimulationProgram program;
    *admission_status = SHADOWSPILL_STATUS_OK;
    memset(admission_result, 0, sizeof(*admission_result));
    if (problem->context.admission == NULL) {
        shadowspill_bind_indexed_schedule(problem->context.simulation, schedule, &program);
    } else {
        *admission_status = shadowspill_admit_indexed_schedule(
            &problem->context,
            schedule,
            admission_workspace,
            &program,
            admission_result
        );
        if (*admission_status == SHADOWSPILL_STATUS_REPLAY_INFEASIBLE) {
            memset(result, 0, sizeof(*result));
            return 0;
        }
        if (*admission_status != SHADOWSPILL_STATUS_OK) {
            return -1;
        }
    }
    *result = (ShadowSpillSimulationResult){
        .task_intervals = workspace->tasks,
        .task_interval_capacity = workspace->task_capacity,
        .transfer_intervals = workspace->transfers,
        .transfer_interval_capacity = workspace->transfer_capacity,
        .device_peaks = workspace->peaks,
        .device_peak_capacity = workspace->device_capacity,
        /* One slot: the count still reports the true total, and repair only
         * ever aims at the first place the plan came up short. */
        .capacity_violations = first_violation,
        .capacity_violation_capacity = 1U,
    };
    (void)shadowspill_simulate(&program, result);
    return 0;
}


/* Grow the placement buffers to hold one plan's operations and leases. */
int shadowspill_candidate_place_plan(
    const ShadowSpillPressureFitProblem *problem,
    CandidateWorkspace *workspace,
    const ShadowSpillSimulationResult *simulation,
    uint64_t *required_bytes
) {
    const ShadowSpillIndexedSchedule *schedule = &workspace->schedule.value;
    const ShadowSpillAdmissionFacts *admission = problem->context.placement;
    if (admission == NULL) {
        return -1;
    }
    /* A result the admission replay refused is zeroed, and a zeroed status
     * reads as OK, so a plan can arrive here with no intervals behind it.
     * Lease lifetimes are derived from those intervals, so there is nothing
     * to place. */
    if (simulation->task_intervals == NULL ||
        simulation->transfer_intervals == NULL ||
        simulation->makespan_ns == 0U) {
        return -1;
    }
    uint64_t operation_capacity = 0U;
    uint64_t lease_capacity = 0U;
    ShadowSpillStatus bounds_status = shadowspill_admission_operation_bounds(
        problem->context.simulation, admission, schedule,
        &operation_capacity, &lease_capacity
    );
    if (bounds_status != SHADOWSPILL_STATUS_OK) {
        return -1;
    }
    PlacementWorkspace *place = &workspace->placement;
    if (shadowspill_candidate_placement_reserve(
            place, operation_capacity, lease_capacity,
            problem->residency->alias_count,
            /* Flattened allocation steps, which is where the topology ends --
             * not the slot count, which counts something else. */
            admission->task_allocation_offsets[admission->task_count]
        ) != 0) {
        return -1;
    }
    place->operations = (ShadowSpillAdmissionOperations){
        .lease_ids = place->lease_ids,
        .dependency_ids = place->dependency_ids,
        .bytes = place->bytes,
        .alignments = place->alignments,
        .kinds = place->kinds,
        .purposes = place->purposes,
        .boundaries = place->boundaries,
        .indices = place->indices,
        .allocation_offsets = place->allocation_offsets,
        .operation_capacity = operation_capacity,
        .lease_aliases = place->lease_aliases,
        .lease_starts = place->lease_starts,
        .lease_retires = place->lease_retires,
        .lease_capacity = lease_capacity,
    };
    ShadowSpillStatus operations_status = shadowspill_build_admission_operations(
        problem->context.simulation, admission, schedule, &place->operations
    );
    if (operations_status != SHADOWSPILL_STATUS_OK) {
        return -1;
    }
    /* Aliases the plan leaves resident on the device hold their lease past the
     * step, so placement has to keep room for them. */
    uint32_t dynamic_count = 0U;
    for (uint32_t index = 0U; index < schedule->final_count; ++index) {
        if (schedule->final_locations[index] == SHADOWSPILL_MEMORY_DEVICE) {
            place->dynamic_aliases[dynamic_count++] = schedule->final_aliases[index];
        }
    }
    ShadowSpillLeaseLifetimeProblem lifetime_problem = {
        .abi_version = SHADOWSPILL_ABI_VERSION,
        .operations = &place->operations,
        .admission = admission,
        .schedule = schedule,
        .task_intervals = simulation->task_intervals,
        .task_interval_count = simulation->task_interval_count,
        .transfer_intervals = simulation->transfer_intervals,
        .transfer_interval_count = simulation->transfer_interval_count,
        .makespan_ns = simulation->makespan_ns,
        .dynamic_aliases = place->dynamic_aliases,
        .dynamic_alias_count = dynamic_count,
    };
    ShadowSpillLeaseLifetimeResult lifetime_result = {
        .lifetimes = place->lifetimes,
        .identities = place->identities,
        .allocation_step_leases = place->allocation_step_leases,
        .alias_leases = place->alias_leases,
    };
    ShadowSpillStatus lifetime_status =
        shadowspill_build_lease_lifetimes(&lifetime_problem, &lifetime_result);
    if (lifetime_status != SHADOWSPILL_STATUS_OK) {
        return -1;
    }
    /* Fixed leases occupy the prefix, so placement runs on it without a copy.
     * The leases of aliases the reducer may not cut are left out: they take
     * static homes in the resident slice, laid out after the rest below. */
    const uint8_t *eligible = problem->residency->alias_evict_eligible;
    for (uint64_t lease = 0U; lease < lifetime_result.fixed_count; ++lease) {
        const uint32_t alias = place->identities[lease].alias;
        place->excluded[lease] = eligible != NULL &&
            alias != SHADOWSPILL_PLANNER_NO_INDEX && eligible[alias] == 0U;
    }
    ShadowSpillPlacementProblem placement_problem = {
        .abi_version = SHADOWSPILL_ABI_VERSION,
        .lifetime_count = (uint32_t)lifetime_result.fixed_count,
        .lifetimes = place->lifetimes,
        .excluded = place->excluded,
    };
    ShadowSpillPlacementResult placement_result = {
        .required_bytes = 0U,
        .offsets = place->offsets,
    };
    ShadowSpillStatus place_status =
        shadowspill_place_lifetimes(&placement_problem, &placement_result);
    if (place_status != SHADOWSPILL_STATUS_OK) {
        return -1;
    }
    /* The lease whose end is the extent: the placer's span is the largest
     * offset plus size over the leases it placed, so one of them ends there. */
    place->placed_count = lifetime_result.fixed_count;
    place->extent_bytes = placement_result.required_bytes;
    place->extent_lease = SHADOWSPILL_ADMISSION_NO_LEASE;
    for (uint64_t lease = 0U; lease < lifetime_result.fixed_count; ++lease) {
        if (place->excluded[lease] == 0U &&
            place->offsets[lease] + place->lifetimes[lease].bytes ==
                placement_result.required_bytes) {
            place->extent_lease = lease;
            break;
        }
    }
    /* The resident slice follows the main assignment: each lease left out
     * takes the next aligned home, in lease order, and the fixed range ends
     * past the last of them. */
    uint64_t extent = placement_result.required_bytes;
    for (uint64_t lease = 0U; lease < lifetime_result.fixed_count; ++lease) {
        if (place->excluded[lease] != 0U &&
            shadowspill_resident_home(
                &extent,
                place->lifetimes[lease].bytes,
                place->lifetimes[lease].alignment,
                &place->offsets[lease]
            ) != 0) {
            return -1;
        }
    }
    /* What the pool has to hold is the fixed range plus the leases that
     * outlive the step, which are placed outside it. The certificate adds the
     * same two, so measuring only the range reports a plan as fitting that
     * the certificate then refuses. */
    uint64_t dynamic_bytes = 0U;
    for (uint64_t lease = lifetime_result.fixed_count;
         lease < lifetime_result.lifetime_count;
         ++lease) {
        if (place->lifetimes[lease].bytes >
            UINT64_MAX - dynamic_bytes) {
            return -1;
        }
        dynamic_bytes += place->lifetimes[lease].bytes;
    }
    if (extent > UINT64_MAX - dynamic_bytes) {
        return -1;
    }
    *required_bytes = extent + dynamic_bytes;
    return 0;
}

int shadowspill_candidate_record_fetch_constraint(
    CandidateWorkspace *workspace,
    ShadowSpillFetchTriggerConstraint incoming
) {
    for (uint32_t index = 0U; index < workspace->fetch_constraint_count;
         ++index) {
        ShadowSpillFetchTriggerConstraint *current =
            &workspace->fetch_constraints[index];
        if (current->alias != incoming.alias ||
            current->consumer_task != incoming.consumer_task) {
            continue;
        }
        uint32_t minimum = current->minimum_trigger > incoming.minimum_trigger
            ? current->minimum_trigger
            : incoming.minimum_trigger;
        uint32_t maximum = current->maximum_trigger < incoming.maximum_trigger
            ? current->maximum_trigger
            : incoming.maximum_trigger;
        if (minimum > maximum) {
            return 1;
        }
        current->minimum_trigger = minimum;
        current->maximum_trigger = maximum;
        return 0;
    }
    if (workspace->fetch_constraint_count ==
        workspace->fetch_constraint_capacity) {
        uint32_t capacity = workspace->fetch_constraint_capacity == 0U
            ? 8U
            : workspace->fetch_constraint_capacity * 2U;
        if (capacity < workspace->fetch_constraint_capacity) {
            return -1;
        }
        ShadowSpillFetchTriggerConstraint *constraints = realloc(
            workspace->fetch_constraints,
            (size_t)capacity * sizeof(*constraints)
        );
        if (constraints == NULL) {
            return -1;
        }
        workspace->fetch_constraints = constraints;
        workspace->fetch_constraint_capacity = capacity;
    }
    workspace->fetch_constraints[workspace->fetch_constraint_count++] =
        incoming;
    return 0;
}


void shadowspill_candidate_residency_options(
    CandidateWorkspace *workspace,
    uint8_t strategy,
    ShadowSpillPressureFitResidencyOptions *options
) {
    *options = (ShadowSpillPressureFitResidencyOptions){
        .minimize_transfer =
            strategy == SHADOWSPILL_PRESSUREFIT_RESIDENCY_HEADROOM_TRANSFER ||
                strategy == SHADOWSPILL_PRESSUREFIT_RESIDENCY_TIGHT_TRANSFER
            ? 1U
            : 0U,
        .fetch_headroom =
            strategy == SHADOWSPILL_PRESSUREFIT_RESIDENCY_HEADROOM_STALL ||
                strategy == SHADOWSPILL_PRESSUREFIT_RESIDENCY_HEADROOM_TRANSFER
            ? 1U
            : 0U,
        .seed_resident = workspace->packed_seed_resident,
        .seed_breaks = workspace->packed_seed_breaks,
        .extra_pressure_bytes = workspace->extra_pressure,
    };
}

static ShadowSpillStatus reduce(
    const ShadowSpillPressureFitProblem *problem,
    CandidateWorkspace *workspace,
    const ShadowSpillPressureFitResidencyOptions *options,
    uint8_t *resident,
    uint8_t *breaks,
    ShadowSpillPressureFitResidencyResult *result
) {
    uint64_t cells = (uint64_t)problem->residency->alias_count *
        problem->residency->boundary_count;
    *result = (ShadowSpillPressureFitResidencyResult){
        .resident = resident,
        .resident_capacity = cells,
        .breaks = breaks,
        .break_capacity = cells,
        /* Records what this reduction gives up, when anyone is recording.
         * Several reductions can run before the record is drained, so each
         * appends after the last rather than from the start. A full scratch
         * records nothing further: the reducer stops at its capacity. */
        .cut_aliases = workspace->cut_scratch == NULL
            ? NULL
            : workspace->cut_scratch + workspace->cut_scratch_count,
        .cut_capacity =
            workspace->cut_scratch_capacity - workspace->cut_scratch_count,
    };
    const ShadowSpillStatus reduced = shadowspill_pressurefit_reduce_residency_reusing(
        problem->residency,
        options,
        result,
        workspace->residency_workspace
    );
    workspace->cut_scratch_count += result->cut_count;
    return reduced;
}

/* Reduce a residency and name it.
 *
 * Reductions are not cached. A repair trajectory cuts aliases monotonically,
 * so it never revisits a residency, and the only reuse a cache ever found
 * was a strategy's base residency across its rule variants -- half a percent
 * of reductions when measured, against two packed bitmaps retained per
 * visited residency. The fingerprint is what other stages key on. */
ShadowSpillStatus shadowspill_candidate_reduce_residency(
    const ShadowSpillPressureFitProblem *problem,
    CandidateWorkspace *workspace,
    const ShadowSpillPressureFitResidencyOptions *options,
    uint8_t strategy,
    uint8_t *resident,
    uint8_t *breaks,
    ShadowSpillPressureFitResidencyResult *result
) {
    (void)strategy;
    ShadowSpillPressureFitResidencyResult computed;
    ShadowSpillStatus status = reduce(
        problem,
        workspace,
        options,
        resident,
        breaks,
        &computed
    );
    workspace->current_residency = (Fingerprint){
        .low = shadowspill_candidate_hash_bytes(
            shadowspill_candidate_hash_bytes(
                UINT64_C(1469598103934665603),
                resident,
                workspace->packed_cell_count
            ),
            breaks,
            workspace->packed_cell_count
        ),
        .high = shadowspill_candidate_hash_bytes_high(
            shadowspill_candidate_hash_bytes_high(
                UINT64_C(1099511628211),
                resident,
                workspace->packed_cell_count
            ),
            breaks,
            workspace->packed_cell_count
        ),
    };
    *result = (ShadowSpillPressureFitResidencyResult){
        .status = (uint32_t)status,
        .error_device = computed.error_device,
        .error_boundary = computed.error_boundary,
        .required_bytes = computed.required_bytes,
        .capacity_bytes = computed.capacity_bytes,
        .resident = resident,
        .resident_capacity = workspace->cell_count,
        .breaks = breaks,
        .break_capacity = workspace->cell_count,
    };
    return status;
}

