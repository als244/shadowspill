/* Preparing the problem, then searching it. */
#include "internal.h"

static ShadowSpillStatus prepare_problem(
    const ShadowSpillIndexedProblem *source,
    const ShadowSpillPressureFitOptions *options,
    PreparedProblem *prepared
) {
    memset(prepared, 0, sizeof(*prepared));
    prepared->error_device = UINT32_MAX;
    prepared->error_alias = UINT32_MAX;
    prepared->error_boundary = INT32_MIN;
    const ShadowSpillSimulationProgram *program = source->context.simulation;
    if (shadowspill_problem_allocate_prepared_buffers(program, prepared) != 0) {
        return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    }
    if (shadowspill_problem_bind_residency(program->alias_count, program->initial_count,
                       program->initial_aliases, program->initial_locations,
                       prepared->initial_location) != 0 ||
        shadowspill_problem_bind_residency(program->alias_count, program->final_count,
                       program->final_aliases, program->final_locations,
                       prepared->final_location) != 0) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    ShadowSpillStatus status = shadowspill_problem_derive_task_facts(program, prepared);
    if (status != SHADOWSPILL_STATUS_OK) {
        return status;
    }
    status = shadowspill_problem_build_sparse_lists(program, prepared);
    if (status != SHADOWSPILL_STATUS_OK) {
        return status;
    }
    if (source->context.admission != NULL) {
        if (program->device_count != 1U ||
            source->context.admission->object_capacity_bytes == 0U ||
            source->context.admission->object_capacity_bytes >
                source->context.admission->pool_capacity_bytes) {
            return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
        }
        prepared->device_capacity_bytes[0] =
            source->context.admission->object_capacity_bytes;
    }
    status = shadowspill_problem_finalize_alias_facts(program, prepared);
    if (status != SHADOWSPILL_STATUS_OK) {
        return status;
    }
    status = shadowspill_problem_reserve_resident_slice(source, options, prepared);
    if (status != SHADOWSPILL_STATUS_OK) {
        return status;
    }
    status = shadowspill_problem_finalize_boundary_capacities(program, prepared);
    if (status != SHADOWSPILL_STATUS_OK) {
        return status;
    }
    status = shadowspill_problem_validate_required_floor(program, prepared);
    if (status != SHADOWSPILL_STATUS_OK) {
        return status;
    }
    prepared->residency = (ShadowSpillPressureFitResidencyProblem){
        .abi_version = SHADOWSPILL_ABI_VERSION,
        .alias_count = program->alias_count,
        .boundary_count = program->task_count + 1U,
        .device_count = program->device_count,
        .alias_size_bytes = program->alias_size_bytes,
        .alias_device = program->alias_device,
        .alias_retain_spill_copy = program->alias_retain_spill_copy,
        .initial_location = prepared->initial_location,
        .final_location = prepared->final_location,
        .anchors = prepared->anchors,
        .productions = prepared->productions,
        .latest_access_task = prepared->latest_access_task,
        .anchor_offsets = prepared->sparse.anchor_offsets,
        .anchor_positions = prepared->sparse.anchor_positions,
        .anchor_tasks = prepared->sparse.anchor_tasks,
        .reserved_offsets = prepared->sparse.reserved_offsets,
        .reserved_positions = prepared->sparse.reserved_positions,
        .alias_evict_eligible = prepared->evict_eligible,
        .fixed_fetch_trigger = prepared->fixed_fetch_trigger,
        .output_reservations = prepared->output_reservations,
        .write_prefix = prepared->write_prefix,
        .first_input_task = prepared->first_input_task,
        .fetch_runtime_ns = prepared->fetch_runtime_ns,
        .evict_runtime_ns = prepared->evict_runtime_ns,
        .task_ideal_end_ns = prepared->task_ideal_end_ns,
        .device_capacity_bytes = prepared->device_capacity_bytes,
        .boundary_capacity_bytes = prepared->boundary_capacity_bytes,
        .device_priority = source->device_priority,
    };

    shadowspill_problem_build_anchor_seed(program, prepared);
    if (options->initial_placement == SHADOWSPILL_PRESSUREFIT_INITIAL_PLACEMENT_GREEDY) {
        status = shadowspill_problem_greedily_place_initial_aliases(program, prepared);
        if (status != SHADOWSPILL_STATUS_OK) {
            return status;
        }
    }

    prepared->problem = (ShadowSpillPressureFitProblem){
        .abi_version = SHADOWSPILL_ABI_VERSION,
        .context = source->context,
        .residency = &prepared->residency,
        .seed_resident = prepared->seed_resident,
        .seed_breaks = prepared->seed_breaks,
        .incumbent = source->incumbent,
    };
    /* The same machine and names, over the resolved program this one plans. */
    prepared->problem.context.simulation = program;
    return SHADOWSPILL_STATUS_OK;
}

ShadowSpillStatus shadowspill_pressurefit_search(
    const ShadowSpillIndexedProblem *problems,
    uint32_t problem_count,
    const ShadowSpillPressureFitOptions *options,
    ShadowSpillPressureFitResult *results
) {
    if (problems == NULL || results == NULL || problem_count == 0U) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    for (uint32_t index = 0U; index < problem_count; ++index) {
        memset(&results[index], 0, sizeof(results[index]));
        results[index].selected_candidate_index = SHADOWSPILL_PLANNER_NO_INDEX;
        results[index].status = SHADOWSPILL_STATUS_INVALID_ARGUMENT;
        if (!shadowspill_problem_program_problem_valid(&problems[index], options)) {
            return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
        }
    }
    PreparedProblem *prepared = calloc(problem_count, sizeof(*prepared));
    ShadowSpillPressureFitProblem *derived =
        calloc(problem_count, sizeof(*derived));
    if (prepared == NULL || derived == NULL) {
        free(prepared);
        free(derived);
        return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    }

    /* Deriving the residency problems from their Programs, which the
     * evaluation below never sees and so could never account for. A Program
     * that cannot be prepared is a fact about that Program alone: its result
     * says so, and the others are evaluated together as if it were absent, so
     * a resolution that does not fit at this capacity never silences the ones
     * that do. */
    uint32_t *evaluated = calloc(problem_count, sizeof(*evaluated));
    if (evaluated == NULL) {
        free(prepared);
        free(derived);
        return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    }
    const uint64_t prepare_started = shadowspill_monotonic_ns();
    ShadowSpillStatus status = SHADOWSPILL_STATUS_OK;
    ShadowSpillStatus first_refusal = SHADOWSPILL_STATUS_OK;
    uint32_t evaluated_count = 0U;
    for (uint32_t index = 0U; index < problem_count; ++index) {
        ShadowSpillStatus prepared_status =
            prepare_problem(&problems[index], options, &prepared[index]);
        if (prepared_status == SHADOWSPILL_STATUS_OK) {
            derived[evaluated_count] = prepared[index].problem;
            evaluated[evaluated_count] = index;
            ++evaluated_count;
        } else {
            results[index].status = prepared_status;
            if (first_refusal == SHADOWSPILL_STATUS_OK) {
                first_refusal = prepared_status;
            }
        }
    }
    const uint64_t prepare_ns = shadowspill_monotonic_ns() - prepare_started;
    if (evaluated_count == problem_count) {
        status = shadowspill_pressurefit_evaluate_resolved(
            derived, problem_count, options, results
        );
    } else if (evaluated_count > 0U) {
        /* The evaluation takes its problems and results side by side, so the
         * prepared ones are evaluated in a compact batch and each result is
         * put back where its Program is; a result owns its buffers, and the
         * copy takes them with it. */
        ShadowSpillPressureFitResult *compact =
            calloc(evaluated_count, sizeof(*compact));
        if (compact == NULL) {
            status = SHADOWSPILL_STATUS_INTERNAL_FAILURE;
        } else {
            status = shadowspill_pressurefit_evaluate_resolved(
                derived, evaluated_count, options, compact
            );
            for (uint32_t slot = 0U; slot < evaluated_count; ++slot) {
                results[evaluated[slot]] = compact[slot];
            }
            free(compact);
        }
    } else {
        status = first_refusal;
    }
    for (uint32_t index = 0U; index < problem_count; ++index) {
        if (prepared[index].problem.residency == NULL) {
            continue;
        }
        results[index].evict_ineligible_aliases =
            prepared[index].evict_ineligible_aliases;
        results[index].evict_ineligible_bytes =
            prepared[index].evict_ineligible_bytes;
        /* The slice and the eligibility outlive preparation: the result owns
         * them now. */
        results[index].resident_slice_bytes = prepared[index].resident_slice_bytes;
        results[index].alias_evict_eligible = prepared[index].evict_eligible;
        prepared[index].resident_slice_bytes = NULL;
        prepared[index].evict_eligible = NULL;
    }
    const uint64_t teardown_started = shadowspill_monotonic_ns();
    for (uint32_t index = 0U; index < problem_count; ++index) {
        shadowspill_problem_prepared_problem_destroy(&prepared[index]);
    }
    free(evaluated);
    free(prepared);
    free(derived);
    const uint64_t teardown_ns = shadowspill_monotonic_ns() - teardown_started;

    /* Both spans sit outside the evaluation's own, so they extend the total
     * as well as their own sections, and the identity still holds. Preparing
     * and releasing are shared by the call, so each result reports the whole
     * span rather than a share of it. */
    for (uint32_t index = 0U; index < problem_count; ++index) {
        results[index].work.sections.prepare_ns += prepare_ns;
        results[index].work.sections.teardown_ns += teardown_ns;
        results[index].work.sections.total_ns += prepare_ns + teardown_ns;
    }
    return status;
}

ShadowSpillStatus shadowspill_pressurefit_preflight(
    const ShadowSpillIndexedProblem *problem,
    ShadowSpillPressureFitPreflightResult *result
) {
    if (result == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    memset(result, 0, sizeof(*result));
    result->status = SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    result->error_device = UINT32_MAX;
    result->error_alias = UINT32_MAX;
    result->error_boundary = INT32_MIN;
    const ShadowSpillPressureFitOptions options = {
        .initial_placement = SHADOWSPILL_PRESSUREFIT_INITIAL_PLACEMENT_REQUIRED,
    };
    if (!shadowspill_problem_program_problem_valid(problem, &options)) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }

    PreparedProblem prepared = {0};
    ShadowSpillStatus status = prepare_problem(problem, &options, &prepared);
    result->status = status;
    result->failure_kind = prepared.failure_kind;
    result->error_device = prepared.error_device;
    result->error_alias = prepared.error_alias;
    result->error_boundary = prepared.error_boundary;
    result->required_bytes = prepared.failure_required_bytes;
    result->capacity_bytes = prepared.failure_capacity_bytes;
    shadowspill_problem_prepared_problem_destroy(&prepared);
    return status;
}
