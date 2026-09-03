#include <stdint.h>
#include <stdlib.h>

#include <shadowspill/planner.h>

static ShadowSpillSimulationProgram make_program(
    uint64_t runtime_ns,
    const ShadowSpillSimulationDevice *device,
    const uint32_t *task_device,
    const uint8_t *task_kind,
    const uint32_t *task_lane,
    const uint64_t *task_workspace,
    const uint32_t *empty_offsets
) {
    static uint64_t runtimes[2];
    static uint32_t next_runtime = 0U;
    uint32_t slot = next_runtime++;
    runtimes[slot] = runtime_ns;
    return (ShadowSpillSimulationProgram){
        .abi_version = SHADOWSPILL_ABI_VERSION,
        .device_count = 1U,
        .task_count = 1U,
        .spill_capacity_bytes = 1U,
        .devices = device,
        .task_device = task_device,
        .task_resource_kind = task_kind,
        .task_resource_lane = task_lane,
        .task_runtime_ns = &runtimes[slot],
        .task_workspace_bytes = task_workspace,
        .dependency_offsets = empty_offsets,
        .input_offsets = empty_offsets,
        .output_offsets = empty_offsets,
        .mutation_offsets = empty_offsets,
    };
}

int main(void) {
    const ShadowSpillSimulationDevice device = {
        .capacity_bytes = 1U,
        .fetch_bandwidth_bytes_per_second = 1U,
        .evict_bandwidth_bytes_per_second = 1U,
    };
    const uint32_t task_device[] = {0U};
    const uint8_t task_kind[] = {0U};
    const uint32_t task_lane[] = {0U};
    const uint64_t task_workspace[] = {0U};
    const uint32_t empty_offsets[] = {0U, 0U};
    ShadowSpillSimulationProgram slower = make_program(
        200U,
        &device,
        task_device,
        task_kind,
        task_lane,
        task_workspace,
        empty_offsets
    );
    ShadowSpillSimulationProgram faster = make_program(
        100U,
        &device,
        task_device,
        task_kind,
        task_lane,
        task_workspace,
        empty_offsets
    );
    const ShadowSpillPlanCandidate candidates[] = {
        {.program = &slower, .candidate_id = 10U, .selection_id = 20U},
        {.program = &faster, .candidate_id = 11U, .selection_id = 21U},
    };
    ShadowSpillCandidateResult candidate_results[2] = {{0}};
    ShadowSpillPlanSelectionResult result = {
        .candidate_results = candidate_results,
        .candidate_result_capacity = 2U,
    };

    if (shadowspill_abi_version() != SHADOWSPILL_ABI_VERSION) {
        return EXIT_FAILURE;
    }
    if (shadowspill_select_plan(candidates, 2U, &result) !=
        SHADOWSPILL_STATUS_OK) {
        return EXIT_FAILURE;
    }
    if (result.selected_index != 1U || result.selected_candidate_id != 11U ||
        result.selected_selection_id != 21U ||
        result.selected_makespan_ns != 100U ||
        result.valid_candidate_count != 2U ||
        result.candidate_result_count != 2U ||
        candidate_results[0].makespan_ns != 200U ||
        candidate_results[1].makespan_ns != 100U) {
        return EXIT_FAILURE;
    }

    const uint64_t alias_sizes[] = {64U, 64U};
    const uint32_t alias_devices[] = {0U, 0U};
    const uint8_t retain_spill[] = {1U, 1U};
    const int8_t initial_locations[] = {0, 1};
    const int8_t final_locations[] = {-1, -1};
    const uint8_t anchors[] = {
        1U, 0U, 1U,
        0U, 1U, 0U,
    };
    const uint8_t empty_cells[] = {
        0U, 0U, 0U,
        0U, 0U, 0U,
    };
    const uint32_t latest_access[] = {
        SHADOWSPILL_PLANNER_NO_INDEX,
        SHADOWSPILL_PLANNER_NO_INDEX,
        SHADOWSPILL_PLANNER_NO_INDEX,
        SHADOWSPILL_PLANNER_NO_INDEX,
        1U,
        SHADOWSPILL_PLANNER_NO_INDEX,
    };
    const uint32_t first_input_tasks[] = {0U, 1U};
    const uint64_t transfer_runtimes[] = {1U, 1U};
    const uint64_t task_ends[] = {10U, 20U};
    const uint64_t capacities[] = {64U};
    const uint64_t boundary_capacities[] = {64U, 64U, 64U};
    const uint32_t priorities[] = {0U};
    const uint8_t seed_resident[] = {
        1U, 1U, 1U,
        0U, 1U, 0U,
    };
    const uint8_t seed_breaks[] = {0U, 0U, 0U, 0U, 0U, 0U};
    const uint64_t extra_pressure[] = {0U, 0U, 0U};
    uint8_t reduced_resident[6] = {0};
    uint8_t reduced_breaks[6] = {0};
    const ShadowSpillResidencyProblem residency_problem = {
        .abi_version = SHADOWSPILL_ABI_VERSION,
        .alias_count = 2U,
        .boundary_count = 3U,
        .device_count = 1U,
        .alias_size_bytes = alias_sizes,
        .alias_device = alias_devices,
        .alias_retain_spill_copy = retain_spill,
        .initial_location = initial_locations,
        .final_location = final_locations,
        .anchors = anchors,
        .productions = empty_cells,
        .latest_access_task = latest_access,
        .output_reservations = empty_cells,
        .write_prefix = empty_cells,
        .first_input_task = first_input_tasks,
        .fetch_runtime_ns = transfer_runtimes,
        .evict_runtime_ns = transfer_runtimes,
        .task_ideal_end_ns = task_ends,
        .device_capacity_bytes = capacities,
        .boundary_capacity_bytes = boundary_capacities,
        .device_priority = priorities,
    };
    const ShadowSpillResidencyOptions residency_options = {
        .seed_resident = seed_resident,
        .seed_breaks = seed_breaks,
        .extra_pressure_bytes = extra_pressure,
    };
    ShadowSpillResidencyResult residency_result = {
        .resident = reduced_resident,
        .resident_capacity = 6U,
        .breaks = reduced_breaks,
        .break_capacity = 6U,
    };
    if (shadowspill_reduce_residency(
            &residency_problem,
            &residency_options,
            &residency_result
        ) != SHADOWSPILL_STATUS_OK) {
        return EXIT_FAILURE;
    }
    const uint8_t expected_resident[] = {
        1U, 0U, 1U,
        0U, 1U, 0U,
    };
    const uint8_t expected_breaks[] = {
        1U, 0U, 0U,
        0U, 0U, 0U,
    };
    for (uint32_t index = 0U; index < 6U; ++index) {
        if (reduced_resident[index] != expected_resident[index] ||
            reduced_breaks[index] != expected_breaks[index]) {
            return EXIT_FAILURE;
        }
    }

    const uint64_t problem_alias_size[] = {32U};
    const ShadowSpillSimulationDevice problem_device = {
        .capacity_bytes = 64U,
        .fetch_bandwidth_bytes_per_second = 1U,
        .evict_bandwidth_bytes_per_second = 1U,
    };
    const uint32_t problem_alias_device[] = {0U};
    const uint8_t problem_retain_spill[] = {1U};
    const int8_t problem_initial_location[] = {0};
    const int8_t problem_final_location[] = {-1};
    const uint8_t problem_anchors[] = {1U, 0U};
    const uint8_t problem_zero_cells[] = {0U, 0U};
    const uint32_t problem_latest_access[] = {
        0U,
        SHADOWSPILL_PLANNER_NO_INDEX,
    };
    const uint32_t problem_first_input[] = {0U};
    const uint64_t problem_transfer_runtime[] = {1U};
    const uint64_t problem_task_end[] = {10U};
    const uint64_t problem_capacity[] = {64U};
    const uint64_t problem_boundary_capacity[] = {64U, 64U};
    const uint32_t problem_priority[] = {0U};
    const uint8_t problem_seed_resident[] = {1U, 0U};
    const uint8_t problem_seed_breaks[] = {0U, 0U};
    const ShadowSpillResidencyProblem problem_residency = {
        .abi_version = SHADOWSPILL_ABI_VERSION,
        .alias_count = 1U,
        .boundary_count = 2U,
        .device_count = 1U,
        .alias_size_bytes = problem_alias_size,
        .alias_device = problem_alias_device,
        .alias_retain_spill_copy = problem_retain_spill,
        .initial_location = problem_initial_location,
        .final_location = problem_final_location,
        .anchors = problem_anchors,
        .productions = problem_zero_cells,
        .latest_access_task = problem_latest_access,
        .output_reservations = problem_zero_cells,
        .write_prefix = problem_zero_cells,
        .first_input_task = problem_first_input,
        .fetch_runtime_ns = problem_transfer_runtime,
        .evict_runtime_ns = problem_transfer_runtime,
        .task_ideal_end_ns = problem_task_end,
        .device_capacity_bytes = problem_capacity,
        .boundary_capacity_bytes = problem_boundary_capacity,
        .device_priority = problem_priority,
    };
    const uint64_t problem_alias_version[] = {0U};
    const uint64_t problem_task_runtime[] = {10U};
    const uint64_t problem_task_workspace[] = {0U};
    const uint32_t problem_input_offsets[] = {0U, 1U};
    const uint32_t problem_input_aliases[] = {0U};
    const uint32_t problem_initial_aliases[] = {0U};
    const uint8_t problem_initial_locations[] = {
        SHADOWSPILL_MEMORY_DEVICE,
    };
    const ShadowSpillSimulationProgram problem_simulation = {
        .abi_version = SHADOWSPILL_ABI_VERSION,
        .device_count = 1U,
        .alias_count = 1U,
        .task_count = 1U,
        .initial_count = 1U,
        .input_count = 1U,
        .spill_capacity_bytes = 64U,
        .devices = &problem_device,
        .alias_device = problem_alias_device,
        .alias_size_bytes = problem_alias_size,
        .alias_initial_version = problem_alias_version,
        .alias_retain_spill_copy = problem_retain_spill,
        .task_device = task_device,
        .task_resource_kind = task_kind,
        .task_resource_lane = task_lane,
        .task_runtime_ns = problem_task_runtime,
        .task_workspace_bytes = problem_task_workspace,
        .dependency_offsets = empty_offsets,
        .input_offsets = problem_input_offsets,
        .input_aliases = problem_input_aliases,
        .output_offsets = empty_offsets,
        .mutation_offsets = empty_offsets,
        .initial_aliases = problem_initial_aliases,
        .initial_locations = problem_initial_locations,
    };
    const char *problem_alias_names[] = {"alias"};
    const char *problem_task_names[] = {"task"};
    const ShadowSpillPressureFitProblem problem = {
        .abi_version = SHADOWSPILL_ABI_VERSION,
        .residency = &problem_residency,
        .simulation = &problem_simulation,
        .seed_resident = problem_seed_resident,
        .seed_breaks = problem_seed_breaks,
        .alias_json_names = problem_alias_names,
        .task_json_names = problem_task_names,
    };
    const uint8_t problem_strategies[] = {SHADOWSPILL_RESIDENCY_TIGHT_STALL};
    const uint8_t problem_rules[] = {SHADOWSPILL_FETCH_LATEST_SAFE};
    /* Plain emission only. All three axes are lists, and the candidate count
     * is their product, so leaving this one empty asks for no candidates. */
    const uint8_t problem_modes[] = {0U};
    const ShadowSpillPressureFitProblemOptions problem_options = {
        .coalescing_modes = problem_modes,
        .coalescing_mode_count = 1U,
        .residency_strategies = problem_strategies,
        .residency_strategy_count = 1U,
        .fetch_rules = problem_rules,
        .fetch_rule_count = 1U,
        .max_repair_attempts = 1U,
    };
    ShadowSpillPressureFitProblemResult problem_result = {0};
    if (shadowspill_evaluate_pressurefit_problems(
            &problem,
            1U,
            &problem_options,
            &problem_result
        ) != SHADOWSPILL_STATUS_OK ||
        problem_result.candidate_count != 1U ||
        problem_result.selected_candidate_index != 0U ||
        problem_result.work.simulation_calls != 1U ||
        problem_result.work.schedule_emissions != 1U ||
        problem_result.candidates[0].work.simulation_calls != 1U ||
        problem_result.candidates[0].work.schedule_emissions != 1U ||
        problem_result.selected_schedule.action_count != 1U ||
        problem_result.selected_schedule.action_kinds[0] !=
            SHADOWSPILL_MEMORY_RELEASE) {
        shadowspill_pressurefit_problem_result_destroy(&problem_result);
        return EXIT_FAILURE;
    }
    shadowspill_pressurefit_problem_result_destroy(&problem_result);

    const ShadowSpillPressureFitProgramProblem program_problem = {
        .abi_version = SHADOWSPILL_ABI_VERSION,
        .simulation = &problem_simulation,
        .device_priority = problem_priority,
        .alias_json_names = problem_alias_names,
        .task_json_names = problem_task_names,
    };
    ShadowSpillPressureFitPreflightResult preflight = {0};
    if (shadowspill_validate_pressurefit_program_problem(
            &program_problem,
            &preflight
        ) != SHADOWSPILL_STATUS_OK ||
        preflight.status != SHADOWSPILL_STATUS_OK ||
        preflight.failure_kind != SHADOWSPILL_PREFLIGHT_NONE) {
        return EXIT_FAILURE;
    }
    const uint64_t excessive_workspace[] = {65U};
    ShadowSpillSimulationProgram oversized_simulation = problem_simulation;
    oversized_simulation.task_workspace_bytes = excessive_workspace;
    ShadowSpillPressureFitProgramProblem oversized_problem = program_problem;
    oversized_problem.simulation = &oversized_simulation;
    if (shadowspill_validate_pressurefit_program_problem(
            &oversized_problem,
            &preflight
        ) != SHADOWSPILL_STATUS_ANALYTIC_INFEASIBLE ||
        preflight.status != SHADOWSPILL_STATUS_ANALYTIC_INFEASIBLE ||
        preflight.failure_kind != SHADOWSPILL_PREFLIGHT_WORKSPACE_CAPACITY ||
        preflight.error_boundary != 0 || preflight.required_bytes != 65U ||
        preflight.capacity_bytes != 64U) {
        return EXIT_FAILURE;
    }
    ShadowSpillPressureFitProblemResult program_problem_result = {0};
    if (shadowspill_evaluate_pressurefit_program_problems(
            &program_problem,
            1U,
            &problem_options,
            &program_problem_result
        ) != SHADOWSPILL_STATUS_OK ||
        program_problem_result.candidate_count != 1U ||
        program_problem_result.selected_candidate_index != 0U ||
        program_problem_result.work.simulation_calls != 1U ||
        program_problem_result.work.schedule_emissions != 1U ||
        program_problem_result.candidates[0].work.simulation_calls != 1U ||
        program_problem_result.candidates[0].work.schedule_emissions != 1U ||
        program_problem_result.selected_schedule.action_count != 1U ||
        program_problem_result.selected_schedule.action_kinds[0] !=
            SHADOWSPILL_MEMORY_RELEASE) {
        shadowspill_pressurefit_problem_result_destroy(&program_problem_result);
        return EXIT_FAILURE;
    }
    shadowspill_pressurefit_problem_result_destroy(&program_problem_result);
    return EXIT_SUCCESS;
}
