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
        .abi_version = SHADOWSPILL_SIMULATOR_ABI_VERSION,
        .device_count = 1U,
        .task_count = 1U,
        .host_capacity_bytes = 1U,
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

    if (shadowspill_planner_abi_version() != SHADOWSPILL_PLANNER_ABI_VERSION) {
        return EXIT_FAILURE;
    }
    if (shadowspill_select_plan(candidates, 2U, &result) !=
        SHADOWSPILL_PLANNER_OK) {
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
    const uint8_t retain_host[] = {1U, 1U};
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
        .abi_version = SHADOWSPILL_PLANNER_ABI_VERSION,
        .alias_count = 2U,
        .boundary_count = 3U,
        .device_count = 1U,
        .alias_size_bytes = alias_sizes,
        .alias_device = alias_devices,
        .alias_retain_spill_copy = retain_host,
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
        ) != SHADOWSPILL_PLANNER_OK) {
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

    const uint64_t context_alias_size[] = {32U};
    const ShadowSpillSimulationDevice context_device = {
        .capacity_bytes = 64U,
        .fetch_bandwidth_bytes_per_second = 1U,
        .evict_bandwidth_bytes_per_second = 1U,
    };
    const uint32_t context_alias_device[] = {0U};
    const uint8_t context_retain_spill[] = {1U};
    const int8_t context_initial_location[] = {0};
    const int8_t context_final_location[] = {-1};
    const uint8_t context_anchors[] = {1U, 0U};
    const uint8_t context_zero_cells[] = {0U, 0U};
    const uint32_t context_latest_access[] = {
        0U,
        SHADOWSPILL_PLANNER_NO_INDEX,
    };
    const uint32_t context_first_input[] = {0U};
    const uint64_t context_transfer_runtime[] = {1U};
    const uint64_t context_task_end[] = {10U};
    const uint64_t context_capacity[] = {64U};
    const uint32_t context_priority[] = {0U};
    const uint8_t context_seed_resident[] = {1U, 0U};
    const uint8_t context_seed_breaks[] = {0U, 0U};
    const ShadowSpillResidencyProblem context_residency = {
        .abi_version = SHADOWSPILL_PLANNER_ABI_VERSION,
        .alias_count = 1U,
        .boundary_count = 2U,
        .device_count = 1U,
        .alias_size_bytes = context_alias_size,
        .alias_device = context_alias_device,
        .alias_retain_spill_copy = context_retain_spill,
        .initial_location = context_initial_location,
        .final_location = context_final_location,
        .anchors = context_anchors,
        .productions = context_zero_cells,
        .latest_access_task = context_latest_access,
        .output_reservations = context_zero_cells,
        .write_prefix = context_zero_cells,
        .first_input_task = context_first_input,
        .fetch_runtime_ns = context_transfer_runtime,
        .evict_runtime_ns = context_transfer_runtime,
        .task_ideal_end_ns = context_task_end,
        .device_capacity_bytes = context_capacity,
        .device_priority = context_priority,
    };
    const uint64_t context_alias_version[] = {0U};
    const uint64_t context_task_runtime[] = {10U};
    const uint64_t context_task_workspace[] = {0U};
    const uint32_t context_input_offsets[] = {0U, 1U};
    const uint32_t context_input_aliases[] = {0U};
    const ShadowSpillSimulationProgram context_simulation = {
        .abi_version = SHADOWSPILL_SIMULATOR_ABI_VERSION,
        .device_count = 1U,
        .alias_count = 1U,
        .task_count = 1U,
        .input_count = 1U,
        .host_capacity_bytes = 64U,
        .devices = &context_device,
        .alias_device = context_alias_device,
        .alias_size_bytes = context_alias_size,
        .alias_initial_version = context_alias_version,
        .alias_retain_spill_copy = context_retain_spill,
        .task_device = task_device,
        .task_resource_kind = task_kind,
        .task_resource_lane = task_lane,
        .task_runtime_ns = context_task_runtime,
        .task_workspace_bytes = context_task_workspace,
        .dependency_offsets = empty_offsets,
        .input_offsets = context_input_offsets,
        .input_aliases = context_input_aliases,
        .output_offsets = empty_offsets,
        .mutation_offsets = empty_offsets,
    };
    const char *context_alias_names[] = {"alias"};
    const char *context_task_names[] = {"task"};
    const ShadowSpillPressureFitContext context = {
        .abi_version = SHADOWSPILL_PLANNER_ABI_VERSION,
        .residency = &context_residency,
        .simulation = &context_simulation,
        .seed_resident = context_seed_resident,
        .seed_breaks = context_seed_breaks,
        .alias_json_names = context_alias_names,
        .task_json_names = context_task_names,
    };
    const uint8_t context_strategies[] = {SHADOWSPILL_RESIDENCY_TIGHT_STALL};
    const uint8_t context_rules[] = {SHADOWSPILL_PREFETCH_LATEST_SAFE};
    const ShadowSpillPressureFitContextOptions context_options = {
        .residency_strategies = context_strategies,
        .residency_strategy_count = 1U,
        .prefetch_rules = context_rules,
        .prefetch_rule_count = 1U,
        .max_repair_attempts = 1U,
    };
    ShadowSpillPressureFitContextResult context_result = {0};
    if (shadowspill_evaluate_pressurefit_context(
            &context,
            &context_options,
            &context_result
        ) != SHADOWSPILL_PLANNER_OK ||
        context_result.candidate_count != 1U ||
        context_result.selected_candidate_index != 0U ||
        context_result.selected_schedule.action_count != 1U ||
        context_result.selected_schedule.action_kinds[0] !=
            SHADOWSPILL_MEMORY_RELEASE) {
        shadowspill_pressurefit_context_result_destroy(&context_result);
        return EXIT_FAILURE;
    }
    shadowspill_pressurefit_context_result_destroy(&context_result);
    return EXIT_SUCCESS;
}
