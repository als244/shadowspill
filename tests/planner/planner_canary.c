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
        .h2d_bandwidth_bytes_per_second = 1U,
        .d2h_bandwidth_bytes_per_second = 1U,
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
        .alias_retain_host = retain_host,
        .initial_location = initial_locations,
        .final_location = final_locations,
        .anchors = anchors,
        .productions = empty_cells,
        .latest_access_task = latest_access,
        .output_reservations = empty_cells,
        .write_prefix = empty_cells,
        .first_input_task = first_input_tasks,
        .h2d_runtime_ns = transfer_runtimes,
        .d2h_runtime_ns = transfer_runtimes,
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
    for (uint32_t index = 0U; index < 6U; ++index) {
        if (reduced_resident[index] != expected_resident[index]) {
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
