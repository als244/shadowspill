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
    return EXIT_SUCCESS;
}
