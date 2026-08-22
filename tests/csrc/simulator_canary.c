#include <stdint.h>
#include <stdlib.h>

#include "shadowspill/simulator.h"

int main(void) {
    const ShadowSpillSimulationDevice devices[] = {
        {
            .capacity_bytes = 512U,
            .fetch_bandwidth_bytes_per_second = 1000000000U,
            .evict_bandwidth_bytes_per_second = 1000000000U,
        },
    };
    const uint32_t alias_device[] = {0U, 0U};
    const uint64_t alias_size[] = {64U, 128U};
    const uint64_t alias_version[] = {0U, 0U};
    const uint8_t alias_spill[] = {0U, 0U};
    const uint32_t task_device[] = {0U};
    const uint8_t task_kind[] = {0U};
    const uint32_t task_lane[] = {0U};
    const uint64_t task_runtime[] = {100U};
    const uint64_t task_workspace[] = {16U};
    const uint32_t empty_offsets[] = {0U, 0U};
    const uint32_t input_offsets[] = {0U, 1U};
    const uint32_t inputs[] = {0U};
    const uint32_t output_offsets[] = {0U, 1U};
    const uint32_t outputs[] = {1U};
    const uint32_t initial_aliases[] = {0U};
    const uint8_t initial_locations[] = {SHADOWSPILL_MEMORY_DEVICE};
    const uint32_t final_aliases[] = {1U};
    const uint8_t final_locations[] = {SHADOWSPILL_MEMORY_DEVICE};
    const ShadowSpillSimulationProgram program = {
        .abi_version = SHADOWSPILL_SIMULATOR_ABI_VERSION,
        .device_count = 1U,
        .alias_count = 2U,
        .task_count = 1U,
        .initial_count = 1U,
        .final_count = 1U,
        .input_count = 1U,
        .output_count = 1U,
        .spill_capacity_bytes = 512U,
        .devices = devices,
        .alias_device = alias_device,
        .alias_size_bytes = alias_size,
        .alias_initial_version = alias_version,
        .alias_retain_spill_copy = alias_spill,
        .task_device = task_device,
        .task_resource_kind = task_kind,
        .task_resource_lane = task_lane,
        .task_runtime_ns = task_runtime,
        .task_workspace_bytes = task_workspace,
        .dependency_offsets = empty_offsets,
        .input_offsets = input_offsets,
        .input_aliases = inputs,
        .output_offsets = output_offsets,
        .output_aliases = outputs,
        .mutation_offsets = empty_offsets,
        .initial_aliases = initial_aliases,
        .initial_locations = initial_locations,
        .final_aliases = final_aliases,
        .final_locations = final_locations,
    };
    ShadowSpillTaskInterval task_intervals[1] = {{0}};
    ShadowSpillTransferInterval transfer_intervals[1] = {{0}};
    ShadowSpillDevicePeak peaks[1] = {{0}};
    ShadowSpillSimulationResult result = {
        .task_intervals = task_intervals,
        .task_interval_capacity = 1U,
        .transfer_intervals = transfer_intervals,
        .transfer_interval_capacity = 1U,
        .device_peaks = peaks,
        .device_peak_capacity = 1U,
    };

    if (shadowspill_simulator_abi_version() !=
        SHADOWSPILL_SIMULATOR_ABI_VERSION) {
        return EXIT_FAILURE;
    }
    if (shadowspill_simulate(&program, &result) != SHADOWSPILL_SIMULATION_OK) {
        return EXIT_FAILURE;
    }
    if (result.makespan_ns != 100U || result.task_interval_count != 1U ||
        result.transfer_interval_count != 0U ||
        result.device_peaks[0].total_bytes != 208U ||
        result.task_intervals[0].start_ns != 0U ||
        result.task_intervals[0].end_ns != 100U) {
        return EXIT_FAILURE;
    }

    const uint32_t one_alias_device[] = {0U};
    const uint64_t one_alias_size[] = {64U};
    const uint64_t one_alias_version[] = {0U};
    const uint8_t retained_spill[] = {1U};
    const uint32_t no_inputs_offsets[] = {0U, 0U};
    const uint32_t one_output_offsets[] = {0U, 1U};
    const uint32_t one_output[] = {0U};
    const uint32_t one_initial_alias[] = {0U};
    const uint8_t one_initial_device[] = {SHADOWSPILL_MEMORY_DEVICE};
    const uint32_t release_task[] = {0U};
    const uint32_t release_alias[] = {0U};
    const uint8_t release_kind[] = {SHADOWSPILL_MEMORY_RELEASE};
    const uint32_t one_final_alias[] = {0U};
    const uint8_t one_final_spill[] = {SHADOWSPILL_MEMORY_SPILL};
    const ShadowSpillSimulationProgram stale_spill_program = {
        .abi_version = SHADOWSPILL_SIMULATOR_ABI_VERSION,
        .device_count = 1U,
        .alias_count = 1U,
        .task_count = 1U,
        .action_count = 1U,
        .initial_count = 1U,
        .final_count = 1U,
        .output_count = 1U,
        .spill_capacity_bytes = 512U,
        .devices = devices,
        .alias_device = one_alias_device,
        .alias_size_bytes = one_alias_size,
        .alias_initial_version = one_alias_version,
        .alias_retain_spill_copy = retained_spill,
        .task_device = task_device,
        .task_resource_kind = task_kind,
        .task_resource_lane = task_lane,
        .task_runtime_ns = task_runtime,
        .task_workspace_bytes = task_workspace,
        .dependency_offsets = no_inputs_offsets,
        .input_offsets = no_inputs_offsets,
        .output_offsets = one_output_offsets,
        .output_aliases = one_output,
        .mutation_offsets = no_inputs_offsets,
        .action_trigger_tasks = release_task,
        .action_aliases = release_alias,
        .action_kinds = release_kind,
        .initial_aliases = one_initial_alias,
        .initial_locations = one_initial_device,
        .final_aliases = one_final_alias,
        .final_locations = one_final_spill,
    };
    result.task_interval_count = 0U;
    result.transfer_interval_count = 0U;
    if (shadowspill_simulate(&stale_spill_program, &result) !=
        SHADOWSPILL_SIMULATION_FINAL_RESIDENCY) {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
