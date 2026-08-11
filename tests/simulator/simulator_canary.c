#include <stdint.h>
#include <stdlib.h>

#include "shadowspill/simulator.h"

int main(void) {
    const ShadowSpillSimulationDevice devices[] = {
        {
            .capacity_bytes = 512U,
            .h2d_bandwidth_bytes_per_second = 1000000000U,
            .d2h_bandwidth_bytes_per_second = 1000000000U,
        },
    };
    const uint32_t alias_device[] = {0U, 0U};
    const uint64_t alias_size[] = {64U, 128U};
    const uint64_t alias_version[] = {0U, 0U};
    const uint8_t alias_host[] = {0U, 0U};
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
        .host_capacity_bytes = 512U,
        .devices = devices,
        .alias_device = alias_device,
        .alias_size_bytes = alias_size,
        .alias_initial_version = alias_version,
        .alias_retain_host_backing = alias_host,
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
    return EXIT_SUCCESS;
}
