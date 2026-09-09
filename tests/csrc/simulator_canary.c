#include <stdint.h>
#include <stdlib.h>

#include "shadowspill/simulator.h"

/* `final_location` for a value the final residency does not name. */
#define NO_FINAL_RESIDENCY 0xFFU

/*
 * One 64-byte alias, initially on the device, and one task of 100 ns that
 * writes it when `writes` is set. The actions all trigger at that task.
 */
static ShadowSpillStatus simulate_one_alias(
    const ShadowSpillSimulationDevice *devices,
    int writes,
    uint8_t retained,
    const uint8_t *kinds,
    uint32_t action_count,
    uint8_t final_location,
    ShadowSpillSimulationResult *result
) {
    static const uint32_t alias_device[] = {0U};
    static const uint64_t alias_size[] = {64U};
    static const uint64_t alias_version[] = {0U};
    static const uint32_t task_device[] = {0U};
    static const uint8_t task_kind[] = {0U};
    static const uint32_t task_lane[] = {0U};
    static const uint64_t task_runtime[] = {100U};
    static const uint64_t task_workspace[] = {16U};
    static const uint32_t none_offsets[] = {0U, 0U};
    static const uint32_t one_offsets[] = {0U, 1U};
    static const uint32_t alias_zero[] = {0U};
    static const uint32_t triggers[] = {0U, 0U, 0U};
    static const uint32_t aliases[] = {0U, 0U, 0U};
    static const uint8_t initial_device[] = {SHADOWSPILL_MEMORY_DEVICE};
    const uint8_t retain[] = {retained};
    const uint8_t final_locations[] = {final_location};
    const ShadowSpillSimulationProgram program = {
        .abi_version = SHADOWSPILL_ABI_VERSION,
        .device_count = 1U,
        .alias_count = 1U,
        .task_count = 1U,
        .action_count = action_count,
        .initial_count = 1U,
        .final_count = final_location == NO_FINAL_RESIDENCY ? 0U : 1U,
        .output_count = writes ? 1U : 0U,
        .spill_capacity_bytes = 512U,
        .devices = devices,
        .alias_device = alias_device,
        .alias_size_bytes = alias_size,
        .alias_initial_version = alias_version,
        .alias_retain_spill_copy = retain,
        .task_device = task_device,
        .task_resource_kind = task_kind,
        .task_resource_lane = task_lane,
        .task_runtime_ns = task_runtime,
        .task_workspace_bytes = task_workspace,
        .dependency_offsets = none_offsets,
        .input_offsets = none_offsets,
        .output_offsets = writes ? one_offsets : none_offsets,
        .output_aliases = alias_zero,
        .mutation_offsets = none_offsets,
        .action_trigger_tasks = triggers,
        .action_aliases = aliases,
        .action_kinds = kinds,
        .initial_aliases = alias_zero,
        .initial_locations = initial_device,
        .final_aliases =
            final_location == NO_FINAL_RESIDENCY ? NULL : alias_zero,
        .final_locations =
            final_location == NO_FINAL_RESIDENCY ? NULL : final_locations,
    };
    result->task_interval_count = 0U;
    result->transfer_interval_count = 0U;
    return shadowspill_simulate(&program, result);
}

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
        .abi_version = SHADOWSPILL_ABI_VERSION,
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
    ShadowSpillTransferInterval transfer_intervals[2] = {{0}};
    ShadowSpillDevicePeak peaks[1] = {{0}};
    ShadowSpillSimulationResult result = {
        .task_intervals = task_intervals,
        .task_interval_capacity = 1U,
        .transfer_intervals = transfer_intervals,
        .transfer_interval_capacity = 2U,
        .device_peaks = peaks,
        .device_peak_capacity = 1U,
    };

    if (shadowspill_abi_version() !=
        SHADOWSPILL_ABI_VERSION) {
        return EXIT_FAILURE;
    }
    if (shadowspill_simulate(&program, &result) != SHADOWSPILL_STATUS_OK) {
        return EXIT_FAILURE;
    }
    if (result.makespan_ns != 100U || result.task_interval_count != 1U ||
        result.transfer_interval_count != 0U ||
        result.device_peaks[0].total_bytes != 208U ||
        result.task_intervals[0].start_ns != 0U ||
        result.task_intervals[0].end_ns != 100U) {
        return EXIT_FAILURE;
    }


    /* A release of the copy the task has written would drop the only
     * current version of a value the final residency still needs: refused
     * at the release, whether or not the alias retains a spill copy. */
    const uint8_t release[] = {SHADOWSPILL_MEMORY_RELEASE};
    if (simulate_one_alias(
            devices, 1, 1U, release, 1U, SHADOWSPILL_MEMORY_SPILL, &result
        ) != SHADOWSPILL_STATUS_INVALID_RELEASE ||
        simulate_one_alias(
            devices, 1, 0U, release, 1U, SHADOWSPILL_MEMORY_SPILL, &result
        ) != SHADOWSPILL_STATUS_INVALID_RELEASE) {
        return EXIT_FAILURE;
    }
    /* A value nothing needs any more is released freely, whatever its
     * retained spill copy holds. */
    if (simulate_one_alias(
            devices, 1, 1U, release, 1U, NO_FINAL_RESIDENCY, &result
        ) != SHADOWSPILL_STATUS_OK ||
        result.makespan_ns != 100U || result.transfer_interval_count != 0U) {
        return EXIT_FAILURE;
    }
    /* A retained spill copy the task has written and nothing refreshed is
     * stale at the end. */
    if (simulate_one_alias(
            devices, 1, 1U, NULL, 0U, SHADOWSPILL_MEMORY_SPILL, &result
        ) != SHADOWSPILL_STATUS_FINAL_RESIDENCY) {
        return EXIT_FAILURE;
    }
    /* A write-back refreshes the spill copy on the evict lane and keeps the
     * device copy: the object bytes never drop. */
    const uint8_t write_back[] = {SHADOWSPILL_MEMORY_WRITE_BACK};
    if (simulate_one_alias(
            devices, 1, 1U, write_back, 1U, SHADOWSPILL_MEMORY_SPILL, &result
        ) != SHADOWSPILL_STATUS_OK ||
        result.makespan_ns != 164U || result.transfer_interval_count != 1U ||
        result.transfer_intervals[0].kind != SHADOWSPILL_MEMORY_WRITE_BACK ||
        result.transfer_intervals[0].direction != SHADOWSPILL_TRANSFER_EVICT ||
        result.transfer_intervals[0].start_ns != 100U ||
        result.transfer_intervals[0].end_ns != 164U ||
        result.device_peaks[0].object_bytes != 64U) {
        return EXIT_FAILURE;
    }
    /* A write-back of a copy that is already current costs nothing. */
    if (simulate_one_alias(
            devices, 0, 1U, write_back, 1U, SHADOWSPILL_MEMORY_SPILL, &result
        ) != SHADOWSPILL_STATUS_OK ||
        result.makespan_ns != 100U || result.transfer_interval_count != 0U) {
        return EXIT_FAILURE;
    }
    /* A release behind a pending write-back waits for the copy to land. */
    const uint8_t write_back_then_release[] = {
        SHADOWSPILL_MEMORY_WRITE_BACK,
        SHADOWSPILL_MEMORY_RELEASE,
    };
    if (simulate_one_alias(
            devices,
            1,
            1U,
            write_back_then_release,
            2U,
            SHADOWSPILL_MEMORY_SPILL,
            &result
        ) != SHADOWSPILL_STATUS_OK ||
        result.makespan_ns != 164U || result.transfer_interval_count != 1U ||
        result.transfer_intervals[0].kind != SHADOWSPILL_MEMORY_WRITE_BACK) {
        return EXIT_FAILURE;
    }
    /* A second departure while one is on the lane has no version to carry. */
    const uint8_t write_back_twice[] = {
        SHADOWSPILL_MEMORY_WRITE_BACK,
        SHADOWSPILL_MEMORY_WRITE_BACK,
    };
    if (simulate_one_alias(
            devices, 1, 1U, write_back_twice, 2U, SHADOWSPILL_MEMORY_SPILL, &result
        ) != SHADOWSPILL_STATUS_INVALID_WRITE_BACK) {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
