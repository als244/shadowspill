#include <shadowspill/backend_mock.h>
#include <shadowspill/runtime.h>

#include <stdint.h>
#include <stdio.h>

int main(void) {
    const ShadowSpillMockBackendConfig mock_config = {
        .abi_version = SHADOWSPILL_MOCK_BACKEND_ABI_VERSION,
        .fetch_delay_nanoseconds = 1000U,
        .evict_delay_nanoseconds = 2000U,
    };
    ShadowSpillMockBackend *mock = NULL;
    if (shadowspill_mock_backend_create(&mock_config, &mock) != 0) {
        return 1;
    }
    ShadowSpillMockRuntimeTopology topology;
    shadowspill_mock_runtime_topology(
        mock, 4096U, 4096U, 8U, 1000U, &topology
    );
    ShadowSpillRuntime *runtime = NULL;
    if (shadowspill_runtime_create(&topology.runtime, &runtime) !=
        SHADOWSPILL_STATUS_OK) {
        shadowspill_mock_backend_destroy(mock);
        return 2;
    }
    const ShadowSpillTransferCalibrationConfig calibration = {
        .abi_version = SHADOWSPILL_ABI_VERSION,
        .small_copy_bytes = 32U,
        .large_copy_bytes = 512U,
        .warmup_copies = 1U,
        .measured_copies = 3U,
        .provenance = SHADOWSPILL_TRANSFER_PROFILE_RECALIBRATION,
    };
    if (shadowspill_runtime_calibrate_transfer_capabilities(
            runtime, &calibration, NULL, 0U
        ) != SHADOWSPILL_STATUS_OK) {
        fprintf(stderr, "all-route calibration failed\n");
        return 3;
    }
    ShadowSpillTransferProfile profiles[4] = {0};
    uint32_t count = 0U;
    uint64_t generation = 0U;
    if (shadowspill_runtime_transfer_profiles(
            runtime, profiles, 4U, &count, &generation
        ) != SHADOWSPILL_STATUS_OK || count != 4U || generation != 1U) {
        fprintf(stderr, "invalid first matrix snapshot\n");
        return 4;
    }
    for (uint32_t index = 0U; index < count; ++index) {
        if (!profiles[index].available || !profiles[index].calibrated ||
            profiles[index].generation != generation ||
            (profiles[index].source_pool_id !=
                 profiles[index].destination_pool_id &&
             (profiles[index].bandwidth_bytes_per_second == 0U ||
              profiles[index].solo_bandwidth_bytes_per_second == 0U ||
              profiles[index].concurrent_bandwidth_bytes_per_second == 0U ||
              profiles[index].calibration_mode !=
                  SHADOWSPILL_TRANSFER_CALIBRATION_BIDIRECTIONAL ||
              profiles[index].concurrent_route_count != 2U))) {
            fprintf(stderr, "invalid matrix cell %u\n", index);
            return 5;
        }
    }
    const ShadowSpillTransferRouteKey selected = {
        .source_pool_id = 1U,
        .destination_pool_id = 0U,
    };
    if (shadowspill_runtime_calibrate_transfer_capabilities(
            runtime, &calibration, &selected, 1U
        ) != SHADOWSPILL_STATUS_OK ||
        shadowspill_runtime_transfer_profiles(
            runtime, profiles, 4U, &count, &generation
        ) != SHADOWSPILL_STATUS_OK || generation != 2U) {
        fprintf(stderr, "selected-route recalibration failed\n");
        return 6;
    }
    const uint32_t selected_index =
        selected.source_pool_id * 2U + selected.destination_pool_id;
    if (profiles[selected_index].calibration_mode !=
            SHADOWSPILL_TRANSFER_CALIBRATION_SOLO ||
        profiles[selected_index].concurrent_route_count != 1U ||
        profiles[selected_index].concurrent_bandwidth_bytes_per_second != 0U ||
        profiles[selected_index].bandwidth_bytes_per_second !=
            profiles[selected_index].solo_bandwidth_bytes_per_second) {
        fprintf(stderr, "selected route did not publish a solo profile\n");
        return 7;
    }
    shadowspill_runtime_destroy(runtime);
    shadowspill_mock_backend_destroy(mock);
    return 0;
}
