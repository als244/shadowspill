#include "shadowspill/simulator.h"

uint32_t shadowspill_simulator_abi_version(void) {
    return SHADOWSPILL_SIMULATOR_ABI_VERSION;
}

const char *shadowspill_simulation_status_string(
    ShadowSpillSimulationStatus status
) {
    switch (status) {
        case SHADOWSPILL_SIMULATION_OK:
            return "ok";
        case SHADOWSPILL_SIMULATION_INVALID_ARGUMENT:
            return "invalid argument";
        case SHADOWSPILL_SIMULATION_ALLOCATION_FAILURE:
            return "allocation failure";
        case SHADOWSPILL_SIMULATION_INITIAL_DEVICE_CAPACITY:
            return "initial device capacity exceeded";
        case SHADOWSPILL_SIMULATION_INITIAL_HOST_CAPACITY:
            return "initial host capacity exceeded";
        case SHADOWSPILL_SIMULATION_TASK_INPUT_DEADLOCK:
            return "task input cannot become resident";
        case SHADOWSPILL_SIMULATION_TASK_DEVICE_CAPACITY:
            return "task output and workspace exceed device capacity";
        case SHADOWSPILL_SIMULATION_PREFETCH_DEVICE_CAPACITY:
            return "prefetch cannot reserve device capacity";
        case SHADOWSPILL_SIMULATION_OFFLOAD_HOST_CAPACITY:
            return "offload cannot reserve host capacity";
        case SHADOWSPILL_SIMULATION_TRANSFER_DEADLOCK:
            return "transfer has no progress source";
        case SHADOWSPILL_SIMULATION_INVALID_RELEASE:
            return "release has no ready device copy";
        case SHADOWSPILL_SIMULATION_RELEASE_TRANSFER_CONFLICT:
            return "release conflicts with a transfer";
        case SHADOWSPILL_SIMULATION_INVALID_OFFLOAD:
            return "offload has no ready device source";
        case SHADOWSPILL_SIMULATION_INVALID_PREFETCH:
            return "prefetch has no host source or duplicates a device copy";
        case SHADOWSPILL_SIMULATION_FINAL_RESIDENCY:
            return "required final residency was not reached";
        case SHADOWSPILL_SIMULATION_INTERNAL_ERROR:
            return "internal simulator invariant failed";
        default:
            return "unknown simulator status";
    }
}
