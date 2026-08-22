#include "shadowspill/simulator.h"

uint32_t shadowspill_simulator_abi_version(void) {
    return SHADOWSPILL_SIMULATOR_ABI_VERSION;
}

const char *shadowspill_simulation_status_string(ShadowSpillSimulationStatus status) {
    return shadowspill_status_string(status);
}

