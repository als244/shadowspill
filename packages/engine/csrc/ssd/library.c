#include "internal.h"

static const ShadowSpillLibraryDescription description = {
    .abi_version = SHADOWSPILL_ABI_VERSION,
    .name = "ssd",
    .pool_memory = &shadowspill_ssd_pool_memory,
    .pool_memory_count = 1U,
    .lanes = shadowspill_ssd_lanes,
    .lane_count = 2U,
};

SHADOWSPILL_SSD_API const ShadowSpillLibraryDescription *
shadowspill_library_describe(void) {
    return &description;
}
