/* What this library offers a runtime: one pool kind, and nothing yet that
   moves its bytes. */
#include "../internal.h"

#include <shadowspill/runtime/library.h>

/*
 * A loader reads this, copies the entries into a runtime config, and that is
 * the whole of the loading story. Nothing here runs when the library is
 * opened: every entry is a function pointer nobody has called yet. So a box
 * with no daemon reachable -- and, once there is a lane, no NIC at all -- loads
 * this successfully and fails with a reason the first time a Remote pool is
 * created, where there is somewhere to report it.
 */
static const ShadowSpillLibraryDescription description = {
    .abi_version = SHADOWSPILL_ABI_VERSION,
    .name = "network",
    .pool_memory = &shadowspill_remote_pool_memory,
    .pool_memory_count = 1U,
    /* Both directions: a route is directed, and a spill topology has two. */
    .lanes = shadowspill_remote_lanes,
    .lane_count = 2U,
};

SHADOWSPILL_NETWORK_API const ShadowSpillLibraryDescription *
shadowspill_library_describe(void) {
    return &description;
}
