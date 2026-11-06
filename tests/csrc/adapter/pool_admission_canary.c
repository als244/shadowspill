#include <shadowspill/pytorch_adapter.h>
#include <shadowspill/runtime.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MIB(n) ((uint64_t)(n) << 20U)
#define REQUIRE(c) do { if (!(c)) { fprintf(stderr, "pool admission line %d\n", __LINE__); return 1; } } while (0)

int main(int argc, char **argv) {
    REQUIRE(argc == 3);
    const int mode = atoi(argv[2]);
    void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    REQUIRE(library != NULL);
    void (*set_process)(uint64_t) = NULL;
    void (*set_device)(uint64_t, uint64_t) = NULL;
    void *symbol = dlsym(library, "shadowspill_test_process_bytes");
    memcpy(&set_process, &symbol, sizeof(symbol));
    symbol = dlsym(library, "shadowspill_test_device_bytes");
    memcpy(&set_device, &symbol, sizeof(symbol));
    REQUIRE(set_process != NULL && set_device != NULL);
    set_process(MIB(4));
    set_device(MIB(mode == 4 ? 100 : 32), MIB(96));
    const ShadowSpillPytorchPoolConfig pools[2] = {
        {.pool_id = 0, .kind = SHADOWSPILL_POOL_DEVICE},
        {.pool_id = 1, .kind = SHADOWSPILL_POOL_PINNED_HOST, .capacity_bytes = MIB(1)},
    };
    const ShadowSpillPytorchRouteConfig routes[2] = {
        {.route_id = 0, .source_pool_id = 1, .destination_pool_id = 0, .name = "fetch"},
        {.route_id = 1, .source_pool_id = 0, .destination_pool_id = 1, .name = "evict"},
    };
    ShadowSpillPytorchAdapterConfig config = {
        .abi_version = SHADOWSPILL_PYTORCH_ADAPTER_ABI_VERSION,
        .device_ordinal = 0, .device_budget_bytes = MIB(64),
        .external_headroom_bytes = MIB(4), .external_baseline_bytes = MIB(8),
        .physical_limit_bytes = MIB(mode == 1 ? 40 : 80),
        .unattributed_reservation_bytes = MIB(mode == 3 ? 60 : 16),
        .allocator_pool_id = 0, .pools = pools, .pool_count = 2,
        .routes = routes, .route_count = 2,
        .backend_library = argv[1], .disable_numa_binding = 1,
    };
    if (mode == 2) config.device_budget_bytes = MIB(32);
    const ShadowSpillStatus result = shadowspill_pytorch_allocator_bootstrap(&config);
    if (mode >= 3) {
        REQUIRE(result == (mode == 4 ? SHADOWSPILL_STATUS_BACKEND_FAILURE : SHADOWSPILL_STATUS_OUT_OF_MEMORY));
    } else {
        REQUIRE(result == SHADOWSPILL_STATUS_OK);
        ShadowSpillPytorchPhysicalAdmission admission = {0};
        REQUIRE(shadowspill_pytorch_physical_admission(&admission) == SHADOWSPILL_STATUS_OK);
        const uint64_t effective = MIB(mode == 0 ? 48 : mode == 1 ? 40 : 32);
        REQUIRE(admission.device_budget_bytes == config.device_budget_bytes);
        REQUIRE(admission.available_process_bytes == MIB(64));
        REQUIRE(admission.effective_limit_bytes == effective);
        REQUIRE(admission.allocator_pool_bytes == effective - MIB(12));
        REQUIRE(admission.unattributed_reservation_bytes == MIB(16));
        REQUIRE(shadowspill_pytorch_allocator_close() == SHADOWSPILL_STATUS_OK);
    }
    REQUIRE(dlclose(library) == 0);
    return 0;
}
