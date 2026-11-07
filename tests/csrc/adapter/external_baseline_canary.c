#include <shadowspill/pytorch_adapter.h>
#include <shadowspill/runtime.h>

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MIB(n) ((uint64_t)(n) << 20U)
#define REQUIRE(condition) do { if (!(condition)) { \
    fprintf(stderr, "external baseline canary: line %d\n", __LINE__); \
    return 1; } } while (0)

int main(int argc, char **argv) {
    REQUIRE(argc == 4);
    const uint64_t floor = strtoull(argv[2], NULL, 10);
    const uint8_t strict = (uint8_t)strtoul(argv[3], NULL, 10);
    void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    REQUIRE(library != NULL);
    void *symbol = dlsym(library, "shadowspill_test_process_bytes");
    void (*set_process)(uint64_t) = NULL;
    _Static_assert(sizeof(set_process) == sizeof(symbol), "function pointer size");
    memcpy(&set_process, &symbol, sizeof(symbol));
    REQUIRE(set_process != NULL);
    set_process(MIB(4));
    const ShadowSpillPytorchPoolConfig pools[2] = {
        {.pool_id = 0, .kind = SHADOWSPILL_POOL_DEVICE},
        {.pool_id = 1, .kind = SHADOWSPILL_POOL_PINNED_HOST,
         .capacity_bytes = MIB(1)},
    };
    const ShadowSpillPytorchRouteConfig routes[2] = {
        {.route_id = 0, .source_pool_id = 1, .destination_pool_id = 0,
         .name = "fetch"},
        {.route_id = 1, .source_pool_id = 0, .destination_pool_id = 1,
         .name = "evict"},
    };
    ShadowSpillPytorchAdapterConfig config = {
        .abi_version = SHADOWSPILL_PYTORCH_ADAPTER_ABI_VERSION,
        .device_ordinal = 0, .device_budget_bytes = MIB(64),
        .external_headroom_bytes = MIB(4),
        .external_baseline_bytes = UINT64_MAX,
        .reject_overbudget = strict,
        .allocator_pool_id = 0, .pools = pools, .pool_count = 2,
        .routes = routes, .route_count = 2,
        .backend_library = argv[1], .disable_numa_binding = 1,
    };
    config.abi_version = 1U;
    REQUIRE(shadowspill_pytorch_allocator_bootstrap(&config) ==
            SHADOWSPILL_STATUS_INVALID_ARGUMENT);
    config.abi_version = 2U;
    REQUIRE(shadowspill_pytorch_allocator_bootstrap(&config) ==
            SHADOWSPILL_STATUS_INVALID_ARGUMENT);
    config.abi_version = SHADOWSPILL_PYTORCH_ADAPTER_ABI_VERSION;
    REQUIRE(shadowspill_pytorch_allocator_bootstrap(&config) ==
            SHADOWSPILL_STATUS_INVALID_ARGUMENT);
    config.external_baseline_bytes = MIB(60);
    REQUIRE(shadowspill_pytorch_allocator_bootstrap(&config) ==
            SHADOWSPILL_STATUS_INVALID_ARGUMENT);
    config.external_baseline_bytes = floor;
    REQUIRE(shadowspill_pytorch_allocator_bootstrap(&config) ==
            SHADOWSPILL_STATUS_OK);
    ShadowSpillPytorchPhysicalAdmission admission = {0};
    REQUIRE(shadowspill_pytorch_physical_admission(&admission) ==
            SHADOWSPILL_STATUS_OK);
    const uint64_t reserved = floor > MIB(4) ? floor : MIB(4);
    const uint64_t available = MIB(64) - reserved - MIB(4);
    /* The mock backend requires 256-byte alignment. A non-page-aligned
       budget must keep its usable tail rather than discard a whole page. */
    const uint64_t pool = available - available % 256U;
    REQUIRE(admission.baseline_bytes == MIB(4));
    REQUIRE(admission.reserved_baseline_bytes == reserved);
    REQUIRE(admission.allocator_pool_bytes == pool);
    REQUIRE(admission.external_headroom_bytes == MIB(4));

    /* Mature growth consumes the pre-reserved floor; it does not spend the
       caller's growth margin a second time. Raw growth remains observable. */
    set_process(pool + reserved + MIB(3));
    REQUIRE(shadowspill_pytorch_check_physical_budget() == SHADOWSPILL_STATUS_OK);
    ShadowSpillPytorchAdapterStatistics statistics = {0};
    REQUIRE(shadowspill_pytorch_allocator_statistics(&statistics) ==
            SHADOWSPILL_STATUS_OK);
    REQUIRE(statistics.observed_external_high_water_bytes ==
            reserved - MIB(4) + MIB(3));
    const uint64_t allowance = reserved;
    REQUIRE(shadowspill_pytorch_seal_physical_budget(allowance, 256) ==
            SHADOWSPILL_STATUS_OK);

    /* A non-aligned floor leaves a little unused physical budget. The external
       allowance still applies, independently of the whole-process cap. */
    set_process(pool + reserved + MIB(4) + 1);
    REQUIRE(shadowspill_pytorch_check_physical_budget() ==
            (strict ? SHADOWSPILL_STATUS_PLAN_VIOLATION : SHADOWSPILL_STATUS_OK));
    set_process(pool + reserved);
    REQUIRE(shadowspill_pytorch_allocator_close() == SHADOWSPILL_STATUS_OK);
    REQUIRE(dlclose(library) == 0);
    return 0;
}
