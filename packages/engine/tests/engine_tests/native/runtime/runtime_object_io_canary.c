/* Range I/O preserves neighboring bytes and the full-object API's strictness. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <shadowspill/backend_mock.h>
#include <shadowspill/runtime.h>

#define CHECK(expression) do { if (!(expression)) { \
    fprintf(stderr, "object I/O check failed at line %d\n", __LINE__); \
    return 1; \
} } while (0)

int main(void) {
    ShadowSpillBackend backend = {0};
    const ShadowSpillMockBackendConfig backend_config = {0};
    CHECK(shadowspill_mock_backend_create(&backend_config, &backend) == 0);
    ShadowSpillMockRuntimeTopology topology;
    shadowspill_mock_runtime_topology(&backend, 4096U, 4096U, 8U, 1000U, &topology);
    ShadowSpillRuntime *runtime = NULL;
    CHECK(shadowspill_runtime_create(&topology.runtime, &runtime) == SHADOWSPILL_STATUS_OK);
    const ShadowSpillObjectDescription object = {
        .object_id = 1U, .size_bytes = 32U, .initial_version = 1U,
        .initial_pool_id = 1U, .initially_resident = 1U,
    };
    CHECK(shadowspill_register_object(runtime, &object) == SHADOWSPILL_STATUS_OK);
    unsigned char original[32], output[32], patch[17];
    memset(original, 0xA5, sizeof(original));
    memset(patch, 0x19, sizeof(patch));
    CHECK(shadowspill_write_object(runtime, 1U, 1U, original, sizeof(original)) ==
          SHADOWSPILL_STATUS_OK);
    CHECK(shadowspill_write_object_range(runtime, 1U, 1U, patch, 3U, sizeof(patch)) ==
          SHADOWSPILL_STATUS_OK);
    CHECK(shadowspill_read_object(runtime, 1U, 1U, output, sizeof(output)) ==
          SHADOWSPILL_STATUS_OK);
    memcpy(original + 3U, patch, sizeof(patch));
    CHECK(memcmp(original, output, sizeof(output)) == 0);
    memset(output, 0, sizeof(output));
    CHECK(shadowspill_read_object_range(runtime, 1U, 1U, output, 3U, sizeof(patch)) ==
          SHADOWSPILL_STATUS_OK);
    CHECK(memcmp(output, patch, sizeof(patch)) == 0);
    CHECK(shadowspill_write_object(runtime, 1U, 1U, patch, sizeof(patch)) !=
          SHADOWSPILL_STATUS_OK);
    CHECK(shadowspill_read_object(runtime, 1U, 1U, output, sizeof(patch)) !=
          SHADOWSPILL_STATUS_OK);
    CHECK(shadowspill_write_object_range(runtime, 1U, 1U, patch, 31U, 2U) !=
          SHADOWSPILL_STATUS_OK);
    CHECK(shadowspill_read_object_range(runtime, 1U, 1U, output, UINT64_MAX, 2U) !=
          SHADOWSPILL_STATUS_OK);
    CHECK(shadowspill_read_object_range(runtime, 1U, 1U, output, 1U, UINT64_MAX) !=
          SHADOWSPILL_STATUS_OK);
    CHECK(shadowspill_read_object_range(runtime, 1U, 1U, NULL, 32U, 0U) ==
          SHADOWSPILL_STATUS_OK);
    CHECK(shadowspill_read_object_range(runtime, 2U, 1U, output, 0U, 1U) !=
          SHADOWSPILL_STATUS_OK);
    shadowspill_runtime_destroy(runtime);
    shadowspill_backend_destroy(&backend);
    return 0;
}
