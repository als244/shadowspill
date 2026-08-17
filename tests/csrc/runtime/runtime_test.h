#ifndef SHADOWSPILL_RUNTIME_TEST_H
#define SHADOWSPILL_RUNTIME_TEST_H

#include <stdint.h>

#include <shadowspill/backend_mock.h>
#include <shadowspill/runtime.h>

static inline ShadowSpillRuntimeStatus shadowspill_test_create_runtime(
    ShadowSpillMockBackend *backend,
    uint64_t execution_pool_bytes,
    uint64_t spill_pool_bytes,
    uint64_t minimum_alignment,
    uint64_t worker_poll_nanoseconds,
    ShadowSpillRuntime **runtime
) {
    ShadowSpillMockRuntimeTopology topology;
    shadowspill_mock_runtime_topology(
        backend,
        execution_pool_bytes,
        spill_pool_bytes,
        minimum_alignment,
        worker_poll_nanoseconds,
        &topology
    );
    return shadowspill_runtime_create(&topology.runtime, runtime);
}

#endif
