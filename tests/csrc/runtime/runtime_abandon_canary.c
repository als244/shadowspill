/*
 * Abandoning a runtime returns without waiting, and still releases what the
 * runtime owns.
 *
 * A process that is exiting cannot finish outstanding work, and the drain in
 * shadowspill_runtime_close has no escape but a latched failure that will
 * never appear when something outside ShadowSpill called exit(). Waiting there
 * does not delay the exit, it prevents it: exit runs its handlers before
 * _exit, so a handler that blocks leaves the process alive with every thread
 * parked.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <shadowspill/backend_mock.h>
#include <shadowspill/runtime.h>

int main(void) {
    ShadowSpillBackend mock = {0};
    const ShadowSpillMockBackendConfig mock_config = {
        .event_delay_nanoseconds = 2000000U,
    };
    if (shadowspill_mock_backend_create(&mock_config, &mock) != 0) {
        return EXIT_FAILURE;
    }
    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillMockRuntimeTopology topology;
    shadowspill_mock_runtime_topology(&mock, 4096U, 1U, 1U, 10000U, &topology);
    if (shadowspill_runtime_create(&topology.runtime, &runtime) !=
        SHADOWSPILL_STATUS_OK) {
        return EXIT_FAILURE;
    }

    uint64_t actions = UINT64_MAX;
    uint64_t retirements = UINT64_MAX;
    if (shadowspill_runtime_abandon(runtime, &actions, &retirements) !=
        SHADOWSPILL_STATUS_OK) {
        (void)fprintf(stderr, "abandoning a healthy runtime reported a failure\n");
        return EXIT_FAILURE;
    }
    if (actions != 0U || retirements != 0U) {
        (void)fprintf(
            stderr,
            "an idle runtime reported %llu action(s) and %llu retirement(s)\n",
            (unsigned long long)actions,
            (unsigned long long)retirements
        );
        return EXIT_FAILURE;
    }

    /* Idempotent, like close: a second call is a no-op, not a fault. */
    if (shadowspill_runtime_abandon(runtime, NULL, NULL) !=
        SHADOWSPILL_STATUS_OK) {
        (void)fprintf(stderr, "abandoning twice reported a failure\n");
        return EXIT_FAILURE;
    }
    /* And closing an abandoned runtime is a no-op rather than a second drain. */
    if (shadowspill_runtime_close(runtime) != SHADOWSPILL_STATUS_OK) {
        (void)fprintf(stderr, "closing an abandoned runtime reported a failure\n");
        return EXIT_FAILURE;
    }

    shadowspill_runtime_destroy(runtime);
    shadowspill_backend_destroy(&mock);
    (void)printf("runtime abandon canary passed\n");
    return EXIT_SUCCESS;
}
