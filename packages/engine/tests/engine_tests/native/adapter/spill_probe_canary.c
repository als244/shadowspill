/* Capacity-bound probes must include lane resources, release them on failures,
   and leave the process-global execution allocator untouched. */
#define _GNU_SOURCE
#include <shadowspill/pytorch_adapter.h>
#include <shadowspill/ssd.h>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define MIB(n) ((uint64_t)(n) << 20U)
#define REQUIRE(condition) do { if (!(condition)) { \
    fprintf(stderr, "spill probe canary: line %d: %s\n", __LINE__, #condition); \
    return 1; } } while (0)

static int balanced(const ShadowSpillBackendStatistics *s) {
    return s->device_allocations == s->device_frees &&
        s->bytes_device_allocated == s->bytes_device_freed &&
        s->pinned_host_registrations == s->pinned_host_unregistrations &&
        s->bytes_pinned_host_registered == s->bytes_pinned_host_unregistered &&
        s->streams_created == s->streams_destroyed &&
        s->events_created == s->events_destroyed;
}

int main(int argc, char **argv) {
    REQUIRE(argc == 2 || argc == 4);
    const char *extension = argc == 4 ? argv[2] : NULL;
    char directory[4096] = {0};
    if (extension != NULL) {
        REQUIRE(snprintf(directory, sizeof(directory), "%s/probe-XXXXXX", argv[3]) > 0);
        REQUIRE(mkdtemp(directory) != NULL);
    }
    ShadowSpillSSDConfiguration config = {
        .directory = directory, .staging_bytes = MIB(1),
        .chunk_bytes = 65536, .queue_depth = 2,
    };
    const uint8_t kind = extension == NULL
        ? SHADOWSPILL_POOL_PINNED_HOST : SHADOWSPILL_SSD_POOL_KIND;
    void *configuration = extension == NULL ? NULL : &config;
    void *probe = (void *)(uintptr_t)1U;
    REQUIRE(shadowspill_pytorch_spill_probe_open(NULL, 0, NULL, &probe) ==
            SHADOWSPILL_STATUS_INVALID_ARGUMENT && probe == NULL);
    REQUIRE(shadowspill_pytorch_spill_probe_open(argv[1], 0, "/missing", &probe) !=
            SHADOWSPILL_STATUS_OK && probe == NULL);
    REQUIRE(shadowspill_pytorch_spill_probe_open(argv[1], 0, extension, &probe) ==
            SHADOWSPILL_STATUS_OK);
    ShadowSpillBackendPhysicalMemory physical = {0};
    ShadowSpillBackendStatistics statistics = {0};
    ShadowSpillPytorchPhysicalAdmission admission = {0};
    REQUIRE(shadowspill_pytorch_spill_probe_snapshot(probe, &physical, &statistics) ==
            SHADOWSPILL_STATUS_OK && balanced(&statistics));
    uint64_t values[10] = {0};
    REQUIRE(shadowspill_pytorch_spill_probe_measure(probe, 4096, 65536, 1, 2, values) ==
            SHADOWSPILL_STATUS_INVALID_ARGUMENT);
    REQUIRE(shadowspill_pytorch_spill_probe_acquire(probe, kind, configuration,
            0, MIB(64), 1) == SHADOWSPILL_STATUS_INVALID_ARGUMENT);
    /* Unknown pool kind fails after opening the device pool. It must unwind
       that allocation and leave the session available for another probe. */
    REQUIRE(shadowspill_pytorch_spill_probe_acquire(probe, 233U, NULL,
            MIB(2), MIB(64), 1) != SHADOWSPILL_STATUS_OK);
    REQUIRE(shadowspill_pytorch_spill_probe_snapshot(probe, &physical, &statistics) ==
            SHADOWSPILL_STATUS_OK && balanced(&statistics));
    if (extension != NULL) {
        /* The import scratch and first lane fit, but the second lane does
           not. Release the first lane's registered staging on this path. */
        const uint64_t registered = statistics.bytes_pinned_host_registered;
        config.staging_bytes = 4U * config.chunk_bytes;
        REQUIRE(shadowspill_pytorch_spill_probe_acquire(probe, kind, &config,
                MIB(2), MIB(64), 1) != SHADOWSPILL_STATUS_OK);
        REQUIRE(shadowspill_pytorch_spill_probe_snapshot(probe, &physical, &statistics) ==
                SHADOWSPILL_STATUS_OK && balanced(&statistics));
        REQUIRE(statistics.bytes_pinned_host_registered > registered);
        config.staging_bytes = MIB(1);
    }
    for (unsigned cycle = 0; cycle < 3; ++cycle) {
        const uint64_t allocated = statistics.bytes_device_allocated;
        const uint64_t registered = statistics.bytes_pinned_host_registered;
        REQUIRE(shadowspill_pytorch_spill_probe_acquire(probe, kind, configuration,
                MIB(2), MIB(64), 1) == SHADOWSPILL_STATUS_OK);
        REQUIRE(shadowspill_pytorch_spill_probe_snapshot(probe, &physical, &statistics) ==
                SHADOWSPILL_STATUS_OK);
        REQUIRE(statistics.bytes_device_allocated - allocated == MIB(2));
        const uint64_t pinned = statistics.bytes_pinned_host_registered - registered;
        if (extension == NULL) REQUIRE(pinned == MIB(64));
        else REQUIRE(pinned > 0 && pinned < MIB(64)); /* real SSD lane staging */
        REQUIRE(shadowspill_pytorch_spill_probe_acquire(probe, kind, configuration,
                MIB(2), MIB(64), 1) == SHADOWSPILL_STATUS_INVALID_ARGUMENT);
        REQUIRE(shadowspill_pytorch_spill_probe_measure(probe, 4096, 65536, 1, 2, values) ==
                SHADOWSPILL_STATUS_OK);
        REQUIRE(shadowspill_pytorch_physical_admission(&admission) ==
                SHADOWSPILL_STATUS_CLOSED);
        shadowspill_pytorch_spill_probe_release(probe);
        shadowspill_pytorch_spill_probe_release(probe);
        REQUIRE(shadowspill_pytorch_spill_probe_snapshot(probe, &physical, &statistics) ==
                SHADOWSPILL_STATUS_OK && balanced(&statistics));
    }
    shadowspill_pytorch_spill_probe_destroy(probe);
    shadowspill_pytorch_spill_probe_destroy(NULL);
    REQUIRE(shadowspill_pytorch_spill_probe_create(argv[1], 0, extension,
            kind, configuration, MIB(2), &probe) == SHADOWSPILL_STATUS_OK);
    REQUIRE(shadowspill_pytorch_spill_probe_snapshot(probe, &physical, &statistics) ==
            SHADOWSPILL_STATUS_OK && statistics.bytes_device_allocated == MIB(2));
    shadowspill_pytorch_spill_probe_destroy(probe); /* active-pool close */
    REQUIRE(shadowspill_pytorch_physical_admission(&admission) ==
            SHADOWSPILL_STATUS_CLOSED);
    if (extension != NULL) REQUIRE(rmdir(directory) == 0);
    return 0;
}
