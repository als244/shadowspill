/*
 * Bytes move to another machine and come back unchanged.
 *
 * The mock backend's copies are real memcpy and its device memory is ordinary
 * host memory, so the whole staged path runs on a CPU with a genuine NIC in
 * the middle: device -> ring -> RDMA write -> tubingen, and back. What is
 * being certified is the lane, not the accelerator.
 *
 * Needs a daemon, named by SHADOWSPILL_NETWORK_PEER as host:port, and skips
 * when there is none.
 */

#include <dlfcn.h>
#include <pthread.h>
#include <time.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <shadowspill/backend_mock.h>
#include <shadowspill/runtime.h>

#include "../../../csrc/src/runtime/internal.h"
#include "../../../csrc/src/runtime/transfers/internal.h"
#include "../../../csrc/network/internal.h"

#define DEVICE_POOL 0U
#define REMOTE_POOL 1U
#define FETCH_ROUTE 0U
#define EVICT_ROUTE 1U
/* Overridable so the same canary can measure a rate and be interrupted
   mid-transfer, not only prove correctness at a small size. */
#define DEFAULT_payload_bytes (256U << 10U)

static const char *library_path(void) {
    const char *configured = getenv("SHADOWSPILL_NETWORK_LIBRARY");
    return configured != NULL ? configured : "./libshadowspill_network.so";
}

static const ShadowSpillLibraryDescription *load_library(void **handle) {
    *handle = dlopen(library_path(), RTLD_NOW | RTLD_LOCAL);
    if (*handle == NULL) {
        fprintf(stderr, "remote transfer: %s\n", dlerror());
        return NULL;
    }
    union {
        void *object;
        ShadowSpillLibraryDescribe describe;
    } describe = {
        .object = dlsym(*handle, SHADOWSPILL_LIBRARY_DESCRIBE_SYMBOL)
    };
    if (describe.object == NULL) {
        fprintf(stderr, "remote transfer: no descriptor\n");
        return NULL;
    }
    const ShadowSpillLibraryDescription *description = describe.describe();
    if (description == NULL || description->lane_count != 2U) {
        fprintf(stderr, "remote transfer: the library offers no lanes\n");
        return NULL;
    }
    return description;
}

static double seconds_now(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
}

static int split_peer(char *peer, const char **host, const char **port) {
    char *const colon = strrchr(peer, ':');
    if (colon == NULL || colon == peer || colon[1] == '\0') {
        return -1;
    }
    *colon = '\0';
    *host = peer;
    *port = colon + 1;
    return 0;
}

/* A pattern a wrong offset or a truncated transfer cannot reproduce: every
   8-byte word carries its own index. */
static void fill(uint64_t *words, uint64_t bytes, uint64_t salt) {
    for (uint64_t index = 0U; index < bytes / sizeof(*words); ++index) {
        words[index] = index ^ salt;
    }
}

static int check(const uint64_t *words, uint64_t bytes, uint64_t salt) {
    for (uint64_t index = 0U; index < bytes / sizeof(*words); ++index) {
        if (words[index] != (index ^ salt)) {
            fprintf(
                stderr,
                "remote transfer: word %llu reads %llx, expected %llx\n",
                (unsigned long long)index, (unsigned long long)words[index],
                (unsigned long long)(index ^ salt)
            );
            return -1;
        }
    }
    return 0;
}

/*
 * Both directions at once.
 *
 * Every check above runs an evict, then a fetch. Calibration does not: it
 * measures the two routes under simultaneous traffic, which is the point of
 * measuring them that way. That difference hid a deadlock -- the two lanes
 * shared a completion queue and took each other's completions, each recording
 * what it took where the other could not find it.
 *
 * Sequential transfers can never show this. Concurrent ones show it on the
 * first attempt.
 */
typedef struct ConcurrentArm {
    ShadowSpillRouteState *route;
    void *destination;
    const void *source;
    uint64_t bytes;
    int failed;
} ConcurrentArm;

static void *run_arm(void *argument) {
    ConcurrentArm *arm = argument;
    if (arm->route->operations->copy(
            arm->route->lane, arm->destination, arm->source, arm->bytes
        ) != 0 ||
        arm->route->operations->synchronize(arm->route->lane) != 0) {
        arm->failed = 1;
    }
    return NULL;
}

static int both_directions_at_once(
    ShadowSpillRuntime *runtime,
    const ShadowSpillAllocation *device,
    const ShadowSpillAllocation *stored,
    const ShadowSpillAllocation *second_device,
    const ShadowSpillAllocation *second_stored,
    uint64_t bytes
) {
    ConcurrentArm arms[2] = {
        {
            .route = &runtime->routes[EVICT_ROUTE],
            .destination = stored->pointer,
            .source = device->pointer,
            .bytes = bytes,
        },
        {
            .route = &runtime->routes[FETCH_ROUTE],
            .destination = second_device->pointer,
            .source = second_stored->pointer,
            .bytes = bytes,
        },
    };
    pthread_t threads[2];
    for (unsigned index = 0U; index < 2U; ++index) {
        if (pthread_create(&threads[index], NULL, run_arm, &arms[index]) != 0) {
            fprintf(stderr, "remote transfer: could not start arm %u\n", index);
            return -1;
        }
    }
    int failed = 0;
    for (unsigned index = 0U; index < 2U; ++index) {
        (void)pthread_join(threads[index], NULL);
        failed = failed || arms[index].failed;
    }
    if (failed) {
        fprintf(stderr, "remote transfer: a concurrent arm failed\n");
    }
    return failed ? -1 : 0;
}

int main(void) {
    uint64_t payload_bytes = DEFAULT_payload_bytes;
    const char *const configured = getenv("SHADOWSPILL_TRANSFER_BYTES");
    if (configured != NULL) {
        payload_bytes = strtoull(configured, NULL, 0);
        payload_bytes -= payload_bytes % sizeof(uint64_t);
    }
    if (payload_bytes == 0U) {
        payload_bytes = DEFAULT_payload_bytes;
    }
    /* Repeats, so a small transfer can be timed at all: one 4 KiB copy is
       microseconds and the clock's own cost would dominate. */
    uint64_t repeats = 1U;
    const char *const repeated = getenv("SHADOWSPILL_TRANSFER_REPEATS");
    if (repeated != NULL) {
        repeats = strtoull(repeated, NULL, 0);
    }
    if (repeats == 0U) {
        repeats = 1U;
    }
    const uint64_t region_bytes = payload_bytes * 2U;
    char *const peer = getenv("SHADOWSPILL_NETWORK_PEER");
    if (peer == NULL || peer[0] == '\0') {
        fprintf(stderr, "remote transfer: SHADOWSPILL_NETWORK_PEER unset; skipping\n");
        return 0;
    }
    char copy[256];
    const char *host = NULL;
    const char *port = NULL;
    if (snprintf(copy, sizeof(copy), "%s", peer) < 0 ||
        split_peer(copy, &host, &port) != 0) {
        fprintf(stderr, "remote transfer: PEER must read host:port\n");
        return 1;
    }
    void *handle = NULL;
    const ShadowSpillLibraryDescription *library = load_library(&handle);
    if (library == NULL) {
        return 1;
    }

    ShadowSpillBackend mock = {0};
    const ShadowSpillMockBackendConfig mock_config = {0};
    if (shadowspill_mock_backend_create(&mock_config, &mock) != 0) {
        fprintf(stderr, "remote transfer: the mock backend would not start\n");
        return 1;
    }
    ShadowSpillRemotePoolConfiguration remote = {
        .host = host, .port = port, .selector = "host"
    };
    const ShadowSpillMemoryPoolDescription pools[2] = {
        {
            .pool_id = DEVICE_POOL,
            .kind = SHADOWSPILL_POOL_DEVICE,
            .capacity_bytes = region_bytes,
            .minimum_alignment = 1U,
        },
        {
            .pool_id = REMOTE_POOL,
            .kind = SHADOWSPILL_POOL_REMOTE,
            .capacity_bytes = region_bytes,
            .minimum_alignment = 1U,
            .configuration = &remote,
        },
    };
    const ShadowSpillTransferRouteDescription routes[2] = {
        {
            .route_id = FETCH_ROUTE,
            .name = "remote_fetch",
            .source_pool_id = REMOTE_POOL,
            .destination_pool_id = DEVICE_POOL,
        },
        {
            .route_id = EVICT_ROUTE,
            .name = "remote_evict",
            .source_pool_id = DEVICE_POOL,
            .destination_pool_id = REMOTE_POOL,
        },
    };
    const ShadowSpillRuntimeConfig config = {
        .abi_version = SHADOWSPILL_ABI_VERSION,
        .backend = &mock,
        .pools = pools,
        .pool_count = 2U,
        .routes = routes,
        .route_count = 2U,
        .lanes = library->lanes,
        .lane_count = library->lane_count,
        .pool_memory = library->pool_memory,
        .pool_memory_count = library->pool_memory_count,
        .worker_poll_nanoseconds = 1000U,
    };
    ShadowSpillRuntime *runtime = NULL;
    if (shadowspill_runtime_create(&config, &runtime) != SHADOWSPILL_STATUS_OK) {
        fprintf(
            stderr,
            "remote transfer: create failed; is a daemon listening on %s:%s?\n",
            host, port
        );
        shadowspill_backend_destroy(&mock);
        return 1;
    }

    ShadowSpillBackendStream compute = 0U;
    int failed = mock.create_stream(mock.state, &compute) != 0;
    ShadowSpillAllocation device = {0};
    ShadowSpillAllocation stored = {0};
    failed = failed || shadowspill_memory_pool_allocate(
        runtime, DEVICE_POOL, payload_bytes, 8U, compute, &device
    ) != SHADOWSPILL_STATUS_OK;
    failed = failed || shadowspill_memory_pool_allocate(
        runtime, REMOTE_POOL, payload_bytes, 8U, compute, &stored
    ) != SHADOWSPILL_STATUS_OK;

    /* The device pool is the mock's malloc, so this is readable here. The
       remote one is a PROT_NONE reservation and reading it would fault, which
       is why only the lane ever touches it. */
    if (!failed) {
        fill(device.pointer, payload_bytes, 0xA5A5A5A5U);
    }

    /* Evict: device -> remote. Timed on its own, because a rate that includes
       filling and checking the pattern is not the lane's rate. */
    ShadowSpillRouteState *const evict = &runtime->routes[EVICT_ROUTE];
    const double evict_started = seconds_now();
    for (uint64_t pass = 0U; pass < repeats && !failed; ++pass) {
        failed = failed || evict->operations->copy(
            evict->lane, stored.pointer, device.pointer, payload_bytes
        ) != 0;
        failed = failed || evict->operations->synchronize(evict->lane) != 0;
    }
    const double evict_seconds = (seconds_now() - evict_started) / (double)repeats;

    /* Destroy the evidence: if the fetch does nothing, the check below fails
       rather than reading what was already there. */
    if (!failed) {
        memset(device.pointer, 0, payload_bytes);
    }

    /* Fetch: remote -> device. */
    ShadowSpillRouteState *const fetch = &runtime->routes[FETCH_ROUTE];
    const double fetch_started = seconds_now();
    for (uint64_t pass = 0U; pass < repeats && !failed; ++pass) {
        failed = failed || fetch->operations->copy(
            fetch->lane, device.pointer, stored.pointer, payload_bytes
        ) != 0;
        failed = failed || fetch->operations->synchronize(fetch->lane) != 0;
    }
    const double fetch_seconds = (seconds_now() - fetch_started) / (double)repeats;

    failed = failed || check(device.pointer, payload_bytes, 0xA5A5A5A5U) != 0;

    /* Now both directions at once, on separate ranges so the two arms do not
       race over the same bytes -- what is under test is the lanes, not the
       memory. */
    ShadowSpillAllocation second_device = {0};
    ShadowSpillAllocation second_stored = {0};
    failed = failed || shadowspill_memory_pool_allocate(
        runtime, DEVICE_POOL, payload_bytes, 8U, compute, &second_device
    ) != SHADOWSPILL_STATUS_OK;
    failed = failed || shadowspill_memory_pool_allocate(
        runtime, REMOTE_POOL, payload_bytes, 8U, compute, &second_stored
    ) != SHADOWSPILL_STATUS_OK;
    if (!failed) {
        /* Seed the second remote range so the concurrent fetch has something
           to bring back, and check it afterwards. */
        fill(second_device.pointer, payload_bytes, 0x5A5A5A5AU);
        failed = runtime->routes[EVICT_ROUTE].operations->copy(
            runtime->routes[EVICT_ROUTE].lane, second_stored.pointer,
            second_device.pointer, payload_bytes
        ) != 0 ||
        runtime->routes[EVICT_ROUTE].operations->synchronize(
            runtime->routes[EVICT_ROUTE].lane
        ) != 0;
        memset(second_device.pointer, 0, payload_bytes);
    }
    failed = failed || both_directions_at_once(
        runtime, &device, &stored, &second_device, &second_stored, payload_bytes
    ) != 0;
    failed = failed || check(second_device.pointer, payload_bytes, 0x5A5A5A5AU) != 0;

    if (runtime != NULL) {
        shadowspill_runtime_destroy(runtime);
    }
    if (compute != 0U) {
        (void)mock.destroy_stream(mock.state, compute);
    }
    shadowspill_backend_destroy(&mock);
    if (handle != NULL) {
        (void)dlclose(handle);
    }
    if (!failed) {
        const double mib = (double)payload_bytes / (double)(1U << 20U);
        printf(
            "remote transfer canary passed: %llu KiB to %s:%s and back, "
            "sequentially and both directions at once\n"
            "  evict %.6f s (%.1f MiB/s), fetch %.6f s (%.1f MiB/s)\n",
            (unsigned long long)(payload_bytes >> 10U), host, port,
            evict_seconds, mib / evict_seconds,
            fetch_seconds, mib / fetch_seconds
        );
    }
    return failed ? 1 : 0;
}
