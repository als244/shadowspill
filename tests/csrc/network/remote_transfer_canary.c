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

#include <stdatomic.h>
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
    /* Both directions for each of the two local pool kinds. */
    if (description == NULL || description->lane_count != 4U) {
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

static int all_zero(const uint64_t *words, uint64_t bytes) {
    for (uint64_t index = 0U; index < bytes / sizeof(*words); ++index) {
        if (words[index] != 0U) {
            return 0;
        }
    }
    return 1;
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
    uint64_t handle = 0U;
    if (arm->route->operations->copy(
            arm->route->lane, arm->destination, arm->source, arm->bytes, &handle
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

/*
 * Consecutive transfers on one lane, with nothing waited for between them.
 *
 * Every other case here issues a copy and synchronizes, so the lane's chunk
 * ring drains at every transfer boundary and the pipeline that spans them is
 * never entered. This is the case that enters it: `PIECES` transfers handed to
 * the lane back to back, then one `synchronize`. The lane must post a later
 * transfer's chunks into slots an earlier one has released, retire completions
 * in order across the boundary, and store a NIC count that never goes
 * backwards -- and if it gets any of that wrong the payload comes back wrong,
 * because each piece carries its own index.
 *
 * It is also the only case where a transfer is in flight while the next one is
 * being planned, which is what `chunks_planned` is read under a lock for.
 */
#define PIECES 8U

static int consecutive_transfers_pipeline(
    ShadowSpillRuntime *runtime,
    const ShadowSpillAllocation *device,
    const ShadowSpillAllocation *stored,
    uint64_t bytes
) {
    const uint64_t piece = (bytes / PIECES) & ~(uint64_t)7U;
    if (piece == 0U) {
        return 0;
    }
    ShadowSpillRouteState *const evict = &runtime->routes[EVICT_ROUTE];
    ShadowSpillRouteState *const fetch = &runtime->routes[FETCH_ROUTE];
    uint64_t lane_handle = 0U;
    int failed = 0;

    fill(device->pointer, piece * PIECES, 0xC3C3C3C3U);
    for (unsigned index = 0U; index < PIECES && !failed; ++index) {
        const uint64_t offset = (uint64_t)index * piece;
        failed = evict->operations->copy(
            evict->lane, (char *)stored->pointer + offset,
            (char *)device->pointer + offset, piece, &lane_handle
        ) != 0;
    }
    failed = failed || evict->operations->synchronize(evict->lane) != 0;
    if (!failed) {
        memset(device->pointer, 0, piece * PIECES);
    }
    for (unsigned index = 0U; index < PIECES && !failed; ++index) {
        const uint64_t offset = (uint64_t)index * piece;
        failed = fetch->operations->copy(
            fetch->lane, (char *)device->pointer + offset,
            (char *)stored->pointer + offset, piece, &lane_handle
        ) != 0;
    }
    failed = failed || fetch->operations->synchronize(fetch->lane) != 0;
    failed = failed || check(device->pointer, piece * PIECES, 0xC3C3C3C3U) != 0;
    if (failed) {
        fprintf(
            stderr,
            "remote transfer: %u consecutive transfers did not survive the "
            "pipeline\n",
            PIECES
        );
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
    /* No trace is running here, so every lane answers 0 and this is unread. */
    uint64_t lane_handle = 0U;
    const double evict_started = seconds_now();
    for (uint64_t pass = 0U; pass < repeats && !failed; ++pass) {
        failed = failed || evict->operations->copy(
            evict->lane, stored.pointer, device.pointer, payload_bytes, &lane_handle
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
            fetch->lane, device.pointer, stored.pointer, payload_bytes, &lane_handle
        ) != 0;
        failed = failed || fetch->operations->synchronize(fetch->lane) != 0;
    }
    const double fetch_seconds = (seconds_now() - fetch_started) / (double)repeats;

    failed = failed || check(device.pointer, payload_bytes, 0xA5A5A5A5U) != 0;

    /*
     * A transfer the runtime ordered behind something. The dependency is an
     * event recorded on the compute stream behind a value wait this canary
     * releases from the host, so until the release the event is not complete
     * and a lane honouring the order has not touched the pool. On the direct
     * path nothing stands between the NIC and the pool, so this is the one
     * obligation the lane's thread has before it posts. An evict that posted
     * early carries the pattern from before the release; a fetch that did has
     * written its destination before it.
     */
    ShadowSpillBackendSignals release_signals = 0U;
    uint64_t *release = NULL;
    ShadowSpillBackendEvent dependency = 0U;
    failed = failed || mock.allocate_signals(
        mock.state, 1U, &release_signals, &release
    ) != 0;
    failed = failed || mock.create_event(mock.state, &dependency, 0U) != 0;
    const struct timespec long_enough = {.tv_sec = 0, .tv_nsec = 20000000L};
    if (!failed) {
        failed = mock.wait_value(mock.state, compute, release_signals, 0U, 1U) != 0 ||
            mock.record_event(mock.state, dependency, compute) != 0 ||
            evict->operations->wait(evict->lane, dependency) != 0 ||
            evict->operations->copy(
                evict->lane, stored.pointer, device.pointer, payload_bytes,
                &lane_handle
            ) != 0;
        /* Long enough for a post that ignored the gate to have completed. */
        (void)nanosleep(&long_enough, NULL);
        fill(device.pointer, payload_bytes, 0x3C3C3C3CU);
        atomic_store_explicit((_Atomic uint64_t *)release, 1U, memory_order_release);
        failed = failed || evict->operations->synchronize(evict->lane) != 0;
        memset(device.pointer, 0, payload_bytes);
        failed = failed || fetch->operations->copy(
            fetch->lane, device.pointer, stored.pointer, payload_bytes,
            &lane_handle
        ) != 0 || fetch->operations->synchronize(fetch->lane) != 0;
        failed = failed || check(device.pointer, payload_bytes, 0x3C3C3C3CU) != 0;
        if (failed) {
            fprintf(stderr, "remote transfer: an evict posted before its dependency\n");
        }
    }
    if (!failed) {
        memset(device.pointer, 0, payload_bytes);
        failed = mock.wait_value(mock.state, compute, release_signals, 0U, 2U) != 0 ||
            mock.record_event(mock.state, dependency, compute) != 0 ||
            fetch->operations->wait(fetch->lane, dependency) != 0 ||
            fetch->operations->copy(
                fetch->lane, device.pointer, stored.pointer, payload_bytes,
                &lane_handle
            ) != 0;
        (void)nanosleep(&long_enough, NULL);
        const int untouched = all_zero(device.pointer, payload_bytes);
        atomic_store_explicit((_Atomic uint64_t *)release, 2U, memory_order_release);
        failed = failed || fetch->operations->synchronize(fetch->lane) != 0;
        failed = failed || check(device.pointer, payload_bytes, 0x3C3C3C3CU) != 0;
        if (!untouched) {
            fprintf(stderr, "remote transfer: a fetch posted before its dependency\n");
            failed = 1;
        }
    }

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
            second_device.pointer, payload_bytes, &lane_handle
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

    failed = failed || consecutive_transfers_pipeline(
        runtime, &device, &stored, payload_bytes
    ) != 0;

    if (runtime != NULL) {
        shadowspill_runtime_destroy(runtime);
    }
    if (dependency != 0U) {
        (void)mock.destroy_event(mock.state, dependency);
    }
    if (release_signals != 0U) {
        (void)mock.free_signals(mock.state, release_signals);
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
            "sequentially, both directions at once, and %u consecutive "
            "transfers pipelined\n"
            "  evict %.6f s (%.1f MiB/s), fetch %.6f s (%.1f MiB/s)\n",
            (unsigned long long)(payload_bytes >> 10U), host, port, PIECES,
            evict_seconds, mib / evict_seconds,
            fetch_seconds, mib / fetch_seconds
        );
    }
    return failed ? 1 : 0;
}
