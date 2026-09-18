/*
 * A pool whose memory is on another machine behaves like one whose memory is
 * here.
 *
 * What this proves is the claim the kind lookup rests on: nothing in the
 * memory subsystem reads through a pool's address, so a region a daemon
 * allocated elsewhere serves leases, ranges and retirement exactly as a local
 * one does. Every pointer below is an address in the daemon's space and is
 * never dereferenced.
 *
 * It needs a daemon, named by SHADOWSPILL_NETWORK_PEER as host:port, and skips
 * when there is none -- which is also what the `network` ctest label means at
 * this stage: a daemon is reachable, not that a NIC moved a byte.
 */

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <shadowspill/backend_mock.h>
#include <shadowspill/runtime.h>

#include "../../../csrc/network/internal.h"

/* 256 KiB against the 64 KiB limit `main` forces below, so the payload crosses
   in four pieces rather than one. */
#define EDGE_WORDS (32U << 10U)
#define EDGE_MESSAGE_BYTES "65536"
#define REMOTE_POOL_ID 2U
#define REMOTE_CAPACITY (1U << 20U)

/* Where the network library is, relative to the build directory the canary
   runs from. The ctest entry sets it; this is the fallback for running it by
   hand from a build tree. */
static const char *library_path(void) {
    const char *configured = getenv("SHADOWSPILL_NETWORK_LIBRARY");
    return configured != NULL ? configured : "./libshadowspill_network.so";
}

/*
 * Load the library exactly as the layer above the runtime does: dlopen, one
 * dlsym, read the descriptor. Nothing of the library has run when this
 * returns -- which is the property being relied on, since this box may have no
 * daemon at all and the load must still succeed.
 */
static const ShadowSpillLibraryDescription *load_library(void **handle) {
    *handle = dlopen(library_path(), RTLD_NOW | RTLD_LOCAL);
    if (*handle == NULL) {
        fprintf(stderr, "remote pool: %s\n", dlerror());
        return NULL;
    }
    union {
        void *object;
        ShadowSpillLibraryDescribe describe;
    } describe = {
        .object = dlsym(*handle, SHADOWSPILL_LIBRARY_DESCRIBE_SYMBOL)
    };
    if (describe.object == NULL) {
        fprintf(stderr, "remote pool: the library exports no descriptor\n");
        return NULL;
    }
    const ShadowSpillLibraryDescription *description = describe.describe();
    if (description == NULL ||
        description->abi_version != SHADOWSPILL_ABI_VERSION ||
        description->pool_memory_count != 1U) {
        fprintf(stderr, "remote pool: the descriptor is not this build's\n");
        return NULL;
    }
    return description;
}

/* Split "host:port" in place. */
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

#define LEASE_COUNT 4U

/*
 * Four leases with a free in the middle, so best-fit has a hole to weigh and
 * retirement has something to retire. What comes back is each lease's offset
 * from the first, which is the only thing about a remote address this process
 * may look at.
 */
static int lease_offsets(
    ShadowSpillRuntime *runtime,
    uint32_t pool_id,
    ShadowSpillBackendStream stream,
    uint64_t offsets[LEASE_COUNT]
) {
    ShadowSpillAllocation allocations[LEASE_COUNT] = {{0}};
    const uint64_t sizes[LEASE_COUNT] = {4096U, 2048U, 4096U, 2048U};
    for (uint32_t index = 0U; index < 3U; ++index) {
        if (shadowspill_memory_pool_allocate(
                runtime, pool_id, sizes[index], 1U, stream, &allocations[index]
            ) != SHADOWSPILL_STATUS_OK) {
            fprintf(stderr, "remote pool: lease %u refused on pool %u\n",
                    index, pool_id);
            return -1;
        }
    }
    if (shadowspill_memory_pool_free(
            runtime, pool_id, allocations[1].allocation_id, stream
        ) != SHADOWSPILL_STATUS_OK ||
        shadowspill_memory_pool_allocate(
            runtime, pool_id, sizes[3], 1U, stream, &allocations[3]
        ) != SHADOWSPILL_STATUS_OK) {
        fprintf(stderr, "remote pool: free or reuse refused on pool %u\n",
                pool_id);
        return -1;
    }
    const char *const base = allocations[0].pointer;
    for (uint32_t index = 0U; index < LEASE_COUNT; ++index) {
        if (allocations[index].charged_bytes != sizes[index]) {
            fprintf(stderr, "remote pool: lease %u mischarged on pool %u\n",
                    index, pool_id);
            return -1;
        }
        /* Pointer arithmetic on an address this machine cannot read. That it
           is meaningful here and meaningless to dereference is the whole
           point. */
        offsets[index] =
            (uint64_t)((const char *)allocations[index].pointer - base);
    }
    return 0;
}

/*
 * A device pool, a pinned-host pool and a remote pool, with the two ordinary
 * routes between the first two. The remote pool has no route: there is no lane
 * that moves bytes to it yet, and a route whose kind pair no lane serves is
 * refused at create -- correctly.
 */
static int leases_behave_as_they_do_locally(
    const ShadowSpillLibraryDescription *library,
    const char *host,
    const char *port
) {
    ShadowSpillBackend mock = {0};
    const ShadowSpillMockBackendConfig mock_config = {0};
    if (shadowspill_mock_backend_create(&mock_config, &mock) != 0) {
        fprintf(stderr, "remote pool: the mock backend would not start\n");
        return -1;
    }
    ShadowSpillMockRuntimeTopology local;
    /* The pinned-host pool gets the remote pool's capacity: the comparison
       below is of two kinds, and a different capacity would make it a
       comparison of two sizes. */
    shadowspill_mock_runtime_topology(
        &mock, REMOTE_CAPACITY, REMOTE_CAPACITY, 1U, 1000U, &local
    );

    ShadowSpillRemotePoolConfiguration remote = {
        .host = host, .port = port, .selector = "host"
    };
    ShadowSpillMemoryPoolDescription pools[3] = {
        local.pools[0],
        local.pools[1],
        {
            .pool_id = REMOTE_POOL_ID,
            .kind = SHADOWSPILL_POOL_REMOTE,
            .capacity_bytes = REMOTE_CAPACITY,
            .minimum_alignment = 1U,
            .configuration = &remote,
        },
    };
    ShadowSpillRuntimeConfig config = local.runtime;
    config.pools = pools;
    config.pool_count = 3U;
    config.pool_memory = library->pool_memory;
    config.pool_memory_count = library->pool_memory_count;

    ShadowSpillRuntime *runtime = NULL;
    ShadowSpillBackendStream compute = 0U;
    int failed = shadowspill_runtime_create(&config, &runtime) !=
            SHADOWSPILL_STATUS_OK ||
        mock.create_stream(mock.state, &compute) != 0;
    if (failed) {
        fprintf(
            stderr,
            "remote pool: create failed; is a daemon listening on %s:%s?\n",
            host, port
        );
    }

    /*
     * The same sequence on the pinned-host pool and on the remote one, and the
     * two must lay out identically. Comparing offsets rather than asserting a
     * placement is the honest form of the claim: it says the remote pool
     * behaves as a local one does, whatever the allocator's policy happens to
     * be, instead of restating that policy here where it would rot.
     */
    uint64_t local_offsets[LEASE_COUNT] = {0};
    uint64_t remote_offsets[LEASE_COUNT] = {0};
    failed = failed || lease_offsets(runtime, 1U, compute, local_offsets) != 0;
    failed = failed ||
        lease_offsets(runtime, REMOTE_POOL_ID, compute, remote_offsets) != 0;
    for (uint32_t index = 0U; !failed && index < LEASE_COUNT; ++index) {
        if (local_offsets[index] != remote_offsets[index]) {
            fprintf(
                stderr,
                "remote pool: lease %u sits at +%llu, local sits at +%llu\n",
                index, (unsigned long long)remote_offsets[index],
                (unsigned long long)local_offsets[index]
            );
            failed = 1;
        }
    }

    /*
     * Bytes across the pool's edge, in more pieces than one.
     *
     * `write` and `read` are how state reaches a region this process cannot
     * address, and the port will not carry a message longer than its own
     * limit -- a gigabyte here -- so a large enough object crosses in pieces.
     * `SHADOWSPILL_NETWORK_MESSAGE_BYTES` lowers that limit so the pieces are
     * reachable with a payload measured in kilobytes.
     *
     * A loop that posted only the first piece, or computed the second piece's
     * offset wrongly, or returned on a completion belonging to another work
     * request, all fail here. Until now nothing exercised `write` or `read` at
     * all: the gate was the first thing that ran them.
     */
    static uint64_t written[EDGE_WORDS];
    static uint64_t restored[EDGE_WORDS];
    for (uint64_t index = 0U; index < EDGE_WORDS; ++index) {
        written[index] = index * 0x9E3779B97F4A7C15ULL;
        restored[index] = 0U;
    }
    const ShadowSpillObjectDescription crossing = {
        .object_id = 41U,
        .size_bytes = sizeof(written),
        .initial_version = 1U,
        .retain_spill_copy = 1U,
        .initial_pool_id = REMOTE_POOL_ID,
        .initially_resident = 1U,
    };
    failed = failed ||
        shadowspill_register_object(runtime, &crossing) !=
            SHADOWSPILL_STATUS_OK ||
        shadowspill_write_object(
            runtime, crossing.object_id, REMOTE_POOL_ID,
            written, sizeof(written)
        ) != SHADOWSPILL_STATUS_OK ||
        shadowspill_read_object(
            runtime, crossing.object_id, REMOTE_POOL_ID,
            restored, sizeof(restored)
        ) != SHADOWSPILL_STATUS_OK;
    if (failed) {
        fprintf(stderr, "remote pool: state would not cross the edge\n");
    }
    for (uint64_t index = 0U; !failed && index < EDGE_WORDS; ++index) {
        if (restored[index] != written[index]) {
            fprintf(
                stderr,
                "remote pool: word %llu came back %llx, expected %llx\n",
                (unsigned long long)index,
                (unsigned long long)restored[index],
                (unsigned long long)written[index]
            );
            failed = 1;
        }
    }

    ShadowSpillMemoryPoolStatistics statistics = {0};
    failed = failed || shadowspill_memory_pool_statistics(
        runtime, REMOTE_POOL_ID, &statistics
    ) != SHADOWSPILL_STATUS_OK;
    if (!failed && statistics.capacity_bytes != REMOTE_CAPACITY) {
        fprintf(
            stderr,
            "remote pool: capacity reads %llu, the config declared %u\n",
            (unsigned long long)statistics.capacity_bytes, REMOTE_CAPACITY
        );
        failed = 1;
    }

    /* Destroy gives the region back over the control channel and hangs up. */
    if (runtime != NULL) {
        shadowspill_runtime_destroy(runtime);
    }
    if (compute != 0U) {
        (void)mock.destroy_stream(mock.state, compute);
    }
    shadowspill_backend_destroy(&mock);
    return failed ? -1 : 0;
}

/* A pool whose kind no entry serves is refused, and the refusal is the lookup
   failing rather than a range check -- so it happens with the library absent,
   not with a kind out of range. */
static int an_unserved_kind_is_refused(void) {
    ShadowSpillBackend mock = {0};
    const ShadowSpillMockBackendConfig mock_config = {0};
    if (shadowspill_mock_backend_create(&mock_config, &mock) != 0) {
        fprintf(stderr, "remote pool: the mock backend would not start\n");
        return -1;
    }
    ShadowSpillMockRuntimeTopology local;
    shadowspill_mock_runtime_topology(&mock, 4096U, 4096U, 1U, 1000U, &local);
    ShadowSpillMemoryPoolDescription pools[3] = {
        local.pools[0], local.pools[1],
        {
            .pool_id = REMOTE_POOL_ID,
            .kind = SHADOWSPILL_POOL_REMOTE,
            .capacity_bytes = REMOTE_CAPACITY,
            .minimum_alignment = 1U,
        },
    };
    ShadowSpillRuntimeConfig config = local.runtime;
    config.pools = pools;
    config.pool_count = 3U;
    ShadowSpillRuntime *runtime = NULL;
    const ShadowSpillStatus status = shadowspill_runtime_create(
        &config, &runtime
    );
    if (runtime != NULL) {
        shadowspill_runtime_destroy(runtime);
    }
    shadowspill_backend_destroy(&mock);
    if (status != SHADOWSPILL_STATUS_INVALID_ARGUMENT) {
        fprintf(
            stderr,
            "remote pool: an unregistered kind returned %d, expected %d\n",
            (int)status, (int)SHADOWSPILL_STATUS_INVALID_ARGUMENT
        );
        return -1;
    }
    return 0;
}

/*
 * No daemon there. Create must fail, promptly and with nothing orphaned --
 * this is what a mistyped host or a peer that has not been started looks like,
 * and it must not be a hang.
 */
static int an_unreachable_daemon_fails_create(
    const ShadowSpillLibraryDescription *library
) {
    ShadowSpillBackend mock = {0};
    const ShadowSpillMockBackendConfig mock_config = {0};
    if (shadowspill_mock_backend_create(&mock_config, &mock) != 0) {
        fprintf(stderr, "remote pool: the mock backend would not start\n");
        return -1;
    }
    ShadowSpillMockRuntimeTopology local;
    shadowspill_mock_runtime_topology(
        &mock, REMOTE_CAPACITY, REMOTE_CAPACITY, 1U, 1000U, &local
    );
    /* Port 1 is reserved and nothing listens on it. */
    ShadowSpillRemotePoolConfiguration remote = {
        .host = "127.0.0.1", .port = "1", .selector = "host"
    };
    ShadowSpillMemoryPoolDescription pools[3] = {
        local.pools[0], local.pools[1],
        {
            .pool_id = REMOTE_POOL_ID,
            .kind = SHADOWSPILL_POOL_REMOTE,
            .capacity_bytes = REMOTE_CAPACITY,
            .minimum_alignment = 1U,
            .configuration = &remote,
        },
    };
    ShadowSpillRuntimeConfig config = local.runtime;
    config.pools = pools;
    config.pool_count = 3U;
    config.pool_memory = library->pool_memory;
    config.pool_memory_count = library->pool_memory_count;
    ShadowSpillRuntime *runtime = NULL;
    const ShadowSpillStatus status = shadowspill_runtime_create(
        &config, &runtime
    );
    if (runtime != NULL) {
        shadowspill_runtime_destroy(runtime);
    }
    shadowspill_backend_destroy(&mock);
    if (status == SHADOWSPILL_STATUS_OK) {
        fprintf(stderr, "remote pool: create succeeded with no daemon\n");
        return -1;
    }
    return 0;
}

int main(void) {
    char *const peer = getenv("SHADOWSPILL_NETWORK_PEER");
    if (peer == NULL || peer[0] == '\0') {
        fprintf(
            stderr,
            "remote pool: SHADOWSPILL_NETWORK_PEER is unset; skipping\n"
        );
        return 0;
    }
    char copy[256];
    const char *host = NULL;
    const char *port = NULL;
    if (snprintf(copy, sizeof(copy), "%s", peer) < 0 ||
        split_peer(copy, &host, &port) != 0) {
        fprintf(
            stderr,
            "remote pool: SHADOWSPILL_NETWORK_PEER must read host:port\n"
        );
        return 1;
    }
    /*
     * Lower the largest message the state path will post, so the round trip
     * below crosses in four pieces rather than one.
     *
     * What this covers is the piece loop and the completion accounting: every
     * piece posted at the right offset, and each one waited for by its own
     * work request rather than by whichever completion arrived. A payload that
     * would defeat the *length* arithmetic has to exceed four gigabytes, which
     * is not something to allocate in a canary -- so that part is reasoned
     * about rather than tested, and this covers the loop that carries it.
     */
    if (setenv("SHADOWSPILL_NETWORK_MESSAGE_BYTES", EDGE_MESSAGE_BYTES, 1) != 0) {
        fprintf(stderr, "remote pool: could not set the message limit\n");
        return 1;
    }
    void *handle = NULL;
    const ShadowSpillLibraryDescription *library = load_library(&handle);
    int failed = library == NULL;
    failed = failed || an_unserved_kind_is_refused() != 0;
    failed = failed || an_unreachable_daemon_fails_create(library) != 0;
    failed = failed ||
        leases_behave_as_they_do_locally(library, host, port) != 0;
    if (handle != NULL) {
        (void)dlclose(handle);
    }
    if (!failed) {
        printf("remote pool canary passed against %s:%s\n", host, port);
    }
    return failed ? 1 : 0;
}
