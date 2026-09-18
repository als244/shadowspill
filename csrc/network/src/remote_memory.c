/* Where a Remote pool's memory comes from: a daemon on another machine. */

/* MAP_ANONYMOUS is not in the strict ISO C11 the tree compiles as; the same
   line and the same reason as memory_pool/internal.h. */
#define _DEFAULT_SOURCE

#include "../internal.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

/*
 * The `pool_memory` entry for SHADOWSPILL_POOL_REMOTE. It is reached by the
 * same lookup as the two kinds the runtime implements itself, and it is the
 * whole of what the memory subsystem learns about a remote region: an address,
 * a capacity, and a way to give both back.
 *
 * What comes back from `acquire` is an address in the *daemon's* space. It is
 * never dereferenced here, and the runtime does not dereference it either --
 * offsets are added to it and the result is handed to a lane. That is what
 * lets one pool implementation serve memory that is not on this machine.
 */

/*
 * WHY A POOL'S BASE IS A LOCAL ADDRESS THAT CANNOT BE READ.
 *
 * `acquire` reserves `capacity` bytes of this process's address space with
 * `PROT_NONE` and returns *that* as the pool's base, keeping the daemon's
 * address beside it. Two consequences, and both are the point.
 *
 * **The address is unique.** Two daemons on two machines can perfectly well
 * hand back the same numeric address -- they are addresses in two unrelated
 * address spaces -- so a lane given a bare remote address could not tell which
 * region, which key or which connection it belonged to. A local reservation is
 * unique by construction, because the kernel says so.
 *
 * **Dereferencing it faults.** The claim the whole extension rests on is that
 * nothing outside a lane reads through a pool address. `PROT_NONE` turns that
 * from a claim into an enforced invariant: code that breaks it stops at the
 * instruction that did, rather than reading plausible rubbish or corrupting
 * something that is genuinely mapped. It costs address space and no physical
 * memory.
 *
 * The lane translates back -- base + offset in local terms becomes remote
 * address + the same offset -- which it can do because it finds the region
 * from the address it was given.
 */
typedef struct RemoteRegion {
    /* These fields, in this order, are ShadowSpillRemoteRegion -- what a lane
       is allowed to see. Anything private goes after them. */
    ShadowSpillControlChannel channel;
    void *reservation;
    uint64_t capacity;
    uint64_t address;
    uint32_t key;
    ShadowSpillEndpoint endpoint;

    /* How many of this region's queue pairs a lane has taken. Two lanes serve
       a remote pool -- one per direction -- and they run at once, so each
       takes its own. */
    atomic_uint claimed_queue_pairs;
    /* Serialises `write`, which uses the queue pair no lane may claim. */
    pthread_mutex_t write_lock;
    struct RemoteRegion *next;
} RemoteRegion;

/*
 * The last queue pair is the region's own, for putting state into the pool
 * before any plan exists. Lanes claim from the front and never reach it, so a
 * write never contends with a transfer for a completion queue -- the same
 * separation that stopped the two lanes deadlocking each other.
 */
static uint32_t write_queue_pair(const RemoteRegion *region) {
    return region->endpoint.queue_pair_count - 1U;
}

int shadowspill_remote_region_claim_queue_pair(
    const ShadowSpillRemoteRegion *region
) {
    RemoteRegion *const owned = (RemoteRegion *)(uintptr_t)region;
    const unsigned claimed = atomic_fetch_add_explicit(
        &owned->claimed_queue_pairs, 1U, memory_order_acq_rel
    );
    /* One fewer than exist: the last is the region's own, for `write`. */
    if (claimed + 1U >= region->endpoint.queue_pair_count) {
        return -1;
    }
    return (int)claimed;
}

/*
 * Every region this process holds, so a lane can turn an address back into the
 * connection and key that serve it.
 *
 * A list rather than a table keyed by pool id: pool ids are the *runtime's*
 * and two runtimes in one process would collide, where a reservation is unique
 * process-wide. Entries live exactly as long as their pool -- added by
 * `acquire`, removed by `release` -- so the list holds nothing a pool does not.
 */
static RemoteRegion *regions;
static pthread_mutex_t regions_lock = PTHREAD_MUTEX_INITIALIZER;

const ShadowSpillRemoteRegion *shadowspill_remote_region_for(const void *address) {
    pthread_mutex_lock(&regions_lock);
    const RemoteRegion *found = NULL;
    for (const RemoteRegion *region = regions; region != NULL;
         region = region->next) {
        const char *const base = region->reservation;
        if ((const char *)address >= base &&
            (const char *)address < base + region->capacity) {
            found = region;
            break;
        }
    }
    pthread_mutex_unlock(&regions_lock);
    return (const ShadowSpillRemoteRegion *)found;
}

/*
 * Two steps, each of which can fail, unwound in reverse: connect, then
 * allocate. Nothing is left on either machine when this returns non-zero --
 * create fails and the runtime unwinds, rather than coming up with a pool that
 * has no memory behind it.
 *
 * Capacity is the pool's, declared in its configuration. A daemon that cannot
 * honour it fails here rather than serving less, so what a plan was built
 * against is what it runs against.
 */
static int remote_acquire(
    void *configuration, uint64_t capacity, void **base, void **state
) {
    const ShadowSpillRemotePoolConfiguration *remote = configuration;
    if (remote == NULL || remote->host == NULL || remote->port == NULL ||
        base == NULL || state == NULL) {
        return -1;
    }
    RemoteRegion *region = calloc(1U, sizeof(*region));
    if (region == NULL) {
        return -1;
    }
    region->channel.socket = -1;
    region->capacity = capacity;
    if (pthread_mutex_init(&region->write_lock, NULL) != 0) {
        free(region);
        return -1;
    }
    if (shadowspill_control_connect(
            &region->channel, remote->host, remote->port
        ) != 0) {
        /* The first thing that can go wrong and the one with the least to go
           on: a pool create reports a status and no message, so an address
           that does not resolve, a daemon that is not running and a port that
           is firewalled all arrive as the same number. Name what was tried.
           A host that resolves only through an ssh alias is the common case --
           `getaddrinfo` does not read ssh_config. */
        fprintf(
            stderr,
            "shadowspill network: could not reach the memory daemon at %s:%s\n",
            remote->host, remote->port
        );
        free(region);
        return -1;
    }
    if (shadowspill_control_allocate(
            &region->channel, capacity, remote->selector, &region->address,
            &region->key
        ) != 0) {
        shadowspill_control_close(&region->channel);
        free(region);
        return -1;
    }
    /* Address space only: no pages are committed, and any read or write of it
       faults at the instruction responsible. See the note above. */
    region->reservation = mmap(
        NULL, (size_t)capacity, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0
    );
    if (region->reservation == MAP_FAILED) {
        (void)shadowspill_control_free(&region->channel, region->address);
        shadowspill_control_close(&region->channel);
        free(region);
        return -1;
    }
    /*
     * Bring the queue pairs up here, so a pool that cannot move bytes fails at
     * create rather than at its first transfer. Four steps now, each unwound
     * in reverse: connect, allocate, reserve, handshake.
     */
    /*
     * At least one queue pair per direction, because the fetch lane and the
     * evict lane each take one and they run at the same time. Asking for fewer
     * would make one of them fail to claim; asking for more is the tuning
     * knob's business.
     */
    ShadowSpillNetworkTuning tuning;
    shadowspill_network_tuning_read(&tuning);
    if (shadowspill_endpoint_open(
            &region->endpoint, remote->host, 0U, tuning.queue_pairs + 1U
        ) != 0) {
        (void)munmap(region->reservation, (size_t)capacity);
        (void)shadowspill_control_free(&region->channel, region->address);
        shadowspill_control_close(&region->channel);
        free(region);
        return -1;
    }
    for (uint32_t index = 0U; index < region->endpoint.queue_pair_count;
         ++index) {
        ShadowSpillEndpointIdentity mine;
        ShadowSpillEndpointIdentity theirs;
        memset(&theirs, 0, sizeof(theirs));
        if (shadowspill_endpoint_identity(&region->endpoint, index, &mine) != 0 ||
            shadowspill_control_connect_endpoint(&region->channel, &mine, &theirs)
                != 0 ||
            shadowspill_endpoint_connect(&region->endpoint, index, &theirs) != 0) {
            shadowspill_endpoint_close(&region->endpoint);
            (void)munmap(region->reservation, (size_t)capacity);
            (void)shadowspill_control_free(&region->channel, region->address);
            shadowspill_control_close(&region->channel);
            free(region);
            return -1;
        }
    }
    pthread_mutex_lock(&regions_lock);
    region->next = regions;
    regions = region;
    pthread_mutex_unlock(&regions_lock);
    *base = region->reservation;
    *state = region;
    return 0;
}

/*
 * Move bytes across the pool's edge, for a region this process cannot address.
 *
 * `local` is ordinary memory the caller owns -- a model's weights on their way
 * into the spill pool before there is a plan, or a checkpoint's on their way
 * out -- so it is registered for the duration and unregistered after.
 * Registering rather than staging through a buffer means no copy at all, and
 * the cost is paid once per object rather than once per chunk.
 *
 * One body serves both directions because they differ in one opcode: the
 * registration, the bounds, the queue pair and the wait are the same work, and
 * a second copy of them would be a second place for the bounds check to be
 * wrong. `IBV_ACCESS_LOCAL_WRITE` covers both -- it is what the NIC needs in
 * order to write a read's result into `local`, and is harmless on a write.
 *
 * Synchronous: when this returns the bytes have landed. Both directions are
 * called while state is imported or exported, from that thread, with nothing
 * else moving.
 */
static int remote_transfer(
    void *state,
    uint64_t offset,
    void *local,
    uint64_t bytes,
    enum ibv_wr_opcode opcode,
    const char *what
) {
    RemoteRegion *region = state;
    if (region == NULL || local == NULL || bytes == 0U) {
        return region != NULL && bytes == 0U ? 0 : -1;
    }
    if (offset + bytes > region->capacity) {
        return -1;
    }
    struct ibv_mr *const registration = ibv_reg_mr(
        region->endpoint.protection_domain, local, (size_t)bytes,
        IBV_ACCESS_LOCAL_WRITE
    );
    if (registration == NULL) {
        fprintf(
            stderr,
            "shadowspill network: could not register %llu bytes of state for "
            "%s\n",
            (unsigned long long)bytes, what
        );
        return -1;
    }
    pthread_mutex_lock(&region->write_lock);
    const uint32_t index = write_queue_pair(region);
    /*
     * In pieces the port will carry. A queue pair refuses a work request
     * longer than `max_msg_sz`, and a scatter-gather element's length is 32
     * bits besides -- so an object larger than either has to cross in more
     * than one message. Sending it as one used to truncate the length to 32
     * bits and move the wrong number of bytes without saying so.
     *
     * One piece at a time, because this is the state path rather than the
     * transfer path: it already registers and unregisters per call, it runs
     * while nothing else is moving, and pipelining it would be optimising the
     * half of the system that was deliberately left unoptimised.
     */
    const uint64_t limit = region->endpoint.max_message_bytes;
    uint64_t moved = 0U;
    int failed = 0;
    while (!failed && moved < bytes) {
        const uint64_t piece = bytes - moved < limit ? bytes - moved : limit;
        struct ibv_sge element = {
            .addr = (uint64_t)(uintptr_t)local + moved,
            .length = (uint32_t)piece,
            .lkey = registration->lkey,
        };
        struct ibv_send_wr request = {
            /* Names the piece, so a completion says which one it is for. */
            .wr_id = moved,
            .sg_list = &element,
            .num_sge = 1,
            .opcode = opcode,
            .send_flags = IBV_SEND_SIGNALED,
            .wr = {.rdma = {
                .remote_addr = region->address + offset + moved,
                .rkey = region->key,
            }},
        };
        struct ibv_send_wr *bad = NULL;
        if (ibv_post_send(
                region->endpoint.queue_pairs[index], &request, &bad
            ) != 0) {
            failed = 1;
            break;
        }
        for (;;) {
            struct ibv_wc completion;
            const int taken = ibv_poll_cq(
                region->endpoint.completion_queues[index], 1, &completion
            );
            if (taken < 0) {
                failed = 1;
                break;
            }
            if (taken == 0) {
                continue;
            }
            if (completion.status != IBV_WC_SUCCESS) {
                fprintf(
                    stderr, "shadowspill network: state %s failed (%s)\n",
                    what, ibv_wc_status_str(completion.status)
                );
                failed = 1;
                break;
            }
            if (completion.wr_id != request.wr_id) {
                /* This queue pair is the region's own and is held under
                   `write_lock`, so nothing else may be posting to it. A
                   completion for another work request means that is no longer
                   true, and taking it for this one would report a transfer
                   that has not finished. */
                fprintf(
                    stderr,
                    "shadowspill network: state %s took a completion for "
                    "another request\n",
                    what
                );
                failed = 1;
                break;
            }
            break;
        }
        moved += piece;
    }
    pthread_mutex_unlock(&region->write_lock);
    (void)ibv_dereg_mr(registration);
    return failed ? -1 : 0;
}

static int remote_write(
    void *state, uint64_t offset, const void *source, uint64_t bytes
) {
    /* The cast drops a const the verb does not honour anyway: a write reads
       `local`, and the registration is the same either way. */
    return remote_transfer(
        state, offset, (void *)(uintptr_t)source, bytes, IBV_WR_RDMA_WRITE,
        "import"
    );
}

static int remote_read(
    void *state, uint64_t offset, void *destination, uint64_t bytes
) {
    return remote_transfer(
        state, offset, destination, bytes, IBV_WR_RDMA_READ, "export"
    );
}

/*
 * Give the region back and hang up. The free is asked for explicitly so an
 * orderly shutdown is orderly on both ends, but closing the socket frees it
 * too -- that backstop is what runs when this process dies without getting
 * here, and it is the reason the daemon keeps no registry.
 */
static int remote_release(void *state, void *base, uint64_t capacity) {
    (void)base;
    (void)capacity;
    RemoteRegion *region = state;
    if (region == NULL) {
        return -1;
    }
    pthread_mutex_lock(&regions_lock);
    for (RemoteRegion **link = &regions; *link != NULL; link = &(*link)->next) {
        if (*link == region) {
            *link = region->next;
            break;
        }
    }
    pthread_mutex_unlock(&regions_lock);
    /* Reverse of acquire: the queue pairs before the region they reach, and
       the connection last -- closing it is what frees everything on the far
       side if any step above failed to. */
    pthread_mutex_destroy(&region->write_lock);
    shadowspill_endpoint_close(&region->endpoint);
    const int status = shadowspill_control_free(
        &region->channel, region->address
    );
    shadowspill_control_close(&region->channel);
    if (region->reservation != NULL) {
        (void)munmap(region->reservation, (size_t)region->capacity);
    }
    free(region);
    return status;
}

/*
 * `configuration` is NULL: which machine a pool talks to is the *pool's*
 * configuration, not the kind's, so this one entry serves any number of remote
 * pools on any number of machines.
 */
const ShadowSpillPoolMemoryDescription shadowspill_remote_pool_memory = {
    .kind = SHADOWSPILL_POOL_REMOTE,
    .acquire = remote_acquire,
    .release = remote_release,
    /* Neither is optional for this kind: the region is a PROT_NONE
       reservation, so the runtime's ordinary copy would fault on the first
       byte in either direction. */
    .write = remote_write,
    .read = remote_read,
    .configuration = NULL,
};
