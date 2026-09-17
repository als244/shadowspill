/* Where a Remote pool's memory comes from: a daemon on another machine. */
#include "../internal.h"

#include <stdlib.h>
#include <string.h>

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

typedef struct RemoteRegion {
    ShadowSpillControlChannel channel;
    uint64_t address;
    uint32_t key;
} RemoteRegion;

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
    if (shadowspill_control_connect(
            &region->channel, remote->host, remote->port
        ) != 0) {
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
    /* The daemon's address, widened to a pointer so one pool implementation
       serves every kind. Nothing on this machine may read through it. */
    *base = (void *)(uintptr_t)region->address;
    *state = region;
    return 0;
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
    const int status = shadowspill_control_free(
        &region->channel, region->address
    );
    shadowspill_control_close(&region->channel);
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
    .configuration = NULL,
};
