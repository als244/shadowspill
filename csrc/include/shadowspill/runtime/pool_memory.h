/* Where a pool's memory comes from, and how the runtime finds out. */

#ifndef SHADOWSPILL_RUNTIME_POOL_MEMORY_H
#define SHADOWSPILL_RUNTIME_POOL_MEMORY_H

#include <stdint.h>

#include <shadowspill/shadowspill.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------------
 * Pool memory
 *
 * A pool owns one bounded region and suballocates leases from it. What varies
 * between kinds is only how that region is obtained and what an address in it
 * means; everything else about a pool -- ranges, leases, retirement, the
 * release frontier -- is indifferent to both.
 *
 * That indifference is the whole reason this is a lookup rather than a switch.
 * Nothing in the memory subsystem dereferences an allocation's pointer: it is
 * assigned, compared, handed out, and never read through. So a region on
 * another machine is the same object as a local one, and a kind that lives
 * there needs no special case anywhere.
 *
 * An entry is found by kind, from the `pool_memory` list on the runtime config.
 * The runtime seeds that list with the kinds it implements and appends whatever
 * a caller registered, so one lookup serves both.
 */

typedef struct ShadowSpillPoolMemoryDescription {
    /* A ShadowSpillPoolKind value. Two entries claiming one kind fails create. */
    uint8_t kind;

    /*
     * Obtain `capacity` bytes and report where they start.
     *
     * `state` is whatever this kind needs to remember in order to release the
     * region later -- a connection, a key, a handle -- and is handed back to
     * `release` untouched. A kind with nothing to remember leaves it NULL.
     *
     * `base` is not required to be dereferenceable by this process. It is an
     * address in the pool's own space, and the only arithmetic done on it is
     * adding an offset. Whether that yields something this machine can read is
     * the kind's business and nobody else's.
     *
     * Called once per pool at create, from the thread that creates the runtime,
     * and reports failure by returning non-zero -- so it needs no way to reach
     * the runtime, unlike a lane, which may fail on a thread of its own.
     */
    int (*acquire)(
        void *configuration,
        uint64_t capacity,
        void **base,
        void **state
    );

    /* Give the region back. Receives what `acquire` produced. */
    int (*release)(void *state, void *base, uint64_t capacity);

    /* Passed to `acquire` untouched. Whoever registers the entry decides what
       it points at; for a kind that needs no configuration it is NULL. */
    void *configuration;
} ShadowSpillPoolMemoryDescription;

#ifdef __cplusplus
}
#endif

#endif /* SHADOWSPILL_RUNTIME_POOL_MEMORY_H */
