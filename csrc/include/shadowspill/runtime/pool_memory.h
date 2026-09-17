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

    /*
     * The way in and the way out. Both **optional**, and optional together:
     * NULL means this process can dereference the region itself, so the
     * runtime moves the bytes with an ordinary copy. That is what the two
     * built-in kinds do. A kind whose region this process cannot address
     * implements both -- and must, because the alternative is a fault at the
     * first byte.
     *
     * `source` and `destination` are pointers **in the runtime process**, and
     * that is the honest statement of the contract: not "host memory", which
     * would promise something about the machine, but "an address this process
     * can use". Whoever calls these already holds such a pointer, because
     * every caller is inside this process. Serving a client that is not is a
     * different problem, and would need a different entry than either of
     * these; nothing here pretends to solve it.
     *
     * `offset` is measured from the pool's base, so neither entry needs to
     * know what a pool address means -- which is the same property that lets
     * `acquire` report a base this process cannot read.
     *
     * They exist because moving bytes across the pool's edge is a property of
     * the memory rather than of a transfer. Importing a model's state and
     * reading a checkpoint back are not scheduled transfers on any route: they
     * happen outside a plan, against ordinary memory the caller owns. Routing
     * either through a lane would need a lane for a pair of kinds that is not
     * a route.
     *
     * Called from the thread importing or exporting state, never from the
     * worker, and synchronous: when one returns 0 the bytes have landed.
     */
    int (*write)(
        void *state,
        uint64_t offset,
        const void *source,
        uint64_t bytes
    );

    /* Take `bytes` from this pool at `offset` and put them at `destination`.
       The counterpart of `write`, under every rule above. */
    int (*read)(
        void *state,
        uint64_t offset,
        void *destination,
        uint64_t bytes
    );

    /* Passed to `acquire` untouched. Whoever registers the entry decides what
       it points at; for a kind that needs no configuration it is NULL. */
    void *configuration;
} ShadowSpillPoolMemoryDescription;

#ifdef __cplusplus
}
#endif

#endif /* SHADOWSPILL_RUNTIME_POOL_MEMORY_H */
