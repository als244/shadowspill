/* The struct every lane embeds, and the counters every lane keeps. */

#ifndef SHADOWSPILL_RUNTIME_LANE_BASE_H
#define SHADOWSPILL_RUNTIME_LANE_BASE_H

#include <stdatomic.h>
#include <stdint.h>

#include <shadowspill/backend.h>
#include <shadowspill/runtime/lane.h>

/*
 * This header is for code that *implements* a lane, in the tree or in a loaded
 * library, and for the runtime reading a lane's counters. A caller that only
 * declares a runtime includes <shadowspill/runtime/lane.h>, where a lane is an
 * opaque pointer -- which is what it should be to anyone not implementing one,
 * and what keeps this layout out of the umbrella header a framework adapter
 * pulls into its C++ translation units.
 *
 * So: C only, and free to use `_Atomic`.
 */

/*
 * One lane, made per route at create.
 *
 * Every lane holds the same fields and counts the same seven, so they are
 * here rather than written once per transport. A transport embeds this as its
 * **first member** and casts between the two:
 *
 *     typedef struct { ShadowSpillLane base; ... } MyLane;
 *
 * which is what makes `ShadowSpillLane *` mean one thing everywhere, in the
 * runtime and in a library the runtime loaded alike.
 *
 * The runtime fills every field here before the lane's first call, so `create`
 * does not set them and cannot set them wrong. What a transport adds after the
 * base is its own.
 */

/* Where one of a lane's pools lives: its memory from `address` for `bytes`.
   A transport whose hardware must be made able to reach a pool -- a NIC
   registering it -- does that at create, once, from these. */
typedef struct ShadowSpillLaneRange {
    void *address;
    uint64_t bytes;
} ShadowSpillLaneRange;

struct ShadowSpillLane {
    ShadowSpillRuntime *runtime;
    const ShadowSpillBackend *backend;
    /* The route's stream. A lane may use the backend on it; see `create`. */
    ShadowSpillBackendStream stream;
    /* The directional pool-kind pair this lane serves, from its description.
       A transport that copies one way for a fetch and the other for an evict
       reads its direction from these rather than keeping a flag of its own. */
    uint8_t from_kind;
    uint8_t to_kind;
    /* Where the two pools live, in the order of the kinds above. */
    ShadowSpillLaneRange from_range;
    ShadowSpillLaneRange to_range;

    _Atomic uint64_t copies;      /* transfers accepted */
    _Atomic uint64_t chunks;      /* pieces the hardware was handed */
    _Atomic uint64_t bytes;       /* bytes accepted, summed over copies */
    _Atomic uint64_t signals;     /* completion signals issued */
    _Atomic uint64_t waits;       /* dependency waits enqueued */
    _Atomic uint64_t retries;     /* waits that asked to be retried */
    _Atomic uint64_t failures;    /* transfers that did not land */
};

/*
 * Add to one of the counters above, and read one back.
 *
 * Relaxed, in one place, so no transport picks an ordering by accident: a
 * counter is read for a report and never to order anything.
 */
static inline void shadowspill_lane_counted(_Atomic uint64_t *counter, uint64_t by) {
    atomic_fetch_add_explicit(counter, by, memory_order_relaxed);
}

static inline uint64_t shadowspill_lane_count(const _Atomic uint64_t *counter) {
    return atomic_load_explicit(counter, memory_order_relaxed);
}

#endif /* SHADOWSPILL_RUNTIME_LANE_BASE_H */
