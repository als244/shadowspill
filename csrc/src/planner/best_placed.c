#include <stdatomic.h>
#include <stdlib.h>

#include <shadowspill/planner.h>

/*
 * The best makespan any caller has actually placed.
 *
 * Placement is expensive and a plan no better than one already placed cannot
 * win even if it places, so a search consults this before measuring. The
 * object is deliberately generic: it knows nothing about candidates, resolved
 * programs or calls, so the same code shares one gate between the candidates
 * of a single call or between concurrent calls, depending only on which
 * object the caller passes.
 *
 * Lock-free because the operation is a minimum: readers take a relaxed
 * snapshot, and a writer that loses a race simply retries against the value
 * that beat it. A stale read only costs a measurement that would have been
 * skipped, never a wrong answer.
 */
struct ShadowSpillBestPlaced {
    /* Zero means nothing has been placed yet, so nothing is ruled out. */
    _Atomic uint64_t makespan_ns;
};

ShadowSpillBestPlaced *shadowspill_best_placed_create(void) {
    ShadowSpillBestPlaced *best = calloc(1U, sizeof(*best));
    if (best == NULL) {
        return NULL;
    }
    atomic_store_explicit(&best->makespan_ns, 0U, memory_order_relaxed);
    return best;
}

void shadowspill_best_placed_destroy(ShadowSpillBestPlaced *best) {
    free(best);
}

uint64_t shadowspill_best_placed_get(const ShadowSpillBestPlaced *best) {
    if (best == NULL) {
        return 0U;
    }
    return atomic_load_explicit(&best->makespan_ns, memory_order_acquire);
}

int shadowspill_best_placed_offer(
    ShadowSpillBestPlaced *best,
    uint64_t makespan_ns
) {
    if (best == NULL || makespan_ns == 0U) {
        return 0;
    }
    uint64_t current =
        atomic_load_explicit(&best->makespan_ns, memory_order_relaxed);
    while (current == 0U || makespan_ns < current) {
        if (atomic_compare_exchange_weak_explicit(
                &best->makespan_ns,
                &current,
                makespan_ns,
                memory_order_acq_rel,
                memory_order_relaxed
            )) {
            return 1;
        }
        /* `current` now holds whatever beat us; the loop re-tests against it. */
    }
    return 0;
}

int shadowspill_best_placed_admits(
    const ShadowSpillBestPlaced *best,
    uint64_t makespan_ns
) {
    uint64_t bound = shadowspill_best_placed_get(best);
    return bound == 0U || makespan_ns < bound;
}
