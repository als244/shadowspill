#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#include <shadowspill/planner.h>

#include "candidates_internal.h"

/*
 * The best plan any caller has actually placed.
 *
 * Placement is expensive and a plan no better than one already placed cannot
 * win even if it places, so a search consults this before measuring. The
 * object is deliberately generic: it knows nothing about candidates, resolved
 * programs or calls, so the same code shares one gate between the candidates
 * of a single call or between concurrent calls, depending only on which
 * object the caller passes.
 *
 * Two levels of synchronisation, because the two operations have very
 * different frequencies. `admits` runs at every local minimum of every
 * candidate and reads one atomic word, so it never waits. `offer` and `read`
 * touch the whole record and take a spin lock, which is affordable because a
 * placement that succeeds is rare next to the checks that precede it, and the
 * critical section is a fixed-size copy.
 *
 * A stale `admits` read costs at most a measurement that would have been
 * skipped, never a wrong answer: the best plan that will ever be placed is
 * better than everything already placed, so it is never the one refused.
 */
struct ShadowSpillBestPlaced {
    /* Mirrors record.makespan_ns so the hot path needs no lock. */
    _Atomic uint64_t makespan_ns;
    atomic_flag guard;
    ShadowSpillBestPlacedRecord record;
    /* The plan itself, owned. The record names a plan by digest, and the
     * buffer a candidate built it in is reused by the next candidate and
     * thrown away when that candidate ends, so a record that did not keep a
     * copy can name a plan nobody still has. Replaced in place when a better
     * plan arrives, which reuses these buffers rather than freeing them. */
    ShadowSpillScheduleStorage plan;
};

static void lock(ShadowSpillBestPlaced *best) {
    while (atomic_flag_test_and_set_explicit(&best->guard, memory_order_acquire)) {
        /* Held only for a fixed-size copy, so spinning beats descheduling. */
    }
}

static void unlock(ShadowSpillBestPlaced *best) {
    atomic_flag_clear_explicit(&best->guard, memory_order_release);
}

ShadowSpillBestPlaced *shadowspill_best_placed_create(void) {
    ShadowSpillBestPlaced *best = calloc(1U, sizeof(*best));
    if (best == NULL) {
        return NULL;
    }
    atomic_store_explicit(&best->makespan_ns, 0U, memory_order_relaxed);
    atomic_flag_clear(&best->guard);
    return best;
}

void shadowspill_best_placed_destroy(ShadowSpillBestPlaced *best) {
    if (best == NULL) {
        return;
    }
    shadowspill_schedule_storage_destroy(&best->plan);
    free(best);
}

int shadowspill_best_placed_admits(
    const ShadowSpillBestPlaced *best,
    uint64_t makespan_ns
) {
    if (best == NULL) {
        return 1;
    }
    uint64_t bound =
        atomic_load_explicit(&best->makespan_ns, memory_order_acquire);
    return bound == 0U || makespan_ns < bound;
}

int shadowspill_best_placed_offer(
    ShadowSpillBestPlaced *best,
    const ShadowSpillBestPlacedRecord *record,
    const ShadowSpillScheduleStorage *plan
) {
    if (best == NULL || record == NULL || plan == NULL ||
        record->makespan_ns == 0U) {
        return 0;
    }
    int replaced = 0;
    lock(best);
    if (best->record.makespan_ns == 0U ||
        record->makespan_ns < best->record.makespan_ns) {
        /* Sized on first use from the plan being kept; afterwards the copy
         * grows only if a later plan needs more room. */
        if (best->plan.initial_capacity == 0U &&
            shadowspill_schedule_storage_create(
                plan->initial_capacity, &best->plan
            ) != 0) {
            unlock(best);
            return 0;
        }
        if (shadowspill_schedule_storage_copy(&best->plan, plan) != 0) {
            unlock(best);
            return 0;
        }
        best->record = *record;
        atomic_store_explicit(
            &best->makespan_ns, record->makespan_ns, memory_order_release
        );
        replaced = 1;
    }
    unlock(best);
    return replaced;
}

void shadowspill_best_placed_read(
    const ShadowSpillBestPlaced *best,
    ShadowSpillBestPlacedRecord *record
) {
    if (record == NULL) {
        return;
    }
    if (best == NULL) {
        memset(record, 0, sizeof(*record));
        return;
    }
    ShadowSpillBestPlaced *mutable_best = (ShadowSpillBestPlaced *)best;
    lock(mutable_best);
    *record = best->record;
    unlock(mutable_best);
}
