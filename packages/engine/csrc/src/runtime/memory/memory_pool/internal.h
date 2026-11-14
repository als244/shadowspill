#ifndef SHADOWSPILL_MEMORY_POOL_INTERNAL_H
#define SHADOWSPILL_MEMORY_POOL_INTERNAL_H

/*
 * The pool itself: its memory, its records, and the leases it hands out.
 *
 * A pool is a bump allocator with a release frontier. Memory holds the mapping
 * and the reservations over it; records holds the lease and use tables it
 * owns; locks holds the several locks and what each one covers; leases is
 * the frontier -- reserve, adopt, retire, release -- and causal is the part
 * of that frontier a successor may be promised before its predecessor
 * releases. Every ``_locked`` name is called with the pool's lock held.
 */

#define _DEFAULT_SOURCE
#include "../../internal.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <string.h>

void shadowspill_pool_cpu_relax(void);

void *shadowspill_memory_pool_pointer(
    const ShadowSpillMemoryPool *pool,
    uint64_t offset
);

int shadowspill_pool_handoff_causal_range_locked(
    ShadowSpillMemoryLease *successor,
    ShadowSpillMemoryLeaseState successor_state,
    ShadowSpillEventLease **dependency_event
);

#endif  /* SHADOWSPILL_MEMORY_POOL_INTERNAL_H */
