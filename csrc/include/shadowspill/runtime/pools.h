/* Allocating from a pool, and giving it back. */

#ifndef SHADOWSPILL_RUNTIME_POOLS_H
#define SHADOWSPILL_RUNTIME_POOLS_H

#include <stdint.h>

#include <shadowspill/shadowspill.h>
#include <shadowspill/backend.h>
#include <shadowspill/runtime/vocabulary.h>
#include <shadowspill/runtime/descriptions.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------------
 * Pools and allocation
 *
 * Serving one allocation, freeing it, and naming the stream that
 * used it. Called from whichever thread is dispatching.
 */

/*
 * Synchronously leases an aligned range from the existing slab; it never grows
 * physical storage. The returned pointer remains valid until logical free and
 * all recorded streams retire it. This call may block only when already
 * pending work can make a suitable range available. Otherwise it returns and
 * latches NO_PROGRESS.
 */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_memory_pool_allocate(
    ShadowSpillRuntime *runtime,
    uint32_t pool_id,
    uint64_t bytes,
    uint64_t alignment,
    ShadowSpillBackendStream stream,
    ShadowSpillAllocation *allocation
);

/*
 * Resolves an exact live slab address to its allocation identity and current
 * generation. This read-only lookup exists for framework allocator callbacks
 * whose free/record-stream protocols carry an address rather than an ID.
 */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_memory_pool_allocation_for_pointer(
    ShadowSpillRuntime *runtime,
    uint32_t pool_id,
    const void *pointer,
    ShadowSpillAllocation *allocation
);

/*
 * Performs logical free immediately. A later allocation on the sole recorded
 * stream may reuse the whole pending block by adding its retirement event as a
 * stream dependency. Global and background-transfer reuse waits for every
 * recorded stream to retire. Plan-owned allocations ignore framework logical
 * free until a plan action releases or evicts the owning object.
 */
SHADOWSPILL_API ShadowSpillStatus shadowspill_memory_pool_free(
    ShadowSpillRuntime *runtime,
    uint32_t pool_id,
    uint64_t allocation_id,
    ShadowSpillBackendStream stream
);

/*
 * Reserves [offset, offset + bytes) of a pool for the runtime's lifetime,
 * for state that already lies there: placed objects borrow leases inside
 * it and releasing them does not free it. Fails when any of the range is
 * already in use.
 */
SHADOWSPILL_API ShadowSpillStatus shadowspill_memory_pool_reserve_range(
    ShadowSpillRuntime *runtime,
    uint32_t pool_id,
    uint64_t offset,
    uint64_t bytes
);

/*
 * The address a pool's offset 0 presents, which every lease pointer in the
 * pool is an offset from. For a pool this process cannot dereference it is
 * the same token the lease pointers are built on.
 */
SHADOWSPILL_API ShadowSpillStatus shadowspill_memory_pool_base(
    ShadowSpillRuntime *runtime,
    uint32_t pool_id,
    const void **base
);

/* Adds a borrowed stream token to an allocation's retirement set. */
SHADOWSPILL_API ShadowSpillStatus
shadowspill_memory_pool_record_stream(
    ShadowSpillRuntime *runtime,
    uint32_t pool_id,
    uint64_t allocation_id,
    ShadowSpillBackendStream stream
);

#ifdef __cplusplus
}
#endif

#endif
