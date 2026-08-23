#ifndef SHADOWSPILL_BACKEND_H
#define SHADOWSPILL_BACKEND_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SHADOWSPILL_MEMORY_POOL_BACKEND_ABI_VERSION 1U
#define SHADOWSPILL_TRANSFER_ROUTE_ABI_VERSION 1U
#define SHADOWSPILL_SYNCHRONIZATION_BACKEND_ABI_VERSION 1U

typedef struct ShadowSpillBackendStream {
    uintptr_t words[2];
} ShadowSpillBackendStream;

typedef struct ShadowSpillBackendEvent {
    uintptr_t words[2];
} ShadowSpillBackendEvent;

/*
 * Owns the physical arena for one MemoryPool. The state pointer and callbacks carry
 * the concrete meaning of that storage; the neutral runtime sees only one
 * contiguous byte-addressable arena. An execution-pool backend must return
 * addresses that the selected framework accelerator can consume. Spill pools
 * may use any address token understood by their directed transfer routes.
 */
typedef struct ShadowSpillMemoryPoolBackend {
    uint32_t abi_version;
    void *state;
    int (*allocate_arena)(void *state, uint64_t bytes, void **base);
    /* Idempotently close and release this pool's physical arena. */
    int (*close)(void *state, void *base);
} ShadowSpillMemoryPoolBackend;

/*
 * One directed copy capability between two pool identities. Direction is an
 * immutable property of the route, never an argument interpreted by the
 * runtime. A route owns its lanes; submitted copies are asynchronous and
 * ordered within a lane. Completion events recorded on a lane must be
 * compatible with the runtime's execution-event backend.
 */
typedef struct ShadowSpillTransferRoute {
    uint32_t abi_version;
    uint32_t source_pool_id;
    uint32_t destination_pool_id;
    void *state;
    int (*create_lane)(void *state, ShadowSpillBackendStream *lane);
    int (*destroy_lane)(void *state, ShadowSpillBackendStream lane);
    int (*copy_async)(
        void *state,
        void *destination,
        const void *source,
        uint64_t bytes,
        ShadowSpillBackendStream lane
    );
    int (*synchronize_lane)(void *state, ShadowSpillBackendStream lane);
} ShadowSpillTransferRoute;

/*
 * Event operations shared by execution streams and transfer-route lanes.
 * Streams and events are opaque backend tokens. The runtime borrows state,
 * which must outlive it, and owns every event successfully created here.
 * record_event and wait_event enqueue without synchronizing the host;
 * query_event is a nonblocking poll.
 */
typedef struct ShadowSpillSynchronizationBackend {
    uint32_t abi_version;
    void *state;
    int (*create_event)(void *state, ShadowSpillBackendEvent *event);
    int (*destroy_event)(void *state, ShadowSpillBackendEvent event);
    int (*record_event)(
        void *state,
        ShadowSpillBackendEvent event,
        ShadowSpillBackendStream stream
    );
    int (*query_event)(
        void *state,
        ShadowSpillBackendEvent event,
        int *complete
    );
    int (*wait_event)(
        void *state,
        ShadowSpillBackendStream stream,
        ShadowSpillBackendEvent event
    );
} ShadowSpillSynchronizationBackend;

#ifdef __cplusplus
}
#endif

#endif
