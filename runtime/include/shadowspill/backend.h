#ifndef SHADOWSPILL_BACKEND_H
#define SHADOWSPILL_BACKEND_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SHADOWSPILL_BACKEND_ABI_VERSION 1U
#define SHADOWSPILL_MEMORY_POOL_BACKEND_ABI_VERSION 1U
#define SHADOWSPILL_TRANSFER_ROUTE_ABI_VERSION 1U

typedef struct ShadowSpillBackendStream {
    uintptr_t words[2];
} ShadowSpillBackendStream;

typedef struct ShadowSpillBackendEvent {
    uintptr_t words[2];
} ShadowSpillBackendEvent;

/*
 * Owns the physical arena for one MemoryPool. The context and callbacks carry
 * the concrete meaning of that storage; the neutral runtime sees only one
 * contiguous byte-addressable arena. An execution-pool backend must return
 * addresses that the selected framework accelerator can consume. Spill pools
 * may use any address token understood by their directed transfer routes.
 */
typedef struct ShadowSpillMemoryPoolBackend {
    uint32_t abi_version;
    void *context;
    int (*allocate_arena)(void *context, uint64_t bytes, void **base);
    /* Idempotently close and release this pool's physical arena. */
    int (*close)(void *context, void *base);
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
    void *context;
    int (*create_lane)(void *context, ShadowSpillBackendStream *lane);
    int (*destroy_lane)(void *context, ShadowSpillBackendStream lane);
    int (*copy_async)(
        void *context,
        void *destination,
        const void *source,
        uint64_t bytes,
        ShadowSpillBackendStream lane
    );
    int (*synchronize_lane)(void *context, ShadowSpillBackendStream lane);
} ShadowSpillTransferRoute;

typedef enum ShadowSpillTransferKind {
    SHADOWSPILL_TRANSFER_FETCH = 0,
    SHADOWSPILL_TRANSFER_EVICT = 1,
} ShadowSpillTransferKind;

/*
 * Framework-neutral backend operations. Every operation returns zero on
 * success and nonzero on failure. Stream and event values are opaque tokens
 * created and interpreted solely by the selected backend.
 *
 * The runtime copies this table but borrows context, which must outlive it.
 * Execution and spill arenas are owned by the runtime after a successful
 * allocate call and are returned exactly once through the matching free call.
 * Created streams and events follow the same ownership rule.
 *
 * copy_async, record_event, and wait_event enqueue work without synchronizing
 * the host. Memory passed to copy_async must remain valid through the recorded
 * completion event. query_event is a nonblocking poll. synchronize_stream is
 * used only at explicit lifecycle boundaries.
 *
 * A backend must tolerate calls from the frontend thread and the runtime
 * worker thread. ShadowSpill serializes state transitions, but the backend
 * remains responsible for the thread-safety of its own context.
 */
typedef struct ShadowSpillBackend {
    uint32_t abi_version;
    void *context;

    int (*allocate_execution)(void *context, uint64_t bytes, void **pointer);
    int (*free_execution)(void *context, void *pointer);
    int (*allocate_spill)(void *context, uint64_t bytes, void **pointer);
    int (*free_spill)(void *context, void *pointer);

    int (*create_stream)(
        void *context,
        ShadowSpillTransferKind kind,
        ShadowSpillBackendStream *stream
    );
    int (*destroy_stream)(void *context, ShadowSpillBackendStream stream);
    int (*create_event)(void *context, ShadowSpillBackendEvent *event);
    int (*destroy_event)(void *context, ShadowSpillBackendEvent event);
    int (*record_event)(
        void *context,
        ShadowSpillBackendEvent event,
        ShadowSpillBackendStream stream
    );
    int (*query_event)(
        void *context,
        ShadowSpillBackendEvent event,
        int *complete
    );
    int (*wait_event)(
        void *context,
        ShadowSpillBackendStream stream,
        ShadowSpillBackendEvent event
    );
    int (*copy_async)(
        void *context,
        void *destination,
        const void *source,
        uint64_t bytes,
        ShadowSpillTransferKind kind,
        ShadowSpillBackendStream stream
    );
    int (*synchronize_stream)(
        void *context,
        ShadowSpillBackendStream stream
    );
} ShadowSpillBackend;

#ifdef __cplusplus
}
#endif

#endif
