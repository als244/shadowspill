#ifndef SHADOWSPILL_BACKEND_H
#define SHADOWSPILL_BACKEND_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * The backend contract: a flat table of driver-level calls that one shared
 * object per accelerator provider implements. Everything built from these
 * calls -- pools, routes, lanes, event pools -- is ShadowSpill's, so a backend
 * carries no policy and no object lifetime of its own beyond the provider
 * context. The one version below is checked at load; a backend built against
 * another version is refused.
 */
#define SHADOWSPILL_BACKEND_ABI_VERSION 1U

#define SHADOWSPILL_BACKEND_PROVIDER_NAME_CAPACITY 16U

/* Opaque provider tokens. The runtime stores and returns them unread. */
typedef struct ShadowSpillBackendStream {
    uintptr_t words[2];
} ShadowSpillBackendStream;

typedef struct ShadowSpillBackendEvent {
    uintptr_t words[2];
} ShadowSpillBackendEvent;

typedef uint64_t ShadowSpillProfilerRange;

typedef struct ShadowSpillBackendConfig {
    uint32_t abi_version;
    int32_t device_ordinal;
} ShadowSpillBackendConfig;

/* The alignment device allocations want, and the platform's short lowercase
 * name for diagnostics ("mock" for the mock backend). */
typedef struct ShadowSpillBackendCapabilities {
    int32_t device_ordinal;
    uint64_t minimum_alignment;
    char provider[SHADOWSPILL_BACKEND_PROVIDER_NAME_CAPACITY];
} ShadowSpillBackendCapabilities;

/* The accelerator's memory as the platform accounts for it right now. */
typedef struct ShadowSpillBackendPhysicalMemory {
    uint64_t process_bytes;
    uint64_t device_used_bytes;
    uint64_t device_total_bytes;
} ShadowSpillBackendPhysicalMemory;

/* Counters of driver calls made through this table. A backend without a
 * notion of one reports zero; provider_activations counts the times the
 * provider's context had to be made current on the calling thread. */
typedef struct ShadowSpillBackendStatistics {
    uint64_t device_allocations;
    uint64_t device_frees;
    uint64_t bytes_device_allocated;
    uint64_t bytes_device_freed;
    uint64_t pinned_host_registrations;
    uint64_t pinned_host_unregistrations;
    uint64_t bytes_pinned_host_registered;
    uint64_t bytes_pinned_host_unregistered;
    uint64_t streams_created;
    uint64_t streams_destroyed;
    uint64_t events_created;
    uint64_t events_destroyed;
    uint64_t copies_host_to_device;
    uint64_t copies_device_to_host;
    uint64_t copies_device_to_device;
    uint64_t bytes_host_to_device;
    uint64_t bytes_device_to_host;
    uint64_t bytes_device_to_device;
    uint64_t event_queries;
    uint64_t stream_waits;
    uint64_t stream_synchronizations;
    uint64_t provider_activations;
} ShadowSpillBackendStatistics;

/*
 * Every entry receives `state`, the provider object shadowspill_backend_create()
 * made. Entries return 0 on success and nonzero on failure unless documented
 * otherwise. Calls arrive from the caller's threads and from the runtime
 * worker; the backend serializes what its provider requires. The profiler
 * entries are optional: NULL means the runtime treats them as no-ops.
 */
typedef struct ShadowSpillBackend {
    uint32_t abi_version;
    void *state;

    /* Memory. Device memory is the backend's to allocate; host memory is
       ShadowSpill's, allocated by the pool and registered here so the
       provider can copy from it asynchronously. Frees and unregistrations
       carry the byte count so the backend keeps no size bookkeeping. */
    int (*allocate_device)(void *state, uint64_t bytes, void **address);
    int (*free_device)(void *state, void *address, uint64_t bytes);
    int (*register_host_memory)(void *state, void *address, uint64_t bytes);
    int (*unregister_host_memory)(void *state, void *address, uint64_t bytes);

    /* Streams: ordered queues of copies and events. wrap_stream turns the
       integer handle the framework exposes for its own stream into a token. */
    int (*create_stream)(void *state, ShadowSpillBackendStream *stream);
    int (*destroy_stream)(void *state, ShadowSpillBackendStream stream);
    int (*synchronize_stream)(void *state, ShadowSpillBackendStream stream);
    ShadowSpillBackendStream (*wrap_stream)(
        void *state,
        uint64_t framework_stream_handle
    );

    /* Copies: asynchronous, ordered on the stream, between memory the two
       calls above made or registered. */
    int (*copy_host_to_device)(
        void *state,
        void *device,
        const void *host,
        uint64_t bytes,
        ShadowSpillBackendStream stream
    );
    int (*copy_device_to_host)(
        void *state,
        void *host,
        const void *device,
        uint64_t bytes,
        ShadowSpillBackendStream stream
    );
    int (*copy_device_to_device)(
        void *state,
        void *destination,
        const void *source,
        uint64_t bytes,
        ShadowSpillBackendStream stream
    );

    /* Events. A dependency event (timing clear) is the fast kind that
       record, query, and wait work with. A timing event carries a device
       timestamp when recorded; elapsed_nanoseconds reads the device-clock
       interval between two of them: 0 with the interval, 1 while either is
       still pending, -1 when the pair cannot be measured. record and wait
       enqueue without blocking the host; query is a nonblocking poll. */
    int (*create_event)(
        void *state,
        ShadowSpillBackendEvent *event,
        uint8_t timing
    );
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
    int (*elapsed_nanoseconds)(
        void *state,
        ShadowSpillBackendEvent from,
        ShadowSpillBackendEvent to,
        uint64_t *nanoseconds
    );

    /* Facts. */
    int (*capabilities)(void *state, ShadowSpillBackendCapabilities *capabilities);
    int (*physical_memory)(void *state, ShadowSpillBackendPhysicalMemory *memory);
    void (*statistics)(void *state, ShadowSpillBackendStatistics *statistics);

    /* Profiler, optional. Names and ranges are best-effort diagnostics and
       never change execution semantics. */
    void (*name_thread)(void *state, const char *name);
    void (*name_stream)(
        void *state,
        ShadowSpillBackendStream stream,
        const char *name
    );
    void (*profiler_enable)(void *state, uint8_t enabled);
    ShadowSpillProfilerRange (*range_begin)(void *state, const char *name);
    void (*range_end)(void *state, ShadowSpillProfilerRange range);
} ShadowSpillBackend;

/* The two symbols every backend shared object exports. create() fills the
 * table and returns 0, or returns nonzero leaving nothing to destroy;
 * destroy() releases the provider object and zeroes the table. */
typedef int (*ShadowSpillBackendCreate)(
    const ShadowSpillBackendConfig *config,
    ShadowSpillBackend *backend
);
typedef void (*ShadowSpillBackendDestroy)(ShadowSpillBackend *backend);

#define SHADOWSPILL_BACKEND_CREATE_SYMBOL "shadowspill_backend_create"
#define SHADOWSPILL_BACKEND_DESTROY_SYMBOL "shadowspill_backend_destroy"

int shadowspill_backend_create(
    const ShadowSpillBackendConfig *config,
    ShadowSpillBackend *backend
);
void shadowspill_backend_destroy(ShadowSpillBackend *backend);

#ifdef __cplusplus
}
#endif

#endif
