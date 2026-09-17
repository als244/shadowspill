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
/* A stream and an event are each one opaque word only the backend reads --
   a driver handle, a pointer, an index, whatever it keeps them in -- exactly
   as a profiler range is. Zero is "none": for a stream it is the backend's
   default stream, and for an event it is no event. */
typedef uint64_t ShadowSpillBackendStream;
typedef uint64_t ShadowSpillBackendEvent;
/* A block of words a stream can wait on; one opaque word, as above. Zero is
   "none". */
typedef uint64_t ShadowSpillBackendSignals;

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
    uint64_t stream_writes;
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
    /*
     * Words a stream can wait on and a host thread can store to. Separate from
     * the pair above because a provider may require them mapped a particular
     * way for a stream to read them at all, in which case one word has two
     * addresses -- and only the backend should know that.
     *
     * `signals` names the block; `host` is where the caller stores a value, one
     * word per index. The address a stream waits on never leaves the backend,
     * which is why `wait_value` below takes an index and not a pointer.
     */
    int (*allocate_signals)(
        void *state,
        uint32_t count,
        ShadowSpillBackendSignals *signals,
        uint64_t **host
    );
    int (*free_signals)(void *state, ShadowSpillBackendSignals signals);

    /* Streams: ordered queues of copies and events. A stream the backend made
       comes from create_stream; a stream someone else owns is named by the
       integer its owner knows it by, and resolve_stream answers with the word
       this backend knows it by. A handle of 0 is the backend's default. */
    int (*create_stream)(void *state, ShadowSpillBackendStream *stream);
    int (*destroy_stream)(void *state, ShadowSpillBackendStream stream);
    int (*synchronize_stream)(void *state, ShadowSpillBackendStream stream);
    ShadowSpillBackendStream (*resolve_stream)(
        void *state,
        uint64_t stream_handle
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
    /*
     * Holds `stream` until the word at `index` of `signals` reaches
     * `value`, comparing greater-or-equal so a value already past it does
     * not stall. `wait_event` orders a stream behind work the device will do;
     * this orders it behind work the device cannot see -- a transfer some other
     * hardware is performing, whose completion only the host learns about.
     *
     * Without it a lane that does not move bytes on a stream has no way to make
     * a stream wait for it: the consumer's `wait_event` can be enqueued before
     * the transfer finishes, and a wait on an event not yet recorded does not
     * wait at all.
     */
    int (*wait_value)(
        void *state,
        ShadowSpillBackendStream stream,
        ShadowSpillBackendSignals signals,
        uint32_t index,
        uint64_t value
    );
    /*
     * The mirror of `wait_value`: the *stream* stores `value`, and a host
     * thread polling the word learns how far the stream has got.
     *
     * It exists for the same reason as the wait and answers the opposite
     * question. A lane that stages through a host buffer must not hand the
     * next piece of that buffer to hardware until the device has finished
     * reading what is already there, and only the device can say when that is.
     * An event cannot serve: a lane enqueues every piece of a transfer up
     * front, so one event per buffer slot would be recorded several times
     * before any query, and a query reports the most recent capture -- it
     * would answer about work that cannot run yet. A value the stream writes
     * as it passes is monotonic and has no such ambiguity.
     *
     * Enqueued on the stream like any other work, so it happens in order and
     * says what it means: everything before it on this stream is done.
     */
    int (*write_value)(
        void *state,
        ShadowSpillBackendStream stream,
        ShadowSpillBackendSignals signals,
        uint32_t index,
        uint64_t value
    );
    /* Blocks the calling thread until the device reaches the event. The
     * stream wait above orders one stream behind another; this one is how a
     * host thread waits for work it submitted. */
    int (*synchronize_event)(
        void *state,
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
