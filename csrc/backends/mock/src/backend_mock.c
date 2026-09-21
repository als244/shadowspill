#include <shadowspill/backend_mock.h>

#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*
 * A stream is an ordered queue, and something has to drain it.
 *
 * `backend.h` calls a stream "an ordered queue of copies and events", and until
 * a lane needed to hold one on a word the mock could get away with not being
 * one: every entry performed its side effect as it was called, and only the
 * clock was deferred. That is wrong in exactly the way a lane whose bytes do
 * not move on a stream depends on being right. A copy enqueued behind an
 * unsatisfied wait ran anyway -- reading a staging buffer the hardware had not
 * filled -- because the wait was consulted by whoever came to observe the
 * stream and by nothing on the stream itself.
 *
 * So an entry appends a `MockOperation` and returns, and the caller then drains
 * the stream as far as it will go. For a stream with nothing to wait for that
 * is the whole queue before the call returns, which is what every caller that
 * never waits already had.
 *
 * A queue stopped at a wait is resumed by the backend's drain thread, because
 * nothing satisfies a wait through this table: the word a `wait_value` awaits
 * is stored by a host thread writing memory this backend never sees. Looking
 * again is the only mechanism there is.
 *
 * What the queue does *not* model is the stream's own clock: an entry runs as
 * soon as it reaches the head and its wait is satisfied, rather than when the
 * entry before it was projected to finish. The clock stays a projection that
 * observers wait for, which is what it always was; the queue adds ordering,
 * which is what was missing.
 */

/* How long a drainer waits before looking again at a queue it cannot advance.
   Short enough not to distort a measured transfer rate, long enough not to be
   a spin. */
#define DRAIN_POLL_NANOSECONDS 20000U

typedef enum MockOperationKind {
    MOCK_COPY = 0,
    /* A span of work with no side effect but the clock: the test hook. */
    MOCK_COMPUTE,
    MOCK_WAIT_VALUE,
    MOCK_WRITE_VALUE,
    MOCK_WAIT_EVENT,
    MOCK_RECORD_EVENT,
} MockOperationKind;

typedef struct MockEvent MockEvent;

typedef struct MockOperation {
    MockOperationKind kind;
    struct MockOperation *next;

    /* MOCK_COPY */
    void *destination;
    const void *source;
    uint64_t bytes;

    /* MOCK_COPY and MOCK_COMPUTE: what this entry adds to the stream's clock. */
    uint64_t duration_nanoseconds;

    /* MOCK_WAIT_VALUE and MOCK_WRITE_VALUE */
    uint64_t *word;
    uint64_t value;

    /* MOCK_WAIT_EVENT and MOCK_RECORD_EVENT */
    MockEvent *event;
    /* Records only, and the reason a superseded record cannot stamp its event:
       re-recording bumps the event's stamp, so an older entry still in a queue
       finds the two disagree and does nothing. */
    uint64_t stamp;
} MockOperation;

/*
 * Three states rather than two flags, because "queued but never recorded" is
 * not a thing an event can be.
 */
typedef enum MockEventState {
    /* Never recorded. Query, wait and elapsed all refuse it. */
    MOCK_EVENT_UNRECORDED = 0,
    /* Recorded, and the record has not reached the head of its stream yet. */
    MOCK_EVENT_QUEUED,
    /* The record ran: `ready_nanoseconds` is the instant it stamped. */
    MOCK_EVENT_STAMPED,
} MockEventState;

struct MockEvent {
    uint64_t ready_nanoseconds;
    uint64_t stamp;
    MockEventState state;
    /*
     * Who still refers to this event: its creator until `destroy_event`, and
     * every queued record or wait until the entry is consumed. It is freed
     * when the last goes, because a caller may destroy an event a stream has
     * yet to wait on -- legal, since the wait captured the record -- and a
     * queue entry reading a freed event reads whatever the allocator left
     * there. That once read as a timestamp thirty hours ahead of the clock,
     * and a wait that never passed.
     */
    uint32_t references;
};

typedef struct MockStream {
    uint64_t ready_nanoseconds;
    MockOperation *head;
    MockOperation *tail;
    /*
     * Set for as long as a drainer is inside this stream, including while it
     * has released the backend's lock to perform an entry. It is what makes one
     * drainer at a time true, and what keeps `destroy_stream` from freeing a
     * stream another thread is working on.
     */
    int draining;
    struct MockStream *next;
} MockStream;

struct ShadowSpillMockBackend {
    pthread_mutex_t mutex;
    /* Raised when a stream gains work, so the drain thread sleeps rather than
       polls while every queue is empty. */
    pthread_cond_t queued;
    ShadowSpillMockBackendConfig config;
    ShadowSpillBackendStatistics statistics;
    uint64_t operation_count;
    uint64_t fail_operation;
    /* The stream a framework handle of 0 names; see resolve_stream. */
    MockStream default_stream;
    /* Every live stream, the default one first. */
    MockStream *streams;
    pthread_t drain_thread;
    int stopping;
};

static uint64_t now_nanoseconds(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        return 0U;
    }
    return (uint64_t)value.tv_sec * 1000000000U + (uint64_t)value.tv_nsec;
}

static void sleep_briefly(void) {
    struct timespec delay = {.tv_nsec = DRAIN_POLL_NANOSECONDS};
    (void)nanosleep(&delay, NULL);
}

static int operation_fails(ShadowSpillMockBackend *backend) {
    pthread_mutex_lock(&backend->mutex);
    const uint64_t operation = ++backend->operation_count;
    const int fails =
        backend->fail_operation != 0U && operation == backend->fail_operation;
    pthread_mutex_unlock(&backend->mutex);
    return fails;
}

static void count(ShadowSpillMockBackend *backend, uint64_t *counter, uint64_t by) {
    pthread_mutex_lock(&backend->mutex);
    *counter += by;
    pthread_mutex_unlock(&backend->mutex);
}

static MockStream *stream_pointer(ShadowSpillBackendStream stream) {
    return (MockStream *)stream;
}

static MockEvent *event_pointer(ShadowSpillBackendEvent event) {
    return (MockEvent *)event;
}

/* ---------------------------------------------------------------- memory */

static int allocate_device(void *state, uint64_t bytes, void **address) {
    ShadowSpillMockBackend *backend = state;
    if (address == NULL || bytes > SIZE_MAX || operation_fails(backend)) {
        return -1;
    }
    *address = malloc(bytes == 0U ? 1U : (size_t)bytes);
    if (*address == NULL) {
        return -1;
    }
    count(backend, &backend->statistics.device_allocations, 1U);
    count(backend, &backend->statistics.bytes_device_allocated, bytes);
    return 0;
}

static int free_device(void *state, void *address, uint64_t bytes) {
    ShadowSpillMockBackend *backend = state;
    if (operation_fails(backend)) {
        return -1;
    }
    free(address);
    count(backend, &backend->statistics.device_frees, 1U);
    count(backend, &backend->statistics.bytes_device_freed, bytes);
    return 0;
}

static int register_host_memory(void *state, void *address, uint64_t bytes) {
    ShadowSpillMockBackend *backend = state;
    if (address == NULL || bytes == 0U || operation_fails(backend)) {
        return -1;
    }
    count(backend, &backend->statistics.pinned_host_registrations, 1U);
    count(backend, &backend->statistics.bytes_pinned_host_registered, bytes);
    return 0;
}

static int unregister_host_memory(void *state, void *address, uint64_t bytes) {
    ShadowSpillMockBackend *backend = state;
    if (address == NULL || operation_fails(backend)) {
        return -1;
    }
    count(backend, &backend->statistics.pinned_host_unregistrations, 1U);
    count(backend, &backend->statistics.bytes_pinned_host_unregistered, bytes);
    return 0;
}

/* ----------------------------------------------------------- the queue */

/* Drop one hold on an event, freeing it with the last. The backend's lock is
   held. */
static void release_event_locked(MockEvent *event) {
    if (event != NULL && --event->references == 0U) {
        free(event);
    }
}

/* Whether an event has completed: its record has run, its instant has passed.
   The backend's lock is held. */
static int event_complete_locked(const MockEvent *event) {
    return event->state == MOCK_EVENT_STAMPED &&
           now_nanoseconds() >= event->ready_nanoseconds;
}

/* Whether the entry at the head may run. The backend's lock is held.
   Only the two waits can answer no, which is what makes them waits. */
static int entry_may_run_locked(const MockOperation *entry) {
    switch (entry->kind) {
    case MOCK_WAIT_VALUE:
        return __atomic_load_n(entry->word, __ATOMIC_ACQUIRE) >= entry->value;
    case MOCK_WAIT_EVENT:
        return event_complete_locked(entry->event);
    default:
        return 1;
    }
}

/*
 * The entry's side effect, performed with the backend's lock released.
 *
 * Nothing here touches stream or event state: the clock and the stamps are
 * applied afterwards, under the lock, and the entry stays at the head until
 * they are -- so an empty queue really does mean the bytes have landed.
 */
static void perform_entry(const MockOperation *entry) {
    switch (entry->kind) {
    case MOCK_COPY:
        if (entry->bytes != 0U) {
            memcpy(entry->destination, entry->source, (size_t)entry->bytes);
        }
        break;
    case MOCK_WRITE_VALUE:
        /* Released, so a host thread that sees the word sees the copies this
           entry followed. That is the whole question the word answers. */
        __atomic_store_n(entry->word, entry->value, __ATOMIC_RELEASE);
        break;
    default:
        break;
    }
}

/* What the entry did to the clock, applied with the backend's lock held. */
static void apply_entry_locked(
    ShadowSpillMockBackend *backend, MockStream *stream, const MockOperation *entry
) {
    const uint64_t now = now_nanoseconds();
    switch (entry->kind) {
    case MOCK_COPY:
    case MOCK_COMPUTE:
        if (stream->ready_nanoseconds < now) {
            stream->ready_nanoseconds = now;
        }
        stream->ready_nanoseconds += entry->duration_nanoseconds;
        break;
    case MOCK_WAIT_EVENT:
        if (entry->event->ready_nanoseconds > stream->ready_nanoseconds) {
            stream->ready_nanoseconds = entry->event->ready_nanoseconds;
        }
        break;
    case MOCK_RECORD_EVENT:
        /* A later record has superseded this one, so it stamps nothing. */
        if (entry->event->stamp != entry->stamp) {
            break;
        }
        entry->event->ready_nanoseconds =
            (stream->ready_nanoseconds > now ? stream->ready_nanoseconds : now) +
            backend->config.event_delay_nanoseconds;
        entry->event->state = MOCK_EVENT_STAMPED;
        break;
    default:
        break;
    }
}

/*
 * Run the stream's entries in order until one may not run yet, or none is left.
 *
 * Called with the backend's lock held and returns with it held, having released
 * it around each entry's side effect -- a memcpy of an arbitrary size is not
 * something to hold a backend-wide lock across, and two streams have to be able
 * to copy at once.
 */
static void drain_locked(ShadowSpillMockBackend *backend, MockStream *stream) {
    if (stream->draining) {
        return;
    }
    stream->draining = 1;
    for (;;) {
        MockOperation *const entry = stream->head;
        if (entry == NULL || !entry_may_run_locked(entry)) {
            break;
        }
        pthread_mutex_unlock(&backend->mutex);
        perform_entry(entry);
        pthread_mutex_lock(&backend->mutex);
        apply_entry_locked(backend, stream, entry);
        stream->head = entry->next;
        if (stream->head == NULL) {
            stream->tail = NULL;
        }
        release_event_locked(entry->event);
        free(entry);
    }
    stream->draining = 0;
}

/*
 * Append an entry and run what can be run.
 *
 * `entry` is already filled in by the caller; this takes ownership of it and
 * frees it whether or not the queue advances.
 */
static void submit(
    ShadowSpillMockBackend *backend, MockStream *stream, MockOperation *entry
) {
    pthread_mutex_lock(&backend->mutex);
    entry->next = NULL;
    if (stream->tail != NULL) {
        stream->tail->next = entry;
    } else {
        stream->head = entry;
    }
    stream->tail = entry;
    drain_locked(backend, stream);
    const int left = stream->head != NULL;
    pthread_mutex_unlock(&backend->mutex);
    if (left) {
        /* Only a queue that stopped needs the thread, and only then is waking
           it worth a system call. */
        pthread_cond_signal(&backend->queued);
    }
}

static MockOperation *entry_for(MockOperationKind kind) {
    MockOperation *entry = calloc(1U, sizeof(*entry));
    if (entry != NULL) {
        entry->kind = kind;
    }
    return entry;
}

/*
 * Throw away whatever a stream still holds.
 *
 * Destroying a stream with work on it abandons that work, which is what
 * destroying a stream means. Records are the exception and are stamped on the
 * way out: an event left queued would never complete, and a caller waiting on
 * one would wait for a stream that no longer exists. The backend's lock is
 * held, and the caller has established that no drainer is inside.
 */
static void discard_queue_locked(
    ShadowSpillMockBackend *backend, MockStream *stream
) {
    while (stream->head != NULL) {
        MockOperation *const entry = stream->head;
        stream->head = entry->next;
        if (entry->kind == MOCK_RECORD_EVENT) {
            apply_entry_locked(backend, stream, entry);
        }
        release_event_locked(entry->event);
        free(entry);
    }
    stream->tail = NULL;
}

/*
 * Resume the queues that stopped.
 *
 * Nothing reaches this backend when a value wait is satisfied -- the word is
 * stored by a host thread writing memory -- so a stopped queue is found by
 * looking, and this is what looks. The cursor advances while the lock is held
 * and `drain_locked` holds `draining` across every window in which it is not,
 * so a stream cannot be destroyed under it.
 */
static void *drain_streams(void *argument) {
    ShadowSpillMockBackend *backend = argument;
    pthread_mutex_lock(&backend->mutex);
    for (;;) {
        if (backend->stopping) {
            break;
        }
        int waiting = 0;
        for (MockStream *stream = backend->streams; stream != NULL;) {
            drain_locked(backend, stream);
            waiting = waiting || stream->head != NULL;
            stream = stream->next;
        }
        if (waiting) {
            pthread_mutex_unlock(&backend->mutex);
            sleep_briefly();
            pthread_mutex_lock(&backend->mutex);
        } else {
            pthread_cond_wait(&backend->queued, &backend->mutex);
        }
    }
    pthread_mutex_unlock(&backend->mutex);
    return NULL;
}

/* --------------------------------------------------------------- streams */

static int create_stream(void *state, ShadowSpillBackendStream *stream) {
    ShadowSpillMockBackend *backend = state;
    if (stream == NULL || operation_fails(backend)) {
        return -1;
    }
    MockStream *created = calloc(1U, sizeof(*created));
    if (created == NULL) {
        return -1;
    }
    pthread_mutex_lock(&backend->mutex);
    created->next = backend->streams;
    backend->streams = created;
    pthread_mutex_unlock(&backend->mutex);
    *stream = (ShadowSpillBackendStream)(uintptr_t)created;
    count(backend, &backend->statistics.streams_created, 1U);
    return 0;
}

static void unlink_stream_locked(
    ShadowSpillMockBackend *backend, MockStream *stream
) {
    MockStream **link = &backend->streams;
    while (*link != NULL && *link != stream) {
        link = &(*link)->next;
    }
    if (*link == stream) {
        *link = stream->next;
    }
}

static int destroy_stream(void *state, ShadowSpillBackendStream stream) {
    ShadowSpillMockBackend *backend = state;
    if (operation_fails(backend)) {
        return -1;
    }
    MockStream *target = stream_pointer(stream);
    /* The default stream lives inside the backend and goes with it. Freeing it
       here would be freeing the middle of the backend. */
    if (target == NULL || target == &backend->default_stream) {
        return -1;
    }
    for (;;) {
        pthread_mutex_lock(&backend->mutex);
        drain_locked(backend, target);
        if (!target->draining) {
            discard_queue_locked(backend, target);
            unlink_stream_locked(backend, target);
            pthread_mutex_unlock(&backend->mutex);
            break;
        }
        pthread_mutex_unlock(&backend->mutex);
        sleep_briefly();
    }
    free(target);
    count(backend, &backend->statistics.streams_destroyed, 1U);
    return 0;
}

static int synchronize_stream(void *state, ShadowSpillBackendStream stream) {
    ShadowSpillMockBackend *backend = state;
    if (operation_fails(backend)) {
        return -1;
    }
    MockStream *target = stream_pointer(stream);
    if (target == NULL) {
        return -1;
    }
    for (;;) {
        pthread_mutex_lock(&backend->mutex);
        drain_locked(backend, target);
        const int idle = target->head == NULL && !target->draining;
        const uint64_t ready = target->ready_nanoseconds;
        pthread_mutex_unlock(&backend->mutex);
        if (idle && now_nanoseconds() >= ready) {
            break;
        }
        sleep_briefly();
    }
    count(backend, &backend->statistics.stream_synchronizations, 1U);
    return 0;
}

/* A framework handle is a MockStream the framework created through this
 * table, except 0, which is the default stream: the one a driver runs work
 * on when the caller names no stream. Without it a caller with no stream of
 * its own could not drive the table at all. */
static ShadowSpillBackendStream resolve_stream(
    void *state, uint64_t stream_handle
) {
    ShadowSpillMockBackend *backend = state;
    /* The mock keeps its own stream objects, so 0 -- the default stream --
       has to name one it owns. Every other handle is already one of ours. */
    return stream_handle == 0U
        ? (ShadowSpillBackendStream)(uintptr_t)&backend->default_stream
        : (ShadowSpillBackendStream)stream_handle;
}

/* ---------------------------------------------------------------- copies */

static int delayed_copy(
    ShadowSpillMockBackend *backend,
    void *destination,
    const void *source,
    uint64_t bytes,
    ShadowSpillBackendStream stream,
    uint64_t delay_nanoseconds,
    uint64_t *copies,
    uint64_t *copied_bytes
) {
    if ((bytes != 0U && (destination == NULL || source == NULL)) ||
        bytes > SIZE_MAX || operation_fails(backend)) {
        return -1;
    }
    MockStream *target = stream_pointer(stream);
    if (target == NULL) {
        return -1;
    }
    MockOperation *entry = entry_for(MOCK_COPY);
    if (entry == NULL) {
        return -1;
    }
    entry->destination = destination;
    entry->source = source;
    entry->bytes = bytes;
    entry->duration_nanoseconds = delay_nanoseconds;
    /* Counted where it was asked for rather than where it runs, so the numbers
       a caller reads back describe the calls it made. */
    pthread_mutex_lock(&backend->mutex);
    ++*copies;
    *copied_bytes += bytes;
    pthread_mutex_unlock(&backend->mutex);
    submit(backend, target, entry);
    return 0;
}

static int copy_host_to_device(
    void *state, void *device, const void *host, uint64_t bytes,
    ShadowSpillBackendStream stream
) {
    ShadowSpillMockBackend *backend = state;
    return delayed_copy(
        backend, device, host, bytes, stream,
        backend->config.fetch_delay_nanoseconds,
        &backend->statistics.copies_host_to_device,
        &backend->statistics.bytes_host_to_device
    );
}

static int copy_device_to_host(
    void *state, void *host, const void *device, uint64_t bytes,
    ShadowSpillBackendStream stream
) {
    ShadowSpillMockBackend *backend = state;
    return delayed_copy(
        backend, host, device, bytes, stream,
        backend->config.evict_delay_nanoseconds,
        &backend->statistics.copies_device_to_host,
        &backend->statistics.bytes_device_to_host
    );
}

static int copy_device_to_device(
    void *state, void *destination, const void *source, uint64_t bytes,
    ShadowSpillBackendStream stream
) {
    ShadowSpillMockBackend *backend = state;
    return delayed_copy(
        backend, destination, source, bytes, stream, 0U,
        &backend->statistics.copies_device_to_device,
        &backend->statistics.bytes_device_to_device
    );
}

/* ---------------------------------------------------------------- events */

static int create_event(void *state, ShadowSpillBackendEvent *event, uint8_t timing) {
    ShadowSpillMockBackend *backend = state;
    (void)timing;
    if (event == NULL || operation_fails(backend)) {
        return -1;
    }
    MockEvent *created = calloc(1U, sizeof(*created));
    if (created == NULL) {
        return -1;
    }
    created->references = 1U;
    *event = (ShadowSpillBackendEvent)(uintptr_t)created;
    count(backend, &backend->statistics.events_created, 1U);
    return 0;
}

static int destroy_event(void *state, ShadowSpillBackendEvent event) {
    ShadowSpillMockBackend *backend = state;
    if (operation_fails(backend)) {
        return -1;
    }
    MockEvent *target = event_pointer(event);
    if (target != NULL) {
        /* The creator's hold. A queue that still refers to the event keeps it
           until that entry is consumed. */
        pthread_mutex_lock(&backend->mutex);
        release_event_locked(target);
        pthread_mutex_unlock(&backend->mutex);
    }
    count(backend, &backend->statistics.events_destroyed, 1U);
    return 0;
}

/*
 * The event is recorded at once and complete later.
 *
 * Recording has to take effect immediately, because a caller may wait on or
 * query the event the instant this returns and an unrecorded event is refused.
 * What waits is completion: the event is stamped when the entry reaches the
 * head of this stream, so an event recorded behind a wait cannot complete until
 * that wait does.
 */
static int record_event(
    void *state, ShadowSpillBackendEvent event, ShadowSpillBackendStream stream
) {
    ShadowSpillMockBackend *backend = state;
    if (operation_fails(backend)) {
        return -1;
    }
    MockEvent *target = event_pointer(event);
    MockStream *source = stream_pointer(stream);
    if (target == NULL || source == NULL) {
        return -1;
    }
    MockOperation *entry = entry_for(MOCK_RECORD_EVENT);
    if (entry == NULL) {
        return -1;
    }
    entry->event = target;
    pthread_mutex_lock(&backend->mutex);
    entry->stamp = ++target->stamp;
    target->state = MOCK_EVENT_QUEUED;
    ++target->references;
    pthread_mutex_unlock(&backend->mutex);
    submit(backend, source, entry);
    return 0;
}

static int query_event(void *state, ShadowSpillBackendEvent event, int *complete) {
    ShadowSpillMockBackend *backend = state;
    if (complete == NULL || operation_fails(backend)) {
        return -1;
    }
    MockEvent *target = event_pointer(event);
    if (target == NULL) {
        return -1;
    }
    pthread_mutex_lock(&backend->mutex);
    ++backend->statistics.event_queries;
    if (target->state == MOCK_EVENT_UNRECORDED) {
        pthread_mutex_unlock(&backend->mutex);
        return -1;
    }
    *complete = event_complete_locked(target);
    pthread_mutex_unlock(&backend->mutex);
    return 0;
}

/*
 * Signal words, and a stream that waits on one.
 *
 * The mock has no device, so "the stream waits until the word reaches a value"
 * becomes "the entries behind the wait do not run until the host stores that
 * value". A wait on a word already past what it awaits runs at once; one on a
 * word that has not arrived holds the queue, and the backend's drain thread is
 * what notices when it does. That is the same shape a driver's value wait has,
 * which is what the lane layer above is written against.
 */
typedef struct MockSignals {
    uint64_t *words;
    uint32_t count;
} MockSignals;

static MockSignals *signals_pointer(ShadowSpillBackendSignals signals) {
    return (MockSignals *)(uintptr_t)signals;
}

static int allocate_signals(
    void *state,
    uint32_t count,
    ShadowSpillBackendSignals *signals,
    uint64_t **host
) {
    ShadowSpillMockBackend *backend = state;
    if (operation_fails(backend)) {
        return -1;
    }
    if (count == 0U || signals == NULL || host == NULL) {
        return -1;
    }
    MockSignals *created = calloc(1U, sizeof(*created));
    if (created == NULL) {
        return -1;
    }
    created->words = calloc(count, sizeof(*created->words));
    if (created->words == NULL) {
        free(created);
        return -1;
    }
    created->count = count;
    *signals = (ShadowSpillBackendSignals)(uintptr_t)created;
    *host = created->words;
    return 0;
}

static int free_signals(void *state, ShadowSpillBackendSignals signals) {
    ShadowSpillMockBackend *backend = state;
    if (operation_fails(backend)) {
        return -1;
    }
    MockSignals *target = signals_pointer(signals);
    if (target == NULL) {
        return -1;
    }
    free(target->words);
    free(target);
    return 0;
}

static int wait_value(
    void *state,
    ShadowSpillBackendStream stream,
    ShadowSpillBackendSignals signals,
    uint32_t index,
    uint64_t value
) {
    ShadowSpillMockBackend *backend = state;
    if (operation_fails(backend)) {
        return -1;
    }
    MockStream *target = stream_pointer(stream);
    MockSignals *block = signals_pointer(signals);
    if (target == NULL || block == NULL || index >= block->count) {
        return -1;
    }
    MockOperation *entry = entry_for(MOCK_WAIT_VALUE);
    if (entry == NULL) {
        return -1;
    }
    entry->word = &block->words[index];
    entry->value = value;
    count(backend, &backend->statistics.stream_waits, 1U);
    submit(backend, target, entry);
    return 0;
}

/*
 * The mirror of `wait_value`. The stream stores the word when it reaches this
 * entry, which is after everything queued before it has run -- so a host thread
 * that reads the word learns how far the stream has got, which is the whole
 * question it answers.
 */
static int write_value(
    void *state,
    ShadowSpillBackendStream stream,
    ShadowSpillBackendSignals signals,
    uint32_t index,
    uint64_t value
) {
    ShadowSpillMockBackend *backend = state;
    if (operation_fails(backend)) {
        return -1;
    }
    MockStream *target = stream_pointer(stream);
    MockSignals *block = signals_pointer(signals);
    if (target == NULL || block == NULL || index >= block->count) {
        return -1;
    }
    MockOperation *entry = entry_for(MOCK_WRITE_VALUE);
    if (entry == NULL) {
        return -1;
    }
    entry->word = &block->words[index];
    entry->value = value;
    count(backend, &backend->statistics.stream_writes, 1U);
    submit(backend, target, entry);
    return 0;
}

static int wait_event(
    void *state, ShadowSpillBackendStream stream, ShadowSpillBackendEvent event
) {
    ShadowSpillMockBackend *backend = state;
    if (operation_fails(backend)) {
        return -1;
    }
    MockStream *target = stream_pointer(stream);
    MockEvent *source = event_pointer(event);
    if (target == NULL || source == NULL) {
        return -1;
    }
    MockOperation *entry = entry_for(MOCK_WAIT_EVENT);
    if (entry == NULL) {
        return -1;
    }
    pthread_mutex_lock(&backend->mutex);
    const int recorded = source->state != MOCK_EVENT_UNRECORDED;
    if (recorded) {
        /* The entry's hold, so the caller may destroy the event before this
           stream reaches the wait. */
        ++source->references;
    }
    pthread_mutex_unlock(&backend->mutex);
    if (!recorded) {
        free(entry);
        return -1;
    }
    entry->event = source;
    count(backend, &backend->statistics.stream_waits, 1U);
    submit(backend, target, entry);
    return 0;
}

static int synchronize_event(void *state, ShadowSpillBackendEvent event) {
    ShadowSpillMockBackend *backend = state;
    if (operation_fails(backend)) {
        return -1;
    }
    MockEvent *target = event_pointer(event);
    if (target == NULL) {
        return -1;
    }
    /* The mock's events complete on the host clock, so waiting for one is
     * waiting for the stream to reach its record and for that instant to pass.
     * The drain thread is what advances a stream nobody is calling into. */
    for (;;) {
        pthread_mutex_lock(&backend->mutex);
        const int recorded = target->state != MOCK_EVENT_UNRECORDED;
        const int complete = event_complete_locked(target);
        pthread_mutex_unlock(&backend->mutex);
        if (!recorded) {
            return -1;
        }
        if (complete) {
            return 0;
        }
        sleep_briefly();
    }
}

static int elapsed_nanoseconds(
    void *state, ShadowSpillBackendEvent from, ShadowSpillBackendEvent to,
    uint64_t *nanoseconds
) {
    ShadowSpillMockBackend *backend = state;
    if (nanoseconds == NULL || operation_fails(backend)) {
        return -1;
    }
    MockEvent *origin = event_pointer(from);
    MockEvent *target = event_pointer(to);
    if (origin == NULL || target == NULL) {
        return -1;
    }
    pthread_mutex_lock(&backend->mutex);
    if (origin->state == MOCK_EVENT_UNRECORDED ||
        target->state == MOCK_EVENT_UNRECORDED) {
        pthread_mutex_unlock(&backend->mutex);
        return -1;
    }
    if (!event_complete_locked(origin) || !event_complete_locked(target)) {
        pthread_mutex_unlock(&backend->mutex);
        return 1;
    }
    *nanoseconds = target->ready_nanoseconds > origin->ready_nanoseconds
        ? target->ready_nanoseconds - origin->ready_nanoseconds
        : 0U;
    pthread_mutex_unlock(&backend->mutex);
    return 0;
}

/* ----------------------------------------------------------------- facts */

static int capabilities(void *state, ShadowSpillBackendCapabilities *out) {
    if (state == NULL || out == NULL) {
        return -1;
    }
    *out = (ShadowSpillBackendCapabilities){
        .device_ordinal = 0,
        .minimum_alignment = 256U,
        .provider = "mock",
    };
    return 0;
}

static int physical_memory(void *state, ShadowSpillBackendPhysicalMemory *out) {
    if (state == NULL || out == NULL) {
        return -1;
    }
    /* Host memory stands in for the device: nothing is used, and the total
       is large enough for any budget a test asks for. */
    *out = (ShadowSpillBackendPhysicalMemory){
        .device_total_bytes = UINT64_C(1) << 40U,
    };
    return 0;
}

static void statistics(void *state, ShadowSpillBackendStatistics *out) {
    ShadowSpillMockBackend *backend = state;
    if (backend == NULL || out == NULL) {
        return;
    }
    pthread_mutex_lock(&backend->mutex);
    *out = backend->statistics;
    pthread_mutex_unlock(&backend->mutex);
}

/* -------------------------------------------------------------- lifetime */

static ShadowSpillBackend interface_for(ShadowSpillMockBackend *backend) {
    return (ShadowSpillBackend){
        .abi_version = SHADOWSPILL_BACKEND_ABI_VERSION,
        .state = backend,
        .allocate_device = allocate_device,
        .free_device = free_device,
        .register_host_memory = register_host_memory,
        .unregister_host_memory = unregister_host_memory,
        .allocate_signals = allocate_signals,
        .free_signals = free_signals,
        .wait_value = wait_value,
        .write_value = write_value,
        .create_stream = create_stream,
        .destroy_stream = destroy_stream,
        .synchronize_stream = synchronize_stream,
        .resolve_stream = resolve_stream,
        .copy_host_to_device = copy_host_to_device,
        .copy_device_to_host = copy_device_to_host,
        .copy_device_to_device = copy_device_to_device,
        .create_event = create_event,
        .destroy_event = destroy_event,
        .record_event = record_event,
        .query_event = query_event,
        .wait_event = wait_event,
        .synchronize_event = synchronize_event,
        .elapsed_nanoseconds = elapsed_nanoseconds,
        .capabilities = capabilities,
        .physical_memory = physical_memory,
        .statistics = statistics,
    };
}

int shadowspill_mock_backend_create(
    const ShadowSpillMockBackendConfig *config,
    ShadowSpillBackend *backend
) {
    if (config == NULL || backend == NULL) {
        return -1;
    }
    ShadowSpillMockBackend *mock = calloc(1U, sizeof(*mock));
    if (mock == NULL) {
        return -1;
    }
    mock->config = *config;
    if (pthread_mutex_init(&mock->mutex, NULL) != 0) {
        free(mock);
        return -1;
    }
    if (pthread_cond_init(&mock->queued, NULL) != 0) {
        pthread_mutex_destroy(&mock->mutex);
        free(mock);
        return -1;
    }
    mock->streams = &mock->default_stream;
    if (pthread_create(&mock->drain_thread, NULL, drain_streams, mock) != 0) {
        pthread_cond_destroy(&mock->queued);
        pthread_mutex_destroy(&mock->mutex);
        free(mock);
        return -1;
    }
    *backend = interface_for(mock);
    return 0;
}

SHADOWSPILL_BACKEND_MOCK_API int shadowspill_backend_create(
    const ShadowSpillBackendConfig *config,
    ShadowSpillBackend *backend
) {
    if (config == NULL || backend == NULL ||
        config->abi_version != SHADOWSPILL_BACKEND_ABI_VERSION) {
        return -1;
    }
    const ShadowSpillMockBackendConfig mock_config = {0};
    return shadowspill_mock_backend_create(&mock_config, backend);
}

SHADOWSPILL_BACKEND_MOCK_API void shadowspill_backend_destroy(
    ShadowSpillBackend *backend
) {
    if (backend == NULL || backend->state == NULL) {
        return;
    }
    ShadowSpillMockBackend *mock = backend->state;
    pthread_mutex_lock(&mock->mutex);
    mock->stopping = 1;
    pthread_cond_broadcast(&mock->queued);
    pthread_mutex_unlock(&mock->mutex);
    (void)pthread_join(mock->drain_thread, NULL);
    /* Streams a caller did not destroy go with the backend, and so does
       anything still queued on them: there is nothing left to observe it. */
    while (mock->streams != NULL) {
        MockStream *const stream = mock->streams;
        mock->streams = stream->next;
        discard_queue_locked(mock, stream);
        if (stream != &mock->default_stream) {
            free(stream);
        }
    }
    pthread_cond_destroy(&mock->queued);
    pthread_mutex_destroy(&mock->mutex);
    free(mock);
    memset(backend, 0, sizeof(*backend));
}

/* -------------------------------------------------------------- topology */

void shadowspill_mock_runtime_topology(
    const ShadowSpillBackend *backend,
    uint64_t execution_pool_bytes,
    uint64_t spill_pool_bytes,
    uint64_t minimum_alignment,
    uint64_t worker_poll_nanoseconds,
    ShadowSpillMockRuntimeTopology *topology
) {
    if (topology == NULL || backend == NULL) {
        return;
    }
    memset(topology, 0, sizeof(*topology));
    topology->backend = *backend;
    topology->pools[0] = (ShadowSpillMemoryPoolDescription){
        .pool_id = 0U,
        .kind = SHADOWSPILL_POOL_DEVICE,
        .capacity_bytes = execution_pool_bytes,
        .minimum_alignment = minimum_alignment,
    };
    topology->pools[1] = (ShadowSpillMemoryPoolDescription){
        .pool_id = 1U,
        .kind = SHADOWSPILL_POOL_PINNED_HOST,
        .capacity_bytes = spill_pool_bytes,
        .minimum_alignment = 1U,
    };
    topology->routes[0] = (ShadowSpillTransferRouteDescription){
        .route_id = 0U,
        .name = "shadowspill_fetch",
        .source_pool_id = 1U,
        .destination_pool_id = 0U,
    };
    topology->routes[1] = (ShadowSpillTransferRouteDescription){
        .route_id = 1U,
        .name = "shadowspill_evict",
        .source_pool_id = 0U,
        .destination_pool_id = 1U,
    };
    topology->runtime = (ShadowSpillRuntimeConfig){
        .abi_version = SHADOWSPILL_ABI_VERSION,
        .backend = &topology->backend,
        .pools = topology->pools,
        .pool_count = 2U,
        .routes = topology->routes,
        .route_count = 2U,
        .worker_poll_nanoseconds = worker_poll_nanoseconds,
    };
}

/* ------------------------------------------------------------ test hooks */

int shadowspill_mock_enqueue_compute(
    const ShadowSpillBackend *backend,
    ShadowSpillBackendStream stream,
    uint64_t duration_nanoseconds
) {
    ShadowSpillMockBackend *mock = backend == NULL ? NULL : backend->state;
    if (mock == NULL || operation_fails(mock)) {
        return -1;
    }
    MockStream *target = stream_pointer(stream);
    if (target == NULL) {
        return -1;
    }
    MockOperation *entry = entry_for(MOCK_COMPUTE);
    if (entry == NULL) {
        return -1;
    }
    entry->duration_nanoseconds = duration_nanoseconds;
    submit(mock, target, entry);
    return 0;
}

void shadowspill_mock_fail_operation(
    const ShadowSpillBackend *backend, uint64_t operation_number
) {
    ShadowSpillMockBackend *mock = backend == NULL ? NULL : backend->state;
    if (mock == NULL) {
        return;
    }
    pthread_mutex_lock(&mock->mutex);
    mock->fail_operation = operation_number;
    pthread_mutex_unlock(&mock->mutex);
}

void shadowspill_mock_fail_next_operation(const ShadowSpillBackend *backend) {
    ShadowSpillMockBackend *mock = backend == NULL ? NULL : backend->state;
    if (mock == NULL) {
        return;
    }
    pthread_mutex_lock(&mock->mutex);
    mock->fail_operation = mock->operation_count + 1U;
    pthread_mutex_unlock(&mock->mutex);
}

void shadowspill_mock_backend_statistics(
    const ShadowSpillBackend *backend, ShadowSpillMockBackendStatistics *statistics
) {
    ShadowSpillMockBackend *mock = backend == NULL ? NULL : backend->state;
    if (mock == NULL || statistics == NULL) {
        return;
    }
    pthread_mutex_lock(&mock->mutex);
    statistics->operation_count = mock->operation_count;
    pthread_mutex_unlock(&mock->mutex);
}
