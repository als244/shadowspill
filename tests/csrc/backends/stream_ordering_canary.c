/*
 * A stream is an ordered queue, and a value wait is a wait.
 *
 * `backend.h` says both, and a backend can satisfy every other check in the
 * contract while honouring neither: recording a wait and enforcing one are
 * different properties, and only the first is visible to a caller that never
 * puts work behind a wait. The mock did exactly that for as long as nothing
 * asked -- copies ran as they were called, and the pending wait was consulted
 * only by whoever came to observe the stream.
 *
 * This is what a lane whose bytes do not move on a stream rests on, so it is
 * checked here rather than first exercised by a NIC. Driven through the table
 * alone, so it runs against any backend: the mock with no accelerator, and a
 * provider's driver where there is one.
 *
 *   1. work behind an unsatisfied wait has not happened -- the copy, the value
 *      the stream writes, and the event recorded after them;
 *   2. satisfying the wait lets all three through;
 *   3. the value the stream wrote means the copy before it landed, which is
 *      the question a lane staging through a ring asks of every slot.
 */
#include <shadowspill/backend.h>

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define PAYLOAD_BYTES 4096U
/* What the gated copy's destination holds until the copy runs. */
#define UNWRITTEN 0xA5U
/* Word 0 is the gate a host thread opens; word 1 is what the stream reports
   through, which is the pair of directions a staged transfer needs. */
#define GATE 0U
#define PROGRESS 1U
#define REPORTED 9U
/* Long enough that a backend running gated work early has done it by the time
   the assertions run, and it is not waiting on anything: nothing stores the
   gate word until they have. */
#define SETTLE_NANOSECONDS 5000000U
/* A backend that never lets the gated work through fails here rather than
   hanging until ctest kills it. */
#define DEADLINE_SECONDS 10.0

#define FAIL(message)                                                          \
    do {                                                                       \
        fprintf(stderr, "stream ordering canary: %s\n", message);              \
        return 1;                                                              \
    } while (0)

static double seconds_now(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
}

static void settle(void) {
    struct timespec delay = {.tv_nsec = SETTLE_NANOSECONDS};
    (void)nanosleep(&delay, NULL);
}

static int all_bytes_are(const unsigned char *bytes, uint64_t count, unsigned char value) {
    for (uint64_t index = 0U; index < count; ++index) {
        if (bytes[index] != value) {
            return 0;
        }
    }
    return 1;
}

/* Every word carries its own index, so a partial or misaddressed copy cannot
   reproduce it. */
static void fill(uint64_t *words, uint64_t bytes) {
    for (uint64_t index = 0U; index < bytes / sizeof(*words); ++index) {
        words[index] = index * 0x9E3779B97F4A7C15ULL;
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        FAIL("usage: stream_ordering_canary <backend shared object>");
    }
    void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (library == NULL) {
        FAIL(dlerror());
    }
    union {
        void *object;
        ShadowSpillBackendCreate create;
    } create = {.object = dlsym(library, SHADOWSPILL_BACKEND_CREATE_SYMBOL)};
    union {
        void *object;
        ShadowSpillBackendDestroy destroy;
    } destroy = {.object = dlsym(library, SHADOWSPILL_BACKEND_DESTROY_SYMBOL)};
    if (create.object == NULL || destroy.object == NULL) {
        FAIL("the backend does not export both contract symbols");
    }
    const ShadowSpillBackendConfig config = {
        .abi_version = SHADOWSPILL_BACKEND_ABI_VERSION,
        .device_ordinal = 0,
    };
    ShadowSpillBackend backend = {0};
    if (create.create(&config, &backend) != 0) {
        FAIL("create failed");
    }

    static uint64_t source[PAYLOAD_BYTES / sizeof(uint64_t)];
    static uint64_t destination[PAYLOAD_BYTES / sizeof(uint64_t)];
    fill(source, sizeof(source));
    memset(destination, UNWRITTEN, sizeof(destination));

    ShadowSpillBackendSignals signals = 0U;
    uint64_t *words = NULL;
    ShadowSpillBackendStream stream = 0U;
    void *device = NULL;
    ShadowSpillBackendEvent arrived = 0U;
    if (backend.allocate_signals(backend.state, 2U, &signals, &words) != 0 ||
        words == NULL) {
        FAIL("could not allocate signal words");
    }
    if (backend.create_stream(backend.state, &stream) != 0 ||
        backend.create_event(backend.state, &arrived, 0U) != 0 ||
        backend.allocate_device(backend.state, sizeof(source), &device) != 0 ||
        backend.register_host_memory(backend.state, source, sizeof(source)) != 0 ||
        backend.register_host_memory(
            backend.state, destination, sizeof(destination)
        ) != 0) {
        FAIL("could not set up a stream, an event, and the two ends of a copy");
    }

    /* The payload reaches the device with nothing gated, so the gated copy
       below has something real to bring back. */
    if (backend.copy_host_to_device(
            backend.state, device, source, sizeof(source), stream
        ) != 0 ||
        backend.synchronize_stream(backend.state, stream) != 0) {
        FAIL("the ungated copy out to the device did not complete");
    }

    /*
     * Everything from here is behind a wait on a word no one has stored.
     *
     * The order is the one a staged fetch issues: wait for the hardware to have
     * filled the buffer, copy it, then report that the stream is finished with
     * it -- and an event recorded last, which is what `signal` leaves for the
     * runtime to wait on.
     */
    if (backend.wait_value(backend.state, stream, signals, GATE, 1U) != 0) {
        FAIL("a wait on a value not yet stored was refused");
    }
    if (backend.copy_device_to_host(
            backend.state, destination, device, sizeof(destination), stream
        ) != 0 ||
        backend.write_value(
            backend.state, stream, signals, PROGRESS, REPORTED
        ) != 0 ||
        backend.record_event(backend.state, arrived, stream) != 0) {
        FAIL("the gated copy, report and record were refused");
    }

    settle();

    if (!all_bytes_are((const unsigned char *)destination, sizeof(destination), UNWRITTEN)) {
        FAIL("a copy behind an unsatisfied wait landed anyway");
    }
    if (__atomic_load_n(&words[PROGRESS], __ATOMIC_ACQUIRE) != 0U) {
        FAIL("a value written behind an unsatisfied wait was stored anyway");
    }
    int complete = 1;
    if (backend.query_event(backend.state, arrived, &complete) != 0) {
        FAIL("an event recorded behind an unsatisfied wait could not be queried");
    }
    if (complete) {
        FAIL("an event recorded behind an unsatisfied wait reported complete");
    }

    /* Open it, the way a lane's own thread does: a plain store to the word. */
    __atomic_store_n(&words[GATE], 1U, __ATOMIC_RELEASE);

    const double deadline = seconds_now() + DEADLINE_SECONDS;
    while (__atomic_load_n(&words[PROGRESS], __ATOMIC_ACQUIRE) != REPORTED) {
        if (seconds_now() > deadline) {
            FAIL("the stream never got past a wait whose value had been stored");
        }
    }
    /*
     * Read without synchronizing, on purpose. What the report means is that the
     * stream is past the copy, and a lane reuses a staging slot on exactly that
     * -- so if the bytes are not here, the word said so too early.
     */
    if (memcmp(destination, source, sizeof(source)) != 0) {
        FAIL("the stream reported past a copy whose bytes had not landed");
    }

    if (backend.synchronize_stream(backend.state, stream) != 0) {
        FAIL("the stream did not drain once its wait was satisfied");
    }
    if (backend.query_event(backend.state, arrived, &complete) != 0 || !complete) {
        FAIL("the event behind the satisfied wait never completed");
    }

    /* A wait on a value already past what it awaits does not hold the stream:
       greater-or-equal, which is what lets a lane enqueue a wait after the
       value it names has been and gone. */
    if (backend.wait_value(backend.state, stream, signals, GATE, 1U) != 0 ||
        backend.synchronize_stream(backend.state, stream) != 0) {
        FAIL("a wait on a value already stored held the stream");
    }

    if (backend.destroy_event(backend.state, arrived) != 0 ||
        backend.destroy_stream(backend.state, stream) != 0 ||
        backend.unregister_host_memory(backend.state, source, sizeof(source)) != 0 ||
        backend.unregister_host_memory(
            backend.state, destination, sizeof(destination)
        ) != 0 ||
        backend.free_device(backend.state, device, sizeof(source)) != 0 ||
        backend.free_signals(backend.state, signals) != 0) {
        FAIL("teardown failed");
    }
    destroy.destroy(&backend);
    (void)dlclose(library);
    printf("stream ordering canary: ok\n");
    return 0;
}
