/* Loads a backend the way the adapter does, by path through the two exported
 * symbols, and drives every entry of the contract once. The mock backend is
 * the subject because it needs no accelerator. */
#include <shadowspill/backend.h>

#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

#define FAIL(message)                                                          \
    do {                                                                       \
        fprintf(stderr, "backend contract canary: %s\n", message);            \
        return 1;                                                              \
    } while (0)

int main(int argc, char **argv) {
    if (argc != 2) {
        FAIL("usage: backend_contract_canary <backend shared object>");
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
    const ShadowSpillBackendConfig wrong = {.abi_version = 0U, .device_ordinal = 0};
    ShadowSpillBackend backend = {0};
    if (create.create(&wrong, &backend) == 0) {
        FAIL("a wrong contract version was accepted");
    }
    const ShadowSpillBackendConfig config = {
        .abi_version = SHADOWSPILL_BACKEND_ABI_VERSION,
        .device_ordinal = 0,
    };
    if (create.create(&config, &backend) != 0) {
        FAIL("create failed");
    }
    if (backend.abi_version != SHADOWSPILL_BACKEND_ABI_VERSION || backend.state == NULL ||
        backend.allocate_device == NULL || backend.free_device == NULL ||
        backend.register_host_memory == NULL || backend.unregister_host_memory == NULL ||
        backend.allocate_signals == NULL || backend.free_signals == NULL ||
        backend.wait_value == NULL ||
        backend.create_stream == NULL || backend.destroy_stream == NULL ||
        backend.synchronize_stream == NULL || backend.resolve_stream == NULL ||
        backend.copy_host_to_device == NULL || backend.copy_device_to_host == NULL ||
        backend.copy_device_to_device == NULL || backend.create_event == NULL ||
        backend.destroy_event == NULL || backend.record_event == NULL ||
        backend.query_event == NULL || backend.wait_event == NULL ||
        backend.synchronize_event == NULL ||
        backend.elapsed_nanoseconds == NULL || backend.capabilities == NULL ||
        backend.physical_memory == NULL || backend.statistics == NULL) {
        FAIL("the table is incomplete");
    }
    /*
     * The sequence a lane that does not move bytes on a stream depends on: the
     * stream is held until the host stores the generation, and a wait on a
     * generation already stored does not hold it at all. Proven here, on the
     * mock, so the mechanism is not first exercised by a NIC.
     */
    ShadowSpillBackendSignals signals = 0U;
    uint64_t *words = NULL;
    if (backend.allocate_signals(backend.state, 2U, &signals, &words) != 0 ||
        words == NULL || words[0] != 0U || words[1] != 0U) {
        FAIL("signal words did not come back zeroed");
    }
    ShadowSpillBackendStream signal_stream = 0U;
    if (backend.create_stream(backend.state, &signal_stream) != 0) {
        FAIL("could not create a stream to wait on a value");
    }
    /* Already satisfied: generation 0 is what the word holds. */
    if (backend.wait_value(backend.state, signal_stream, signals, 0U, 0U) != 0) {
        FAIL("a wait on a generation already stored was refused");
    }
    /* Not yet: the host has not stored 7. Then it does, and the stream may go. */
    if (backend.wait_value(backend.state, signal_stream, signals, 1U, 7U) != 0) {
        FAIL("a wait on a generation not yet stored was refused");
    }
    words[1] = 7U;
    if (backend.synchronize_stream(backend.state, signal_stream) != 0) {
        FAIL("the stream did not pass once the generation was stored");
    }
    if (backend.wait_value(backend.state, signal_stream, signals, 2U, 1U) == 0) {
        FAIL("an index past the end of the block was accepted");
    }
    if (backend.destroy_stream(backend.state, signal_stream) != 0 ||
        backend.free_signals(backend.state, signals) != 0) {
        FAIL("signal teardown failed");
    }

    ShadowSpillBackendCapabilities capabilities = {0};
    if (backend.capabilities(backend.state, &capabilities) != 0 ||
        capabilities.minimum_alignment == 0U || capabilities.provider[0] == '\0') {
        FAIL("capabilities are incomplete");
    }
    ShadowSpillBackendPhysicalMemory memory = {0};
    if (backend.physical_memory(backend.state, &memory) != 0 ||
        memory.device_total_bytes == 0U) {
        FAIL("physical memory is incomplete");
    }
    void *device = NULL;
    char host[64] = "the quick brown fox";
    char back[64] = {0};
    if (backend.allocate_device(backend.state, sizeof(host), &device) != 0 || device == NULL) {
        FAIL("allocate_device failed");
    }
    if (backend.register_host_memory(backend.state, host, sizeof(host)) != 0 ||
        backend.register_host_memory(backend.state, back, sizeof(back)) != 0) {
        FAIL("register_host_memory failed");
    }
    ShadowSpillBackendStream stream = 0U;
    if (backend.create_stream(backend.state, &stream) != 0) {
        FAIL("create_stream failed");
    }
    ShadowSpillBackendEvent start = 0U;
    ShadowSpillBackendEvent end = 0U;
    if (backend.create_event(backend.state, &start, 1U) != 0 ||
        backend.create_event(backend.state, &end, 1U) != 0 ||
        backend.record_event(backend.state, start, stream) != 0) {
        FAIL("event creation or record failed");
    }
    if (backend.copy_host_to_device(backend.state, device, host, sizeof(host), stream) != 0 ||
        backend.copy_device_to_device(backend.state, device, device, 0U, stream) != 0 ||
        backend.copy_device_to_host(backend.state, back, device, sizeof(back), stream) != 0 ||
        backend.record_event(backend.state, end, stream) != 0 ||
        backend.wait_event(backend.state, stream, end) != 0 ||
        backend.synchronize_event(backend.state, end) != 0 ||
        backend.synchronize_stream(backend.state, stream) != 0) {
        FAIL("copies or stream ordering failed");
    }
    if (strcmp(host, back) != 0) {
        FAIL("the round trip did not preserve the bytes");
    }
    int complete = 0;
    uint64_t nanoseconds = 0U;
    if (backend.query_event(backend.state, end, &complete) != 0 || !complete ||
        backend.elapsed_nanoseconds(backend.state, start, end, &nanoseconds) != 0) {
        FAIL("event query or elapsed time failed");
    }
    const ShadowSpillBackendStream resolved = backend.resolve_stream(backend.state, 7U);
    if (resolved != 7U) {
        FAIL("resolve_stream does not carry a handle the backend already knows");
    }
    if (backend.destroy_event(backend.state, start) != 0 ||
        backend.destroy_event(backend.state, end) != 0 ||
        backend.destroy_stream(backend.state, stream) != 0 ||
        backend.unregister_host_memory(backend.state, host, sizeof(host)) != 0 ||
        backend.unregister_host_memory(backend.state, back, sizeof(back)) != 0 ||
        backend.free_device(backend.state, device, sizeof(host)) != 0) {
        FAIL("teardown of streams, events, or memory failed");
    }
    ShadowSpillBackendStatistics statistics;
    memset(&statistics, 0xFF, sizeof(statistics));
    backend.statistics(backend.state, &statistics);
    if (statistics.copies_host_to_device != 1U || statistics.copies_device_to_host != 1U ||
        statistics.device_allocations != 1U || statistics.device_frees != 1U ||
        statistics.pinned_host_registrations != 2U || statistics.bytes_pinned_host_registered != 128U ||
        statistics.bytes_pinned_host_unregistered != 128U || statistics.bytes_device_allocated != 64U ||
        statistics.bytes_device_freed != 64U || statistics.events_created != 2U ||
        /* Two streams now: the transfer stream, and the one the value-wait
           sequence above waits on. */
        statistics.streams_created != 2U) {
        FAIL("statistics do not count the calls made");
    }
    destroy.destroy(&backend);
    if (backend.state != NULL) {
        FAIL("destroy left the table populated");
    }
    (void)dlclose(library);
    printf("backend contract canary: ok (%s)\n", capabilities.provider);
    return 0;
}
