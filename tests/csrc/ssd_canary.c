#define _GNU_SOURCE
#include "../../csrc/ssd/internal.h"

#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "SSD canary line %d: %s\n", __LINE__, #condition); return 1; \
} } while (0)

static int entries(const char *directory) {
    DIR *dir = opendir(directory);
    if (dir == NULL) return -1;
    int count = 0;
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL)
        if (strcmp(entry->d_name, ".") && strcmp(entry->d_name, "..")) ++count;
    closedir(dir);
    return count;
}

/* Simulate a device that accepts commands without reaching the producer gate.
   Destruction must stop the I/O worker without synchronizing that device. */
static unsigned destroy_sync_calls;
static int held_value(void *state, ShadowSpillBackendStream stream,
                      ShadowSpillBackendSignals signals, uint32_t index,
                      uint64_t value) {
    (void)state; (void)stream; (void)signals; (void)index; (void)value;
    return 0;
}
static int held_copy(void *state, void *destination, const void *source,
                     uint64_t bytes, ShadowSpillBackendStream stream) {
    (void)state; (void)destination; (void)source; (void)bytes; (void)stream;
    return 0;
}
static int forbidden_sync(void *state, ShadowSpillBackendStream stream) {
    (void)state; (void)stream;
    ++destroy_sync_calls;
    return -1;
}

int main(int argc, char **argv) {
    CHECK(argc == 4);
    void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (library == NULL) fprintf(stderr, "%s\n", dlerror());
    CHECK(library != NULL);
    union { void *symbol; const ShadowSpillLibraryDescription *(*call)(void); }
        describe = {.symbol = dlsym(library, "shadowspill_library_describe")};
    CHECK(describe.symbol != NULL);
    const ShadowSpillLibraryDescription *extension = describe.call();
    CHECK(extension->pool_memory_count == 1 && extension->lane_count == 2);
    const ShadowSpillPoolMemoryDescription *pool = extension->pool_memory;
    void *backend_library = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
    CHECK(backend_library != NULL);
    union { void *symbol; ShadowSpillBackendCreate call; } create = {
        .symbol = dlsym(backend_library, SHADOWSPILL_BACKEND_CREATE_SYMBOL)};
    union { void *symbol; ShadowSpillBackendDestroy call; } destroy = {
        .symbol = dlsym(backend_library, SHADOWSPILL_BACKEND_DESTROY_SYMBOL)};
    CHECK(create.symbol && destroy.symbol);
    ShadowSpillBackend backend = {0};
    const ShadowSpillBackendConfig device_config = {
        .abi_version = SHADOWSPILL_BACKEND_ABI_VERSION, .device_ordinal = 0,
    };
    CHECK(create.call(&device_config, &backend) == 0);
    char path[4096];
    CHECK(snprintf(path, sizeof(path), "%s/shadowspill-ssd-XXXXXX", argv[3]) > 0);
    CHECK(mkdtemp(path) != NULL);
    ShadowSpillSSDConfiguration config = {
        .directory = path, .staging_bytes = 1U << 20,
        .chunk_bytes = 65536, .queue_depth = 2,
    };
    const uint64_t capacity = (2U << 20) + 317;
    void *base = NULL, *state = NULL;
    CHECK(pool->acquire(&config, capacity, &base, &state) == 0);
    CHECK(entries(path) == 0);
    SSDRegion *region = state;
    const int fd = region->fd;
    CHECK((fcntl(fd, F_GETFL) & O_DIRECT) != 0);
    struct stat file;
    CHECK(fstat(fd, &file) == 0 && file.st_nlink == 0 && file.st_size >= (off_t)capacity);
    unsigned char *expected = NULL, *actual = NULL;
    const size_t host_bytes = ((size_t)capacity + 4095U) / 4096U * 4096U;
    CHECK(posix_memalign((void **)&expected, 4096, host_bytes) == 0);
    CHECK(posix_memalign((void **)&actual, 4096, host_bytes) == 0);
    for (uint64_t i = 0; i < capacity; ++i) expected[i] = (unsigned char)(i * 17U + i / 251U);
    CHECK(pool->write(state, 0, expected, capacity) == 0);
    CHECK(pool->read(state, 0, actual, capacity) == 0);
    CHECK(memcmp(expected, actual, (size_t)capacity) == 0);
    CHECK(pool->read(state, capacity + 1, actual, 0) != 0);
    CHECK(pool->write(state, capacity - 3, expected, 4) != 0);
    CHECK(pool->write(state, capacity, NULL, 0) == 0);

    void *device = NULL;
    CHECK(backend.allocate_device(backend.state, capacity, &device) == 0);
    CHECK(backend.register_host_memory(backend.state, expected, capacity) == 0);
    CHECK(backend.register_host_memory(backend.state, actual, capacity) == 0);
    ShadowSpillBackendStream stream[2] = {0};
    ShadowSpillLane *lanes[2] = {0};
    for (unsigned i = 0; i < 2; ++i) {
        CHECK(backend.create_stream(backend.state, &stream[i]) == 0);
        const ShadowSpillLaneDescription *description = &extension->lanes[i];
        ShadowSpillLane common = {
            .backend = &backend, .stream = stream[i],
            .from_kind = description->from_kind, .to_kind = description->to_kind,
            .from_range = {.address = i == 0 ? base : device, .bytes = capacity},
            .to_range = {.address = i == 0 ? device : base, .bytes = capacity},
        };
        CHECK(description->create(&common, NULL, &lanes[i]) == 0);
    }
    const ShadowSpillLaneOperations *fetch = extension->lanes[0].operations;
    const ShadowSpillLaneOperations *evict = extension->lanes[1].operations;
    uint64_t handle = 0;
    CHECK(fetch->copy(lanes[0], device, base, capacity, &handle) == 0);
    CHECK(fetch->synchronize(lanes[0]) == 0);
    CHECK(backend.copy_device_to_host(backend.state, actual, device, capacity, stream[0]) == 0);
    CHECK(backend.synchronize_stream(backend.state, stream[0]) == 0);
    CHECK(memcmp(expected, actual, (size_t)capacity) == 0);

    /* Cross sector/chunk edges and repeatedly recycle both work and staging
       rings. Adjacent one-byte objects must survive the sector read/modify/write. */
    for (unsigned iteration = 0; iteration < 301; ++iteration) {
        const uint64_t start = (uint64_t)iteration * 673 % 65536;
        const uint64_t bytes = iteration % 2 ? 1 : 65537;
        memset(expected + start, (int)(iteration % 256), (size_t)bytes);
        CHECK(backend.copy_host_to_device(backend.state, device, expected + start, bytes, stream[1]) == 0);
        CHECK(evict->copy(lanes[1], (char *)base + start, device, bytes, &handle) == 0);
        CHECK(evict->synchronize(lanes[1]) == 0);
    }
    CHECK(pool->read(state, 0, actual, capacity) == 0);
    CHECK(memcmp(expected, actual, (size_t)capacity) == 0);

    /* Completion events must include SSD I/O, rather than only host staging. */
    ShadowSpillBackendEvent complete;
    CHECK(backend.create_event(backend.state, &complete, 0) == 0);
    CHECK(fetch->copy(lanes[0], device, base, capacity, &handle) == 0);
    CHECK(fetch->signal(lanes[0], handle, complete) == 0);
    CHECK(backend.synchronize_event(backend.state, complete) == 0);
    CHECK(backend.copy_device_to_host(backend.state, actual, device, capacity, stream[0]) == 0);
    CHECK(backend.synchronize_stream(backend.state, stream[0]) == 0);
    CHECK(memcmp(expected, actual, (size_t)capacity) == 0);
    CHECK(backend.destroy_event(backend.state, complete) == 0);

    /* A fetch must not read SSD bytes before its dependency event. Change
       the source while the event is deliberately blocked: an early disk
       read would return the old pattern even if its GPU copy waits correctly. */
    ShadowSpillBackendSignals gate;
    uint64_t *gate_host;
    ShadowSpillBackendStream producer;
    ShadowSpillBackendEvent dependency;
    CHECK(backend.allocate_signals(backend.state, 1, &gate, &gate_host) == 0);
    CHECK(backend.create_stream(backend.state, &producer) == 0);
    CHECK(backend.create_event(backend.state, &dependency, 0) == 0);
    CHECK(backend.wait_value(backend.state, producer, gate, 0, 1) == 0);
    CHECK(backend.record_event(backend.state, dependency, producer) == 0);
    CHECK(fetch->wait(lanes[0], dependency) == 0);
    CHECK(fetch->copy(lanes[0], device, base, capacity, &handle) == 0);
    const struct timespec delay = {.tv_nsec = 20000000L};
    nanosleep(&delay, NULL);
    memset(expected, 73, (size_t)capacity);
    CHECK(pool->write(state, 0, expected, capacity) == 0);
    atomic_store_explicit((_Atomic uint64_t *)gate_host, 1, memory_order_release);
    CHECK(fetch->synchronize(lanes[0]) == 0);
    CHECK(backend.copy_device_to_host(backend.state, actual, device, capacity, stream[0]) == 0);
    CHECK(backend.synchronize_stream(backend.state, stream[0]) == 0);
    CHECK(memcmp(expected, actual, (size_t)capacity) == 0);

    /* Likewise an eviction must follow its GPU producer and must not publish
       completion merely when the device-to-staging copy has finished. */
    memset(expected, 91, (size_t)capacity);
    CHECK(backend.wait_value(backend.state, producer, gate, 0, 2) == 0);
    CHECK(backend.copy_host_to_device(backend.state, device, expected, capacity, producer) == 0);
    CHECK(backend.record_event(backend.state, dependency, producer) == 0);
    CHECK(evict->wait(lanes[1], dependency) == 0);
    CHECK(evict->copy(lanes[1], base, device, capacity, &handle) == 0);
    nanosleep(&delay, NULL);
    CHECK(pool->read(state, 0, actual, capacity) == 0);
    for (uint64_t i = 0; i < capacity; ++i) CHECK(actual[i] == 73);
    CHECK(backend.create_event(backend.state, &complete, 0) == 0);
    CHECK(evict->signal(lanes[1], handle, complete) == 0);
    atomic_store_explicit((_Atomic uint64_t *)gate_host, 2, memory_order_release);
    CHECK(backend.synchronize_event(backend.state, complete) == 0);
    CHECK(pool->read(state, 0, actual, capacity) == 0);
    CHECK(memcmp(expected, actual, (size_t)capacity) == 0);
    CHECK(backend.destroy_event(backend.state, complete) == 0);
    CHECK(backend.destroy_event(backend.state, dependency) == 0);
    CHECK(backend.destroy_stream(backend.state, producer) == 0);
    CHECK(backend.free_signals(backend.state, gate) == 0);

    ShadowSpillBackend held_backend = backend;
    held_backend.write_value = held_value;
    held_backend.wait_value = held_value;
    held_backend.copy_host_to_device = held_copy;
    held_backend.copy_device_to_host = held_copy;
    held_backend.synchronize_stream = forbidden_sync;
    for (unsigned i = 0; i < 2; ++i) {
        ShadowSpillLane held_base = *lanes[i], *held_lane = NULL;
        held_base.backend = &held_backend;
        const ShadowSpillLaneDescription *description = &extension->lanes[i];
        CHECK(description->create(&held_base, NULL, &held_lane) == 0);
        CHECK(description->operations->copy(held_lane,
            i == 0 ? device : base, i == 0 ? base : device, capacity, &handle) == 0);
        nanosleep(&delay, NULL);
        description->operations->destroy(held_lane);
    }
    CHECK(destroy_sync_calls == 0);

    /* A failed read must release stream waits and report failure, including
       when queued work exceeds the staging depth. Bound by CTest's timeout. */
    CHECK(ftruncate(fd, 0) == 0);
    CHECK(fetch->copy(lanes[0], device, base, capacity, &handle) == 0);
    CHECK(fetch->synchronize(lanes[0]) != 0);
    for (unsigned i = 0; i < 2; ++i) {
        extension->lanes[i].operations->destroy(lanes[i]);
        CHECK(backend.destroy_stream(backend.state, stream[i]) == 0);
    }
    CHECK(region->staging_reserved == config.chunk_bytes);
    CHECK(backend.unregister_host_memory(backend.state, actual, capacity) == 0);
    CHECK(backend.unregister_host_memory(backend.state, expected, capacity) == 0);
    CHECK(backend.free_device(backend.state, device, capacity) == 0);
    free(actual);
    free(expected);
    CHECK(pool->release(state, base, capacity) == 0);
    CHECK(fcntl(fd, F_GETFD) == -1);
    CHECK(entries(path) == 0 && rmdir(path) == 0);
    destroy.call(&backend);
    dlclose(backend_library);
    dlclose(library);
    puts("SSD pool/lanes: direct I/O, partial sectors, ring reuse, completion, failure, close passed");
    return 0;
}
