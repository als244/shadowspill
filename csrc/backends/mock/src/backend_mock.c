#include <shadowspill/backend_mock.h>

#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct MockStream {
    uint64_t ready_nanoseconds;
} MockStream;

typedef struct MockEvent {
    uint64_t ready_nanoseconds;
    int recorded;
} MockEvent;

struct ShadowSpillMockBackend {
    pthread_mutex_t mutex;
    ShadowSpillMockBackendConfig config;
    ShadowSpillBackendStatistics statistics;
    uint64_t operation_count;
    uint64_t fail_operation;
};

static uint64_t now_nanoseconds(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        return 0U;
    }
    return (uint64_t)value.tv_sec * 1000000000U + (uint64_t)value.tv_nsec;
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
    return (MockStream *)stream.words[0];
}

static MockEvent *event_pointer(ShadowSpillBackendEvent event) {
    return (MockEvent *)event.words[0];
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
    *stream = (ShadowSpillBackendStream){.words = {(uintptr_t)created, 0U}};
    count(backend, &backend->statistics.streams_created, 1U);
    return 0;
}

static int destroy_stream(void *state, ShadowSpillBackendStream stream) {
    ShadowSpillMockBackend *backend = state;
    if (operation_fails(backend)) {
        return -1;
    }
    free(stream_pointer(stream));
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
        const uint64_t ready = target->ready_nanoseconds;
        pthread_mutex_unlock(&backend->mutex);
        if (now_nanoseconds() >= ready) {
            break;
        }
        struct timespec delay = {.tv_nsec = 100000U};
        (void)nanosleep(&delay, NULL);
    }
    count(backend, &backend->statistics.stream_synchronizations, 1U);
    return 0;
}

static ShadowSpillBackendStream wrap_stream(
    void *state, uint64_t framework_stream_handle
) {
    (void)state;
    return (ShadowSpillBackendStream){
        .words = {(uintptr_t)framework_stream_handle, 0U},
    };
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
    pthread_mutex_lock(&backend->mutex);
    if (bytes != 0U) {
        memcpy(destination, source, (size_t)bytes);
    }
    const uint64_t now = now_nanoseconds();
    if (target->ready_nanoseconds < now) {
        target->ready_nanoseconds = now;
    }
    target->ready_nanoseconds += delay_nanoseconds;
    ++*copies;
    *copied_bytes += bytes;
    pthread_mutex_unlock(&backend->mutex);
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
    *event = (ShadowSpillBackendEvent){.words = {(uintptr_t)created, 0U}};
    count(backend, &backend->statistics.events_created, 1U);
    return 0;
}

static int destroy_event(void *state, ShadowSpillBackendEvent event) {
    ShadowSpillMockBackend *backend = state;
    if (operation_fails(backend)) {
        return -1;
    }
    free(event_pointer(event));
    count(backend, &backend->statistics.events_destroyed, 1U);
    return 0;
}

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
    pthread_mutex_lock(&backend->mutex);
    const uint64_t now = now_nanoseconds();
    const uint64_t ready =
        source->ready_nanoseconds > now ? source->ready_nanoseconds : now;
    target->ready_nanoseconds = ready + backend->config.event_delay_nanoseconds;
    target->recorded = 1;
    pthread_mutex_unlock(&backend->mutex);
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
    if (!target->recorded) {
        pthread_mutex_unlock(&backend->mutex);
        return -1;
    }
    *complete = now_nanoseconds() >= target->ready_nanoseconds;
    pthread_mutex_unlock(&backend->mutex);
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
    pthread_mutex_lock(&backend->mutex);
    if (!source->recorded) {
        pthread_mutex_unlock(&backend->mutex);
        return -1;
    }
    if (source->ready_nanoseconds > target->ready_nanoseconds) {
        target->ready_nanoseconds = source->ready_nanoseconds;
    }
    ++backend->statistics.stream_waits;
    pthread_mutex_unlock(&backend->mutex);
    return 0;
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
    if (!origin->recorded || !target->recorded) {
        pthread_mutex_unlock(&backend->mutex);
        return -1;
    }
    const uint64_t now = now_nanoseconds();
    if (now < origin->ready_nanoseconds || now < target->ready_nanoseconds) {
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
        .create_stream = create_stream,
        .destroy_stream = destroy_stream,
        .synchronize_stream = synchronize_stream,
        .wrap_stream = wrap_stream,
        .copy_host_to_device = copy_host_to_device,
        .copy_device_to_host = copy_device_to_host,
        .copy_device_to_device = copy_device_to_device,
        .create_event = create_event,
        .destroy_event = destroy_event,
        .record_event = record_event,
        .query_event = query_event,
        .wait_event = wait_event,
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
    pthread_mutex_lock(&mock->mutex);
    const uint64_t now = now_nanoseconds();
    if (target->ready_nanoseconds < now) {
        target->ready_nanoseconds = now;
    }
    target->ready_nanoseconds += duration_nanoseconds;
    pthread_mutex_unlock(&mock->mutex);
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
