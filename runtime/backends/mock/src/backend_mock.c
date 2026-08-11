#define _POSIX_C_SOURCE 200809L

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
    ShadowSpillMockBackendStatistics statistics;
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
    uint64_t operation = ++backend->statistics.operation_count;
    int fails = backend->fail_operation != 0U &&
        operation == backend->fail_operation;
    pthread_mutex_unlock(&backend->mutex);
    return fails;
}

static MockStream *stream_pointer(ShadowSpillBackendStream stream) {
    return (MockStream *)stream.words[0];
}

static MockEvent *event_pointer(ShadowSpillBackendEvent event) {
    return (MockEvent *)event.words[0];
}

static int allocate_device(void *context, uint64_t bytes, void **pointer) {
    ShadowSpillMockBackend *backend = context;
    if (pointer == NULL || operation_fails(backend)) {
        return -1;
    }
    *pointer = malloc(bytes == 0U ? 1U : (size_t)bytes);
    if (*pointer == NULL) {
        return -1;
    }
    pthread_mutex_lock(&backend->mutex);
    ++backend->statistics.device_allocations;
    pthread_mutex_unlock(&backend->mutex);
    return 0;
}

static int free_device(void *context, void *pointer) {
    ShadowSpillMockBackend *backend = context;
    if (operation_fails(backend)) {
        return -1;
    }
    free(pointer);
    return 0;
}

static int allocate_host(void *context, uint64_t bytes, void **pointer) {
    ShadowSpillMockBackend *backend = context;
    if (pointer == NULL || operation_fails(backend)) {
        return -1;
    }
    *pointer = malloc(bytes == 0U ? 1U : (size_t)bytes);
    if (*pointer == NULL) {
        return -1;
    }
    pthread_mutex_lock(&backend->mutex);
    ++backend->statistics.host_allocations;
    pthread_mutex_unlock(&backend->mutex);
    return 0;
}

static int free_host(void *context, void *pointer) {
    return free_device(context, pointer);
}

static int create_stream(
    void *context,
    ShadowSpillTransferKind kind,
    ShadowSpillBackendStream *stream
) {
    (void)kind;
    ShadowSpillMockBackend *backend = context;
    if (stream == NULL || operation_fails(backend)) {
        return -1;
    }
    MockStream *created = calloc(1U, sizeof(*created));
    if (created == NULL) {
        return -1;
    }
    stream->words[0] = (uintptr_t)created;
    stream->words[1] = 0U;
    return 0;
}

static int destroy_stream(void *context, ShadowSpillBackendStream stream) {
    ShadowSpillMockBackend *backend = context;
    if (operation_fails(backend)) {
        return -1;
    }
    free(stream_pointer(stream));
    return 0;
}

static int create_event(void *context, ShadowSpillBackendEvent *event) {
    ShadowSpillMockBackend *backend = context;
    if (event == NULL || operation_fails(backend)) {
        return -1;
    }
    MockEvent *created = calloc(1U, sizeof(*created));
    if (created == NULL) {
        return -1;
    }
    event->words[0] = (uintptr_t)created;
    event->words[1] = 0U;
    return 0;
}

static int destroy_event(void *context, ShadowSpillBackendEvent event) {
    ShadowSpillMockBackend *backend = context;
    if (operation_fails(backend)) {
        return -1;
    }
    free(event_pointer(event));
    return 0;
}

static int record_event(
    void *context,
    ShadowSpillBackendEvent event,
    ShadowSpillBackendStream stream
) {
    ShadowSpillMockBackend *backend = context;
    if (operation_fails(backend)) {
        return -1;
    }
    MockEvent *target = event_pointer(event);
    MockStream *source = stream_pointer(stream);
    if (target == NULL || source == NULL) {
        return -1;
    }
    pthread_mutex_lock(&backend->mutex);
    uint64_t now = now_nanoseconds();
    uint64_t ready = source->ready_nanoseconds > now
        ? source->ready_nanoseconds
        : now;
    target->ready_nanoseconds = ready + backend->config.event_delay_nanoseconds;
    target->recorded = 1;
    pthread_mutex_unlock(&backend->mutex);
    return 0;
}

static int query_event(
    void *context,
    ShadowSpillBackendEvent event,
    int *complete
) {
    ShadowSpillMockBackend *backend = context;
    if (complete == NULL || operation_fails(backend)) {
        return -1;
    }
    MockEvent *target = event_pointer(event);
    if (target == NULL) {
        return -1;
    }
    pthread_mutex_lock(&backend->mutex);
    if (!target->recorded) {
        pthread_mutex_unlock(&backend->mutex);
        return -1;
    }
    *complete = now_nanoseconds() >= target->ready_nanoseconds;
    pthread_mutex_unlock(&backend->mutex);
    return 0;
}

static int wait_event(
    void *context,
    ShadowSpillBackendStream stream,
    ShadowSpillBackendEvent event
) {
    ShadowSpillMockBackend *backend = context;
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

static int copy_async(
    void *context,
    void *destination,
    const void *source,
    uint64_t bytes,
    ShadowSpillTransferKind kind,
    ShadowSpillBackendStream stream
) {
    ShadowSpillMockBackend *backend = context;
    if ((bytes != 0U && (destination == NULL || source == NULL)) ||
        operation_fails(backend)) {
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
    uint64_t now = now_nanoseconds();
    if (target->ready_nanoseconds < now) {
        target->ready_nanoseconds = now;
    }
    target->ready_nanoseconds += kind == SHADOWSPILL_TRANSFER_TO_DEVICE
        ? backend->config.h2d_delay_nanoseconds
        : backend->config.d2h_delay_nanoseconds;
    if (kind == SHADOWSPILL_TRANSFER_TO_DEVICE) {
        ++backend->statistics.copies_to_device;
    } else {
        ++backend->statistics.copies_to_host;
    }
    pthread_mutex_unlock(&backend->mutex);
    return 0;
}

static int synchronize_stream(
    void *context,
    ShadowSpillBackendStream stream
) {
    ShadowSpillMockBackend *backend = context;
    if (operation_fails(backend)) {
        return -1;
    }
    MockStream *target = stream_pointer(stream);
    if (target == NULL) {
        return -1;
    }
    for (;;) {
        pthread_mutex_lock(&backend->mutex);
        uint64_t ready = target->ready_nanoseconds;
        pthread_mutex_unlock(&backend->mutex);
        if (now_nanoseconds() >= ready) {
            break;
        }
        struct timespec delay = {.tv_nsec = 100000U};
        (void)nanosleep(&delay, NULL);
    }
    pthread_mutex_lock(&backend->mutex);
    ++backend->statistics.stream_synchronizations;
    pthread_mutex_unlock(&backend->mutex);
    return 0;
}

int shadowspill_mock_backend_create(
    const ShadowSpillMockBackendConfig *config,
    ShadowSpillMockBackend **output
) {
    if (config == NULL || output == NULL ||
        config->abi_version != SHADOWSPILL_BACKEND_ABI_VERSION) {
        return -1;
    }
    *output = calloc(1U, sizeof(**output));
    if (*output == NULL) {
        return -1;
    }
    (*output)->config = *config;
    if (pthread_mutex_init(&(*output)->mutex, NULL) != 0) {
        free(*output);
        *output = NULL;
        return -1;
    }
    return 0;
}

void shadowspill_mock_backend_destroy(ShadowSpillMockBackend *backend) {
    if (backend == NULL) {
        return;
    }
    pthread_mutex_destroy(&backend->mutex);
    free(backend);
}

ShadowSpillBackend shadowspill_mock_backend_vtable(
    ShadowSpillMockBackend *backend
) {
    return (ShadowSpillBackend){
        .abi_version = SHADOWSPILL_BACKEND_ABI_VERSION,
        .context = backend,
        .allocate_device = allocate_device,
        .free_device = free_device,
        .allocate_host = allocate_host,
        .free_host = free_host,
        .create_stream = create_stream,
        .destroy_stream = destroy_stream,
        .create_event = create_event,
        .destroy_event = destroy_event,
        .record_event = record_event,
        .query_event = query_event,
        .wait_event = wait_event,
        .copy_async = copy_async,
        .synchronize_stream = synchronize_stream,
    };
}

int shadowspill_mock_create_compute_stream(
    ShadowSpillMockBackend *backend,
    ShadowSpillBackendStream *stream
) {
    return create_stream(backend, SHADOWSPILL_TRANSFER_TO_DEVICE, stream);
}

int shadowspill_mock_destroy_compute_stream(
    ShadowSpillMockBackend *backend,
    ShadowSpillBackendStream stream
) {
    return destroy_stream(backend, stream);
}

int shadowspill_mock_enqueue_compute(
    ShadowSpillMockBackend *backend,
    ShadowSpillBackendStream stream,
    uint64_t duration_nanoseconds
) {
    if (operation_fails(backend)) {
        return -1;
    }
    MockStream *target = stream_pointer(stream);
    if (target == NULL) {
        return -1;
    }
    pthread_mutex_lock(&backend->mutex);
    uint64_t now = now_nanoseconds();
    if (target->ready_nanoseconds < now) {
        target->ready_nanoseconds = now;
    }
    target->ready_nanoseconds += duration_nanoseconds;
    pthread_mutex_unlock(&backend->mutex);
    return 0;
}

void shadowspill_mock_fail_operation(
    ShadowSpillMockBackend *backend,
    uint64_t operation_number
) {
    pthread_mutex_lock(&backend->mutex);
    backend->fail_operation = operation_number;
    pthread_mutex_unlock(&backend->mutex);
}

void shadowspill_mock_fail_next_operation(ShadowSpillMockBackend *backend) {
    pthread_mutex_lock(&backend->mutex);
    backend->fail_operation = backend->statistics.operation_count + 1U;
    pthread_mutex_unlock(&backend->mutex);
}

void shadowspill_mock_backend_statistics(
    ShadowSpillMockBackend *backend,
    ShadowSpillMockBackendStatistics *statistics
) {
    pthread_mutex_lock(&backend->mutex);
    *statistics = backend->statistics;
    pthread_mutex_unlock(&backend->mutex);
}
