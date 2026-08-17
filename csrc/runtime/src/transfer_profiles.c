#include "internal.h"

#include <pthread.h>
#include <sched.h>
#include <stddef.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t monotonic_nanoseconds(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        return 0U;
    }
    return (uint64_t)value.tv_sec * 1000000000U + (uint64_t)value.tv_nsec;
}

static uint32_t profile_index(
    const ShadowSpillRuntime *runtime,
    uint32_t source_pool_id,
    uint32_t destination_pool_id
) {
    return source_pool_id * runtime->pool_count + destination_pool_id;
}

ShadowSpillTransferRoute *shadowspill_transfer_route(
    ShadowSpillRuntime *runtime,
    uint32_t source_pool_id,
    uint32_t destination_pool_id
) {
    if (runtime == NULL) {
        return NULL;
    }
    if (runtime->fetch_route.source_pool_id == source_pool_id &&
        runtime->fetch_route.destination_pool_id == destination_pool_id) {
        return &runtime->fetch_route;
    }
    if (runtime->evict_route.source_pool_id == source_pool_id &&
        runtime->evict_route.destination_pool_id == destination_pool_id) {
        return &runtime->evict_route;
    }
    return NULL;
}

ShadowSpillBackendStream *shadowspill_transfer_route_lane(
    ShadowSpillRuntime *runtime,
    const ShadowSpillTransferRoute *route
) {
    if (runtime == NULL || route == NULL) {
        return NULL;
    }
    if (route == &runtime->fetch_route) {
        return &runtime->fetch_stream;
    }
    if (route == &runtime->evict_route) {
        return &runtime->evict_stream;
    }
    return NULL;
}

int shadowspill_transfer_profiles_initialize(ShadowSpillRuntime *runtime) {
    if (runtime == NULL || runtime->pool_count == 0U ||
        runtime->pool_count > UINT32_MAX / runtime->pool_count) {
        return -1;
    }
    const uint32_t count = runtime->pool_count * runtime->pool_count;
    ShadowSpillTransferProfile *profiles = calloc(
        (size_t)count, sizeof(*profiles)
    );
    if (profiles == NULL || pthread_rwlock_init(
            &runtime->transfer_profiles_lock, NULL
        ) != 0) {
        free(profiles);
        return -1;
    }
    runtime->transfer_profiles = profiles;
    runtime->transfer_profile_count = count;
    runtime->transfer_profile_generation = 0U;
    runtime->transfer_profiles_initialized = 1U;
    for (uint32_t source = 0U; source < runtime->pool_count; ++source) {
        for (uint32_t destination = 0U;
             destination < runtime->pool_count; ++destination) {
            ShadowSpillTransferProfile *profile = &profiles[
                profile_index(runtime, source, destination)
            ];
            *profile = (ShadowSpillTransferProfile){
                .abi_version = SHADOWSPILL_TRANSFER_PROFILE_ABI_VERSION,
                .source_pool_id = source,
                .destination_pool_id = destination,
                .bandwidth_bytes_per_second = source == destination
                    ? UINT64_MAX
                    : 0U,
                .available = source == destination ||
                    shadowspill_transfer_route(
                        runtime, source, destination
                    ) != NULL,
                .calibrated = source == destination,
                .calibration_mode = source == destination
                    ? SHADOWSPILL_TRANSFER_CALIBRATION_IDENTITY
                    : SHADOWSPILL_TRANSFER_CALIBRATION_SOLO,
                .concurrent_route_count = source == destination ? 0U : 1U,
            };
        }
    }
    return 0;
}

void shadowspill_transfer_profiles_destroy(ShadowSpillRuntime *runtime) {
    if (runtime == NULL || !runtime->transfer_profiles_initialized) {
        return;
    }
    pthread_rwlock_destroy(&runtime->transfer_profiles_lock);
    free(runtime->transfer_profiles);
    runtime->transfer_profiles = NULL;
    runtime->transfer_profile_count = 0U;
    runtime->transfer_profile_generation = 0U;
    runtime->transfer_profiles_initialized = 0U;
}

static int route_selected(
    uint32_t source,
    uint32_t destination,
    const ShadowSpillTransferRouteKey *routes,
    uint32_t route_count
) {
    if (route_count == 0U) {
        return source != destination;
    }
    for (uint32_t index = 0U; index < route_count; ++index) {
        if (routes[index].source_pool_id == source &&
            routes[index].destination_pool_id == destination) {
            return 1;
        }
    }
    return 0;
}

static int reserve_probe_ranges(
    ShadowSpillMemoryPool *source,
    ShadowSpillMemoryPool *destination,
    uint64_t bytes,
    uint64_t *source_offset,
    uint64_t *destination_offset
) {
    ShadowSpillMemoryPool *first = source->pool_id < destination->pool_id
        ? source
        : destination;
    ShadowSpillMemoryPool *second = first == source ? destination : source;
    pthread_mutex_lock(&first->lock);
    pthread_mutex_lock(&second->lock);
    int status = shadowspill_memory_pool_reserve_locked(
        source,
        bytes,
        source->minimum_alignment,
        SHADOWSPILL_MEMORY_FIRST_FIT,
        source_offset
    );
    if (status == 0) {
        status = shadowspill_memory_pool_reserve_locked(
            destination,
            bytes,
            destination->minimum_alignment,
            SHADOWSPILL_MEMORY_FIRST_FIT,
            destination_offset
        );
        if (status != 0) {
            (void)shadowspill_memory_pool_release_locked(
                source, *source_offset, bytes
            );
        }
    }
    pthread_mutex_unlock(&second->lock);
    pthread_mutex_unlock(&first->lock);
    return status;
}

static void release_probe_ranges(
    ShadowSpillMemoryPool *source,
    ShadowSpillMemoryPool *destination,
    uint64_t bytes,
    uint64_t source_offset,
    uint64_t destination_offset
) {
    ShadowSpillMemoryPool *first = source->pool_id < destination->pool_id
        ? source
        : destination;
    ShadowSpillMemoryPool *second = first == source ? destination : source;
    pthread_mutex_lock(&first->lock);
    pthread_mutex_lock(&second->lock);
    (void)shadowspill_memory_pool_release_locked(
        destination, destination_offset, bytes
    );
    (void)shadowspill_memory_pool_release_locked(
        source, source_offset, bytes
    );
    pthread_mutex_unlock(&second->lock);
    pthread_mutex_unlock(&first->lock);
}

static int measure_copy(
    const ShadowSpillTransferRoute *route,
    ShadowSpillBackendStream lane,
    void *destination,
    const void *source,
    uint64_t bytes,
    uint32_t copies,
    uint64_t *average_nanoseconds
) {
    uint64_t total = 0U;
    for (uint32_t copy = 0U; copy < copies; ++copy) {
        const uint64_t begin = monotonic_nanoseconds();
        if (begin == 0U || route->copy_async(
                route->context, destination, source, bytes, lane
            ) != 0 || route->synchronize_lane(route->context, lane) != 0) {
            return -1;
        }
        const uint64_t end = monotonic_nanoseconds();
        if (end < begin || UINT64_MAX - total < end - begin) {
            return -1;
        }
        total += end - begin;
    }
    *average_nanoseconds = total / copies;
    return 0;
}

static int measure_copy_batch(
    const ShadowSpillTransferRoute *route,
    ShadowSpillBackendStream lane,
    void *destination,
    const void *source,
    uint64_t bytes,
    uint32_t copies,
    uint64_t *elapsed_nanoseconds
) {
    const uint64_t begin = monotonic_nanoseconds();
    if (begin == 0U) {
        return -1;
    }
    for (uint32_t copy = 0U; copy < copies; ++copy) {
        if (route->copy_async(
                route->context, destination, source, bytes, lane
            ) != 0) {
            return -1;
        }
    }
    if (route->synchronize_lane(route->context, lane) != 0) {
        return -1;
    }
    const uint64_t end = monotonic_nanoseconds();
    if (end <= begin) {
        return -1;
    }
    *elapsed_nanoseconds = end - begin;
    return 0;
}

static uint64_t measured_bandwidth(
    uint64_t bytes, uint32_t copies, uint64_t elapsed_nanoseconds
) {
    if (bytes == 0U || copies == 0U || elapsed_nanoseconds == 0U) {
        return 0U;
    }
    const uint64_t total_bytes = bytes <= UINT64_MAX / copies
        ? bytes * copies
        : UINT64_MAX;
    if (total_bytes <= UINT64_MAX / 1000000000U) {
        return total_bytes * 1000000000U / elapsed_nanoseconds;
    }
    return (total_bytes / elapsed_nanoseconds) * 1000000000U;
}

static int calibrate_route(
    ShadowSpillRuntime *runtime,
    ShadowSpillTransferRoute *route,
    const ShadowSpillTransferCalibrationConfig *config,
    ShadowSpillTransferProfile *profile
) {
    ShadowSpillMemoryPool *source = shadowspill_runtime_pool(
        runtime, route->source_pool_id
    );
    ShadowSpillMemoryPool *destination = shadowspill_runtime_pool(
        runtime, route->destination_pool_id
    );
    ShadowSpillBackendStream *lane = shadowspill_transfer_route_lane(
        runtime, route
    );
    uint64_t source_offset = 0U;
    uint64_t destination_offset = 0U;
    if (source == NULL || destination == NULL || lane == NULL ||
        reserve_probe_ranges(
            source,
            destination,
            config->large_copy_bytes,
            &source_offset,
            &destination_offset
        ) != 0) {
        return -1;
    }
    void *source_pointer = shadowspill_memory_pool_pointer(
        source, source_offset
    );
    void *destination_pointer = shadowspill_memory_pool_pointer(
        destination, destination_offset
    );
    int status = 0;
    for (uint32_t warmup = 0U; warmup < config->warmup_copies; ++warmup) {
        if (route->copy_async(
                route->context,
                destination_pointer,
                source_pointer,
                config->large_copy_bytes,
                *lane
            ) != 0 || route->synchronize_lane(route->context, *lane) != 0) {
            status = -1;
            break;
        }
    }
    uint64_t small_nanoseconds = 0U;
    uint64_t large_nanoseconds = 0U;
    if (status == 0 && (measure_copy(
            route,
            *lane,
            destination_pointer,
            source_pointer,
            config->small_copy_bytes,
            config->measured_copies,
            &small_nanoseconds
        ) != 0 || measure_copy_batch(
            route,
            *lane,
            destination_pointer,
            source_pointer,
            config->large_copy_bytes,
            config->measured_copies,
            &large_nanoseconds
        ) != 0)) {
        status = -1;
    }
    release_probe_ranges(
        source,
        destination,
        config->large_copy_bytes,
        source_offset,
        destination_offset
    );
    if (status != 0) {
        return status;
    }
    uint64_t bandwidth = measured_bandwidth(
        config->large_copy_bytes,
        config->measured_copies,
        large_nanoseconds
    );
    uint64_t latency = small_nanoseconds;
    if (bandwidth != 0U &&
        config->small_copy_bytes <= UINT64_MAX / 1000000000U) {
        const uint64_t payload =
            config->small_copy_bytes * 1000000000U / bandwidth;
        latency = small_nanoseconds > payload
            ? small_nanoseconds - payload
            : 0U;
    }
    profile->latency_nanoseconds = latency;
    profile->bandwidth_bytes_per_second = bandwidth == 0U ? 1U : bandwidth;
    profile->solo_bandwidth_bytes_per_second =
        profile->bandwidth_bytes_per_second;
    profile->concurrent_bandwidth_bytes_per_second = 0U;
    profile->solo_measurement_nanoseconds = large_nanoseconds;
    profile->concurrent_measurement_nanoseconds = 0U;
    profile->small_copy_bytes = config->small_copy_bytes;
    profile->large_copy_bytes = config->large_copy_bytes;
    profile->measured_copies = config->measured_copies;
    profile->calibrated_timestamp_nanoseconds = monotonic_nanoseconds();
    profile->available = 1U;
    profile->calibrated = 1U;
    profile->provenance = config->provenance;
    profile->calibration_mode = SHADOWSPILL_TRANSFER_CALIBRATION_SOLO;
    profile->concurrent_route_count = 1U;
    return 0;
}

typedef struct ShadowSpillCalibrationProbe {
    ShadowSpillTransferRoute *route;
    ShadowSpillBackendStream lane;
    ShadowSpillMemoryPool *source_pool;
    ShadowSpillMemoryPool *destination_pool;
    uint64_t bytes;
    uint64_t source_offset;
    uint64_t destination_offset;
    const void *source_pointer;
    void *destination_pointer;
} ShadowSpillCalibrationProbe;

static int prepare_probe(
    ShadowSpillRuntime *runtime,
    ShadowSpillTransferRoute *route,
    uint64_t bytes,
    ShadowSpillCalibrationProbe *probe
) {
    ShadowSpillMemoryPool *source = shadowspill_runtime_pool(
        runtime, route->source_pool_id
    );
    ShadowSpillMemoryPool *destination = shadowspill_runtime_pool(
        runtime, route->destination_pool_id
    );
    ShadowSpillBackendStream *lane = shadowspill_transfer_route_lane(
        runtime, route
    );
    if (source == NULL || destination == NULL || lane == NULL) {
        return -1;
    }
    uint64_t source_offset = 0U;
    uint64_t destination_offset = 0U;
    if (reserve_probe_ranges(
            source,
            destination,
            bytes,
            &source_offset,
            &destination_offset
        ) != 0) {
        return -1;
    }
    *probe = (ShadowSpillCalibrationProbe){
        .route = route,
        .lane = *lane,
        .source_pool = source,
        .destination_pool = destination,
        .bytes = bytes,
        .source_offset = source_offset,
        .destination_offset = destination_offset,
        .source_pointer = shadowspill_memory_pool_pointer(
            source, source_offset
        ),
        .destination_pointer = shadowspill_memory_pool_pointer(
            destination, destination_offset
        ),
    };
    return 0;
}

static void release_probe(ShadowSpillCalibrationProbe *probe) {
    release_probe_ranges(
        probe->source_pool,
        probe->destination_pool,
        probe->bytes,
        probe->source_offset,
        probe->destination_offset
    );
}

typedef struct ShadowSpillCalibrationGate {
    _Atomic uint32_t ready;
    _Atomic uint8_t start;
} ShadowSpillCalibrationGate;

typedef struct ShadowSpillCalibrationJob {
    ShadowSpillCalibrationProbe *probe;
    ShadowSpillCalibrationGate *gate;
    uint32_t copies;
    uint64_t elapsed_nanoseconds;
    int status;
} ShadowSpillCalibrationJob;

static void *run_calibration_job(void *context) {
    ShadowSpillCalibrationJob *job = context;
    atomic_fetch_add_explicit(&job->gate->ready, 1U, memory_order_release);
    while (!atomic_load_explicit(&job->gate->start, memory_order_acquire)) {
        sched_yield();
    }
    job->status = measure_copy_batch(
        job->probe->route,
        job->probe->lane,
        job->probe->destination_pointer,
        job->probe->source_pointer,
        job->probe->bytes,
        job->copies,
        &job->elapsed_nanoseconds
    );
    return NULL;
}

static int measure_concurrent_pair(
    ShadowSpillCalibrationProbe *first,
    ShadowSpillCalibrationProbe *second,
    uint32_t copies,
    uint64_t *first_nanoseconds,
    uint64_t *second_nanoseconds
) {
    ShadowSpillCalibrationGate gate = {0};
    ShadowSpillCalibrationJob jobs[2] = {
        {.probe = first, .gate = &gate, .copies = copies, .status = -1},
        {.probe = second, .gate = &gate, .copies = copies, .status = -1},
    };
    pthread_t threads[2];
    if (pthread_create(&threads[0], NULL, run_calibration_job, &jobs[0]) != 0) {
        return -1;
    }
    if (pthread_create(&threads[1], NULL, run_calibration_job, &jobs[1]) != 0) {
        atomic_store_explicit(&gate.start, 1U, memory_order_release);
        (void)pthread_join(threads[0], NULL);
        return -1;
    }
    while (atomic_load_explicit(&gate.ready, memory_order_acquire) != 2U) {
        sched_yield();
    }
    atomic_store_explicit(&gate.start, 1U, memory_order_release);
    const int first_join = pthread_join(threads[0], NULL);
    const int second_join = pthread_join(threads[1], NULL);
    if (first_join != 0 || second_join != 0 ||
        jobs[0].status != 0 || jobs[1].status != 0) {
        return -1;
    }
    *first_nanoseconds = jobs[0].elapsed_nanoseconds;
    *second_nanoseconds = jobs[1].elapsed_nanoseconds;
    return 0;
}

static int calibrate_reverse_pair(
    ShadowSpillRuntime *runtime,
    ShadowSpillTransferRoute *first_route,
    ShadowSpillTransferRoute *second_route,
    const ShadowSpillTransferCalibrationConfig *config,
    ShadowSpillTransferProfile *first_profile,
    ShadowSpillTransferProfile *second_profile
) {
    ShadowSpillCalibrationProbe first = {0};
    ShadowSpillCalibrationProbe second = {0};
    if (prepare_probe(
            runtime, first_route, config->large_copy_bytes, &first
        ) != 0) {
        return -1;
    }
    if (prepare_probe(
            runtime, second_route, config->large_copy_bytes, &second
        ) != 0) {
        release_probe(&first);
        return -1;
    }
    uint64_t first_nanoseconds = 0U;
    uint64_t second_nanoseconds = 0U;
    int status = 0;
    if (config->warmup_copies != 0U && measure_concurrent_pair(
            &first,
            &second,
            config->warmup_copies,
            &first_nanoseconds,
            &second_nanoseconds
        ) != 0) {
        status = -1;
    }
    if (status == 0 && measure_concurrent_pair(
            &first,
            &second,
            config->measured_copies,
            &first_nanoseconds,
            &second_nanoseconds
        ) != 0) {
        status = -1;
    }
    release_probe(&second);
    release_probe(&first);
    if (status != 0) {
        return status;
    }
    ShadowSpillTransferProfile *profiles[2] = {
        first_profile, second_profile
    };
    const uint64_t measurements[2] = {
        first_nanoseconds, second_nanoseconds
    };
    for (uint32_t index = 0U; index < 2U; ++index) {
        const uint64_t bandwidth = measured_bandwidth(
            config->large_copy_bytes,
            config->measured_copies,
            measurements[index]
        );
        profiles[index]->concurrent_bandwidth_bytes_per_second =
            bandwidth == 0U ? 1U : bandwidth;
        profiles[index]->bandwidth_bytes_per_second =
            profiles[index]->concurrent_bandwidth_bytes_per_second;
        profiles[index]->concurrent_measurement_nanoseconds =
            measurements[index];
        profiles[index]->calibration_mode =
            SHADOWSPILL_TRANSFER_CALIBRATION_BIDIRECTIONAL;
        profiles[index]->concurrent_route_count = 2U;
        profiles[index]->calibrated_timestamp_nanoseconds =
            monotonic_nanoseconds();
    }
    return 0;
}

ShadowSpillRuntimeStatus shadowspill_runtime_calibrate_transfer_capabilities(
    ShadowSpillRuntime *runtime,
    const ShadowSpillTransferCalibrationConfig *provided_config,
    const ShadowSpillTransferRouteKey *routes,
    uint32_t route_count
) {
    if (runtime == NULL || (route_count != 0U && routes == NULL)) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    ShadowSpillTransferCalibrationConfig config = provided_config == NULL
        ? (ShadowSpillTransferCalibrationConfig){
            .abi_version = SHADOWSPILL_TRANSFER_PROFILE_ABI_VERSION,
            .small_copy_bytes = 4096U,
            .large_copy_bytes = 256U << 20U,
            .warmup_copies = 4U,
            .measured_copies = 16U,
            .provenance = SHADOWSPILL_TRANSFER_PROFILE_RECALIBRATION,
        }
        : *provided_config;
    if (config.abi_version != SHADOWSPILL_TRANSFER_PROFILE_ABI_VERSION ||
        config.small_copy_bytes == 0U ||
        config.large_copy_bytes < config.small_copy_bytes ||
        config.measured_copies == 0U ||
        config.provenance > SHADOWSPILL_TRANSFER_PROFILE_RECALIBRATION) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    for (uint32_t index = 0U; index < route_count; ++index) {
        if (routes[index].source_pool_id >= runtime->pool_count ||
            routes[index].destination_pool_id >= runtime->pool_count ||
            routes[index].source_pool_id == routes[index].destination_pool_id ||
            shadowspill_transfer_route(
                runtime,
                routes[index].source_pool_id,
                routes[index].destination_pool_id
            ) == NULL) {
            return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
        }
    }
    ShadowSpillRuntimeStatus idle = shadowspill_runtime_wait_idle(runtime);
    if (idle != SHADOWSPILL_RUNTIME_OK) {
        return idle;
    }
    ShadowSpillTransferProfile *next = malloc(
        (size_t)runtime->transfer_profile_count * sizeof(*next)
    );
    if (next == NULL) {
        return SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE;
    }
    pthread_rwlock_rdlock(&runtime->transfer_profiles_lock);
    memcpy(
        next,
        runtime->transfer_profiles,
        (size_t)runtime->transfer_profile_count * sizeof(*next)
    );
    pthread_rwlock_unlock(&runtime->transfer_profiles_lock);
    for (uint32_t source = 0U; source < runtime->pool_count; ++source) {
        for (uint32_t destination = 0U;
             destination < runtime->pool_count; ++destination) {
            if (!route_selected(
                    source, destination, routes, route_count
                )) {
                continue;
            }
            ShadowSpillTransferRoute *route = shadowspill_transfer_route(
                runtime, source, destination
            );
            if (route == NULL || calibrate_route(
                    runtime,
                    route,
                    &config,
                    &next[profile_index(runtime, source, destination)]
                ) != 0) {
                free(next);
                return SHADOWSPILL_RUNTIME_BACKEND_FAILURE;
            }
        }
    }
    for (uint32_t source = 0U; source < runtime->pool_count; ++source) {
        for (uint32_t destination = source + 1U;
             destination < runtime->pool_count; ++destination) {
            if (!route_selected(
                    source, destination, routes, route_count
                ) || !route_selected(
                    destination, source, routes, route_count
                )) {
                continue;
            }
            ShadowSpillTransferRoute *forward = shadowspill_transfer_route(
                runtime, source, destination
            );
            ShadowSpillTransferRoute *reverse = shadowspill_transfer_route(
                runtime, destination, source
            );
            if (forward != NULL && reverse != NULL && calibrate_reverse_pair(
                    runtime,
                    forward,
                    reverse,
                    &config,
                    &next[profile_index(runtime, source, destination)],
                    &next[profile_index(runtime, destination, source)]
                ) != 0) {
                free(next);
                return SHADOWSPILL_RUNTIME_BACKEND_FAILURE;
            }
        }
    }
    pthread_rwlock_wrlock(&runtime->transfer_profiles_lock);
    const uint64_t generation = ++runtime->transfer_profile_generation;
    for (uint32_t index = 0U;
         index < runtime->transfer_profile_count; ++index) {
        next[index].generation = generation;
    }
    ShadowSpillTransferProfile *previous = runtime->transfer_profiles;
    runtime->transfer_profiles = next;
    pthread_rwlock_unlock(&runtime->transfer_profiles_lock);
    free(previous);
    return SHADOWSPILL_RUNTIME_OK;
}

ShadowSpillRuntimeStatus shadowspill_runtime_transfer_profiles(
    ShadowSpillRuntime *runtime,
    ShadowSpillTransferProfile *profiles,
    uint32_t capacity,
    uint32_t *count,
    uint64_t *generation
) {
    if (runtime == NULL || count == NULL || generation == NULL ||
        (profiles == NULL && capacity != 0U)) {
        return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
    }
    pthread_rwlock_rdlock(&runtime->transfer_profiles_lock);
    *count = runtime->transfer_profile_count;
    *generation = runtime->transfer_profile_generation;
    if (profiles != NULL) {
        if (capacity < runtime->transfer_profile_count) {
            pthread_rwlock_unlock(&runtime->transfer_profiles_lock);
            return SHADOWSPILL_RUNTIME_INVALID_ARGUMENT;
        }
        memcpy(
            profiles,
            runtime->transfer_profiles,
            (size_t)runtime->transfer_profile_count * sizeof(*profiles)
        );
    }
    pthread_rwlock_unlock(&runtime->transfer_profiles_lock);
    return SHADOWSPILL_RUNTIME_OK;
}
