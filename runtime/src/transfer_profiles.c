#include "internal.h"

#include <pthread.h>
#include <stddef.h>
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
        ) != 0 || measure_copy(
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
    uint64_t bandwidth = 0U;
    uint64_t latency = small_nanoseconds;
    if (config->large_copy_bytes > config->small_copy_bytes &&
        large_nanoseconds > small_nanoseconds) {
        const uint64_t byte_delta =
            config->large_copy_bytes - config->small_copy_bytes;
        const uint64_t time_delta = large_nanoseconds - small_nanoseconds;
        if (byte_delta <= UINT64_MAX / 1000000000U) {
            bandwidth = byte_delta * 1000000000U / time_delta;
        } else {
            bandwidth = (byte_delta / time_delta) * 1000000000U;
        }
        if (bandwidth != 0U &&
            config->small_copy_bytes <= UINT64_MAX / 1000000000U) {
            const uint64_t payload =
                config->small_copy_bytes * 1000000000U / bandwidth;
            latency = small_nanoseconds > payload
                ? small_nanoseconds - payload
                : 0U;
        }
    }
    if (bandwidth == 0U) {
        bandwidth = config->large_copy_bytes <= UINT64_MAX / 1000000000U
            ? config->large_copy_bytes * 1000000000U /
                (large_nanoseconds == 0U ? 1U : large_nanoseconds)
            : config->large_copy_bytes /
                (large_nanoseconds == 0U ? 1U : large_nanoseconds) *
                1000000000U;
    }
    profile->latency_nanoseconds = latency;
    profile->bandwidth_bytes_per_second = bandwidth == 0U ? 1U : bandwidth;
    profile->small_copy_bytes = config->small_copy_bytes;
    profile->large_copy_bytes = config->large_copy_bytes;
    profile->measured_copies = config->measured_copies;
    profile->calibrated_timestamp_nanoseconds = monotonic_nanoseconds();
    profile->available = 1U;
    profile->calibrated = 1U;
    profile->provenance = config->provenance;
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
            .large_copy_bytes = 64U << 20U,
            .warmup_copies = 2U,
            .measured_copies = 5U,
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
