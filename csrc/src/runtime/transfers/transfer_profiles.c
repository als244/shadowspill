#include "../internal.h"
#include "../../common/platform.h"

#include <pthread.h>
#include <stddef.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define SHADOWSPILL_CALIBRATION_BATCH_SAMPLES 3U

static uint64_t median_batch_nanoseconds(const uint64_t values[3]) {
    if (values[0] < values[1]) {
        if (values[1] < values[2]) {
            return values[1];
        }
        return values[0] < values[2] ? values[2] : values[0];
    }
    if (values[0] < values[2]) {
        return values[0];
    }
    return values[1] < values[2] ? values[2] : values[1];
}

static uint32_t profile_index(
    const ShadowSpillRuntime *runtime,
    uint32_t source_pool_id,
    uint32_t destination_pool_id
) {
    return source_pool_id * runtime->pool_count + destination_pool_id;
}

static ShadowSpillRouteState *route_between(
    ShadowSpillRuntime *runtime,
    uint32_t source_pool_id,
    uint32_t destination_pool_id
) {
    if (runtime == NULL) {
        return NULL;
    }
    for (uint32_t route_id = 0U; route_id < runtime->route_count; ++route_id) {
        ShadowSpillRouteState *route = &runtime->routes[route_id];
        if (route->source_pool_id == source_pool_id &&
            route->destination_pool_id == destination_pool_id) {
            return route;
        }
    }
    return NULL;
}

static ShadowSpillBackendStream *stream_of(
    ShadowSpillRuntime *runtime,
    const ShadowSpillRouteState *route
) {
    if (runtime == NULL || route == NULL) {
        return NULL;
    }
    for (uint32_t route_id = 0U; route_id < runtime->route_count; ++route_id) {
        if (route == &runtime->routes[route_id]) {
            return &runtime->routes[route_id].stream;
        }
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
                .abi_version = SHADOWSPILL_ABI_VERSION,
                .source_pool_id = source,
                .destination_pool_id = destination,
                .bandwidth_bytes_per_second = source == destination
                    ? UINT64_MAX
                    : 0U,
                .available = source == destination ||
                    route_between(
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

/* How many per-copy samples the typical is taken from. More than this and the
   extra copies are still timed, they simply do not widen the window the median
   is read out of. */
#define SHADOWSPILL_CALIBRATION_LATENCY_SAMPLES 64U

static int compare_nanoseconds(const void *first, const void *second) {
    const uint64_t left = *(const uint64_t *)first;
    const uint64_t right = *(const uint64_t *)second;
    return left < right ? -1 : (left > right ? 1 : 0);
}

/*
 * What one copy costs, start to finish, as the **typical** of its samples.
 *
 * The median rather than the mean, and no subtraction, because this is a
 * latency measurement and that is how latency is measured: `ib_write_lat` and
 * `ib_read_lat` ping-pong a two-byte message and report a typical, a minimum
 * and percentiles, never a mean and never a figure derived from a second
 * measurement.
 *
 * The mean was wrong twice over. One descheduled copy in sixteen moves it by
 * more than the quantity being measured -- these samples routinely span
 * threefold -- and what it fed was then *reduced by a payload term computed
 * from the bandwidth probe*, coupling two measurements so that shrinking the
 * bandwidth probe drove the reported latency to zero. It is what a small
 * transfer costs; nothing is deducted from it.
 */
static int measure_copy(
    const ShadowSpillRouteState *route,
    void *destination,
    const void *source,
    uint64_t bytes,
    uint32_t copies,
    uint64_t *typical_nanoseconds
) {
    uint64_t samples[SHADOWSPILL_CALIBRATION_LATENCY_SAMPLES];
    uint32_t kept = 0U;
    /* Calibration runs with no trace, so every lane keeps nothing and hands
       back handle 0. Named once here rather than at four call sites. */
    uint64_t ignored = 0U;
    for (uint32_t copy = 0U; copy < copies; ++copy) {
        const uint64_t begin = shadowspill_monotonic_ns();
        if (begin == 0U ||
            route->operations->copy(
                route->lane, destination, source, bytes, &ignored
            ) != 0 ||
            route->operations->synchronize(route->lane) != 0) {
            return -1;
        }
        const uint64_t end = shadowspill_monotonic_ns();
        if (end < begin) {
            return -1;
        }
        if (kept < SHADOWSPILL_CALIBRATION_LATENCY_SAMPLES) {
            samples[kept++] = end - begin;
        }
    }
    if (kept == 0U) {
        return -1;
    }
    qsort(samples, kept, sizeof(samples[0]), compare_nanoseconds);
    *typical_nanoseconds = samples[kept / 2U];
    return 0;
}

static int measure_copy_batch(
    const ShadowSpillRouteState *route,
    void *destination,
    const void *source,
    uint64_t bytes,
    uint32_t copies,
    uint64_t *elapsed_nanoseconds
) {
    const uint64_t begin = shadowspill_monotonic_ns();
    if (begin == 0U) {
        return -1;
    }
    uint64_t ignored = 0U;
    for (uint32_t copy = 0U; copy < copies; ++copy) {
        if (route->operations->copy(
                route->lane, destination, source, bytes, &ignored
            ) != 0) {
            return -1;
        }
    }
    if (route->operations->synchronize(route->lane) != 0) {
        return -1;
    }
    const uint64_t end = shadowspill_monotonic_ns();
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
    ShadowSpillRouteState *route,
    const ShadowSpillTransferCalibrationConfig *config,
    ShadowSpillTransferProfile *profile
) {
    ShadowSpillMemoryPool *source = shadowspill_runtime_pool(
        runtime, route->source_pool_id
    );
    ShadowSpillMemoryPool *destination = shadowspill_runtime_pool(
        runtime, route->destination_pool_id
    );
    ShadowSpillBackendStream *stream = stream_of(
        runtime, route
    );
    uint64_t source_offset = 0U;
    uint64_t destination_offset = 0U;
    if (source == NULL || destination == NULL || stream == NULL ||
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
    uint64_t ignored = 0U;
    for (uint32_t warmup = 0U; warmup < config->warmup_copies; ++warmup) {
        if (route->operations->copy(
                route->lane, destination_pointer, source_pointer,
                config->large_copy_bytes, &ignored
            ) != 0 ||
            route->operations->synchronize(route->lane) != 0) {
            status = -1;
            break;
        }
    }
    uint64_t small_nanoseconds = 0U;
    uint64_t large_measurements[SHADOWSPILL_CALIBRATION_BATCH_SAMPLES] = {0};
    if (status == 0 && measure_copy(
            route,
            destination_pointer,
            source_pointer,
            config->small_copy_bytes,
            config->measured_copies,
            &small_nanoseconds
        ) != 0) {
        status = -1;
    }
    for (uint32_t sample = 0U;
         status == 0 && sample < SHADOWSPILL_CALIBRATION_BATCH_SAMPLES;
         ++sample) {
        if (measure_copy_batch(
                route,
                destination_pointer,
                source_pointer,
                config->large_copy_bytes,
                config->measured_copies,
                &large_measurements[sample]
            ) != 0) {
            status = -1;
        }
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
    const uint64_t large_nanoseconds = median_batch_nanoseconds(
        large_measurements
    );
    uint64_t bandwidth = measured_bandwidth(
        config->large_copy_bytes,
        config->measured_copies,
        large_nanoseconds
    );
    /* What the small copy cost, not what is left of it after a payload term
       computed from a different measurement. See `measure_copy`. */
    profile->latency_nanoseconds = small_nanoseconds;
    profile->bandwidth_bytes_per_second = bandwidth == 0U ? 1U : bandwidth;
    profile->solo_bandwidth_bytes_per_second =
        profile->bandwidth_bytes_per_second;
    profile->concurrent_bandwidth_bytes_per_second = 0U;
    profile->solo_measurement_nanoseconds = large_nanoseconds;
    profile->concurrent_measurement_nanoseconds = 0U;
    profile->small_copy_bytes = config->small_copy_bytes;
    profile->large_copy_bytes = config->large_copy_bytes;
    profile->measured_copies = config->measured_copies;
    profile->calibrated_timestamp_nanoseconds = shadowspill_monotonic_ns();
    profile->available = 1U;
    profile->calibrated = 1U;
    profile->provenance = config->provenance;
    profile->calibration_mode = SHADOWSPILL_TRANSFER_CALIBRATION_SOLO;
    profile->concurrent_route_count = 1U;
    return 0;
}

typedef struct ShadowSpillCalibrationProbe {
    ShadowSpillRouteState *route;
    ShadowSpillBackendStream stream;
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
    ShadowSpillRouteState *route,
    uint64_t bytes,
    ShadowSpillCalibrationProbe *probe
) {
    ShadowSpillMemoryPool *source = shadowspill_runtime_pool(
        runtime, route->source_pool_id
    );
    ShadowSpillMemoryPool *destination = shadowspill_runtime_pool(
        runtime, route->destination_pool_id
    );
    ShadowSpillBackendStream *stream = stream_of(
        runtime, route
    );
    if (source == NULL || destination == NULL || stream == NULL) {
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
        .stream = *stream,
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

/*
 * Both routes at once, measured from one thread.
 *
 * Each lane is asked to signal an event once everything issued on it has
 * landed, and the two events are then polled -- so one thread records when
 * each lane finished. Blocking in one lane's `synchronize` instead would
 * inflate whichever finished first by however long the other still had to
 * run, which is the direction that matters here: the number this produces is
 * the concurrent rate, and the slower lane would otherwise decide both.
 *
 * It is also the path a real transfer takes. The worker issues copies and
 * reads an event; calibration now measures what it measures the same way,
 * rather than through a mechanism only it uses.
 *
 * The two probes start a fraction apart, because issuing the first batch
 * precedes issuing the second. Issuing is enqueueing -- a lane's `copy`
 * returns without waiting for anything -- so for a probe large enough to be
 * worth timing, the two are in flight together for substantially all of it.
 */
static int measure_concurrent_pair(
    ShadowSpillRuntime *runtime,
    ShadowSpillCalibrationProbe *first,
    ShadowSpillCalibrationProbe *second,
    uint32_t copies,
    uint64_t *first_nanoseconds,
    uint64_t *second_nanoseconds
) {
    ShadowSpillCalibrationProbe *const probes[2] = {first, second};
    ShadowSpillEventLease *leases[2] = {NULL, NULL};
    uint64_t begin[2] = {0U, 0U};
    uint64_t finished[2] = {0U, 0U};
    int status = 0;
    /* The last handle each probe's lane handed back: 0 from a lane that keeps
       nothing without a trace, and the transfer to ask about from one that
       answers for its own. */
    uint64_t last_handle[2] = {0U, 0U};

    for (unsigned index = 0U; index < 2U; ++index) {
        if (shadowspill_event_lease_acquire(
                runtime, &runtime->events, &leases[index]
            ) != SHADOWSPILL_STATUS_OK) {
            status = -1;
            break;
        }
    }
    /*
     * Issued a copy at a time, alternating, and both clocks started together.
     *
     * Draining one probe's whole batch before starting the other's cost the
     * first probe its own dispatch: its window contained the second's, the two
     * finished together because they contend for one link, and the first
     * therefore reported about 25 % low. Which route that was fell out of pool
     * registration order, which no caller would expect a measurement to depend
     * on.
     *
     * The assumption that made draining look safe was that "issuing is
     * enqueueing" -- true of a lane whose `copy` hands a stream one operation,
     * false of one that enqueues device work per chunk. The remote lane issues
     * 1024 operations for a 16-copy batch, about 0.5 s against a 1.4 s
     * transfer. Alternating costs nothing where the assumption did hold.
     *
     * Dispatch time is still inside both windows, and that is deliberate: the
     * worker pays it on the production path too, so a rate measured without it
     * would flatter the system. What this removes is the asymmetry.
     */
    for (unsigned index = 0U; status == 0 && index < 2U; ++index) {
        begin[index] = shadowspill_monotonic_ns();
        if (begin[index] == 0U) {
            status = -1;
        }
    }
    for (uint32_t copy = 0U; status == 0 && copy < copies; ++copy) {
        for (unsigned index = 0U; status == 0 && index < 2U; ++index) {
            const ShadowSpillRouteState *const route = probes[index]->route;
            if (route->operations->copy(
                    route->lane,
                    probes[index]->destination_pointer,
                    probes[index]->source_pointer,
                    probes[index]->bytes,
                    &last_handle[index]
                ) != 0) {
                status = -1;
            }
        }
    }
    for (unsigned index = 0U; status == 0 && index < 2U; ++index) {
        const ShadowSpillRouteState *const route = probes[index]->route;
        shadowspill_event_lease_issued_by(
            leases[index], route, last_handle[index]
        );
        if (route->operations->signal(
                route->lane, last_handle[index], leases[index]->event
            ) != 0) {
            status = -1;
        }
    }
    while (status == 0 && (finished[0] == 0U || finished[1] == 0U)) {
        for (unsigned index = 0U; index < 2U; ++index) {
            int complete = 0;
            if (finished[index] != 0U) {
                continue;
            }
            if (shadowspill_event_lease_landed(
                    runtime, leases[index], &complete
                ) != 0) {
                status = -1;
                break;
            }
            if (complete) {
                finished[index] = shadowspill_monotonic_ns();
            }
        }
        if (status == 0 && (finished[0] == 0U || finished[1] == 0U)) {
            shadowspill_thread_yield();
        }
    }
    for (unsigned index = 0U; index < 2U; ++index) {
        if (leases[index] != NULL) {
            (void)shadowspill_event_lease_release(runtime, leases[index]);
        }
    }
    if (status != 0 || finished[0] <= begin[0] || finished[1] <= begin[1]) {
        return -1;
    }
    *first_nanoseconds = finished[0] - begin[0];
    *second_nanoseconds = finished[1] - begin[1];
    return 0;
}

static int calibrate_reverse_pair(
    ShadowSpillRuntime *runtime,
    ShadowSpillRouteState *first_route,
    ShadowSpillRouteState *second_route,
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
            runtime,
            &first,
            &second,
            config->warmup_copies,
            &first_nanoseconds,
            &second_nanoseconds
        ) != 0) {
        status = -1;
    }
    uint64_t first_measurements[SHADOWSPILL_CALIBRATION_BATCH_SAMPLES] = {0};
    uint64_t second_measurements[SHADOWSPILL_CALIBRATION_BATCH_SAMPLES] = {0};
    for (uint32_t sample = 0U;
         status == 0 && sample < SHADOWSPILL_CALIBRATION_BATCH_SAMPLES;
         ++sample) {
        if (measure_concurrent_pair(
                runtime,
                &first,
                &second,
                config->measured_copies,
                &first_measurements[sample],
                &second_measurements[sample]
            ) != 0) {
            status = -1;
        }
    }
    release_probe(&second);
    release_probe(&first);
    if (status != 0) {
        return status;
    }
    first_nanoseconds = median_batch_nanoseconds(first_measurements);
    second_nanoseconds = median_batch_nanoseconds(second_measurements);
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
            shadowspill_monotonic_ns();
    }
    return 0;
}

ShadowSpillStatus shadowspill_runtime_calibrate_transfer_capabilities(
    ShadowSpillRuntime *runtime,
    const ShadowSpillTransferCalibrationConfig *provided_config,
    const ShadowSpillTransferRouteKey *routes,
    uint32_t route_count
) {
    if (runtime == NULL || (route_count != 0U && routes == NULL)) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    ShadowSpillTransferCalibrationConfig config = provided_config == NULL
        ? (ShadowSpillTransferCalibrationConfig){
            .abi_version = SHADOWSPILL_ABI_VERSION,
            .small_copy_bytes = 4096U,
            .large_copy_bytes = 256U << 20U,
            .warmup_copies = 4U,
            .measured_copies = 16U,
            .provenance = SHADOWSPILL_TRANSFER_PROFILE_RECALIBRATION,
        }
        : *provided_config;
    if (config.abi_version != SHADOWSPILL_ABI_VERSION ||
        config.small_copy_bytes == 0U ||
        config.large_copy_bytes < config.small_copy_bytes ||
        config.measured_copies == 0U ||
        config.provenance > SHADOWSPILL_TRANSFER_PROFILE_RECALIBRATION) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    for (uint32_t index = 0U; index < route_count; ++index) {
        if (routes[index].source_pool_id >= runtime->pool_count ||
            routes[index].destination_pool_id >= runtime->pool_count ||
            routes[index].source_pool_id == routes[index].destination_pool_id ||
            route_between(
                runtime,
                routes[index].source_pool_id,
                routes[index].destination_pool_id
            ) == NULL) {
            return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
        }
    }
    ShadowSpillStatus idle = shadowspill_runtime_wait_idle(runtime);
    if (idle != SHADOWSPILL_STATUS_OK) {
        return idle;
    }
    ShadowSpillTransferProfile *next = malloc(
        (size_t)runtime->transfer_profile_count * sizeof(*next)
    );
    if (next == NULL) {
        return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
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
            ShadowSpillRouteState *route = route_between(
                runtime, source, destination
            );
            if (route == NULL || calibrate_route(
                    runtime,
                    route,
                    &config,
                    &next[profile_index(runtime, source, destination)]
                ) != 0) {
                free(next);
                return SHADOWSPILL_STATUS_BACKEND_FAILURE;
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
            ShadowSpillRouteState *forward = route_between(
                runtime, source, destination
            );
            ShadowSpillRouteState *reverse = route_between(
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
                return SHADOWSPILL_STATUS_BACKEND_FAILURE;
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
    return SHADOWSPILL_STATUS_OK;
}

ShadowSpillStatus shadowspill_runtime_transfer_profiles(
    ShadowSpillRuntime *runtime,
    ShadowSpillTransferProfile *profiles,
    uint32_t capacity,
    uint32_t *count,
    uint64_t *generation
) {
    if (runtime == NULL || count == NULL || generation == NULL ||
        (profiles == NULL && capacity != 0U)) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    pthread_rwlock_rdlock(&runtime->transfer_profiles_lock);
    *count = runtime->transfer_profile_count;
    *generation = runtime->transfer_profile_generation;
    if (profiles != NULL) {
        if (capacity < runtime->transfer_profile_count) {
            pthread_rwlock_unlock(&runtime->transfer_profiles_lock);
            return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
        }
        memcpy(
            profiles,
            runtime->transfer_profiles,
            (size_t)runtime->transfer_profile_count * sizeof(*profiles)
        );
    }
    pthread_rwlock_unlock(&runtime->transfer_profiles_lock);
    return SHADOWSPILL_STATUS_OK;
}

/*
 * What the lane serving one route has moved.
 *
 * Nothing here knows which kind of lane answers: the route holds an operations
 * table and this asks it, so a registered lane and a built-in are reached by
 * the same call. A lane entitled to keep no count says so by leaving the entry
 * NULL, and UNSUPPORTED carries that up rather than zeroes a caller would read
 * as "moved nothing".
 */
ShadowSpillStatus shadowspill_route_lane_statistics(
    ShadowSpillRuntime *runtime,
    uint32_t route_id,
    ShadowSpillLaneStatistics *statistics
) {
    if (runtime == NULL || statistics == NULL ||
        route_id >= runtime->route_count) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    const ShadowSpillRouteState *const route = &runtime->routes[route_id];
    if (route->operations == NULL || route->lane == NULL) {
        return SHADOWSPILL_STATUS_UNSUPPORTED;
    }
    const ShadowSpillLane *const lane = route->lane;
    /* The seven come from the lane's own struct, so they are the counts it
       kept rather than a copy it made. Only the timing pair is asked for. */
    *statistics = (ShadowSpillLaneStatistics){
        .copies = shadowspill_lane_count(&lane->copies),
        .chunks = shadowspill_lane_count(&lane->chunks),
        .bytes = shadowspill_lane_count(&lane->bytes),
        .signals = shadowspill_lane_count(&lane->signals),
        .waits = shadowspill_lane_count(&lane->waits),
        .retries = shadowspill_lane_count(&lane->retries),
        .failures = shadowspill_lane_count(&lane->failures),
        .timed = 0U,
    };
    ShadowSpillLaneTiming timing = {0};
    if (route->operations->timing != NULL &&
        route->operations->timing(lane, &timing) == 0) {
        statistics->timed = 1U;
        statistics->posted_to_completion_seconds =
            timing.posted_to_completion_seconds;
        statistics->longest_completion_seconds =
            timing.longest_completion_seconds;
    }
    return SHADOWSPILL_STATUS_OK;
}
