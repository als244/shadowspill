/* Runtime-facing operations. Only this calling thread enqueues device work. */
#include "lane_internal.h"

#include <stdio.h>
#include <stdlib.h>

void ssd_lane_fail(SSDLane *lane) {
    if (atomic_exchange_explicit(&lane->failed, 1, memory_order_acq_rel)) return;
    shadowspill_lane_counted(&lane->base.failures, 1U);
    if (lane->base.runtime != NULL)
        shadowspill_lane_latch_failure(lane->base.runtime,
            SHADOWSPILL_STATUS_BACKEND_FAILURE, SHADOWSPILL_FAILURE_REASON_TRANSFER_REJECTED);
    pthread_mutex_lock(&lane->lock);
    /* Release stream waits only after recording the failure. A failed transfer
       must fail the step, rather than hang its teardown behind an I/O word. */
    ssd_report(lane, SSD_IO, lane->chunks);
    ssd_report(lane, SSD_DONE, lane->accepted);
    pthread_cond_broadcast(&lane->changed);
    pthread_mutex_unlock(&lane->lock);
}

static int lane_wait(ShadowSpillLane *base, ShadowSpillBackendEvent event) {
    shadowspill_lane_counted(&base->waits, 1U);
    SSDLane *lane = (SSDLane *)base;
    if (!lane->writing && base->backend->wait_event(base->backend->state,
            lane->readiness_stream, event) != 0) {
        ssd_lane_fail(lane);
        return -1;
    }
    if (base->backend->wait_event(base->backend->state, base->stream, event) != 0) {
        ssd_lane_fail((SSDLane *)base);
        return -1;
    }
    return 0;
}

static int in_range(ShadowSpillLaneRange range, const void *pointer, uint64_t bytes) {
    const uintptr_t start = (uintptr_t)range.address, at = (uintptr_t)pointer;
    return at >= start && at - start <= range.bytes && bytes <= range.bytes - (at - start);
}

static int enqueue(SSDLane *lane, const SSDWork *work) {
    const ShadowSpillBackend *backend = lane->base.backend;
    const ShadowSpillBackendStream stream = lane->base.stream;
    if (backend->write_value(backend->state,
                            lane->writing ? stream : lane->readiness_stream,
                            lane->signals, SSD_GATE, work->sequence) != 0) return -1;
    for (uint64_t index = 0; index < work->chunks; ++index) {
        const SSDPiece piece = ssd_piece(lane, work, index);
        const uint64_t number = work->first_chunk + index;
        void *host = (char *)piece.slot + piece.prefix;
        void *device = (char *)work->device + piece.device_offset;
        if (lane->writing) {
            if (number >= lane->depth && backend->wait_value(
                    backend->state, stream, lane->signals, SSD_IO,
                    number + 1U - lane->depth) != 0) return -1;
            if (backend->copy_device_to_host(backend->state, host, device,
                                             piece.bytes, stream) != 0) return -1;
        } else {
            if (backend->wait_value(backend->state, stream, lane->signals,
                                   SSD_IO, number + 1U) != 0) return -1;
            if (backend->copy_host_to_device(backend->state, device, host,
                                             piece.bytes, stream) != 0) return -1;
        }
        if (backend->write_value(backend->state, stream, lane->signals,
                                SSD_DEVICE, number + 1U) != 0) return -1;
    }
    return 0;
}

static int lane_copy(ShadowSpillLane *base, void *destination, const void *source,
                     uint64_t bytes, uint64_t *handle) {
    SSDLane *lane = (SSDLane *)base;
    if (handle == NULL || !in_range(base->from_range, source, bytes) ||
        !in_range(base->to_range, destination, bytes)) return -1;
    *handle = 0U;
    SSDTrace *trace = NULL;
    if (base->runtime != NULL && shadowspill_lane_trace_active(base->runtime)) {
        trace = calloc(1U, sizeof(*trace));
        if (trace == NULL) return -1;
        trace->issued = ssd_now();
        trace->bytes = bytes;
    }
    pthread_mutex_lock(&lane->lock);
    while (lane->accepted - lane->retired == SSD_WORK_SLOTS &&
           !atomic_load_explicit(&lane->failed, memory_order_acquire))
        pthread_cond_wait(&lane->changed, &lane->lock);
    if (atomic_load_explicit(&lane->failed, memory_order_acquire)) {
        pthread_mutex_unlock(&lane->lock);
        free(trace);
        return -1;
    }
    const uintptr_t remote = (uintptr_t)(lane->writing ? destination : source);
    const uint64_t offset = remote - (uintptr_t)lane->region->base;
    const uint64_t prefix = offset % lane->region->alignment;
    const uint64_t chunk = lane->region->config.chunk_bytes;
    const uint64_t chunks = bytes == 0U ? 0U : (bytes + prefix + chunk - 1U) / chunk;
    SSDWork work = {
        .device = lane->writing ? (void *)(uintptr_t)source : destination,
        .offset = offset, .bytes = bytes, .chunks = chunks,
        .first_chunk = lane->chunks, .sequence = lane->accepted + 1U, .trace = trace,
    };
    if (trace != NULL) {
        trace->chunks = chunks;
        trace->next = lane->traces;
        lane->traces = trace;
        *handle = (uint64_t)(uintptr_t)trace;
    }
    lane->work[lane->accepted % SSD_WORK_SLOTS] = work;
    lane->chunks += chunks;
    ++lane->accepted;
    pthread_cond_signal(&lane->changed);
    pthread_mutex_unlock(&lane->lock);
    shadowspill_lane_counted(&base->copies, 1U);
    shadowspill_lane_counted(&base->bytes, bytes);
    /* Publish before enqueuing: the device may fill its command queue while
       waiting for this thread's I/O completion words. */
    if (enqueue(lane, &work) != 0) {
        ssd_lane_fail(lane);
        return -1;
    }
    return 0;
}

static int lane_signal(ShadowSpillLane *base, uint64_t handle,
                       ShadowSpillBackendEvent event) {
    (void)handle;
    SSDLane *lane = (SSDLane *)base;
    const ShadowSpillBackend *backend = base->backend;
    shadowspill_lane_counted(&base->signals, 1U);
    if (backend->wait_value(backend->state, base->stream, lane->signals,
                            SSD_DONE, lane->accepted) != 0 ||
        backend->record_event(backend->state, event, base->stream) != 0) {
        ssd_lane_fail(lane);
        return -1;
    }
    return 0;
}

static int lane_synchronize(ShadowSpillLane *base) {
    SSDLane *lane = (SSDLane *)base;
    pthread_mutex_lock(&lane->lock);
    while (lane->retired < lane->accepted &&
           !atomic_load_explicit(&lane->failed, memory_order_acquire))
        pthread_cond_wait(&lane->changed, &lane->lock);
    pthread_mutex_unlock(&lane->lock);
    const int status = base->backend->synchronize_stream(base->backend->state, base->stream);
    return status == 0 && !atomic_load_explicit(&lane->failed, memory_order_acquire) ? 0 : -1;
}

static int lane_transfer(ShadowSpillLane *base, uint64_t handle,
                         ShadowSpillLaneTransfer *result) {
    SSDLane *lane = (SSDLane *)base;
    SSDTrace *trace = (SSDTrace *)(uintptr_t)handle;
    if (trace == NULL || result == NULL) return -1;
    pthread_mutex_lock(&lane->lock);
    SSDTrace **link = &lane->traces;
    while (*link != NULL && *link != trace) link = &(*link)->next;
    if (*link == NULL) {
        pthread_mutex_unlock(&lane->lock);
        return -1;
    }
    *link = trace->next;
    pthread_mutex_unlock(&lane->lock);
    (void)ssd_word(lane, SSD_DONE);
    *result = (ShadowSpillLaneTransfer){
        .issued_at_nanoseconds = shadowspill_lane_origin_instant(base->runtime, trace->issued),
        .started_at_nanoseconds = shadowspill_lane_origin_instant(base->runtime, trace->started),
        .finished_at_nanoseconds = shadowspill_lane_origin_instant(base->runtime, trace->finished),
        .bytes = trace->bytes, .chunks = trace->chunks,
    };
    free(trace);
    return 0;
}

static void lane_destroy(ShadowSpillLane *base) {
    SSDLane *lane = (SSDLane *)base;
    if (lane->thread_started) {
        /* Normal close has already drained the route. Abandon must also be
           able to stop a worker waiting for a producer that will never run. */
        atomic_store_explicit(&lane->stopping, 1, memory_order_release);
        pthread_mutex_lock(&lane->lock);
        pthread_cond_signal(&lane->changed);
        pthread_mutex_unlock(&lane->lock);
        (void)pthread_join(lane->thread, NULL);
    }
    ssd_io_destroy(lane);
    if (lane->words != NULL) {
        ssd_report(lane, SSD_IO, lane->chunks);
        ssd_report(lane, SSD_DONE, lane->accepted);
    }
    const ShadowSpillBackend *backend = base->backend;
    if (lane->readiness_created)
        (void)backend->destroy_stream(backend->state, lane->readiness_stream);
    if (lane->signals != 0U) (void)backend->free_signals(backend->state, lane->signals);
    if (lane->registered)
        (void)backend->unregister_host_memory(backend->state, lane->ring, lane->ring_bytes);
    free(lane->ring);
    free(lane->edges);
    if (lane->reserved_bytes) ssd_staging_release(lane->region, lane->reserved_bytes);
    while (lane->traces != NULL) {
        SSDTrace *next = lane->traces->next;
        free(lane->traces);
        lane->traces = next;
    }
    pthread_cond_destroy(&lane->changed);
    pthread_mutex_destroy(&lane->lock);
    free(lane);
}

static int lane_create(const ShadowSpillLane *base, void *configuration,
                       ShadowSpillLane **result) {
    (void)configuration;
    SSDLane *lane = calloc(1U, sizeof(*lane));
    if (lane == NULL) return -1;
    lane->base = *base;
    lane->writing = base->to_kind == SHADOWSPILL_SSD_POOL_KIND;
    lane->region = ssd_region_find(lane->writing ? base->to_range.address : base->from_range.address);
    if (lane->region == NULL || pthread_mutex_init(&lane->lock, NULL) != 0) {
        free(lane);
        return -1;
    }
    if (pthread_cond_init(&lane->changed, NULL) != 0) {
        pthread_mutex_destroy(&lane->lock);
        free(lane);
        return -1;
    }
    SSDRegion *region = lane->region;
    lane->depth = region->config.queue_depth;
    if (lane->depth > SIZE_MAX / region->config.chunk_bytes) goto fail;
    lane->ring_bytes = lane->depth * region->config.chunk_bytes;
    if (lane->ring_bytes > SIZE_MAX - region->alignment) goto fail;
    const uint64_t staging = lane->ring_bytes + region->alignment;
    if (ssd_staging_reserve(region, staging) != 0) goto fail;
    lane->reserved_bytes = staging;
    if (posix_memalign(&lane->ring, (size_t)region->alignment, (size_t)lane->ring_bytes) != 0 ||
        posix_memalign(&lane->edges, (size_t)region->alignment, (size_t)region->alignment) != 0)
        goto fail;
    const ShadowSpillBackend *backend = base->backend;
    if (backend->register_host_memory(backend->state, lane->ring, lane->ring_bytes) != 0)
        goto fail;
    lane->registered = 1;
    if (backend->allocate_signals(backend->state, SSD_SIGNALS, &lane->signals, &lane->words) != 0 ||
        ssd_io_create(lane) != 0) goto fail;
    if (!lane->writing) {
        if (backend->create_stream(backend->state, &lane->readiness_stream) != 0)
            goto fail;
        lane->readiness_created = 1;
    }
    if (pthread_create(&lane->thread, NULL, ssd_io_thread, lane) != 0) goto fail;
    lane->thread_started = 1;
    *result = &lane->base;
    return 0;
fail:
    lane_destroy(&lane->base);
    return -1;
}

static const ShadowSpillLaneOperations operations = {
    .wait = lane_wait, .copy = lane_copy, .signal = lane_signal,
    .synchronize = lane_synchronize, .transfer = lane_transfer, .destroy = lane_destroy,
};

const ShadowSpillLaneDescription shadowspill_ssd_lanes[2] = {
    {.from_kind = SHADOWSPILL_SSD_POOL_KIND, .to_kind = 0U,
     .operations = &operations, .create = lane_create},
    {.from_kind = 0U, .to_kind = SHADOWSPILL_SSD_POOL_KIND,
     .operations = &operations, .create = lane_create},
};
