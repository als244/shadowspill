/* Linux direct AIO owns disk work; this thread never calls the device backend. */
#define _GNU_SOURCE
#include "lane_internal.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

SSDPiece ssd_piece(const SSDLane *lane, const SSDWork *work, uint64_t index) {
    const uint64_t chunk = lane->region->config.chunk_bytes;
    const uint64_t alignment = lane->region->alignment;
    const uint64_t head = work->offset % alignment;
    const uint64_t prefix = index == 0U ? head : 0U;
    const uint64_t device_offset = index == 0U ? 0U : index * chunk - head;
    const uint64_t remaining = work->bytes - device_offset;
    const uint64_t bytes = remaining < chunk - prefix ? remaining : chunk - prefix;
    return (SSDPiece){
        .slot = (char *)lane->ring +
            ((work->first_chunk + index) % lane->depth) * chunk,
        .file_offset = work->offset - head + index * chunk,
        .disk_bytes = (prefix + bytes + alignment - 1U) / alignment * alignment,
        .prefix = prefix, .bytes = bytes, .device_offset = device_offset,
    };
}

int ssd_io_create(SSDLane *lane) {
    lane->requests = calloc(lane->depth, sizeof(*lane->requests));
    lane->events = calloc(lane->depth, sizeof(*lane->events));
    lane->landed = calloc(lane->depth, sizeof(*lane->landed));
    if (lane->requests == NULL || lane->events == NULL || lane->landed == NULL)
        return -1;
    if (syscall(SYS_io_setup, lane->depth, &lane->aio) != 0) {
        fprintf(stderr, "shadowspill SSD: io_setup: %s\n", strerror(errno));
        return -1;
    }
    return 0;
}

void ssd_io_destroy(SSDLane *lane) {
    /* io_destroy drains outstanding requests before the ring is released. */
    if (lane->aio != 0U) (void)syscall(SYS_io_destroy, lane->aio);
    lane->aio = 0U;
    free(lane->landed);
    free(lane->events);
    free(lane->requests);
}

/* Only boundary sectors need read/modify/write. Serialize those with other
   partial writers so distinct leases sharing a sector cannot overwrite each
   other. Interior aligned chunks keep their full asynchronous queue depth. */
static int write_edge(SSDLane *lane, const SSDPiece *piece) {
    SSDRegion *region = lane->region;
    const uint64_t alignment = region->alignment;
    int status = 0;
    pthread_mutex_lock(&region->state_lock);
    if (piece->prefix != 0U) {
        status = ssd_direct_io(region->fd, lane->edges, alignment,
                               piece->file_offset, 0);
        if (status == 0)
            memcpy(piece->slot, lane->edges, (size_t)piece->prefix);
    }
    const uint64_t end = piece->prefix + piece->bytes;
    if (status == 0 && end < piece->disk_bytes) {
        const uint64_t tail = piece->disk_bytes - alignment;
        status = ssd_direct_io(region->fd, lane->edges, alignment,
                               piece->file_offset + tail, 0);
        if (status == 0)
            memcpy((char *)piece->slot + end,
                   (char *)lane->edges + end - tail,
                   (size_t)(piece->disk_bytes - end));
    }
    if (status == 0)
        status = ssd_direct_io(region->fd, piece->slot, piece->disk_bytes,
                               piece->file_offset, 1);
    pthread_mutex_unlock(&region->state_lock);
    return status;
}

static int post(SSDLane *lane, SSDWork *work, uint64_t index) {
    const uint64_t number = work->first_chunk + index;
    const uint64_t ready = lane->writing ? number + 1U
        : (number < lane->depth ? 0U : number + 1U - lane->depth);
    if (ssd_word(lane, SSD_DEVICE) < ready) return 0;
    const SSDPiece piece = ssd_piece(lane, work, index);
    const uint64_t slot = number % lane->depth;
    if (work->trace != NULL && work->trace->started == 0U)
        work->trace->started = ssd_now();
    if (lane->writing && (piece.prefix != 0U || piece.bytes != piece.disk_bytes)) {
        if (write_edge(lane, &piece) != 0) return -1;
        lane->landed[slot] = 1U;
    } else {
        struct iocb *request = &lane->requests[slot];
        *request = (struct iocb){
            .aio_data = number,
            .aio_lio_opcode = lane->writing ? IOCB_CMD_PWRITE : IOCB_CMD_PREAD,
            .aio_fildes = (uint32_t)lane->region->fd,
            .aio_buf = (uint64_t)(uintptr_t)piece.slot,
            .aio_nbytes = piece.disk_bytes,
            .aio_offset = (int64_t)piece.file_offset,
        };
        const long result = syscall(SYS_io_submit, lane->aio, 1L, &request);
        if (result != 1L) {
            if (result == 0L || errno == EAGAIN || errno == EINTR) return 0;
            fprintf(stderr, "shadowspill SSD: io_submit: %s\n", strerror(errno));
            return -1;
        }
    }
    shadowspill_lane_counted(&lane->base.chunks, 1U);
    return 1;
}

static int collect(SSDLane *lane) {
    const struct timespec immediate = {0};
    const long count = syscall(SYS_io_getevents, lane->aio, 0L,
                               (long)lane->depth, lane->events, &immediate);
    if (count < 0L) return errno == EINTR ? 0 : -1;
    for (long i = 0; i < count; ++i) {
        const struct io_event *event = &lane->events[i];
        const uint64_t slot = event->data % lane->depth;
        if (event->res < 0 || event->res2 != 0 ||
            (uint64_t)event->res != lane->requests[slot].aio_nbytes) {
            fprintf(stderr, "shadowspill SSD: %s completion failed "
                    "(result=%" PRId64 ", expected=%" PRIu64 ")\n",
                    lane->writing ? "write" : "read", (int64_t)event->res,
                    (uint64_t)lane->requests[slot].aio_nbytes);
            return -1;
        }
        lane->landed[slot] = 1U;
    }
    return 0;
}

static void pause_io(void) {
    /* Active work polls completion words without a scheduler wake-up delay.
       Idle lanes sleep on changed; they do not consume a core without work. */
#if defined(__x86_64__) || defined(__i386__)
    __asm__ volatile("pause" ::: "memory");
#elif defined(__aarch64__)
    __asm__ volatile("yield" ::: "memory");
#else
    atomic_signal_fence(memory_order_seq_cst);
#endif
}

static int process(SSDLane *lane, SSDWork *work) {
    uint64_t posted = 0U, retired = 0U;
    while (ssd_word(lane, SSD_GATE) < work->sequence) {
        if (ssd_stopped(lane)) return -1;
        pause_io();
    }
    if (work->trace != NULL) work->trace->started = ssd_now();
    while (retired < work->chunks) {
        if (ssd_stopped(lane)) return -1;
        const uint64_t before_posted = posted, before_retired = retired;
        while (posted < work->chunks && posted - retired < lane->depth) {
            const int result = post(lane, work, posted);
            if (result < 0) return -1;
            if (result == 0) break;
            ++posted;
        }
        if (collect(lane) != 0) return -1;
        while (retired < posted &&
               lane->landed[(work->first_chunk + retired) % lane->depth]) {
            lane->landed[(work->first_chunk + retired) % lane->depth] = 0U;
            ++retired;
        }
        if (retired != before_retired)
            ssd_report(lane, SSD_IO, work->first_chunk + retired);
        if (posted == before_posted && retired == before_retired) pause_io();
    }
    if (work->trace != NULL && !lane->writing) {
        while (ssd_word(lane, SSD_DEVICE) < work->first_chunk + work->chunks) {
            if (ssd_stopped(lane)) return -1;
            pause_io();
        }
    }
    return 0;
}

/* Read ahead across object boundaries. The global chunk number identifies
   both a ring slot and the monotonically published I/O/device acknowledgments.
   A slot is not reused until the HtoD stream acknowledges its previous owner.
   Per-object source dependencies are opened on the separate readiness stream. */
static void *fetch_io_thread(SSDLane *lane) {
    uint64_t next_work = 0U, index = 0U, posted = 0U, completed = 0U;
    (void)pthread_setname_np(pthread_self(), "ssd.fetch");
    for (;;) {
        pthread_mutex_lock(&lane->lock);
        while (lane->retired == lane->accepted && !ssd_stopped(lane))
            pthread_cond_wait(&lane->changed, &lane->lock);
        const uint64_t accepted = lane->accepted;
        pthread_mutex_unlock(&lane->lock);
        if (ssd_stopped(lane)) return NULL;

        while (next_work < accepted && posted - completed < lane->depth) {
            pthread_mutex_lock(&lane->lock);
            SSDWork work = lane->work[next_work % SSD_WORK_SLOTS];
            pthread_mutex_unlock(&lane->lock);
            if (ssd_word(lane, SSD_GATE) < work.sequence) break;
            if (work.chunks == 0U && work.trace != NULL && work.trace->started == 0U)
                work.trace->started = ssd_now();
            if (index < work.chunks) {
                const int status = post(lane, &work, index);
                if (status < 0) goto fail;
                if (status == 0) break;
                ++index;
                ++posted;
            }
            if (index == work.chunks) {
                ++next_work;
                index = 0U;
            }
        }

        if (collect(lane) != 0) goto fail;
        const uint64_t before = completed;
        while (completed < posted && lane->landed[completed % lane->depth]) {
            lane->landed[completed % lane->depth] = 0U;
            ++completed;
        }
        if (completed != before) ssd_report(lane, SSD_IO, completed);

        pthread_mutex_lock(&lane->lock);
        while (lane->retired < accepted) {
            SSDWork *work = &lane->work[lane->retired % SSD_WORK_SLOTS];
            const uint64_t end = work->first_chunk + work->chunks;
            /* next_work also distinguishes posted zero-byte objects. */
            if (lane->retired >= next_work || completed < end ||
                (work->trace != NULL && ssd_word(lane, SSD_DEVICE) < end)) break;
            if (work->trace != NULL) work->trace->finished = ssd_now();
            ssd_report(lane, SSD_DONE, work->sequence);
            ++lane->retired;
            pthread_cond_broadcast(&lane->changed);
        }
        pthread_mutex_unlock(&lane->lock);
        pause_io();
    }
fail:
    if (!atomic_load_explicit(&lane->stopping, memory_order_acquire))
        ssd_lane_fail(lane);
    return NULL;
}

void *ssd_io_thread(void *argument) {
    SSDLane *lane = argument;
    if (!lane->writing) return fetch_io_thread(lane);
    (void)pthread_setname_np(pthread_self(), lane->writing ? "ssd.evict" : "ssd.fetch");
    for (;;) {
        pthread_mutex_lock(&lane->lock);
        while (lane->retired == lane->accepted && !ssd_stopped(lane))
            pthread_cond_wait(&lane->changed, &lane->lock);
        if (ssd_stopped(lane)) {
            pthread_mutex_unlock(&lane->lock);
            return NULL;
        }
        SSDWork work = lane->work[lane->retired % SSD_WORK_SLOTS];
        pthread_mutex_unlock(&lane->lock);
        if (process(lane, &work) != 0) {
            if (!atomic_load_explicit(&lane->stopping, memory_order_acquire))
                ssd_lane_fail(lane);
            return NULL;
        }
        if (work.trace != NULL) work.trace->finished = ssd_now();
        /* Publish metadata before the event can complete and retire it. */
        ssd_report(lane, SSD_DONE, work.sequence);
        pthread_mutex_lock(&lane->lock);
        ++lane->retired;
        pthread_cond_broadcast(&lane->changed);
        pthread_mutex_unlock(&lane->lock);
    }
}
