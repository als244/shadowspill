#ifndef SHADOWSPILL_SSD_LANE_INTERNAL_H
#define SHADOWSPILL_SSD_LANE_INTERNAL_H

#include "internal.h"
#include <linux/aio_abi.h>
#include <time.h>

#define SSD_WORK_SLOTS 128U
enum { SSD_IO, SSD_DEVICE, SSD_GATE, SSD_DONE, SSD_SIGNALS };

typedef struct SSDTrace {
    uint64_t issued, started, finished, bytes, chunks;
    struct SSDTrace *next;
} SSDTrace;

typedef struct SSDWork {
    void *device;
    uint64_t offset, bytes, chunks, first_chunk, sequence;
    SSDTrace *trace;
} SSDWork;

typedef struct SSDLane {
    ShadowSpillLane base;
    SSDRegion *region;
    int writing;
    /* Source readiness must not queue behind preceding fetch copies. */
    ShadowSpillBackendStream readiness_stream;
    int readiness_created;
    void *ring, *edges;
    uint64_t ring_bytes, reserved_bytes;
    uint32_t depth;
    int registered;
    ShadowSpillBackendSignals signals;
    uint64_t *words;
    aio_context_t aio;
    struct iocb *requests;
    struct io_event *events;
    uint8_t *landed;
    pthread_t thread;
    int thread_started;
    _Atomic int stopping, failed;
    pthread_mutex_t lock;
    pthread_cond_t changed;
    SSDWork work[SSD_WORK_SLOTS];
    uint64_t accepted, retired, chunks;
    SSDTrace *traces;
} SSDLane;

typedef struct SSDPiece {
    void *slot;
    uint64_t file_offset, disk_bytes, prefix, bytes, device_offset;
} SSDPiece;

static inline uint64_t ssd_now(void) {
    struct timespec at;
    clock_gettime(CLOCK_MONOTONIC, &at);
    return (uint64_t)at.tv_sec * 1000000000U + (uint64_t)at.tv_nsec;
}

static inline uint64_t ssd_word(const SSDLane *lane, unsigned index) {
    return atomic_load_explicit((_Atomic uint64_t *)&lane->words[index],
                                memory_order_acquire);
}

static inline void ssd_report(SSDLane *lane, unsigned index, uint64_t value) {
    atomic_store_explicit((_Atomic uint64_t *)&lane->words[index], value,
                          memory_order_release);
}

static inline int ssd_stopped(const SSDLane *lane) {
    return atomic_load_explicit(&lane->stopping, memory_order_acquire) ||
        atomic_load_explicit(&lane->failed, memory_order_acquire);
}

SSDPiece ssd_piece(const SSDLane *lane, const SSDWork *work, uint64_t index);
int ssd_io_create(SSDLane *lane);
void ssd_io_destroy(SSDLane *lane);
void *ssd_io_thread(void *argument);
void ssd_lane_fail(SSDLane *lane);

#endif
