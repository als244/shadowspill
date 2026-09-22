/* The remote lane's measuring instrument: every instant of a transfer that had
   the lane to itself, on both of the threads that carry it. */

#include "remote_lane_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * THE INSTANTS ONE TRANSFER PASSES THROUGH, on the two threads that carry it.
 *
 * Two chains rather than one, because after `copy` publishes a transfer the
 * worker and the lane's thread run at the same time: the thread can have
 * posted the transfer before the worker has returned. A single ordered list
 * of stamps cannot partition that, so each chain is ordered on its own, and a
 * handoff between them is a difference of two absolute instants.
 *
 * Captured only while measuring, and only for a transfer that had the lane to
 * itself: nothing accepted since the last `synchronize`, nothing in flight,
 * one piece. That is the transfer a latency figure describes, and the only
 * one whose instants are not queueing behind a neighbour's.
 */

void remote_lane_stamp(RemoteLane *lane, const Work *work, unsigned point) {
    if (work == NULL || work->timeline_row < 0) {
        return;
    }
    double *const at = lane->timeline[work->timeline_row].at;
    if (at[point] == 0.0) {
        at[point] = remote_lane_seconds_now();
    }
}

void remote_lane_capture_timeline(RemoteLane *lane, Work *work, double entered) {
    if (lane->timeline_rows >= TIMELINE_ROWS ||
        lane->accepted != lane->retired ||
        lane->accepted != lane->synchronized_through ||
        work->chunks != 1U) {
        return;
    }
    TimelineRow *const row = &lane->timeline[lane->timeline_rows];
    memset(row, 0, sizeof(*row));
    row->bytes = work->bytes;
    row->first_chunk = work->first_chunk;
    row->at[TIMELINE_ENTERED] = entered;
    work->timeline_row = (int32_t)lane->timeline_rows++;
}

const Work *remote_lane_captured_transfer(const RemoteLane *lane) {
    if (lane->accepted != lane->synchronized_through + 1U) {
        return NULL;
    }
    const Work *const work =
        &lane->work_ring[(lane->accepted - 1U) % WORK_SLOTS];
    return work->timeline_row >= 0 ? work : NULL;
}

/*
 * A batch's first transfer was claimed with the lane to itself and captured,
 * and the rest of the batch was then accepted behind it -- so no `synchronize`
 * will ever cover it alone and its row cannot be completed. It is the last row
 * claimed, since nothing accepted after it qualified.
 */
void remote_lane_release_batch_row(RemoteLane *lane, uint64_t accepted) {
    const uint64_t batch = accepted - lane->synchronized_through;
    if (batch < 2U || batch > WORK_SLOTS) {
        return;
    }
    Work *const first =
        &lane->work_ring[lane->synchronized_through % WORK_SLOTS];
    if (first->timeline_row >= 0 &&
        first->timeline_row + 1 == (int32_t)lane->timeline_rows) {
        memset(&lane->timeline[first->timeline_row], 0, sizeof(TimelineRow));
        --lane->timeline_rows;
        first->timeline_row = -1;
    }
}

/*
 * Where a staged fetch's time after the NIC goes: the route stream is waiting
 * on the word the thread stored at `TIMELINE_REPORTED`; it then copies the
 * piece in and stores the device's word; and `synchronize_stream` reports only
 * when all of it is done. Seen from here, the store separates the device
 * finishing from the host noticing that it has. Bounded, because a stream
 * that failed never stores it. An evict, and either direction on the direct
 * path, has no such store: the NIC's completion was the landing.
 */
void remote_lane_watch_device(RemoteLane *lane, const Work *work) {
    if (!lane->stages || work->to_remote) {
        remote_lane_stamp(lane, work, TIMELINE_DEVICE_SEEN);
        return;
    }
    const uint64_t last = work->first_chunk + work->chunks;
    const double deadline = remote_lane_seconds_now() + 1.0;
    while (!remote_lane_device_reached(lane, last)) {
        if (remote_lane_seconds_now() >= deadline) {
            ++lane->device_watch_timeouts;
            return;
        }
#if defined(__x86_64__) || defined(__i386__)
        __builtin_ia32_pause();
#elif defined(__aarch64__)
        __asm__ volatile("yield");
#endif
    }
    remote_lane_stamp(lane, work, TIMELINE_DEVICE_SEEN);
}

/* ----------------------------------------------------------- the report */

typedef struct TimelineSpan {
    const char *name;
    unsigned from;
    unsigned to;
} TimelineSpan;

static int compare_doubles(const void *first, const void *second) {
    const double left = *(const double *)first;
    const double right = *(const double *)second;
    return left < right ? -1 : (left > right ? 1 : 0);
}

/*
 * The captured transfers, as medians: each instant after `copy` was entered,
 * in the order they happen, then the spans that say where the time went, then
 * every row. Only a row with every instant counts; one without is a transfer
 * captured at its claim and then not synchronized alone, or one whose device
 * store was never seen, and either is reported as a count.
 */
static void report_timeline(const RemoteLane *lane) {
    static const char *const point_names[TIMELINE_POINTS] = {
        "worker  copy entered",
        "worker  slot published",
        "worker  device side enqueued",
        "worker  synchronize entered",
        "worker  saw the thread retire",
        "worker  saw the device's store",
        "worker  stream synchronized",
        "thread  saw the transfer",
        "thread  first piece may post",
        "thread  posted",
        "thread  completion polled",
        "thread  reported to the stream",
        "thread  retired",
    };
    static const TimelineSpan spans[] = {
        {"worker: claim and plan", TIMELINE_ENTERED, TIMELINE_PUBLISHED},
        {"worker: enqueue the device side", TIMELINE_PUBLISHED, TIMELINE_ENQUEUED},
        {"worker: enqueued to sync entered", TIMELINE_ENQUEUED, TIMELINE_SYNC_ENTERED},
        {"worker: waits for the thread", TIMELINE_SYNC_ENTERED, TIMELINE_SYNC_DRAINED},
        {"worker: device finishes after", TIMELINE_SYNC_DRAINED, TIMELINE_DEVICE_SEEN},
        {"worker: stream synchronize", TIMELINE_DEVICE_SEEN, TIMELINE_SYNC_RETURNED},
        {"thread: picks it up", TIMELINE_PUBLISHED, TIMELINE_SEEN},
        {"thread: waits for the device", TIMELINE_SEEN, TIMELINE_READY},
        {"thread: ibv_post_send", TIMELINE_READY, TIMELINE_POSTED},
        {"thread: NIC, posted to polled", TIMELINE_POSTED, TIMELINE_COMPLETED},
        {"thread: report to the stream", TIMELINE_COMPLETED, TIMELINE_REPORTED},
        {"thread: retire under the lock", TIMELINE_REPORTED, TIMELINE_RETIRED},
        {"handoff: retired to sync sees", TIMELINE_RETIRED, TIMELINE_SYNC_DRAINED},
        {"device: NIC word to its store", TIMELINE_REPORTED, TIMELINE_DEVICE_SEEN},
        {"end to end", TIMELINE_ENTERED, TIMELINE_SYNC_RETURNED},
    };
    const TimelineRow *rows[TIMELINE_ROWS];
    uint32_t complete = 0U;
    for (uint32_t which = 0U; which < lane->timeline_rows; ++which) {
        const TimelineRow *const row = &lane->timeline[which];
        unsigned point = 0U;
        while (point < TIMELINE_POINTS && row->at[point] != 0.0) {
            ++point;
        }
        if (point == TIMELINE_POINTS) {
            rows[complete++] = row;
        }
    }
    if (complete == 0U) {
        fprintf(
            stderr,
            "shadowspill network:   timeline -- no transfer had the lane to "
            "itself (%u rows claimed, %llu device watches timed out)\n",
            lane->timeline_rows,
            (unsigned long long)lane->device_watch_timeouts
        );
        return;
    }
    fprintf(
        stderr,
        "shadowspill network:   timeline -- %u transfers of %llu bytes that had "
        "the lane to themselves (%u rows incomplete), %s; median us after "
        "copy() was entered\n",
        complete, (unsigned long long)rows[0]->bytes,
        lane->timeline_rows - complete,
        lane->stages ? "staged through the ring" : "posted from the pool"
    );
    double values[TIMELINE_ROWS];
    double offsets[TIMELINE_POINTS];
    unsigned order[TIMELINE_POINTS];
    for (unsigned point = 0U; point < TIMELINE_POINTS; ++point) {
        for (uint32_t which = 0U; which < complete; ++which) {
            values[which] =
                (rows[which]->at[point] - rows[which]->at[TIMELINE_ENTERED])
                * 1e6;
        }
        qsort(values, complete, sizeof(values[0]), compare_doubles);
        offsets[point] = values[complete / 2U];
        order[point] = point;
    }
    /* In the order they happen, which interleaves the two chains. */
    for (unsigned outer = 1U; outer < TIMELINE_POINTS; ++outer) {
        for (unsigned inner = outer;
             inner > 0U && offsets[order[inner]] < offsets[order[inner - 1U]];
             --inner) {
            const unsigned swap = order[inner];
            order[inner] = order[inner - 1U];
            order[inner - 1U] = swap;
        }
    }
    for (unsigned rank = 0U; rank < TIMELINE_POINTS; ++rank) {
        fprintf(
            stderr, "shadowspill network:     %-36s %8.2f\n",
            point_names[order[rank]], offsets[order[rank]]
        );
    }
    fprintf(stderr, "shadowspill network:   spans -- median, min us\n");
    for (size_t span = 0U; span < sizeof(spans) / sizeof(spans[0]); ++span) {
        for (uint32_t which = 0U; which < complete; ++which) {
            values[which] =
                (rows[which]->at[spans[span].to]
                 - rows[which]->at[spans[span].from]) * 1e6;
        }
        qsort(values, complete, sizeof(values[0]), compare_doubles);
        fprintf(
            stderr, "shadowspill network:     %-36s %8.2f  %8.2f\n",
            spans[span].name, values[complete / 2U], values[0]
        );
    }
    /* Every row, for a reader after the distribution rather than its middle:
       each instant in the enum's order, us after entry, and the ring slot the
       piece used. */
    for (uint32_t which = 0U; which < complete; ++which) {
        fprintf(
            stderr, "shadowspill network:     row %2u chunk %llu slot %llu:",
            which, (unsigned long long)rows[which]->first_chunk,
            (unsigned long long)(rows[which]->first_chunk
                                 % lane->tuning.ring_slots)
        );
        for (unsigned point = 1U; point < TIMELINE_POINTS; ++point) {
            fprintf(
                stderr, " %.2f",
                (rows[which]->at[point] - rows[which]->at[TIMELINE_ENTERED])
                * 1e6
            );
        }
        fputc('\n', stderr);
    }
    if (lane->device_watch_timeouts != 0U) {
        fprintf(
            stderr, "shadowspill network:   %llu device watches timed out\n",
            (unsigned long long)lane->device_watch_timeouts
        );
    }
}

void remote_lane_report(const RemoteLane *lane) {
    report_timeline(lane);
    fprintf(
        stderr,
        "shadowspill network:   completion queue -- %llu polls, %llu empty,"
        " %.1f polls per piece\n",
        (unsigned long long)lane->polls,
        (unsigned long long)lane->polls_empty,
        shadowspill_lane_count(&lane->base.chunks) != 0U
            ? (double)lane->polls
              / (double)shadowspill_lane_count(&lane->base.chunks)
            : 0.0
    );
    const uint64_t timed = atomic_load_explicit(
        &lane->stat_timed, memory_order_relaxed
    );
    if (timed != 0U) {
        fprintf(
            stderr,
            "shadowspill network:   posted to completion -- %llu pieces, "
            "%.2f us mean, %.2f us longest\n",
            (unsigned long long)timed,
            (double)atomic_load_explicit(
                &lane->stat_completion_micros, memory_order_relaxed
            ) / (double)timed,
            (double)atomic_load_explicit(
                &lane->stat_longest_micros, memory_order_relaxed
            )
        );
    }
}
