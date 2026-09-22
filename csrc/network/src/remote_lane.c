/* What moves bytes between a pool on this machine and a pool on another. */

/* pthread_setname_np is a GNU extension; the define has to precede the first
   system header. */
#define _GNU_SOURCE

#include "remote_lane_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * THE DIRECT PATH, WHICH IS THE NORM.
 *
 * A transfer is one end in the pool this lane connects and the other end in a
 * pool on the peer. The NIC reads or writes the peer's memory itself, and it
 * reads or writes ours the same way: the pool is registered with the NIC once,
 * at create, and every transfer is posted straight from the pool's memory in
 * as few pieces as the port will carry. No byte passes through the host on
 * the way, and no device copy is involved at all. Where the NIC cannot
 * address the pool -- device memory on a platform that exports neither a
 * dma-buf nor peer memory -- the lane stages through a host ring instead,
 * which is the fallback in remote_lane_staging.c and changes nothing here but
 * where a piece is posted from and what must be true before it is.
 *
 * WHO DOES WHAT. Everything device-side is enqueued at dispatch, on the
 * route's stream, by the thread that called `copy` -- the runtime's worker --
 * behind the waits the runtime put there: under staging the copies across the
 * ring, on the direct path a single store of the transfer's number into the
 * gate word, which is how the thread learns that the route stream has passed
 * the transfer's dependencies without the thread ever asking the driver. The
 * lane's own thread does the rest, and its whole part is memory: it posts to
 * the NIC when the words say it may, reaps completions, retires pieces in
 * order, and stores the NIC's word. It makes no device call, for the reason
 * the header gives.
 *
 * THE OBLIGATION, AND HOW THIS LANE MEETS IT. A lane makes the event it was
 * given complete when the bytes have landed. `signal` enqueues, on the route
 * stream, a value wait on the NIC's word for everything planned so far and
 * records the event behind it; the thread's store is what lets it pass, and
 * everything downstream sees an ordinary event.
 *
 * WHY A QUEUE AND A THREAD. `copy` must return promptly, or there is no
 * overlap and the whole exercise is pointless. So `copy` appends to one FIFO
 * and returns, and one thread drains it in order: posts what the queue pair
 * has room for, reaps completions, retires pieces in order, and reports. It
 * watches rather than sleeps, by default: a condition variable costs about
 * twenty microseconds to wake from and the NIC answers a small transfer in a
 * few.
 */

/* ------------------------------------------------------------ transfers */

Work *remote_lane_owner_of_chunk(RemoteLane *lane, uint64_t chunk) {
    for (uint32_t step = 0U; step < WORK_SLOTS; ++step) {
        Work *const candidate =
            &lane->work_ring[(lane->retired + step) % WORK_SLOTS];
        if (atomic_load_explicit(&candidate->handle, memory_order_acquire)
                != 0U &&
            chunk >= candidate->first_chunk &&
            chunk < candidate->first_chunk + candidate->chunks) {
            return candidate;
        }
    }
    return NULL;
}

/* How many pieces may be on the wire at once: as many as the queue pair was
   built to hold on the direct path, and one slot's worth each when staging. */
static uint32_t outstanding_limit(const RemoteLane *lane) {
    if (lane->stages) {
        return lane->tuning.ring_slots;
    }
    return lane->tuning.send_depth < PIECES_IN_FLIGHT
        ? lane->tuning.send_depth : PIECES_IN_FLIGHT;
}

/* The NIC has finished every piece below `reached`. Releases whatever waits
   on it: the route stream's next staged copy, and the event behind `signal`'s
   wait. Monotonic, because pieces retire in order. */
static void report_nic(RemoteLane *lane, uint64_t reached) {
    atomic_store_explicit(
        (_Atomic uint64_t *)&lane->signal_host[SIGNAL_NIC], reached,
        memory_order_release
    );
}

/*
 * May piece `chunk` of `work` go to the NIC? Staged: when the device has
 * done its part for the slot. Direct: when the route stream has stored this
 * transfer's number into the gate, which it does only after the waits the
 * runtime enqueued for the transfer -- the one ordering obligation a post
 * that touches the pool directly has. Reads of words, never device calls.
 */
static int may_post(const RemoteLane *lane, const Work *work, uint64_t chunk) {
    if (lane->stages) {
        return remote_lane_stage_ready(lane, work, chunk);
    }
    return remote_lane_word(lane, SIGNAL_GATE)
        >= atomic_load_explicit(&work->handle, memory_order_relaxed);
}

/*
 * Post one piece of one transfer, whose lane-global number is `chunk`.
 *
 * Direct: from the pool's memory, at the piece's offset, with the pool's
 * key. Staged: from the piece's ring slot. Returns 1 when the piece went to
 * the NIC, 0 when it may not go yet -- the caller retires and asks again --
 * and -1 when the post failed.
 */
static int post_one_piece(RemoteLane *lane, Work *work, uint64_t chunk) {
    if (!may_post(lane, work, chunk)) {
        return 0;
    }
    remote_lane_stamp(lane, work, TIMELINE_READY);
    const uint64_t offset = (chunk - work->first_chunk) * lane->piece_bytes;
    const uint64_t bytes = work->bytes - offset < lane->piece_bytes
        ? work->bytes - offset : lane->piece_bytes;
    void *const address = lane->stages
        ? remote_lane_stage_slot(lane, chunk)
        : (char *)work->local + offset;
    const struct ibv_mr *const registration = lane->stages
        ? lane->ring_registration : lane->pool_registration;
    struct ibv_sge element = {
        .addr = (uint64_t)(uintptr_t)address,
        .length = (uint32_t)bytes,
        .lkey = registration->lkey,
    };
    struct ibv_send_wr request = {
        /* The piece's lane-global number, which names the transfer it belongs
           to as well as its place in the pipeline: a completion has to say
           which transfer it finished. */
        .wr_id = chunk,
        .sg_list = &element,
        .num_sge = 1,
        .opcode = work->to_remote ? IBV_WR_RDMA_WRITE : IBV_WR_RDMA_READ,
        .send_flags = IBV_SEND_SIGNALED,
        .wr = {.rdma = {
            .remote_addr = lane->region->address + work->remote_offset + offset,
            .rkey = lane->region->key,
        }},
    };
    struct ibv_send_wr *bad = NULL;
    /* This lane's own queue pair, never another's: two lanes polling one
       completion queue take each other's completions. */
    if (ibv_post_send(
            lane->region->endpoint.queue_pairs[lane->queue_pair], &request, &bad
        ) != 0) {
        return -1;
    }
    shadowspill_lane_counted(&lane->base.chunks, 1U);
    if (lane->measuring) {
        lane->chunk_posted_at[chunk % PIECES_IN_FLIGHT] =
            remote_lane_seconds_now();
        remote_lane_stamp(lane, work, TIMELINE_POSTED);
    }
    ++lane->chunk_posted;
    return 1;
}

/*
 * Take whatever completions have arrived and retire what that makes retirable,
 * in order. **Never waits**: the thread alternates between posting and
 * retiring, and whichever is ready first is what it does. Completions may
 * arrive out of order, so `chunk_landed` records what has arrived and the
 * oldest piece is retired only when it is among them.
 */
static int collect_and_retire(RemoteLane *lane, uint64_t *retired) {
    struct ibv_cq *const queue =
        lane->region->endpoint.completion_queues[lane->queue_pair];
    for (;;) {
        struct ibv_wc completion;
        const int taken = ibv_poll_cq(queue, 1, &completion);
        if (lane->measuring) {
            ++lane->polls;
            if (taken == 0) {
                ++lane->polls_empty;
            }
        }
        if (taken < 0) {
            return -1;
        }
        if (taken == 0) {
            break;
        }
        if (completion.status != IBV_WC_SUCCESS) {
            fprintf(
                stderr, "shadowspill network: transfer failed (%s)\n",
                ibv_wc_status_str(completion.status)
            );
            return -1;
        }
        lane->chunk_landed[completion.wr_id % PIECES_IN_FLIGHT] = 1U;
        if (lane->measuring) {
            remote_lane_stamp(
                lane, remote_lane_owner_of_chunk(lane, completion.wr_id),
                TIMELINE_COMPLETED
            );
            /* Posted to observed, which is what this lane can see: not time on
               the wire, and separating the two is the point. */
            const double posted =
                lane->chunk_posted_at[completion.wr_id % PIECES_IN_FLIGHT];
            if (posted != 0.0) {
                const uint64_t micros = (uint64_t)(
                    (remote_lane_seconds_now() - posted) * 1e6
                );
                (void)atomic_fetch_add_explicit(
                    &lane->stat_completion_micros, micros, memory_order_relaxed
                );
                uint64_t longest = atomic_load_explicit(
                    &lane->stat_longest_micros, memory_order_relaxed
                );
                while (micros > longest &&
                       !atomic_compare_exchange_weak_explicit(
                           &lane->stat_longest_micros, &longest, micros,
                           memory_order_relaxed, memory_order_relaxed
                       )) {
                }
                (void)atomic_fetch_add_explicit(
                    &lane->stat_timed, 1U, memory_order_relaxed
                );
            }
        }
    }
    while (lane->chunk_retired < lane->chunk_posted &&
           lane->chunk_landed[lane->chunk_retired % PIECES_IN_FLIGHT]) {
        lane->chunk_landed[lane->chunk_retired % PIECES_IN_FLIGHT] = 0U;
        ++lane->chunk_retired;
        ++*retired;
    }
    if (*retired != 0U) {
        report_nic(lane, lane->chunk_retired);
        if (lane->measuring) {
            remote_lane_stamp(
                lane, remote_lane_owner_of_chunk(lane, lane->chunk_retired - 1U),
                TIMELINE_REPORTED
            );
        }
    }
    return 0;
}

/*
 * Post whatever the queue pair, and under staging the ring and the device,
 * allow. A transfer is started when its last piece is posted; the counts of
 * transfers and of pieces advance independently, which is the whole of what
 * "the pipeline spans transfers" means.
 */
static int post_what_it_can(
    RemoteLane *lane, uint64_t accepted, uint64_t *started, uint64_t *posted
) {
    while (*started < accepted) {
        Work *const work = &lane->work_ring[*started % WORK_SLOTS];
        remote_lane_stamp(lane, work, TIMELINE_SEEN);
        if (lane->chunk_posted - lane->chunk_retired >= outstanding_limit(lane)) {
            break;
        }
        const int first = lane->chunk_posted == work->first_chunk ? 1 : 0;
        const int sent = post_one_piece(lane, work, lane->chunk_posted);
        if (sent < 0) {
            return -1;
        }
        if (sent == 0) {
            /* The device, or the route stream, is behind. Retire, which is
               what moves them. */
            break;
        }
        if (first && work->traced) {
            /* Bytes start moving here, not when the transfer was accepted:
               everything before this was the dependency it was given. */
            work->started_host_ns = remote_lane_monotonic_ns();
        }
        ++*posted;
        if (lane->chunk_posted == work->first_chunk + work->chunks) {
            ++*started;
        }
    }
    return 0;
}

/*
 * Everything the thread does between two looks at its queue: post what it
 * can, reap what has arrived, and again while either made progress. Returns
 * once it has made progress and then found nothing more to do, so the caller
 * can notice newly accepted work; the lock is therefore off the per-piece
 * path. A round that does neither pauses and tries again, because both are
 * questions whose answer arrives from hardware.
 */
static int pump(RemoteLane *lane, uint64_t accepted, uint64_t *started) {
    int progressed = 0;
    for (;;) {
        uint64_t posted = 0U;
        uint64_t retired = 0U;
        if (post_what_it_can(lane, accepted, started, &posted) != 0) {
            return -1;
        }
        if (lane->chunk_posted != lane->chunk_retired &&
            collect_and_retire(lane, &retired) != 0) {
            return -1;
        }
        if (posted != 0U || retired != 0U) {
            progressed = 1;
            continue;
        }
        if (progressed || lane->chunk_posted == lane->chunk_retired) {
            return 0;
        }
#if defined(__x86_64__) || defined(__i386__)
        __builtin_ia32_pause();
#elif defined(__aarch64__)
        __asm__ volatile("yield");
#endif
    }
}

/*
 * A failure anywhere reports **every piece planned**, in all three words: the
 * route stream's staged copies and `signal`'s event wait on the NIC's, the
 * thread reads the other two, and a failure that left any unstored would be
 * a step that never ends rather than one that fails. The latch is what makes
 * it fail.
 */
static void fail_everything(RemoteLane *lane) {
    shadowspill_lane_counted(&lane->base.failures, 1U);
    pthread_mutex_lock(&lane->lock);
    const uint64_t planned = lane->chunks_planned;
    const uint64_t accepted = lane->accepted;
    pthread_mutex_unlock(&lane->lock);
    report_nic(lane, planned);
    atomic_store_explicit(
        (_Atomic uint64_t *)&lane->signal_host[SIGNAL_DEVICE], planned,
        memory_order_release
    );
    atomic_store_explicit(
        (_Atomic uint64_t *)&lane->signal_host[SIGNAL_GATE], accepted,
        memory_order_release
    );
    atomic_store_explicit(&lane->failed, 1U, memory_order_release);
    shadowspill_lane_latch_failure(
        lane->base.runtime, SHADOWSPILL_STATUS_BACKEND_FAILURE,
        SHADOWSPILL_FAILURE_REASON_TRANSFER_REJECTED
    );
}

/* Watch for work before blocking for it. Forever by default -- a core per
   lane while the runtime is open, taken deliberately; bounded again by
   `SHADOWSPILL_NETWORK_SPIN_NANOSECONDS`, and zero blocks at once. */
static void spin_briefly(RemoteLane *lane) {
    if (lane->tuning.spin_nanoseconds == 0U) {
        return;
    }
    const int forever =
        lane->tuning.spin_nanoseconds == SHADOWSPILL_NETWORK_SPIN_FOREVER;
    const double deadline = forever
        ? 0.0
        : remote_lane_seconds_now()
          + (double)lane->tuning.spin_nanoseconds * 1e-9;
    while (atomic_load_explicit(&lane->queued, memory_order_acquire) == 0ULL &&
           atomic_load_explicit(&lane->stopping, memory_order_acquire) == 0U) {
        if (!forever && remote_lane_seconds_now() >= deadline) {
            return;
        }
#if defined(__x86_64__) || defined(__i386__)
        __builtin_ia32_pause();
#elif defined(__aarch64__)
        __asm__ volatile("yield");
#endif
    }
}

/* Named like the worker's thread, so `ps -T` and a profiler say which lane a
   core belongs to: the OS name, and the backend's profiler name where it has
   one. Three letters after the prefix is what the 16-byte cap leaves; a
   pinned-host pair shares the names of the device pair, since a process
   rarely runs both. */
static void name_lane_thread(const RemoteLane *lane) {
    const char *const name = lane->base.to_kind == SHADOWSPILL_POOL_REMOTE
        ? "shadowspill.evc" : "shadowspill.fch";
    (void)pthread_setname_np(pthread_self(), name);
    const ShadowSpillBackend *const backend = lane->base.backend;
    if (backend->name_thread != NULL) {
        backend->name_thread(backend->state, name);
    }
}

static void *lane_thread(void *argument) {
    RemoteLane *lane = argument;
    name_lane_thread(lane);
    for (;;) {
        /* Only when the pipeline is empty: with pieces outstanding the thread
           already has work. */
        if (lane->chunk_posted == lane->chunk_retired) {
            spin_briefly(lane);
        }
        pthread_mutex_lock(&lane->lock);
        while (lane->started == lane->accepted &&
               lane->chunk_posted == lane->chunk_retired &&
               atomic_load_explicit(&lane->stopping, memory_order_acquire) == 0U) {
            pthread_cond_wait(&lane->work_ready, &lane->lock);
        }
        const uint64_t accepted = lane->accepted;
        uint64_t started = lane->started;
        if (started == accepted && lane->chunk_posted == lane->chunk_retired) {
            pthread_mutex_unlock(&lane->lock);
            return NULL;
        }
        pthread_mutex_unlock(&lane->lock);

        const int failed = pump(lane, accepted, &started) != 0;
        if (failed) {
            fail_everything(lane);
        }

        pthread_mutex_lock(&lane->lock);
        const uint64_t retired_before = lane->retired;
        if (started != lane->started) {
            (void)atomic_fetch_sub_explicit(
                &lane->queued, started - lane->started, memory_order_acq_rel
            );
            lane->started = started;
        }
        /* A transfer is retired when its last piece is, which is the moment
           its bytes are where the NIC put them. */
        while (lane->retired < lane->started) {
            Work *const oldest = &lane->work_ring[lane->retired % WORK_SLOTS];
            if (!failed &&
                lane->chunk_retired < oldest->first_chunk + oldest->chunks) {
                break;
            }
            remote_lane_stamp(lane, oldest, TIMELINE_RETIRED);
            if (oldest->traced) {
                /* A failed transfer is retired here too and keeps whatever it
                   had, so a trace shows where it stopped. */
                oldest->finished_host_ns = remote_lane_monotonic_ns();
            }
            ++lane->retired;
        }
        if (failed) {
            /* Nothing will retire these; let every waiter go. */
            lane->retired = lane->started = lane->accepted;
            lane->chunk_retired = lane->chunk_posted;
        }
        /* Broadcast when a transfer retired -- what `synchronize` waits for
           and what frees a slot for `claim_slot` -- and not per piece. */
        if (lane->retired != retired_before) {
            pthread_cond_broadcast(&lane->drained);
        }
        pthread_mutex_unlock(&lane->lock);
    }
}

/* -------------------------------------------------------- the operations */

/* Take the next slot, waiting if every one is still in flight. The thread
   never waits on the caller, so the wait terminates; in practice a route
   cannot get 256 transfers ahead of a lane that is draining them. Returns
   with the lock held, so the caller fills the slot before anything sees it. */
static Work *claim_slot(RemoteLane *lane, uint64_t *handle) {
    pthread_mutex_lock(&lane->lock);
    while (lane->accepted - lane->retired == WORK_SLOTS) {
        pthread_cond_wait(&lane->drained, &lane->lock);
    }
    Work *work = &lane->work_ring[lane->accepted % WORK_SLOTS];
    /* Invalidated before a field is overwritten, so a reader still holding
       the previous occupant's handle fails its check rather than reading a
       half-built record. `publish_slot` stores the new handle last. */
    atomic_store_explicit(&work->handle, 0U, memory_order_release);
    *handle = lane->accepted + 1U;
    return work;
}

/* Publish the slot the caller has filled, and wake the thread. Signalled
   after unlocking, so the thread does not wake only to block on the mutex. */
static void publish_slot(RemoteLane *lane, Work *work, uint64_t handle) {
    remote_lane_stamp(lane, work, TIMELINE_PUBLISHED);
    atomic_store_explicit(&work->handle, handle, memory_order_release);
    (void)atomic_fetch_add_explicit(&lane->queued, 1ULL, memory_order_release);
    ++lane->accepted;
    pthread_mutex_unlock(&lane->lock);
    pthread_cond_signal(&lane->work_ready);
}

/*
 * Order this lane's next transfer behind `event`, on the route's stream:
 * everything `copy` enqueues for the transfer goes behind it there. Never
 * returns 1: this lane can always enqueue a device-side wait.
 */
static int remote_wait(ShadowSpillLane *lane, ShadowSpillBackendEvent event) {
    shadowspill_lane_counted(&lane->waits, 1U);
    return lane->backend->wait_event(lane->backend->state, lane->stream, event)
        == 0 ? 0 : -1;
}

/*
 * What this transfer is, before anything is issued. Direction is the lane's,
 * from its kinds; the remote end is an address in the region's reservation,
 * turned back into an offset; the piece count is decided here, once, so the
 * thread never recomputes it. Under the lane's lock, because the piece
 * numbering is shared with the thread.
 */
static int plan_transfer(
    RemoteLane *lane,
    Work *work,
    void *destination,
    const void *source,
    uint64_t bytes
) {
    const int to_remote = lane->base.to_kind == SHADOWSPILL_POOL_REMOTE;
    const char *const remote = to_remote ? (const char *)destination : (const char *)source;
    const char *const reservation = lane->region->reservation;
    if (remote < reservation ||
        (uint64_t)(remote - reservation) + bytes > lane->region->capacity) {
        /* Not an address in the pool this lane reaches: the route resolved to
           the wrong lane, or the caller handed over a foreign pointer. */
        return -1;
    }
    work->to_remote = (uint8_t)to_remote;
    work->local = to_remote ? (void *)(uintptr_t)source : destination;
    work->remote_offset = (uint64_t)(remote - reservation);
    work->bytes = bytes;
    work->chunks = bytes == 0U
        ? 1U
        : (uint32_t)((bytes + lane->piece_bytes - 1U) / lane->piece_bytes);
    work->first_chunk = lane->chunks_planned;
    lane->chunks_planned += work->chunks;
    work->traced =
        shadowspill_lane_trace_active(lane->base.runtime) != 0 ? 1U : 0U;
    work->issued_host_ns = work->traced ? remote_lane_monotonic_ns() : 0U;
    work->started_host_ns = 0U;
    work->finished_host_ns = 0U;
    return 0;
}

/*
 * Plan it, hand it to the thread, then give the route stream the device's
 * side. Nothing is waited for here: this runs on the runtime's worker.
 *
 * The thread is given the transfer *before* the device side is enqueued, and
 * the order matters: every value wait enqueued below is satisfied by this
 * lane's thread, and a driver asked to enqueue behind a wait nobody can
 * satisfy yet may hold the caller. Handing the work over first is safe
 * because the waits are greater-or-equal -- one enqueued after its value was
 * stored passes at once -- and stream order is unchanged. The plan is copied
 * out first because the slot belongs to the thread the moment it is
 * published.
 */
static int remote_copy(
    ShadowSpillLane *lane,
    void *destination,
    const void *source,
    uint64_t bytes,
    uint64_t *handle
) {
    RemoteLane *self = (RemoteLane *)lane;
    const double entered = self->measuring ? remote_lane_seconds_now() : 0.0;
    *handle = 0U;
    shadowspill_lane_counted(&lane->copies, 1U);
    shadowspill_lane_counted(&lane->bytes, bytes);

    uint64_t claimed = 0U;
    Work *work = claim_slot(self, &claimed);
    work->timeline_row = -1;
    if (plan_transfer(self, work, destination, source, bytes) != 0) {
        /* Never published, so `accepted` did not move and the next claim takes
           this same slot; its handle is already 0. */
        pthread_mutex_unlock(&self->lock);
        shadowspill_lane_counted(&lane->failures, 1U);
        return -1;
    }
    if (self->measuring) {
        remote_lane_capture_timeline(self, work, entered);
    }
    const Work plan = *work;
    publish_slot(self, work, claimed);
    *handle = claimed;

    const ShadowSpillBackend *const backend = lane->backend;
    int enqueued;
    if (self->stages) {
        enqueued = remote_lane_stage_enqueue(self, &plan);
    } else {
        /* The gate: the transfer's number, stored by the route stream once it
           has passed whatever the runtime enqueued ahead of this. The thread
           posts nothing of the transfer before it reads it. */
        enqueued = backend->write_value(
            backend->state, lane->stream, self->signals, SIGNAL_GATE, claimed
        );
    }
    if (enqueued != 0) {
        shadowspill_lane_counted(&lane->failures, 1U);
        return -1;
    }
    remote_lane_stamp(self, &plan, TIMELINE_ENQUEUED);
    return 0;
}

/* An instant this lane observed, placed on the trace origin's axis. Zero is
   "never reached this point". */
static uint64_t instant_on_origin(ShadowSpillLane *lane, uint64_t monotonic_ns) {
    return monotonic_ns == 0U
        ? SHADOWSPILL_LANE_NO_TIME
        : shadowspill_lane_origin_instant(lane->runtime, monotonic_ns);
}

/* What one transfer did. Read without the lock, the handle checked on both
   sides of the fields: a `copy` claiming the slot rewrites the handle first,
   so a match before and after means nothing overwrote what was read. */
static int remote_transfer(
    ShadowSpillLane *lane, uint64_t handle, ShadowSpillLaneTransfer *transfer
) {
    RemoteLane *self = (RemoteLane *)lane;
    if (handle == 0U) {
        return -1;
    }
    const Work *work = &self->work_ring[(handle - 1U) % WORK_SLOTS];
    if (atomic_load_explicit(&work->handle, memory_order_acquire) != handle) {
        return -1;
    }
    const ShadowSpillLaneTransfer read = {
        .issued_at_nanoseconds = instant_on_origin(lane, work->issued_host_ns),
        .started_at_nanoseconds = instant_on_origin(lane, work->started_host_ns),
        .finished_at_nanoseconds =
            instant_on_origin(lane, work->finished_host_ns),
        .bytes = work->bytes,
        .chunks = work->chunks,
    };
    if (atomic_load_explicit(&work->handle, memory_order_acquire) != handle) {
        return -1;
    }
    *transfer = read;
    return 0;
}

/*
 * Make `event` complete once everything queued before it has landed.
 *
 * Two calls on the route stream, the same on either path: a value wait on
 * the last report this lane's thread will make for everything planned so
 * far, and the record behind it. Where the transfer had device work, that
 * work is already ahead of this on the same stream. The wait is enqueued
 * here, on the dispatching thread, rather than left to the lane's thread to
 * arrange -- which could not, see the header -- and it covers every piece
 * planned so far, so the handle says nothing this does not already know.
 */
static int remote_signal(
    ShadowSpillLane *lane, uint64_t handle, ShadowSpillBackendEvent event
) {
    (void)handle;
    RemoteLane *self = (RemoteLane *)lane;
    const ShadowSpillBackend *const backend = lane->backend;
    shadowspill_lane_counted(&lane->signals, 1U);
    pthread_mutex_lock(&self->lock);
    const uint64_t planned = self->chunks_planned;
    pthread_mutex_unlock(&self->lock);
    if (planned != 0U && backend->wait_value(
            backend->state, lane->stream, self->signals, SIGNAL_NIC, planned
        ) != 0) {
        return -1;
    }
    return backend->record_event(backend->state, event, lane->stream) == 0
        ? 0 : -1;
}

/*
 * Block until everything issued has landed: every transfer retired by the
 * thread, then the route stream synchronized, which covers the device's
 * side and the events. Watches for the drain before blocking on it, for the
 * reason the thread does: a condition variable costs about twenty
 * microseconds to wake from, and a small transfer finishes in fewer.
 */
static int remote_synchronize(ShadowSpillLane *base_lane) {
    RemoteLane *lane = (RemoteLane *)base_lane;
    const Work *captured = NULL;
    if (lane->measuring) {
        pthread_mutex_lock(&lane->lock);
        captured = remote_lane_captured_transfer(lane);
        pthread_mutex_unlock(&lane->lock);
        remote_lane_stamp(lane, captured, TIMELINE_SYNC_ENTERED);
    }
    if (lane->tuning.spin_nanoseconds != 0U) {
        const int forever =
            lane->tuning.spin_nanoseconds == SHADOWSPILL_NETWORK_SPIN_FOREVER;
        const double deadline = forever
            ? 0.0
            : remote_lane_seconds_now()
              + (double)lane->tuning.spin_nanoseconds * 1e-9;
        for (;;) {
            pthread_mutex_lock(&lane->lock);
            const int drained = lane->retired == lane->accepted;
            pthread_mutex_unlock(&lane->lock);
            if (drained ||
                (!forever && remote_lane_seconds_now() >= deadline)) {
                break;
            }
#if defined(__x86_64__) || defined(__i386__)
            __builtin_ia32_pause();
#elif defined(__aarch64__)
            __asm__ volatile("yield");
#endif
        }
    }
    pthread_mutex_lock(&lane->lock);
    while (lane->retired != lane->accepted) {
        pthread_cond_wait(&lane->drained, &lane->lock);
    }
    remote_lane_stamp(lane, captured, TIMELINE_SYNC_DRAINED);
    const uint64_t accepted = lane->accepted;
    if (lane->measuring) {
        remote_lane_release_batch_row(lane, accepted);
    }
    pthread_mutex_unlock(&lane->lock);
    if (atomic_load_explicit(&lane->failed, memory_order_acquire) != 0U) {
        return -1;
    }
    if (captured != NULL) {
        remote_lane_watch_device(lane, captured);
    }
    const ShadowSpillBackend *const backend = lane->base.backend;
    const int status =
        backend->synchronize_stream(backend->state, lane->base.stream);
    remote_lane_stamp(lane, captured, TIMELINE_SYNC_RETURNED);
    if (lane->measuring) {
        lane->synchronized_through = accepted;
    }
    return status == 0 ? 0 : -1;
}

/* What a transfer cost this lane, without locking: every field is atomic and
   a reader wants a recent picture, not a consistent instant. */
static int remote_timing(
    const ShadowSpillLane *lane, ShadowSpillLaneTiming *timing
) {
    if (lane == NULL || timing == NULL) {
        return -1;
    }
    const RemoteLane *self = (const RemoteLane *)lane;
    if (atomic_load_explicit(&self->stat_timed, memory_order_relaxed) == 0U) {
        /* No clock was read on the transfer path. Refusing is how a reader
           learns that, rather than seeing a zero it cannot interpret. */
        return -1;
    }
    timing->posted_to_completion_seconds =
        (double)atomic_load_explicit(
            &self->stat_completion_micros, memory_order_relaxed
        ) / 1e6;
    timing->longest_completion_seconds =
        (double)atomic_load_explicit(
            &self->stat_longest_micros, memory_order_relaxed
        ) / 1e6;
    return 0;
}

static void remote_destroy(ShadowSpillLane *base_lane) {
    RemoteLane *lane = (RemoteLane *)base_lane;
    if (lane == NULL) {
        return;
    }
    if (lane->measuring) {
        /* On stderr, because a lane has no other channel out and the reader is
           the person who set the variable. */
        remote_lane_report(lane);
    }
    if (lane->thread_started) {
        atomic_store_explicit(&lane->stopping, 1U, memory_order_release);
        pthread_mutex_lock(&lane->lock);
        pthread_cond_broadcast(&lane->work_ready);
        pthread_mutex_unlock(&lane->lock);
        (void)pthread_join(lane->thread, NULL);
    }
    /* The pool's and the ring's registrations are the region's and go with
       it; the ring itself is this lane's. */
    remote_lane_stage_destroy(lane);
    if (lane->signal_host != NULL) {
        (void)lane->base.backend->free_signals(
            lane->base.backend->state, lane->signals
        );
    }
    pthread_cond_destroy(&lane->drained);
    pthread_cond_destroy(&lane->work_ready);
    pthread_mutex_destroy(&lane->lock);
    free(lane);
}

static const ShadowSpillLaneOperations remote_operations = {
    .wait = remote_wait,
    .copy = remote_copy,
    .signal = remote_signal,
    .synchronize = remote_synchronize,
    .transfer = remote_transfer,
    .destroy = remote_destroy,
    .timing = remote_timing,
};

static const char *kind_name(uint8_t kind) {
    switch (kind) {
    case SHADOWSPILL_POOL_DEVICE:
        return "device";
    case SHADOWSPILL_POOL_PINNED_HOST:
        return "pinned host";
    default:
        return "local";
    }
}

/*
 * Made once per route, and this is where the lane probes: which region the
 * route reaches, a queue pair of its own from it, and whether the NIC can
 * address the local pool -- a registration of the whole pool, through the
 * backend's dma-buf where it exports one and plainly otherwise, kept by the
 * region and shared with the lane serving the other direction. If it can, the
 * direct path; if it cannot, the ring. Then the three words, and the thread.
 */
static int remote_create(
    const ShadowSpillLane *base, void *configuration, ShadowSpillLane **created
) {
    (void)configuration;
    if (base == NULL || created == NULL) {
        return -1;
    }
    RemoteLane *lane = calloc(1U, sizeof(*lane));
    if (lane == NULL) {
        return -1;
    }
    /* Copied in first, so everything below -- and `remote_destroy` on every
       failure path below -- can reach the backend through it. */
    lane->base = *base;
    const ShadowSpillBackend *const backend = base->backend;
    shadowspill_network_tuning_read(&lane->tuning);
    lane->measuring = getenv("SHADOWSPILL_NETWORK_MEASURE") != NULL;
    for (uint32_t index = 0U; index < WORK_SLOTS; ++index) {
        /* Not captured until the timeline says so; 0 would be a row. */
        lane->work_ring[index].timeline_row = -1;
    }
    atomic_init(&lane->stopping, 0U);
    atomic_init(&lane->failed, 0U);
    atomic_init(&lane->queued, 0ULL);
    if (pthread_mutex_init(&lane->lock, NULL) != 0 ||
        pthread_cond_init(&lane->work_ready, NULL) != 0 ||
        pthread_cond_init(&lane->drained, NULL) != 0) {
        free(lane);
        return -1;
    }

    const int to_remote = base->to_kind == SHADOWSPILL_POOL_REMOTE;
    const ShadowSpillLaneRange remote = to_remote ? base->to_range : base->from_range;
    const ShadowSpillLaneRange local = to_remote ? base->from_range : base->to_range;
    const uint8_t local_kind = to_remote ? base->from_kind : base->to_kind;
    lane->region = shadowspill_remote_region_for(remote.address);
    if (lane->region == NULL) {
        fprintf(
            stderr,
            "shadowspill network: this route's remote pool is not a region "
            "this process holds\n"
        );
        remote_destroy(&lane->base);
        return -1;
    }
    lane->queue_pair = shadowspill_remote_region_claim_queue_pair(lane->region);
    if (lane->queue_pair < 0) {
        fprintf(
            stderr,
            "shadowspill network: no queue pair left for this lane; raise "
            "SHADOWSPILL_NETWORK_QUEUE_PAIRS above %u\n",
            lane->region->endpoint.queue_pair_count
        );
        remote_destroy(&lane->base);
        return -1;
    }
    /* Device memory reaches the NIC through the backend, where the backend
       has a way; host memory reaches it plainly. */
    lane->pool_registration = shadowspill_remote_region_register_local(
        lane->region,
        local_kind == SHADOWSPILL_POOL_DEVICE ? backend : NULL,
        local.address, local.bytes
    );
    if (lane->pool_registration != NULL) {
        lane->stages = 0U;
        lane->piece_bytes = lane->region->endpoint.max_message_bytes != 0U
            ? lane->region->endpoint.max_message_bytes
            : lane->tuning.chunk_bytes;
    } else {
        fprintf(
            stderr,
            "shadowspill network: the NIC cannot address the %s pool; "
            "staging through a host ring of %u x %llu bytes\n",
            kind_name(local_kind), lane->tuning.ring_slots,
            (unsigned long long)lane->tuning.chunk_bytes
        );
        lane->stages = 1U;
        lane->piece_bytes = lane->tuning.chunk_bytes;
        if (remote_lane_stage_create(lane) != 0) {
            remote_destroy(&lane->base);
            return -1;
        }
    }
    /* Three words, once per lane: what the NIC has finished, what the device
       has, and which transfer the route stream has passed. */
    if (backend->allocate_signals(
            backend->state, SIGNAL_WORDS, &lane->signals, &lane->signal_host
        ) != 0) {
        remote_destroy(&lane->base);
        return -1;
    }
    for (uint32_t index = 0U; index < SIGNAL_WORDS; ++index) {
        lane->signal_host[index] = 0U;
    }
    if (pthread_create(&lane->thread, NULL, lane_thread, lane) != 0) {
        remote_destroy(&lane->base);
        return -1;
    }
    lane->thread_started = 1U;
    *created = &lane->base;
    return 0;
}

/*
 * Both directions, for each local pool the NIC may address or stage for --
 * device memory, and pinned host memory, which the NIC always can and which
 * is how the direct path runs on a machine whose device cannot be reached.
 * Registered by kind pair like every other lane, so the runtime finds this
 * one exactly as it finds the built-in.
 */
const ShadowSpillLaneDescription shadowspill_remote_lanes[4] = {
    {
        .from_kind = SHADOWSPILL_POOL_REMOTE,
        .to_kind = SHADOWSPILL_POOL_DEVICE,
        .operations = &remote_operations,
        .create = remote_create,
        .configuration = NULL,
    },
    {
        .from_kind = SHADOWSPILL_POOL_DEVICE,
        .to_kind = SHADOWSPILL_POOL_REMOTE,
        .operations = &remote_operations,
        .create = remote_create,
        .configuration = NULL,
    },
    {
        .from_kind = SHADOWSPILL_POOL_REMOTE,
        .to_kind = SHADOWSPILL_POOL_PINNED_HOST,
        .operations = &remote_operations,
        .create = remote_create,
        .configuration = NULL,
    },
    {
        .from_kind = SHADOWSPILL_POOL_PINNED_HOST,
        .to_kind = SHADOWSPILL_POOL_REMOTE,
        .operations = &remote_operations,
        .create = remote_create,
        .configuration = NULL,
    },
};
