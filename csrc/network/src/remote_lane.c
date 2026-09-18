/* What moves bytes between a device pool and a pool on another machine. */

/* MAP_ANONYMOUS is not in the strict ISO C11 the tree compiles as. */
#define _DEFAULT_SOURCE

#include "../internal.h"

#include <pthread.h>
#include <stdatomic.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

/*
 * THE OBLIGATION, AND HOW THIS LANE MEETS IT.
 *
 * A lane makes the event it was given complete when the bytes have landed, and
 * the runtime does not drive it. This one cannot let a stream order that event
 * behind a copy, because its bytes do not move on a stream: the NIC completes
 * on its own schedule.
 *
 * So it uses the backend's value wait, on **its own** stream:
 *
 *   signal(event)  ->  wait_value(lane stream, signals, 0, chunks so far)
 *                      record_event(event, lane stream)
 *   its thread     ->  stores that count once everything queued before
 *                      the signal has landed, which releases the wait and
 *                      completes the event
 *
 * The route's stream is never written. The runtime waits on the *event*, an
 * ordinary backend event that happens to have been recorded on another stream
 * -- legal, and indistinguishable downstream from the built-in lane's. That is
 * why this needed no new entry in the lane table.
 *
 * WHY A QUEUE AND A THREAD. `copy` must return promptly, or there is no
 * overlap and the whole exercise is pointless. So `copy` and `signal` both
 * append to one FIFO and return, and one thread drains it in order. Order is
 * the reason it is a FIFO rather than a counter: a signal must store its
 * count after the copies queued before it and not before, and a list
 * preserves that without anyone reasoning about it.
 *
 * STAGING, AND WHY BOTH DIRECTIONS NEED IT. There is no `nvidia_peermem` here,
 * so the NIC cannot reach device memory and every byte passes through a host
 * ring the lane owns, registered with the NIC *and* with the backend -- the
 * second registration is what makes the device copy a real asynchronous DMA
 * rather than a bounce through the driver's own staging buffer.
 *
 *   fetch   NIC reads remote -> ring -> copy_host_to_device -> device
 *   evict   device -> copy_device_to_host -> ring -> NIC writes remote
 *
 * They are mirror images, and which side fills a slot inverts between them.
 * This path is deliberately not optimised: it exists because this hardware has
 * no GPUDirect, and a deployment that reaches the device directly never enters
 * it.
 */

#define RING_REGISTRATION_CACHE 4U

typedef struct RingRegistration {
    const ShadowSpillRemoteRegion *region;
    struct ibv_mr *registration;
    /*
     * The queue pair this lane took from that region, and its completion
     * queue. **Exclusive to this lane**: a fetch and an evict on one pool run
     * at the same time, and two lanes polling one completion queue take each
     * other's completions -- each recording what it took where the other
     * cannot find it, so both wait for something that already arrived.
     */
    int queue_pair;
} RingRegistration;

/*
 * Two words, and what each answers.
 *
 * `SIGNAL_NIC` is stored by this lane's thread and waited on by the stream:
 * "the NIC has finished chunk k". `SIGNAL_DEVICE` is stored by the *stream*
 * and polled by the thread: "the device has finished chunk k". Between them
 * they express the only cross-domain dependencies staging has, and they are
 * monotonic counters rather than per-chunk objects -- which is why two words
 * serve any number of transfers and no pool is needed anywhere.
 */
#define SIGNAL_NIC 0U
#define SIGNAL_DEVICE 1U
#define SIGNAL_WORDS 2U

/*
 * One transfer, as the lane's thread sees it: hardware work and nothing else.
 *
 * There is no signal kind any more. The completion event is ordered by the
 * stream, where the device work already is, so nothing has to be queued behind
 * the copies to store a word at the right moment.
 */
typedef struct Work {
    const ShadowSpillRemoteRegion *region;
    /* The end that is not remote: device memory the NIC reaches directly on
       the core path, and the device side of the staged copies otherwise. */
    void *local;
    uint64_t remote_offset;
    uint64_t bytes;
    uint8_t to_remote;
    /* Chunking is decided when the device side is enqueued, so the thread and
       the stream agree on it without recomputing. */
    uint32_t chunks;
    /*
      * This transfer's first chunk, numbered over the lane's whole life rather
      * than from zero each time -- which is what lets the ring hold one
      * transfer's chunks while the next is already posting into it.
      *
      * Both words count the same thing on this scale: chunk k of this transfer
      * is chunk `first_chunk + k`, and a side reports it finished by storing
      * `first_chunk + k + 1`. One numbering, so a reader never has to ask
      * which of the two a number belongs to.
      */
    uint64_t first_chunk;

    /*
     * Which transfer owns this slot, counted from 1 over the lane's life, and
     * 0 when nothing does. It is the handle `copy` hands back, and it is what
     * lets `transfer` tell "the slot still holds my transfer" from "the slot
     * has been taken by a later one" without any lock.
     */
    _Atomic uint64_t handle;
} Work;

/*
 * How many transfers the lane can hold.
 *
 * The queue used to be a linked list of `calloc`ed nodes, which put a host
 * allocation on the transfer path and freed it on the lane thread. A ring
 * removes both and gives every transfer a record that outlives it just long
 * enough for the runtime to read -- so the work queue and the per-transfer
 * report are one structure rather than two.
 *
 * Far larger than a route can have in flight. `copy` waits rather than fails
 * if it ever fills; see `claim_slot`.
 */
#define WORK_SLOTS 256U

typedef struct RemoteLane {
    /* Runtime, backend, the route's stream, this lane's two kinds, and the
       seven counters every lane keeps. Filled by the runtime and copied in by
       `create`; see <shadowspill/runtime/lane_base.h>. */
    ShadowSpillLane base;
    ShadowSpillNetworkTuning tuning;

    /* Host memory every byte passes through: registered with the backend once,
       and with a protection domain per region that uses it. */
    void *ring;
    uint64_t ring_bytes;
    RingRegistration registrations[RING_REGISTRATION_CACHE];
    uint32_t registration_count;

    /*
     * Whether this lane has to stage. Decided once, at create, and never asked
     * again on a transfer: it is a property of what the NIC can reach, not of
     * what is being moved.
     *
     * It is 1 on every box without peer memory, which is this one. The probe
     * that would find device memory registrable belongs at create beside the
     * endpoint, and when it lands this is the only line that changes -- the
     * core path is already written and already what `post_one_chunk` and
     * `outstanding_limit` select when this is 0.
     */
    uint8_t stages;

    /* One word the thread stores and the lane's stream waits on. */
    ShadowSpillBackendSignals signals;
    uint64_t *signal_host;
    /*
     * Chunks planned over this lane's life. Both words are counts on this
     * scale, so they keep rising across transfers and a reader never has to
     * know which transfer a number came from. Written and read by whoever
     * calls the lane, never by its thread, under the lane's lock.
     */
    uint64_t chunks_planned;

    /*
     * THE CHUNK RING, and it belongs to the lane rather than to a call.
     *
     * Chunk numbers already run across the lane's whole life -- a transfer owns
     * `[first_chunk, first_chunk + chunks)` -- so a slot is `chunk % ring_slots`
     * and the pipeline never has to be drained between transfers. It used to be
     * drained all the same, because `posted`, `completed` and `landed[]` were
     * locals in `run_copy` and it did not return until its own last completion
     * arrived. Consecutive transfers therefore overlapped not at all.
     *
     * Here instead: `chunk_posted` and `chunk_retired` keep rising, retirement
     * is in order, and the thread posts the next transfer's chunks into slots
     * the previous one has released. Touched only by the lane's thread.
     *
     * `flight_region` is which region's queue pair the outstanding chunks are
     * on, or NULL when none are. A lane may reach more than one remote pool and
     * each gets its own queue pair, and order across two queue pairs is not
     * guaranteed -- so the pipeline spans transfers to **one** region, and a
     * transfer to a different one waits for the ring to drain first. One route
     * reaches one remote pool, so that wait is a correctness fallback rather
     * than something the spill path meets.
     */
    const ShadowSpillRemoteRegion *flight_region;
    const RingRegistration *flight_claim;
    uint64_t chunk_posted;
    uint64_t chunk_retired;
    uint8_t chunk_landed[SHADOWSPILL_NETWORK_MAX_RING_SLOTS];
    /* Which queue pair the next chunk goes to, when there is more than one. */
    uint64_t posted;

    /*
     * Where a transfer's time goes, split between the host copy and the NIC.
     * Accumulated only when asked for: the question "why is this not at line
     * rate" is otherwise answered by guessing, and guessing was wrong once.
     *
     * **The split is only meaningful with one slot.** With more, a wait
     * returns immediately for a chunk that landed while the host was copying,
     * so the phases no longer partition the wall clock and the two figures
     * sum to less than the elapsed time -- which is the overlap working, not
     * an error. Read the end-to-end rate for the answer; read the split to
     * find out which stage is the ceiling.
     */
    uint8_t measuring;
    double staging_seconds;
    double link_seconds;
    uint64_t measured_bytes;

    /*
     * What a transfer cost this lane, reported through the contract's `timing`
     * entry. The seven counters are not here: they are in the base, kept
     * through `shadowspill_lane_counted`, and the runtime reads them straight
     * out -- so this lane no longer copies them anywhere.
     *
     * A clock read per chunk is cheap but not free, so it is taken only while
     * a trace is running, and `timed` says which case a reader is looking at
     * -- otherwise "not measured" and "instant" are the same zero.
     *
     * Written by the lane thread and the dispatching thread, read by whoever
     * collects diagnostics, so each is atomic rather than locked: this must
     * never make a transfer wait on a reader.
     *
     * Microseconds, as integers, so the accumulation is atomic without a lock
     * and a reader divides once rather than every producer rounding.
     */
    _Atomic uint64_t stat_completion_micros;
    _Atomic uint64_t stat_longest_micros;
    _Atomic uint64_t stat_timed;

    /*
     * Where a *small* transfer's fixed cost goes. Measured in stages because
     * the total (12 us against a NIC that answers in 3.9) is mostly handoff,
     * and which handoff matters for what to do about it.
     */
    double enqueue_seconds;      /* copy(): allocate, lock, signal            */
    double wakeup_seconds;       /* signal -> the lane thread running         */
    double work_seconds;         /* the thread doing the transfer             */
    double handback_seconds;     /* the thread finishing -> synchronize waking */
    double stream_seconds;       /* synchronize_stream after that             */
    /* Inside the work stage, which is most of it. */
    double lookup_seconds;       /* finding the ring's registration           */
    double memcpy_seconds;       /* the backend's staging copy alone         */
    double streamsync_seconds;   /* the stream sync after that copy           */
    double post_seconds;         /* ibv_post_send                             */
    double poll_seconds;         /* ibv_poll_cq until the completion          */
    uint64_t registrations_made;  /* should be one per region, ever            */
    uint64_t transfers;
    double queued_at;
    double finished_at;

    /* Non-zero while work is queued, readable without the lock so the thread
       can watch for it while spinning. */
    atomic_ullong queued;
    /*
     * The ring, and three counts that never wrap: transfers accepted, taken by
     * the thread, and finished by it. Slot k of a count is `count % WORK_SLOTS`.
     *
     * `accepted - retired` is work accepted and not yet finished, which is
     * **not** the same as work still waiting. The thread takes an item before
     * doing it, so `started == accepted` means "nothing waiting to start", not
     * "nothing in flight". `synchronize` must wait for `retired`, or it can
     * return while the transfer it was told to wait for is still on the wire.
     */
    Work work_ring[WORK_SLOTS];
    uint64_t accepted;
    uint64_t started;
    uint64_t retired;

    pthread_t thread;
    pthread_mutex_t lock;
    pthread_cond_t work_ready;
    pthread_cond_t drained;
    atomic_uint stopping;
    atomic_uint failed;
    uint8_t thread_started;
} RemoteLane;

static double seconds_now(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
}

/* ------------------------------------------------------------ the ring */

/* Register the ring with the protection domain serving `region`, once each. A
   lane may reach more than one remote pool, and a registration belongs to one
   protection domain. */
static const RingRegistration *ring_registration_for(
    RemoteLane *lane, const ShadowSpillRemoteRegion *region
) {
    for (uint32_t index = 0U; index < lane->registration_count; ++index) {
        if (lane->registrations[index].region == region) {
            return &lane->registrations[index];
        }
    }
    if (lane->registration_count >= RING_REGISTRATION_CACHE) {
        return NULL;
    }
    const int claimed = shadowspill_remote_region_claim_queue_pair(region);
    if (claimed < 0) {
        fprintf(
            stderr,
            "shadowspill network: no queue pair left for this lane; raise "
            "SHADOWSPILL_NETWORK_QUEUE_PAIRS above %u\n",
            region->endpoint.queue_pair_count
        );
        return NULL;
    }
    struct ibv_mr *registration = ibv_reg_mr(
        region->endpoint.protection_domain, lane->ring,
        (size_t)lane->ring_bytes, IBV_ACCESS_LOCAL_WRITE
    );
    if (registration == NULL) {
        return NULL;
    }
    lane->registrations[lane->registration_count++] = (RingRegistration){
        .region = region, .registration = registration, .queue_pair = claimed
    };
    ++lane->registrations_made;
    return &lane->registrations[lane->registration_count - 1U];
}

/* --------------------------------------------------------- one transfer */

/* Post one chunk. Does not wait: waiting is separate so that a chunk's host
   copy can run while an earlier chunk is still on the wire. */
static int post_chunk(
    const ShadowSpillRemoteRegion *region,
    const RingRegistration *claim,
    void *slot,
    uint64_t chunk_number,
    uint64_t remote_offset,
    uint64_t bytes,
    int to_remote
) {
    struct ibv_sge element = {
        .addr = (uint64_t)(uintptr_t)slot,
        .length = (uint32_t)bytes,
        .lkey = claim->registration->lkey,
    };
    struct ibv_send_wr request = {
        /*
         * **The chunk's lane-global number**, which names the transfer it
         * belongs to as well as its place in the ring: a transfer owns a
         * contiguous run of these, and the slot is the number modulo
         * `ring_slots`. It used to be the slot index alone, which was enough
         * while a call drained its own chunks before returning and is not
         * enough now that retirement crosses transfers -- a completion has to
         * say which transfer it finished.
         */
        .wr_id = chunk_number,
        .sg_list = &element,
        .num_sge = 1,
        .opcode = to_remote ? IBV_WR_RDMA_WRITE : IBV_WR_RDMA_READ,
        .send_flags = IBV_SEND_SIGNALED,
        .wr = {.rdma = {
            .remote_addr = region->address + remote_offset,
            .rkey = region->key,
        }},
    };
    struct ibv_send_wr *bad = NULL;
    /* This lane's own queue pair, never another's. */
    return ibv_post_send(
        region->endpoint.queue_pairs[claim->queue_pair], &request, &bad
    ) == 0 ? 0 : -1;
}

/*
 * Wait until slot `wanted` has completed, remembering any other slot that
 * completes first.
 *
 * It matches by work-request id rather than counting arrivals. A single queue
 * pair completes in the order it was posted, so counting would work -- but
 * several do not complete in order with respect to each other, and this lane
 * spreads chunks across them. Counting would then reclaim a slot whose bytes
 * were still on the wire.
 *
 * Blocking here costs nothing the runtime is waiting for: this is the lane's
 * own thread, and the worker is already dispatching the next action.
 */
/* The NIC has finished every chunk below `reached`. Releases whatever the
   stream is waiting on for them -- a fetch's copy out of the slot, an evict's
   reuse of the slot, and a transfer's own completion event. Monotonic, because
   chunks retire in order. */
static void report_nic(RemoteLane *lane, uint64_t reached) {
    atomic_store_explicit(
        (_Atomic uint64_t *)&lane->signal_host[SIGNAL_NIC],
        reached,
        memory_order_release
    );
}

/*
 * Take whatever completions have arrived and retire what that makes retirable,
 * in order. **Never waits**, which is the point: the thread alternates between
 * posting and retiring, and whichever is ready first is what it does. Blocking
 * here for a completion would give up the chance to post the moment the device
 * catches up, and with two ring slots there is no slack to absorb that -- it
 * measured about 1.5 % of a large transfer's rate.
 *
 * In order, which is what lets the ring span transfers: a slot is free once the
 * chunk that occupied it retires, whichever transfer that chunk belonged to.
 * Completions may arrive out of order even on one queue pair's completion
 * queue, so `chunk_landed` records what has arrived and the oldest is retired
 * only when it is among them.
 */
static int collect_and_retire(RemoteLane *lane, uint64_t *retired) {
    const uint64_t slots = lane->tuning.ring_slots;
    struct ibv_cq *const queue =
        lane->flight_region->endpoint.completion_queues[
            lane->flight_claim->queue_pair];
    for (;;) {
        struct ibv_wc completion;
        /* This lane's own completion queue. Nothing else polls it, so a
           completion taken here is always this lane's. */
        const int taken = ibv_poll_cq(queue, 1, &completion);
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
        lane->chunk_landed[completion.wr_id % slots] = 1U;
    }
    while (lane->chunk_retired < lane->chunk_posted &&
           lane->chunk_landed[lane->chunk_retired % slots]) {
        lane->chunk_landed[lane->chunk_retired % slots] = 0U;
        ++lane->chunk_retired;
        ++*retired;
    }
    if (*retired != 0U) {
        report_nic(lane, lane->chunk_retired);
    }
    return 0;
}

/*
 * Has the stream finished with chunks up to `reached`?
 *
 * A read of a word the stream stores, never a device call: this runs on the
 * lane's thread, and the thread must be able to make progress while the stream
 * is waiting on something only this thread will store. A device call here --
 * even an asynchronous one -- can block on a stalled stream, and that is a
 * cycle, because the thread is what unstalls it.
 *
 * **It asks rather than waits**, and that is load-bearing now that the ring
 * spans transfers. The next transfer's device work sits behind the previous
 * transfer's `signal` on the route stream, so the device cannot reach it until
 * this thread has retired every chunk of the previous one. A thread spinning
 * here for that would be waiting on itself. Asking lets the caller stop posting
 * and go retire, which is what releases the stream.
 */
static int device_reached(const RemoteLane *lane, uint64_t reached) {
    const _Atomic uint64_t *const word =
        (const _Atomic uint64_t *)&lane->signal_host[SIGNAL_DEVICE];
    return atomic_load_explicit(word, memory_order_acquire) >= reached ? 1 : 0;
}

/*
 * A bounded spin before sleeping.
 *
 * Waking this thread costs ~1.8 us of the ~12 us a small transfer takes, and
 * transfers arrive in batches at a task boundary -- so the next one is often
 * already on its way when this one finishes. Watching for it briefly catches
 * that case without a context switch.
 *
 * **Bounded is the whole point.** A lane thread that spins without limit is a
 * core per lane, which is why the contract says a lane should block. This
 * spins for a few microseconds and then blocks properly, so an idle runtime
 * costs nothing.
 */
static void spin_briefly(RemoteLane *lane) {
    if (lane->tuning.spin_nanoseconds == 0U) {
        return;
    }
    const double deadline =
        seconds_now() + (double)lane->tuning.spin_nanoseconds * 1e-9;
    while (atomic_load_explicit(&lane->queued, memory_order_acquire) == 0ULL &&
           atomic_load_explicit(&lane->stopping, memory_order_acquire) == 0U) {
        if (seconds_now() >= deadline) {
            return;
        }
#if defined(__x86_64__) || defined(__i386__)
        __builtin_ia32_pause();
#elif defined(__aarch64__)
        __asm__ volatile("yield");
#endif
    }
}

/*
 * Where chunk `index` lives in the ring. Staging only: without it the bytes
 * are already where the NIC wants them.
 */
static char *stage_slot(
    const RemoteLane *lane, uint64_t chunk) {
    const uint64_t slot = chunk % lane->tuning.ring_slots;
    return (char *)lane->ring + slot * lane->tuning.chunk_bytes;
}

/*
 * Has the device done its part for chunk `chunk`?
 *
 * An evict needs the device to have *filled* this chunk's slot; a fetch needs
 * it to have *drained* the slot this chunk is about to reuse, which for the
 * first `ring_slots` chunks has never been used. Both read a word the stream
 * stores, never a device call -- see `device_reached`, and see there for why
 * this asks rather than waits.
 */
static int stage_device_ready(
    const RemoteLane *lane, const Work *work, uint64_t chunk
) {
    const uint64_t slots = lane->tuning.ring_slots;
    /*
     * Chunks are numbered across the lane's whole life, not within a transfer,
     * and the difference is a correctness one rather than a tidiness one.
     *
     * A fetch's slot is drained by the device, and the copy that drains it is
     * queued on the route stream -- so when a transfer's last completion
     * arrives its slots may still hold bytes nobody has copied out. Counting
     * from zero each transfer would let the next one's first `ring_slots`
     * chunks skip this wait and post over them. Counting globally makes the
     * ring one continuous pipeline instead, which is also why it never has to
     * be drained between transfers.
     *
     * An evict is safe either way, because the device fills its slots from the
     * route stream and the stream orders one transfer's copies after the
     * previous one's. It waits here all the same: the condition is the same
     * question asked of the other side.
     */
    if (work->to_remote) {
        return device_reached(lane, chunk + 1U);
    }
    return chunk < slots ? 1 : device_reached(lane, chunk - slots + 1U);
}

/*
 * One transfer's hardware, and nothing else.
 *
 * This is the core path, and it is written for the case where the NIC reaches
 * device memory: post the bytes where they already are, watch for the
 * completion, say so. The report is what releases the stream -- the transfer's
 * completion event on every path, and additionally the device copies when
 * there are any.
 *
 * Staging enters through three `if`s and nothing else: it decides how many
 * pieces there are, where each piece lives, and whether the device has to be
 * waited for first. Everything around them is the same work in both cases,
 * which is the point -- a box with peer memory runs this function with the
 * branches not taken, not a different function.
 *
 * Up to `ring_slots` posts are outstanding while staging, so a chunk's wire
 * time overlaps its neighbour's device copy. Completions on one queue pair
 * arrive in the order they were posted, so reporting them in order keeps the
 * counter monotonic, which is all the stream's waits require.
 */
/*
 * Post one chunk, whose lane-global number is `chunk`.
 *
 * Returns 1 when it went to the NIC, 0 when the device has not done its half
 * yet -- the caller must then retire rather than wait, see
 * `stage_device_ready` -- and -1 when the post failed.
 */
static int post_one_chunk(RemoteLane *lane, const Work *work, uint64_t chunk) {
    if (lane->stages && !stage_device_ready(lane, work, chunk)) {
        return 0;
    }
    const uint64_t chunk_bytes = lane->tuning.chunk_bytes;
    const uint64_t offset = (chunk - work->first_chunk) * chunk_bytes;
    const uint64_t bytes = lane->stages
        ? (work->bytes - offset < chunk_bytes ? work->bytes - offset : chunk_bytes)
        : work->bytes;
    char *const address = lane->stages
        ? stage_slot(lane, chunk)
        : (char *)work->local + offset;
    if (post_chunk(
            work->region, lane->flight_claim, address, chunk,
            work->remote_offset + offset, bytes, work->to_remote
        ) != 0) {
        return -1;
    }
    shadowspill_lane_counted(&lane->base.chunks, 1U);
    ++lane->chunk_posted;
    return 1;
}

/* How many chunks may be outstanding at once. One slot's worth when the NIC
   reaches device memory directly, because then there is no ring to fill. */
static uint32_t outstanding_limit(const RemoteLane *lane) {
    return lane->stages ? lane->tuning.ring_slots : 1U;
}

/*
 * Everything the lane's thread does, once.
 *
 * Post whatever the ring, the device and the queue pair allow, then retire the
 * oldest outstanding chunk. Both halves are bounded and neither blocks on the
 * other, which is what makes the loop able to unstall the stream it is waiting
 * on -- posting stops when the device is behind, and retiring is what lets the
 * device catch up.
 *
 * `started` and `retired` are transfer counts and `chunk_posted`/`chunk_retired`
 * chunk counts; a transfer is started when its last chunk is posted and retired
 * when its last chunk retires. They advance independently, which is the whole
 * of what "the pipeline spans transfers" means.
 */
static int post_what_it_can(
    RemoteLane *lane, uint64_t accepted, uint64_t *started, uint64_t *posted
) {
    while (*started < accepted) {
        Work *const work = &lane->work_ring[*started % WORK_SLOTS];
        if (lane->flight_region != NULL && lane->flight_region != work->region) {
            /* Another region means another queue pair, and order across two of
               them is not guaranteed. Drain before crossing. */
            break;
        }
        if (lane->chunk_posted - lane->chunk_retired >= outstanding_limit(lane)) {
            break;
        }
        if (lane->flight_region == NULL) {
            const RingRegistration *const claim =
                ring_registration_for(lane, work->region);
            if (claim == NULL) {
                return -1;
            }
            lane->flight_region = work->region;
            lane->flight_claim = claim;
        }
        const int sent = post_one_chunk(lane, work, lane->chunk_posted);
        if (sent < 0) {
            return -1;
        }
        if (sent == 0) {
            /* The device is behind. Retire, which is what moves it. */
            break;
        }
        ++*posted;
        if (lane->chunk_posted == work->first_chunk + work->chunks) {
            ++*started;
        }
    }
    return 0;
}

/*
 * Everything the lane's thread does between two looks at its queue.
 *
 * Post whatever the ring, the device and the queue pair allow; take whatever
 * completions have arrived. Neither half waits on the other, which is what
 * makes the loop able to unstall the stream it depends on -- posting stops when
 * the device is behind, and retiring is what lets the device catch up. A round
 * that does neither pauses and tries again, because both are questions whose
 * answer arrives from hardware.
 *
 * Returns once it has made progress and then found nothing more to do, so the
 * caller can notice newly accepted work; the lock is therefore off the
 * per-chunk path.
 *
 * `started` and `retired` are transfer counts and `chunk_posted`/`chunk_retired`
 * chunk counts; a transfer is started when its last chunk is posted and retired
 * when its last chunk retires. They advance independently, which is the whole
 * of what "the pipeline spans transfers" means.
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
        /* Chunks are outstanding and neither side is ready. */
#if defined(__x86_64__) || defined(__i386__)
        __builtin_ia32_pause();
#elif defined(__aarch64__)
        __asm__ volatile("yield");
#endif
    }
}

/*
 * A failure anywhere in the pipeline reports **every chunk planned**, not just
 * the failed transfer's.
 *
 * The stream is waiting on these words -- for a staged transfer's device
 * copies, and for the completion event on every transfer -- and a failure that
 * left them unstored would be a step that never ends rather than one that
 * fails. The latch is what makes it fail. Planned rather than posted, because
 * a transfer that never reached the NIC is waited on exactly the same way.
 */
static void fail_everything(RemoteLane *lane) {
    shadowspill_lane_counted(&lane->base.failures, 1U);
    pthread_mutex_lock(&lane->lock);
    const uint64_t planned = lane->chunks_planned;
    pthread_mutex_unlock(&lane->lock);
    report_nic(lane, planned);
    atomic_store_explicit(&lane->failed, 1U, memory_order_release);
    shadowspill_lane_latch_failure(
        lane->base.runtime, SHADOWSPILL_STATUS_BACKEND_FAILURE,
        SHADOWSPILL_FAILURE_REASON_TRANSFER_REJECTED
    );
}

static void *lane_thread(void *argument) {
    RemoteLane *lane = argument;
    for (;;) {
        /*
         * Only when the pipeline is empty. `spin_briefly` waits for work to
         * *arrive*, and with chunks outstanding the thread already has work --
         * spinning a full window before every retire costs more than the wakeup
         * it saves, and measured about 1.5 % of a large transfer's rate.
         */
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
        if (lane->measuring && started < accepted) {
            lane->wakeup_seconds += seconds_now() - lane->queued_at;
        }
        pthread_mutex_unlock(&lane->lock);

        const double work_started = lane->measuring ? seconds_now() : 0.0;
        const int failed = pump(lane, accepted, &started) != 0;
        if (failed) {
            fail_everything(lane);
        }

        pthread_mutex_lock(&lane->lock);
        const uint64_t retired_before = lane->retired;
        if (lane->measuring) {
            lane->work_seconds += seconds_now() - work_started;
        }
        if (started != lane->started) {
            (void)atomic_fetch_sub_explicit(
                &lane->queued, started - lane->started, memory_order_acq_rel
            );
            lane->started = started;
        }
        /*
         * A transfer is retired when its last chunk is, which is the moment its
         * bytes are on the far side. `synchronize` waits for this and not for
         * the list being empty, because the thread takes a transfer before
         * doing it.
         */
        while (lane->retired < lane->started) {
            const Work *const oldest =
                &lane->work_ring[lane->retired % WORK_SLOTS];
            if (!failed &&
                lane->chunk_retired < oldest->first_chunk + oldest->chunks) {
                break;
            }
            ++lane->retired;
            if (lane->measuring) {
                ++lane->transfers;
            }
        }
        if (failed) {
            /* Nothing will retire these; let every waiter go. */
            lane->retired = lane->started = lane->accepted;
            lane->chunk_retired = lane->chunk_posted;
        }
        if (lane->retired == lane->accepted && lane->measuring) {
            lane->finished_at = seconds_now();
        }
        /*
         * Broadcast when a *transfer* retired, which is both what `synchronize`
         * waits for and what frees a work-ring slot for `claim_slot`. Not on
         * every chunk: a chunk frees nothing either of them is waiting on, and
         * at a chunk apiece this is on the transfer path.
         */
        if (lane->retired != retired_before) {
            pthread_cond_broadcast(&lane->drained);
        }
        pthread_mutex_unlock(&lane->lock);
    }
}

/*
 * Take the next slot, waiting if every one is still in flight.
 *
 * The wait terminates, and the reason is worth stating because a lane blocking
 * in `copy` is otherwise exactly the deadlock this design avoids: the thread
 * never waits on the caller. Its only wait is for device work the *same* call
 * to `copy` already enqueued -- transfer N's device half is issued during
 * copy N, before copy N+1 can be reached -- so it always drains and a slot
 * always comes free. In practice it never runs: a route cannot get 256
 * transfers ahead of a lane that is draining them.
 *
 * Returns the claimed slot with the lock held, so the caller fills it before
 * anything can see it.
 */
static Work *claim_slot(RemoteLane *lane, uint64_t *handle) {
    pthread_mutex_lock(&lane->lock);
    while (lane->accepted - lane->retired == WORK_SLOTS) {
        pthread_cond_wait(&lane->drained, &lane->lock);
    }
    Work *work = &lane->work_ring[lane->accepted % WORK_SLOTS];
    /*
     * Invalidated before a single field is overwritten, so a `transfer` still
     * holding the previous occupant's handle fails its check rather than
     * reading a half-built record. The new handle is stored by `publish_slot`,
     * after every field is filled -- the two stores bracket construction.
     */
    atomic_store_explicit(&work->handle, 0U, memory_order_release);
    *handle = lane->accepted + 1U;
    return work;
}

/* Publish the slot the caller has filled, and wake the thread. */
static void publish_slot(RemoteLane *lane, Work *work, uint64_t handle) {
    atomic_store_explicit(&work->handle, handle, memory_order_release);
    if (lane->measuring) {
        lane->queued_at = seconds_now();
    }
    (void)atomic_fetch_add_explicit(&lane->queued, 1ULL, memory_order_release);
    ++lane->accepted;
    pthread_mutex_unlock(&lane->lock);
    /*
     * Signalled *after* unlocking. Signalling while holding the lock wakes the
     * thread only for it to block immediately on the mutex the signaller still
     * holds -- a handoff through the scheduler that buys nothing. With the
     * spin above, the thread has usually seen the atomic and is already
     * heading for the lock before this runs at all.
     */
    pthread_cond_signal(&lane->work_ready);
}

/* ------------------------------------------------------- the operations */

/*
 * Order this lane's work behind `event`, on **its own** stream. An evict's
 * `copy_device_to_host` reads device memory the trigger event protects, and
 * that copy runs here -- so the wait has to be here too, not only on the
 * route's stream.
 *
 * Never returns 1: this lane can always enqueue a device-side wait, because it
 * has a stream to enqueue it on.
 */
static int remote_wait(ShadowSpillLane *lane, ShadowSpillBackendEvent event) {
    shadowspill_lane_counted(&lane->waits, 1U);
    return lane->backend->wait_event(lane->backend->state, lane->stream, event)
        == 0 ? 0 : -1;
}

/*
 * What this transfer is, before anything is issued.
 *
 * Exactly one end is remote -- a route joins a device pool and a remote one --
 * and which end says the direction. The chunk count is decided here, once, so
 * the thread and the stream agree on it without either recomputing it: one
 * piece when the NIC reaches device memory, and as many as the ring holds
 * when it does not.
 */
static int plan_transfer(
    RemoteLane *lane,
    Work *work,
    void *destination,
    const void *source,
    uint64_t bytes
) {
    const ShadowSpillRemoteRegion *const from =
        shadowspill_remote_region_for(source);
    const ShadowSpillRemoteRegion *const to =
        shadowspill_remote_region_for(destination);
    if ((from == NULL) == (to == NULL)) {
        /* Both or neither: this lane was asked for a copy it does not serve,
           which means the route resolved to the wrong lane. */
        return -1;
    }
    work->bytes = bytes;
    if (to != NULL) {
        work->region = to;
        work->to_remote = 1U;
        work->local = (void *)(uintptr_t)source;
        work->remote_offset =
            (uint64_t)((const char *)destination - (const char *)to->reservation);
    } else {
        work->region = from;
        work->to_remote = 0U;
        work->local = destination;
        work->remote_offset =
            (uint64_t)((const char *)source - (const char *)from->reservation);
    }
    const uint64_t chunk_bytes = lane->tuning.chunk_bytes;
    work->chunks = lane->stages
        ? (uint32_t)((bytes + chunk_bytes - 1U) / chunk_bytes)
        : 1U;
    /* Under the lane's lock, because `claim_slot` holds it: the chunk
       numbering is shared with the thread and was read unprotected before the
       ring gave the claim a lock to sit under. */
    work->first_chunk = lane->chunks_planned;
    lane->chunks_planned += work->chunks;
    return 0;
}

/*
 * The device's half of a staged transfer, enqueued here and nowhere else.
 *
 * All of it goes on the route stream, in chunk order, at dispatch. The lane
 * still makes device calls -- these are they -- but they are made by the
 * runtime's own thread, on the runtime's own stream, before the lane's thread
 * sees the transfer at all. What matters is that the lane's *thread* makes
 * none, so it can always reach the reports the stream is waiting for. Each
 * chunk waits on what the thread will report
 * and, when it is done, writes what the thread will wait for:
 *
 *   fetch   wait: the NIC filled this chunk's slot  -> copy it to the device
 *   evict   wait: the NIC emptied the slot being reused -> refill it
 *
 * and both then store a rising count so the thread knows the device is
 * finished with that slot.
 */
static int stage_enqueue_device_work(RemoteLane *lane, const Work *work) {
    const ShadowSpillBackend *const backend = lane->base.backend;
    const uint64_t chunk_bytes = lane->tuning.chunk_bytes;
    const uint32_t slots = lane->tuning.ring_slots;
    for (uint32_t index = 0U; index < work->chunks; ++index) {
        const uint64_t offset = (uint64_t)index * chunk_bytes;
        const uint64_t chunk = work->bytes - offset < chunk_bytes
            ? work->bytes - offset : chunk_bytes;
        const uint64_t chunk_number = work->first_chunk + index;
        char *const slot = stage_slot(lane, chunk_number);
        char *const local = (char *)work->local + offset;
        if (work->to_remote) {
            /*
             * Before overwriting a slot, wait for the NIC to have finished the
             * chunk that last occupied it -- `slots` chunks ago on the lane's
             * global numbering, **not** on this transfer's.
             *
             * It was `index >= slots`, which is the same thing only for the
             * first transfer: from the second on it let the opening `slots`
             * chunks skip the wait entirely and the device wrote over slots the
             * NIC was still reading. Harmless while a transfer's chunks were
             * drained before the next one began, and a silent corruption once
             * the ring spans transfers -- one byte of one word, in a payload
             * that otherwise arrives intact.
             */
            if (chunk_number >= slots && backend->wait_value(
                    backend->state, lane->base.stream, lane->signals, SIGNAL_NIC,
                    chunk_number + 1U - slots
                ) != 0) {
                return -1;
            }
            if (backend->copy_device_to_host(
                    backend->state, slot, local, chunk, lane->base.stream
                ) != 0) {
                return -1;
            }
        } else {
            if (backend->wait_value(
                    backend->state, lane->base.stream, lane->signals, SIGNAL_NIC,
                    chunk_number + 1U
                ) != 0) {
                return -1;
            }
            if (backend->copy_host_to_device(
                    backend->state, local, slot, chunk, lane->base.stream
                ) != 0) {
                return -1;
            }
        }
        if (backend->write_value(
                backend->state, lane->base.stream, lane->signals, SIGNAL_DEVICE,
                chunk_number + 1U
            ) != 0) {
            return -1;
        }
    }
    return 0;
}

/* Plan it, give the device its half if there is one, hand the rest to the
   thread. Nothing is waited for here: this runs on the runtime's worker. */
static int remote_copy(
    ShadowSpillLane *lane,
    void *destination,
    const void *source,
    uint64_t bytes,
    uint64_t *handle
) {
    RemoteLane *self = (RemoteLane *)lane;
    const double entered = self->measuring ? seconds_now() : 0.0;
    *handle = 0U;
    shadowspill_lane_counted(&lane->copies, 1U);
    shadowspill_lane_counted(&lane->bytes, bytes);

    uint64_t claimed = 0U;
    Work *work = claim_slot(self, &claimed);
    if (plan_transfer(self, work, destination, source, bytes) != 0) {
        /* Never published, so `accepted` did not move and the next claim takes
           this same slot; its handle is already 0. */
        pthread_mutex_unlock(&self->lock);
        shadowspill_lane_counted(&lane->failures, 1U);
        return -1;
    }
    /*
     * The thread is given the transfer *before* the device side is enqueued,
     * and the order matters.
     *
     * Every chunk's value wait is satisfied by this lane's thread. Enqueuing
     * them all first would leave this thread -- the runtime's worker -- issuing
     * device calls behind value waits nobody can satisfy yet, and an
     * outstanding value wait blocks calls on other streams too. `copy` would
     * then block, which the contract forbids and which is the deadlock this
     * design exists to avoid, merely moved onto the caller.
     *
     * Handing the work over first is safe because the wait is greater-or-equal:
     * a wait enqueued after its value was already stored passes at once.
     * Stream order is unchanged -- every copy is still enqueued before this
     * returns, so `signal`'s event still lands behind all of them.
     *
     * The plan is copied out first because the slot belongs to the thread the
     * moment it is published: the thread may finish the transfer and a later
     * `copy` may claim the slot while this function is still enqueuing the
     * device side. Everything the device side needs is a handful of scalars,
     * so a copy costs nothing and removes the question.
     */
    const Work plan = *work;
    publish_slot(self, work, claimed);
    *handle = claimed;
    if (self->stages && stage_enqueue_device_work(self, &plan) != 0) {
        return -1;
    }
    if (self->measuring) {
        self->enqueue_seconds += seconds_now() - entered;
    }
    return 0;
}

/*
 * What one transfer did. No instants: this lane's clock is the host's, and
 * nothing anchors that to the trace's device-side origin -- which is what the
 * pair of interval entries it had to leave NULL used to mean. What it does
 * know is how much it moved and in how many pieces, and the pieces are the
 * number worth having here: the ratio of bytes to chunks is what says whether
 * a slow transfer was one long wait or many short ones.
 */
static int remote_transfer(
    ShadowSpillLane *lane, uint64_t handle, ShadowSpillLaneTransfer *transfer
) {
    RemoteLane *self = (RemoteLane *)lane;
    if (handle == 0U) {
        return -1;
    }
    const Work *work = &self->work_ring[(handle - 1U) % WORK_SLOTS];
    /*
     * Read without the lock, and the handle is checked on both sides of the
     * fields. A `copy` claiming this slot rewrites the handle first, so a
     * match before and after means nothing overwrote what was read between
     * them -- and taking the lane's lock here would let a trace's reader stall
     * a transfer, which is the trade this avoids everywhere else too.
     */
    if (atomic_load_explicit(&work->handle, memory_order_acquire) != handle) {
        return -1;
    }
    const ShadowSpillLaneTransfer read = {
        .started_at_nanoseconds = SHADOWSPILL_LANE_NO_TIME,
        .finished_at_nanoseconds = SHADOWSPILL_LANE_NO_TIME,
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
 * Two calls, and the same two whichever path the transfer took. The wait is on
 * the last report this lane's thread will make, so the event cannot complete
 * before the hardware is done; where the transfer also had device work, that
 * work is already ahead of this on the same stream and the wait is satisfied
 * by the time the stream reaches it.
 *
 * The wait is enqueued here, on the dispatching thread, rather than left to
 * the lane's thread to arrange: the stream must already be waiting before the
 * word can be stored, or a store that landed first would make the wait pass
 * for the wrong reason.
 */
static int remote_signal(
    ShadowSpillLane *lane, uint64_t handle, ShadowSpillBackendEvent event
) {
    /* The value wait covers every chunk planned so far, which is every
       transfer issued before this signal and not only the one named. So the
       handle says nothing this does not already know. */
    (void)handle;
    RemoteLane *self = (RemoteLane *)lane;
    const ShadowSpillBackend *const backend = lane->backend;
    shadowspill_lane_counted(&lane->signals, 1U);
    if (self->chunks_planned != 0U && backend->wait_value(
            backend->state, lane->stream, self->signals, SIGNAL_NIC,
            self->chunks_planned
        ) != 0) {
        return -1;
    }
    return backend->record_event(backend->state, event, lane->stream) == 0
        ? 0 : -1;
}

/*
 * Block until everything issued has landed.
 *
 * Waits on `outstanding`, not on the list being empty. The list empties when
 * the thread *takes* an item, which is before it does it -- so a wait on the
 * list could return with the transfer still on the wire, and would, whenever
 * `synchronize` was called in the window between the two. That the canary
 * passed anyway is luck about timing, not evidence.
 */
static int remote_synchronize(ShadowSpillLane *base_lane) {
    RemoteLane *lane = (RemoteLane *)base_lane;
    pthread_mutex_lock(&lane->lock);
    while (lane->retired != lane->accepted) {
        pthread_cond_wait(&lane->drained, &lane->lock);
    }
    const double woken = lane->measuring ? seconds_now() : 0.0;
    if (lane->measuring && lane->finished_at != 0.0) {
        lane->handback_seconds += woken - lane->finished_at;
        lane->finished_at = 0.0;
    }
    pthread_mutex_unlock(&lane->lock);
    if (atomic_load_explicit(&lane->failed, memory_order_acquire) != 0U) {
        return -1;
    }
    const int status = lane->base.backend->synchronize_stream(
        lane->base.backend->state, lane->base.stream
    );
    if (lane->measuring) {
        lane->stream_seconds += seconds_now() - woken;
    }
    return status == 0 ? 0 : -1;
}

/*
 * No intervals. The two entries are optional as a pair, and this lane cannot
 * place an instant on the trace's clock: its bytes move on a NIC, whose
 * completions are not events on the stream the trace is read against. Its
 * transfers are recorded untimed rather than timed wrongly.
 */

static void remote_destroy(ShadowSpillLane *base_lane) {
    RemoteLane *lane = (RemoteLane *)base_lane;
    if (lane == NULL) {
        return;
    }
    if (lane->measuring && lane->measured_bytes != 0U) {
        const double mib = (double)lane->measured_bytes / (double)(1U << 20U);
        fprintf(
            stderr,
            "shadowspill network: %.0f MiB -- host copy %.4f s (%.0f MiB/s), "
            "link %.4f s (%.0f MiB/s), serial total %.4f s (%.0f MiB/s)\n",
            mib, lane->staging_seconds, mib / lane->staging_seconds,
            lane->link_seconds, mib / lane->link_seconds,
            lane->staging_seconds + lane->link_seconds,
            mib / (lane->staging_seconds + lane->link_seconds)
        );
        if (lane->transfers != 0U) {
            const double each = 1e6 / (double)lane->transfers;
            fprintf(
                stderr,
                "shadowspill network: %llu transfers, per transfer -- enqueue "
                "%.2f us, wake %.2f us, work %.2f us, hand back %.2f us, "
                "stream %.2f us\n",
                (unsigned long long)lane->transfers,
                lane->enqueue_seconds * each, lane->wakeup_seconds * each,
                lane->work_seconds * each, lane->handback_seconds * each,
                lane->stream_seconds * each
            );
            fprintf(
                stderr,
                "shadowspill network:   within work -- lookup %.2f us, staging copy "
                "%.2f us, stream sync %.2f us, post %.2f us, poll %.2f us "
                "(%llu registrations made, %u cached)\n",
                lane->lookup_seconds * each, lane->memcpy_seconds * each,
                lane->streamsync_seconds * each, lane->post_seconds * each,
                lane->poll_seconds * each,
                (unsigned long long)lane->registrations_made,
                lane->registration_count
            );
        }
    }
    if (lane->thread_started) {
        atomic_store_explicit(&lane->stopping, 1U, memory_order_release);
        pthread_mutex_lock(&lane->lock);
        pthread_cond_broadcast(&lane->work_ready);
        pthread_mutex_unlock(&lane->lock);
        (void)pthread_join(lane->thread, NULL);
    }
    for (uint32_t index = 0U; index < lane->registration_count; ++index) {
        (void)ibv_dereg_mr(lane->registrations[index].registration);
    }
    if (lane->ring != NULL) {
        (void)lane->base.backend->unregister_host_memory(
            lane->base.backend->state, lane->ring, lane->ring_bytes
        );
        (void)munmap(lane->ring, (size_t)lane->ring_bytes);
    }
    if (lane->signal_host != NULL) {
        (void)lane->base.backend->free_signals(lane->base.backend->state, lane->signals);
    }
    pthread_cond_destroy(&lane->drained);
    pthread_cond_destroy(&lane->work_ready);
    pthread_mutex_destroy(&lane->lock);
    free(lane);
}

/*
 * What a transfer cost this lane. Read without locking: every field is atomic
 * and the reader wants a recent picture, not a consistent instant -- taking the
 * lane's lock here would make a diagnostics call able to stall a transfer,
 * which is the wrong trade for a number nobody acts on within a microsecond.
 *
 * The counters are not here. They are in the base, and the runtime reads them.
 */
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

static const ShadowSpillLaneOperations remote_operations = {
    .wait = remote_wait,
    .copy = remote_copy,
    .signal = remote_signal,
    .synchronize = remote_synchronize,
    .transfer = remote_transfer,
    .destroy = remote_destroy,
    .timing = remote_timing,
};

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
    atomic_init(&lane->stopping, 0U);
    atomic_init(&lane->failed, 0U);
    atomic_init(&lane->queued, 0ULL);
    if (pthread_mutex_init(&lane->lock, NULL) != 0 ||
        pthread_cond_init(&lane->work_ready, NULL) != 0 ||
        pthread_cond_init(&lane->drained, NULL) != 0) {
        free(lane);
        return -1;
    }
    /*
     * The ring: an anonymous mapping the lane owns, registered with the
     * backend so the device copies are real asynchronous DMA. Its NIC
     * registrations come later, one per region, because a registration belongs
     * to a protection domain and the regions are not known yet.
     */
    lane->ring_bytes = lane->tuning.chunk_bytes * lane->tuning.ring_slots;
    lane->ring = mmap(
        NULL, (size_t)lane->ring_bytes, PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0
    );
    if (lane->ring == MAP_FAILED) {
        lane->ring = NULL;
        remote_destroy(&lane->base);
        return -1;
    }
    if (backend->register_host_memory(
            backend->state, lane->ring, lane->ring_bytes
        ) != 0) {
        (void)munmap(lane->ring, (size_t)lane->ring_bytes);
        lane->ring = NULL;
        remote_destroy(&lane->base);
        return -1;
    }
    /*
     * Two words, and both are counters rather than per-transfer objects, so
     * this allocation happens once per lane and never again -- which matters,
     * because registering host memory with the driver is not cheap and this is
     * the only place it is paid.
     *
     * Transfers on a route are ordered, so one rising count per direction
     * describes all of them: what the NIC has finished, and what the device
     * has finished.
     */
    if (backend->allocate_signals(
            backend->state, SIGNAL_WORDS, &lane->signals, &lane->signal_host
        ) != 0) {
        remote_destroy(&lane->base);
        return -1;
    }
    for (uint32_t index = 0U; index < SIGNAL_WORDS; ++index) {
        lane->signal_host[index] = 0U;
    }
    /* No peer memory on this box, so every transfer stages. The probe that
       would decide otherwise belongs here; nothing else would change. */
    lane->stages = 1U;
    if (pthread_create(&lane->thread, NULL, lane_thread, lane) != 0) {
        remote_destroy(&lane->base);
        return -1;
    }
    lane->thread_started = 1U;
    *created = &lane->base;
    return 0;
}

/*
 * Both directions, because a route is directed and a spill topology has two.
 * Registered by kind pair like every other lane, so the runtime finds this one
 * exactly as it finds the built-in.
 */
const ShadowSpillLaneDescription shadowspill_remote_lanes[2] = {
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
};
