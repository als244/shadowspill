/* Staging: how the remote lane moves bytes when the NIC cannot address the
   pool -- through a host ring, one piece at a time, with the device copying
   each piece across the pool's edge. */

/* MAP_ANONYMOUS is not in the strict ISO C11 the tree compiles as; the define
   has to precede the first system header. */
#define _DEFAULT_SOURCE

#include "remote_lane_internal.h"

#include <stdio.h>
#include <sys/mman.h>

/*
 * WHY THIS EXISTS, AND WHEN. The direct path posts the pool's memory to the
 * NIC. That needs the NIC to address the pool: host memory it always can, and
 * device memory it can where the platform exports it -- through a dma-buf the
 * backend hands over, or through peer memory in the kernel. Where neither
 * holds, this is the fallback: a ring of host memory the NIC *can* address,
 * registered with the backend as well so the device's copies into and out of
 * it are real asynchronous DMA, and a transfer crossing it in pieces the size
 * of a slot.
 *
 *   fetch   NIC reads the peer -> slot -> copy_host_to_device -> pool
 *   evict   pool -> copy_device_to_host -> slot -> NIC writes the peer
 *
 * The device's side is enqueued at dispatch, all of it, on the route stream,
 * by the runtime's worker -- behind the waits the runtime put there, so the
 * copies honour whatever the transfer was ordered behind, and behind value
 * waits on the NIC's word, so a slot is copied out only once filled and
 * refilled only once emptied. Each piece then stores the device's word, which
 * is how the thread knows a slot is ready to post or free to reuse. The
 * thread makes no device call; see the header for why that is a rule.
 *
 * It exists for hardware without peer memory; a deployment that reaches the
 * device directly never enters it.
 */

int remote_lane_stage_create(RemoteLane *lane) {
    const ShadowSpillBackend *const backend = lane->base.backend;
    lane->ring_bytes = lane->tuning.chunk_bytes * lane->tuning.ring_slots;
    lane->ring = mmap(
        NULL, (size_t)lane->ring_bytes, PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0
    );
    if (lane->ring == MAP_FAILED) {
        lane->ring = NULL;
        return -1;
    }
    if (backend->register_host_memory(
            backend->state, lane->ring, lane->ring_bytes
        ) != 0) {
        (void)munmap(lane->ring, (size_t)lane->ring_bytes);
        lane->ring = NULL;
        return -1;
    }
    /* Host memory, so the region registers it plainly; NULL backend says so.
       The region keeps the registration for its own life. */
    lane->ring_registration = shadowspill_remote_region_register_local(
        lane->region, NULL, lane->ring, lane->ring_bytes
    );
    if (lane->ring_registration == NULL) {
        fprintf(
            stderr,
            "shadowspill network: the NIC would not register the staging "
            "ring, %llu bytes\n",
            (unsigned long long)lane->ring_bytes
        );
        return -1;
    }
    return 0;
}

void remote_lane_stage_destroy(RemoteLane *lane) {
    if (lane->ring == NULL) {
        return;
    }
    (void)lane->base.backend->unregister_host_memory(
        lane->base.backend->state, lane->ring, lane->ring_bytes
    );
    (void)munmap(lane->ring, (size_t)lane->ring_bytes);
    lane->ring = NULL;
}

void *remote_lane_stage_slot(const RemoteLane *lane, uint64_t chunk) {
    const uint64_t slot = chunk % lane->tuning.ring_slots;
    return (char *)lane->ring + slot * lane->tuning.chunk_bytes;
}

/*
 * Pieces are numbered over the lane's whole life, and the difference is a
 * correctness one: a slot is reused every `ring_slots` pieces *across*
 * transfers, so whether the NIC has emptied it or the device has drained it
 * is a question about the lane's count, not this transfer's. Counting from
 * zero each transfer would let a transfer's opening pieces skip the check
 * and copy over -- or post over -- a slot the other side was still using.
 *
 *   fetch   wait: the NIC filled this piece's slot      -> copy it to the pool
 *   evict   wait: the NIC emptied the slot being reused -> refill it
 *
 * and both then store the device's word for the piece.
 */
int remote_lane_stage_enqueue(RemoteLane *lane, const Work *plan) {
    const ShadowSpillBackend *const backend = lane->base.backend;
    const ShadowSpillBackendStream stream = lane->base.stream;
    const uint64_t chunk_bytes = lane->tuning.chunk_bytes;
    const uint32_t slots = lane->tuning.ring_slots;
    for (uint32_t index = 0U; index < plan->chunks; ++index) {
        const uint64_t offset = (uint64_t)index * chunk_bytes;
        const uint64_t bytes = plan->bytes - offset < chunk_bytes
            ? plan->bytes - offset : chunk_bytes;
        const uint64_t chunk = plan->first_chunk + index;
        void *const slot = remote_lane_stage_slot(lane, chunk);
        char *const local = (char *)plan->local + offset;
        if (plan->to_remote) {
            if (chunk >= slots && backend->wait_value(
                    backend->state, stream, lane->signals, SIGNAL_NIC,
                    chunk + 1U - slots
                ) != 0) {
                return -1;
            }
            if (backend->copy_device_to_host(
                    backend->state, slot, local, bytes, stream
                ) != 0) {
                return -1;
            }
        } else {
            if (backend->wait_value(
                    backend->state, stream, lane->signals, SIGNAL_NIC,
                    chunk + 1U
                ) != 0) {
                return -1;
            }
            if (backend->copy_host_to_device(
                    backend->state, local, slot, bytes, stream
                ) != 0) {
                return -1;
            }
        }
        if (backend->write_value(
                backend->state, stream, lane->signals, SIGNAL_DEVICE,
                chunk + 1U
            ) != 0) {
            return -1;
        }
    }
    return 0;
}

/* An evict's piece may post once the device has filled its slot; a fetch's
   once the device has drained the slot's previous occupant, which the first
   ring's worth never had. Asked, never waited for: if the answer is no the
   thread goes and retires, which is what moves the device. */
int remote_lane_stage_ready(
    const RemoteLane *lane, const Work *work, uint64_t chunk
) {
    const uint64_t slots = lane->tuning.ring_slots;
    if (work->to_remote) {
        return remote_lane_device_reached(lane, chunk + 1U);
    }
    return chunk < slots
        ? 1 : remote_lane_device_reached(lane, chunk - slots + 1U);
}
