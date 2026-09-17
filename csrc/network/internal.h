/* What csrc/network/ holds, and nothing outside it sees. */

#ifndef SHADOWSPILL_NETWORK_INTERNAL_H
#define SHADOWSPILL_NETWORK_INTERNAL_H

#include <stdint.h>

/* The contract this library implements, and the kind enum naming its entry.
   Nothing else of the runtime's is needed, and nothing of the runtime's
   internals is reachable. */
#include <shadowspill/runtime/descriptions.h>
#include <shadowspill/runtime/pool_memory.h>

/* This library links verbs from Phase 4 onward; libshadowspill still does
   not, and neither does anything that includes it. */
#include <infiniband/verbs.h>

/* The one symbol this library exports; everything else stays hidden, as the
   build's default visibility says. */
#define SHADOWSPILL_NETWORK_API __attribute__((visibility("default")))

/*
 * This directory owns no public header. Everything it implements is declared
 * by the runtime -- a pool-memory pair here, a lane later -- so what it exports
 * is one static descriptor and nothing else. A caller that loads it therefore
 * needs the runtime's headers and none of these.
 *
 * Nothing in memory/, transfers/ or sync/ may include this file.
 */

/* ------------------------------------------------------------------------
 * The control channel
 *
 * One TCP connection to one daemon, carrying three verbs. It is how a region on
 * another machine is obtained, connected to, and given back. It moves no
 * payload: the bytes travel over the NIC, addressed by what these verbs return.
 *
 *   allocate <bytes> <selector>       ->  ok <address> <rkey>      | error ...
 *   connect <qpn> <psn> <lid> <gid>   ->  ok <qpn> <psn> <lid> <gid> | error ...
 *   free <address>                    ->  ok                       | error ...
 *
 * Line-oriented text, one request per line, because being readable by hand was
 * the argument for a TCP channel rather than a second RDMA path. There is no
 * version handshake: a mismatched pair is diagnosed by reading the two sources.
 * Text is what makes that affordable -- an unrecognised verb or a wrong arity
 * is a visible error line, where binary framing could misread a field and
 * return a plausible address. The residual risk is a verb whose *meaning*
 * changes while its shape does not, so **extend the protocol rather than edit
 * it** -- which is exactly what `connect` did to a two-verb protocol.
 *
 * WHY THE HANDSHAKE IS HERE AND NOT IN librdmacm. An RC queue pair is
 * established two-sided: both ends create one and exchange a queue-pair
 * number, a packet sequence number and a GID before either can post. That is
 * three short fields each way, and a channel that already exists carries them
 * -- where rdmacm would add a library on both sides, its own event channel and
 * its own port, and a second connection mechanism beside this one. The
 * handshake stays readable by hand like the rest.
 *
 * `connect` is ordered: the daemon transitions *its* queue pair to RTR and RTS
 * before it replies, so by the time this side reads the reply the far end is
 * ready to receive. This side then transitions its own. Neither end posts
 * before both have.
 */

/* Longest line either side sends or accepts, newline included. A GID is 32 hex
   characters, an address and a key are hex, a selector is short. */
#define SHADOWSPILL_NETWORK_LINE_BYTES 512U

/*
 * What the two ends tell each other to bring an RC queue pair up.
 *
 * Both a LID and a GID travel, because which one matters depends on the link
 * and neither side should have to ask. **InfiniBand addresses by LID**, with
 * no global route header unless the path crosses subnets; **RoCE has no LID at
 * all** and addresses by GID. Carrying both costs 4 hex characters and means
 * one protocol serves either link, with each end reading the field its own
 * port's link layer calls for.
 *
 * The GID is 32 hex characters -- the 16 bytes `ibv_query_gid` fills in, most
 * significant first.
 */
typedef struct ShadowSpillEndpointIdentity {
    uint32_t queue_pair_number;
    uint32_t packet_sequence_number;
    /* Meaningful on InfiniBand; zero and ignored on RoCE. */
    uint16_t local_identifier;
    uint8_t global_identifier[16];
} ShadowSpillEndpointIdentity;

typedef struct ShadowSpillControlChannel {
    int socket;
} ShadowSpillControlChannel;

/*
 * Connect to `host`:`port`. Both borrowed for the call.
 *
 * Returns 0, or -1 with the channel left closed -- there is no partial state
 * to unwind, which is why every failure in this file reports the same way.
 */
int shadowspill_control_connect(
    ShadowSpillControlChannel *channel, const char *host, const char *port
);

/* Close the connection. Idempotent, and the daemon frees everything it held
   for this connection when it sees the close -- the backstop that runs when
   `free` never arrives. */
void shadowspill_control_close(ShadowSpillControlChannel *channel);

/*
 * Ask the daemon for `capacity` bytes of the memory `selector` names, and
 * report where they live in its address space.
 *
 * `selector` is opaque here: it is forwarded as given and never parsed, so
 * which memory a daemon can serve is the daemon's business and adding one
 * needs no change on this side. `address` is an address on *that* machine and
 * is never dereferenced here.
 */
int shadowspill_control_allocate(
    ShadowSpillControlChannel *channel,
    uint64_t capacity,
    const char *selector,
    uint64_t *address,
    uint32_t *key
);

/* Give a region back. Called at pool close, before the connection is shut. */
int shadowspill_control_free(
    ShadowSpillControlChannel *channel, uint64_t address
);

/*
 * Exchange queue-pair identities. `local` is what this side has created and is
 * sent as given; `remote` is filled in with what the daemon created.
 *
 * The daemon has already moved its own queue pair to RTS when this returns 0,
 * so the caller may transition its own and then post. A non-zero return leaves
 * nothing to unwind on this side -- the daemon frees what it built when the
 * connection closes, the same backstop that frees a region.
 */
int shadowspill_control_connect_endpoint(
    ShadowSpillControlChannel *channel,
    const ShadowSpillEndpointIdentity *local,
    ShadowSpillEndpointIdentity *remote
);

/* ------------------------------------------------------------------------
 * Tuning
 *
 * Every knob that affects how fast or how reliably bytes move, in one place,
 * read once from the environment. **None of it is in a pool's configuration**,
 * for the reason the endpoint's own detection is not: these are properties of
 * a NIC and a fabric, not of a plan, and a caller cannot know them.
 *
 * They are environment variables rather than a config struct a caller fills in
 * because the shape of an investigation is "run it again with one thing
 * changed", and that should not require rebuilding a configuration. The
 * resolved values are logged at create, so a run's own output says what it
 * used.
 *
 * The defaults are what this hardware wanted. What each one is *for*:
 *
 *   QUEUE_PAIRS        One saturates 25 Gb/s; a faster card may need several,
 *                      because a queue pair is served by one send engine.
 *   SEND_DEPTH         Work requests outstanding per queue pair. Too few and
 *                      the pipe drains between completions.
 *   RECEIVE_DEPTH      Nearly irrelevant here -- one-sided reads and writes
 *                      post no receives -- but a queue pair still has one.
 *   OUTSTANDING_READS  **Caps RDMA read bandwidth directly.** An RC queue pair
 *                      may have only this many reads in flight at once, so it
 *                      bounds the fetch direction no matter how deep the send
 *                      queue is. The HCA has a hard limit; asking for more
 *                      than it allows fails the transition.
 *   CHUNK_BYTES        How large one piece of a transfer is, and so how much
 *                      one ring slot holds.
 *   RING_SLOTS         How many chunks may be in flight. Two is enough to
 *                      overlap the host copy with the transfer, which is what
 *                      matters: measured with one slot, the link ran at
 *                      2857 MiB/s -- 98 % of what `ib_read_bw` gets -- while
 *                      the whole transfer ran at 2406, and that serialisation
 *                      was the entire gap. More than two buys little, since
 *                      one stage is six times faster than the other.
 *   SIGNAL_EVERY       Completions are expensive; only every Nth work request
 *                      need be signalled, with the last always signalled.
 *   SPIN_NANOSECONDS   How long the lane's thread watches for the next
 *                      transfer before sleeping. Waking it costs ~1.8 us of a
 *                      small transfer's ~12, and transfers arrive in batches,
 *                      so a brief watch often catches the next one without a
 *                      context switch. Zero disables it; it is bounded so that
 *                      an idle lane still costs no CPU.
 *   TRAFFIC_CLASS      RoCE congestion control reads this. Zero is right on a
 *                      quiet lab subnet and is exactly what you would change
 *                      first on a fabric with PFC or DCQCN.
 *   SERVICE_LEVEL      The InfiniBand equivalent, and a RoCE priority.
 *   PATH_MTU           Discovered from the port. Lowering it can help under
 *                      congestion, which is the only reason to set it.
 *   TIMEOUT / RETRIES  How long a queue pair persists against a peer that has
 *                      stopped answering, which is how long a dead daemon
 *                      takes to become an error rather than a hang.
 *   DEVICE / GID_INDEX Override the endpoint's own detection.
 */
typedef struct ShadowSpillNetworkTuning {
    uint32_t queue_pairs;
    uint32_t send_depth;
    uint32_t receive_depth;
    uint32_t outstanding_reads;
    uint64_t chunk_bytes;
    uint32_t ring_slots;
    uint32_t signal_every;
    /* How long the lane's thread watches for more work before sleeping. */
    uint64_t spin_nanoseconds;
    uint32_t traffic_class;
    uint32_t service_level;
    /* Zero means "whatever the port negotiated". */
    uint32_t path_mtu;
    uint32_t timeout;
    uint32_t retry_count;
    uint32_t rnr_retry;
    const char *device;
    /* Negative means "discover one". */
    int gid_index;
} ShadowSpillNetworkTuning;

/* Read every knob once and report what was resolved. Safe to call repeatedly;
   it recomputes rather than caching, and nothing here is hot. */
void shadowspill_network_tuning_read(ShadowSpillNetworkTuning *tuning);

/* One line per value, to stderr, so a run says what it used. */
void shadowspill_network_tuning_report(const ShadowSpillNetworkTuning *tuning);

/* ------------------------------------------------------------------------
 * The endpoint
 *
 * One RDMA device, one queue pair, one peer. Everything about it is
 * discovered: which device reaches the peer, which of its ports is active,
 * whether that port is InfiniBand or RoCE, which GID to advertise, and what
 * MTU the port negotiated. **None of it appears in a pool's configuration**,
 * because none of it is a property of the plan -- a caller cannot know this
 * machine's hardware and does not have to.
 *
 * `SHADOWSPILL_NETWORK_DEVICE` names a device when detection is wrong. It is
 * an environment variable rather than a config field deliberately: an escape
 * hatch a caller never has to know exists.
 */

/*
 * WHY THERE ARE SEVERAL QUEUE PAIRS, WHEN ONE IS ENOUGH HERE.
 *
 * One queue pair reaches line rate on a 25 Gb/s link -- measured, 2921 MiB/s
 * against `ib_read_bw`'s 2921 MiB/s. It stops being enough on a faster one: a
 * single queue pair is served by one of the NIC's send engines, and past some
 * rate the way to use the rest is to post across several. Which rate that is
 * depends on the card, so it is a number to configure rather than a fact to
 * hard-code.
 *
 * Making this an array now costs a loop; retrofitting it later would touch the
 * handshake, the protocol and the posting path at once. The count is
 * `SHADOWSPILL_NETWORK_QUEUE_PAIRS`, default 1 -- an environment variable
 * rather than a pool configuration field, because it is a property of this
 * machine's NIC and not of the plan.
 *
 * Transfers are spread across them round-robin. They share one completion
 * queue, so the lane's thread still blocks in one place however many there
 * are.
 */
#define SHADOWSPILL_NETWORK_MAX_QUEUE_PAIRS 16U
#define SHADOWSPILL_NETWORK_MAX_RING_SLOTS 8U

/*
 * ONE COMPLETION QUEUE PER QUEUE PAIR, NOT ONE PER ENDPOINT.
 *
 * A shared completion queue looks like an economy and is a correctness bug.
 * `ibv_poll_cq` returns whatever completed, not what the caller was waiting
 * for, so two lanes polling one queue **consume each other's completions** --
 * and each records what it took in its own state, where the other can never
 * find it. Both then wait for something that already arrived.
 *
 * It survived every sequential test. Calibration is the first thing to drive a
 * fetch and an evict at the same time, which is precisely what it exists to
 * measure, and it deadlocked on the first attempt.
 */
typedef struct ShadowSpillEndpoint {
    struct ibv_context *device;
    struct ibv_pd *protection_domain;
    struct ibv_cq *completion_queues[SHADOWSPILL_NETWORK_MAX_QUEUE_PAIRS];
    struct ibv_qp *queue_pairs[SHADOWSPILL_NETWORK_MAX_QUEUE_PAIRS];
    uint32_t queue_pair_count;
    uint8_t port;
    int gid_index;
    uint8_t link_layer;
    uint16_t local_identifier;
    enum ibv_mtu path_mtu;
    char name[64];
    /* Resolved once at open, so everything downstream reads the same numbers
       and a run's log says which they were. */
    ShadowSpillNetworkTuning tuning;
} ShadowSpillEndpoint;

/*
 * Open the device that can reach `host` and build `queue_pairs` of them, each
 * in INIT, over one completion queue.
 *
 * `depth` is the one number that sizes each queue pair, the completion queue
 * and later the staging ring -- they are the same quantity. `queue_pairs` of
 * zero means read `SHADOWSPILL_NETWORK_QUEUE_PAIRS`, defaulting to one.
 */
int shadowspill_endpoint_open(
    ShadowSpillEndpoint *endpoint,
    const char *host,
    uint32_t depth,
    uint32_t queue_pairs
);

/* What to send the peer so it can reach queue pair `index`. Each is connected
   separately, so each is handshaken separately. */
int shadowspill_endpoint_identity(
    const ShadowSpillEndpoint *endpoint,
    uint32_t index,
    ShadowSpillEndpointIdentity *identity
);

/* INIT -> RTR -> RTS for one queue pair against what the peer sent back.
   Nothing may be posted on it before this, and everything after. */
int shadowspill_endpoint_connect(
    ShadowSpillEndpoint *endpoint,
    uint32_t index,
    const ShadowSpillEndpointIdentity *peer
);

void shadowspill_endpoint_close(ShadowSpillEndpoint *endpoint);

/* ------------------------------------------------------------------------
 * The Remote pool kind
 *
 * What `ShadowSpillMemoryPoolDescription.configuration` points at for a Remote
 * pool. Borrowed for `acquire`, which copies what it needs -- so a caller may
 * build one on the stack.
 *
 * It names a machine and what to ask it for, and nothing else: the capacity is
 * the pool's, and the key comes back from the daemon.
 */
typedef struct ShadowSpillRemotePoolConfiguration {
    const char *host;
    const char *port;
    /* Which memory the daemon should serve, in its own vocabulary. NULL means
       "host", the only kind a daemon serves today. */
    const char *selector;
} ShadowSpillRemotePoolConfiguration;

/* The entry to append to `ShadowSpillRuntimeConfig.pool_memory`. An object
   rather than a getter, so the library descriptor that points at it is a
   static initializer and nothing at all runs before the loader reads it. */
extern const ShadowSpillPoolMemoryDescription shadowspill_remote_pool_memory;

/*
 * What a lane needs to know about a region it was handed an address in: where
 * the bytes really are, and the key and connection that reach them.
 *
 * The first fields of the region record, exposed read-only. A lane is given
 * local addresses -- unique, and unreadable by construction -- and turns one
 * back into a remote address by taking its offset from `reservation` and
 * adding it to `address`.
 */
typedef struct ShadowSpillRemoteRegion {
    ShadowSpillControlChannel channel;
    void *reservation;
    uint64_t capacity;
    uint64_t address;
    uint32_t key;
    /*
     * One connection to one daemon means one endpoint, so it belongs here
     * rather than to a lane: several routes may move bytes to the same remote
     * pool, and they share the queue pairs rather than each building their
     * own. It is built during `acquire`, which is where bring-up belongs --
     * there is a failure path and a strict unwind there already, and a lane's
     * `create` would be a second place to write one.
     */
    ShadowSpillEndpoint endpoint;
} ShadowSpillRemoteRegion;

/* Which region an address belongs to, or NULL when it is local. This is how a
   lane tells the two ends of a copy apart: exactly one of them is remote. */
const ShadowSpillRemoteRegion *shadowspill_remote_region_for(const void *address);

/*
 * Claim a queue pair, and its completion queue, for one lane's exclusive use.
 *
 * Returns the index claimed, or -1 when the region has none left. Two lanes
 * serve a remote pool -- one per direction -- and they run at the same time,
 * so each needs its own or they steal each other's completions. Claims are
 * never released: a lane lives as long as its runtime.
 */
int shadowspill_remote_region_claim_queue_pair(
    const ShadowSpillRemoteRegion *region
);

/* The lane that moves bytes to and from a remote pool, one entry per
   direction, appended to `ShadowSpillRuntimeConfig.lanes`. */
extern const ShadowSpillLaneDescription shadowspill_remote_lanes[2];

#endif /* SHADOWSPILL_NETWORK_INTERNAL_H */
