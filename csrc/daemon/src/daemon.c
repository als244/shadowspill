/*
 * shadowspill_memory_daemon -- holds memory on this machine so a ShadowSpill
 * runtime on another can use it as a pool.
 *
 * Links ibverbs and libc, and not libshadowspill: the far side of a remote
 * pool is not a second ShadowSpill. See ../README.md for the protocol and for
 * what this process does between bring-up and teardown (nothing).
 */

/* MAP_ANONYMOUS is not in the strict ISO C11 the tree compiles as; the same
   line and the same reason as memory_pool/internal.h. */
#define _DEFAULT_SOURCE

#include <arpa/inet.h>
#include <errno.h>
#include <inttypes.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <unistd.h>

#include <infiniband/verbs.h>

#define LINE_BYTES 512U

/*
 * One number for the selector's length, spelled both as a buffer size and as
 * the width sscanf is told. A literal "%255s" written beside a buffer somebody
 * later shrinks is a stack overwrite with nothing to notice it, so neither is
 * written by hand.
 */
#define SELECTOR_WIDTH 63
#define STRINGIFY_INNER(value) #value
#define STRINGIFY(value) STRINGIFY_INNER(value)
#define SELECTOR_SCAN "%" STRINGIFY(SELECTOR_WIDTH) "s"

/*
 * One connection at a time, so every region here belongs to the connection
 * being served and there is nothing to lock. A list rather than a single slot
 * because a runtime may hold more than one remote pool on one daemon in a
 * future that costs nothing to allow: allocate appends, free unlinks, and
 * closing the connection walks what is left.
 */
typedef struct Region {
    struct ibv_mr *registration;
    void *base;
    uint64_t bytes;
    struct Region *next;
} Region;

/*
 * The queue pair this connection's peer posts against. One per connection,
 * built by `connect` and torn down with everything else when the socket
 * closes.
 *
 * It exists so that an rkey means something: a memory region's key is usable
 * only by a queue pair in the *same protection domain*, so the far side cannot
 * simply invent one. That is the whole of why this process takes part in
 * connection setup despite doing nothing at all during a transfer -- the data
 * path is one-sided, the *setup* is not.
 */
/*
 * The queue pairs this connection's peer posts against. The peer decides how
 * many it wants and asks for them one at a time, so this grows by one per
 * `connect`: a peer that saturates its link with one asks once, and a peer on
 * a faster card asks several times. They share a completion queue this end
 * never reads, because this end never posts.
 */
#define MAX_QUEUE_PAIRS 16

typedef struct Endpoint {
    struct ibv_cq *completion_queue;
    struct ibv_qp *queue_pairs[MAX_QUEUE_PAIRS];
    uint32_t queue_pair_count;
    uint32_t packet_sequence_number;
    uint8_t port;
    int gid_index;
    /* What this port is: InfiniBand addresses by LID, RoCE by GID, and the
       two take different paths through the RTR transition below. */
    uint8_t link_layer;
    uint16_t local_identifier;
    enum ibv_mtu path_mtu;
} Endpoint;

typedef struct Daemon {
    struct ibv_context *device;
    struct ibv_pd *protection_domain;
    uint8_t port;
    /* -1 means "discover it"; --gid overrides. */
    int gid_index;
    Region *regions;
    Endpoint endpoint;
} Daemon;

/* Depth of the queue pair and its completion queue. This end posts nothing --
   every transfer is the peer's one-sided read or write -- so the depth is what
   the peer may have outstanding against us, not what we will ever use. */
#define ENDPOINT_DEPTH 64

/* Defined with the rest of the queue-pair code below; released here by the
   same teardown that releases every region, since both die with the
   connection they belong to. */
static void endpoint_destroy(Endpoint *endpoint);

/* ---------------------------------------------------------------- verbs */

/* Does this device have a port a peer could reach us on? */
static int has_an_active_port(struct ibv_context *device) {
    struct ibv_device_attr attributes;
    if (ibv_query_device(device, &attributes) != 0) {
        return 0;
    }
    for (uint8_t port = 1U; port <= attributes.phys_port_cnt; ++port) {
        struct ibv_port_attr port_attributes;
        if (ibv_query_port(device, port, &port_attributes) == 0 &&
            port_attributes.state == IBV_PORT_ACTIVE) {
            return 1;
        }
    }
    return 0;
}

/*
 * Pick a device. A named one is taken as given -- the operator said so.
 * Unnamed, prefer one with an ACTIVE port and fall back to any that opens.
 *
 * The preference is not cosmetic. Registering memory needs no link at all, so
 * a daemon on a device whose port is down serves `allocate` perfectly and
 * hands back a key no peer can ever use. The boxes here make that easy to hit:
 * the active device is mlx5_1 on one and mlx5_0 on the other, so "the first
 * device that opens" is right on one machine and silently wrong on the other.
 */
static struct ibv_context *open_device(const char *name) {
    int count = 0;
    struct ibv_device **devices = ibv_get_device_list(&count);
    if (devices == NULL || count == 0) {
        fprintf(stderr, "daemon: no RDMA device found\n");
        if (devices != NULL) {
            ibv_free_device_list(devices);
        }
        return NULL;
    }
    struct ibv_context *opened = NULL;
    struct ibv_context *fallback = NULL;
    const char *fallback_name = NULL;
    for (int index = 0; index < count; ++index) {
        const char *device_name = ibv_get_device_name(devices[index]);
        if (name != NULL && strcmp(name, device_name) != 0) {
            continue;
        }
        struct ibv_context *candidate = ibv_open_device(devices[index]);
        if (candidate == NULL) {
            continue;
        }
        if (name != NULL || has_an_active_port(candidate)) {
            opened = candidate;
            fprintf(stderr, "daemon: using %s\n", device_name);
            break;
        }
        if (fallback == NULL) {
            fallback = candidate;
            fallback_name = device_name;
        } else {
            (void)ibv_close_device(candidate);
        }
    }
    if (opened == NULL && fallback != NULL) {
        opened = fallback;
        fallback = NULL;
        fprintf(
            stderr,
            "daemon: using %s, whose ports are all down -- it can register "
            "memory but no peer can reach it\n",
            fallback_name
        );
    }
    if (fallback != NULL) {
        (void)ibv_close_device(fallback);
    }
    if (opened == NULL) {
        fprintf(
            stderr, "daemon: could not open %s\n", name != NULL ? name : "any device"
        );
    }
    ibv_free_device_list(devices);
    return opened;
}

/*
 * An anonymous private mapping rather than malloc, so the registered region is
 * page-aligned and the C allocator never hands part of it to anything else.
 * The same reasoning as a pinned-host pool's memory on the other side.
 */
static Region *region_allocate(Daemon *daemon, uint64_t bytes) {
    void *base = mmap(
        NULL, (size_t)bytes, PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0
    );
    if (base == MAP_FAILED) {
        return NULL;
    }
    struct ibv_mr *registration = ibv_reg_mr(
        daemon->protection_domain, base, (size_t)bytes,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
            IBV_ACCESS_REMOTE_WRITE
    );
    if (registration == NULL) {
        (void)munmap(base, (size_t)bytes);
        return NULL;
    }
    Region *region = calloc(1U, sizeof(*region));
    if (region == NULL) {
        (void)ibv_dereg_mr(registration);
        (void)munmap(base, (size_t)bytes);
        return NULL;
    }
    *region = (Region){
        .registration = registration,
        .base = base,
        .bytes = bytes,
        .next = daemon->regions,
    };
    daemon->regions = region;
    return region;
}

static void region_destroy(Region *region) {
    fprintf(
        stderr, "daemon: releasing %" PRIu64 " bytes at %p\n", region->bytes,
        region->base
    );
    (void)ibv_dereg_mr(region->registration);
    (void)munmap(region->base, (size_t)region->bytes);
    free(region);
}

/* Free everything this connection held. Runs on an orderly close and on a
   peer that vanished alike, which is why there is no registry and no
   heartbeat: both look the same from here and both end here. */
static void release_all(Daemon *daemon) {
    if (daemon->endpoint.queue_pair_count != 0U ||
        daemon->endpoint.completion_queue != NULL) {
        fprintf(
            stderr, "daemon: tearing down %u queue pair(s)\n",
            (unsigned)daemon->endpoint.queue_pair_count
        );
        endpoint_destroy(&daemon->endpoint);
    }
    Region *region = daemon->regions;
    while (region != NULL) {
        Region *next = region->next;
        region_destroy(region);
        region = next;
    }
    daemon->regions = NULL;
}

static int release_one(Daemon *daemon, uint64_t address) {
    Region **link = &daemon->regions;
    while (*link != NULL) {
        Region *region = *link;
        if ((uint64_t)(uintptr_t)region->base == address) {
            *link = region->next;
            region_destroy(region);
            return 0;
        }
        link = &region->next;
    }
    return -1;
}

/* ------------------------------------------------------- the queue pair */

/*
 * A GID on the wire is 32 hex characters, most significant byte first. The
 * same two functions exist on the other side; they are written twice rather
 * than shared because this program deliberately links nothing of
 * ShadowSpill's, and a wire format simple enough to write twice is a wire
 * format simple enough to read by hand.
 */
static void format_gid(const uint8_t bytes[16], char text[33]) {
    static const char digits[] = "0123456789abcdef";
    for (unsigned index = 0U; index < 16U; ++index) {
        text[index * 2U] = digits[(bytes[index] >> 4) & 0x0FU];
        text[index * 2U + 1U] = digits[bytes[index] & 0x0FU];
    }
    text[32] = '\0';
}

static int parse_gid(const char *text, uint8_t bytes[16]) {
    if (strlen(text) != 32U) {
        return -1;
    }
    for (unsigned index = 0U; index < 16U; ++index) {
        unsigned value = 0U;
        if (sscanf(text + index * 2U, "%2x", &value) != 1) {
            return -1;
        }
        bytes[index] = (uint8_t)value;
    }
    return 0;
}

/*
 * Which GID this port should advertise.
 *
 * A RoCE port carries several, and they are not interchangeable. On both boxes
 * here the table is the same four: link-local RoCE v1, link-local RoCE v2,
 * IPv4-mapped RoCE v1, IPv4-mapped RoCE v2. **Index 0 is the wrong one** -- it
 * is link-local v1, and two machines on a routed subnet need the IPv4-mapped
 * v2 entry, which happens to be index 3 here and is not guaranteed to be
 * index 3 anywhere else.
 *
 * So it is discovered rather than assumed: prefer RoCE v2 with an IPv4-mapped
 * address, then any RoCE v2, then give up rather than silently advertise a GID
 * no peer can route to. An operator who knows better passes --gid.
 *
 * Getting this wrong does not fail loudly. The handshake succeeds, both queue
 * pairs reach RTS, and then every transfer times out with nothing to say why.
 */
static int is_ipv4_mapped(const union ibv_gid *gid) {
    static const uint8_t prefix[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xFF};
    return memcmp(gid->raw, prefix, sizeof(prefix)) == 0;
}

static int find_gid_index(struct ibv_context *device, uint8_t port) {
    struct ibv_port_attr attributes;
    if (ibv_query_port(device, port, &attributes) != 0) {
        return -1;
    }
    /* InfiniBand does not choose a GID this way: it addresses by LID, and
       index 0 is the right and only answer when a GID is needed at all. */
    if (attributes.link_layer == IBV_LINK_LAYER_INFINIBAND) {
        return 0;
    }
    int any_roce_v2 = -1;
    for (int index = 0; index < (int)attributes.gid_tbl_len; ++index) {
        struct ibv_gid_entry entry;
        if (ibv_query_gid_ex(device, port, (uint32_t)index, &entry, 0) != 0) {
            continue;
        }
        if (entry.gid_type != IBV_GID_TYPE_ROCE_V2) {
            continue;
        }
        if (is_ipv4_mapped(&entry.gid)) {
            return index;
        }
        if (any_roce_v2 < 0) {
            any_roce_v2 = index;
        }
    }
    return any_roce_v2;
}

/* Which port of the opened device is usable. */
static int find_active_port(Daemon *daemon) {
    struct ibv_device_attr attributes;
    if (ibv_query_device(daemon->device, &attributes) != 0) {
        return -1;
    }
    for (uint8_t port = 1U; port <= attributes.phys_port_cnt; ++port) {
        struct ibv_port_attr port_attributes;
        if (ibv_query_port(daemon->device, port, &port_attributes) == 0 &&
            port_attributes.state == IBV_PORT_ACTIVE) {
            daemon->port = port;
            return 0;
        }
    }
    return -1;
}

static void endpoint_destroy(Endpoint *endpoint) {
    for (uint32_t index = 0U; index < endpoint->queue_pair_count; ++index) {
        if (endpoint->queue_pairs[index] != NULL) {
            (void)ibv_destroy_qp(endpoint->queue_pairs[index]);
        }
    }
    if (endpoint->completion_queue != NULL) {
        (void)ibv_destroy_cq(endpoint->completion_queue);
    }
    *endpoint = (Endpoint){0};
}

/*
 * Discover the port, GID and MTU once, on the first `connect`. Everything here
 * is a property of this machine's hardware and identical for every queue pair
 * the peer asks for.
 */
static int endpoint_prepare(Daemon *daemon, Endpoint *endpoint) {
    if (daemon->port == 0U && find_active_port(daemon) != 0) {
        fprintf(stderr, "daemon: no active port to connect on\n");
        return -1;
    }
    endpoint->port = daemon->port;
    struct ibv_port_attr port_attributes;
    if (ibv_query_port(daemon->device, endpoint->port, &port_attributes) != 0) {
        return -1;
    }
    endpoint->link_layer = port_attributes.link_layer;
    endpoint->local_identifier = port_attributes.lid;
    /* From the port, not a constant. A hardcoded 1024 would quietly cap a
       port that negotiated 4096 at a quarter of its frame size. */
    endpoint->path_mtu = port_attributes.active_mtu;
    endpoint->gid_index = daemon->gid_index >= 0
        ? daemon->gid_index
        : find_gid_index(daemon->device, endpoint->port);
    if (endpoint->gid_index < 0) {
        fprintf(
            stderr,
            "daemon: port %u advertises no RoCE v2 GID; pass --gid if you know "
            "which index to use\n",
            (unsigned)endpoint->port
        );
        return -1;
    }
    endpoint->packet_sequence_number = 0U;
    endpoint->completion_queue = ibv_create_cq(
        daemon->device, ENDPOINT_DEPTH * MAX_QUEUE_PAIRS, NULL, NULL, 0
    );
    return endpoint->completion_queue == NULL ? -1 : 0;
}

/* One more queue pair for the peer, in INIT. It cannot receive yet: that needs
   the peer's identity, which arrives in the same request that asks for this. */
static int endpoint_add_queue_pair(Daemon *daemon, Endpoint *endpoint) {
    if (endpoint->queue_pair_count >= MAX_QUEUE_PAIRS) {
        return -1;
    }
    if (endpoint->completion_queue == NULL &&
        endpoint_prepare(daemon, endpoint) != 0) {
        return -1;
    }
    struct ibv_qp_init_attr init = {
        .send_cq = endpoint->completion_queue,
        .recv_cq = endpoint->completion_queue,
        .qp_type = IBV_QPT_RC,
        .cap = {
            .max_send_wr = ENDPOINT_DEPTH,
            .max_recv_wr = ENDPOINT_DEPTH,
            .max_send_sge = 1,
            .max_recv_sge = 1,
        },
    };
    struct ibv_qp *const created =
        ibv_create_qp(daemon->protection_domain, &init);
    if (created == NULL) {
        return -1;
    }
    struct ibv_qp_attr attributes = {
        .qp_state = IBV_QPS_INIT,
        .pkey_index = 0,
        .port_num = endpoint->port,
        /* The peer reads and writes our memory; we initiate nothing. */
        .qp_access_flags = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE |
                           IBV_ACCESS_LOCAL_WRITE,
    };
    if (ibv_modify_qp(
            created, &attributes,
            IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                IBV_QP_ACCESS_FLAGS
        ) != 0) {
        (void)ibv_destroy_qp(created);
        return -1;
    }
    endpoint->queue_pairs[endpoint->queue_pair_count++] = created;
    return 0;
}

/*
 * INIT -> RTR -> RTS, using what the peer just told us. After this the peer
 * may post; this end still posts nothing, ever.
 *
 * Each transition reports which one failed and why. A handshake has several
 * ways to fail that look identical from the other end -- an unroutable GID, a
 * GID index the port does not have, an MTU the path cannot carry -- and this
 * runs on a machine nobody is watching, so the log is the only witness.
 */
static int endpoint_connect(
    Endpoint *endpoint,
    struct ibv_qp *queue_pair,
    uint32_t peer_queue_pair,
    uint32_t peer_sequence,
    uint16_t peer_identifier,
    const uint8_t peer_gid[16]
) {
    struct ibv_qp_attr ready_to_receive = {
        .qp_state = IBV_QPS_RTR,
        .path_mtu = endpoint->path_mtu,
        .dest_qp_num = peer_queue_pair,
        .rq_psn = peer_sequence,
        .max_dest_rd_atomic = 16,
        .min_rnr_timer = 12,
        .ah_attr = {.port_num = endpoint->port},
    };
    /*
     * The one place the two link layers differ. InfiniBand reaches a peer by
     * LID on the local subnet and needs no global route header; RoCE has no
     * LID and reaches it by GID, always through a GRH. Getting this backwards
     * is not a loud failure -- both ends still reach RTS, and every transfer
     * then times out with nothing to say why.
     */
    if (endpoint->link_layer == IBV_LINK_LAYER_INFINIBAND) {
        ready_to_receive.ah_attr.is_global = 0;
        ready_to_receive.ah_attr.dlid = peer_identifier;
        ready_to_receive.ah_attr.sl = 0;
    } else {
        ready_to_receive.ah_attr.is_global = 1;
        ready_to_receive.ah_attr.grh.hop_limit = 64;
        ready_to_receive.ah_attr.grh.sgid_index = (uint8_t)endpoint->gid_index;
        memcpy(ready_to_receive.ah_attr.grh.dgid.raw, peer_gid, 16U);
    }
    const int to_receive = ibv_modify_qp(
        queue_pair, &ready_to_receive,
        IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
            IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER
    );
    if (to_receive != 0) {
        fprintf(
            stderr,
            "daemon: INIT->RTR failed (%s) for peer queue pair %" PRIx32
            " on port %u gid index %d\n",
            strerror(to_receive), peer_queue_pair, (unsigned)endpoint->port,
            endpoint->gid_index
        );
        return -1;
    }
    struct ibv_qp_attr ready_to_send = {
        .qp_state = IBV_QPS_RTS,
        .timeout = 14,
        .retry_cnt = 7,
        .rnr_retry = 7,
        .sq_psn = endpoint->packet_sequence_number,
        .max_rd_atomic = 16,
    };
    const int to_send = ibv_modify_qp(
        queue_pair, &ready_to_send,
        IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
            IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC
    );
    if (to_send != 0) {
        fprintf(stderr, "daemon: RTR->RTS failed (%s)\n", strerror(to_send));
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------- protocol */

static int write_all(int socket, const char *bytes, size_t count) {
    size_t written = 0U;
    while (written < count) {
        const ssize_t step = send(
            socket, bytes + written, count - written, MSG_NOSIGNAL
        );
        if (step < 0 && errno == EINTR) {
            continue;
        }
        if (step <= 0) {
            return -1;
        }
        written += (size_t)step;
    }
    return 0;
}

static int reply(int socket, const char *line) {
    char framed[LINE_BYTES];
    const int written = snprintf(framed, sizeof(framed), "%s\n", line);
    if (written <= 0 || (size_t)written >= sizeof(framed)) {
        return -1;
    }
    return write_all(socket, framed, (size_t)written);
}

/*
 * Serve one request. Every path answers exactly one line, including every
 * error -- a client blocked on a reply that never comes is the one failure
 * mode worse than a refusal.
 */
static int serve(Daemon *daemon, int socket, const char *request) {
    uint64_t bytes = 0U;
    char selector[SELECTOR_WIDTH + 1];
    uint64_t address = 0U;

    if (sscanf(
            request, "allocate %" SCNu64 " " SELECTOR_SCAN, &bytes, selector
        ) == 2) {
        if (strcmp(selector, "host") != 0) {
            /* The selector stays in the protocol even though only one value is
               served: it is what keeps one Remote kind from becoming two. */
            return reply(socket, "error only the host selector is served");
        }
        if (bytes == 0U) {
            return reply(socket, "error zero bytes");
        }
        Region *region = region_allocate(daemon, bytes);
        if (region == NULL) {
            return reply(socket, "error allocation or registration failed");
        }
        fprintf(
            stderr, "daemon: serving %" PRIu64 " bytes at %p\n", bytes,
            region->base
        );
        char line[LINE_BYTES];
        (void)snprintf(
            line, sizeof(line), "ok %" PRIx64 " %" PRIx32,
            (uint64_t)(uintptr_t)region->base, region->registration->rkey
        );
        return reply(socket, line);
    }
    uint32_t peer_queue_pair = 0U;
    uint32_t peer_sequence = 0U;
    uint32_t peer_identifier = 0U;
    char peer_gid_text[64];
    if (sscanf(
            request, "connect %" SCNx32 " %" SCNx32 " %" SCNx32 " %63s",
            &peer_queue_pair, &peer_sequence, &peer_identifier, peer_gid_text
        ) == 4) {
        uint8_t peer_gid[16];
        if (parse_gid(peer_gid_text, peer_gid) != 0) {
            return reply(socket, "error malformed gid");
        }
        /* Each request adds one. A peer that saturates its link with a single
           queue pair asks once; one on a faster card asks several times, and
           they differ only in which of ours each is paired with. */
        if (endpoint_add_queue_pair(daemon, &daemon->endpoint) != 0) {
            return reply(socket, "error could not create a queue pair");
        }
        struct ibv_qp *const created =
            daemon->endpoint.queue_pairs[daemon->endpoint.queue_pair_count - 1U];
        union ibv_gid local_gid;
        if (ibv_query_gid(
                daemon->device, daemon->endpoint.port,
                daemon->endpoint.gid_index, &local_gid
            ) != 0) {
            endpoint_destroy(&daemon->endpoint);
            return reply(socket, "error could not read a gid");
        }
        (void)0;
        /*
         * Transition before replying, so that when the peer reads this line
         * the far end is already able to receive. The peer then brings its own
         * up and posts; neither end posts before both are ready.
         */
        if (endpoint_connect(
                &daemon->endpoint, created, peer_queue_pair, peer_sequence,
                (uint16_t)peer_identifier, peer_gid
            ) != 0) {
            endpoint_destroy(&daemon->endpoint);
            return reply(socket, "error could not connect the queue pair");
        }
        char gid_text[33];
        format_gid(local_gid.raw, gid_text);
        fprintf(
            stderr,
            "daemon: connected queue pair %" PRIx32 " to peer %" PRIx32
            " (%u in this connection)\n",
            created->qp_num, peer_queue_pair,
            (unsigned)daemon->endpoint.queue_pair_count
        );
        char line[LINE_BYTES];
        (void)snprintf(
            line, sizeof(line), "ok %" PRIx32 " %" PRIx32 " %" PRIx16 " %s",
            created->qp_num, daemon->endpoint.packet_sequence_number,
            daemon->endpoint.local_identifier, gid_text
        );
        return reply(socket, line);
    }
    if (sscanf(request, "free %" SCNx64, &address) == 1) {
        return release_one(daemon, address) == 0
            ? reply(socket, "ok")
            : reply(socket, "error no such region");
    }
    return reply(socket, "error unrecognised request");
}

static void serve_connection(Daemon *daemon, int socket) {
    char line[LINE_BYTES];
    size_t length = 0U;
    for (;;) {
        char byte = 0;
        const ssize_t step = recv(socket, &byte, 1U, 0);
        if (step < 0 && errno == EINTR) {
            continue;
        }
        if (step <= 0) {
            break;
        }
        if (byte == '\n') {
            line[length] = '\0';
            length = 0U;
            if (serve(daemon, socket, line) != 0) {
                break;
            }
            continue;
        }
        if (length + 1U >= sizeof(line)) {
            /* An over-long line would otherwise be split into two requests,
               the second of them nonsense. Answer once and hang up. */
            (void)reply(socket, "error request too long");
            break;
        }
        line[length++] = byte;
    }
    release_all(daemon);
}

/* ----------------------------------------------------------------- main */

static int listening_socket(const char *host, uint16_t port) {
    const int listener = socket(AF_INET, SOCK_STREAM, 0);
    if (listener < 0) {
        return -1;
    }
    const int reuse = 1;
    (void)setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    struct sockaddr_in address = {
        .sin_family = AF_INET,
        .sin_port = htons(port),
    };
    if (host == NULL) {
        address.sin_addr.s_addr = htonl(INADDR_ANY);
    } else if (inet_pton(AF_INET, host, &address.sin_addr) != 1) {
        fprintf(stderr, "daemon: --host must be an IPv4 address\n");
        (void)close(listener);
        return -1;
    }
    if (bind(listener, (struct sockaddr *)&address, sizeof(address)) != 0 ||
        listen(listener, 1) != 0) {
        perror("daemon: bind");
        (void)close(listener);
        return -1;
    }
    return listener;
}

static void usage(void) {
    fprintf(
        stderr,
        "usage: shadowspill_memory_daemon --port N [--host A.B.C.D] "
        "[--device NAME] [--gid INDEX]\n"
    );
}

int main(int argc, char **argv) {
    const char *host = NULL;
    const char *device_name = NULL;
    long port = 0;
    int requested_gid = -1;
    for (int index = 1; index < argc; ++index) {
        const char *flag = argv[index];
        const char *value = index + 1 < argc ? argv[index + 1] : NULL;
        if (value == NULL) {
            usage();
            return 2;
        }
        if (strcmp(flag, "--port") == 0) {
            port = strtol(value, NULL, 10);
        } else if (strcmp(flag, "--host") == 0) {
            host = value;
        } else if (strcmp(flag, "--device") == 0) {
            device_name = value;
        } else if (strcmp(flag, "--gid") == 0) {
            requested_gid = (int)strtol(value, NULL, 10);
        } else {
            usage();
            return 2;
        }
        ++index;
    }
    if (port <= 0 || port > 65535) {
        usage();
        return 2;
    }
    /* A peer that hangs up mid-reply must not kill the process; every write
       already reports its own failure. */
    (void)signal(SIGPIPE, SIG_IGN);

    Daemon daemon = {0};
    /*
     * Not zero. Zero is a *valid* GID index -- the link-local RoCE v1 entry --
     * so it cannot also mean "nobody chose one". Left at zero the daemon
     * silently skipped discovery and advertised a GID no routed peer can
     * reach, which surfaces as "INIT->RTR: Invalid argument" with nothing to
     * connect it to the cause.
     */
    daemon.gid_index = requested_gid;
    daemon.device = open_device(device_name);
    if (daemon.device == NULL) {
        return 1;
    }
    daemon.protection_domain = ibv_alloc_pd(daemon.device);
    if (daemon.protection_domain == NULL) {
        fprintf(stderr, "daemon: could not allocate a protection domain\n");
        (void)ibv_close_device(daemon.device);
        return 1;
    }
    const int listener = listening_socket(host, (uint16_t)port);
    if (listener < 0) {
        (void)ibv_dealloc_pd(daemon.protection_domain);
        (void)ibv_close_device(daemon.device);
        return 1;
    }
    fprintf(stderr, "daemon: listening on port %ld\n", port);
    /* One connection at a time, served to completion, then the next. That is
       what keeps this process stateless and lock-free. */
    for (;;) {
        const int connection = accept(listener, NULL, NULL);
        if (connection < 0) {
            if (errno == EINTR) {
                continue;
            }
            break;
        }
        serve_connection(&daemon, connection);
        (void)close(connection);
    }
    (void)close(listener);
    (void)ibv_dealloc_pd(daemon.protection_domain);
    (void)ibv_close_device(daemon.device);
    return 0;
}
