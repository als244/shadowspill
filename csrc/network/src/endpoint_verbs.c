/* The local end of an RC connection: one device, one queue pair, one peer. */
#include "../internal.h"

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

/*
 * WHAT NEVER REACHES A POOL'S CONFIGURATION. A remote pool is configured with
 * a host, a port and a selector, and nothing else: no device name, no GID
 * index, no link layer, no MTU, no queue depth. Every one of those is
 * discovered here, because every one of them is a property of *this machine's*
 * hardware rather than of the plan, and a caller who had to supply them would
 * be supplying facts it cannot know and the runtime can.
 *
 * Detection, in order:
 *   which local NIC reaches the peer  -> which RDMA device owns that NIC
 *   which of its ports is active      -> InfiniBand or RoCE
 *   InfiniBand: address by LID        -> RoCE: discover a routable GID
 *   the port's own negotiated MTU
 *
 * SHADOWSPILL_NETWORK_DEVICE overrides the first step for the case where
 * detection is wrong. It is an environment variable rather than a config field
 * on purpose: an escape hatch that a caller never has to know exists.
 */

static int is_ipv4_mapped(const union ibv_gid *gid) {
    static const uint8_t prefix[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xFF};
    return memcmp(gid->raw, prefix, sizeof(prefix)) == 0;
}

/*
 * Which of this machine's interfaces would carry a packet to `host`.
 *
 * Asked by connecting a UDP socket, which sends nothing: the kernel performs
 * the route lookup to choose a source address, and `getsockname` reports what
 * it chose. That is the routing table's own answer rather than a guess from
 * netmasks, and it is right when several interfaces could plausibly serve.
 */
static int interface_toward(const char *host, char name[IF_NAMESIZE]) {
    struct addrinfo hints = {.ai_family = AF_UNSPEC, .ai_socktype = SOCK_DGRAM};
    struct addrinfo *peer = NULL;
    if (getaddrinfo(host, "9", &hints, &peer) != 0 || peer == NULL) {
        return -1;
    }
    const int probe = socket(peer->ai_family, SOCK_DGRAM, 0);
    if (probe < 0) {
        freeaddrinfo(peer);
        return -1;
    }
    struct sockaddr_storage chosen;
    socklen_t length = sizeof(chosen);
    const int connected = connect(probe, peer->ai_addr, peer->ai_addrlen) == 0 &&
        getsockname(probe, (struct sockaddr *)&chosen, &length) == 0;
    (void)close(probe);
    freeaddrinfo(peer);
    if (!connected) {
        return -1;
    }
    struct ifaddrs *interfaces = NULL;
    if (getifaddrs(&interfaces) != 0) {
        return -1;
    }
    int found = -1;
    for (struct ifaddrs *entry = interfaces; entry != NULL;
         entry = entry->ifa_next) {
        if (entry->ifa_addr == NULL ||
            entry->ifa_addr->sa_family != chosen.ss_family) {
            continue;
        }
        const size_t compare = chosen.ss_family == AF_INET
            ? sizeof(struct in_addr) : sizeof(struct in6_addr);
        const void *mine = chosen.ss_family == AF_INET
            ? (const void *)&((struct sockaddr_in *)&chosen)->sin_addr
            : (const void *)&((struct sockaddr_in6 *)&chosen)->sin6_addr;
        const void *theirs = chosen.ss_family == AF_INET
            ? (const void *)&((struct sockaddr_in *)entry->ifa_addr)->sin_addr
            : (const void *)&((struct sockaddr_in6 *)entry->ifa_addr)->sin6_addr;
        if (memcmp(mine, theirs, compare) == 0) {
            snprintf(name, IF_NAMESIZE, "%s", entry->ifa_name);
            found = 0;
            break;
        }
    }
    freeifaddrs(interfaces);
    return found;
}

/* Does this RDMA device own that network interface? Its netdevs are listed in
   sysfs, which is how a RoCE device is tied to the NIC it shares. */
static int device_owns_interface(
    struct ibv_device *device, const char *interface
) {
    char path[512];
    snprintf(
        path, sizeof(path), "/sys/class/infiniband/%s/device/net/%s",
        ibv_get_device_name(device), interface
    );
    return access(path, F_OK) == 0;
}

static int port_is_active(struct ibv_context *device, uint8_t port) {
    struct ibv_port_attr attributes;
    return ibv_query_port(device, port, &attributes) == 0 &&
        attributes.state == IBV_PORT_ACTIVE;
}

static int first_active_port(struct ibv_context *device, uint8_t *port) {
    struct ibv_device_attr attributes;
    if (ibv_query_device(device, &attributes) != 0) {
        return -1;
    }
    for (uint8_t candidate = 1U; candidate <= attributes.phys_port_cnt;
         ++candidate) {
        if (port_is_active(device, candidate)) {
            *port = candidate;
            return 0;
        }
    }
    return -1;
}

/*
 * Open the device that can reach `host`.
 *
 * The route lookup is the good answer and the active-port scan is the
 * fallback, because a machine with one RDMA NIC does not need the lookup and a
 * machine with two does. Picking "the first device that opens" is what the
 * daemon did in the phase before this one, and on this very box that is a
 * device whose ports are all down -- which registers memory perfectly and
 * hands back a key nothing can reach.
 */
static struct ibv_context *open_device_toward(
    const char *host, uint8_t *port, char chosen_name[64]
) {
    const char *requested = getenv("SHADOWSPILL_NETWORK_DEVICE");
    char interface[IF_NAMESIZE] = {0};
    const int routed = requested == NULL && interface_toward(host, interface) == 0;

    int count = 0;
    struct ibv_device **devices = ibv_get_device_list(&count);
    if (devices == NULL || count == 0) {
        if (devices != NULL) {
            ibv_free_device_list(devices);
        }
        return NULL;
    }
    struct ibv_context *opened = NULL;
    /* Two passes: the device that owns the routed interface, then any device
       with an active port. A named device skips both. */
    for (int pass = 0; pass < 2 && opened == NULL; ++pass) {
        for (int index = 0; index < count; ++index) {
            const char *name = ibv_get_device_name(devices[index]);
            if (requested != NULL && strcmp(requested, name) != 0) {
                continue;
            }
            if (requested == NULL && pass == 0 &&
                !(routed && device_owns_interface(devices[index], interface))) {
                continue;
            }
            struct ibv_context *candidate = ibv_open_device(devices[index]);
            if (candidate == NULL) {
                continue;
            }
            if (first_active_port(candidate, port) != 0 && requested == NULL) {
                (void)ibv_close_device(candidate);
                continue;
            }
            snprintf(chosen_name, 64, "%s", name);
            opened = candidate;
            break;
        }
        if (requested != NULL) {
            break;
        }
    }
    ibv_free_device_list(devices);
    return opened;
}

/* Which GID to advertise. InfiniBand addresses by LID and needs no choice;
   RoCE needs the routable one, which is the IPv4-mapped v2 entry on a routed
   subnet and is not at a fixed index. */
static int find_gid_index(struct ibv_context *device, uint8_t port) {
    struct ibv_port_attr attributes;
    if (ibv_query_port(device, port, &attributes) != 0) {
        return -1;
    }
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

void shadowspill_endpoint_close(ShadowSpillEndpoint *endpoint) {
    if (endpoint == NULL) {
        return;
    }
    for (uint32_t index = 0U; index < endpoint->queue_pair_count; ++index) {
        if (endpoint->queue_pairs[index] != NULL) {
            (void)ibv_destroy_qp(endpoint->queue_pairs[index]);
        }
        if (endpoint->completion_queues[index] != NULL) {
            (void)ibv_destroy_cq(endpoint->completion_queues[index]);
        }
    }
    endpoint->queue_pair_count = 0U;
    if (endpoint->protection_domain != NULL) {
        (void)ibv_dealloc_pd(endpoint->protection_domain);
    }
    if (endpoint->device != NULL) {
        (void)ibv_close_device(endpoint->device);
    }
    *endpoint = (ShadowSpillEndpoint){0};
}

int shadowspill_endpoint_open(
    ShadowSpillEndpoint *endpoint,
    const char *host,
    uint32_t depth,
    uint32_t queue_pairs
) {
    if (endpoint == NULL || host == NULL) {
        return -1;
    }
    *endpoint = (ShadowSpillEndpoint){0};
    shadowspill_network_tuning_read(&endpoint->tuning);
    /* An explicit argument wins over the environment; zero means "whatever was
       configured", which is what every caller but a test passes. */
    if (queue_pairs != 0U) {
        endpoint->tuning.queue_pairs =
            queue_pairs > SHADOWSPILL_NETWORK_MAX_QUEUE_PAIRS
                ? SHADOWSPILL_NETWORK_MAX_QUEUE_PAIRS : queue_pairs;
    }
    if (depth != 0U) {
        endpoint->tuning.send_depth = depth;
    }
    queue_pairs = endpoint->tuning.queue_pairs;
    depth = endpoint->tuning.send_depth;
    shadowspill_network_tuning_report(&endpoint->tuning);
    endpoint->device = open_device_toward(host, &endpoint->port, endpoint->name);
    if (endpoint->device == NULL) {
        return -1;
    }
    struct ibv_port_attr port_attributes;
    if (ibv_query_port(endpoint->device, endpoint->port, &port_attributes) != 0) {
        shadowspill_endpoint_close(endpoint);
        return -1;
    }
    endpoint->link_layer = port_attributes.link_layer;
    endpoint->local_identifier = port_attributes.lid;
    /* Discovered rather than configured, like the MTU above it: a caller
       cannot know it and a wrong guess is a refused work request. */
    endpoint->max_message_bytes = port_attributes.max_msg_sz;
    if (endpoint->max_message_bytes == 0U) {
        shadowspill_endpoint_close(endpoint);
        return -1;
    }
    /* A knob may only lower it. Asking for more than the port carries is not a
       preference the NIC will honour -- it refuses the work request -- so a
       larger value is silently the port's, which is what it would have been. */
    if (endpoint->tuning.message_bytes != 0U &&
        endpoint->tuning.message_bytes < endpoint->max_message_bytes) {
        endpoint->max_message_bytes =
            (uint32_t)endpoint->tuning.message_bytes;
    }
    endpoint->path_mtu = endpoint->tuning.path_mtu != 0U
        ? (enum ibv_mtu)endpoint->tuning.path_mtu
        : port_attributes.active_mtu;
    endpoint->gid_index = endpoint->tuning.gid_index >= 0
        ? endpoint->tuning.gid_index
        : find_gid_index(endpoint->device, endpoint->port);
    if (endpoint->gid_index < 0) {
        shadowspill_endpoint_close(endpoint);
        return -1;
    }
    endpoint->protection_domain = ibv_alloc_pd(endpoint->device);
    if (endpoint->protection_domain == NULL) {
        shadowspill_endpoint_close(endpoint);
        return -1;
    }
    for (uint32_t index = 0U; index < queue_pairs; ++index) {
        /* One completion queue each. A shared one lets two lanes consume each
           other's completions; see the note in internal.h. */
        endpoint->completion_queues[index] = ibv_create_cq(
            endpoint->device, (int)depth, NULL, NULL, 0
        );
        if (endpoint->completion_queues[index] == NULL) {
            endpoint->queue_pair_count = index;
            shadowspill_endpoint_close(endpoint);
            return -1;
        }
        struct ibv_qp_init_attr init = {
            .send_cq = endpoint->completion_queues[index],
            .recv_cq = endpoint->completion_queues[index],
            .qp_type = IBV_QPT_RC,
            .sq_sig_all = 0,
            .cap = {
                .max_send_wr = depth,
                .max_recv_wr = endpoint->tuning.receive_depth,
                .max_send_sge = 1,
                .max_recv_sge = 1,
            },
        };
        /* Counted before the queue pair exists, so the completion queue just
           made is unwound too if creating it fails. */
        endpoint->queue_pair_count = index + 1U;
        endpoint->queue_pairs[index] =
            ibv_create_qp(endpoint->protection_domain, &init);
        if (endpoint->queue_pairs[index] == NULL) {
            shadowspill_endpoint_close(endpoint);
            return -1;
        }
        struct ibv_qp_attr attributes = {
            .qp_state = IBV_QPS_INIT,
            .pkey_index = 0,
            .port_num = endpoint->port,
            .qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                               IBV_ACCESS_REMOTE_WRITE,
        };
        if (ibv_modify_qp(
                endpoint->queue_pairs[index], &attributes,
                IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                    IBV_QP_ACCESS_FLAGS
            ) != 0) {
            shadowspill_endpoint_close(endpoint);
            return -1;
        }
    }
    return 0;
}

int shadowspill_endpoint_identity(
    const ShadowSpillEndpoint *endpoint,
    uint32_t index,
    ShadowSpillEndpointIdentity *identity
) {
    if (endpoint == NULL || identity == NULL ||
        index >= endpoint->queue_pair_count ||
        endpoint->queue_pairs[index] == NULL) {
        return -1;
    }
    union ibv_gid gid;
    if (ibv_query_gid(
            endpoint->device, endpoint->port, endpoint->gid_index, &gid
        ) != 0) {
        return -1;
    }
    identity->queue_pair_number = endpoint->queue_pairs[index]->qp_num;
    identity->packet_sequence_number = 0U;
    identity->local_identifier = endpoint->local_identifier;
    memcpy(identity->global_identifier, gid.raw, 16U);
    return 0;
}

int shadowspill_endpoint_connect(
    ShadowSpillEndpoint *endpoint,
    uint32_t index,
    const ShadowSpillEndpointIdentity *peer
) {
    if (endpoint == NULL || peer == NULL ||
        index >= endpoint->queue_pair_count ||
        endpoint->queue_pairs[index] == NULL) {
        return -1;
    }
    struct ibv_qp *const queue_pair = endpoint->queue_pairs[index];
    struct ibv_qp_attr ready_to_receive = {
        .qp_state = IBV_QPS_RTR,
        .path_mtu = endpoint->path_mtu,
        .dest_qp_num = peer->queue_pair_number,
        .rq_psn = peer->packet_sequence_number,
        .max_dest_rd_atomic = (uint8_t)endpoint->tuning.outstanding_reads,
        .min_rnr_timer = 12,
        .ah_attr = {
            .port_num = endpoint->port,
            .sl = (uint8_t)endpoint->tuning.service_level,
        },
    };
    /* The one place the link layers differ; see the daemon's copy of this. */
    if (endpoint->link_layer == IBV_LINK_LAYER_INFINIBAND) {
        ready_to_receive.ah_attr.is_global = 0;
        ready_to_receive.ah_attr.dlid = peer->local_identifier;
    } else {
        ready_to_receive.ah_attr.is_global = 1;
        ready_to_receive.ah_attr.grh.hop_limit = 64;
        /* What RoCE congestion control reads. Zero on a quiet subnet, and the
           first thing to change on a fabric with PFC or DCQCN. */
        ready_to_receive.ah_attr.grh.traffic_class =
            (uint8_t)endpoint->tuning.traffic_class;
        ready_to_receive.ah_attr.grh.sgid_index = (uint8_t)endpoint->gid_index;
        memcpy(ready_to_receive.ah_attr.grh.dgid.raw, peer->global_identifier, 16U);
    }
    const int to_receive = ibv_modify_qp(
        queue_pair, &ready_to_receive,
        IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
            IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER
    );
    if (to_receive != 0) {
        fprintf(
            stderr,
            "shadowspill network: INIT->RTR failed (%s) on %s port %u, %s\n",
            strerror(to_receive), endpoint->name, (unsigned)endpoint->port,
            endpoint->link_layer == IBV_LINK_LAYER_INFINIBAND
                ? "InfiniBand" : "RoCE"
        );
        return -1;
    }
    struct ibv_qp_attr ready_to_send = {
        .qp_state = IBV_QPS_RTS,
        /* How long this queue pair keeps trying a peer that has stopped
           answering, which is how long a dead daemon takes to become an error
           rather than a hang. */
        .timeout = (uint8_t)endpoint->tuning.timeout,
        .retry_cnt = (uint8_t)endpoint->tuning.retry_count,
        .rnr_retry = (uint8_t)endpoint->tuning.rnr_retry,
        .sq_psn = 0U,
        .max_rd_atomic = (uint8_t)endpoint->tuning.outstanding_reads,
    };
    const int to_send = ibv_modify_qp(
        queue_pair, &ready_to_send,
        IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
            IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC
    );
    if (to_send != 0) {
        fprintf(
            stderr, "shadowspill network: RTR->RTS failed (%s)\n",
            strerror(to_send)
        );
        return -1;
    }
    return 0;
}
