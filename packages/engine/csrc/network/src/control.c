/* The TCP control channel: how a region on another machine is obtained. */
#include "../internal.h"

#include <errno.h>
#include <inttypes.h>
#include <netdb.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

/*
 * A request is written whole, a reply is read to the first newline, and the
 * channel is used by one thread at a time -- pool create and pool close, both
 * on the thread that creates the runtime. So there is no framing state to keep
 * between calls and no lock: a partial read cannot be observed, because the
 * only reader is the caller that just wrote.
 */

static int write_all(int socket, const char *bytes, size_t count) {
    size_t written = 0U;
    while (written < count) {
        const ssize_t step = send(
            socket, bytes + written, count - written, MSG_NOSIGNAL
        );
        if (step < 0) {
            if (errno == EINTR) {
                continue;
            }
            return -1;
        }
        if (step == 0) {
            return -1;
        }
        written += (size_t)step;
    }
    return 0;
}

/*
 * Read one line, newline stripped. One byte at a time, which is the honest
 * shape for a channel that carries a handful of messages per run: it needs no
 * buffer surviving the call, so nothing can be left behind for a later reader
 * to lose.
 */
static int read_line(int socket, char *line, size_t capacity) {
    size_t length = 0U;
    while (length + 1U < capacity) {
        char byte = 0;
        const ssize_t step = recv(socket, &byte, 1U, 0);
        if (step < 0) {
            if (errno == EINTR) {
                continue;
            }
            return -1;
        }
        if (step == 0) {
            /* The peer closed mid-reply. A daemon that died holding our region
               is exactly the case the caller must not mistake for success. */
            return -1;
        }
        if (byte == '\n') {
            line[length] = '\0';
            return 0;
        }
        line[length++] = byte;
    }
    return -1;
}

static int exchange(
    ShadowSpillControlChannel *channel, const char *request, char *reply
) {
    if (channel == NULL || channel->socket < 0) {
        return -1;
    }
    if (write_all(channel->socket, request, strlen(request)) != 0) {
        return -1;
    }
    return read_line(channel->socket, reply, SHADOWSPILL_NETWORK_LINE_BYTES);
}

int shadowspill_control_connect(
    ShadowSpillControlChannel *channel, const char *host, const char *port
) {
    if (channel == NULL || host == NULL || port == NULL) {
        return -1;
    }
    channel->socket = -1;
    const struct addrinfo hints = {
        .ai_family = AF_UNSPEC,
        .ai_socktype = SOCK_STREAM,
    };
    struct addrinfo *candidates = NULL;
    if (getaddrinfo(host, port, &hints, &candidates) != 0) {
        return -1;
    }
    for (struct addrinfo *candidate = candidates; candidate != NULL;
         candidate = candidate->ai_next) {
        const int opened = socket(
            candidate->ai_family, candidate->ai_socktype, candidate->ai_protocol
        );
        if (opened < 0) {
            continue;
        }
        if (connect(opened, candidate->ai_addr, candidate->ai_addrlen) == 0) {
            channel->socket = opened;
            break;
        }
        (void)close(opened);
    }
    freeaddrinfo(candidates);
    return channel->socket < 0 ? -1 : 0;
}

void shadowspill_control_close(ShadowSpillControlChannel *channel) {
    if (channel == NULL || channel->socket < 0) {
        return;
    }
    (void)close(channel->socket);
    channel->socket = -1;
}

int shadowspill_control_allocate(
    ShadowSpillControlChannel *channel,
    uint64_t capacity,
    const char *selector,
    uint64_t *address,
    uint32_t *key
) {
    if (address == NULL || key == NULL || capacity == 0U) {
        return -1;
    }
    char request[SHADOWSPILL_NETWORK_LINE_BYTES];
    const int written = snprintf(
        request, sizeof(request), "allocate %" PRIu64 " %s\n", capacity,
        selector != NULL ? selector : "host"
    );
    if (written <= 0 || (size_t)written >= sizeof(request)) {
        return -1;
    }
    char reply[SHADOWSPILL_NETWORK_LINE_BYTES];
    if (exchange(channel, request, reply) != 0) {
        return -1;
    }
    uint64_t replied_address = 0U;
    uint64_t replied_key = 0U;
    /*
     * A reply that is not exactly "ok <address> <rkey>" fails, including an
     * "error <reason>" line. The reason is the daemon's to print and this
     * side's to not invent: a caller sees that the allocation was refused,
     * and the daemon's own output says why.
     */
    if (sscanf(reply, "ok %" SCNx64 " %" SCNx64, &replied_address, &replied_key)
            != 2 ||
        replied_key > UINT32_MAX) {
        return -1;
    }
    *address = replied_address;
    *key = (uint32_t)replied_key;
    return 0;
}

int shadowspill_control_free(
    ShadowSpillControlChannel *channel, uint64_t address
) {
    char request[SHADOWSPILL_NETWORK_LINE_BYTES];
    const int written = snprintf(
        request, sizeof(request), "free %" PRIx64 "\n", address
    );
    if (written <= 0 || (size_t)written >= sizeof(request)) {
        return -1;
    }
    char reply[SHADOWSPILL_NETWORK_LINE_BYTES];
    if (exchange(channel, request, reply) != 0) {
        return -1;
    }
    return strcmp(reply, "ok") == 0 ? 0 : -1;
}

/* ------------------------------------------------------------ handshake */

/*
 * A GID travels as 32 hex characters -- the 16 bytes `ibv_query_gid` fills in,
 * most significant first. Formatting and parsing it by hand keeps the wire
 * format independent of any verbs type, which is what lets the daemon and this
 * side be compiled against different headers on different machines.
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

int shadowspill_control_connect_endpoint(
    ShadowSpillControlChannel *channel,
    const ShadowSpillEndpointIdentity *local,
    ShadowSpillEndpointIdentity *remote
) {
    if (local == NULL || remote == NULL) {
        return -1;
    }
    char gid_text[33];
    format_gid(local->global_identifier, gid_text);
    char request[SHADOWSPILL_NETWORK_LINE_BYTES];
    const int written = snprintf(
        request, sizeof(request), "connect %" PRIx32 " %" PRIx32 " %" PRIx16 " %s\n",
        local->queue_pair_number, local->packet_sequence_number,
        local->local_identifier, gid_text
    );
    if (written <= 0 || (size_t)written >= sizeof(request)) {
        return -1;
    }
    char reply[SHADOWSPILL_NETWORK_LINE_BYTES];
    if (exchange(channel, request, reply) != 0) {
        return -1;
    }
    uint32_t queue_pair = 0U;
    uint32_t sequence = 0U;
    uint16_t identifier = 0U;
    char remote_gid[64];
    /*
     * Anything that is not exactly "ok <qpn> <psn> <lid> <gid>" fails, an
     * "error <reason>" line included. The reason is the daemon's to print and
     * this side's not to invent.
     */
    if (sscanf(
            reply, "ok %" SCNx32 " %" SCNx32 " %" SCNx16 " %63s", &queue_pair,
            &sequence, &identifier, remote_gid
        ) != 4 ||
        parse_gid(remote_gid, remote->global_identifier) != 0) {
        return -1;
    }
    remote->queue_pair_number = queue_pair;
    remote->packet_sequence_number = sequence;
    remote->local_identifier = identifier;
    return 0;
}
