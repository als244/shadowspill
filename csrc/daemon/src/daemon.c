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

typedef struct Daemon {
    struct ibv_context *device;
    struct ibv_pd *protection_domain;
    Region *regions;
} Daemon;

/* ---------------------------------------------------------------- verbs */

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
    for (int index = 0; index < count; ++index) {
        const char *device_name = ibv_get_device_name(devices[index]);
        if (name != NULL && strcmp(name, device_name) != 0) {
            continue;
        }
        opened = ibv_open_device(devices[index]);
        if (opened != NULL) {
            fprintf(stderr, "daemon: using %s\n", device_name);
            break;
        }
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
        "[--device NAME]\n"
    );
}

int main(int argc, char **argv) {
    const char *host = NULL;
    const char *device_name = NULL;
    long port = 0;
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
