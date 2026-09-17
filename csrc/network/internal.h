/* What csrc/network/ holds, and nothing outside it sees. */

#ifndef SHADOWSPILL_NETWORK_INTERNAL_H
#define SHADOWSPILL_NETWORK_INTERNAL_H

#include <stdint.h>

/* The contract this library implements, and the kind enum naming its entry.
   Nothing else of the runtime's is needed, and nothing of the runtime's
   internals is reachable. */
#include <shadowspill/runtime/descriptions.h>
#include <shadowspill/runtime/pool_memory.h>

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
 * One TCP connection to one daemon, carrying two verbs. It is how a region on
 * another machine is obtained and given back, and it moves no payload: the
 * bytes travel over the NIC, addressed by what these verbs return.
 *
 *   allocate <bytes> <selector>   ->  ok <address> <rkey>  |  error <reason>
 *   free <address>                ->  ok                   |  error <reason>
 *
 * Line-oriented text, one request per line, because being readable by hand was
 * the argument for a TCP channel rather than a second RDMA path. There is no
 * version handshake: a mismatched pair is diagnosed by reading the two sources.
 * Text is what makes that affordable -- an unrecognised verb or a wrong arity
 * is a visible error line, where binary framing could misread a field and
 * return a plausible address. The residual risk is a verb whose *meaning*
 * changes while its shape does not, so extend the protocol rather than edit it.
 */

/* Longest line either side sends or accepts, newline included. An address and
   a key are hex, a selector is short, and neither verb will grow. */
#define SHADOWSPILL_NETWORK_LINE_BYTES 512U

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

#endif /* SHADOWSPILL_NETWORK_INTERNAL_H */
