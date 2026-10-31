# The remote memory daemon

`shadowspill_memory_daemon` holds memory on one machine so that a ShadowSpill
runtime on another can use it as a pool.

It links **ibverbs and libc, and not `libshadowspill`**. That is the point: the
far side of a remote pool is not a second ShadowSpill, it is a process that
owns a registered region and answers three requests about it.

## Running one

```
shadowspill_memory_daemon --port 7654 [--host 0.0.0.0] [--device mlx5_1] [--gid 3]
```

`--gid` names the GID index the port advertises. Without it the daemon takes
the port's first RoCE v2 entry with an IPv4-mapped address, then any RoCE v2
entry, and refuses rather than advertise one no peer can route to; an
InfiniBand port uses index 0.

One daemon serves concurrent connections in independent workers. Each connection
owns its device handle, protection domain, registered regions and queue pairs;
disconnecting releases only that client's resources. Distributed ranks can use
one endpoint without serializing pool initialization.
Each rank still owns a separate spill pool with its own capacity and leases.
The daemon needs memory for the sum of those capacities; its address identifies
the service, not a shared pool.

There is **no authentication**, which is a decision rather than an oversight:
this is a development daemon for a lab segment. Bind it to the interface you
mean.

## The protocol

Line-oriented text, one request per line, one reply per line.

| request | reply |
|---|---|
| `allocate <bytes> <selector>` | `ok <address> <rkey>` or `error <reason>` |
| `connect <qp> <psn> <lid> <gid>` | `ok <qp> <psn> <lid> <gid>` or `error <reason>` |
| `free <address>` | `ok` or `error <reason>` |

`<bytes>` is decimal; every other number is hex, without a prefix. `connect`
carries the peer's queue pair number, packet sequence number, local identifier
and GID (32 hex digits), and its reply carries the same four for the queue pair
the daemon created to pair with it.

`<selector>` says which memory to serve, in the daemon's own vocabulary. Only
`host` is served, and anything else is refused with a reason. The selector is
in the protocol so that serving another kind of memory adds a selector value
rather than a second `Remote` pool kind.

Text rather than binary framing, because being readable by hand was the whole
argument for a TCP channel beside the RDMA path: `nc` is a working client.

There is **no version handshake**. A daemon started out of band can be older
than the runtime talking to it, and that is diagnosed by reading the two
sources. Text is what makes the omission affordable — an unrecognised verb or a
wrong arity is a visible error line, where binary framing could misread a field
and hand back a plausible address. The residual risk is a verb whose *meaning*
changes while its shape does not, so **extend the protocol rather than edit
it**.

## What it does between bring-up and teardown

Nothing, and that is a consequence of one-sided verbs rather than an
aspiration. Every transfer is an RDMA read or write posted by the *remote*
side against the address and key returned here, and serviced by this machine's
NIC; this process is never notified and has no completion to handle.

"One-sided" describes the data path, not connection setup: an RC queue pair is
established two-sided, and an rkey is usable only by a queue pair in the same
protection domain as its memory region. So this process owns the device handle,
the protection domain and the registrations, and takes part in the handshake:
each `connect` creates one queue pair and brings it up against the peer's
before replying with its numbers, so this end can already receive when the
peer brings its own up and posts. A peer that wants several queue pairs asks
once for each.

## When the socket closes

Everything allocated for that connection is freed and unregistered, and its
queue pairs destroyed, whether the close was orderly or a crash. That backstop
is why the daemon needs no registry and no heartbeat: a clean shutdown and a
dead peer look the same from here, and both end with the memory released.

The one case it does not cover is a half-open connection — the machine
vanishing rather than the process — where TCP keepalive is the slow backstop. A
heartbeat would only shorten that window, so there is not one.
