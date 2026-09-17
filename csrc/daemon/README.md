# The remote memory daemon

`shadowspill_memory_daemon` holds memory on one machine so that a ShadowSpill
runtime on another can use it as a pool.

It links **ibverbs and libc, and not `libshadowspill`**. That is the point: the
far side of a remote pool is not a second ShadowSpill, it is a process that
owns a registered region and answers two questions about it.

## Running one

```
shadowspill_memory_daemon --port 7654 [--host 0.0.0.0] [--device mlx5_1]
```

One daemon serves **one connection at a time**, which makes it stateless and
lock-free: everything it holds belongs to the connection it is holding it for.
Two remote pools on a box is two daemons on two ports.

There is **no authentication**, which is a decision rather than an oversight:
this is a development daemon for a lab segment, and it is the first thing to
revisit if any of this heads somewhere deployed. Bind it to the interface you
mean.

## The protocol

Line-oriented text, one request per line, one reply per line.

| request | reply |
|---|---|
| `allocate <bytes> <selector>` | `ok <address> <rkey>` or `error <reason>` |
| `free <address>` | `ok` or `error <reason>` |

`<bytes>` is decimal; `<address>` and `<rkey>` are hex, without a prefix.

`<selector>` says which memory to serve, in the daemon's own vocabulary. Today
only `host` is served and anything else is refused with a reason — but the
selector **stays in the protocol**, because it is what keeps one `Remote` pool
kind from turning into two the day a daemon can serve device memory.

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
the protection domain and the registrations, and will own a queue pair and take
part in the handshake when there is a data path to set up.

## When the socket closes

Everything allocated for that connection is freed and unregistered, whether the
close was orderly or a crash. That backstop is why the daemon needs no registry
and no heartbeat: a clean shutdown and a dead peer look the same from here, and
both end with the memory released.

The one case it does not cover is a half-open connection — the machine
vanishing rather than the process — where TCP keepalive is the slow backstop. A
heartbeat would only shorten that window, so there is not one.
