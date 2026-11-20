# Node registration and control

The [service architecture](README.md) uses node-initiated registration. The
following protocol is the implementation contract for remote node daemons.

## Join algorithm

1. The node daemon acquires its local ownership lock, confirms that previous owned
   workers have stopped and probes its assigned resources.
2. It authenticates to the configured service URL and sends a bounded `Join`
   record: supported protocol versions, stable node ID, fresh process incarnation,
   hardware, installed software capabilities and optional worker network addresses.
3. The service checks the credential's allowed identity and pools, negotiates a
   protocol version and rejects conflicting live incarnations. A node daemon cannot
   evict another process by registering its name.
4. `Joined` returns the controller incarnation, membership ID, authorized pool
   and heartbeat/lease durations. The node becomes an entry in the registry used
   by hardware inspection and selection. Joining reserves no training resources.
5. The node daemon maintains an outbound command subscription and sends heartbeats
   and results. The service delivers work through that subscription; it opens no
   incoming connection to a node daemon control port.

A hostname is display/contact information, not node identity. The service's
trust policy binds credentials to identities. Client and node credentials have
different permissions: a client cannot register hardware, and a node daemon cannot
submit another user's training requests. TLS verifies the service identity;
Internet connections do not use plaintext bearer credentials.

An expired membership cannot become live again through a late heartbeat. A new
membership requires reconciliation and confirmed cleanup of the old one. A
controller restart changes its incarnation and cannot continue old assignments.

## Registry and compatibility

The registry combines observed capacity, membership liveness and confirmed
resource claims. It records assigned device IDs/types/capacities, CPU and host
limits, runtime/source capabilities and operator-provided worker interfaces.
The service owns pool membership and selection policy.

Preview checks compatible entries without reserving them. Creation rechecks and
claims the complete selected set. Join reports capabilities; preparation and
saved-Plan admission check exact installed components, source identities and
target requirements. Different GPU types may use different compatible binaries
and Plans. Never infer measured bandwidth or performance from a network address.

## Outbound HTTP control

Use authenticated HTTPS long polling with the existing bounded record codecs.
The node daemon keeps a command request pending. The service replies immediately when
work appears, and the node daemon immediately renews its request. This avoids a polling
sleep before delivering work. Heartbeats, replies and artifact streams use
separate requests, so a long build or transfer cannot block membership renewal.

Long-poll timeouts must fit the deployment's proxy/request limits. Bound pending
requests, queued bytes and message sizes; overload applies backpressure. These
are transport concerns, separate from how long a training operation may run.
[RFC 6202](https://www.rfc-editor.org/rfc/rfc6202) describes long polling and its
timeout/intermediary constraints.

The membership state machine is independent of the HTTP transport. There is one
implementation of its semantics; a future duplex transport does not require a
different public training API.

## Message contract

| Exchange | Meaning |
| --- | --- |
| `Join` / `Joined` | Authenticate and establish one versioned membership |
| `Heartbeat` / acknowledgement | Renew authority and reconcile active assignment IDs |
| `Poll` / assignment | Deliver bounded work over the node daemon's outbound request |
| `Claim` / accepted or rejected | Reserve a complete local resource subset |
| `Stage` / ready | Verify artifacts, launch owned processes and reach their startup barrier |
| Worker envelopes / replies | Preserve the existing ordered worker protocol |
| `Stop` / stopped | Confirm termination and cleanup before releasing resources |

Each assignment/result carries `(controller incarnation, membership ID,
assignment ID)`. Worker messages also retain run, rank and invocation tickets.
An assignment's payload is immutable. Late or incorrectly scoped messages are
rejected. Commands select installed phase/worker operations; they do not supply
shell commands, arbitrary imports or executable uploads.

Retries of the same ID and identical payload may return a recorded receipt;
they never enqueue the operation again. Reusing an ID with a different payload
is an error. Receipt retention is bounded by explicit acknowledgement/retirement;
an expired ID is rejected rather than interpreted as new. This deduplication
applies within a live membership. A crash does not establish whether an uncertain
optimizer update executed; [recovery](execution.md#failure-and-recovery) applies.

## Files and checkpoints

Bulk data uses assignment-scoped artifact grants and admitted streams, separate
from control messages. Existing length/content checks must complete before files
become ready. Logs have explicit size bounds and drain independently of workers.

Each node daemon has local working/cache paths and may locate frozen checkpoints under
its own model root. A common checkpoint identity need not have the same path on
every machine. Registration does not rehash all model weights or copy them through
the controller. Exact source/content validation belongs to preparation and
initialization, where verified identities can be reused.

---

Previous: [Service and node daemons](README.md) ·
[Architecture index](../README.md) ·
Next: [Assignments and recovery](execution.md)
