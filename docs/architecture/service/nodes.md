# Node registration and control

A node daemon registers an operator-assigned machine with the [service](README.md)
and keeps that membership alive over outbound requests. This page describes the
protocol between them.

## Join algorithm

1. The node daemon takes its node and device ownership locks and probes its
   assigned resources. If its journal names an earlier incarnation, it sends
   `Reconcile` for it: holding the locks shows that incarnation's children have
   exited.
2. It journals its own incarnation, then sends a bounded `Join` to the
   configured service URL: the protocol version, a fresh process incarnation,
   its hardware (including the stable node ID and pool), installed component
   packages, an optional worker address, and the previous membership scope when
   it has one. A `Join` whose version differs from the service's is refused.
3. The service checks that the credential authorizes that node ID and pool. It
   refuses a join that would replace a live membership, or an earlier one that
   is not reconciled. A node daemon cannot evict another process by registering
   its name.
4. `Joined` returns the membership scope (controller incarnation and membership
   ID) and the heartbeat, lease and poll durations. The node becomes an entry in
   the registry that hardware listings read. Joining reserves no resources.
5. The node daemon sends heartbeats, polls for commands and returns receipts,
   all over outbound requests; the service opens no connection to it.

A hostname is display/contact information, not node identity. The service's
trust policy binds credentials to identities. Client and node credentials have
different permissions: a client cannot register hardware, and a node daemon cannot
submit another user's training requests. TLS verifies the service identity;
Internet connections do not use plaintext bearer credentials.

An expired membership cannot become live again through a late heartbeat. A new
membership requires reconciliation and confirmed cleanup of the old one. A
controller restart changes its incarnation and cannot continue old assignments.

## Registry

The registry combines observed capacity, membership liveness and confirmed
resource claims. It records each node's devices with their IDs, types and
capacities, its CPU and host limits, its installed packages and its optional
worker address. Hardware listings show registered nodes beside the service's
own; preview and creation select from the service's own node only.

## Outbound HTTP control

Every exchange is one bounded request from the node daemon, over HTTPS unless
the service is on a loopback address. TLS must verify the service's certificate
and hostname.

The node daemon keeps one command poll pending. The service answers it as soon
as a command is queued, or with an acknowledgement once the poll duration
passes, and the node daemon polls again at once. Heartbeats and receipts use
separate requests, so a long command cannot block membership renewal. The node
daemon's request timeout must exceed the poll duration, and so must any proxy's
([RFC 6202](https://www.rfc-editor.org/rfc/rfc6202) describes long polling
behind intermediaries). The registry bounds the commands pending per node and in
all, with room for one stop beyond each bound.

The membership state machine (`NodeRegistry`) is independent of the HTTP
transport.

## Message contract

| Exchange | Meaning |
| --- | --- |
| `Join` / `Joined` | Authenticate and establish one versioned membership |
| `Heartbeat` / `Acknowledged` | Renew the lease and report the active assignment IDs |
| `Poll` / `Command` or `Acknowledged` | Deliver the next command over the node daemon's outbound request, or report which receipts the service has retired |
| `Receipt` / `Acknowledged` | Return one command's result |
| `Leave` / `Acknowledged` | End the membership after local cleanup, releasing its claims |
| `Reconcile` / `Reconcile` | Confirm that a journaled earlier incarnation's children have exited |

A command is `claim`, which reserves a complete local resource subset (bytes on
each named device, and host bytes), or `stop`, which releases an assignment's
resources. The node daemon refuses any other operation. A command names
installed code; it carries no shell command, import or executable.

Each command and receipt carries the membership scope and a sequence, and a
command names its assignment. A command's payload is immutable. Late or
incorrectly scoped messages are rejected.

A claim for an assignment that already exists is refused rather than enqueued
again; its recorded receipt is the answer. A receipt cannot change once
recorded. Receipt retention is bounded by explicit acknowledgement and
retirement, and an assignment ID stays known, so a reused one is rejected rather
than interpreted as new. This deduplication applies within a live membership.

---

Previous: [Service and node daemons](README.md) ·
[Architecture index](../README.md) ·
Next: [Assignments and recovery](execution.md)
