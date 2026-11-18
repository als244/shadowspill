# Assignments and recovery

The service owns one ordered training run across its selected agents. Agents
own local resources and processes. This division preserves the existing training
state machine and [durable result semantics](../service-records.md).

## Select, claim and start

1. Resolve the client's logical parallelism and optional resource selectors
   against fresh registry entries. Automatic selection uses equivalent compatible
   devices; mixing hardware types requires explicit setup.
2. Record the full logical-worker mapping and request each node's complete local
   resource claim in parallel. Local admission reserves devices and host capacity
   atomically before launching children.
3. If any node refuses, stop/release successful partial claims. No collective work
   starts. Resources without confirmed cleanup stay unavailable.
4. Stage verified Plans/inputs and launch owned processes. For distributed work,
   preflight actual peer connectivity and communicator startup with a deadline.
5. Wait for all members to become ready, then release the phase. A receipt from
   an agent is not proof that its worker is ready or that training completed.

Preview is advisory and makes no claims. Creation may fail if capacity has become
busy; waiting is explicit. Exact-device selection never silently remaps devices.
Changing GPU class or raising a budget requires a changed setup request.

Independent phase jobs and planning searches can run in parallel. Equivalent
homogeneous workers use the same complete selected Plan with rank-specific
shards, inputs and communicator bindings. Different targets need compatible
target-specific Plans. Collective constraints include communication within weight
fetches, gradient spills and optimizer tasks, not only explicit model operations.

Distributed training candidates share a traversal ordering. Its `depth` is the
number of complete forward/backward passes; `breadth` is the number of
microbatches visited within each pass. Select these jointly across workers,
including heterogeneous workers, while allowing their token capacities to differ.
An independently fastest depth on each GPU can repeat sharded-weight fetches a
different number of times. Equal depth is only an initial constraint: final
Plans must still agree on collective participation and issue order, since
different spill decisions can introduce incompatible communication.

If independently selected Plans disagree, preparation tries a common logical
schedule from the searched candidates. Each worker keeps its own compiled code,
measured costs and tensor sizes. The common retention, recomputation and transfer
decisions are simulated and physically admitted on every worker; only a candidate
that fits all workers and has matching collective order can run. This requires
matching task/storage topology. Failure to find such a candidate is reported as
search exhaustion, rather than relaxing the communication or memory checks.

## Runtime commands and workload shares

The client continues to submit a global batch through `step()`,
`forward_backward()` or `forward()`, and optimizer controls through `optim_step()`.
Packing and worker selection happen automatically. `validate_batch()` uses the
prepared layouts to reject batches that cannot fit before submission.

Advanced heterogeneous setup associates each worker with resource constraints,
memory/preparation settings and an optional relative input share. This is input
capacity, not a measured speed ratio. It applies to independent input partitions;
cooperating copies do not multiply the global input count.

For example, four 24K buffers and four 8K buffers represent a 3:1 share and 128K
scheduled positions. Whole-sequence packing can change the ratio of real input
tokens. Padding still counts against the global cap. Global objective scaling
and requested batch membership stay unchanged. Different counts must not let a
worker skip a required collective; all complete schedules must agree on their
collective participation.

The preview reports resolved shapes/counts/capacities before preparation. Actual
step-time predictions require measured profiles. Runtime calls do not receive
node addresses, worker settings, workload ratios or Plan choices.

## Failure and recovery

Heartbeats renew a membership lease using local monotonic clocks; synchronized
wall clocks are unnecessary. Renewals are correlated, and the agent bounds an
extension from its request-send time so a delayed reply cannot grant authority
indefinitely. An operator's allocation expiry bounds all leases.

| Event | Required behavior |
| --- | --- |
| Individual HTTP exchange fails | Retry transport with bounded backoff while membership is valid; reconcile receipts without recreating commands |
| Heartbeats become stale | Exclude the node from new selection |
| Membership lease expires | Fence the membership permanently and stop its owned workers |
| Agent dies | Local supervision/watchdog stops owned children; restart confirms cleanup before offering resources |
| Executing member is lost | Fail the group, stop peers and preserve durable completed outcomes |
| Controller restarts | Use a new incarnation; interrupt active runs and reject old assignment messages |
| Cleanup fails | Keep resources unavailable and retain owners for a cleanup retry |

A lease timeout alone never proves that a worker stopped. The controller waits
for confirmed cleanup before reusing resources. The agent's local ownership lock
and supervisor must cover crashes as well as graceful shutdown.

An uncertain optimizer update is never replayed. Recovery creates a fresh worker
group from a completed checkpoint. This protocol does not claim exactly-once
execution after a crash, automatic live failover between controllers or changes
to a running world's size.

Normal shutdown drains or cancels work, stops children, closes outstanding file
owners and only then releases claims. `Stop` is idempotent within its membership.
Unregistering does not release the operator's scheduler allocation.

---

Previous: [Node registration and control](nodes.md) ·
[Architecture index](../README.md) ·
Next: [Durable service records](../service-records.md)
