# Assignments and recovery

The service owns one ordered training run on the devices of its own host, and
the run's processes and resources there. This division preserves the training
state machine and [durable result semantics](../service-records.md).

## Select, reserve and start

1. Resolve the client's logical parallelism and optional `Resources` selectors
   against the service's own node. Automatic selection uses equivalent devices;
   a selection that mixes GPU types is refused.
2. Unless the client supplied saved Plans, run the preparation phases on the
   selected devices. Plan-set selection refuses a set whose spill pools and
   trace budgets need more host memory than the node leaves beside the
   service, the run's arena and the workers' fixed allowances; supplied Plans
   are checked the same way.
3. Reserve every worker's device and host capacity, and its startup files and
   logs, before launching any child. Resources without confirmed cleanup stay
   unavailable.
4. Stage the verified Plans and launch the owned worker processes. Distributed
   workers rendezvous through a file the run owns.
5. Wait for every worker to report ready. A worker that exits is seen at once;
   `worker_startup_seconds` bounds one that hangs.

Registered node daemons take no part: no run is placed on them.

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
microbatches visited within each pass. Every worker's Plan for a candidate uses
the same ordering. An independently fastest depth on each GPU can repeat
sharded-weight fetches a different number of times. Equal depth is only an
initial constraint: final Plans must still agree on collective participation and
issue order, since different spill decisions can introduce incompatible
communication.

Matching collective order does not require matching local schedules. Workers
may retain or recompute different intermediates and fetch or spill at different
task boundaries when those choices preserve collective participation and order.
Preparation keeps such independently admitted Plans unchanged. This applies to
both replicated and sharded storage; replicated weight fetches are local, while
sharded weight fetches can introduce collectives.

If independently selected Plans disagree, preparation tries a common logical
schedule from the searched candidates. Each worker keeps its own compiled code,
measured costs and tensor sizes. The common retention, recomputation and transfer
decisions are simulated and physically admitted on every worker; only a candidate
that fits all workers and has matching collective order can run. This requires
matching task/storage topology. Failure to find such a candidate is reported as
search exhaustion, rather than relaxing the communication or memory checks.
This fallback restricts local scheduling choices and can sacrifice some
worker-specific optimization; it does not require equal wall-clock timings,
tensor sizes, kernel implementations or physical memory offsets.

## Runtime commands

The client submits a global batch through `step()`, `forward_backward()` or
`forward()`, and optimizer controls through `optim_step()`. Packing and worker
selection happen automatically. `validate_batch()` uses the prepared layouts to
reject batches that cannot fit before submission. Batch limits bound the whole
batch across workers, and scheduled padding counts toward them.

The preview reports resolved shapes, counts and capacities before preparation;
step-time predictions require measured profiles. Runtime calls receive no device
placements or Plan choices.

## Failure and recovery

Heartbeats renew a node daemon's membership lease using local monotonic clocks;
synchronized wall clocks are unnecessary. Renewals are correlated, and the node
daemon bounds an extension from its request-send time so a delayed reply cannot
grant authority indefinitely.

| Event | Required behavior |
| --- | --- |
| Individual node HTTP exchange fails | Retry while membership is valid; return receipts again without repeating commands |
| Heartbeats stop | Expire the membership and drop the node from the inventory |
| Membership lease expires | Fence the membership permanently; the node daemon releases its claims |
| Node daemon dies | A restart takes its ownership locks and reconciles its predecessor before joining again |
| A run's worker is lost | Fail the group, stop the other workers and preserve durable completed outcomes |
| Controller restarts | Use a new incarnation; mark active runs stopped and reject old membership messages |
| Cleanup fails | Record the run's outcome, keep resources unavailable and retain owners for a cleanup retry |

A lease timeout alone never proves that a node's claims were released: the
registry keeps claimed capacity owned until a confirmed stop, a leave or a
reconciliation.

An uncertain optimizer update is never replayed. Recovery creates a fresh worker
group from a completed checkpoint. The service does not claim exactly-once
execution after a crash, automatic live failover between controllers or changes
to a running world's size.

Normal shutdown drains or cancels work, stops children, closes outstanding file
owners and only then releases resources. `stop` is idempotent within a
membership. Leaving does not release the operator's scheduler allocation.

---

Previous: [Node registration and control](nodes.md) ·
[Architecture index](../README.md) ·
Next: [Durable service records](../service-records.md)
