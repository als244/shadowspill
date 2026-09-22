# The remote performance gate

The performance gate's matrix with one substitution: the spill pool is a region
a daemon holds on another machine instead of pinned host memory. Same cells,
same manifests, same protocol -- three groups of four steps, no checkpoint --
and the same budgets. That one difference is the variable under test.

```text
qualification/remote_perf/
└── matrix.py    the performance matrix, spilling to a peer
```

It has no cell list of its own. The cells come from the performance gate, so
the two cannot drift apart: a cell added there is a cell this gate runs.

## What it judges against

The cells are judged against floors of their own,
`remote_regression_tokens_per_second` in `workloads.full_model`, measured with
the pool on a peer, at the same 0.95 margin the local floors use. The local
floors do not apply: the link is about 3 GB/s against 25 GB/s to pinned host
memory, so a transfer-bound step runs several times slower, and a local number
says nothing about a remote run. Every cell runs even after one misses, so a
run records all three numbers.

## Running it

It is **not** in the default gate run, because it needs a memory daemon
reachable over RDMA. Name one in `SHADOWSPILL_NETWORK_PEER` as `host:port`, an
address rather than an ssh alias:

```bash
SHADOWSPILL_NETWORK_PEER=192.168.50.32:17800 python -m tools.qualification.gates remote_perf
```

With no peer named the gate **skips and succeeds**, writing a summary that says
so. Results land under `qualification/results/remote_perf_<run>/`, one artifact
and log per cell, beside the local matrix's.

The peer's pool defaults to the manifests' 112 GiB spill budget, and the peer
has to hold it. `--remote-spill-gib` on the matrix makes the pool smaller, and
the run says so, because a smaller pool is a second thing differing from the
local run: planning sees a smaller pool and may choose differently. A
three-cell run takes about an hour.
