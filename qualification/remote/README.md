# The remote gate

The numerical gate's matrix with one substitution: the spill pool is a region a
daemon holds on another machine instead of pinned host memory. Same programs,
same references, same tolerances, same routes, same cells, same spill budget.
That one difference is the variable under test, and keeping everything else
identical is what makes a pass mean something -- a disagreement can only be the
pool.

```text
qualification/remote/
└── matrix.py    the numerical matrix, spilling to a peer
```

It has no cell list and no budget of its own. Both come from the numerical
gate, so the two cannot drift apart: a cell added there is a cell this gate
runs.

## Running it

It is **not** in the default gate run, because it needs a memory daemon
reachable over RDMA. Name one in `SHADOWSPILL_NETWORK_PEER` as `host:port`:

```bash
SHADOWSPILL_NETWORK_PEER=192.168.50.32:17800 python -m tools.qualification.gates remote
```

**Give an address, never an ssh alias.** The control channel resolves the host
through `getaddrinfo`, which does not read `~/.ssh/config`, so a name that
works for `ssh` fails here -- and it fails as an unexplained create failure
rather than as a name lookup.

With no peer named the gate **skips and succeeds**, writing a summary that says
so: a machine with no peer is not a machine with a broken runtime.

## What it does not measure

There is deliberately no performance cell. The link is an order of magnitude
slower than local memory, so a rate measured over it is not comparable with
anything and is not a baseline. What this gate answers is whether the numbers
come out the same, which is a question the hardware can answer honestly. A step
here is several times a local one, and that is the link rather than a
regression.
