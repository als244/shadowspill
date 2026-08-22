# Fixed-offset placement

Every lease that a schedule creates has to be given one address, fixed before
the step runs, and the addresses have to fit in the execution pool. This page
is the contract for that assignment: what goes in, what comes out, how each
offset is chosen, and why the structure that chooses it looks the way it does.

[From a resolved program to leases](admission-leases.md) covers where the
leases come from; [physical admission](physical-admission.md) covers what the
resulting layout is used for.

## Input

One record per lease, and nothing else:

| field | meaning |
|---|---|
| `bytes` | how much the lease occupies |
| `alignment` | the offset must be a multiple of this |
| `predicted_start_ns`, `predicted_end_ns` | when it is live, half-open |
| `causal_start`, `causal_end` | its position in the operation order |
| `lease_id` | identity, used only to break ties |

The lifetime is half-open, so leases that merely touch are **not** live at the
same moment and may share an address.

The two pairs of boundaries do different jobs, and the distinction is what
makes the result safe under imperfect predictions: the predicted lifetime
chooses offsets, the causal boundaries decide whether a shared address is
legal. See [timings choose the offsets](#timings-choose-the-offsets-causality-makes-them-safe)
below. Everything else about a lease — what it is for, which task owns it,
whether it is an output — is irrelevant here.

## Output

An offset per lease, in input order, and the total bytes the assignment spans.
That span is the fixed slice; admission adds the caller-owned reserve and the
scratch reserve to get what the pool must hold.

## What makes it hard

Two leases may share an address only if their lifetimes do not overlap:

\[
[t_i^0,t_i^1)\cap[t_j^0,t_j^1)\ne\varnothing
\Longrightarrow
[o_i,o_i+s_i)\cap[o_j,o_j+s_j)=\varnothing.
\]

Minimising the span under that constraint is dynamic storage allocation, which
is NP-hard, so the assignment is greedy and the question is only how good the
greedy is. The floor it is measured against is the **peak live bytes** — the
most that is ever simultaneously live — because no assignment can be smaller
than that.

## The assignment

Leases are placed one at a time, in a fixed order, each taking the lowest
address that clears everything it overlaps in time.

**Order:** largest first; among equal sizes, longest-lived first; then
earliest; then lowest lease id. The first two are the heuristic — big,
long-lived leases are the hardest to fit, so they are placed while the space
is empty. The last two only break ties, and exist so the result does not
depend on the order the leases arrived in.

**Offset:** walk the addresses already taken, in address order, and take the
first aligned gap wide enough; if none, take the address just past them all.

## The occupancy index

Choosing an offset needs the addresses in use by leases live at the same time.
Finding those is the whole cost, so it has a structure: a segment tree over the
distinct lifetime endpoints, where a lease is recorded on the nodes that cover
its lifetime, and a query gathers the nodes its own lifetime touches.

**The nodes hold merged address ranges, not leases.** The offset depends only
on the *union* of the addresses in use, never on which lease owns what, and a
packed layout's union collapses hard — measured at 10 to 17 disjoint ranges
where 200 to 430 leases overlap. Inserting a range into a union can only merge
neighbours, never split them, so a node's list stays as short as the layout is
contiguous.

Three things were measured rather than assumed, and each ruled out an
alternative that looked better on paper:

- **Sorting the gathered ranges beats sweeping them.** A sweep that jumps past
  the furthest conflicting range rescans everything once per layer, and packed
  leases stack into many layers; it measured slower on every scenario.
- **Reporting each lease once matters.** A lease is recorded on several
  covering nodes, so one query can reach it repeatedly — 2.1 to 2.5 times.
  Sorting those repeats was pure waste.
- **The comparison is a single load, so an indirect comparator costs more than
  the comparison.** The sort is specialised rather than generic.

## Cost

`O(n log n + sum_i k_i log k_i)`, where `k_i` is the number of already-placed
leases overlapping lease `i`. `sum_i k_i` is the number of time-overlapping
pairs, which is `Theta(n^2)` when everything is live at once, so the worst case
is quadratic. Real layouts are far sparser — mean `k` about 1.5% of `n` — but
growth is superlinear.

Measured on real plans: 14.6 ms at 2,543 leases, 68 ms at 17,250.

## Timings choose the offsets; causality makes them safe

Placement decides two leases may share an address by looking at their
*predicted* lifetimes. Those predictions come from a simulation and will not
match a real run exactly. The assignment is still correct, and the reason is
that predicted time is used for one job only.

Each lease carries a second pair of boundaries alongside its predicted
lifetime: `causal_start` and `causal_end`, which are **positions in the
operation order**, not times — the sequence number of the operation that
acquires the lease and of the one that retires it. Before any two leases are
allowed to share an address, the layout checks

```text
predecessor.causal_end <= successor.causal_start
```

and refuses the layout otherwise. That is a statement about the order
operations happen in, which no amount of timing drift can change: the
predecessor's retirement *precedes* the successor's acquisition in the
schedule itself.

The runtime then enforces it. Every shared address produces a reuse
dependency, and a successor is held out of its transfer lane until the
predecessor has published the completion that frees the range. The wait is on
an event, never on a predicted time.

So the two roles are cleanly separated:

| what | uses | if it is wrong |
|---|---|---|
| choosing an offset | predicted lifetimes | the layout is bigger or smaller than it needed to be |
| allowing a shared address | causal order | cannot be wrong; it is a property of the schedule |
| honouring a shared address | published completions | cannot be wrong; the successor waits |

**Bad predictions cost bytes and time, never correctness.** A slow task
delays the successor that reuses its address, which is why the reuse slack
below is worth reporting — but the successor waits rather than writing into a
range someone still holds.

## Judging the result

Two numbers, both cheap and both worth reporting:

- `fragmentation_bytes` = slice size − peak live bytes, and
  `fragmentation_ratio` = slice size ÷ peak live bytes. **1.00 means the
  assignment is perfect** and the only way to shrink is to keep less resident.
  Measured range on this corpus: 1.00 for every layout that fit, 1.04 to 1.25
  for layouts that missed.
- The reuse edges the assignment creates, and how much slack each has. Sharing
  an address is what saves the bytes, and it is not free: the successor cannot
  start until the predecessor releases. Those edges cost nothing in simulation
  — placement only shares an address between lifetimes already disjoint in the
  predicted timeline — but they are what a real run has to honour, so their
  slack is the layout's exposure to timing drift. Measured, 13 to 24% of edges
  have exactly zero slack.

The two move in opposite directions: packing tighter reduces the first and
increases the second, and the assignment currently prices only the first.
