# Fixed-offset placement

Every lease that a schedule creates has to be given one address, fixed before
the step runs, and the addresses have to fit in the execution pool. This page
is the contract for that assignment: what goes in, what comes out, how each
offset is chosen, and why the structure that chooses it looks the way it does.

[From a resolved program to leases](admission-leases.md) covers where the
leases come from; [physical admission](physical-admission.md) covers what the
resulting layout is used for.

## Input

Four numbers per lease, and nothing else:

| field | meaning |
|---|---|
| `bytes` | how much the lease occupies |
| `alignment` | the offset must be a multiple of this |
| `start_ns`, `end_ns` | when it is live, half-open |

The lifetime is half-open, so leases that merely touch are **not** live at the
same moment and may share an address.

Placement is not told which lease a record belongs to. Identity would only
ever break a tie between records equal in every key, and the input index does
that, so the layout depends on the records and the order they arrive in and on
nothing else. Everything else a lease carries — what it is for, which task
owns it, whether it is an output, and the causal boundaries below — belongs to
the caller and never crosses into the placer.

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
earliest; then lowest input index. The first two are the heuristic — big,
long-lived leases are the hardest to fit, so they are placed while the space
is empty. The last two only break ties, and exist so that records equal in
every key still get a defined order.

**Offset:** walk the addresses already taken, in address order, and take the
first aligned gap wide enough; if none, take the address just past them all.

## The resident slice

Objects under `minimum_object_bytes_evict_eligible` are never cut, so their
leases run from their first access to their last: long-lived and small, the
worst shape for a placer whose other leases come and go. Placing them among
the rest would leave holes the size of nothing useful across the whole layout,
and they are too small for packing them on their own to be worth the
machinery. So every lease of such an object gets a static home instead: a
range of its own in the resident slice, which follows the main assignment at
the end of the fixed range. The leases left out of the assignment above are
walked in lease order, each taking the next offset that satisfies its
alignment — 256 bytes, like every lease — and the fixed range ends past the
last of them. Nothing in the slice is ever reused, so nothing in it needs a
completion dependency, and the certificate's `fixed_slice_bytes` is the main
assignment's extent plus `resident_slice_bytes`.

The slice's size is known before the search. At problem preparation the
planner sums the homes such an object will need — one, plus one for every task
that mutates it in place, since that task holds both generations — each
rounded up to the alignment, and hands the reducer the device's capacity
minus that sum, so these objects are never charged per boundary again. Every
plan the search measures lays its slice out from the leases the plan actually
has, exactly as the final layout does. The emitter fetches each such object
at a trigger chosen once, as late as the ideal timeline allows, so no fetch
rule moves a lifetime the slice was sized without.

## The occupancy index

Choosing an offset needs the addresses in use by leases live at the same time.
Finding those is the whole cost, so it has a structure: a segment tree over the
distinct lifetime endpoints, where a lease is recorded on the nodes that cover
its lifetime, and a query gathers the nodes its own lifetime touches.

**The nodes hold merged address ranges, not leases.** The offset depends only
on the *union* of the addresses in use, never on which lease owns what, and a
packed layout's union collapses hard. Inserting a range into a union can only
merge neighbours, never split them, so a node's list stays as short as the
layout is contiguous — and that is what keeps a query cheap, because a query
pays for the ranges each covering node holds.

Three things were measured rather than assumed, and each ruled out an
alternative that looked better on paper:

- **Sorting the gathered ranges beats sweeping them.** A sweep that jumps past
  the furthest conflicting range rescans everything once per layer, and packed
  leases stack into many layers; it measured slower on every scenario.
- **Merging beats de-duplicating.** An earlier design stored leases on the
  nodes and skipped the repeats a query reached through several covering nodes
  at once — 2.1 to 2.5 of them per lease. Merging is strictly better: it
  collapses neighbours that de-duplication cannot, and it still leaves the
  same per-node duplication, now cheap enough not to be worth removing. That
  duplication is why a query can return more ranges than the lease has
  conflicts, and the [cost section](#cost) shows it losing to merging on every
  plan above about 8,000 leases.
- **The comparison is a single load, so an indirect comparator costs more than
  the comparison.** The sort is specialised rather than generic.

## Cost

Placing one lease is: gather the occupied ranges it could collide with, sort
them by address, walk them for the first gap, insert the chosen range. Sorting
is the only superlinear step, so the total is

```text
O(n log n  +  sum_i r_i log r_i)
```

where `r_i` is the number of address ranges the gather returns for lease `i`.
The `n log n` is the one-off ordering of the leases themselves. Everything
else lives in `r_i`, so `r_i` is the number that matters.

**Why the sort exists.** Lowest-fit needs the occupied ranges *in address
order* — it walks them from zero and stops at the first gap wide enough. The
index is keyed by time, not address, so what it returns is a bag of ranges
gathered from the tree nodes covering the lease's lifetime, in node order.
Sorting it is what turns a set of conflicts into a walkable address line.

**What `r_i` is bounded by.** Two things, and neither is `n`:

- Ranges are stored on every node covering their interval, and a query touches
  `O(log n)` nodes per level of the tree, so `r_i` grows with how many nodes
  the lease's lifetime spans. Long-lived leases pay the most.
- Each node holds a *merged* union, so its list is short when the layout is
  contiguous. This is where the packing pays for itself twice: a tight layout
  is both smaller and cheaper to extend.

The loose upper bound is `k_i`, the number of already-placed leases that
overlap lease `i` in time — but `r_i` is not bounded by `k_i` in either
direction, because merging pushes it down while per-node duplication pushes it
up.

**Measured on the thirteen captured plans.** `k` is the conflict count, `r` is
what actually got sorted:

| plan | `n` | peak live | mean `k` | `Σk` | mean `r` | `Σr` |
|---|---|---|---|---|---|---|
| olmoe fast | 2,542 | 287 | 198 | 0.50 M | 196 | 0.50 M |
| llama3 fast | 3,038 | 304 | 128 | 0.39 M | 151 | 0.46 M |
| olmoe medium | 8,678 | 285 | 199 | 1.73 M | 99 | 0.86 M |
| llama3 slow | 16,892 | 320 | 238 | 4.01 M | 84 | 1.41 M |
| qwen35 slow | 27,234 | 388 | 226 | 6.15 M | 104 | 2.82 M |

Three things this shows:

- **The quadratic worst case is nowhere near.** `Σk` is 15.6% of `n²/2` on the
  smallest plan and 1.7% on the largest; it *falls* as plans grow, because
  peak live leases stay near 300 no matter how long the step is. Peak live is
  set by the model's working set, not by the step's length.
- **Merging is what makes it cheap, and it pays more as plans grow.** `Σr` is
  the same as `Σk` at 2,542 leases and a third of it at 16,892: there is
  little to collapse in a small layout and a lot in a large one. Within one
  family, `Σr` grows *sublinearly* in `n` — llama3 fast to slow is 5.6× the
  leases for 3.1× the sorted volume.
- **The distribution is skewed, not flat.** On the largest plan the median
  lease gathers 27 ranges and the worst gathers 15,407. A handful of
  long-lived leases — the resident parameters — dominate the total, which is
  why the sort is specialised rather than generic: most calls are tiny.

The profile agrees with the model. On llama3 slow, `sort_by_address` is 42% of
the run, the gather 14%, the moves the sort makes 10%, and insertion under 1%.

End to end, one placement call: 14.5 ms at 3,039 leases, 65 ms at 17,250.

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

Four numbers describe a layout, each computed where the information exists:

- `slack_bytes` on the fixed layout (`FixedLayout.slack_bytes`,
  `src/shadowspill/planner/admission/layout/model.py`) is the pool capacity
  minus the bytes the assignment requires: how much of the pool the layout
  leaves untouched.
- `predicted_fragmentation_bytes` in the execution plan's admission block is
  the planner's estimate of slab bytes that lifetimes will leave unusable; it
  can never exceed the slab.
- `peak_fragmentation_bytes` from admission replay
  (`<shadowspill/admission_replay.h>`) is the largest gap the replayed
  allocator actually left between live ranges while honouring the layout.
- `external_fragmentation_bytes` in the runtime statistics is the same
  quantity observed on the real pool: free bytes that no single request can
  use because they are not contiguous.

The first two are predictions the layout is built against; the last two are
what the allocator, replayed or real, made of it. When the replayed peak is
larger than the prediction, the lifetimes were wrong, not the assignment:
sharing an address is only ever granted between lifetimes already disjoint
in the predicted timeline, so a bad prediction costs bytes and a successor's
wait, never correctness.
