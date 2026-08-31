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

Two numbers, both cheap and both worth reporting. Neither is implemented;
the metrics the code does compute are `predicted_fragmentation_bytes`,
`peak_fragmentation_bytes`, and `external_fragmentation_bytes`, which
measure different quantities:

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
