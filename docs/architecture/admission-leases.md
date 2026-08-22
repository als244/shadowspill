# From a resolved program to leases

Physical admission turns a schedule into a set of **leases** — one per object
generation that occupies execution memory — and then places them at fixed
addresses. This page is the contract for that derivation: what is fixed before
it starts, what the walk produces, and the rules that decide a lease's
identity. [Physical admission](physical-admission.md) covers what happens to
the leases afterwards.

## What is fixed, and at which level

Four levels, each fixing something the level below varies:

| level | per plan | fixed here | varies below |
|---|---|---|---|
| plan | 1 | program, admission topology, simulation config, pool capacity, alignment | which resolved program |
| **resolved program** | a handful | the executing task set, its runtimes and object accesses | which candidate policy |
| candidate | tens per resolved program | residency strategy, prefetch rule, coalescing | the target capacity |
| probe | a few per candidate | — | the repaired schedule |

A **resolved program** is the Program with every alternative fixed, leaving one
concrete task set. Recomputation is the frontend alternative that produces
them, but the core is unaware of that: it plans memory for whatever tasks
execute. A hand-authored Program with no alternatives resolves to exactly one,
and nothing downstream can tell the difference.

Resolved programs are independent problems. Nothing flows between them; each
is planned separately and the best result is taken.

### The shared setup

Everything a measurement needs that a schedule does not change is fixed per
resolved program, so it is prepared once and reused across every candidate and
every probe beneath it:

- the **selected task set**, with each task's runtime, workspace and object
  accesses;
- the **compiled simulation template**, derived from that task set;
- the **compiled admission topology**, derived from the template — per-task
  allocation steps, fresh outputs, replacements, storage handoffs, alignment;
- the **compute floor**, the critical path through the selected tasks, which
  no schedule for this resolved program can beat.

Only the schedule varies below that, so a measurement is a function of the
setup and one schedule. Building the setup per probe instead is measurably
wasteful — template and topology compilation cost tens of milliseconds against
calls of a few milliseconds — but the reason to hold it once is that it says
plainly what a repair can and cannot change.

## The walk

Executing a schedule implies a sequence of **pool operations**: `RESERVE`,
`ACQUIRE`, `ACQUIRE_RESERVED`, `BEGIN_RETIREMENT`, `COMPLETE_RETIREMENT`,
`RELEASE`, `PUBLISH_DEPENDENCY`. These are allocator calls, not the schedule's
memory *actions* — see [that distinction](physical-admission.md#two-vocabularies-actions-and-operations);
one action implies about three operations, and a task acquires and retires
leases with no action at all.

`csrc/planner/src/admission/operations.c` derives the sequence in causal
order. Each operation records where it sits and why the lease exists.

### Where an operation sits

| boundary | names | index means |
|---|---|---|
| `INITIAL` | neither a task nor an action | **nothing — it carries no identity** |
| `TASK_START`, `TASK_COMPLETION` | a task | the task |
| `ACTION_TRIGGER`, `ACTION_COMPLETION` | an action **and** its triggering task | the action |

The last row is the one that catches readers: an action-boundary operation
belongs to both. The action names what happened; the triggering task names
when. Recovering the task means looking the action's trigger up in the
schedule, not reading the operation's index.

### Why a lease exists

| purpose | the lease is |
|---|---|
| `INITIAL_OBJECT` | an object already resident when the step begins |
| `TASK_WORKSPACE` | anonymous scratch for one task |
| `TASK_OUTPUT` | a fresh output a task produces |
| `MUTATION_REPLACEMENT` | the generation an in-place mutation supersedes |
| `RELEASE` | retired by a scheduled release action |
| `EVICTION` | retired by an offload, once the copy lands |
| `FETCH_DESTINATION` | the destination a fetch reserves at its trigger |
| `TERMINAL_COMPLETION` | the completion of an eviction's retirement |

Purpose is finer than boundary and is **not recoverable from the operation
stream**: at a task start a lease may be workspace, an output, or a
replacement, and the emitted record is identical in all three. It is decided
where it is known, during the walk, and recorded per operation.

Two rules are easy to get wrong:

- **An eviction's retirement changes purpose between begin and completion.**
  The begin is `EVICTION`; the completion is `TERMINAL_COMPLETION`, because by
  then the copy has landed and the lease is simply gone. Lifetime construction
  dates evicted leases from the *transfer* interval and everything else from a
  *task* interval, so conflating the two misdates every evicted lease.
- **A task allocation's alias comes from its allocation step**, not from the
  lease. Anonymous workspace has no alias.

## Things that happen without an operation

Two transitions move or reuse a lease and emit nothing. Both must be replayed
from the topology; neither is visible in the stream.

**Slot reuse.** A task may free an allocation slot and reallocate it. The
second allocation reuses the same lease and emits no operation of its own, so
a lease is tied to its allocation step through the slot, not through the
stream. The reusing step also **overwrites the lease's provenance**, so a
lease records what it most recently became rather than what it first was. That
is deliberate: a slot first used as workspace and then retained as an output
really is an output by the end of the task, and the fixed/dynamic check below
depends on seeing it that way.

**Storage handoff.** A live lease moves from a source alias to a destination
without allocating. The active-alias map has to follow it or it ends holding
aliases that no longer own anything.

## What the walk produces

Besides the operations, four maps, each the subset of the walk with one
meaning:

| map | keyed by | used for |
|---|---|---|
| `initial_alias_leases` | alias | the certificate's initial-object leases |
| `task_allocation_leases` | (task, allocation ordinal) | binding runtime allocations to planned offsets |
| `action_destination_leases` | action index | binding a fetch's destination to its planned offset |
| `active_aliases` | alias | which lease currently owns each alias |

`active_aliases` is the live state, not a history: acquisitions add,
retirements remove, handoffs move. At the end of the walk it is exactly the
final residency.

## Fixed and dynamic

Every lease is placed at a fixed address **except** the final leases of
caller-owned aliases — the outputs that remain device-resident when the step
ends, which the caller may hold across later invocations. Those are excluded
from the fixed slice and covered by `dynamic_reserve_bytes` instead, so no
later invocation can plan an address the caller is still using.

Two consequences worth stating plainly, because both are easy to assume
backwards:

- **Task workspace is fixed, not dynamic.** It is placed inside the slice with
  a lifetime equal to its task's execution, and its offsets are reused across
  tasks. "Dynamic" here means caller-owned, not transient.
- **Only a real output may escape.** A caller-owned lease must have purpose
  `TASK_OUTPUT` or `FETCH_DESTINATION`; anything else is a bug in the walk.
  This is the check that the provenance-overwrite rule exists to satisfy.

The caller supplies which aliases are caller-owned; the walk resolves each to
the lease that owns it at the end, through `active_aliases`.
