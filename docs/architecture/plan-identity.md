# Plan identity

A plan is the unit of admitted work: one set of tasks, one fixed layout, one
pool pair. A pool, though, outlives any single plan, and more than one plan may
share it. So every question of the form "whose is this?" needs a name for the
plan, and a task id is not one.

## Why a task id cannot answer it

Task ids are plan-local. The frontend mints them from the program's own
numbering, so `task_001112` is id 1112 in *every* plan. Two plans on one pool
both have a task 1112, and a lease recording only that tells you nothing about
which work made it.

A task record's first field is the plan that admitted it, so anything holding a
task can name one. A lease is the case that cannot: it needs the plan's own
number, recorded on it directly.

## The id

`shadowspill_runtime_next_plan_id` hands out the ids. It only counts up, so an
id names one plan for the life of the runtime and is never reissued. The caller
takes an id, then names it in the `ShadowSpillPlanDescription` it creates the
plan from.

The id is taken before creation rather than returned by it because the caller
needs it earlier. Allocation scopes — profiling probes, which run under the plan
but outside any of its tasks — name the plan they are measuring for, and the
runtime has no task there to read a plan from. One number therefore spans the
whole period a plan owns, the profiling that precedes its tasks included.

Creation enforces that an id is used once: an id the runtime did not issue is
refused, and so is one some plan has already been created with, whether that
plan is still live or has since been destroyed.

Zero is not a plan id. It is what a lease made outside any plan reports, so no
plan may hold it.

## What a lease records

Every lease carries a `ShadowSpillAllocationOrigin`: the plan id and the task
id, together. The two travel as one value because neither identifies anything
alone — the plan id says whose work it is, the task id says which step of that
work.

The origin is passed in rather than read from the running thread, because it
belongs to the work and not to the thread doing it. A pre-task action batch
creates leases for a task it is not running, and stamping the dispatching
thread's scope there would attribute the bytes to the wrong task.

`shadowspill_memory_pool_live_allocations` reports both halves, which the
adapter's `PoolAllocation.origin` names, so a range in the pool reads as:

    plan 7 task 1112                   a task of plan 7 made it
    plan 7 profiling scope #40         a probe for plan 7 made it
    runtime object                     the runtime holds it for a named object
    no scope                           no scope was open: nothing a plan made

The first two lines are a closing plan's to release. The last two are not. They
differ from each other, which is why they are not one label: a runtime object's
storage backs something the program named and any number of plans may bind it,
while `no scope` is a provider taking its own workspace between tasks and backs
nothing.

`runtime object` is recorded on the lease and read through the object registry.
It does not appear in `shadowspill_memory_pool_live_allocations`, which covers the
ranges a pool has published, and a registered object's storage is reserved without
being published. [Shared objects](shared-objects.md) has the detail.

## The registry

The runtime keeps one slot per plan id that has ever named a plan, holding that
plan's record while it exists. `csrc/src/runtime/plan/registry.c` is all of it.

It answers two questions. `shadowspill_runtime_plan` maps an id to its record,
and `shadowspill_runtime_plan_state` says what became of an id:

| state | meaning |
|---|---|
| `UNKNOWN` | no plan was ever created with this id |
| `LIVE` | created and open; it may still admit work |
| `CLOSED` | closed, record still present: it admits no further work |
| `DESTROYED` | closed and the record freed; the id stays claimed, because a lease may still carry it |

The state outliving the record is the point, and what it is for is catching a
defect. A plan that closes is expected to release every range its own scopes
made, so **a live allocation naming a `CLOSED` or `DESTROYED` plan should not
exist**. Keeping the id answerable is what turns that from something invisible —
a range at an offset nobody can account for — into a question with an answer.

An object shared between plans is the exception by design: it outlives any one of
them, so a lease that deliberately crosses that boundary names a plan that has
gone. [Shared objects](shared-objects.md) is how that storage is owned.

A range taken outside any plan's scope is not an exception to it either. A
provider allocating its own workspace with no scope open — a handle's working
buffer, say — produces a lease naming no plan at all, so there is no plan whose
state could be the wrong one.

The registry has its own lock and never takes another while holding it, so
asking what an id means does not wait behind plan creation or teardown.

Ids are a dense counter from one, so the slots are indexed by id and the lookup
needs no hashing: a hash table whose hash is the identity and whose buckets hold
one entry each, with the indirection removed. A slot carries a claimed flag
beside its record, which is what separates an id no plan was ever created with
from one whose plan has been freed — neither has a record.

## What it is for

Two things.

**Attribution.** A refused fixed layout lists what occupies the pool. With the
plan on each range, residue left by an earlier plan is visible on sight instead
of being inferred from where the offsets fall.

**Cleanup.** A plan closing can ask which ranges its own scopes made, which is
exactly the set it is entitled to release: everything its tasks and its
profiling probes allocated, and nothing a sibling plan or a provider did.
Without the id it would have to guess from task ids that a sibling plan may also
be using, and would have no way to tell its own probe's leftovers from a
provider's workspace.

See [task boundaries](task-boundaries.md#what-a-scope-owes-when-it-ends) for
what a scope owes the allocations it made, and
[memory runtime](memory-runtime.md) for the lease itself.

Previous: [The planning pipeline](planning-pipeline.md). Next: [Shared
objects](shared-objects.md).
