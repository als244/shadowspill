# Task boundaries

Every planned task runs between two calls. `before_task` makes the task's
inputs real and opens the scope its allocations belong to; `after_task` closes
that scope, publishes what the task produced, and hands the plan's actions to
the worker. Between them the dispatching thread runs the task's kernels and the
runtime does nothing.

Nearly everything ShadowSpill does to memory happens at one of these two
points, which is why they are worth reading before anything else in the
runtime.

Related: [memory runtime](memory-runtime.md) for pools, leases and the worker;
[physical admission](physical-admission.md) for the layout a plan is admitted
under; [Runtime C API](../c/runtime.md) for the exact signatures.

## What each call is responsible for

### `shadowspill_before_task_handle`

```text
before_task(runtime, task, compute_stream) -> bindings
    reject unless the handle belongs to this runtime and is a task
    claim the task invocation            # fails if one is already active
    record a BEFORE_TASK trace event
    acquire the task's input objects:
        for each distinct input:
            refuse if the runtime has already failed
            refuse if the object is not resident, or its version or lease
                differs from what the plan expects
            if a fetch is still in flight, make the compute stream wait on
                its readiness event
            take an owner reference and record the binding
    enter the task scope                 # this thread is now "inside" the task
    return the borrowed binding array
```

The bindings are a borrowed view of an array the task record owns; the caller
must not free them or use them past `after_task`.

The refusal in the middle is how a failure anywhere - including on the worker
thread - stops the next task. Object acquisition consults the failure latch
before it hands any object over, so a runtime that has already failed cannot
start new work.

### `shadowspill_after_task_handle`

```text
after_task(runtime, task, compute_stream)
    status = latched failure, if any
    validate the allocation contract is complete
    publish mutations: bump each mutated object's authoritative version
    if the task has actions or task-owned retirements:
        record a completion event on the compute stream
        attach that event to every lease the task retired
        instantiate the plan's actions against current object state
        publish the action batch and wake the worker
        wait until the worker has issued this batch's fetches
    on failure:
        discard the batch, clear pending handoffs, latch the failure
        publish a retirement event anyway, so allocations freed inside the
            task still have a completion to retire against
    release the completion event
    record an AFTER_TASK trace event
    leave the task scope
```

### `shadowspill_abort_task_handle`

The escape hatch when a frontend opened a scope and cannot reach `after_task`.
It finalizes the task's retirements, clears pending handoffs and leaves the
scope, without publishing mutations or actions. It is only valid from the
thread currently inside that task.

Abort exists because `before_task` and `after_task` are not adjacent: between
them the frontend rebinds storages, assembles arguments, and runs the kernels,
and any of that can raise. What actually calls it:

| Scenario | Where |
|---|---|
| Rebinding a storage onto its planned lease fails | between the two boundaries |
| Assembling the task's arguments raises | between the two boundaries |
| The compiled callable itself raises | during the kernel launch |
| Output classification or publication raises | inside `after_task`'s caller |
| The dispatching thread is interrupted - `KeyboardInterrupt`, a cancelled future | anywhere in between |

In each case the frontend has already claimed the invocation and opened the
scope, so simply propagating the exception would leave the task permanently
"active": the handle refuses a second invocation, the task's freed leases keep
their claim with no completion event coming, and closing the plan waits for a
scope that will never be left.

What abort deliberately does **not** do is publish. A task that did not finish
did not produce its outputs, so its mutations must not bump object versions and
its actions must not be instantiated - a prefetch triggered by a task that
never ran would fetch into a lease the plan expected that task to fill.

Abort is not a way to recover from a *runtime* failure. A latched failure is
not cleared by aborting; the next `before_task` still refuses. Abort only
closes a scope the frontend opened and cannot close normally.

### `shadowspill_submit_action_batch_handle`

An action batch is a task with no compute: a record whose actions are the point.
It enters a task scope and runs the same `after_task` body, so prefetches and
evictions can be issued without a kernel between them.

## What is asynchronous, from the dispatching thread

This is the part worth being exact about.

| Work | When the dispatcher returns from `after_task` |
|---|---|
| The task's kernels | Enqueued on the compute stream, not finished |
| Prefetches this task triggers | **Issued** on the transfer lane, not landed |
| Evictions this task triggers | May not be issued yet |
| Releases this task triggers | May not be issued yet |
| Retirement of leases freed in the task | Queued against the completion event |
| Returning retired ranges to the pool | Done by the worker, later |

`after_task` blocks in exactly one place: it waits until the worker has
*published* the batch's fetches. Published means the copy has been issued on
the transfer lane and its readiness event exists — not that any byte has moved.
That is enough for the next task, because the next `before_task` will insert a
stream wait on that readiness event rather than waiting on the CPU.

Everything else is the worker's, and the dispatcher never waits for it. The
wait itself is an active atomic poll; neither thread enters a condition wait,
a sleep, or a scheduler yield, because the dispatcher is on the critical path
of a step.

`shadowspill_plan_wait_idle` is the only call that waits for all of it.

### Who owns what

The split is worth stating directly, because both threads touch the same
objects, leases and pools.

| | Dispatching thread | Worker thread |
|---|---|---|
| Task scope | Owns it. Enters at `before_task`, leaves at `after_task` or abort | Never enters one |
| Object acquisition | Validates state, takes owner references, inserts stream waits | Never acquires for a task |
| Allocation | Serves every allocation, planned or dynamic, from its own thread | Never allocates for a task |
| Actions | Instantiates and publishes the batch; reserves each destination | Issues the copies, publishes readiness, completes and unlinks them |
| Retirement | Links freed leases to the scope; attaches one completion event | Polls the events, returns the ranges to the pool |
| Failure | Latches what it sees; refuses to start new work once latched | Latches what it sees; the dispatcher hears it at the next boundary |

Two rules follow, and both are load-bearing:

**The worker never blocks the dispatcher except at one point.** That point is
the fetch-publication handshake at the end of `after_task`. Everything else the
worker does - moving bytes, polling completions, returning ranges - happens
while the dispatcher is already running the next task.

**The dispatcher never does the worker's work.** It does not wait for a copy,
poll an event, or return a range, even when that would be convenient. An
allocation that cannot be served waits on the pool for the worker to release
something; it does not go and retire leases itself. This is why a stalled
worker shows up as a blocked allocator rather than as a dispatcher that
silently took over.

The handshake itself is an active atomic poll on both sides. Neither thread
enters a condition wait, a sleep, or a scheduler yield, because both are on the
critical path of a step and a scheduler round trip is longer than the work
being waited for.

## How allocations find their task

The task scope is thread-local. While a thread is inside a task, the scope
names the task, its invocation, the pool its allocations come from, how many
allocations it has made, and which leases it has freed.

```text
memory_pool_allocate(...)
    task  = current task scope, if this thread is inside one
    if inside a task:
        check the request against the task's allocation envelope
        match it against the next step of the task's allocation contract
        if the plan placed this allocation, use its fixed offset
    ...
    commit: charge the scope, advance the allocation ordinal
```

Three consequences follow from the scope being thread-local:

- An allocation made by a framework callback - PyTorch's allocator calling
  into ShadowSpill from inside a kernel launch - is attributed to the right
  task without anyone passing a task ID.
- An allocation made outside any task scope is unattributed and unplanned,
  which is legal but takes the dynamic path rather than a planned offset.
- Two threads can be inside two different tasks at once. Two threads inside
  the *same* task handle is refused, because the handle owns the validation
  and action records that a second invocation would race.

A lease freed inside the task is linked to the scope rather than retired
immediately. `after_task` attaches one completion event to all of them, so the
worker can return every range the task freed against a single fence instead of
one event per free.

## What a plan's actions do here

The plan says which objects to fetch, evict or release, and which task triggers
each. Those actions are not executed at `before_task`; they are instantiated at
the `after_task` of their trigger task, against object state as it is then.

An action's destination lease is reserved before the batch is published, so a
prefetch that cannot fit fails at the boundary that triggered it rather than
somewhere inside the worker. That reservation is why `after_task` can return
`NO_PROGRESS`: the trigger's fetch had nowhere to land and nothing was left to
release for it.

## Failure

Both boundaries return a `ShadowSpillStatus`. A failure latched anywhere -
including on the worker - reaches the caller at the next boundary:
`before_task` refuses to start, and `after_task` folds the latch into its
return rather than reporting its own success. The first cause is preserved:
`shadowspill_runtime_failure()` reports the failure that stopped the runtime,
with the reason naming what was attempted and refused, while the return value
reports what this call saw.

`after_task` cleans up whether or not it succeeded. It is never correct to skip
it because an earlier call failed; the scope stays open and the task's leases
keep their claim until it runs.
