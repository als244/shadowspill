# Failure, abort, and process exit

Four things can go wrong at four different scopes, and they are handled
differently on purpose. A task can fail while a thread is inside it. A planned
callable can fail and has to give the model its storage back. A runtime can
fail and has to stay closable. And the process itself can end while any of
that is in flight, including because something outside ShadowSpill decided so.

The first three assume the program continues. The fourth assumes it does not,
and that difference decides everything below.

Related: [memory runtime](memory-runtime.md) for leases, the worker, and the
failure latch; [task boundaries](task-boundaries.md) for what `abort_task`
does; [errors, failures, and cleanup](../python/failures.md) for the
Python-facing taxonomy and the normal close order.

## A failure inside a task

A failure latched anywhere, including on the worker, reaches the dispatching
thread at the next boundary. `before_task` refuses to start new work once the
latch is set, and `after_task` folds the latch into its return rather than
reporting its own success. The first cause is preserved: the failure names
what was attempted and refused, while the return value says what this call
saw.

A frontend that opened a task scope and cannot reach `after_task` closes it
with `abort_task`, which publishes nothing; [task
boundaries](task-boundaries.md#shadowspill_abort_task_handle) says what it
finalizes and which cases reach it.

## A failure in a planned callable

A callable owns admitted execution state, so closing it is what returns the
model's storage. Its close is ordinary program work: it may wait, because the
program is still running and the worker is still able to make progress.

## A failure in the runtime

`shadowspill_runtime_close` rejects new work, waits until two counters reach
zero or a failure is latched, synchronizes every transfer lane, joins the
worker, and releases everything it owns. It is synchronizing and idempotent,
and it returns the first latched failure. Draining is correct here because the
process continues: outstanding transfers can still complete, and a caller may
go on to use the memory they were writing into.

The two counters are not the same kind of thing, which is what the next
section turns on:

| counter | what it counts | what finishing it needs |
|---|---|---|
| queued actions | fetches, write-backs, evictions and releases a task triggered, published to the worker and not yet complete | bytes still to move on a transfer lane, or a range still to hand back |
| pending retirements | ranges freed inside a task, fenced against a completion event | no bytes move; the worker has to observe that event and return the range to the pool |

## The process exiting

This is the case that behaves differently, and it is worth stating why.

A process can start exiting while a runtime still holds either counter, and
neither can make progress once exit handlers are running. Retirements make the
point sharpest: a runtime can have nothing mid-transfer and still hold freed
ranges waiting on completion events that nobody will ever observe.

It does not have to be ShadowSpill's decision, or Python's: any C code in the
process can call `exit`, and some does. A kernel launcher that finds no kernel
image for the device -- what happens when kernels built for one accelerator
architecture are asked to run on another -- may print a device error and call
`exit(1)` rather than return a status.

`exit` runs registered handlers on the calling thread and only reaches
`_exit`, which asks the kernel to tear the process down, after they all
return. So a handler that blocks does not delay the exit; it prevents it. The
process stays alive with every thread parked, and a parent waiting on it waits
forever.

**So the exit path never waits.** The adapter registers its handler with
`on_exit` rather than `atexit`, which hands it the exit status, and the handler
abandons the runtime instead of closing it. Abandoning skips three things that
can block:

| skipped | why it would block |
|---|---|
| the drain | outstanding work cannot complete once exit handlers are running, and the wait's only escape is a latched failure that will never appear when the exit came from elsewhere |
| lane synchronization | waiting on the device for the work it just declined to finish |
| aborting the open task scope | finalizing retirements takes the pool's foreground lock, which the worker may hold and may never release |

**What it still does.** Abandoning stops the worker, joins it, and releases
what the runtime owns: every route's lane and stream, both event pools, the
transfer profiles, and every memory pool, which is what unregisters pinned
host memory and frees device memory. Nothing is leaked that a running program
would have kept. The distinction is not what gets released; it is that
nothing is waited for.

**What it reports.** The handler writes one line to stderr naming the exit
status, how many actions and retirements were outstanding, and whether a
failure was latched:

```text
ShadowSpill: the process is exiting with status 1 and 0 action(s) and
5 retirement(s) outstanding; ShadowSpill did not wait for them.
Latched failure: none, so the exit came from outside ShadowSpill.
```

A nonzero status with nothing latched is the informative case: no call of ours
failed, so something else ended the process. That combination is otherwise
invisible, because the third party that called `exit` leaves no exception, no
traceback, and no failure of ours to find. The line is silent when the process
exits cleanly with nothing outstanding, which is the ordinary path.

## What this means for a harness

A child that stops making progress is not distinguishable from a child that is
working, so anything running planning or execution in a subprocess should bound
it in time and require the artifact it promised, rather than trusting an exit
status alone. Program collection and planning evaluation already do; see
[qualification](../../qualification/README.md).
