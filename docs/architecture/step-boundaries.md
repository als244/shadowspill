# Step boundaries

A planned callable is built to be invoked repeatedly. Each invocation runs
the same fixed schedule between two boundary regions: the closing writeback
of the invocation before it and the opening restore of its own initial
objects. This page defines that cycle — why repetition is sound, which
synchronization points separate one step from the next, in what order the
opening restore runs, and what "step time" means once the boundary regions
are accounted for.

## The invocation cycle

[Lowering](lowering.md) declares the boundary residency contract:
parameters, external inputs, and optimizer state are spill-resident both
when a step begins and when it ends; only public outputs end on device,
and they leave the cycle as caller-owned storage. The emitted schedule
keeps the declared final residency verbatim, so the state a finished
invocation leaves behind is exactly the state the next invocation's
preconditions require. The plan is closed under repetition: no data
placement carries from one invocation into the next, and what does persist
between invocations is invariant machinery — pools, leases, the fixed
layout slice, admitted task and action batches, and the sealed event pool
(see [Memory runtime](memory-runtime.md)) — never residency.

Within a step, the schedule is free to decide that some of those declared
spill-resident objects should already be on device when the first tasks
run. That solved set is recorded as the schedule's initial device
residency, and realizing it is the opening restore below. The declared
contract is the cross-invocation interface; the initial device set is an
intra-step decision.

Repetition is sound because the contract is enforced at every layer rather
than assumed. The [simulator](simulation.md) rejects a schedule whose
declared final residency is not reached; [physical
admission](physical-admission.md) proves the layout still holds what the
step promised to end holding; and the runtime enforces each transition at
admission time — restoring an object requires a current spill copy and no
device copy, and staging a new microbatch requires the input object to be
spill-resident. A violated precondition is a reported plan violation, not
silent corruption, so each invocation may begin on an assumption that has
been proven rather than sampled.

## What ends a step

The last tasks that touch each spill-final object trigger its terminal
writeback, an ordinary scheduled action handed to the worker at those
tasks' `after_task` boundaries. Public outputs are acquired for the
caller with stream-ordered waits and become the caller's storage. The
callable then returns **without draining**: terminal transfers and lease
retirements are still in flight on the worker when control reaches the
caller, and the compute stream may still be executing the final kernels.
Nothing in the return path synchronizes the host with the device.

## What begins the next one

The next invocation pays for that freedom in a fixed sequence of
synchronization points:

1. **Plan-idle wait.** The host blocks until the previous invocation's
   actions and retirements have fully drained. This is where the terminal
   writeback is actually paid: the new step must not overlap transfers
   that are still completing the old step's contract.
2. **Input staging.** After a runtime-wide idle wait, each microbatch is
   copied on the host into its pinned spill lease. Staging requires the
   input objects to be spill-only — a precondition the previous step's
   contract guarantees.
3. **Opening restore.** One reusable action batch — admitted once, under
   a reserved task identity outside the plan's task range — submits a
   restore for every entry in the schedule's initial device residency.
   The submitting thread records a compute-stream event that triggers the
   batch, hands it to the worker, and returns only once the worker has
   issued every copy onto the strict-FIFO fetch lane. The batch is not
   part of the schedule's actions: it re-establishes the schedule's
   assumed starting state rather than executing the schedule.
4. **First task boundaries.** Each task's `before_task` inserts
   stream-ordered waits for inputs still in flight and for the ranges its
   allocations reuse (see [Task boundaries](task-boundaries.md)). The
   compute stream, not the host, waits for the restore to deliver.

## Restore order

The fetch lane is strictly first-in, first-out, so the order of the
opening restore decides how long the earliest tasks wait. The batch — and
the fixed-layout destinations paired with it position by position — is
ordered by the program's first consuming task: aliases the first task
reads come first, aliases first consumed by the same task follow that
task's own input order, and aliases no task consumes come last in their
emitted order. Under this order a task's inputs arrive no later than the
work ahead of them requires, the first task waits only for its own
inputs, and the remainder of the restore streams in behind compute
instead of in front of it.

## What step time means

Three quantities describe one invocation, and they are not the same
number:

- **The measured step** is the wall time of one full cycle: the plan-idle
  wait, staging, the opening restore, every task, and the terminal
  writeback. Repeated timed invocations measure this cycle — each step's
  closing drain is paid inside the next step's opening wait, so a
  contiguous sequence of invocations divides cleanly into whole cycles.
- **The selected task span** is device time from after the first task's
  readiness waits to the completion of the final task's kernels. It
  excludes both boundary regions by construction and is the number that
  compares directly against profiled task cost.
- **The simulated makespan** is the selected span plus the terminal
  transfer tail. The simulator prices the writeback — it runs until
  terminal transfers drain — but not the opening restore: initial device
  objects are seeded ready at time zero, and no scheduled action can
  precede the first task. The prediction is therefore exact for the span
  and tail and blind to the opening region. With the restore in first-use
  order the unmodeled cost is bounded by the first task's own inputs plus
  the plan-idle and staging waits above, rather than by the size of the
  initial device set.

The [StepResult diagnostics guide](../python/step-diagnostics.md) exposes
each piece: the first task's readiness wait is reported on its own,
outside every span-relative number; the simulated terminal tail is
reported beside the makespan; and dispatch timing separates the plan-idle
wait and the restore submission from task dispatch.

Previous: [Task boundaries](task-boundaries.md). The
[simulation](simulation.md) page defines the prediction this cycle is
measured against.
