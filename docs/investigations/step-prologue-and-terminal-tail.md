# Step Prologue and Terminal Tail

> **Historical, non-normative investigation.** This report preserves
> evidence from the implementation under investigation. Current behavior is
> defined by the [simulation architecture](../architecture/simulation.md)
> and the [StepResult diagnostics guide](../python/step-diagnostics.md).

## Purpose

Every repeated invocation of a planned training callable spends time
outside its selected task span: a startup prologue that restores the
plan's initial device objects from the spill pool, and a terminal tail
that writes the step's final spill residency back out. The simulator
prices the tail and none of the prologue, which made its prediction
systematically optimistic — the mechanism behind the one-sided simulator
error the performance gate tolerates. This note pins the mechanism to
code, decomposes the cost exactly, and records which remedies the
evidence accepted and rejected. Working data and scripts:
`docs/internal/plans/cyclic_placement_0901/`.

## The mechanism in one paragraph

Training lowering declares parameters and inputs spill-resident at both
step boundaries. The planner's emitted schedule overwrites the initial
half with whatever its residency solve kept on device at boundary 0 —
two unremovable "anchor" inputs of the first task plus a much larger
greedy pre-placement — while the final half stays as declared. No
scheduled action can precede the first task, so the simulator seeds the
initial set as ready at t=0 for free, while the runtime must actually
restore it: each invocation first drains the prior step's terminal
transfers, stages the new microbatches into pinned spill, then submits
one out-of-plan FIFO prefetch batch for the whole initial set before the
first task's inputs can be waited on.

## The decomposition (clean-machine qualification, three cells)

Out-of-span cost per step, `median_step − median_selected_span`, with the
exact identity `out-of-span = terminal tail + simulated-span pessimism +
gate error` (residual 0.000000 on every cell):

| cell | step (s) | out-of-span | tail | span pessimism | gate error |
|---|---|---|---|---|---|
| llama3 | 19.7707 | 0.4835 (2.4%) | 0.1696 | −0.3151 | +3.29% |
| qwen35 | 21.8898 | 0.5640 (2.6%) | 0.2422 | −0.0331 | +1.65% |
| olmoe  | 4.9656  | 0.3295 (6.6%) | 0.1223 | +0.0618 | +3.02% |

The measured head — the first task's readiness wait, which by
construction sits outside every span-relative number — is 0.330 / 0.296 /
0.275 s. An nsys trace of thirteen olmoe invocations splits the rest of
the wall: prior-step drain ~220 ms (matching the step-minus-dispatch
residue to 0.3 ms), input staging ~63 ms, initial-batch submission under
1 ms. llama3 is the outlier in kind: half its gate error is in-span
simulator optimism, a profile-fidelity matter unrelated to placement.

## What sets the head: queue order, not queue size

The initial sets are 9.7–11.9 GiB of parameters and buffers — no
optimizer state, no microbatch inputs — and all but the two anchors come
from the greedy pre-placement. The prefetch batch runs in the schedule's
emitted order, which is registration order; that order is first-use order
to within two adjacent swaps, except that one input of the first task
lands at the very end of the queue on every cell (rank 245/245, 261/261,
145/145 counting from one). The traced first kernel starts 282 ms after
submission (olmoe), partway through a ~577 ms wire window — the gate is
the task's last-arriving input, and its position in the queue decides the
wait. Ordered by first use, the same wire delivers the first task's
inputs in ~8–80 ms.

Two further measurements close the causal story. Re-planning with
`InitialPlacement.REQUIRED` collapses the initial set to the two anchors
and shows the greedy head's simulated value is +0.017 to +0.148 s per
cell — less than the real, unpriced head it induces on every cell. And
142 of olmoe's 145 initial objects are evicted mid-step after first use,
so the set the runtime restores each step is one the schedule itself
discards almost immediately.

## Remedies the evidence rejected

Making the step end holding what it assumes it starts holding — so
repeated invocations meet a warm device — was priced directly by the
solver. Requiring final residency to equal the solved initial residency
produced capacity violations on every cell and +1.14 / +0.25 / +0.43 s of
simulated makespan: more than the waste it recovers on two of three
cells. The fixed point itself is well-behaved (the solved initial set is
unchanged under the constraint), but there is no end-of-span headroom for
it — 9.3 / 7.2 / 9.9 GiB of headroom against 11.9 / 10.6 / 9.7 GiB of
initial set, by an occupancy reconstruction that reproduces the
simulator's peak exactly — and olmoe's fetch lane (64% occupied, idle
mostly early in the step) cannot hide the traffic where it would be
needed. Even the minimal variant, persisting only the two anchors,
carries violations on two cells for at most −0.08 s of simulated benefit.

## The adopted remedy

Submit the initial placement batch in plan first-use order. The batch
content, the runtime contract, the planner, and every schema are
unchanged; only the order of the out-of-plan batch and its paired
fixed-layout ordinals move. The head then bounds at own-inputs wire time,
and the simulator's free-head assumption, while still counterfactual,
shrinks from ~275–330 ms of error to tens of milliseconds plus the drain
and staging terms above.
