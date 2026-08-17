# Planning and physical admission

Planning is a sequence of reusable artifact transformations. The public
orchestrators are intentionally small; each artifact can also be constructed
or consumed independently. These transformations select and physically admit
the [logical Program](ir.md); they do not recapture or execute the model.

```text
capture/export and graph lowering
        -> structural compilation and profiling
        -> canonical StepProgram
        -> PressureFitProgram
        -> pressurefit()
        -> fixed physical layout and admission
        -> AnnotatedProgramPlan
        -> materialized callable and PlanReport
```

`make_step_program()` stops before PressureFit. `pressurefit_program()` accepts
that saved program with new budgets or transfer bandwidths, so budget sweeps do
not repeat capture, compilation, or profiling.

## Recomputation portfolio

For a small product of graph-pair choices, PressureFit evaluates every
selection. For large binary save/recompute programs, the bounded
portfolio tests evenly distributed 0%, 25%, 50%, 75%, and 100% recomputation
across flexible groups. Forward-DAG sink groups are pinned to a save option so
terminal heads are not needlessly recomputed. Non-binary groups retain a
deterministic within-group memory-quantile fallback.

Each recomputation selection becomes a parent context whose child
candidates vary residency, eviction, fetch-trigger, and coalescing policies.
Diagnostics retain both levels separately.

## PressureFit

PressureFit works on logical object capacity after provider/fixed-service and
allocator allowances. It evaluates candidate schedules through the required C
planner and simulator. Missing or ABI-incompatible compiled libraries fail;
there is no silent Python fallback. Python simulation remains an explicit
timeline/debugging oracle.

Candidate diagnostics include:

- recomputation context and choice counts;
- candidate policy identity;
- simulation calls and pressure-boundary repairs;
- feasibility or rejection reason;
- predicted makespan, transfer bytes, stalls, and peak memory;
- aggregate and per-context planning work.

## Physical admission

The selected logical schedule is not callable until a complete physical
layout passes. The layout builder combines:

- initial object placement;
- each selected task's strict allocation ABI core;
- output promotion and mutation replacement;
- task-completion retirements;
- triggered transfer reservations;
- causal reuse dependencies;
- caller handoff and terminal residency.

It assigns deterministic offsets, emits reuse dependencies, and re-simulates
the physical schedule. If the layout does not fit, planning reduces logical
object capacity in monotonic increments, reruns PressureFit, and retries
admission. The frontend refinement uses 256 MiB increments through the
first GiB and 512 MiB thereafter. The final report records every attempt and
why it failed or succeeded.

The admitted layout is a certificate for the profiled fixed-shape allocation
contract. Runtime divergence fails before unsafe backend use. Optional scratch
paths remain bounded rather than pretending that all opaque provider behavior
has a compiler proof.

## Transfer bandwidths

Transfer measurement belongs to runtime initialization. Every supported
direction is calibrated independently and then under simultaneous
bidirectional traffic. Planning consumes the conservative per-direction rates
measured during concurrency, plus route latency. `TransferBandwidths` stored in
the program and plan make this input explicit and serializable.

## Plan report

Planning diagnostics are always present. `PlanReport` maps chronological
execution IDs to semantic tasks, unique stages, structural ABIs, selected
graph-pair variants, storage contracts, physical layouts, allocation events,
profile measurements, cache artifacts, transfer calibration, and phase times.
Verbose console output is only presentation; disabling it does not remove the
report.

Previous: [PyTorch capture and lowering](lowering.md). Next:
[Simulation](simulation.md).
