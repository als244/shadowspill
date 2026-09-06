# Planning orchestration

Planning is a sequence of reusable artifact transformations. The public
orchestrators are intentionally small; each artifact can also be constructed
or consumed independently. These transformations select and physically admit
the [logical Program](ir.md); they do not recapture or execute the model.

```text
capture/export and stage partitioning
        -> graph-pair construction
        -> structural compilation and profiling
        -> canonical Program lowering
        -> StepProgram
        -> PressureFitProgram
        -> complete graph-pair selections
        -> pressurefit()
        -> fixed physical layout and admission
        -> AnnotatedProgramPlan
        -> materialized callable and PlanReport
```

`make_step_program()` stops before PressureFit. `pressurefit_program()` accepts
that saved program with new budgets or transfer bandwidths, so budget sweeps do
not repeat capture, compilation, or profiling. Capturing a Program needs the
frontend; planning a saved one does not, so `pressurefit_program()` lives in
`shadowspill.planner` and a sweep never imports torch.

## The walk through the microbatches

An accumulated training step is one forward and one backward per microbatch,
then one optimizer update, and the tasks of every microbatch depend only on
their own data and on the gradients earlier microbatches created. So the
order the step runs its microbatches in is a free choice, and it decides how
much moves over the lanes: a step that runs one microbatch through every
stage before starting the next fetches each stage's parameters once per
microbatch, while a step that runs every microbatch through one stage before
any starts the next fetches them once per pass.

The choice is a `StepDataOrdering`: `depth` passes of `breadth` microbatches
each, with `depth * breadth` the microbatch count. Within a pass the forward
is stage-major and the backward stage-major in reverse. `pair_loss` runs each
microbatch's last stage forward and backward together, because that stage's
saved state -- the logits a loss keeps for its backward -- is the largest per
microbatch, and consuming it as it is produced keeps one copy in flight
rather than a pass's worth. `reverse_breadth` walks a pass's microbatches in
reverse during backward, so the freshest activations go first. Both flags are
on by default and mean nothing at `breadth = 1`, which is the microbatch-major
order every step used before there was a choice; `plan_step()` takes all four.

One consequence reaches the graph pairs. The gradient of a stage is created by
the first backward the walk emits for it and accumulated into by every later
one, so which microbatch runs a stage's creating form and which its
accumulating form follows from the walk (`StepDataOrdering.creates`), not from
the microbatch's position, and capture derives both forms for every
microbatch of an accumulating step; see
[graph-pair construction](graph-pair-construction.md#accumulating-onto-gradients-that-already-exist).

`plan_step_search()` lowers every `depth x breadth` factor pair of a geometry
into its own program over the geometry's one capture and profile set and
plans each under every budget, so the winner at a budget is a walk of a
geometry rather than a geometry alone. On the llama3 step at 6 GiB that
choice took the frontier's winner from 20.6 s to 18.0 s and the 64-microbatch
geometry from 133 s to 22 s; the figures guide's `orderings/` directory shows
the ladder behind each geometry's line.

## Policy selection

[Graph-pair selection](graph-pair-selection.md) constructs the finite set
of legal task-alternative problems. [PressureFit](pressurefit.md) evaluates
residency, eviction, fetch-trigger, and coalescing candidates within each
problem. The two levels remain separate in diagnostics. All the problems go
to PressureFit in one call, in the order they should be searched, so one set
of workers covers them and each prunes against what the others have already
placed.

PressureFit works on logical object capacity after provider/fixed-service and
allocator allowances. It uses the required C planner and simulator; missing or
ABI-incompatible libraries fail closed. Its dedicated page defines the full
input/output contract, mathematical problem, bounded algorithm, repair rules,
and pseudocode.

## Physical admission

The selected logical schedule is not callable until physical admission
assigns its execution-pool ranges, proves every shared-range dependency, and
re-simulates the resulting schedule. Whether it fits is settled during the
search: each candidate measures its own plan's extent against the pool and
gives back what it overran, so the schedule reaching this stage has already
been measured and there is no capacity to lower afterwards.

The complete capacity equations, allocation lifetimes, deterministic placement
algorithm, offset coordinate systems, fixed-core/dynamic-scratch boundary,
runtime sealing, per-candidate capacity refinement, and diagnostics are
documented in
[Physical admission and offset handling](physical-admission.md).

## Transfer bandwidths

Transfer measurement belongs to runtime initialization. Every supported
direction is calibrated independently and then under simultaneous
bidirectional traffic. Planning consumes the conservative per-direction rates
measured during concurrency, plus route latency. `TransferBandwidths` stored in
the program and plan make this input explicit and serializable.

## Plan report

Planning diagnostics are always present. `PlanReport` maps chronological
execution IDs to semantic tasks, unique stages, structural contracts, selected
graph-pair variants, storage contracts, physical layouts, allocation events,
profile measurements, cache artifacts, transfer calibration, and phase times.
Verbose console output is only presentation; disabling it does not remove the
report.

The [PlanReport interpretation guide](../python/plan-report.md) gives the
inspection order, field tables, task/stage lookup workflow, PressureFit search
hierarchy, and common investigations. The [JSON artifact
guide](../python/planning-json.md) documents the portable Program and admitted
plan schemas separately from the callable's in-memory report.

Previous: [Physical admission and offset handling](physical-admission.md). Next:
[Simulation](simulation.md).
