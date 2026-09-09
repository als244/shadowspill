# Planning orchestration

Planning is a sequence of reusable artifact transformations. The public
orchestrators are intentionally small; each artifact can also be constructed
or consumed independently. These transformations select and physically admit
the [logical program](program.md); they do not recapture or execute the model.

```text
capture/export and stage partitioning
        -> graph-pair construction
        -> structural compilation and profiling
        -> canonical ShadowSpillProgram lowering
        -> StepProgram
        -> ShadowSpillPlanningProblem
        -> resolved programs, one per complete graph-pair selection
        -> the PressureFit search
        -> fixed physical layout and admission
        -> AnnotatedProgramPlan
        -> materialized callable and PlanReport
```

`build_step_program()` stops before PressureFit. `plan_program()` accepts that
saved program with new budgets or transfer bandwidths, so budget sweeps do not
repeat capture, compilation, or profiling. Capturing a program needs the
frontend; planning a saved one does not, so `plan_program()` lives in
`shadowspill.planner` and a sweep never imports torch.

## Two layers

Two functions divide the work, and the boundary between them is exactly the
line between *what is asked* and *how it is answered*.

| Layer | Entry | In | Out |
|---|---|---|---|
| the planner | `plan_program()` | a [`ShadowSpillPlanningProblem`](planning-problem.md), a store, and a `SearchOptions` naming the search | `AnnotatedProgramPlan` |
| the search | any [`SearchAlgorithm`](search.md) | a [program](program.md), boundary residency, machine facts, and the generic options -- its own options it already holds | `ProgramPlanResult` |

**`plan_program()` is the entry point, and the only one.** It resolves the
artifact store, turns budgets and bandwidths into a `SimulationConfig` and
`AdmissionFacts`, consults the planning store, runs the search, holds it to
any plan it was handed, certifies the physical layout, and returns an
`AnnotatedProgramPlan`. It is the only one of the two that touches disk, and
the only one that knows what a budget is.

**The search answers one question and is pluggable.** It receives a program
with its alternatives still open, and everything about how to fix them and
which schedule to choose is its own. [Plan search](search.md) states the
contract in full; [PressureFit](pressurefit.md) is the implementation that
ships, and `search_options.algorithm` is where another goes -- an object, so a
search written outside this package needs no registration.

Expanding a program into resolved programs and comparing across them is the
search's own work rather than a third layer above it, because the order in
which resolutions are tried is part of how a search works and not something
the planner can choose on its behalf.

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
order: one microbatch start to finish before the next. `plan_step()` takes all
four.

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
geometry rather than a geometry alone. Which walk wins depends on the budget:
a breadth-first walk keeps a stage's activations together and lets a tight
budget fetch and evict them in bulk, where the depth-first walk of the same
geometry pays for every microbatch's round trip alone.

## Policy selection

[Graph-pair construction](graph-pair-construction.md) and [graph-pair
selection](graph-pair-selection.md) leave the task alternatives *open* in the
program: they say which implementations exist and which complete assignments
are legal, not which one to use. Fixing them is the search's work. It resolves
the program into the assignments it will compare, chooses the order to try
them in, and ranks what it places -- and the planner sees one call and one
answer, which is why it can be handed a different search without changing.

The two levels stay separate in the diagnostics, so a report says which
assignment won and what it cost to find out.

[PressureFit](pressurefit.md), the search that ships, evaluates residency,
eviction, fetch-trigger and coalescing candidates within each resolved
program, against logical object capacity after provider, fixed-service and
allocator allowances. It requires the compiled planner and simulator and fails
closed on a missing or ABI-incompatible library. Its own page defines its
input/output contract, the problem it solves, its bounded algorithm, and its
repair rules.

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

## The artifact store

`plan_program()` reads and writes one versioned store with two independent
trees. `build/` holds what a run pays for and another run can reuse: exports,
compiler caches, graph pairs, and profiles. `planning/` holds what a run
decided: the canonical program each call was given, the requests put to the
search, the plans it chose, and a readable manifest linking one call to the
artifacts behind it. A planning call writes nothing under `build/` and a build
writes nothing under `planning/`, which is what lets one build store serve many
runs that each keep their own plans. `plan_store` puts the `planning` tree
under a directory of its own for exactly that case.

Each tree carries one policy, a `StoreMode`: `contribute` reads the tree and
writes back what it lacks, `reuse` reads and persists nothing, `require` reads
and refuses a miss, and `refresh` ignores what is there and writes over it.
One mode per tree replaces reading, writing, and overwriting as separate
switches, whose combinations included several that meant nothing.

The [artifact store guide](../python/artifact-store.md) documents the layout
and the per-directory contents.

## Plan report

Planning diagnostics are always present. `PlanReport` maps chronological
execution IDs to semantic tasks, unique stages, structural contracts, selected
graph-pair variants, storage contracts, physical layouts, allocation events,
profile measurements, store hits and misses, transfer calibration, and phase
times. Verbose console output is only presentation; disabling it does not
remove the report.

The [PlanReport interpretation guide](../python/plan-report.md) gives the
inspection order, field tables, task/stage lookup workflow, PressureFit search
hierarchy, and common investigations. The [JSON artifact
guide](../python/planning-json.md) documents the portable program and admitted
plan schemas separately from the callable's in-memory report.

Previous: [Physical admission and offset handling](physical-admission.md). Next:
[Simulation](simulation.md).
