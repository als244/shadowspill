# The planning problem

A [program](program.md) says what the work is. A **planning problem** asks a
question about it: *run this program, starting here, ending there, on this
machine — what should the schedule be?* The type is
`ShadowSpillPlanningProblem`, and it is what `plan_program()` is handed; a
[search](search.md) receives its parts.

The distinction is worth holding onto, because the two are otherwise easy to
run together and they have different lifetimes.

## What it adds to a program

| | |
|---|---|
| `program` | The [ShadowSpillProgram](program.md) itself |
| `role` | What it plans: a training `step` or a `forward` |
| `initial_residency` | Where each alias group starts |
| `final_residency` | Where each alias group must end |
| `simulation_config` | The machine: device capacity, spill capacity, transfer bandwidths |
| `admission_facts` | The pool a layout must fit into |
| `source_execution_budget_bytes` | The budget the problem was captured under |
| `maximum_execution_budget_bytes` | The largest budget it may be re-asked at |
| `maximum_spill_budget_bytes` | The spill pool it was profiled against, as a record rather than a ceiling |
| `fixed_execution_bytes` | Device bytes the pool never sees |
| `object_reserve_bytes` | Bytes held back for planned objects |
| `dynamic_scratch_reserve_bytes` | Bytes held back for allocations the plan does not own |

## Why the boundaries belong here and not in the program

Where a step starts and ends is not a property of the work; it is a property
of how the work is being used. The same program planned to end with its
parameters on the device is a different question from the same program
planned to end with them spilled, and both are legitimate. Putting the
boundaries in the problem is what lets one program be asked both.

It is also what makes a step composable with the step before it. See [step
boundaries](step-boundaries.md).

## Why the budget is here and re-askable

`source_execution_budget_bytes` is what the problem was captured under, and
the two maxima bound what it may be re-asked at. Overriding the budget
produces a different machine and therefore a different question — but from
the *same* problem, with no capture, compilation or profiling repeated. That
is the whole reason a frontier sweep is cheap: it is one problem asked many
times.

The execution ceiling exists because a budget larger than the device that
was measured is not a question this problem can answer honestly — the
profiles were taken in a pool of that size, the layout is built against it,
and stretching them past it would report predictions nothing measured.

**Spill has no such ceiling.** Nothing measured depends on how large the
spill pool was: it holds what was evicted, and a task's profile, a
transfer's calibration and the layout are unmoved by its size. The spill
budget reaches exactly one place, the simulator's spill capacity, so a
problem profiled against a small spill pool answers honestly about a large
one — which is what a sweep over spill budgets, or over machines that do not
exist yet, is asking. `maximum_spill_budget_bytes` therefore records the pool
the problem was profiled against and bounds nothing.

Whether a plan can be *run* is a different question, and it is asked where
it can be answered: a plan made through a runtime is checked against the
pool that runtime actually has.

## What it still does not carry

**No policy.** Not which search runs, not what that search may try, not how
many workers it may use. A problem says what question it is, never how to
answer it, so an option added to a search later cannot change the meaning of
a problem already on disk. Policy travels beside the problem, as a
`SearchOptions` — see [plan search](search.md).

**No answer.** The plan, the schedule and the simulation come back from a
search; none of them live here.

## Lifetimes

A program outlives a problem: the same lowered program is asked about at
every budget of a sweep. A problem outlives a plan: the same problem is
re-asked when a search improves, when a different search is tried, or when a
plan store is rebuilt. This is why they are separate types with separate
digests, and why the plan key covers the problem, the machine, the search
and its options rather than the program alone.

## Where it comes from

`build_step_programs(...)` returns one `StepProgram` per ordering, and each
holds its step's problem as `.problem`, the one a sweep plans. A saved problem round-trips through
`ShadowSpillPlanningProblem.from_value()` and is the unit the corpus stores
— see [program collection](../../benchmarking/program_collection/README.md).

Previous: [The ShadowSpillProgram](program.md). Next:
[Simulation](simulation.md).
