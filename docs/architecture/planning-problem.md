# The planning problem

A [program](program.md) says what the work is. A **planning problem** asks a
question about it: *run this program, starting here, ending there, on this
machine — what should the schedule be?* The type is
`ShadowSpillPlanningProblem`, and it is what a [search](search.md) is handed.

The distinction is worth holding onto, because the two are otherwise easy to
run together and they have different lifetimes.

## What it adds to a program

| | |
|---|---|
| `program` | The [ShadowSpillProgram](program.md) itself |
| `role` | Which step this is: `initial`, `recurrent` or `forward` |
| `initial_residency` | Where each alias group starts |
| `final_residency` | Where each alias group must end |
| `simulation_config` | The machine: device capacity, spill capacity, transfer bandwidths |
| `admission_facts` | The pool a layout must fit into |
| `source_execution_budget_bytes` | The budget the problem was captured under |
| `maximum_execution_budget_bytes` | The largest budget it may be re-asked at |
| `maximum_spill_budget_bytes` | The same ceiling for spill |
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

The ceilings exist because a budget larger than the machine that was
measured is not a question this problem can answer honestly — the profiles
were taken under a particular configuration, and stretching them past it
would report predictions nothing measured.

## What it still does not carry

**No policy.** Not which search runs, not what that search may try, not how
many workers it may use. A problem says what question it is, never how to
answer it, so an option added to a search later cannot change the meaning of
a problem already on disk. Policy travels beside the problem, as `options`
and `search_options` — see [plan search](search.md).

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

`build_step_program(...)` returns one per role; `.recurrent` is the steady
step and the one a sweep plans. A saved problem round-trips through
`ShadowSpillPlanningProblem.from_value()` and is the unit the corpus stores
— see [program collection](../../benchmarking/program_collection/README.md).

Previous: [The ShadowSpillProgram](program.md). Next: [Plan
search](search.md).
