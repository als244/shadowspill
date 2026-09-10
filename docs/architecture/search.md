# Plan search

Choosing a schedule is one decision, and it is pluggable. This page states
what the planner asks of a search, what it promises in return, and what a
search is free to decide. [PressureFit](pressurefit.md) is the search that
ships and the only one today; nothing below is written for it in particular.

## Why the seam is here

Two things happen around every search and have nothing to do with how a
schedule is chosen. A budget has to become a machine — capacities and
bandwidths the simulator can price. An answer has to be keyed, so the same
question is not asked twice and a sweep can reuse what an earlier point
found. Both are identical whatever the search.

What is left is one question: *given this program on this machine, with
these boundaries, which schedule should run?* That is the search's, and the
planner is deliberately ignorant of how it is answered. A second search is
therefore a second implementation of `SearchAlgorithm` — not a second planner,
not a flag inside PressureFit.

## What a search is given

| | |
|---|---|
| `program` | The [ShadowSpillProgram](program.md), alternatives still open |
| `initial_residency` | Where each alias group starts |
| `final_residency` | Where each alias group must end |
| `config` | The machine: device capacities, spill capacity, transfer bandwidths |
| `generic` | `GenericPlanningOptions` — what every search is told |
| `workers` | How many threads it may use: zero for every logical CPU, one for serial |
| `admission` | Pool facts, when a caller wants the dynamic-pool replay |
| `placement` | The pool topology a layout must fit into |
| `progress` | One line per phase, or `None` |
| `incumbent` | A plan already in hand for this program, offered as a bound |

`generic` holds what is true of any search: whether it must reproduce
exactly, and the size below which an object is not worth moving at all. A
search's own options are not passed in at all — it was built with them and
holds them as its `options`. The planner carries that record into the plan
key and never reads a field of it, which is what lets a search grow its
candidate space without the planner learning a new word.

`workers` is an argument rather than an option because it changes how long
an answer takes, not which answer is right, so it is no part of the question
the plan is keyed by.

The program arrives with its task alternatives **open**. Expanding them
into resolved programs, deciding which are worth planning and comparing
what each answers is the search's own business — see
[task alternatives](program.md#task-alternatives). The planner hands over
the program as it stands and reads only the one result that comes back.

## What a search answers with

A `ProgramPlanResult`: the schedule, the alternative choices it fixed, the
simulation that priced it, and diagnostics. The schedule must be one the
simulator accepts on the machine it was given — a search that returns
something the simulator rejects is a broken search, not an infeasible
problem.

Two refusals are distinct and both are load-bearing:

- `PlanInfeasibleError` — no schedule fits this machine. The answer does
  not exist, so no other search would find one either.
- `PlanSearchExhaustedError` — one may exist, but this search did not
  reach it. A different search, or the same one told to try harder, might.

Reporting the second as the first tells a caller to buy memory it does not
need; reporting the first as the second sends it looking for a plan that
cannot exist.

## What a search may decide

Everything about *how*: which candidates to build, in what order, how long
to spend, when to stop, what to do with the plan it was handed, and how to
expand alternatives. None of that reaches the planner.

## What a search may not decide

- **The program.** Tasks, objects, profiles and alternatives are given.
- **The boundaries.** Where the program starts and ends is the caller's.
- **The machine.** Capacity is not a knob a search may turn.
- **What the answer costs.** The simulator prices a schedule; a search
  reports what it measured, never what it hoped.

## What the planner promises

**More memory never plans worse.** A plan that fits in less memory fits in
more, so a sweep that plans budgets ascending hands each point the best
plan below it. The guarantee is structural rather than a promise each
search is trusted to keep: the planner replays the incumbent's schedule on
this machine after the search returns, and answers with it when it is
strictly faster and its layout still fits. That is one simulation against a
whole search. A search that used the hint well returns the incumbent
itself, and the check agrees with it.

**An answer is keyed by the whole question.** The plan key covers the
program, the boundaries, the machine, the pool a layout must fit, the
generic options, the name of the search that ran, and what that search was
told. Change any of them and it is a different question with a different
answer. Two things are deliberately outside the key: `workers`, and the
plan in hand, which is provenance — so a run that replans a budget without
the sweep's plan still reads back the sweep's answer.

The key records a search's `name`, a stable string the search chooses, and
never the Python class's name, so renaming or moving the class leaves a
stored corpus reachable.

**The winner is physically admitted.** What comes back from the planner has
passed [physical admission](physical-admission.md), not only the simulator.

## Where it plugs in

`plan_program(problem, search_options=...)` is the seam, and
`SearchOptions.algorithm` is where a search goes. It is the search itself, not
a name to look up: an instance holding its own options. `None` runs the one
that ships, so a call that does not care reads as if there were no seam at
all.

Nothing has to be registered by hand, and no call path resolves a name. A
`SearchAlgorithm` subclass defined outside this package is passed in and
used, which is the whole test of whether the seam is real. Defining it does
record its `name`, which is how a plan read back from an archive can be
given the search that made it.

The PyTorch frontend's `plan_step` builds and plans in one call and takes the
same argument, using the shipped search throughout when it is not given
another -- including for the capacity preflight. A search supplied there
applies to every point it plans; one supplied to `plan_program` applies to the
problem the frontend already built.

## Writing one

This page is the contract. [Writing a search
algorithm](search-algorithm.md) is the reference for satisfying it: the
method to implement, every argument with its type and meaning, what the
defaults are, and a complete example of a search defined outside this
repository.

Previous: [The planning problem](planning-problem.md). Next: [Writing a
search algorithm](search-algorithm.md).
