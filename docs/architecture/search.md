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
| `options` | `SearchOptions` — what every search is told |
| `search_options` | This search's own record, opaque to the planner |
| `admission` | Pool facts, when a caller wants the dynamic-pool replay |
| `placement` | The pool topology a layout must fit into |
| `progress` | One line per phase, or `None` |
| `incumbent` | A plan already in hand for this program, offered as a bound |

`options` holds what is true of any search: how many workers it may use,
whether it must reproduce exactly, and the size below which an object is
not worth moving at all. `search_options` holds everything else. The
planner carries it into the plan key and never reads a field of it, which
is what lets a search grow its candidate space without the planner learning
a new word.

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
program, the boundaries, the machine, `options`, which search ran, and what
that search was told. Change any of them and it is a different question
with a different answer. The plan in hand is deliberately **not** in the
key: it is provenance, so a run that replans a budget without the sweep's
plan still reads back the sweep's answer.

**The winner is physically admitted.** What comes back from the planner has
passed [physical admission](physical-admission.md), not only the simulator.

## Naming a search

A search carries a `name` — a stable string it chooses, `"pressurefit"` —
and that name is what the plan key and the plan manifest record. It is
never the Python callable's name, so renaming or wrapping the callable
leaves a stored corpus reachable. Its options travel beside it as a
serialized record, derived from the option type's own fields, so an option
added later is keyed and archived without a second edit.

## Where it plugs in

`plan_program(problem, search_options=...)` is the seam, and
`SearchOptions.algorithm` is where a search goes. It is the search itself, not
a name to look up: an instance holding its own options. `None` runs the one
that ships, so a call that does not care reads as if there were no seam at
all.

Nothing registers a search. A `SearchAlgorithm` subclass defined outside this
package is passed in and used, which is the whole test of whether the seam is
real.

The PyTorch frontend's `plan_step` builds and plans in one call and takes the
same argument, using the shipped search throughout when it is not given
another -- including for the capacity preflight. A search supplied there
applies to every point it plans; one supplied to `plan_program` applies to the
problem the frontend already built.

## Writing one

This page is the contract. [Writing a search
algorithm](search-algorithm.md) is the reference for satisfying it: the
two methods to implement, every argument with its type and meaning, what
the defaults are, and a complete example of a search defined outside this
repository.

Previous: [The planning problem](planning-problem.md). Next: [Writing a
search algorithm](search-algorithm.md).
