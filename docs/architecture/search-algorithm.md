# Writing a search algorithm

The reference for implementing one. [Plan search](search.md) states the
contract — what the planner asks of a search and promises in return; this
page is how you satisfy it, argument by argument, with a working example.

A search is an object. You subclass `SearchAlgorithm`, give it a name,
implement one method, and pass an instance. There is no registration call,
no plugin manifest and no name to reserve, so a search living outside this
repository is a first-class one.

## The two types

| | |
|---|---|
| `SearchAlgorithm` | The search itself, holding the options it was built with. Subclass this. |
| `SearchOptions` | What one planning call is told about searching: `generic`, the `algorithm` instance, and `workers`. |

They compose one way only:

```python
SearchOptions(
    generic=GenericPlanningOptions(...),      # what any search understands
    algorithm=PressureFit(            # the search, holding its own options
        PressureFitOptions(...),
    ),
)
```

`generic` and `algorithm` are separate because the first is a fact about
the request and the second is a fact about the search. A new search adds
nothing to `GenericPlanningOptions` and changes nothing that reads it.

## What you implement

`__call__` is abstract: a class that leaves it out cannot be instantiated. A
concrete class with no `name` is refused at definition.

### `name`

A class attribute. A stable string, chosen by you, that identifies this
search in a plan key and in a plan manifest.

It is **not** the Python class's name. Renaming or moving the class must
not orphan a stored corpus, so the string is written once and left alone.
Defining the class records the string, which is how a plan read back from an
archive is given the search that made it; two searches answering to one name
would make a stored plan ambiguous, so pick something specific.

```python
class Beam(SearchAlgorithm):
    name = "beam"
```

### `__init__`

Yours to define. Take your options record, validate it, and store it as
`self.options`. Refuse the wrong type here rather than later — this is the
earliest point a mistake can be named, and the planner will not check for
you:

```python
def __init__(self, options: BeamOptions | None = None) -> None:
    if options is not None and not isinstance(options, BeamOptions):
        raise TypeError(f"Beam takes BeamOptions, not {type(options).__name__}")
    self.options = options if options is not None else BeamOptions()
```

An instance must hold no per-call state. The same one serves every budget
of a sweep and every worker of a search, concurrently.

### `preflight(...) -> None`

Refuse a machine no schedule can fit, before a search is paid for.

| argument | type | meaning |
|---|---|---|
| `program` | `ShadowSpillProgram` | The work, alternatives still open |
| `initial_residency` | `tuple[ResidencySpec, ...]` | Where each alias group starts |
| `final_residency` | `tuple[ResidencySpec, ...]` | Where each alias group must end |
| `config` | `SimulationConfig` | The machine: capacities and bandwidths |
| `admission` | `AdmissionFacts \| None` | Pool facts, when a dynamic-pool replay is wanted |
| `generic` | `GenericPlanningOptions` | What every search is told |

Returns `None`. Raises `PlanInfeasibleError` when it can tell nothing
fits.

This is a *necessary-condition* check, not a search: what passes is what
you could reach, not a promise that a plan exists. The inherited default
says nothing at all, which is always a correct answer, so a search with no
cheap way to tell simply does not override it.

### `__call__(...) -> ProgramPlanResult`

Answer with a schedule.

| argument | type | meaning |
|---|---|---|
| `program` | `ShadowSpillProgram` | The work, alternatives still open |
| `initial_residency` | `tuple[ResidencySpec, ...]` | Where each alias group starts |
| `final_residency` | `tuple[ResidencySpec, ...]` | Where each alias group must end |
| `config` | `SimulationConfig` | The machine the plan is priced against |
| `generic` | `GenericPlanningOptions` | What every search is told |
| `workers` | `int` | Threads you may use; `0` means every logical CPU, `1` forces serial |
| `admission` | `AdmissionFacts \| None` | Switches on the dynamic-pool replay |
| `placement` | `AdmissionFacts \| None` | The pool a layout must fit; measure against it as you go |
| `progress` | `(str) -> None \| None` | One line per phase, or `None` |
| `incumbent` | `ProgramPlanResult \| None` | A plan already in hand, offered as a bound |

Your own options are not in that list: you were built with them and read
them off `self.options`.

Returns a `ProgramPlanResult`: the schedule, the alternative choices it
fixed, the simulation that priced it, and diagnostics.

Three things are worth stating plainly:

- **The program arrives with alternatives open.** Expanding them into
  resolved programs, and deciding which are worth planning, is yours. The
  planner hands the program over as it stands.
- **The schedule must be one the simulator accepts** on the machine you
  were given. Returning something it rejects is a broken search, not an
  infeasible problem.
- **`incumbent` is a hint, not an obligation.** Use it to stop measuring
  what cannot win, or ignore it. The planner re-measures it on this
  machine afterwards and answers with it if you did worse, so the
  "more memory never plans worse" guarantee holds either way.

Raise `PlanInfeasibleError` when no schedule fits, and
`PlanSearchExhaustedError` when one might exist but you did not reach it.
[Plan search](search.md#what-a-search-answers-with) says why a caller needs
the two kept apart.

## Defaults

`SearchOptions()` with nothing named runs the search that ships, built
with its own defaults. `workers` defaults to zero, one thread per logical
CPU, and is **not** part of the plan key: it says how much machine to spend
rather than what to decide, so two runs at different worker counts ask the
same question and read back the same answer. The plan report records what
was used.

`GenericPlanningOptions` is what you are handed as `generic`:

| field | default | meaning |
|---|---|---|
| `deterministic` | `False` | Make every candidate's outcome a pure function of its inputs |
| `minimum_object_bytes_evict_eligible` | `1 << 20` | Objects below this stay resident from first to last access |

The shipped search's own options are `PressureFitOptions`, from
`shadowspill.planner.search.algorithms.pressurefit`; its defaults are on
[its page](pressurefit.md).

## Your options record

Subclass `OptionRecord` as a frozen dataclass and give it a `KIND`. The
kind is how a nested record says on the wire what it is, so it reads back
as the type it was written as:

```python
@dataclass(frozen=True, slots=True)
class BeamOptions(OptionRecord):
    KIND: ClassVar[str] = "beam"

    width: int = 4

    def __post_init__(self) -> None:
        if self.width < 1:
            raise ValueError("beam width must be at least one")
```

`to_dict` and `from_dict` come from the base and are derived from your
dataclass's own fields, so an option you add later is keyed, archived and
replayed without a second edit. A stored record missing an option is
refused rather than read as though the absent option held the current
default.

## A complete example

Everything above, in the smallest search that actually works — it fixes
every alternative one way, builds a schedule for it, and prices it with the
simulator:

```python
from dataclasses import dataclass
from typing import ClassVar

from shadowspill.planner import (
    GenericPlanningOptions,
    OptionRecord,
    SearchAlgorithm,
    SearchOptions,
    plan_program,
    toolkit,
)


@dataclass(frozen=True, slots=True)
class FirstFitOptions(OptionRecord):
    KIND: ClassVar[str] = "first_fit"

    prefer_recompute: bool = True


class FirstFit(SearchAlgorithm):
    """Take the first resolution that places, and stop looking."""

    name = "first_fit"

    def __init__(self, options: FirstFitOptions | None = None) -> None:
        if options is not None and not isinstance(options, FirstFitOptions):
            raise TypeError(
                f"FirstFit takes FirstFitOptions, not {type(options).__name__}"
            )
        self.options = options if options is not None else FirstFitOptions()

    def __call__(self, program, *, initial_residency, final_residency=(),
                 config, generic, workers=0, admission=None, placement=None,
                 progress=None, incumbent=None):
        # the toolkit does the parts that are not this search's business
        toolkit.validate_search_inputs(
            program, initial_residency, final_residency, config, admission
        )
        resolved = toolkit.resolutions(program, toolkit.DEFAULT_RESOLUTION_OPTIONS)
        ...   # build a schedule, simulate() it, return a ProgramPlanResult


plan = plan_program(
    problem,
    search_options=SearchOptions(
        generic=GenericPlanningOptions(deterministic=True),
        algorithm=FirstFit(FirstFitOptions(prefer_recompute=False)),
        workers=8,
    ),
)
```

`preflight` is left out, so the inherited one runs and says nothing.
Nothing in that file imports from inside
`shadowspill.planner.search.algorithms.pressurefit`, and nothing is
registered. The planner keys the answer under `"first_fit"`, records
`FirstFitOptions` beside it, and holds the result to any incumbent exactly
as it would PressureFit's.

## What the planner does around you

Stated in full in [plan search](search.md), and worth knowing because it is
work you do not have to do: it turns a budget into a machine, keys the
answer so the same question is not asked twice, replays the incumbent to
hold you to it, and physically admits the winner. See
[PressureFit](pressurefit.md) for what one real implementation does inside
the seam.

Previous: [Plan search](search.md). Next: [PressureFit](pressurefit.md).
