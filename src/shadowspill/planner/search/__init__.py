"""The search protocol: what the planner asks of a search, and promises it.

A search answers one question -- *given this program on this machine, with
these boundaries, what schedule should run?* -- and the planner is
deliberately ignorant of how. PressureFit is the search that ships; the
planner reaches it through this protocol and nothing else, so a second
search is a second implementation of :class:`SearchAlgorithm` rather than a
second planner.

What the planner does around a search is described in
``docs/architecture/search.md``: it fixes the machine from a budget, keys
the answer in the plan store, admits the winner physically, and holds the
search to the plan it was handed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

from shadowspill.ir import ResidencySpec, ShadowSpillProgram
from shadowspill.simulator import (
    SimulationConfig,
    SimulationInfeasibleError,
    simulate,
)

from ..admission import AdmissionFacts
from ..request import GenericPlanningOptions, OptionRecord
from ..result import ProgramPlanResult


class SearchAlgorithm(ABC):
    """One way of choosing a schedule, and the options it was built with.

    A search *is* an instance of this: subclass it, give it a `name`, and
    implement `__call__`. A planning call is handed the instance, so a caller
    extending ShadowSpill with a search of their own looks no name up on the
    calling path.

    `name` identifies the search in a plan key and in a plan manifest. It
    is a stable string chosen by the implementation -- `"pressurefit"` --
    and never the Python class's name, so renaming or moving the class
    leaves a stored corpus reachable. It is also what `named` resolves, which
    is how a plan read back from an archive reaches the search that made it.

    `options` is the search's own record, carried into the plan key whole
    and read by nothing outside the search. An instance holds no per-call
    state, so the same one serves every budget of a sweep and every worker
    of a search.
    """

    #: The stable name this search is keyed by.
    name: ClassVar[str] = ""

    #: This search's own options.
    options: OptionRecord

    #: Every named search, by name. Defining a subclass fills this in, so a
    #: plan read back from an archive can be given the search that made it
    #: without the reader knowing which searches exist.
    _NAMED: ClassVar[dict[str, type[SearchAlgorithm]]] = {}

    def __init_subclass__(cls, **keywords: object) -> None:
        super().__init_subclass__(**keywords)
        if getattr(cls, "__abstractmethods__", None):
            return
        if not cls.name:
            raise TypeError(f"{cls.__name__} must name itself")
        SearchAlgorithm._NAMED[cls.name] = cls

    def __init__(self, options: OptionRecord | None = None) -> None:
        """Build the search with its own options; `None` means its defaults.

        Every search is constructed this way, which is what lets a plan read
        back from an archive be given the search that made it.
        """

        if options is not None:
            self.options = options

    @staticmethod
    def named(name: str) -> type[SearchAlgorithm]:
        """The search class registered under `name`.

        Defining the class is what registers it, so a search implemented
        outside this package is reachable here once its module is imported.
        """

        if name not in SearchAlgorithm._NAMED:
            known = ", ".join(sorted(SearchAlgorithm._NAMED)) or "none"
            raise KeyError(
                f"no search named {name!r}; import the module that defines "
                f"it. Known searches: {known}"
            )
        return SearchAlgorithm._NAMED[name]

    def preflight(  # noqa: B027 -- a documented default, not an omission
        self,
        program: ShadowSpillProgram,
        *,
        initial_residency: tuple[ResidencySpec, ...],
        final_residency: tuple[ResidencySpec, ...] = (),
        config: SimulationConfig,
        admission: AdmissionFacts | None = None,
        generic: GenericPlanningOptions,
    ) -> None:
        """Refuse a machine no schedule can fit, before a search is paid for.

        A necessary-condition check, not a search: what passes here is what
        this search could reach, so a search that cannot cheaply tell may
        do nothing at all. Raises
        :exc:`~shadowspill.errors.PlanInfeasibleError` when it can tell that
        nothing fits.

        The default says nothing, which is always a correct answer: this
        check exists to save a search that would have failed anyway, and a
        search with no cheap way to tell should not pretend otherwise.
        """

    @abstractmethod
    def __call__(
        self,
        program: ShadowSpillProgram,
        *,
        initial_residency: tuple[ResidencySpec, ...],
        final_residency: tuple[ResidencySpec, ...] = (),
        config: SimulationConfig,
        generic: GenericPlanningOptions,
        workers: int = 0,
        admission: AdmissionFacts | None = None,
        placement: AdmissionFacts | None = None,
        progress: Callable[[str], None] | None = None,
        incumbent: ProgramPlanResult | None = None,
    ) -> ProgramPlanResult:
        """Answer with a schedule for ``program``.

        ``program`` may carry task alternatives. Expanding them into
        resolved programs, deciding which are worth planning and comparing
        what each answers is the search's own business: the planner hands
        over the program as it stands and reads only the one result that
        comes back.

        ``initial_residency`` and ``final_residency`` are where each alias
        group must start and end. ``config`` is the machine: capacities and
        transfer bandwidths, already fixed from whatever budget the caller
        asked for. ``generic`` is what every search is told; this search's
        own options are its `options`.

        ``workers`` is how many threads it may use: zero for every
        available logical CPU, one to force serial evaluation. It is an
        argument rather than an option because it changes how long an
        answer takes, not which answer is right, so it is no part of the
        question the plan is keyed by.

        ``admission`` switches on the dynamic-pool replay when a caller
        wants one; ``placement`` is the pool topology a plan's layout must
        fit into, which a search measures against as it goes.

        ``progress`` receives one line per phase when a caller wants to
        watch a long search, and is ``None`` otherwise.

        ``incumbent`` is a plan already in hand for this same program,
        offered as a bound: a search may use it to stop measuring what
        cannot win, and may answer with it. It is a hint and not an
        obligation -- the planner re-measures the incumbent on this machine
        and returns it if this search did worse, so the guarantee that more
        memory never plans worse holds whatever a search does with it.

        Raises :exc:`~shadowspill.errors.PlanInfeasibleError` when no
        schedule fits the machine, and
        :exc:`~shadowspill.errors.PlanSearchExhaustedError` when one might
        exist but this search did not reach it.
        """


@dataclass(frozen=True, slots=True)
class SearchOptions:
    """The whole of what a planning call is told about searching.

    Two halves, so neither can be set without the other being visible:
    `generic` is what any search understands, and `algorithm` is the
    search itself, carrying its own options. The plan key covers both.

    `workers` sits beside them but is **not** part of the key: it says how
    much of the machine to spend, which changes how long an answer takes
    rather than which answer is right. Zero means every available logical
    CPU; one forces serial evaluation. The plan report records what was
    used, so a run is still reproducible from its own record.

    `algorithm` of ``None`` means the search that ships.
    """

    generic: GenericPlanningOptions = field(default_factory=GenericPlanningOptions)
    algorithm: SearchAlgorithm | None = None
    workers: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.generic, GenericPlanningOptions):
            raise TypeError("generic must be a GenericPlanningOptions")
        if self.algorithm is not None and not isinstance(
            self.algorithm, SearchAlgorithm
        ):
            raise TypeError("algorithm must be a SearchAlgorithm")
        if (
            isinstance(self.workers, bool)
            or not isinstance(self.workers, int)
            or self.workers < 0
        ):
            raise ValueError("workers must be a non-negative integer")

    def to_dict(self) -> dict[str, object]:
        """The question, as the plan key and the archived request see it.

        `workers` is deliberately absent: two runs at different worker
        counts ask the same question and must read back the same answer.
        """

        algorithm = self.resolved_algorithm
        return {
            "generic": self.generic.to_dict(),
            "algorithm": {
                "name": algorithm.name,
                "options": algorithm.options.to_dict(),
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], path: str) -> SearchOptions:
        """Read back what :meth:`to_dict` wrote.

        The search is rebuilt from its name and handed its own options, so
        a plan read from an archive says which search made it and what it
        was told. `workers` is not stored and comes back as zero.
        """

        generic = OptionRecord.record_from_value(value.get("generic"))
        if not isinstance(generic, GenericPlanningOptions):
            raise ValueError(f"{path}.generic is not the generic options")
        algorithm = value.get("algorithm")
        if not isinstance(algorithm, Mapping):
            raise ValueError(f"{path}.algorithm is not a mapping")
        name = algorithm.get("name")
        if not isinstance(name, str):
            raise ValueError(f"{path}.algorithm.name is not a string")
        options = OptionRecord.record_from_value(algorithm.get("options"))
        return cls(
            generic=generic,
            algorithm=SearchAlgorithm.named(name)(options),
        )

    @property
    def resolved_algorithm(self) -> SearchAlgorithm:
        """The search that will run: the one named, or the one that ships."""

        return self.algorithm if self.algorithm is not None else default_algorithm()


#: Set by the search that ships, when its module is imported.
_DEFAULT: list[SearchAlgorithm] = []


def set_default_algorithm(algorithm: SearchAlgorithm) -> None:
    """Name the search a call gets when it asks for none."""

    _DEFAULT[:] = [algorithm]


def default_algorithm() -> SearchAlgorithm:
    """The search that ships, as an instance built with its own defaults."""

    if not _DEFAULT:
        raise RuntimeError("no default search algorithm has been installed")
    return _DEFAULT[0]


def answer_no_worse_than(
    result: ProgramPlanResult,
    *,
    incumbent: ProgramPlanResult | None,
    config: SimulationConfig,
    placement: AdmissionFacts | None,
) -> ProgramPlanResult:
    """Hold a search to the plan it was handed.

    The incumbent was found on another machine -- at a smaller capacity,
    usually -- so its own makespan says nothing here. This replays its
    schedule on this one and answers with it when it is strictly faster
    than what the search returned and its layout still fits. That is one
    simulation against a whole search, and it is what makes "more memory
    never plans worse" a property of the planner rather than a promise each
    search is trusted to keep.

    A search that used the hint well returns the incumbent itself, and this
    agrees with it. A search that ignored the hint is corrected here.
    """

    if incumbent is None:
        return result
    try:
        measured = simulate(
            incumbent.program,
            incumbent.schedule,
            selections=incumbent.selections,
            config=config,
        )
    except SimulationInfeasibleError:
        # The plan in hand does not run on this machine at all, so there is
        # nothing to hold the search to.
        return result
    if measured.makespan_ns >= result.simulation.makespan_ns:
        return result
    carried = replace(
        incumbent,
        simulation=measured,
        simulation_config=config,
        admission_facts=result.admission_facts,
        placement_facts=result.placement_facts,
        diagnostics=result.diagnostics,
        # The answer is this call's, whichever plan it settles on, so it
        # names the search that ran and what it was told. Carrying the
        # incumbent's instead would file the plan under one search and key
        # it under another.
        search_options=result.search_options,
    )
    if placement is not None and not _places(carried, placement):
        # Faster, but its layout does not fit the pool. A plan that cannot
        # be admitted is not an answer.
        return result
    return carried


def _places(result: ProgramPlanResult, placement: AdmissionFacts) -> bool:
    """Whether this schedule's layout still fits the pool it must live in."""

    from ..admission.layout import measure_fixed_layout

    try:
        return measure_fixed_layout(result, placement).fits
    except ValueError:
        return False


__all__ = [
    "SearchAlgorithm",
    "SearchOptions",
    "answer_no_worse_than",
    "default_algorithm",
    "set_default_algorithm",
]
