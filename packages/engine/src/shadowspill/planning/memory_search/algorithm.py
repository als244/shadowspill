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
from shadowspill.planning.memory_search.admission import AdmissionFacts
from shadowspill.planning.memory_search.request import (
    GenericPlanningOptions,
    OptionRecord,
)
from shadowspill.planning.memory_search.result import ProgramPlanResult
from shadowspill.simulator import (
    SimulationConfig,
    SimulationInfeasibleError,
    simulate,
)


class SearchAlgorithm(ABC):
    """The base class of a search: one way of choosing a schedule.

    Subclass it, give the subclass a ``name``, implement ``__call__``, which
    answers with a `ProgramPlanResult`, and pass an instance as
    `SearchOptions.algorithm`; nothing is looked up by name on the calling
    path. An instance holds no per-call state, so one serves every budget of
    a sweep and every worker of a search.

    Parameters
    ----------
    options : OptionRecord, optional
        The search's own options, carried whole into the plan key and read
        by nothing outside the search. None keeps the options the subclass
        defines.

    Attributes
    ----------
    name : str
        The stable name the search is keyed and registered by, such as
        ``"pressurefit"``. Set by the subclass, independent of its class name:
        a stored plan names its search by it, and `SearchOptions.from_dict`
        finds the class through `SearchAlgorithm.named`. Defining a concrete
        subclass without one raises `TypeError`.
    """

    name: ClassVar[str] = ""

    options: OptionRecord

    #: Every named search, by name. Defining a subclass fills this in, so a
    #: plan read back from an archive can be given the search that made it
    #: without the reader knowing which searches exist.
    _NAMED: ClassVar[dict[str, type[SearchAlgorithm]]] = {}

    def __init_subclass__(cls, **keywords: object) -> None:
        super().__init_subclass__(**keywords)
        # ABCMeta sets __abstractmethods__ only after this hook, so find here,
        # as it will, whether an abstract method is left.
        inherited = (getattr(b, "__abstractmethods__", ()) for b in cls.__bases__)
        names = {n for group in inherited for n in group}
        if any(
            getattr(getattr(cls, n, None), "__isabstractmethod__", False)
            for n in names | set(vars(cls))
        ):
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
        """Find the search class registered under ``name``.

        Defining a subclass registers it, so a search defined outside
        ShadowSpill is found once its module is imported. ``"pressurefit"``
        is always found.

        Parameters
        ----------
        name : str
            The search's ``name``.

        Returns
        -------
        type
            The `SearchAlgorithm` subclass.

        Raises
        ------
        KeyError
            If no search is registered under ``name``. The message names those
            that are.
        """

        if name == "pressurefit" and name not in SearchAlgorithm._NAMED:
            from shadowspill.planning.memory_search.algorithms.pressurefit import (
                PressureFit,
            )

            return PressureFit
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

        A necessary condition, not a search: passing it promises no plan.
        The base implementation does nothing, which is always correct; a
        search overrides it when it has a cheap way to tell.
        `validate_schedule_feasibility` calls it.

        Parameters
        ----------
        program : ShadowSpillProgram
            The program, its task alternatives still open.
        initial_residency : tuple of ResidencySpec
            Where each alias group is before the first task.
        final_residency : tuple of ResidencySpec, default ()
            Where each alias group must be after the last task.
        config : SimulationConfig
            The machine: capacities and copy rates.
        admission : AdmissionFacts, optional
            The execution pool's physical facts, when the caller has them.
        generic : GenericPlanningOptions
            The options every search is told.

        Raises
        ------
        PlanInfeasibleError
            If the search can tell that no schedule fits.
        """

    def record_trace(
        self,
        result: ProgramPlanResult,
        *,
        options: SearchOptions,
        admission: AdmissionFacts | None = None,
        placement: AdmissionFacts | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        """Record a requested trace of the search behind a plan already found.

        The planner calls it after planning when ``options.trace_directory``
        is set, including when the plan store held the answer. It leaves the
        answer unchanged. The base implementation refuses.

        Parameters
        ----------
        result : ProgramPlanResult
            The answer to trace.
        options : SearchOptions
            The caller's search options, which say where to record and which
            candidates.
        admission : AdmissionFacts, optional
            The physical facts the answer was planned with, if any.
        placement : AdmissionFacts, optional
            The pool the answer's layout was measured against, if any.
        progress : callable, optional
            Receives one line per milestone.

        Raises
        ------
        NotImplementedError
            If the search does not support tracing.
        """
        raise NotImplementedError(f"{self.name} does not support search tracing")

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
        keep_resolutions: bool = False,
    ) -> ProgramPlanResult:
        """Answer with a schedule for ``program`` on this machine.

        The one method a subclass must write. The program arrives with its
        task alternatives open: expanding them into resolved programs,
        choosing which to plan and comparing their plans is the search's own
        work. The schedule must be one `simulate` accepts on ``config``.

        Parameters
        ----------
        program : ShadowSpillProgram
            The program, its task alternatives still open.
        initial_residency : tuple of ResidencySpec
            Where each alias group is before the first task.
        final_residency : tuple of ResidencySpec, default ()
            Where each alias group must be after the last task.
        config : SimulationConfig
            The machine, its capacities already fixed from the caller's
            budget.
        generic : GenericPlanningOptions
            The options every search is told; the search's own are its
            ``options``.
        workers : int, default 0
            Threads the search may use: zero for every logical CPU, one for
            serial evaluation.
        admission : AdmissionFacts, optional
            Turns on a replay of each schedule through the dynamic execution
            pool.
        placement : AdmissionFacts, optional
            The pool a plan's fixed layout must fit, for the search to
            measure against as it goes.
        progress : callable, optional
            Receives one line per phase of the search.
        incumbent : ProgramPlanResult, optional
            A plan for this program found under another budget, offered as a
            bound. A search may use it to stop measuring what cannot win and
            may answer with it; the planner replays it afterwards and answers
            with it when the search did worse.
        keep_resolutions : bool, default False
            Report every resolved program's best plan, decoded and simulated
            like the answer, as `ProgramPlanResult.resolutions`. A search
            that plans one resolution may leave them empty.

        Returns
        -------
        ProgramPlanResult
            The schedule, the alternatives it fixes, its simulation and the
            search's diagnostics.

        Raises
        ------
        PlanInfeasibleError
            If no schedule fits the machine.
        PlanSearchExhaustedError
            If a schedule may exist, but the search did not reach one.
        """


@dataclass(frozen=True, slots=True)
class SearchOptions:
    """Everything a planning call is told about searching.

    ``generic`` and ``algorithm`` are part of the plan key. ``workers`` and
    the trace fields are not: they change how an answer is found or
    recorded, not which answer is right.

    Parameters
    ----------
    generic : GenericPlanningOptions, default GenericPlanningOptions()
        The options every search is told.
    algorithm : SearchAlgorithm, optional
        The search to run, holding its own options. None runs the search
        that ships, PressureFit, with its defaults.
    workers : int, default 0
        Threads the search may use: zero for every logical CPU, one for
        serial evaluation. At least zero. PressureFit records it in
        `PlanningDiagnostics.workers`; the planner does not.
    trace_directory : str, optional
        Directory the search records a trace into after planning, through
        `SearchAlgorithm.record_trace`. The plan is unchanged.
    trace_candidates : tuple of (str, str), optional
        The ``(selection_id, candidate_id)`` pairs to trace, each once. None
        traces the pair that answered. Requires ``trace_directory``.

    Raises
    ------
    ValueError
        If ``trace_directory`` is empty or holds a null character,
        ``trace_candidates`` is malformed or given without it, or ``workers``
        is negative or not an integer.
    TypeError
        If ``generic`` or ``algorithm`` has the wrong type.
    """

    generic: GenericPlanningOptions = field(default_factory=GenericPlanningOptions)
    algorithm: SearchAlgorithm | None = None
    workers: int = 0
    #: Optional diagnostic output; excluded from the request and plan key.
    trace_directory: str | None = None
    #: (selection_id, candidate_id) pairs; None records the answering pair.
    trace_candidates: tuple[tuple[str, str], ...] | None = None

    def __post_init__(self) -> None:
        if self.trace_directory is not None and (
            not isinstance(self.trace_directory, str)
            or not self.trace_directory
            or "\x00" in self.trace_directory
        ):
            raise ValueError("trace_directory must be a non-empty path or None")
        if self.trace_candidates is not None:
            if self.trace_directory is None:
                raise ValueError("trace_candidates requires trace_directory")
            if (
                not isinstance(self.trace_candidates, tuple)
                or not self.trace_candidates
            ):
                raise ValueError("trace_candidates must be a non-empty tuple of pairs")
            for pair in self.trace_candidates:
                if (
                    not isinstance(pair, tuple)
                    or len(pair) != 2
                    or any(not isinstance(value, str) or not value for value in pair)
                ):
                    raise ValueError(
                        "trace_candidates entries must be identifier pairs"
                    )
            if len(set(self.trace_candidates)) != len(self.trace_candidates):
                raise ValueError("trace_candidates must not contain duplicate pairs")
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
        """Write the keyed part of the options, as the plan key records it.

        Returns
        -------
        dict
            ``generic`` as its record, and ``algorithm`` as the search's
            ``name`` and ``options``; the search that ships when
            ``algorithm`` is None.
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
        """Read options back from what `SearchOptions.to_dict` wrote.

        The search is found by its name, through `SearchAlgorithm.named`,
        and built with its own options. ``workers`` comes back as zero and
        the trace fields as None.

        Parameters
        ----------
        value : mapping of str to object
            The written options.
        path : str
            Where the value sits in its record, for error messages.

        Returns
        -------
        SearchOptions
            The options.

        Raises
        ------
        ValueError
            If a part is missing or malformed.
        KeyError
            If no search is registered under the written name.
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
        algorithm_type = SearchAlgorithm.named(name)
        options = OptionRecord.record_from_value(algorithm.get("options"))
        return cls(
            generic=generic,
            algorithm=algorithm_type(options),
        )

    @property
    def resolved_algorithm(self) -> SearchAlgorithm:
        """The search that will run: ``algorithm``, or the one that ships."""

        return self.algorithm if self.algorithm is not None else default_algorithm()


#: Set by the search that ships, when its module is imported.
_DEFAULT: list[SearchAlgorithm] = []


def set_default_algorithm(algorithm: SearchAlgorithm) -> None:
    """Name the search a call gets when it asks for none."""

    _DEFAULT[:] = [algorithm]


def default_algorithm() -> SearchAlgorithm:
    """The search that ships, as an instance built with its own defaults."""

    if not _DEFAULT:
        from shadowspill.planning.memory_search.algorithms.pressurefit import (
            pressurefit,
        )

        set_default_algorithm(pressurefit)
    return _DEFAULT[0]


def answer_no_worse_than(
    result: ProgramPlanResult,
    *,
    incumbent: ProgramPlanResult | None,
    config: SimulationConfig,
    placement: AdmissionFacts | None,
) -> ProgramPlanResult:
    """Answer with the plan a search was handed when it is faster here.

    The planner calls it after every search, which makes "more memory never
    plans worse" its own guarantee rather than each search's. The
    incumbent's schedule is simulated on ``config``, and it replaces
    ``result`` when it runs there, its makespan is strictly smaller and,
    given ``placement``, its fixed layout still fits the pool. It then
    carries the new simulation, and ``result``'s diagnostics, search
    options, admission facts and resolutions.

    Parameters
    ----------
    result : ProgramPlanResult
        What the search answered.
    incumbent : ProgramPlanResult or None
        The plan the search was handed, found for the same program under
        another budget. None returns ``result``.
    config : SimulationConfig
        The machine the search planned for.
    placement : AdmissionFacts or None
        The pool the incumbent's fixed layout must fit; None skips that
        check.

    Returns
    -------
    ProgramPlanResult
        ``result``, or the incumbent carried over to this machine.
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
        # The other resolutions are this search's, not the plan in hand's.
        resolutions=result.resolutions,
    )
    if placement is not None and not _places(carried, placement):
        # Faster, but its layout does not fit the pool. A plan that cannot
        # be admitted is not an answer.
        return result
    return carried


def _places(result: ProgramPlanResult, placement: AdmissionFacts) -> bool:
    """Whether this schedule's layout still fits the pool it must live in."""

    from shadowspill.planning.memory_search.admission.layout import measure_fixed_layout

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
