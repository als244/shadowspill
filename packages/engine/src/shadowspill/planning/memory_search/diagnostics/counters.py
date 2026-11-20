"""What the search did: repairs attempted, and where the time went."""

from __future__ import annotations

from dataclasses import dataclass, field

from shadowspill.planning.memory_search.diagnostics.json import _integer, _mapping

#: The sections that partition a step's total, in the order work reaches them.
#: Nested sections -- admission inside simulation -- are deliberately absent.
SECTION_NAMES = (
    "prepare_ns",
    "setup_ns",
    "reduce_ns",
    "emit_ns",
    "simulate_ns",
    "repair_ns",
    "digest_ns",
    "place_ns",
    "select_ns",
    "teardown_ns",
)

#: What became of a plan the search held, in flag-bit order.
STEP_OUTCOMES = (
    "simulated",
    "measured",
    "placed",
    "refined",
    "best",
    "answer",
    "moved",
)


def _nonnegative(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class PlanningRepairDiagnostics:
    """A search's repairs, by what refused the plan and what the repair did.

    Two of these add with ``+``, field by field.

    Parameters
    ----------
    unclassified_attempts : int, default 0
        Repairs the search did not attribute to a category below.
    admission_fetch_advance_attempts : int, default 0
        Admission refused the plan, and a fetch was moved earlier.
    admission_fetch_delay_attempts : int, default 0
        Admission refused the plan, and a fetch was moved later.
    admission_pressure_boundary_attempts : int, default 0
        Admission refused the plan, and room was made at a pressure
        boundary.
    simulation_fetch_delay_attempts : int, default 0
        Simulation refused the plan, and a fetch was moved later.
    simulation_pressure_boundary_attempts : int, default 0
        Simulation refused the plan, and room was made at a pressure
        boundary.
    layout_fetch_delay_attempts : int, default 0
        The plan's layout overran the pool, and a fetch was moved later so
        it could fit.

    Raises
    ------
    ValueError
        If a count is negative or not an integer.
    """

    unclassified_attempts: int = 0
    admission_fetch_advance_attempts: int = 0
    admission_fetch_delay_attempts: int = 0
    admission_pressure_boundary_attempts: int = 0
    simulation_fetch_delay_attempts: int = 0
    simulation_pressure_boundary_attempts: int = 0
    #: A layout overran the pool and a fetch was moved later so it could fit.
    layout_fetch_delay_attempts: int = 0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _nonnegative(name, getattr(self, name))

    @property
    def total_attempts(self) -> int:
        """Every repair, of every category."""
        return sum(getattr(self, name) for name in self.__dataclass_fields__)

    @property
    def pressure_boundary_attempts(self) -> int:
        """Repairs that made room at a pressure boundary, whatever refused."""
        return (
            self.admission_pressure_boundary_attempts
            + self.simulation_pressure_boundary_attempts
        )

    def __add__(self, other: PlanningRepairDiagnostics) -> PlanningRepairDiagnostics:
        if not isinstance(other, PlanningRepairDiagnostics):
            return NotImplemented
        return PlanningRepairDiagnostics(
            **{
                name: getattr(self, name) + getattr(other, name)
                for name in self.__dataclass_fields__
            }
        )

    def to_dict(self) -> dict[str, object]:
        """Write the counts as a JSON-compatible mapping.

        Returns
        -------
        dict
            The two totals, ``unclassified_attempts``, and the categories
            grouped under ``admission_failure``, ``simulation_failure`` and
            ``layout_miss``.
        """
        return {
            "total_attempts": self.total_attempts,
            "pressure_boundary_attempts": self.pressure_boundary_attempts,
            "unclassified_attempts": self.unclassified_attempts,
            "admission_failure": {
                "fetch_advance_attempts": (self.admission_fetch_advance_attempts),
                "fetch_delay_attempts": self.admission_fetch_delay_attempts,
                "pressure_boundary_attempts": (
                    self.admission_pressure_boundary_attempts
                ),
            },
            "simulation_failure": {
                "fetch_delay_attempts": self.simulation_fetch_delay_attempts,
                "pressure_boundary_attempts": (
                    self.simulation_pressure_boundary_attempts
                ),
            },
            "layout_miss": {
                "fetch_delay_attempts": self.layout_fetch_delay_attempts,
            },
        }

    @classmethod
    def from_value(
        cls, value: object, path: str = "search_repairs"
    ) -> PlanningRepairDiagnostics:
        """Read counts back from what `PlanningRepairDiagnostics.to_dict` wrote.

        A missing count reads as zero.

        Parameters
        ----------
        value : object
            The mapping.
        path : str, default "search_repairs"
            Where the value sits in its record, for error messages.

        Returns
        -------
        PlanningRepairDiagnostics
            The counts.

        Raises
        ------
        ValueError
            If a field is malformed, or a written total disagrees with the
            counts.
        """
        data = _mapping(value, path)
        admission = _mapping(data.get("admission_failure"), f"{path}.admission_failure")
        simulation = _mapping(
            data.get("simulation_failure"), f"{path}.simulation_failure"
        )
        layout = _mapping(data.get("layout_miss", {}), f"{path}.layout_miss")
        result = cls(
            unclassified_attempts=_integer(
                data.get("unclassified_attempts", 0),
                f"{path}.unclassified_attempts",
            ),
            admission_fetch_advance_attempts=_integer(
                admission.get("fetch_advance_attempts", 0),
                f"{path}.admission_failure.fetch_advance_attempts",
            ),
            admission_fetch_delay_attempts=_integer(
                admission.get("fetch_delay_attempts", 0),
                f"{path}.admission_failure.fetch_delay_attempts",
            ),
            admission_pressure_boundary_attempts=_integer(
                admission.get("pressure_boundary_attempts", 0),
                f"{path}.admission_failure.pressure_boundary_attempts",
            ),
            simulation_fetch_delay_attempts=_integer(
                simulation.get("fetch_delay_attempts", 0),
                f"{path}.simulation_failure.fetch_delay_attempts",
            ),
            simulation_pressure_boundary_attempts=_integer(
                simulation.get("pressure_boundary_attempts", 0),
                f"{path}.simulation_failure.pressure_boundary_attempts",
            ),
            layout_fetch_delay_attempts=_integer(
                layout.get("fetch_delay_attempts", 0),
                f"{path}.layout_miss.fetch_delay_attempts",
            ),
        )
        declared = data.get("total_attempts")
        if declared is not None and _integer(declared, f"{path}.total_attempts") != (
            result.total_attempts
        ):
            raise ValueError(f"{path}.total_attempts does not reconcile")
        declared_pressure = data.get("pressure_boundary_attempts")
        if (
            declared_pressure is not None
            and _integer(declared_pressure, f"{path}.pressure_boundary_attempts")
            != result.pressure_boundary_attempts
        ):
            raise ValueError(f"{path}.pressure_boundary_attempts does not reconcile")
        return result


@dataclass(frozen=True, slots=True)
class PlanningSectionTiming:
    """Where a search's time went, in disjoint sections, in nanoseconds.

    Exactly one section is open at a time, so ``total_ns`` equals ``named_ns``
    plus ``residual_ns`` for a candidate, a resolved program and a whole call
    alike. ``admit_ns`` is the exception: admission runs inside simulation, so
    it is part of ``simulate_ns`` and outside that sum. Sections measure work,
    so with several workers a total can exceed the elapsed time. Two of these
    add with ``+``, field by field.

    Parameters
    ----------
    total_ns : int, default 0
        The whole span measured.
    prepare_ns : int, default 0
        Deriving the residency problem from the program; zero for a
        candidate.
    setup_ns : int, default 0
        Schedule facts and the candidate workspace.
    reduce_ns : int, default 0
        Choosing what stays resident, before any repair.
    emit_ns : int, default 0
        Turning residency gaps into an ordered schedule.
    simulate_ns : int, default 0
        Simulating the schedule, admission included.
    repair_ns : int, default 0
        Moving a transfer or making room for one, and reducing again when
        that is what it took.
    digest_ns : int, default 0
        Naming the schedule.
    place_ns : int, default 0
        Measuring whether the plan has a layout that fits.
    select_ns : int, default 0
        Deciding what to answer with, and materializing it.
    teardown_ns : int, default 0
        Releasing what the evaluation held.
    admit_ns : int, default 0
        Admitting the schedule into the pool, within ``simulate_ns``.
    residual_ns : int, default 0
        ``total_ns`` less every named section.

    Raises
    ------
    ValueError
        If a time is negative or not an integer.
    """

    total_ns: int = 0
    #: Deriving the residency problem from the program. Problem level only.
    prepare_ns: int = 0
    #: Schedule facts and the candidate workspace.
    setup_ns: int = 0
    #: Choosing what stays resident, before any candidate repairs it.
    reduce_ns: int = 0
    #: Turning residency gaps into an ordered schedule.
    emit_ns: int = 0
    #: Replaying the schedule for a makespan.
    simulate_ns: int = 0
    #: Moving a transfer or making room for one, and reducing again when that
    #: is what it took.
    repair_ns: int = 0
    #: Naming the schedule.
    digest_ns: int = 0
    #: Measuring whether the plan has a layout that fits.
    place_ns: int = 0
    #: Deciding what to answer with, and materialising it.
    select_ns: int = 0
    #: Releasing everything the evaluation held.
    teardown_ns: int = 0
    #: Admitting the schedule into the pool. Inside ``simulate_ns``.
    admit_ns: int = 0
    #: ``total_ns`` less every named section above.
    residual_ns: int = 0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _nonnegative(name, getattr(self, name))

    @property
    def named_ns(self) -> int:
        """Nanoseconds every named section claimed, ``admit_ns`` excluded."""

        return sum(getattr(self, name) for name in SECTION_NAMES)

    def __add__(self, other: PlanningSectionTiming) -> PlanningSectionTiming:
        if not isinstance(other, PlanningSectionTiming):
            return NotImplemented
        return PlanningSectionTiming(
            **{
                name: getattr(self, name) + getattr(other, name)
                for name in self.__dataclass_fields__
            }
        )

    def to_dict(self) -> dict[str, object]:
        """Write the sections as a JSON-compatible mapping.

        Returns
        -------
        dict
            Every field by name.
        """
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_value(cls, value: object, path: str = "sections") -> PlanningSectionTiming:
        """Read sections back from what `PlanningSectionTiming.to_dict` wrote.

        A missing section reads as zero.

        Parameters
        ----------
        value : object
            The mapping.
        path : str, default "sections"
            Where the value sits in its record, for error messages.

        Returns
        -------
        PlanningSectionTiming
            The sections.

        Raises
        ------
        ValueError
            If a field is malformed.
        """
        data = _mapping(value, path)
        return cls(
            **{
                name: _integer(data.get(name, 0), f"{path}.{name}")
                for name in cls.__dataclass_fields__
            }
        )


@dataclass(frozen=True, slots=True)
class PlanningWorkDiagnostics:
    """What a search did, counted exactly, and where the time went.

    A resolved program's or a call's counts include work shared across its
    candidates, so they need not equal the sum of its candidates'. Two of
    these add with ``+``, field by field.

    Parameters
    ----------
    schedule_emissions : int, default 0
        Schedules emitted.
    schedule_cache_hits : int, default 0
        Emissions served from the cache instead.
    simulation_calls : int, default 0
        Simulations run.
    simulation_cache_hits : int, default 0
        Simulations served from the cache instead.
    admission_calls : int, default 0
        Admission checks run.
    sections : PlanningSectionTiming, default PlanningSectionTiming()
        Where the time went.

    Raises
    ------
    ValueError
        If a count is negative or not an integer.
    """

    schedule_emissions: int = 0
    schedule_cache_hits: int = 0
    simulation_calls: int = 0
    simulation_cache_hits: int = 0
    admission_calls: int = 0
    sections: PlanningSectionTiming = field(default_factory=PlanningSectionTiming)

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if name != "sections":
                _nonnegative(name, getattr(self, name))

    @property
    def simulation_requests(self) -> int:
        """Simulations asked for: those run and those served from the cache."""
        return self.simulation_calls + self.simulation_cache_hits

    def __add__(self, other: PlanningWorkDiagnostics) -> PlanningWorkDiagnostics:
        if not isinstance(other, PlanningWorkDiagnostics):
            return NotImplemented
        return PlanningWorkDiagnostics(
            **{
                name: getattr(self, name) + getattr(other, name)
                for name in self.__dataclass_fields__
            }
        )

    def to_dict(self) -> dict[str, object]:
        """Write the counts as a JSON-compatible mapping.

        Returns
        -------
        dict
            The counts grouped under ``schedule``, ``simulation`` (with
            ``requests``) and ``admission``, and the ``sections``.
        """
        return {
            "schedule": {
                "cache_hits": self.schedule_cache_hits,
                "emissions": self.schedule_emissions,
            },
            "simulation": {
                "requests": self.simulation_requests,
                "calls": self.simulation_calls,
                "cache_hits": self.simulation_cache_hits,
            },
            "admission": {"calls": self.admission_calls},
            "sections": self.sections.to_dict(),
        }

    @classmethod
    def from_value(
        cls, value: object, path: str = "search_work"
    ) -> PlanningWorkDiagnostics:
        """Read counts back from what `PlanningWorkDiagnostics.to_dict` wrote.

        A missing count reads as zero.

        Parameters
        ----------
        value : object
            The mapping.
        path : str, default "search_work"
            Where the value sits in its record, for error messages.

        Returns
        -------
        PlanningWorkDiagnostics
            The counts.

        Raises
        ------
        ValueError
            If a field is malformed, or the written ``requests`` disagrees with
            the counts.
        """
        data = _mapping(value, path)
        schedule = _mapping(data.get("schedule"), f"{path}.schedule")
        simulation = _mapping(data.get("simulation"), f"{path}.simulation")
        admission = _mapping(data.get("admission"), f"{path}.admission")
        result = cls(
            schedule_emissions=_integer(
                schedule.get("emissions", 0), f"{path}.schedule.emissions"
            ),
            schedule_cache_hits=_integer(
                schedule.get("cache_hits", 0), f"{path}.schedule.cache_hits"
            ),
            simulation_calls=_integer(
                simulation.get("calls", 0), f"{path}.simulation.calls"
            ),
            simulation_cache_hits=_integer(
                simulation.get("cache_hits", 0), f"{path}.simulation.cache_hits"
            ),
            admission_calls=_integer(
                admission.get("calls", 0), f"{path}.admission.calls"
            ),
            sections=PlanningSectionTiming.from_value(
                data.get("sections", {}), f"{path}.sections"
            ),
        )
        declared = simulation.get("requests")
        if (
            declared is not None
            and _integer(declared, f"{path}.simulation.requests")
            != result.simulation_requests
        ):
            raise ValueError(f"{path}.simulation.requests does not reconcile")
        return result


@dataclass(frozen=True, slots=True)
class ReductionStep:
    """One plan a candidate held, and what became of it.

    A candidate reaches its answer through a succession of plans: it
    reduces, emits and simulates, then keeps the plan or repairs it and goes
    again. A step is one turn, so `CandidateDiagnostic.steps` in order is the
    candidate's whole trajectory. Steps are recorded only when the search is
    asked for them.

    Parameters
    ----------
    makespan_ns : int
        The plan's simulated makespan, in nanoseconds.
    required_bytes : int
        Bytes its layout spans; zero unless it was measured.
    capacity_bytes : int
        The object capacity it was built against, which falls as the
        candidate gives capacity back.
    cut_aliases : tuple of int
        The objects the reducer cut to reach it, by alias index.
    repairs : int
        Repairs the candidate had made when it reached this plan.
    simulation_status : int
        The simulator's status for it; zero when it simulated.
    capacity_violations : int
        How many times it came up short of capacity and waited.
    simulated : bool
        Whether it was simulated.
    measured : bool
        Whether its layout was measured.
    placed : bool
        Whether its layout fit the pool.
    refined : bool
        Whether capacity was given back after it.
    best : bool
        Whether it was the candidate's best plan so far.
    answer : bool
        Whether the candidate answered with it.
    moved : bool, default False
        Whether its layout overran the pool and a fetch was moved later
        after it.
    """

    makespan_ns: int
    #: Bytes the layout of this plan spans. Zero unless it was measured.
    required_bytes: int
    #: The object capacity this plan was built against, which falls as the
    #: candidate hands capacity back.
    capacity_bytes: int
    #: Objects the reducer cut to reach this plan, by alias index.
    cut_aliases: tuple[int, ...]
    #: Repairs the candidate had made when it reached this plan.
    repairs: int
    simulation_status: int
    #: Places this plan came up short of capacity and waited.
    capacity_violations: int
    simulated: bool
    measured: bool
    placed: bool
    refined: bool
    best: bool
    answer: bool
    #: Its layout overran the pool and a fetch was moved later after it.
    moved: bool = False

    def to_dict(self) -> dict[str, object]:
        """Write the step as a JSON-compatible mapping.

        Returns
        -------
        dict
            The counts by name, and the seven flags under ``outcome``.
        """
        return {
            "makespan_ns": self.makespan_ns,
            "required_bytes": self.required_bytes,
            "capacity_bytes": self.capacity_bytes,
            "cut_aliases": list(self.cut_aliases),
            "repairs": self.repairs,
            "simulation_status": self.simulation_status,
            "capacity_violations": self.capacity_violations,
            "outcome": {name: getattr(self, name) for name in STEP_OUTCOMES},
        }

    @classmethod
    def from_value(cls, value: object, path: str) -> ReductionStep:
        """Read a step back from what `ReductionStep.to_dict` wrote.

        A missing count reads as zero and a missing flag as False.

        Parameters
        ----------
        value : object
            The mapping.
        path : str
            Where the value sits in its record, for error messages.

        Returns
        -------
        ReductionStep
            The step.

        Raises
        ------
        ValueError
            If a field is malformed.
        """
        data = _mapping(value, path)
        outcome = _mapping(data.get("outcome"), f"{path}.outcome")
        aliases = data.get("cut_aliases", [])
        if not isinstance(aliases, list):
            raise ValueError(f"{path}.cut_aliases must be a list")
        return cls(
            makespan_ns=_integer(data.get("makespan_ns", 0), f"{path}.makespan_ns"),
            required_bytes=_integer(
                data.get("required_bytes", 0), f"{path}.required_bytes"
            ),
            capacity_bytes=_integer(
                data.get("capacity_bytes", 0), f"{path}.capacity_bytes"
            ),
            cut_aliases=tuple(
                _integer(alias, f"{path}.cut_aliases[{index}]")
                for index, alias in enumerate(aliases)
            ),
            repairs=_integer(data.get("repairs", 0), f"{path}.repairs"),
            simulation_status=_integer(
                data.get("simulation_status", 0), f"{path}.simulation_status"
            ),
            capacity_violations=_integer(
                data.get("capacity_violations", 0), f"{path}.capacity_violations"
            ),
            **{name: bool(outcome.get(name, False)) for name in STEP_OUTCOMES},
        )
