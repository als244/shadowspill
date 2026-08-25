"""What the search did: repairs attempted, and where the time went."""

from __future__ import annotations

from dataclasses import dataclass, field

from .json import _integer, _mapping

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
)


def _nonnegative(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class PressureFitRepairDiagnostics:
    """Categorized monotonic changes made while repairing one search path."""

    unclassified_attempts: int = 0
    admission_prefetch_advance_attempts: int = 0
    admission_prefetch_delay_attempts: int = 0
    admission_pressure_boundary_attempts: int = 0
    simulation_prefetch_delay_attempts: int = 0
    simulation_pressure_boundary_attempts: int = 0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _nonnegative(name, getattr(self, name))

    @property
    def total_attempts(self) -> int:
        return sum(getattr(self, name) for name in self.__dataclass_fields__)

    @property
    def pressure_boundary_attempts(self) -> int:
        return (
            self.admission_pressure_boundary_attempts
            + self.simulation_pressure_boundary_attempts
        )

    def __add__(
        self, other: PressureFitRepairDiagnostics
    ) -> PressureFitRepairDiagnostics:
        if not isinstance(other, PressureFitRepairDiagnostics):
            return NotImplemented
        return PressureFitRepairDiagnostics(
            **{
                name: getattr(self, name) + getattr(other, name)
                for name in self.__dataclass_fields__
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "total_attempts": self.total_attempts,
            "pressure_boundary_attempts": self.pressure_boundary_attempts,
            "unclassified_attempts": self.unclassified_attempts,
            "admission_failure": {
                "prefetch_advance_attempts": (self.admission_prefetch_advance_attempts),
                "prefetch_delay_attempts": self.admission_prefetch_delay_attempts,
                "pressure_boundary_attempts": (
                    self.admission_pressure_boundary_attempts
                ),
            },
            "simulation_failure": {
                "prefetch_delay_attempts": self.simulation_prefetch_delay_attempts,
                "pressure_boundary_attempts": (
                    self.simulation_pressure_boundary_attempts
                ),
            },
        }

    @classmethod
    def from_value(
        cls, value: object, path: str = "pressurefit_repairs"
    ) -> PressureFitRepairDiagnostics:
        data = _mapping(value, path)
        admission = _mapping(data.get("admission_failure"), f"{path}.admission_failure")
        simulation = _mapping(
            data.get("simulation_failure"), f"{path}.simulation_failure"
        )
        result = cls(
            unclassified_attempts=_integer(
                data.get("unclassified_attempts", 0),
                f"{path}.unclassified_attempts",
            ),
            admission_prefetch_advance_attempts=_integer(
                admission.get("prefetch_advance_attempts", 0),
                f"{path}.admission_failure.prefetch_advance_attempts",
            ),
            admission_prefetch_delay_attempts=_integer(
                admission.get("prefetch_delay_attempts", 0),
                f"{path}.admission_failure.prefetch_delay_attempts",
            ),
            admission_pressure_boundary_attempts=_integer(
                admission.get("pressure_boundary_attempts", 0),
                f"{path}.admission_failure.pressure_boundary_attempts",
            ),
            simulation_prefetch_delay_attempts=_integer(
                simulation.get("prefetch_delay_attempts", 0),
                f"{path}.simulation_failure.prefetch_delay_attempts",
            ),
            simulation_pressure_boundary_attempts=_integer(
                simulation.get("pressure_boundary_attempts", 0),
                f"{path}.simulation_failure.pressure_boundary_attempts",
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
class PressureFitSectionTiming:
    """Disjoint spans of one planning step, as its orchestrator measured them.

    Sections do not overlap: exactly one is open at a time, and the function
    that opens it is the one that names it. ``total_ns`` is the whole span the
    orchestrator covered, and ``residual_ns`` is whatever part of it no named
    section claimed, so

        total_ns == named_ns + residual_ns

    holds at every level -- which is the point of the shape. ``admit_ns`` is
    the exception: admission runs as part of simulating, so it is nested
    inside ``simulate_ns`` rather than beside it, and stands outside that sum.

    ``prepare_ns`` is problem-level only; a candidate never prepares anything.
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
        """Time every named section claimed. Nested sections excluded."""

        return sum(getattr(self, name) for name in SECTION_NAMES)

    def __add__(self, other: PressureFitSectionTiming) -> PressureFitSectionTiming:
        if not isinstance(other, PressureFitSectionTiming):
            return NotImplemented
        return PressureFitSectionTiming(
            **{
                name: getattr(self, name) + getattr(other, name)
                for name in self.__dataclass_fields__
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_value(
        cls, value: object, path: str = "sections"
    ) -> PressureFitSectionTiming:
        data = _mapping(value, path)
        return cls(
            **{
                name: _integer(data.get(name, 0), f"{path}.{name}")
                for name in cls.__dataclass_fields__
            }
        )


@dataclass(frozen=True, slots=True)
class PressureFitWorkDiagnostics:
    """Exact search operations, and the sections the time went to.

    Invocation-level values include shared work performed before or across
    candidates. Consequently they need not equal the sum of candidate values.
    Times are summed section work, not necessarily elapsed wall time when
    independent recomputation problems are evaluated concurrently.
    """

    residency_cache_hits: int = 0
    residency_cache_misses: int = 0
    schedule_emissions: int = 0
    schedule_cache_hits: int = 0
    simulation_calls: int = 0
    simulation_cache_hits: int = 0
    admission_calls: int = 0
    sections: PressureFitSectionTiming = field(default_factory=PressureFitSectionTiming)

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if name != "sections":
                _nonnegative(name, getattr(self, name))

    @property
    def simulation_requests(self) -> int:
        return self.simulation_calls + self.simulation_cache_hits

    def __add__(self, other: PressureFitWorkDiagnostics) -> PressureFitWorkDiagnostics:
        if not isinstance(other, PressureFitWorkDiagnostics):
            return NotImplemented
        return PressureFitWorkDiagnostics(
            **{
                name: getattr(self, name) + getattr(other, name)
                for name in self.__dataclass_fields__
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "residency": {
                "cache_hits": self.residency_cache_hits,
                "evaluations": self.residency_cache_misses,
            },
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
        cls, value: object, path: str = "pressurefit_work"
    ) -> PressureFitWorkDiagnostics:
        data = _mapping(value, path)
        residency = _mapping(data.get("residency"), f"{path}.residency")
        schedule = _mapping(data.get("schedule"), f"{path}.schedule")
        simulation = _mapping(data.get("simulation"), f"{path}.simulation")
        admission = _mapping(data.get("admission"), f"{path}.admission")
        result = cls(
            residency_cache_hits=_integer(
                residency.get("cache_hits", 0), f"{path}.residency.cache_hits"
            ),
            residency_cache_misses=_integer(
                residency.get("evaluations", 0), f"{path}.residency.evaluations"
            ),
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
            sections=PressureFitSectionTiming.from_value(
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

    A candidate reaches its answer by holding a succession of plans: it
    reduces, emits, simulates, and either keeps what it got or repairs it and
    goes again. A step is one turn of that, so the steps in order are the
    search itself rather than a summary of it -- which plan stalled, which one
    was measured, which one gave capacity back, and which one it answered with.

    Recorded only when the caller asks: the trajectory costs an allocation per
    candidate that grows with the search, which is worth paying when
    attributing planner time or explaining a plan, and not otherwise.
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

    def to_dict(self) -> dict[str, object]:
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
