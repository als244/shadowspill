"""What the search did: repairs attempted, and work spent doing them."""

from __future__ import annotations

from dataclasses import dataclass

from .json import _integer, _mapping


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
class PressureFitWorkDiagnostics:
    """Exact search operations and summed component work time.

    Invocation-level values include shared work performed before or across
    candidates. Consequently they need not equal the sum of candidate values.
    Times are summed component work, not necessarily elapsed wall time when
    independent recomputation problems are evaluated concurrently.
    """

    evaluation_time_ns: int = 0
    residency_cache_hits: int = 0
    residency_cache_misses: int = 0
    schedule_emissions: int = 0
    schedule_cache_hits: int = 0
    simulation_calls: int = 0
    simulation_cache_hits: int = 0
    result_simulation_calls: int = 0
    admission_calls: int = 0
    result_admission_calls: int = 0
    residency_time_ns: int = 0
    schedule_time_ns: int = 0
    simulation_time_ns: int = 0
    result_simulation_time_ns: int = 0
    admission_time_ns: int = 0
    result_admission_time_ns: int = 0
    digest_time_ns: int = 0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _nonnegative(name, getattr(self, name))

    @property
    def simulation_requests(self) -> int:
        return self.simulation_calls + self.simulation_cache_hits

    @property
    def total_simulation_calls(self) -> int:
        return self.simulation_calls + self.result_simulation_calls

    @property
    def total_admission_calls(self) -> int:
        return self.admission_calls + self.result_admission_calls

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
            "evaluation": {"summed_wall_time_ns": self.evaluation_time_ns},
            "residency": {
                "cache_hits": self.residency_cache_hits,
                "evaluations": self.residency_cache_misses,
                "summed_work_time_ns": self.residency_time_ns,
            },
            "schedule": {
                "cache_hits": self.schedule_cache_hits,
                "emissions": self.schedule_emissions,
                "summed_work_time_ns": self.schedule_time_ns,
            },
            "simulation": {
                "requests": self.simulation_requests,
                "search_calls": self.simulation_calls,
                "cache_hits": self.simulation_cache_hits,
                "result_materialization_calls": self.result_simulation_calls,
                "total_calls": self.total_simulation_calls,
                "summed_work_time_ns": self.simulation_time_ns,
                "result_materialization_time_ns": (self.result_simulation_time_ns),
            },
            "admission": {
                "search_calls": self.admission_calls,
                "result_materialization_calls": self.result_admission_calls,
                "total_calls": self.total_admission_calls,
                "summed_work_time_ns": self.admission_time_ns,
                "result_materialization_time_ns": self.result_admission_time_ns,
            },
            "digest": {"summed_work_time_ns": self.digest_time_ns},
        }

    @classmethod
    def from_value(
        cls, value: object, path: str = "pressurefit_work"
    ) -> PressureFitWorkDiagnostics:
        data = _mapping(value, path)
        evaluation = _mapping(data.get("evaluation"), f"{path}.evaluation")
        residency = _mapping(data.get("residency"), f"{path}.residency")
        schedule = _mapping(data.get("schedule"), f"{path}.schedule")
        simulation = _mapping(data.get("simulation"), f"{path}.simulation")
        admission = _mapping(data.get("admission"), f"{path}.admission")
        digest = _mapping(data.get("digest"), f"{path}.digest")
        result = cls(
            evaluation_time_ns=_integer(
                evaluation.get("summed_wall_time_ns", 0),
                f"{path}.evaluation.summed_wall_time_ns",
            ),
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
                simulation.get("search_calls", simulation.get("calls", 0)),
                f"{path}.simulation.search_calls",
            ),
            simulation_cache_hits=_integer(
                simulation.get("cache_hits", 0), f"{path}.simulation.cache_hits"
            ),
            result_simulation_calls=_integer(
                simulation.get("result_materialization_calls", 0),
                f"{path}.simulation.result_materialization_calls",
            ),
            admission_calls=_integer(
                admission.get("search_calls", admission.get("calls", 0)),
                f"{path}.admission.search_calls",
            ),
            result_admission_calls=_integer(
                admission.get("result_materialization_calls", 0),
                f"{path}.admission.result_materialization_calls",
            ),
            residency_time_ns=_integer(
                residency.get("summed_work_time_ns", 0),
                f"{path}.residency.summed_work_time_ns",
            ),
            schedule_time_ns=_integer(
                schedule.get("summed_work_time_ns", 0),
                f"{path}.schedule.summed_work_time_ns",
            ),
            simulation_time_ns=_integer(
                simulation.get("summed_work_time_ns", 0),
                f"{path}.simulation.summed_work_time_ns",
            ),
            result_simulation_time_ns=_integer(
                simulation.get("result_materialization_time_ns", 0),
                f"{path}.simulation.result_materialization_time_ns",
            ),
            admission_time_ns=_integer(
                admission.get("summed_work_time_ns", 0),
                f"{path}.admission.summed_work_time_ns",
            ),
            result_admission_time_ns=_integer(
                admission.get("result_materialization_time_ns", 0),
                f"{path}.admission.result_materialization_time_ns",
            ),
            digest_time_ns=_integer(
                digest.get("summed_work_time_ns", 0),
                f"{path}.digest.summed_work_time_ns",
            ),
        )
        expected = {
            "requests": result.simulation_requests,
            "total_calls": result.total_simulation_calls,
        }
        for name, expected_value in expected.items():
            declared = simulation.get(name)
            if (
                declared is not None
                and _integer(declared, f"{path}.simulation.{name}") != expected_value
            ):
                raise ValueError(f"{path}.simulation.{name} does not reconcile")
        declared_admission_calls = admission.get("total_calls")
        if (
            declared_admission_calls is not None
            and _integer(declared_admission_calls, f"{path}.admission.total_calls")
            != result.total_admission_calls
        ):
            raise ValueError(f"{path}.admission.total_calls does not reconcile")
        return result
