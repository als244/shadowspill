"""CPU preparation estimates, separate from measurements and allocation admission."""

from collections.abc import Sequence
from dataclasses import dataclass

from .._records import Record, integer, record
from .operations import OperationKey
from .training import ResolvedTraining


@dataclass(frozen=True)
class PreparationEstimate(Record):
    operation: OperationKey
    geometry_cases: int
    candidate_schedules: int
    rejected_schedules: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "operation", record(OperationKey, self.operation, "operation")
        )
        for name in (
            "geometry_cases",
            "candidate_schedules",
            "rejected_schedules",
        ):
            integer(getattr(self, name), name)
        if not 1 <= self.geometry_cases <= self.candidate_schedules <= 4096:
            raise ValueError("preview requires bounded nonempty preparation cases")
        if self.rejected_schedules > 4096:
            raise ValueError("preview rejected schedules exceed the expansion bound")


@dataclass(frozen=True)
class TrainingPreview(Record):
    config: ResolvedTraining
    preparation: Sequence[PreparationEstimate]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "config", record(ResolvedTraining, self.config, "config")
        )
        estimates = tuple(
            record(PreparationEstimate, item, "preparation")
            for item in self.preparation
        )
        if tuple(item.operation for item in estimates) != tuple(
            key for key in self.config.operations if key.operation != "optim_step"
        ):
            raise ValueError("preview estimates differ from requested operations")
        object.__setattr__(self, "preparation", estimates)

    def summary(self) -> str:
        """Describe requested work without inventing task counts or fresh timings."""
        rows = [
            f"Model: {self.config.model.definition}",
            f"Objective: {self.config.objective.definition}",
            f"Batch limits: {[dict(batch.limits) for batch in self.config.batches]}",
            "Scheduled padding counts toward input capacities.",
        ]
        rows.extend(
            f"{item.operation.operation}/{item.operation.outputs} "
            f"({item.operation.mode}): {item.geometry_cases} "
            f"physical cases before capture expansion/reuse; {item.candidate_schedules} "
            f"candidate schedules, {item.rejected_schedules} rejected."
            for item in self.preparation
        )
        rows.append(
            "CPU description only; no measured timings or resource reservation."
        )
        return "\n".join(rows)
