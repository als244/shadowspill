"""CPU preparation estimates, separate from measurements and allocation admission."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .._records import Record, integer, record
from ..config import Resources
from .hardware import WorkerPlacement
from .operations import OperationKey
from .setup import TrainingSetup
from .training import ResolvedTraining


@dataclass(frozen=True)
class TrainingDescription(Record):
    setup: TrainingSetup | None = None
    config: ResolvedTraining | None = None
    resources: Resources | None = None

    def __post_init__(self) -> None:
        if (self.setup is None) == (self.config is None):
            raise ValueError("description requires setup or resolved configuration")
        for name, kind in (
            ("setup", TrainingSetup),
            ("config", ResolvedTraining),
            ("resources", Resources),
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, record(kind, value, name))

    @classmethod
    def request(
        cls, settings: TrainingSetup | ResolvedTraining, resources: Resources | None
    ) -> TrainingDescription:
        if isinstance(settings, ResolvedTraining):
            return cls(config=settings, resources=resources)
        return cls(setup=settings, resources=resources)


@dataclass(frozen=True)
class PreparationEstimate(Record):
    """How much preparation one compute operation needs, counted before building.

    A schedule is one batch option, build case and microbatch count.

    Parameters
    ----------
    operation : OperationKey
        The compute operation.
    geometry_cases : int
        Number of distinct batch option and build case pairs among the
        candidate schedules: the physical input shapes to build. Capture can
        expand or share them.
    candidate_schedules : int
        Number of schedules that fit the batch limits, padding included.
    rejected_schedules : int
        Number of schedules that exceed a batch limit once padded.
    """

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
    """A run's resolved settings and proposed workers.

    Returned by `ServiceClient.describe_training`. Pass ``config`` and
    ``resources`` to `ServiceClient.create_training_client` to create the run
    as described.

    Parameters
    ----------
    config : ResolvedTraining
        The resolved settings.
    preparation : sequence of PreparationEstimate
        One estimate per compute operation of ``config``, in order, except
        ``optim_step``.
    resources : Resources, optional
        The hardware selection the service resolved, with ``pool`` and
        ``gpu_type`` filled in and any requested devices named by ID.
    hardware : sequence of WorkerPlacement, default ()
        The device proposed for each worker, by rank. Empty exactly when
        ``resources`` is None.
    """

    config: ResolvedTraining
    preparation: Sequence[PreparationEstimate]
    resources: Resources | None = None
    hardware: Sequence[WorkerPlacement] = ()

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
        if self.resources is not None:
            object.__setattr__(
                self, "resources", record(Resources, self.resources, "resources")
            )
        if not isinstance(self.hardware, (list, tuple)) or len(self.hardware) > 1024:
            raise ValueError("preview hardware exceeds its bound")
        hardware = tuple(
            record(WorkerPlacement, row, "worker") for row in self.hardware
        )
        if tuple(row.rank for row in hardware) != tuple(range(len(hardware))):
            raise ValueError("preview hardware requires ordered logical ranks")
        if len({(row.node, row.device.id) for row in hardware}) != len(hardware):
            raise ValueError("preview assigned one device to multiple workers")
        if bool(hardware) != (self.resources is not None):
            raise ValueError(
                "hardware preview requires its resolved resource selection"
            )
        object.__setattr__(self, "hardware", hardware)

    def summary(self) -> str:
        """Describe the preview as text for printing.

        Returns
        -------
        str
            The model, objective and batch limits, each operation's estimate,
            the proposed workers and their memory budgets.
        """
        rows = [
            f"Model: {self.config.model.name}",
            f"Objective: {self.config.objective.name}",
            "Global batch limits: "
            f"{[dict(batch.limits) for batch in self.config.batches]}",
            "Scheduled padding counts toward input capacities.",
        ]
        rows.extend(
            f"{item.operation.operation}/{item.operation.outputs} "
            f"({item.operation.mode}): {item.geometry_cases} "
            "physical cases before capture expansion/reuse; "
            f"{item.candidate_schedules} "
            f"candidate schedules, {item.rejected_schedules} rejected."
            for item in self.preparation
        )
        if self.hardware:
            rows.append("Hardware:")
            rows.extend(
                f"  Worker {row.rank}: {row.node}/{row.device.index}, "
                f"{row.device.gpu_type or 'unknown GPU'}, "
                f"{row.device.capacity_bytes / (1 << 30):.1f} GiB "
                f"({'available' if row.device.available else 'busy'})"
                for row in self.hardware
            )
            rows.append(
                f"Per-worker budgets: {self.config.memory.device_gib:g} GiB device, "
                f"{self.config.memory.spill_gib:g} GiB spill."
            )
        rows.append(
            "CPU description only; no measured timings or resource reservation."
        )
        return "\n".join(rows)
