"""CPU preparation results retain every capability, witness and Plan dependency."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from shadowspill.artifacts.plan import Plan
from shadowspill.planning.shared import SharedMemory
from shadowspill.step.ordering import StepDataOrdering
from shadowspill_client.inputs.coverage import CoverageCertificate
from shadowspill_client.protocol.operations import OperationKey

from .templates import TemplateCandidate, TemplateInventory


@dataclass(frozen=True)
class WorkerPlans:
    main: str
    optimizer: str | None

    def members(self) -> tuple[str, ...]:
        return tuple(name for name in (self.main, self.optimizer) if name is not None)


@dataclass(frozen=True)
class PlannedCandidate:
    id: str
    source: TemplateCandidate
    ordering: StepDataOrdering | None
    workers: Mapping[int, WorkerPlans]
    duration_seconds: float

    @property
    def operation(self) -> OperationKey:
        return self.source.layout.operation

    @property
    def batch(self) -> int:
        return self.source.layout.candidate.batch


@dataclass(frozen=True)
class PlanRejection:
    candidate: str
    operation: OperationKey
    batch: int
    case: int
    count: int
    ordering: str | None
    stage: str
    reason: str
    search_exhausted: bool = False


@dataclass(frozen=True)
class PreparedPlans:
    inventory: TemplateInventory
    families: Mapping[OperationKey, tuple[PlannedCandidate, ...]]
    plans: Mapping[int, Mapping[str, Plan]]
    coverage: Mapping[tuple[OperationKey, int], CoverageCertificate]
    memory: Mapping[int, SharedMemory]
    rejected: tuple[PlanRejection, ...]
    family_attempts: int
