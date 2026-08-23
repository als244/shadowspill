"""Immutable artifacts produced by stage-local AOT differentiation."""

from __future__ import annotations

from dataclasses import dataclass

from shadowspill.pytorch.capture.aot import TrainingObjectiveCapture
from shadowspill.pytorch.capture.artifacts import AotGraphPair

from ..partition.artifacts import PartitionedExport, StageExample


@dataclass(frozen=True, slots=True)
class GraphPairVariant:
    """One labeled AOT pair produced under a recomputation memory budget."""

    option_id: str
    memory_budget: float | None
    pair: AotGraphPair

    def __post_init__(self) -> None:
        if not self.option_id:
            raise ValueError("graph-pair option ID must be non-empty")
        if self.memory_budget is not None and not 0.0 <= self.memory_budget <= 1.0:
            raise ValueError("graph-pair memory budget must be between zero and one")
        if self.pair.recomputation != (self.memory_budget is not None):
            raise ValueError(
                "only min-cut graph-pair variants carry an activation-memory budget"
            )


@dataclass(frozen=True, slots=True)
class TaskGraphPairs:
    """All legal differentiated variants for one structural task contract."""

    structural_contract: str
    root_output_indices: tuple[int, ...]
    variants: tuple[GraphPairVariant, ...]
    reference_option_id: str = "save"

    def __post_init__(self) -> None:
        if not self.structural_contract:
            raise ValueError("graph pairs requires a structural contract")
        if not self.root_output_indices:
            raise ValueError("graph pairs requires differentiable roots")
        option_ids = tuple(item.option_id for item in self.variants)
        if not option_ids or len(set(option_ids)) != len(option_ids):
            raise ValueError("graph pairs option IDs must be unique")
        if self.reference_option_id not in option_ids:
            raise ValueError("graph-pair reference option is absent")

    @property
    def reference(self) -> AotGraphPair:
        """Return the pair that establishes the canonical stage-boundary contract."""

        return self.variant(self.reference_option_id).pair

    def variant(self, option_id: str) -> GraphPairVariant:
        """Return one named variant or fail with an explicit identity error."""

        for item in self.variants:
            if item.option_id == option_id:
                return item
        raise KeyError(option_id)


@dataclass(frozen=True, slots=True)
class DifferentiatedStage:
    """Bind one partition occurrence to its structural graph pairs."""

    example: StageExample
    graph_pairs: TaskGraphPairs

    @property
    def differentiable_output_indices(self) -> tuple[int, ...]:
        return self.graph_pairs.root_output_indices


@dataclass(frozen=True, slots=True)
class PartitionedTrainingCapture:
    """One objective capture decomposed into differentiated training stages."""

    training: TrainingObjectiveCapture
    partitioned: PartitionedExport
    stages: tuple[DifferentiatedStage, ...]


__all__ = [
    "DifferentiatedStage",
    "GraphPairVariant",
    "PartitionedTrainingCapture",
    "TaskGraphPairs",
]
