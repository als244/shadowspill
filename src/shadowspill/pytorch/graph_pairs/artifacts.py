"""Immutable artifacts produced by stage-local AOT differentiation."""

from __future__ import annotations

from dataclasses import dataclass

from ..aot import TrainingObjectiveCapture
from ..capture import AotGraphPair
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
class GraphPairPortfolio:
    """All legal differentiated variants for one structural task ABI."""

    structural_abi: str
    root_output_indices: tuple[int, ...]
    variants: tuple[GraphPairVariant, ...]
    reference_option_id: str = "save"

    def __post_init__(self) -> None:
        if not self.structural_abi:
            raise ValueError("graph-pair portfolio requires a structural ABI")
        if not self.root_output_indices:
            raise ValueError("graph-pair portfolio requires differentiable roots")
        option_ids = tuple(item.option_id for item in self.variants)
        if not option_ids or len(set(option_ids)) != len(option_ids):
            raise ValueError("graph-pair portfolio option IDs must be unique")
        if self.reference_option_id not in option_ids:
            raise ValueError("graph-pair reference option is absent")

    @property
    def reference(self) -> AotGraphPair:
        """Return the pair used to establish the canonical stage boundary ABI."""

        return self.variant(self.reference_option_id).pair

    def variant(self, option_id: str) -> GraphPairVariant:
        """Return one named variant or fail with an explicit identity error."""

        for item in self.variants:
            if item.option_id == option_id:
                return item
        raise KeyError(option_id)


@dataclass(frozen=True, slots=True)
class DifferentiatedStage:
    """Bind one partition occurrence to its structural graph-pair portfolio."""

    example: StageExample
    graph_pairs: GraphPairPortfolio

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
    "GraphPairPortfolio",
    "GraphPairVariant",
    "PartitionedTrainingCapture",
]
