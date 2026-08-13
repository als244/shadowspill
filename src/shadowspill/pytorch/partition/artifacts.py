"""Immutable artifacts produced by PyTorch graph partitioning."""

from __future__ import annotations

from dataclasses import dataclass

from torch.fx import GraphModule

from ..capture import TaskInputProvenance
from ..output_contract import ExplicitMutation


@dataclass(frozen=True, slots=True)
class StageValueSource:
    """Root-graph provenance for one positional stage input."""

    root_input_index: int | None = None
    producer_stage_index: int | None = None
    producer_output_index: int | None = None

    def __post_init__(self) -> None:
        root = self.root_input_index is not None
        produced = self.producer_stage_index is not None
        if root == produced:
            raise ValueError("stage source must be either a root input or stage output")
        if root:
            if self.root_input_index is None or self.root_input_index < 0:
                raise ValueError("stage root-input source is invalid")
            if self.producer_output_index is not None:
                raise ValueError("root-input source cannot name a producer output")
        else:
            if (
                self.producer_stage_index is None
                or self.producer_stage_index < 0
                or self.producer_output_index is None
                or self.producer_output_index < 0
            ):
                raise ValueError("stage-output source is invalid")


@dataclass(frozen=True, slots=True)
class Stage:
    """One semantic partition occurrence in topological model order."""

    stage_id: str
    module_target: str
    graph_module: GraphModule
    input_sources: tuple[StageValueSource | None, ...]
    input_provenance: tuple[TaskInputProvenance, ...]
    mutations: tuple[ExplicitMutation, ...]
    user_output_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class StageExample:
    """One stage paired with representative values for its captured ABI."""

    stage: Stage
    inputs: tuple[object, ...]
    output: object


@dataclass(frozen=True, slots=True)
class StageRecord:
    """One observed call through the split root graph.

    This internal record names fields that were previously encoded as an
    order-sensitive five-tuple.  It is deliberately independent of AOT graph
    pairs: partitioning ends once these stage-local values and sources exist.
    """

    module_target: str
    graph_module: GraphModule
    inputs: tuple[object, ...]
    input_sources: tuple[StageValueSource | None, ...]
    output: object


@dataclass(frozen=True, slots=True)
class PartitionedExport:
    """Executable split root and topologically ordered stage examples."""

    root: GraphModule
    root_inputs: tuple[object, ...]
    root_input_provenance: tuple[TaskInputProvenance, ...]
    stages: tuple[StageExample, ...]
    repeated_groups: tuple[str, ...]
    user_output_indices: tuple[int, ...]
