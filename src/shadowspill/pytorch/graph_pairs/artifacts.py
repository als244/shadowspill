"""Immutable artifacts produced by stage-local AOT differentiation."""

from __future__ import annotations

from dataclasses import dataclass, replace

from shadowspill.pytorch.capture.aot import (
    TrainingObjectiveCapture,
    accumulate_gradient_outputs,
)
from shadowspill.pytorch.capture.artifacts import (
    AotGraphPair,
    GraphArtifact,
    TaskInputRole,
)

from ..partition.artifacts import PartitionedExport, StageExample


@dataclass(frozen=True, slots=True)
class GraphPairVariant:
    """One labeled AOT pair produced under a recomputation memory budget."""

    option_id: str
    memory_budget: float | None
    pair: AotGraphPair
    accumulates: bool = False
    """Whether this backward adds its gradients onto ones it is given.

    Microbatches after the first contribute to gradients that already exist,
    so they run a backward that takes those gradients as further arguments
    and returns the sum. Which form a stage uses follows from its microbatch,
    not from planning, so both forms share one option ID and the planner sees
    the same recomputation choices at every microbatch.
    """

    def __post_init__(self) -> None:
        if not self.option_id:
            raise ValueError("graph-pair option ID must be non-empty")
        if self.memory_budget is not None and not 0.0 <= self.memory_budget <= 1.0:
            raise ValueError("graph-pair memory budget must be between zero and one")
        if self.pair.recomputation != (self.memory_budget is not None):
            raise ValueError(
                "only min-cut graph-pair variants carry an activation-memory budget"
            )

    def accumulating(self) -> GraphPairVariant:
        """Return the form of this variant that adds onto the gradients it is given.

        Only parameter gradients outlive a microbatch; a cotangent belongs to
        the microbatch that produced it. So the parameter gradients this
        backward returns are exactly the ones a later microbatch has to add
        to, and taking them as arguments moves that addition inside the task.
        """

        return replace(
            self,
            pair=replace(
                self.pair,
                backward=accumulate_gradient_outputs(
                    self.pair.backward,
                    parameter_gradient_leaves(self.pair),
                ),
            ),
            accumulates=True,
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
        if not option_ids:
            raise ValueError("graph pairs option IDs must be unique")
        if self.reference_option_id not in option_ids:
            raise ValueError("graph-pair reference option is absent")
        forms = {(item.option_id, item.accumulates) for item in self.variants}
        if len(forms) != len(self.variants):
            raise ValueError("graph pairs option IDs must be unique")

    @property
    def reference(self) -> AotGraphPair:
        """Return the pair that establishes the canonical stage-boundary contract."""

        return self.variant(self.reference_option_id).pair

    def variant(self, option_id: str) -> GraphPairVariant:
        """Return one named variant or fail with an explicit identity error."""

        for item in self.variants:
            if item.option_id == option_id and not item.accumulates:
                return item
        raise KeyError(option_id)

    def accumulating_variants(self) -> tuple[GraphPairVariant, ...]:
        """Derive the accumulating form of every captured variant."""

        return tuple(
            item.accumulating() for item in self.variants if not item.accumulates
        )

    def options(self, *, accumulates: bool) -> tuple[GraphPairVariant, ...]:
        """Return the variants one microbatch may choose between.

        Every microbatch offers the same recomputation choices; which form of
        them it runs follows from its position rather than from planning. A
        step with a single microbatch never asks for the accumulating form, so
        the store never derives it and it is never compiled or profiled.

        Both forms are ordinary variants by the time they get here. Deriving
        one costs a graph capture, so it happens once per structural contract
        in the store and is rebound per occurrence like everything else.
        """

        return tuple(
            item for item in self.variants if item.accumulates == accumulates
        )


def parameter_gradient_leaves(pair: AotGraphPair) -> tuple[int, ...]:
    """Report which backward outputs are gradients of parameters.

    A backward returns one gradient per differentiable forward argument, at
    the argument\'s own position, so the forward\'s provenance says which of
    those outputs belong to parameters. Arguments that need no gradient leave
    a hole, and those are not accumulated onto.
    """

    provenance = pair.forward.input_provenance
    produced = _produced_output_leaves(pair.backward)
    return tuple(
        position
        for position in pair.forward.tensor_argument_positions
        if position < len(provenance)
        and provenance[position].role is TaskInputRole.PARAMETER
        and position in produced
    )


def _produced_output_leaves(backward: GraphArtifact) -> frozenset[int]:
    output = next(
        node for node in backward.graph_module.graph.nodes if node.op == "output"
    )
    return frozenset(
        index for index, value in enumerate(output.args[0]) if value is not None
    )


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
