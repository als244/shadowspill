"""One portable, immutable setup snapshot for inspection and creation."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from .._records import Record, Value, mapping, record, text
from ..config import BatchConfig, Execution, Memory, Model, Objective, Optimizer
from .controls import ControlSchema
from .definitions import DefinitionSource, OutputSet, Participation
from .inputs import InputSchema
from .operations import OperationKey
from .preparation import ResolvedPreparation

type Origin = Literal["explicit", "definition", "preset", "engine_default"]


@dataclass(frozen=True)
class ResolvedTraining(Record):
    """Resolved computation settings; no allocation, weights or live state.

    An absent model initializer remains unselected. Creation must obtain initial
    state from an explicit initializer or checkpoint before making a client ready.
    """

    model: Model
    objective: Objective
    optimizer: Optimizer | None
    batches: Sequence[BatchConfig]
    memory: Memory
    parallelism: Mapping[str, Value]
    execution: Execution
    operations: Sequence[OperationKey]
    preparation: ResolvedPreparation
    definition_sources: Mapping[str, DefinitionSource]
    policy_sources: Mapping[str, DefinitionSource]
    engine_source: DefinitionSource
    client_source: DefinitionSource
    inputs: InputSchema
    loss_controls: ControlSchema
    optimizer_controls: ControlSchema | None
    outputs: Mapping[str, OutputSet]
    participation: Participation
    origins: Mapping[str, Origin]
    reference_definition: str | None = None

    def __post_init__(self) -> None:
        for name, cls in (
            ("model", Model),
            ("objective", Objective),
            ("memory", Memory),
            ("execution", Execution),
            ("preparation", ResolvedPreparation),
            ("engine_source", DefinitionSource),
            ("client_source", DefinitionSource),
            ("inputs", InputSchema),
            ("loss_controls", ControlSchema),
            ("participation", Participation),
        ):
            object.__setattr__(self, name, record(cls, getattr(self, name), name))
        if self.model.definition is None:
            raise ValueError("resolved model requires its installed definition")
        if self.optimizer is not None:
            object.__setattr__(
                self, "optimizer", record(Optimizer, self.optimizer, "optimizer")
            )
        if self.optimizer_controls is not None:
            object.__setattr__(
                self,
                "optimizer_controls",
                record(ControlSchema, self.optimizer_controls, "optimizer_controls"),
            )
        if (self.optimizer is None) != (self.optimizer_controls is None):
            raise ValueError(
                "resolved optimizer and its controls must be supplied together"
            )
        sources = {"model", "objective"} | (
            {"optimizer"} if self.optimizer is not None else set()
        )
        if self.reference_definition is not None:
            text(self.reference_definition, "reference_definition")
            sources.add("reference")
        pretrained = (
            self.model.initialize is not None
            and self.model.initialize.kind == "pretrained"
        )
        if pretrained != (self.reference_definition is not None) or (
            pretrained
            and self.model.initialize is not None
            and self.model.initialize.revision is None
        ):
            raise ValueError(
                "pretrained metadata requires its resolver and pinned revision"
            )
        for name in ("definition_sources", "policy_sources"):
            object.__setattr__(
                self,
                name,
                MappingProxyType(
                    {
                        text(k, name): record(DefinitionSource, v, k)
                        for k, v in getattr(self, name).items()
                    }
                ),
            )
        if self.definition_sources.keys() != sources:
            raise ValueError(
                "resolved setup requires exactly its definition source pins"
            )
        if not isinstance(self.batches, (tuple, list)) or not self.batches:
            raise ValueError("resolved setup requires explicit batch options")
        batches = tuple(record(BatchConfig, b, "batch") for b in self.batches)
        if any(b in batches[:i] for i, b in enumerate(batches)):
            raise ValueError("resolved batch options must be distinct")
        object.__setattr__(self, "batches", batches)
        object.__setattr__(
            self, "parallelism", mapping(self.parallelism, "parallelism")
        )
        operations = tuple(
            record(OperationKey, op, "operation") for op in self.operations
        )
        if (
            not operations
            or len(set(operations)) != len(operations)
            or all(op.operation == "optim_step" for op in operations)
        ):
            raise ValueError(
                "resolved operations require distinct compute combinations"
            )
        dependency = self.optimizer is not None and any(
            op.operation == "forward_backward" for op in operations
        )
        if (OperationKey("optim_step", None) in operations) != dependency:
            raise ValueError(
                "resolved operations must preserve the optimizer-only dependency"
            )
        if self.optimizer is None and any(
            op.operation in ("step", "optim_step") for op in operations
        ):
            raise ValueError("optimizer operations require a resolved optimizer")
        object.__setattr__(self, "operations", operations)
        outputs = {
            text(k, "output"): record(OutputSet, v, k) for k, v in self.outputs.items()
        }
        if outputs.keys() != {
            op.outputs for op in operations if op.outputs is not None
        }:
            raise ValueError(
                "resolved outputs must match exactly the requested combinations"
            )
        object.__setattr__(self, "outputs", MappingProxyType(outputs))
        origins = dict(self.origins)
        if any(
            not isinstance(k, str)
            or not k
            or v not in ("explicit", "definition", "preset", "engine_default")
            for k, v in origins.items()
        ):
            raise ValueError("unknown resolved setting origin")
        object.__setattr__(self, "origins", MappingProxyType(origins))
