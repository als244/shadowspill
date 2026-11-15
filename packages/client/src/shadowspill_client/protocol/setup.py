"""Portable setup keywords, resolved by installed server definitions."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from .._records import Record, Value, mapping, record, text
from ..config import (
    BatchConfig,
    Execution,
    Memory,
    Model,
    Objective,
    Operation,
    Optimizer,
    Preparation,
)


@dataclass(frozen=True)
class TrainingSetup(Record):
    model: Model
    objective: Objective
    batch: BatchConfig | Sequence[BatchConfig]
    memory: Memory
    optimizer: Optimizer | None = None
    parallelism: Mapping[str, Value] | None = None
    execution: Execution | None = None
    operations: Mapping[str, Sequence[str | Operation]] | None = None
    prepare: Preparation | None = None

    def __post_init__(self) -> None:
        for name, cls in (
            ("model", Model),
            ("objective", Objective),
            ("memory", Memory),
            ("optimizer", Optimizer),
            ("execution", Execution),
            ("prepare", Preparation),
        ):
            value = getattr(self, name)
            if value is not None or name in ("model", "objective", "memory"):
                object.__setattr__(self, name, record(cls, value, name))
        batch = self.batch
        batches: tuple[BatchConfig, ...]
        if isinstance(batch, (BatchConfig, Mapping)):
            batches = (record(BatchConfig, batch, "batch"),)
        elif isinstance(batch, (tuple, list)) and 1 <= len(batch) <= 4096:
            batches = tuple(record(BatchConfig, item, "batch") for item in batch)
        else:
            raise ValueError("batch requires one or at most 4096 batch options")
        object.__setattr__(self, "batch", batches)
        if self.parallelism is not None:
            object.__setattr__(
                self, "parallelism", mapping(self.parallelism, "parallelism")
            )
        if self.operations is not None:
            if not isinstance(self.operations, Mapping):
                raise TypeError("operations requires a mapping")
            operations = {}
            for key, values in self.operations.items():
                text(key, "operation")
                if (
                    not isinstance(values, (tuple, list))
                    or not 1 <= len(values) <= 4096
                ):
                    raise ValueError(
                        "operation requires a nonempty bounded output list"
                    )
                operations[key] = tuple(
                    text(value, "output set")
                    if isinstance(value, str)
                    else record(Operation, value, "operation")
                    for value in values
                )
            object.__setattr__(self, "operations", MappingProxyType(operations))
