"""Installed CPU preset and input-geometry functions, independent of model code."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from shadowspill.parallelism import Parallelism
from shadowspill_client import BatchConfig, Preparation
from shadowspill_client.protocol.definitions import DefinitionSource
from shadowspill_client.protocol.inputs import InputSchema

from .contracts import versioned_id

if TYPE_CHECKING:
    from shadowspill.training.inputs.shapes import BufferShape
    from shadowspill.training.inputs.validation import Path

    from .registry import ResolvedModel


@dataclass(frozen=True)
class PreparationContext:
    model: ResolvedModel
    schema: InputSchema
    batches: tuple[BatchConfig, ...]
    parallelism: Parallelism

    def __post_init__(self) -> None:
        batches = tuple(self.batches)
        if not batches or any(not isinstance(batch, BatchConfig) for batch in batches):
            raise ValueError("preparation context requires declared batch options")
        if not isinstance(self.schema, InputSchema) or not isinstance(
            self.parallelism, Parallelism
        ):
            raise TypeError(
                "preparation context requires a schema and generic parallelism"
            )
        object.__setattr__(self, "batches", batches)


@dataclass(frozen=True)
class PreparationDefinition:
    """Only descriptions/geometry; functions never construct models or reserve GPUs.

    ``buffers`` returns one physical template per logical input partition for a
    requested microbatch size. Variable units and coupled-field geometry belong
    to the installed definition, not to a model-name branch in the engine.
    """

    id: str
    defaults: Callable[[PreparationContext], Preparation]
    buffers: Callable[
        [PreparationContext, int, Sequence[Path], int], Sequence[BufferShape]
    ]

    def __post_init__(self) -> None:
        versioned_id(self.id)
        if not callable(self.defaults) or not callable(self.buffers):
            raise TypeError("preparation definition requires CPU functions")


@dataclass(frozen=True)
class ResolvedPreparationDefinition:
    definition: PreparationDefinition
    source: DefinitionSource
