"""Catalog entries and their default text geometry."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from importlib import import_module
from typing import Any

from workloads.providers import ModelImplementation

IMPLEMENTATIONS: tuple[ModelImplementation, ...] = ("pytorch", "mlops")


@dataclass(frozen=True, slots=True)
class Geometry:
    """Where a benchmark of a model starts: the sequence length, the tokens
    one microbatch holds and the tokens one optimizer step consumes."""

    sequence_length: int
    tokens_per_microbatch: int
    tokens_per_step: int

    def __post_init__(self) -> None:
        if self.tokens_per_microbatch % self.sequence_length:
            raise ValueError("the sequence length must divide the microbatch")
        if self.tokens_per_step % self.tokens_per_microbatch:
            raise ValueError("the microbatch must divide the step")

    @property
    def sequences_per_microbatch(self) -> int:
        return self.tokens_per_microbatch // self.sequence_length

    @property
    def accumulation_count(self) -> int:
        return self.tokens_per_step // self.tokens_per_microbatch


@dataclass(frozen=True, slots=True)
class Entry:
    """One catalog model."""

    name: str
    family: str
    size: str
    implementations: tuple[ModelImplementation, ...]
    geometry: Geometry
    mixture: bool = False
    expert_parallel: bool = False
    factory: str | None = None
    factory_arguments: Mapping[str, Any] = field(default_factory=dict)
    description: str = ""

    def config(self, **changes: Any) -> Any:
        """The family's ``Config`` at this size, with ``changes`` in place."""

        presets = import_module(f"workloads.models.{self.family}.config").PRESETS
        try:
            preset = presets[self.size]
        except KeyError as error:
            raise ValueError(
                f"{self.family} has no {self.size!r} preset; it has {list(presets)}"
            ) from error
        try:
            return replace(preset, **changes)
        except TypeError as error:
            raise ValueError(f"{self.name}: {error}") from error

    def model_class(self, implementation: str) -> Any:
        """The ``Model`` class of one of this entry's implementations."""

        if implementation not in self.implementations:
            raise ValueError(
                f"{self.name} has no {implementation!r} implementation; "
                f"it has {', '.join(self.implementations)}"
            )
        return import_module(
            f"workloads.models.{self.family}.implementations.{implementation}"
        ).Model
