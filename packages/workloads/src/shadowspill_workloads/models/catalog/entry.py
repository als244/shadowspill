"""Supplied model architectures and implementations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from importlib import import_module
from typing import Any

from shadowspill_workloads.providers import ModelImplementation

IMPLEMENTATIONS: tuple[ModelImplementation, ...] = ("pytorch", "mlops")


@dataclass(frozen=True, slots=True)
class Entry:
    """One catalog model."""

    name: str
    family: str
    size: str
    implementations: tuple[ModelImplementation, ...]
    mixture: bool = False
    expert_parallel: bool = False
    description: str = ""

    def config(self, **changes: Any) -> Any:
        """The family's ``Config`` at this size, with ``changes`` in place."""

        presets = import_module(
            f"shadowspill_workloads.models.{self.family}.config"
        ).PRESETS
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
            f"shadowspill_workloads.models.{self.family}.implementations.{implementation}"
        ).Model
