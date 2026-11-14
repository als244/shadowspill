"""Logical caller examples; packing and input ownership belong to submission."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class Datum:
    """Named model and objective inputs, with no implicit shifting or batching.

    Arrays and nested values remain caller-owned until a training submission
    snapshots them. Constructing this record does not copy large payloads.
    """

    inputs: Mapping[str, Any]
    loss_inputs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("inputs", "loss_inputs"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"Datum.{name} must be a mapping")
            if any(type(key) is not str or not key for key in value):
                raise ValueError(f"Datum.{name} keys must be nonempty strings")
            object.__setattr__(self, name, MappingProxyType(dict(value)))
