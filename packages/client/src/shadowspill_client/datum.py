"""Logical caller examples; packing and input ownership belong to submission."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class Datum:
    """One example: the model's inputs and the objective's inputs.

    Each mapping is a tree of named fields matching the model's or the
    objective's input schema; a field's path is its keys, such as
    ``("tokens",)``. A leaf is a NumPy array or scalar, a CPU PyTorch tensor,
    a Python bool, int or float, or a nested list of them, with a dtype that
    `TensorField` accepts. The client never shifts or batches fields.

    Construction copies only the two top-level mappings, into read-only views.
    A submission checks every value against the schemas and the run's limits,
    and copies every array before it returns; the arrays may be changed or
    reused afterwards.

    Parameters
    ----------
    inputs : mapping of str to array or mapping
        Model inputs, by field name.
    loss_inputs : mapping of str to array or mapping, default {}
        Objective inputs, by field name, such as targets, masks and weights.

    Raises
    ------
    TypeError
        If ``inputs`` or ``loss_inputs`` is not a mapping.
    ValueError
        If a top-level key is not a nonempty str.
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
