"""The logical parameter used by a rank's computation, before storage sharding."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass(frozen=True)
class Replicated:
    """A parameter every rank computes with whole."""


@dataclass(frozen=True)
class Sharded:
    """A parameter split into equal contiguous slices over each group's ranks.

    Each rank of a group computes with its own slice, in the group's declared
    order; the model supplies the local slice's shape. Corresponding ranks of
    the collection's other groups hold copies of the same slice, whose
    gradients are still reduced.

    Parameters
    ----------
    group : str
        Name of a group collection of the `Parallelism`.
    dim : int
        The dimension the parameter is split along.

    Raises
    ------
    ValueError
        If ``group`` is not a nonempty string.
    TypeError
        If ``dim`` is not an integer.
    """

    group: str
    dim: int

    def __post_init__(self) -> None:
        if not isinstance(self.group, str) or not self.group:
            raise ValueError("placement group must be a nonempty name")
        if type(self.dim) is not int:
            raise TypeError("sharded dimension must be an integer")


type Placement = Replicated | Sharded


def placement_record(value: Placement) -> dict[str, Any]:
    return {"kind": type(value).__name__, **asdict(value)}


def placement_from_record(value: Mapping[str, Any], where: str) -> Placement:
    data = dict(value)
    if "kind" not in data:
        raise ValueError(f"{where}: missing key 'kind'")
    kind = data.pop("kind")
    kinds: dict[str, type[Replicated] | type[Sharded]] = {
        "Replicated": Replicated,
        "Sharded": Sharded,
    }
    if not isinstance(kind, str) or kind not in kinds:
        raise ValueError(f"{where}: unknown parameter placement: {kind}")
    keys = [item.name for item in fields(kinds[kind])]
    for key in keys:
        if key not in data:
            raise ValueError(f"{where}: missing key {key!r}")
    for key in data:
        if key not in keys:
            raise ValueError(f"{where}: unknown key {key!r}")
    return kinds[kind](**data)
