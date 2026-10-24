"""The logical parameter used by a rank's computation, before storage sharding."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Replicated:
    group: str

    def __post_init__(self) -> None:
        if not isinstance(self.group, str) or not self.group:
            raise ValueError("placement group must be a nonempty name")


@dataclass(frozen=True)
class Sharded:
    """Equal contiguous slices of one parameter, in the group's declared order.

    The model supplies its local slice's shape. Repeated group instances own
    copies of corresponding slices, so those copies still require reduction.
    Noncontiguous mappings will be added with distributed input-layout support.
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


def placement_from_record(value: dict[str, Any]) -> Placement:
    fields = dict(value)
    kind = fields.pop("kind")
    if kind == "Replicated":
        return Replicated(**fields)
    if kind == "Sharded":
        return Sharded(**fields)
    raise ValueError(f"unknown parameter placement: {kind}")
