"""Schema-owned capacity dimensions and per-worker resource budgets."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from .._records import Record, Value, mapping, number


@dataclass(frozen=True)
class BatchConfig(Record):
    """Hard bounds and timing hints; recognized keys belong to the input schema."""

    limits: Mapping[str, Value]
    typical: Mapping[str, Value] = field(default_factory=dict)

    def __post_init__(self) -> None:
        limits = mapping(self.limits, "limits")
        if not limits:
            raise ValueError("limits must explicitly describe input capacity")
        for name, value in limits.items():
            number(value, f"limits.{name}", minimum=0)
        object.__setattr__(self, "limits", limits)
        object.__setattr__(self, "typical", mapping(self.typical, "typical"))


@dataclass(frozen=True)
class Memory(Record):
    """Per execution worker budgets, in GiB; node admission is separate."""

    device_gib: float
    spill_gib: float

    def __post_init__(self) -> None:
        if number(self.device_gib, "device_gib") <= 0:
            raise ValueError("device_gib must be positive")
        number(self.spill_gib, "spill_gib", minimum=0)
