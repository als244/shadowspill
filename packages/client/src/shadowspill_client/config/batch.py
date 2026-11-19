"""Schema-owned capacity dimensions and per-worker resource budgets."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from .._records import Record, Value, mapping, number


@dataclass(frozen=True)
class BatchConfig(Record):
    """Limits on one submitted batch, and hints for timing it.

    Parameters
    ----------
    limits : mapping of str to int or float
        Hard limit for each capacity the model's input schema declares,
        naming exactly those capacities, such as ``{"token_capacity": 512,
        "max_sequence_length": 128, "max_sequences": 4}``. Each is at least
        zero and bounds the whole batch of one call, across all workers;
        scheduled padding counts toward it.
    typical : mapping, default {}
        Representative input sizes, such as ``{"sequence_length": 2048}``,
        for the inputs the preparation preset builds to time tasks. The
        preset defines the accepted keys; the generic stacked preset accepts
        none.
    """

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
    """Memory budgets for each worker, in GiB.

    Parameters
    ----------
    device_gib : float
        Device memory budget per worker, in GiB. Positive. It covers the
        Plan's execution pool and fixed overhead, including the headroom
        `Preparation` reserves.
    spill_gib : float
        Spill budget per worker, in GiB. At least zero. The spill pool is
        pinned host memory for values the Plan moves off the device; a worker
        allocates its Plan's requirement plus a small margin, up to this
        budget.
    """

    device_gib: float
    spill_gib: float

    def __post_init__(self) -> None:
        if number(self.device_gib, "device_gib") <= 0:
            raise ValueError("device_gib must be positive")
        number(self.spill_gib, "spill_gib", minimum=0)
