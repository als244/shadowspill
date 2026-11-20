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

    A budget of zero is automatic: the service sizes it when it describes the
    run, and the resolved settings hold the size. An automatic budget is 90%
    of the worker's share of the memory: its GPU's memory for ``device_gib``,
    and its node's host memory divided evenly among the node's GPUs for
    ``spill_gib``.

    Parameters
    ----------
    device_gib : float
        Device memory budget per worker, in GiB. At least zero; zero is
        automatic. It covers the Plan's execution pool and fixed overhead,
        including the headroom `Preparation` reserves.
    spill_gib : float
        Spill budget per worker, in GiB. At least zero; zero is automatic.
        The spill pool is pinned host memory for values the Plan moves off
        the device; a worker allocates its Plan's requirement plus a small
        margin, up to this budget.

    Raises
    ------
    ValueError
        If a budget is negative or not a finite number.
    """

    device_gib: float
    spill_gib: float

    def __post_init__(self) -> None:
        number(self.device_gib, "device_gib", minimum=0)
        number(self.spill_gib, "spill_gib", minimum=0)
