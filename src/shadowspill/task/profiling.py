"""Configurable task profiling policy and the evidence that it completed."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True, slots=True)
class ProfilingOptions:
    """Initialization counts and sustained-task timing budgets.

    Duration targets count device-event time from the exact task. Wall limits
    bound additional repetitions and are checked between invocations, never
    interrupting an invocation or skipping the minimum measurement count.
    Zero duration targets disable their corresponding time floor.
    """

    warmup_iterations: int = 3
    stabilization_iterations: int = 16
    conditioning_seconds: float = 1.0
    conditioning_wall_seconds: float = 3.0
    minimum_samples: int = 15
    measurement_seconds: float = 0.3
    measurement_wall_seconds: float = 2.0
    relative_mad_threshold: float = 0.03
    half_drift_threshold: float = 0.03

    def __post_init__(self) -> None:
        counts = ("warmup_iterations", "stabilization_iterations", "minimum_samples")
        for name in counts:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"profiling {name} must be a positive integer")
        for option in fields(self):
            if option.name in counts:
                continue
            value = getattr(self, option.name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(
                    f"profiling {option.name} must be finite and non-negative"
                )

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> ProfilingOptions:
        if not isinstance(value, dict) or set(value) != {
            item.name for item in fields(cls)
        }:
            raise ValueError("invalid profiling options")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class TimingWindow:
    """Observed task repetitions and whether their device-time floor was met."""

    gpu_ns: int = 0
    wall_ns: int = 0
    iterations: int = 0
    target_met: bool = True

    def __post_init__(self) -> None:
        if (
            any(
                type(value) is not int or value < 0
                for value in (self.gpu_ns, self.wall_ns, self.iterations)
            )
            or type(self.target_met) is not bool
        ):
            raise ValueError("invalid profiling timing window")

    def to_dict(self) -> dict[str, int | bool]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> TimingWindow:
        if not isinstance(value, dict) or set(value) != {
            item.name for item in fields(cls)
        }:
            raise ValueError("invalid profiling timing window")
        return cls(**value)


__all__ = ["ProfilingOptions", "TimingWindow"]
