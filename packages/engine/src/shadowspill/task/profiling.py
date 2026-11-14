"""Configurable task profiling policy and the evidence that it completed."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields, replace
from typing import Literal


@dataclass(frozen=True, slots=True)
class ProfilingOptions:
    """Bounded conditioning and timing of each unique finalized task.

    Duration targets count device-event time from the exact task. Wall limits
    bound additional repetitions and are checked between invocations, never
    interrupting an invocation or skipping the minimum measurement count.
    Count limits also bound tiny tasks that cannot reach a time floor cheaply.
    Distributed workers agree on each repetition. Zero duration targets disable
    their corresponding time floor. These windows do not establish GPU thermal
    equilibrium; see the profiling guide for operating-condition controls.
    """

    conditioning_seconds: float = 0.25
    conditioning_wall_seconds: float = 3.0
    conditioning_max_samples: int = 32
    minimum_samples: int = 3
    maximum_samples: int = 128
    measurement_seconds: float = 0.18
    measurement_wall_seconds: float = 2.0
    relative_mad_threshold: float = 0.03
    half_drift_threshold: float = 0.03

    def __post_init__(self) -> None:
        counts = ("conditioning_max_samples", "minimum_samples", "maximum_samples")
        for name in counts:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"profiling {name} must be a positive integer")
        if self.maximum_samples < self.minimum_samples:
            raise ValueError("profiling maximum_samples must be >= minimum_samples")
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


def resolve_options(
    options: ProfilingOptions | None,
    warmup_seconds: float | None,
    measure_seconds: float | None,
) -> ProfilingOptions:
    """Resolve the two duration shortcuts through the same effective policy."""
    if options is not None:
        if not isinstance(options, ProfilingOptions):
            raise TypeError("options must be ProfilingOptions")
        if warmup_seconds is not None or measure_seconds is not None:
            raise ValueError("pass options or duration shortcuts, not both")
        return options
    default = ProfilingOptions()
    return replace(
        default,
        conditioning_seconds=(
            default.conditioning_seconds if warmup_seconds is None else warmup_seconds
        ),
        measurement_seconds=(
            default.measurement_seconds if measure_seconds is None else measure_seconds
        ),
    )


@dataclass(frozen=True, slots=True)
class TimingWindow:
    """Observed repetitions, device-time coverage, and a limiting stop reason."""

    gpu_ns: int = 0
    wall_ns: int = 0
    iterations: int = 0
    target_met: bool = True
    limit: Literal["samples", "wall"] | None = None

    def __post_init__(self) -> None:
        if (
            any(
                type(value) is not int or value < 0
                for value in (self.gpu_ns, self.wall_ns, self.iterations)
            )
            or type(self.target_met) is not bool
            or self.limit not in (None, "samples", "wall")
        ):
            raise ValueError("invalid profiling timing window")

    def to_dict(self) -> dict[str, int | bool | str | None]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> TimingWindow:
        if not isinstance(value, dict) or set(value) != {
            item.name for item in fields(cls)
        }:
            raise ValueError("invalid profiling timing window")
        return cls(**value)


__all__ = ["ProfilingOptions", "TimingWindow"]
