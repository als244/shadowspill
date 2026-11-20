"""Configurable task profiling policy and the evidence that it completed."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields, replace
from typing import Literal


@dataclass(frozen=True, slots=True)
class ProfilingOptions:
    """Bounds on how each task form is conditioned and timed.

    Time floors count device time measured around the exact task; wall-time
    limits bound further runs, checked between runs and never interrupting
    one. Timing stops once the minimum count, the device-time floor and both
    stability checks are met, or at a wall-time or count limit; the profile
    records which. Distributed workers agree on every repetition. The windows
    do not bring a device to thermal equilibrium.

    Parameters
    ----------
    conditioning_seconds : float, default 0.25
        Device time, in seconds, to run a task before timing it. Zero
        disables this floor.
    conditioning_wall_seconds : float, default 3.0
        Wall-time limit, in seconds, on conditioning runs.
    conditioning_max_samples : int, default 32
        Maximum conditioning runs, even if the time floor is unmet.
    minimum_samples : int, default 3
        Timing samples always taken, whatever the wall-time limit.
    maximum_samples : int, default 128
        Maximum timing samples, at least ``minimum_samples``.
    measurement_seconds : float, default 0.18
        Device time, in seconds, to accumulate across timing samples. Zero
        disables this floor.
    measurement_wall_seconds : float, default 2.0
        Wall-time limit, in seconds, on timing samples after the minimum
        count.
    relative_mad_threshold : float, default 0.03
        Largest median absolute deviation, as a fraction of the median, at
        which samples are stable.
    half_drift_threshold : float, default 0.03
        Largest difference between the medians of the first and second half
        of the samples, as a fraction of the overall median, at which samples
        are stable.

    Raises
    ------
    ValueError
        If a count is not a positive integer, ``maximum_samples`` is below
        ``minimum_samples``, or a duration or threshold is negative or not
        finite.
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
        """Return the options as plain data.

        Returns
        -------
        dict
            Every field, by name.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> ProfilingOptions:
        """Rebuild options from a `to_dict` record.

        Parameters
        ----------
        value : object
            A dict with exactly the option fields.

        Returns
        -------
        ProfilingOptions
            The options.

        Raises
        ------
        ValueError
            If ``value`` is not such a dict, or an option is invalid.
        """
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
    """How one conditioning or timing window ran, as a task profile records it.

    Attributes
    ----------
    gpu_ns : int
        Device time of the window's runs, in nanoseconds.
    wall_ns : int
        Wall time the window took, in nanoseconds.
    iterations : int
        Runs in the window.
    target_met : bool
        Whether ``gpu_ns`` reached the window's device-time floor.
    limit : str or None
        The limit that ended the window early: ``"samples"`` for the count
        limit, ``"wall"`` for the wall-time limit. A window ends early when its
        floor is unmet or, for timing, its samples are unstable. None when it
        ended normally.
    """

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
        """Return the window as plain data.

        Returns
        -------
        dict
            Every field, by name.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> TimingWindow:
        """Rebuild a window from a `to_dict` record.

        Parameters
        ----------
        value : object
            A dict with exactly the window's fields.

        Returns
        -------
        TimingWindow
            The window.

        Raises
        ------
        ValueError
            If ``value`` is not such a dict, or a field is invalid.
        """
        if not isinstance(value, dict) or set(value) != {
            item.name for item in fields(cls)
        }:
            raise ValueError("invalid profiling timing window")
        return cls(**value)


__all__ = ["ProfilingOptions", "TimingWindow"]
