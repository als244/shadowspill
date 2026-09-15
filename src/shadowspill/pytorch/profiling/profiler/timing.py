"""Timing samples of a warmed task, and when they are stable enough to stop."""

from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass

#: Relative spread above which a measurement is reported as unstable.
STABLE_VARIABILITY = 0.03
MAXIMUM_SAMPLES = 15


@dataclass(frozen=True, slots=True)
class TimingObservation:
    samples: tuple[int, ...]
    relative_mad: float
    half_drift: float

    @property
    def variability(self) -> float:
        return max(self.relative_mad, self.half_drift)


def timing_stability(samples: Sequence[int]) -> tuple[float, float]:
    """Return relative MAD and first/second-half median drift."""

    if not samples:
        raise ValueError("timing stability requires at least one sample")
    median = float(statistics.median(samples))
    if median <= 0:
        return (0.0, 0.0) if not any(samples) else (float("inf"), float("inf"))
    mad = float(statistics.median(abs(value - median) for value in samples)) / median
    midpoint = len(samples) // 2
    if midpoint == 0:
        return mad, 0.0
    first = float(statistics.median(samples[:midpoint]))
    second = float(statistics.median(samples[-midpoint:]))
    return mad, abs(first - second) / median


def collect_timing_samples(
    sample: Callable[[], int],
    *,
    minimum: int,
) -> TimingObservation:
    """Sample until the spread settles, two more at a time, up to fifteen.

    Fewer than five requested samples are taken as given: the caller asked
    for a quick reading, not a stable one.
    """

    samples: list[int] = []
    while True:
        target = minimum if not samples else min(MAXIMUM_SAMPLES, len(samples) + 2)
        samples.extend(sample() for _ in range(target - len(samples)))
        relative_mad, half_drift = timing_stability(samples)
        if (
            minimum < 5
            or max(relative_mad, half_drift) <= STABLE_VARIABILITY
            or len(samples) >= MAXIMUM_SAMPLES
        ):
            return TimingObservation(tuple(samples), relative_mad, half_drift)
