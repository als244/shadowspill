"""Condition and measure an exact task with device-time floors and wall limits."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from shadowspill.task.profiling import ProfilingOptions, TimingWindow


@dataclass(frozen=True, slots=True)
class TimingObservation:
    samples: tuple[int, ...]
    relative_mad: float
    half_drift: float
    window: TimingWindow

    @property
    def variability(self) -> float:
        return max(self.relative_mad, self.half_drift)

    def unstable(self, options: ProfilingOptions) -> bool:
        return (
            not self.window.target_met
            or self.relative_mad > options.relative_mad_threshold
            or self.half_drift > options.half_drift_threshold
        )


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


def condition_task(
    sample: Callable[[], int],
    *,
    options: ProfilingOptions,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> TimingWindow:
    """Sustain the exact initialized task before its measurement window."""

    target = round(options.conditioning_seconds * 1e9)
    limit = round(options.conditioning_wall_seconds * 1e9)
    began = clock()
    gpu_ns = iterations = 0
    elapsed = 0
    while gpu_ns < target and elapsed < limit:
        gpu_ns += _sample(sample)
        iterations += 1
        elapsed = clock() - began
    return TimingWindow(gpu_ns, elapsed, iterations, gpu_ns >= target)


def collect_timing_samples(
    sample: Callable[[], int],
    *,
    options: ProfilingOptions,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> TimingObservation:
    """Meet the sample count and device-time floor, bounded by wall time."""

    target = round(options.measurement_seconds * 1e9)
    limit = round(options.measurement_wall_seconds * 1e9)
    began = clock()
    samples: list[int] = []
    gpu_ns = 0
    while True:
        value = _sample(sample)
        samples.append(value)
        gpu_ns += value
        elapsed = clock() - began
        if len(samples) < options.minimum_samples or (
            gpu_ns < target and elapsed < limit
        ):
            continue
        relative_mad, half_drift = timing_stability(samples)
        stable = (
            relative_mad <= options.relative_mad_threshold
            and half_drift <= options.half_drift_threshold
        )
        if (gpu_ns >= target and stable) or elapsed >= limit:
            return TimingObservation(
                tuple(samples),
                relative_mad,
                half_drift,
                TimingWindow(gpu_ns, elapsed, len(samples), gpu_ns >= target),
            )


def _sample(sample: Callable[[], int]) -> int:
    duration = sample()
    if type(duration) is not int or duration < 0:
        raise ValueError("task timing must be a non-negative integer in nanoseconds")
    return duration
