"""Fit a nonnegative startup time plus byte-proportional copy cost."""

from collections.abc import Sequence
from math import isfinite, sqrt


def fit_copy_cost(samples: Sequence[tuple[int, float]]) -> tuple[float, float, float]:
    """Return bandwidth, latency and RMS fit error, in bytes/seconds units."""
    if len(samples) < 2 or len({size for size, _ in samples}) < 2:
        raise ValueError("copy calibration needs at least two distinct sizes")
    if any(
        type(size) is not int or size <= 0 or not isfinite(time) or time <= 0
        for size, time in samples
    ):
        raise ValueError("copy measurements must have positive sizes and durations")
    # Scale bytes before fitting to avoid poorly conditioned normal equations.
    scale = max(size for size, _ in samples)
    x = [size / scale for size, _ in samples]
    y = [time for _, time in samples]
    mx, my = sum(x) / len(x), sum(y) / len(y)
    slope = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True)) / sum(
        (a - mx) ** 2 for a in x
    )
    if slope <= 0:
        raise ValueError("copy timings do not increase with size; retry calibration")
    latency = my - slope * mx
    if latency < 0:
        latency = 0.0
        slope = sum(a * b for a, b in zip(x, y, strict=True)) / sum(a * a for a in x)
    error = sqrt(
        sum((b - latency - slope * a) ** 2 for a, b in zip(x, y, strict=True)) / len(x)
    )
    return scale / slope, latency, error
