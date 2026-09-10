"""Coarsen measured planning inputs so near-identical measurements plan alike.

Transfer calibration, the pool capacity a process ends up with, and a budget
left to default to that capacity all differ slightly from one process to the
next. Planning identity hashes these inputs, so without coarsening every
process would plan afresh and the store would fill with near-duplicate plans.
Values are coarsened only once they reach one quantum; smaller values (unit
tests plan against pools of a few hundred bytes) stay exact.

Bandwidth and latency are coarsened by **magnitude**, not by one fixed step.
A fixed step is wrong at both ends of the range it has to cover: a step coarse
enough to absorb the run-to-run noise of a slow lane erases the difference
between two fast ones, and a step fine enough for a fast lane leaves a slow
one planning afresh every process. So each quantity has bands, and a value is
rounded to the granularity its own size deserves.
"""

from __future__ import annotations

from typing import Final

GIBIBYTE: Final = 1 << 30
GIGABYTE_PER_SECOND: Final = 1_000_000_000
MICROSECOND_NS: Final = 1_000

#: Bandwidth bands, coarsest first: a lane at or above half a gigabyte a
#: second rounds to the half, and a slower one to a tenth, where half a
#: gigabyte would be most of its measured rate.
_BANDWIDTH_BANDS: Final = (
    (GIGABYTE_PER_SECOND // 2, GIGABYTE_PER_SECOND // 2),
    (0, GIGABYTE_PER_SECOND // 10),
)

#: Latency bands, coarsest first, as (the value this applies above, quantum).
#: A millisecond-scale latency is noisy in whole tens of microseconds, while a
#: few-microsecond one is meaningful to the microsecond, so the quantum grows
#: with the value rather than staying where the smallest case needs it.
_LATENCY_BANDS: Final = (
    (10_000 * MICROSECOND_NS, 1_000 * MICROSECOND_NS),
    (1_000 * MICROSECOND_NS, 250 * MICROSECOND_NS),
    (500 * MICROSECOND_NS, 100 * MICROSECOND_NS),
    (100 * MICROSECOND_NS, 50 * MICROSECOND_NS),
)

#: At and above this, latency rounds to whole multiples of it; below, to the
#: microsecond.
_LATENCY_BASE_QUANTUM: Final = 5 * MICROSECOND_NS


def floored(value: int, quantum: int) -> int:
    """``value`` rounded down to a whole number of ``quantum`` once it reaches one."""

    if value < quantum:
        return value
    return value - value % quantum


def nearest(value: int, quantum: int) -> int:
    """``value`` rounded to the nearest whole ``quantum`` once it reaches one."""

    if value < quantum:
        return value
    return ((value + quantum // 2) // quantum) * quantum


def quantized_bandwidth(bytes_per_second: int) -> int:
    """A measured lane rate, coarsened by its own magnitude."""

    for above, quantum in _BANDWIDTH_BANDS:
        if bytes_per_second >= above:
            return nearest(bytes_per_second, quantum)
    return bytes_per_second


def quantized_latency(nanoseconds: int) -> int:
    """A measured per-transfer latency, coarsened by its own magnitude."""

    for above, quantum in _LATENCY_BANDS:
        if nanoseconds > above:
            return nearest(nanoseconds, quantum)
    if nanoseconds >= _LATENCY_BASE_QUANTUM:
        return nearest(nanoseconds, _LATENCY_BASE_QUANTUM)
    return nearest(nanoseconds, MICROSECOND_NS)


__all__ = [
    "GIBIBYTE",
    "GIGABYTE_PER_SECOND",
    "MICROSECOND_NS",
    "floored",
    "nearest",
    "quantized_bandwidth",
    "quantized_latency",
]
