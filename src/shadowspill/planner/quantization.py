"""Coarsen measured planning inputs so near-identical measurements plan alike.

Transfer calibration, the pool capacity a process ends up with, and a budget
left to default to that capacity all differ slightly from one process to the
next.  Planning identity hashes these inputs, so without coarsening every
process would plan afresh and the store would fill with near-duplicate plans.
Values are coarsened only once they reach one quantum; smaller values (unit
tests plan against pools of a few hundred bytes) stay exact.
"""

from __future__ import annotations

from typing import Final

GIBIBYTE: Final = 1 << 30
GIGABYTE_PER_SECOND: Final = 1_000_000_000
MICROSECOND_NS: Final = 1_000


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


__all__ = ["GIBIBYTE", "GIGABYTE_PER_SECOND", "MICROSECOND_NS", "floored", "nearest"]
