"""Axis conventions the figures share."""

from __future__ import annotations


def budget_label(gibibytes: float) -> str:
    """Name an execution budget for a tick.

    A pool's capacity resolves to an exact byte count, so a budget arrives here
    as 29.0137 rather than 29: precision an axis whose points sit a gibibyte
    apart cannot use. A tenth of a gibibyte is as fine as a tick is read, and a
    whole value loses its trailing zero, so 29.0137 reads as 29 and 7.5 stays
    7.5.
    """

    return f"{round(gibibytes, 1):g}"


__all__ = ["budget_label"]
