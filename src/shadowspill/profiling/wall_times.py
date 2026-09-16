"""How long profiling took, in the classes a report attributes separately.

Compiling a task, measuring it, and warming one whose measurement was already
cached are three disjoint costs, and a plan report accounts for them separately
so that a warm run's wall clock is readable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProfilingWallTimes:
    """Disjoint wall-clock classes a planning report attributes separately."""

    compilation_ns: int
    profiling_ns: int
    cached_warmup_ns: int


__all__ = ["ProfilingWallTimes"]
