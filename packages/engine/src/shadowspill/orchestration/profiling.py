"""Resolve one serializable profiling policy for phase workers."""

from collections.abc import Mapping
from typing import Any

from shadowspill import ProfilingOptions
from shadowspill.task.profiling import resolve_options


def profiling_options(config: Mapping[str, Any]) -> ProfilingOptions:
    """Accept a Python policy or partial JSON object; reject conflicting shortcuts."""
    policy = config.get("profiling")
    if isinstance(policy, Mapping):
        policy = ProfilingOptions(**policy)
    return resolve_options(
        policy, config.get("warmup_seconds"), config.get("measure_seconds")
    )
