"""Recipe concurrency and optional PressureFit search-space controls."""

from collections.abc import Mapping
from fractions import Fraction
from typing import Any

from shadowspill.planner import GenericPlanningOptions, SearchOptions
from shadowspill.planner.search.algorithms.pressurefit import (
    PressureFit,
    PressureFitOptions,
)


def search_options(options: Mapping[str, Any]) -> SearchOptions:
    resolutions = options.get("resolution_options")
    tails = options.get("tail_save_counts")
    controls: dict[str, Any] = {}
    if resolutions is not None:
        controls["resolution_options"] = tuple(Fraction(value) for value in resolutions)
    if tails is not None:
        controls["tail_save_counts"] = tuple(tails)
    return SearchOptions(
        workers=options.get("search_workers", 8),
        algorithm=None if not controls else PressureFit(PressureFitOptions(**controls)),
        generic=GenericPlanningOptions(
            minimum_object_bytes_evict_eligible=options.get(
                "minimum_evict_bytes",
                GenericPlanningOptions().minimum_object_bytes_evict_eligible,
            )
        ),
    )
