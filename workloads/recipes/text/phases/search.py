"""Optional search controls; defaults remain the public planner's defaults."""

from fractions import Fraction

from shadowspill.planner import GenericPlanningOptions, SearchOptions
from shadowspill.planner.search.algorithms.pressurefit import (
    PressureFit,
    PressureFitOptions,
)


def search_options(options):
    resolutions = options.get("resolution_options")
    return SearchOptions(
        workers=options.get("search_workers", 0),
        algorithm=None
        if resolutions is None
        else PressureFit(
            PressureFitOptions(
                resolution_options=tuple(Fraction(value) for value in resolutions)
            )
        ),
        generic=GenericPlanningOptions(
            minimum_object_bytes_evict_eligible=options.get(
                "minimum_evict_bytes",
                GenericPlanningOptions().minimum_object_bytes_evict_eligible,
            )
        ),
    )
