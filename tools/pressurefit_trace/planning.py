"""Reproduce the recorded search choices when analyzing a saved trace."""

from shadowspill.step import StepDataOrdering


def orderings(count, mode="factors"):
    if mode not in {"factors", "depth-first"}:
        raise ValueError("ordering must be factors or depth-first")
    if count is None:
        return [None]
    if mode == "depth-first":
        return [StepDataOrdering.depth_first(count)]
    return [
        StepDataOrdering(depth=depth, breadth=count // depth)
        for depth in range(1, count + 1)
        if count % depth == 0
    ]


def search_options(request):
    from fractions import Fraction

    from shadowspill.planner import SearchOptions
    from shadowspill.planner.search.algorithms.pressurefit import PressureFit
    from shadowspill.planner.search.algorithms.pressurefit.options import (
        PressureFitOptions,
    )
    from shadowspill.planner.search.toolkit.resolution import named_resolution_options

    shares = named_resolution_options(request.get("resolution_options", "quarters"))
    return SearchOptions(
        workers=request["search_workers"],
        algorithm=PressureFit(
            PressureFitOptions(
                resolution_options=tuple(Fraction(v) for v in shares),
                tail_save_counts=tuple(request.get("tail_save_counts", (0,))),
            )
        ),
    )
