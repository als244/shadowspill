"""One calibration: plan every candidate, ordering and memory budget on CPU."""

from fractions import Fraction

from shadowspill import plan_program
from shadowspill.artifacts import SweepCase, SweepResult
from shadowspill.errors import PlanInfeasibleError, PlanSearchExhaustedError
from shadowspill.planner import GenericPlanningOptions, SearchOptions
from shadowspill.planner.search.algorithms.pressurefit import PressureFit
from shadowspill.planner.search.algorithms.pressurefit.options import PressureFitOptions
from shadowspill.step import StepDataOrdering


def orderings(program, mode):
    bindings = program.program.bindings
    count = len(bindings.get("training_templates", ()))
    if not count:
        return (None,)
    if mode == "depth-first" or bindings.get("ordering_constraints"):
        return (StepDataOrdering.depth_first(count),)
    return tuple(
        StepDataOrdering(depth, count // depth)
        for depth in range(1, count + 1)
        if count % depth == 0
    )


def plan(programs, budgets, communication, args):
    options = (
        {}
        if args.resolution_options is None
        else {"resolution_options": tuple(Fraction(v) for v in args.resolution_options)}
    )
    search = SearchOptions(
        generic=GenericPlanningOptions(deterministic=True),
        algorithm=PressureFit(PressureFitOptions(**options)),
        workers=args.search_workers,
    )
    cases = []
    for budget in budgets:
        for name, program in programs.items():
            for ordering in orderings(program, args.orderings):
                label = name if ordering is None else f"{name}/{ordering.label}"
                try:
                    result = plan_program(
                        program,
                        memory_budgets=budget,
                        communication_info=communication,
                        search_options=search,
                        ordering=ordering,
                    )
                except (PlanInfeasibleError, PlanSearchExhaustedError) as error:
                    cases.append(
                        SweepCase(
                            label,
                            budget,
                            failure={
                                "type": type(error).__name__,
                                "message": str(error),
                                "diagnostics": [
                                    item.to_dict() for item in error.diagnostics
                                ],
                            },
                        )
                    )
                else:
                    cases.append(SweepCase(label, budget, result))
    return SweepResult(tuple(cases))
