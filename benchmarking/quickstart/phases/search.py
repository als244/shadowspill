"""Enumerate CPU planning choices without rebuilding or remeasuring a candidate."""

from shadowspill import plan_program
from shadowspill.artifacts.sweep import SweepCase, SweepResult
from shadowspill.errors import PlanInfeasibleError, PlanSearchExhaustedError
from shadowspill.step import StepDataOrdering


def search(programs, entries, budgets, communication, options):
    cases, choices = [], {}
    for budget in budgets:
        for entry in entries:
            name = entry["name"]
            for ordering in orderings(entry.get("num_examples")):
                label = "default" if ordering is None else ordering.label
                case_name = f"{name}: {label}"
                choices[case_name] = {"candidate": name, "ordering": label}
                if "failure" in entry:
                    cases.append(SweepCase(case_name, budget, failure=entry["failure"]))
                    continue
                print(
                    f"plan: {case_name}, {budget.execution_bytes / (1 << 30):g} GiB",
                    flush=True,
                )
                try:
                    plan = plan_program(
                        programs[name],
                        memory_budgets=budget,
                        communication_info=communication,
                        search_options=options,
                        ordering=ordering,
                    )
                except (PlanInfeasibleError, PlanSearchExhaustedError) as error:
                    cases.append(
                        SweepCase(
                            case_name,
                            budget,
                            failure={
                                "type": type(error).__name__,
                                "message": str(error),
                                "diagnostics": [
                                    item.to_dict()
                                    if hasattr(item, "to_dict")
                                    else str(item)
                                    for item in error.diagnostics
                                ],
                            },
                        )
                    )
                else:
                    cases.append(SweepCase(case_name, budget, plan))
    return SweepResult(tuple(cases)), choices


def orderings(count):
    if count is None:
        return [None]
    return [
        StepDataOrdering(depth=depth, breadth=count // depth)
        for depth in range(1, count + 1)
        if count % depth == 0
    ]
