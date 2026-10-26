"""CPU-only planning of saved batch candidates, including shared evaluation."""

import json
import sys

from shadowspill import (
    CommunicationInfo,
    MemoryBudgets,
    ProfiledProgram,
    plan_program,
    write_report,
)
from shadowspill.errors import PlanInfeasibleError, PlanSearchExhaustedError
from shadowspill.step import StepDataOrdering
from workloads.recipes.phases import settings


def run(directory):
    root, options = settings(directory)
    inventory = json.loads((root / "candidates.json").read_text())
    for name in ("planning.json", "plans/training.ssp", "plans/evaluation.ssp"):
        (root / name).unlink(missing_ok=True)
    budgets = MemoryBudgets(
        int(options["execution_gib"] * (1 << 30)), int(options["spill_gib"] * (1 << 30))
    )
    communication = CommunicationInfo.load(root / "communication.ssp")
    best, rows = None, []
    for name, geometry in inventory["candidates"].items():
        program = ProfiledProgram.load(root / f"profiled/{name}/training.ssp")
        count = geometry["microbatches"]
        for depth in (d for d in range(1, count + 1) if count % d == 0):
            ordering = StepDataOrdering(depth=depth, breadth=count // depth)
            row = {"candidate": name, **geometry, "ordering": ordering.label}
            try:
                plan = plan_program(
                    program,
                    memory_budgets=budgets,
                    communication_info=communication,
                    ordering=ordering,
                )
            except (PlanInfeasibleError, PlanSearchExhaustedError) as error:
                row["error"] = str(error)
            else:
                row["duration_seconds"] = plan.summary["duration_seconds"]
                plan.save(root / f"search/{name}/{ordering.label}.ssp")
                if (
                    best is None
                    or row["duration_seconds"] < best[0]["duration_seconds"]
                ):
                    best = row, plan
            rows.append(row)
    (root / "search.json").write_text(json.dumps(rows, indent=2) + "\n")
    if best is None:
        raise PlanInfeasibleError("no candidate fits; see search.json", kind="recipe")
    choice, plan = best
    plan.save(root / "plans/training.ssp")
    if options.get("eval_every", 0):
        evaluation = plan_program(
            ProfiledProgram.load(
                root / f"profiled/{choice['candidate']}/evaluation.ssp"
            ),
            memory_budgets=budgets,
            communication_info=communication,
        )
        evaluation.save(root / "plans/evaluation.ssp")
    write_report(plan, root / "report", plots=options.get("plots", True))
    (root / "planning.json").write_text(
        json.dumps({**choice, "skipped": inventory["skipped"]}, indent=2) + "\n"
    )


if __name__ == "__main__":
    run(sys.argv[1])
