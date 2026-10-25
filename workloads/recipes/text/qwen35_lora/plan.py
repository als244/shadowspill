"""CPU-only batch/ordering sweep with jointly admitted training and evaluation."""

import json
import shutil
import sys

from shadowspill import (
    CommunicationInfo,
    MemoryBudgets,
    ProfiledProgram,
    plan_program,
    write_report,
)
from shadowspill.errors import PlanInfeasibleError, PlanSearchExhaustedError
from shadowspill.planning.shared import shared_memory
from shadowspill.step import StepDataOrdering
from workloads.recipes.text.phases import settings

from .candidates import batch_sizes


def run(directory):
    root, options = settings(directory)
    # A failed replan must not leave a runnable selection from the old budget.
    (root / "selected.json").unlink(missing_ok=True)
    for name in ("plans", "reports"):
        if (root / name).exists():
            shutil.rmtree(root / name)
    budgets = MemoryBudgets(
        int(options["execution_gib"] * (1 << 30)),
        int(options["spill_gib"] * (1 << 30)),
    )
    inputs = {
        "memory_budgets": budgets,
        "communication_info": CommunicationInfo.load(root / "communication.ssp"),
    }
    best, rows = None, []
    for size in batch_sizes(options):
        at = root / f"profiled/batch-{size}"
        try:
            evaluation = plan_program(
                ProfiledProgram.load(at / "evaluation.ssp"), **inputs
            )
        except (PlanInfeasibleError, PlanSearchExhaustedError) as error:
            rows.append({"sequences_per_example": size, "evaluation_error": str(error)})
            continue
        profiled = ProfiledProgram.load(at / "training.ssp")
        count = options["sequences_per_step"] // size
        for depth in (d for d in range(1, count + 1) if count % d == 0):
            ordering = StepDataOrdering(depth=depth, breadth=count // depth)
            row = {"sequences_per_example": size, "ordering": ordering.label}
            print(f"planning {row}", flush=True)
            try:
                training = plan_program(profiled, ordering=ordering, **inputs)
                shared_memory(training, {"evaluation": evaluation})
            except (PlanInfeasibleError, PlanSearchExhaustedError) as error:
                row["error"] = str(error)
            else:
                row["duration_seconds"] = training.summary["duration_seconds"]
                destination = root / f"plans/candidates/batch-{size}/{ordering.label}"
                training.save(destination / "training.ssp")
                evaluation.save(destination / "evaluation.ssp")
                if (
                    best is None
                    or row["duration_seconds"] < best[0]["duration_seconds"]
                ):
                    best = row, training, evaluation
            rows.append(row)
    (root / "search.json").write_text(json.dumps(rows, indent=2) + "\n")
    if best is None:
        raise PlanInfeasibleError(
            "no training/evaluation pair fits; see search.json",
            kind="recipe_candidates",
        )
    choice, training, evaluation = best
    for name, plan in (("training", training), ("evaluation", evaluation)):
        plan.save(root / f"plans/{name}.ssp")
        write_report(plan, root / f"reports/{name}", plots=options["plots"])
    (root / "selected.json").write_text(json.dumps(choice, indent=2) + "\n")
    print(f"selected {choice}", flush=True)


if __name__ == "__main__":
    run(sys.argv[1])
