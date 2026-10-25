"""CPU-only size/ordering search. No model, Torch or runtime pools are imported."""

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
from shadowspill.step import StepDataOrdering
from workloads.recipes.phases import settings

from .common import batch_sizes


def run(directory):
    root, options = settings(directory)
    # Never leave the previous budget's executable selected after a failed search.
    for name in ("selected.json", "plan.ssp"):
        (root / name).unlink(missing_ok=True)
    for name in ("plans", "report"):
        if (root / name).exists():
            shutil.rmtree(root / name)
    budgets = MemoryBudgets(
        int(options["execution_gib"] * (1 << 30)), int(options["spill_gib"] * (1 << 30))
    )
    best, rows = None, []
    for size in batch_sizes(options):
        program = ProfiledProgram.load(root / f"profiled/batch-{size}.ssp")
        communication = CommunicationInfo.load(root / f"communication/batch-{size}.ssp")
        count = options["sequences_per_step"] // size
        for depth in (d for d in range(1, count + 1) if count % d == 0):
            ordering = StepDataOrdering(depth=depth, breadth=count // depth)
            row = {"sequences_per_example": size, "ordering": ordering.label}
            print(f"planning {row}", flush=True)
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
                plan.save(root / f"plans/batch-{size}/{ordering.label}.ssp")
                if (
                    best is None
                    or row["duration_seconds"] < best[0]["duration_seconds"]
                ):
                    best = row, plan
            rows.append(row)
    (root / "search.json").write_text(json.dumps(rows, indent=2) + "\n")
    if best is None:
        raise PlanInfeasibleError(
            "no candidate fits; see search.json", kind="recipe_candidates"
        )
    choice, plan = best
    plan.save(root / "plan.ssp")
    write_report(plan, root / "report", plots=options["plots"])
    (root / "selected.json").write_text(json.dumps(choice, indent=2) + "\n")
    print(f"selected {choice}", flush=True)


if __name__ == "__main__":
    run(sys.argv[1])
