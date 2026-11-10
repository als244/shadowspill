"""CPU-only batch/ordering sweep with jointly admitted training and evaluation."""

import json
import shutil

from shadowspill import (
    CommunicationInfo,
    MemoryBudgets,
    PoolMemoryProfile,
    ProfiledProgram,
    plan_program,
    write_report,
)
from shadowspill.errors import PlanInfeasibleError, PlanSearchExhaustedError
from shadowspill.planning.shared import shared_memory
from shadowspill.step import StepDataOrdering
from workloads.recipes.phases import settings

from .candidates import build_candidates, failures
from .search import search_options


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
        "search_options": search_options(options),
        "pool_memory": PoolMemoryProfile.load(options["pool_memory"])
        if options.get("pool_memory")
        else None,
    }
    best, rows = None, []
    rejected = failures(root)
    for candidate in build_candidates(options):
        path = candidate["path"]
        selection = dict(candidate)
        if path in rejected:
            rows.append({**selection, "failure": rejected[path]})
            continue
        at = root / "profiled" / path

        def communication(kind, path=path):
            return CommunicationInfo.load(root / "communication" / path / f"{kind}.ssp")

        evaluation = None
        try:
            if options.get("eval_every", 2):
                evaluation = plan_program(
                    ProfiledProgram.load(at / "evaluation.ssp"),
                    communication_info=communication("evaluation"),
                    **inputs,
                )
        except (PlanInfeasibleError, PlanSearchExhaustedError) as error:
            rows.append({**selection, "evaluation_error": str(error)})
            continue
        profiled = ProfiledProgram.load(at / "training.ssp")
        count = candidate["num_examples"]
        for depth in (d for d in range(1, count + 1) if count % d == 0):
            ordering = StepDataOrdering(depth=depth, breadth=count // depth)
            row = {**selection, "ordering": ordering.label}
            print(f"planning {row}", flush=True)
            try:
                training = plan_program(
                    profiled,
                    ordering=ordering,
                    communication_info=communication("training"),
                    **inputs,
                )
                if evaluation is not None:
                    shared_memory(training, {"evaluation": evaluation})
            except (PlanInfeasibleError, PlanSearchExhaustedError) as error:
                row["error"] = str(error)
            else:
                row["duration_seconds"] = training.summary["duration_seconds"]
                destination = root / "plans/candidates" / path / ordering.label
                training.save(destination / "training.ssp")
                if evaluation is not None:
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
        if plan is None:
            continue
        plan.save(root / f"plans/{name}.ssp")
        write_report(plan, root / f"reports/{name}", plots=options["plots"])
    shutil.copyfile(
        root / "programs" / choice["path"] / "metadata.json", root / "model.json"
    )
    (root / "selected.json").write_text(json.dumps(choice, indent=2) + "\n")
    print(f"selected {choice}", flush=True)


if __name__ == "__main__":
    import sys

    run(sys.argv[1])
