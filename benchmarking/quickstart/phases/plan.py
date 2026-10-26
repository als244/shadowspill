"""CPU-only replanning and reports from saved profile artifacts."""

from shadowspill import (
    CommunicationInfo,
    MemoryBudgets,
    ProfiledProgram,
    write_report,
)
from shadowspill.planner import SearchOptions

from .common import paths, plan_identity, write
from .search import search


def run(request):
    root, index = paths(request)
    # A failed replan must not leave the former selection executable by accident.
    write(root / "plans" / "index.json", [])
    communication = CommunicationInfo.load(
        request.get("communication_info") or root / "communication.ssp"
    )
    programs = {
        entry["name"]: ProfiledProgram.load(root / "profiled" / entry["file"])
        for entry in index["candidates"]
    }
    print("plan: " + ", ".join(programs), flush=True)
    result, choices = search(
        programs,
        index["candidates"],
        [
            MemoryBudgets(budget, request["spill_bytes"])
            for budget in request["search_budgets"]
        ],
        communication,
        SearchOptions(workers=request["search_workers"]),
    )
    result.save(root / "planning" / "sweep.ssp")
    write_report(result, root / "report", plots=request["plots"])
    winners = []
    for index, (budget, plan) in enumerate(result.best_per_budget.items()):
        filename = f"budget-{index:04d}.ssp"
        plan.save(root / "plans" / filename)
        candidate = next(c.candidate for c in result.cases if c.plan is plan)
        winners.append(
            {
                "budget": budget.execution_bytes,
                **choices[candidate],
                "file": filename,
                "predicted_seconds": plan.summary["duration_seconds"],
                "plan_digest": plan_identity(plan),
            }
        )
    write(root / "plans" / "index.json", winners)
