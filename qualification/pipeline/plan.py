"""CPU-only plan generation; no model, PyTorch, allocator or calibration."""

import json
import sys
from pathlib import Path

from shadowspill import CommunicationInfo, MemoryBudgets, ProfiledProgram, plan_program
from shadowspill.errors import PlanInfeasibleError, PlanSearchExhaustedError
from shadowspill.planner import GenericPlanningOptions, SearchOptions
from shadowspill.step import StepDataOrdering


def run(directory: Path) -> None:
    settings = json.loads((directory / "settings.json").read_text())
    ordering = settings.get("data_ordering")
    try:
        plan = plan_program(
            ProfiledProgram.load(directory / "profiled.ssp"),
            memory_budgets=MemoryBudgets(
                settings["execution_bytes"], settings["spill_bytes"]
            ),
            communication_info=CommunicationInfo.load(directory / "communication.ssp"),
            ordering=None
            if ordering is None
            else StepDataOrdering.from_label(ordering),
            optimizer_ordering=settings.get("optimizer_ordering", "stage_interleaved"),
            search_options=SearchOptions(
                generic=GenericPlanningOptions(deterministic=True)
            ),
        )
    except (PlanInfeasibleError, PlanSearchExhaustedError) as error:
        # Preserve the child process's detailed rejection, not just its exit code.
        details: dict[str, object] = {
            "error": str(error),
            "type": type(error).__name__,
            "diagnostics": [item.to_dict() for item in error.diagnostics],
        }
        if isinstance(error, PlanInfeasibleError):
            details.update(
                kind=error.kind,
                required_bytes=error.required_bytes,
                capacity_bytes=error.capacity_bytes,
            )
        (directory / "plan.failure.json").write_text(
            json.dumps(details, indent=2) + "\n"
        )
        raise
    plan.save(directory / "plan.ssp")
    (directory / "summary.json").write_text(json.dumps(plan.summary, indent=2) + "\n")
    if "torch" in sys.modules:
        raise AssertionError("CPU planning imported PyTorch")
    print(f"planned: {plan.summary}", flush=True)


if __name__ == "__main__":
    run(Path(sys.argv[1]))
