"""CPU-only planning and simulated reports from independently saved measurements."""

import sys

from shadowspill import (
    CommunicationInfo,
    MemoryBudgets,
    ProfiledProgram,
    plan_program,
    write_report,
)
from shadowspill.step import StepDataOrdering
from workloads.recipes.text.phases import settings


def run(directory):
    root, options = settings(directory)
    budgets = MemoryBudgets(
        int(options["execution_gib"] * (1 << 30)),
        int(options["spill_gib"] * (1 << 30)),
    )
    communication = CommunicationInfo.load(root / "communication.ssp")
    count = options["sequences_per_step"] // options["sequences_per_example"]
    for name in ("training", "evaluation"):
        print(f"planning {name}", flush=True)
        plan = plan_program(
            ProfiledProgram.load(root / f"profiled/{name}.ssp"),
            memory_budgets=budgets,
            communication_info=communication,
            ordering=StepDataOrdering(depth=1, breadth=count)
            if name == "training"
            else None,
        )
        plan.save(root / f"plans/{name}.ssp")
        write_report(plan, root / f"reports/{name}", plots=options["plots"])
        print(f"{name}: {plan.summary}", flush=True)


if __name__ == "__main__":
    run(sys.argv[1])
