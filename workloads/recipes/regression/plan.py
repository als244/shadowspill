"""CPU-only planning from saved compute and communication evidence."""

import sys

from shadowspill import (
    CommunicationInfo,
    MemoryBudgets,
    ProfiledProgram,
    plan_program,
    write_report,
)
from workloads.recipes.phases import settings


def run(directory):
    root, options = settings(directory)
    plan = plan_program(
        ProfiledProgram.load(root / "profiled.ssp"),
        memory_budgets=MemoryBudgets(
            int(options["execution_gib"] * (1 << 30)),
            int(options["spill_gib"] * (1 << 30)),
        ),
        communication_info=CommunicationInfo.load(root / "communication.ssp"),
    )
    plan.save(root / "plan.ssp")
    write_report(plan, root / "report", plots=options["plots"])
    print(plan.summary, flush=True)


if __name__ == "__main__":
    run(sys.argv[1])
