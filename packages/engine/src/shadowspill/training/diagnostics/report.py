"""Write measured and simulated timelines from the same saved Plan."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from shadowspill.artifacts.plan import Plan
from shadowspill.diagnostics.occupancy import attribute, write_pages
from shadowspill.diagnostics.step import StepDiagnostics
from shadowspill.diagnostics.step_file import write_step_diagnostics
from shadowspill.planner.serialization.simulation_result import simulation_result_value
from shadowspill.planning.costs import simulation_inputs


def write(
    plan: Plan, diagnostics: StepDiagnostics, directory: Path, summary: dict[str, Any]
) -> None:
    _, config = simulation_inputs(
        plan.profiled_program, plan.memory_budgets, plan.communication_info
    )
    assert plan.memory_layout is not None
    selection = {
        "schema": "training_startup_selection/v1",
        "program_digest": plan._costed_program.digest,
        "schedule": plan.schedule.to_dict(),
        "selections": [item.to_dict() for item in plan.selections],
        "simulation": asdict(config),
        "simulation_result": simulation_result_value(plan.timeline),
        "admission_certificate": {"layout": plan.memory_layout.to_dict()},
    }
    program = plan._costed_program.to_dict()
    write_step_diagnostics(directory / "step.json", diagnostics)
    for name, value in (("program", program), ("selection", selection)):
        (directory / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n")
    pages = write_pages(
        [
            attribute(selection, program),
            attribute(selection, program, diagnostics=diagnostics.as_dict()),
        ],
        directory / "timelines",
        title="Startup diagnostics (LR=0)",
    )
    summary["timeline"] = str(pages[-1])
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
