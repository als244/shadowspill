"""CPU-only planning of saved batch candidates, with evaluation admitted jointly."""

import json
import math
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
from workloads.recipes.phases import settings

from . import targets

#: What a budget the config leaves out keeps back: 2 GiB of the device for
#: the display and other processes, a quarter of the host memory that was
#: available when profiling ran.
DEVICE_RESERVE_GIB = 2
HOST_SHARE = 0.75


def budgets(options, hardware):
    """The configured budgets in GiB, or ones the profiled hardware gives."""
    execution, spill = options.get("execution_gib"), options.get("spill_gib")
    if execution is None or spill is None:
        if hardware is None:
            raise ValueError(
                "execution_gib or spill_gib is not set and the profile phase "
                "recorded no hardware.json"
            )
        if execution is None:
            execution = (
                math.floor(hardware["device_total_bytes"] / (1 << 30))
                - DEVICE_RESERVE_GIB
            )
        if spill is None:
            spill = math.floor(
                hardware["host_available_bytes"] * HOST_SHARE / (1 << 30)
            )
    if execution <= 0 or spill < 0:
        raise ValueError(f"budgets must be positive: {execution} GiB, {spill} GiB")
    return execution, spill


def shortfall(error):
    """How far an infeasible candidate missed, when the refusal measured it."""
    required = getattr(error, "required_bytes", None)
    capacity = getattr(error, "capacity_bytes", None)
    if required is None or capacity is None:
        return None
    return required - capacity


def run(directory):
    root, options = settings(directory)
    root = targets.artifacts(root, options)
    inventory = json.loads((root / "candidates.json").read_text())
    for name in ("planning.json", "plans/training.ssp", "plans/evaluation.ssp"):
        (root / name).unlink(missing_ok=True)
    record = root / "hardware.json"
    execution, spill = budgets(
        options, json.loads(record.read_text()) if record.exists() else None
    )
    print(f"budgets: execution {execution} GiB, spill {spill} GiB", flush=True)
    budget = MemoryBudgets(int(execution * (1 << 30)), int(spill * (1 << 30)))
    count = options.get("ranks", 1)

    def communication(name, kind):
        # One worker measured host copies once; ranks measured each Program's
        # collectives in the communicate phase.
        if count == 1:
            return CommunicationInfo.load(root / "communication.ssp")
        return CommunicationInfo.load(root / f"communication/{name}/{kind}.ssp")

    best, rows, misses = None, [], []
    for name, geometry in inventory["candidates"].items():
        # The evaluation Plan shares the worker with the training Plan, so a
        # pair is admitted together here, where a refusal names the candidate,
        # rather than when the worker opens.
        evaluation = None
        if options.get("eval_every", 0):
            try:
                evaluation = plan_program(
                    ProfiledProgram.load(root / f"profiled/{name}/evaluation.ssp"),
                    memory_budgets=budget,
                    communication_info=communication(name, "evaluation"),
                )
            except (PlanInfeasibleError, PlanSearchExhaustedError) as error:
                rows.append(
                    {"candidate": name, **geometry, "evaluation_error": str(error)}
                )
                if (missed := shortfall(error)) is not None:
                    misses.append((missed, error.kind))
                continue
        program = ProfiledProgram.load(root / f"profiled/{name}/training.ssp")
        count = geometry["microbatches"]
        for depth in (d for d in range(1, count + 1) if count % d == 0):
            ordering = StepDataOrdering(depth=depth, breadth=count // depth)
            row = {"candidate": name, **geometry, "ordering": ordering.label}
            try:
                plan = plan_program(
                    program,
                    memory_budgets=budget,
                    communication_info=communication(name, "training"),
                    ordering=ordering,
                )
                if evaluation is not None:
                    shared_memory(plan, {"evaluation": evaluation})
            except (PlanInfeasibleError, PlanSearchExhaustedError) as error:
                row["error"] = str(error)
                if (missed := shortfall(error)) is not None:
                    row["shortfall_bytes"] = missed
                    misses.append((missed, error.kind))
            else:
                row["duration_seconds"] = plan.summary["duration_seconds"]
                plan.save(root / f"search/{name}/{ordering.label}.ssp")
                if (
                    best is None
                    or row["duration_seconds"] < best[0]["duration_seconds"]
                ):
                    best = row, plan, evaluation
            rows.append(row)
    (root / "search.json").write_text(json.dumps(rows, indent=2) + "\n")
    if best is None:
        what = (
            "no training/evaluation pair fits"
            if options.get("eval_every", 0)
            else "no candidate fits"
        )
        nearest = ""
        if misses:
            missed, kind = min(misses)
            nearest = (
                f"; the nearest miss needs {missed / (1 << 30):.2f} GiB more ({kind})"
            )
        raise PlanInfeasibleError(
            f"{what} an execution budget of {execution} GiB and a spill budget of "
            f"{spill} GiB{nearest}; see search.json",
            kind="recipe",
        )
    choice, plan, evaluation = best
    plan.save(root / "plans/training.ssp")
    if evaluation is not None:
        evaluation.save(root / "plans/evaluation.ssp")
    write_report(plan, root / "report", plots=options.get("plots", True))
    (root / "planning.json").write_text(
        json.dumps(
            {
                **choice,
                "execution_gib": execution,
                "spill_gib": spill,
                "skipped": inventory["skipped"],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    run(sys.argv[1])
