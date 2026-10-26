"""CPU search over exported, measured planner inputs."""

import json
import time
from typing import Any

from shadowspill.planner import GenericPlanningOptions, SearchOptions, plan_program
from shadowspill.planner.program_inputs import ShadowSpillPlanningProblem

from .budgets import execution_budget
from .io import write


def plan(args):
    source = args.input.resolve()
    dump = json.loads((source / "dump.json").read_text())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    options = SearchOptions(
        generic=GenericPlanningOptions(deterministic=True), workers=args.workers
    )
    execution, physical = execution_budget(
        dump, execution_gib=args.execution_gib, physical_gib=args.physical_gib
    )
    spill = (
        int(args.spill_gib * 2**30)
        if args.spill_gib is not None
        else dump["spill_budget_bytes"]
    )
    records: list[dict[str, Any]] = []
    winner = None
    for entry in dump["programs"]:
        record = dict(entry)
        started = time.perf_counter()
        problem = ShadowSpillPlanningProblem.from_json(
            (source / entry["problem"]).read_text()
        )
        try:
            selected = plan_program(
                problem,
                execution_budget=execution,
                spill_budget=spill,
                transfer_bandwidths=args.transfer_bandwidths,
                search_options=options,
                artifact_store=output / "artifact_store",
                verbose=True,
            )
            write(output / f"{entry['name']}.plan.json", selected.to_dict())
            record.update(
                makespan_ns=selected.result.simulation.makespan_ns,
                schedule_digest=selected.result.schedule.digest,
                selection_id=selected.result.diagnostics.selected_selection_id,
                candidate_id=selected.result.diagnostics.selected_candidate_id,
            )
            if winner is None or record["makespan_ns"] < records[winner]["makespan_ns"]:
                winner = len(records)
        except RuntimeError as error:
            record["error"] = f"{type(error).__name__}: {error}"
        record["wall_seconds"] = time.perf_counter() - started
        records.append(record)
        print(
            "PLAN",
            {
                key: record.get(key)
                for key in (
                    "name",
                    "geometry",
                    "ordering",
                    "makespan_ns",
                    "candidate_id",
                    "error",
                    "wall_seconds",
                )
            },
            flush=True,
        )
        # Atomic progress evidence, including rejected programs.
        temporary = output / "plans.json.tmp"
        temporary.write_text(
            json.dumps(
                {
                    "source": str(source),
                    "programs": records,
                    "infeasible_geometries": dump.get("infeasible_geometries", []),
                    "winner": winner,
                    "execution_budget_bytes": execution,
                    "physical_budget_bytes": physical,
                    "spill_budget_bytes": spill,
                    "transfer_bandwidths_override": (
                        args.transfer_bandwidths.to_dict()
                        if args.transfer_bandwidths is not None
                        else None
                    ),
                    "search_options": options.to_dict(),
                },
                indent=2,
            )
            + "\n"
        )
        temporary.replace(output / "plans.json")
    if winner is None:
        raise RuntimeError("none of the supplied programs fits")
