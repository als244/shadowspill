"""Prepare video inputs from the exact winners executed by quickstart.

python -m tools.pressurefit_trace.winners QUICKSTART --output OUTPUT
Then pass OUTPUT to tools.pressurefit_trace.matrix render.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

from shadowspill.planner.admission.refinement import placement_facts
from shadowspill.planner.annotated_plan import AnnotatedProgramPlan
from shadowspill.planner.diagnostics import INCUMBENT_CANDIDATE_ID
from shadowspill.planner.search.algorithms.pressurefit.tracing import record_plan_trace
from shadowspill.simulator import simulate

from .data import Trajectory, annotate_recompute
from .matrix import file_digest, write


def logical_trace_result(plan):
    """Replay logical timings rather than using the physical certificate's clock."""
    result = plan.result
    diagnostics = result.diagnostics
    if diagnostics.selected_candidate_id == INCUMBENT_CANDIDATE_ID:
        matches = [
            (parent.selection_id, candidate.candidate_id)
            for parent in diagnostics.resolved_programs
            for candidate in parent.candidate_evaluations
            if candidate.schedule_digest == result.schedule.digest
        ]
        if not matches:
            raise ValueError(
                "winner was carried from an earlier search; its original cut "
                "history must be recovered from the saved planning store"
            )
        diagnostics = replace(
            diagnostics,
            selected_selection_id=matches[0][0],
            selected_candidate_id=matches[0][1],
        )
    simulation = simulate(
        result.program,
        result.schedule,
        selections=result.selections,
        config=result.simulation_config,
    )
    return replace(
        result,
        simulation=simulation,
        diagnostics=diagnostics.replace_selected_makespan(simulation.makespan_ns),
    )


def trace_origin(plan, source):
    """Recover a carried winner's original search from saved budget incumbents."""
    try:
        return plan, logical_trace_result(plan), None
    except ValueError:
        if plan.result.diagnostics.selected_candidate_id != INCUMBENT_CANDIDATE_ID:
            raise
    task_ids = tuple(
        task.task_id
        for task in plan.result.program.selected_tasks(plan.result.selections)
    )
    for path in sorted((Path(source) / "search-winners").glob("*.plan.json")):
        origin = AnnotatedProgramPlan.from_json(path.read_text())
        if origin.result.schedule.digest != plan.result.schedule.digest:
            continue
        if (
            tuple(
                task.task_id
                for task in origin.result.program.selected_tasks(
                    origin.result.selections
                )
            )
            != task_ids
        ):
            continue
        try:
            logical = logical_trace_result(origin)
        except ValueError:
            continue
        return origin, logical, path
    raise ValueError(
        "carried winner has no matching original search in search-winners; "
        "recover its origin from the saved plan store before rendering"
    )


def prepare(quickstart, output, *, rank=0, workload=None):
    quickstart, output = Path(quickstart).resolve(), Path(output).resolve()
    source = quickstart / f"rank-{rank:05d}"
    if not source.is_dir():
        source = quickstart
    executed = sorted((source / "executed").glob("*bytes.json"))
    if not executed:
        raise ValueError(f"no executed winner records in {source}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "pressurefit_winner_videos/v1",
        "source": str(source),
        "rank": rank,
        "cases": [],
        "complete": False,
    }
    for entry in executed:
        execution = json.loads(entry.read_text())
        if execution["status"] != "executed":
            raise ValueError(f"winner did not finish execution: {entry}")
        budget = execution["execution_budget_bytes"]
        bundle = source / "winners" / f"{budget}bytes"
        identity = json.loads((bundle / "bundle.json").read_text())
        plan = AnnotatedProgramPlan.from_json((bundle / "plan.json").read_text())
        if (
            plan.result.schedule.digest != identity["schedule_digest"]
            or identity["capture_identity"] != execution["capture_identity"]
        ):
            raise ValueError(f"export does not identify the executed plan: {bundle}")
        physical = execution["requested_execution_budget_bytes"]
        if len(physical) != 1:
            raise ValueError("winner must have one unambiguous physical budget")
        relative = Path(f"budget_{physical[0] / 2**30:g}GiB")
        case = output / relative
        fingerprint = {
            "plan_sha256": file_digest(bundle / "plan.json"),
            "search_plan_sha256": file_digest(bundle / "search-plan.json"),
            "execution_sha256": file_digest(entry),
        }
        if (case / "case.json").exists():
            record = json.loads((case / "case.json").read_text())
            if record["fingerprint"] != fingerprint:
                raise ValueError("winner changed; use a new output directory")
            manifest["cases"].append(record)
            continue
        case.mkdir(parents=True, exist_ok=True)
        for name in (
            "plan.json",
            "search-plan.json",
            "search-logical.json",
            "semantics.json",
            "bundle.json",
        ):
            shutil.copy2(bundle / name, case / name)
        shutil.copy2(entry, case / "execution.json")
        trace_plan = AnnotatedProgramPlan.from_json(
            (case / "search-plan.json").read_text()
        )
        if trace_plan.result.schedule.digest != plan.result.schedule.digest:
            raise ValueError("search history and executed schedule disagree")
        trace_plan, logical, origin_path = trace_origin(trace_plan, source)
        if origin_path is not None:
            shutil.copy2(origin_path, case / "trace-origin.plan.json")
        trace = record_plan_trace(
            logical,
            options=replace(
                logical.search_options, workers=1, trace_directory=str(case / "traces")
            ),
            placement=placement_facts(
                trace_plan.effective_facts,
                scratch_reserve_bytes=trace_plan.program.dynamic_scratch_reserve_bytes,
            ),
            progress=lambda message: print(message, flush=True),
        )
        semantics = json.loads((case / "semantics.json").read_text())
        # Fixed distributed programs omit the unused backward alternative.
        # Preserve its measured duration in labels for the lime recompute prefix.
        profiles = {p.profile_id: p.runtime_ns for p in plan.result.program.profiles}
        tasks = [
            {"task_id": t.task_id, "profiled_duration_ns": profiles[t.profile_id]}
            for t in plan.result.program.selected_tasks(plan.result.selections)
        ]
        annotate_recompute(tasks, plan.result.program.to_dict())
        for task in tasks:
            semantics.setdefault("tasks", {}).setdefault(task["task_id"], {}).update(
                task
            )
        semantics["program_digest"] = logical.program.digest
        write(trace / "semantic_labels.json", semantics)
        from shadowspill.planner.search.toolkit.resolution import recompute_share

        record = {
            "path": str(relative),
            "fingerprint": fingerprint,
            "status": "ready",
            "rank": rank,
            "geometry": execution["candidate"],
            "microbatches": execution["ordering"]["depth"]
            * execution["ordering"]["breadth"],
            "ordering": execution["ordering_label"],
            "resolution": str(
                recompute_share(plan.result.program, plan.result.selections)
            ),
            "candidate_id": logical.diagnostics.selected_candidate_id,
            "makespan_ns": plan.simulation.makespan_ns,
            "search_makespan_ns": logical.simulation.makespan_ns,
            "trace_origin": None if origin_path is None else str(origin_path),
            "schedule_digest": plan.result.schedule.digest,
            "physical_budget_bytes": physical[0],
            "physical_budget_gib": physical[0] / 2**30,
            "execution_budget_bytes": budget,
            "spill_budget_bytes": execution["spill_budget_bytes"],
            "plan": "plan.json",
            "semantics": "semantics.json",
            "trace": str(trace.relative_to(case)),
            "measured_seconds": execution["median_seconds"],
        }
        write(
            trace / "collection_context.json",
            {
                **record,
                "program_digest": logical.program.digest,
                "workload": workload,
                "metadata": {"representative_rank": rank},
            },
        )
        trajectory = Trajectory.load(trace)
        trajectory.trace.verify()
        record.update(
            answer_reduction=trajectory.trace.answer_reduction,
            reductions=len(trajectory.trace.reductions),
            cuts_per_reduction=[len(r.cuts) for r in trajectory.trace.reductions],
        )
        write(case / "metadata.json", {**record, "workload": workload})
        write(case / "case.json", record)
        manifest["cases"].append(record)
        write(output / "matrix.json", manifest)
    manifest["complete"] = True
    write(output / "matrix.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("quickstart", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--workload")
    args = parser.parse_args(argv)
    prepare(args.quickstart, args.output, rank=args.rank, workload=args.workload)


if __name__ == "__main__":
    main()
