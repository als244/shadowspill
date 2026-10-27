"""Read completed phase artifacts; preserve exact schedules and physical layouts."""

from dataclasses import replace
from pathlib import Path

from benchmarking.quickstart.phases.common import read
from benchmarking.quickstart.phases.search import search_options
from shadowspill import Plan
from shadowspill.planner.search.algorithms.pressurefit.tracing import record_plan_trace
from shadowspill.planner.search.toolkit.resolution import recompute_share

from ..collect.inputs import labels
from ..data import Trajectory, annotate_recompute
from ..matrix import file_digest, write
from ..profile_winners.selection import completed
from .plan import diagnostic_plan


def prepare(quickstart, output, *, rank=0, workload=None):
    root, output = Path(quickstart).resolve(), Path(output).resolve()
    request, index, winners = completed(root)
    if not 0 <= rank < index.get("world_size", 1):
        raise ValueError("rank is outside the saved Program")
    options = search_options(request)
    manifest = {
        "source": str(root),
        "rank": rank,
        "workload": workload,
        "cases": [],
        "complete": False,
    }
    for winner in winners:
        plan_path = root / "plans" / winner["file"]
        execution_path = root / "execution" / plan_path.stem / "result.json"
        fingerprint = {
            "plan_sha256": file_digest(plan_path),
            "execution_sha256": file_digest(execution_path),
            "search_options": options.to_dict(),
            "rank": rank,
            "workload": workload,
        }
        # Use integer bytes in paths; tiny fixtures and nearby budgets must not
        # collide through floating-point formatting of a GiB value.
        relative = Path(f"budget_{winner['budget']}bytes")
        case = output / relative
        if (case / "case.json").exists():
            record = read(case / "case.json")
            if record["fingerprint"] != fingerprint:
                raise ValueError("winner changed; use a new output directory")
        else:
            record = prepare_case(
                Plan.load(plan_path),
                read(execution_path),
                winner,
                case,
                options=options,
                rank=rank,
                workload=workload,
            )
            record.update(path=str(relative), fingerprint=fingerprint, status="ready")
            write(case / "case.json", record)
        manifest["cases"].append(record)
        write(output / "matrix.json", manifest)
    manifest["complete"] = True
    write(output / "matrix.json", manifest)
    return manifest


def prepare_case(plan, execution, winner, case, *, options, rank, workload):
    annotated, logical = diagnostic_plan(plan, options)
    case.mkdir(parents=True, exist_ok=True)
    # Rendering uses the saved certificate and physical clock. Cut replay
    # reproduces the original search candidate using its logical clock.
    (case / "plan.json").write_text(annotated.to_json() + "\n")
    write(case / "execution.json", execution)
    trace = record_plan_trace(
        logical,
        options=replace(
            options,
            generic=replace(options.generic, deterministic=True),
            workers=1,
            trace_directory=str(case / "traces"),
        ),
        placement=plan._admission_facts,
        progress=lambda message: print(message, flush=True),
    )
    semantics = labels(plan.program, logical.program.digest)
    profiles = {p.profile_id: p.runtime_ns for p in logical.program.profiles}
    tasks = [
        {"task_id": t.task_id, "profiled_duration_ns": profiles[t.profile_id]}
        for t in logical.program.selected_tasks(plan.selections)
    ]
    annotate_recompute(tasks, logical.program.to_dict())
    for task in tasks:
        semantics["tasks"][task["task_id"]].update(task)
    write(case / "semantics.json", semantics)
    write(trace / "semantic_labels.json", semantics)
    ordering = plan.program.bindings.get("ordering", {})
    record = {
        "rank": rank,
        "geometry": winner["candidate"],
        "microbatches": ordering.get("depth", 1) * ordering.get("breadth", 1),
        "ordering": winner["ordering"],
        "resolution": str(recompute_share(logical.program, plan.selections)),
        "candidate_id": logical.diagnostics.selected_candidate_id,
        "makespan_ns": plan.timeline.makespan_ns,
        "search_makespan_ns": logical.simulation.makespan_ns,
        "schedule_digest": plan.schedule.digest,
        "physical_budget_bytes": plan.memory_budgets.execution_bytes,
        "physical_budget_gib": plan.memory_budgets.execution_bytes / 2**30,
        "execution_budget_bytes": plan.memory_budgets.execution_bytes
        - plan.fixed_device_bytes,
        "spill_budget_bytes": plan.memory_budgets.spill_bytes,
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
    return record
