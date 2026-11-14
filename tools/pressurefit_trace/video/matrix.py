"""Prepare video inputs for the best policy at every captured resolution."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import replace
from pathlib import Path

from shadowspill.ir import MemoryLocation
from shadowspill.planner import plan_program
from shadowspill.planner.admission.layout.build import build_fixed_layout_admission
from shadowspill.planner.admission.refinement import FixedLayoutAttempt, placement_facts
from shadowspill.planner.annotated_plan import AnnotatedProgramPlan
from shadowspill.planner.program_inputs import ShadowSpillPlanningProblem
from shadowspill.planner.result import certified_result
from shadowspill.planner.search.algorithms.pressurefit.tracing import record_plan_trace
from shadowspill.planner.search.toolkit.resolution import recompute_share

from ..data import Trajectory


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def slug(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "unknown"


def select_resolution(parent, resolution):
    """Return a certified plan and its original search result for cut tracing."""
    started = time.perf_counter_ns()
    diagnostic = next(
        item
        for item in parent.result.diagnostics.resolved_programs
        if item.selection_id == resolution.selection_id
    )
    result = replace(
        parent.result,
        schedule=resolution.schedule,
        selections=resolution.selections,
        simulation=resolution.simulation,
        resident_slice=resolution.resident_slice,
        resolutions=(),
        diagnostics=replace(
            parent.result.diagnostics,
            selected_candidate_id=diagnostic.selected_candidate_id,
            selected_selection_id=diagnostic.selection_id,
            selected_makespan_ns=diagnostic.selected_makespan_ns,
        ),
    )
    admission = build_fixed_layout_admission(
        result,
        parent.effective_facts,
        dynamic_alias_group_ids=frozenset(
            item.alias_group_id
            for item in result.schedule.final_residency
            if item.location is MemoryLocation.DEVICE
        ),
        scratch_reserve_bytes=parent.program.dynamic_scratch_reserve_bytes,
    )
    # The physical replay adds address-reuse dependencies. Its timings can
    # differ from the search simulation even with an identical schedule.
    # Cut recording must replay the original candidate, while plan movies
    # must use the certified physical simulation. Preserve both explicitly.
    logical_result = result
    result = certified_result(result, admission)
    elapsed = time.perf_counter_ns() - started
    facts = parent.effective_facts
    selected = replace(
        parent,
        result=result,
        fixed_layout=admission.layout,
        simulation_admission=admission.simulator_input,
        simulation=admission.simulation,
        attempts=(
            FixedLayoutAttempt(
                facts.object_capacity_bytes,
                facts.object_capacity_bytes,
                admission.layout.required_bytes,
                admission.layout.pool_capacity_bytes,
                True,
                physical_admission_wall_time_ns=elapsed,
                search_diagnostics=result.diagnostics,
            ),
        ),
        wall_time_ns=elapsed,
        plan_from_store=False,
        _digest_cache=[],
    )
    return selected, logical_result


def prepare(planning, output, *, dump=None, workers=4):
    planning, output = Path(planning).resolve(), Path(output).resolve()
    collection = json.loads((planning / "plans.json").read_text())
    source = Path(dump or collection["source"]).resolve()
    capture = json.loads((source / "dump.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "planning": str(planning),
        "source": str(source),
        "plans_sha256": file_digest(planning / "plans.json"),
        "dump_sha256": file_digest(source / "dump.json"),
    }
    manifest_path = output / "matrix.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["identity"] != identity:
            raise ValueError("matrix inputs changed; use a new output directory")
    manifest = {
        "schema": "pressurefit_video_matrix/v1",
        "identity": identity,
        "cases": [],
    }
    for entry in collection["programs"]:
        reference_path = planning / f"{entry['name']}.plan.json"
        if not reference_path.exists():
            manifest["cases"].append(
                {**entry, "status": "infeasible_program", "path": entry["name"]}
            )
            write(manifest_path, manifest)
            continue
        reference = AnnotatedProgramPlan.from_json(reference_path.read_text())
        options = replace(reference.result.search_options, workers=workers)
        if not options.generic.deterministic:
            raise ValueError(
                "matrix preparation requires a deterministic source search"
            )
        problem = ShadowSpillPlanningProblem.from_json(
            (source / entry["problem"]).read_text()
        )
        if problem.digest != reference.program.digest:
            raise ValueError(f"capture and saved plan disagree for {entry['name']}")
        parent = None
        used_names = set()
        for diagnostic in reference.result.diagnostics.resolved_programs:
            share = recompute_share(problem.program, diagnostic.choices)
            label = f"recompute_{share.numerator}" + (
                f"of{share.denominator}" if share.denominator != 1 else ""
            )
            if label in used_names:
                label += (
                    "_"
                    + hashlib.sha256(diagnostic.selection_id.encode()).hexdigest()[:8]
                )
            used_names.add(label)
            relative = (
                Path(f"seqs_{slug(entry.get('geometry', entry['name']))}")
                / f"order_{slug(entry.get('ordering', entry['name']))}"
                / label
            )
            case = output / relative
            fingerprint = {
                "plan_sha256": file_digest(reference_path),
                "selection_id": diagnostic.selection_id,
            }
            saved = case / "case.json"
            if saved.exists():
                record = json.loads(saved.read_text())
                if record["fingerprint"] == fingerprint and record["status"] in {
                    "ready",
                    "infeasible",
                }:
                    manifest["cases"].append(record)
                    write(manifest_path, manifest)
                    continue
            record = {
                "path": str(relative),
                "fingerprint": fingerprint,
                "program": entry["name"],
                "geometry": entry.get("geometry"),
                "microbatches": entry.get("microbatches"),
                "ordering": entry.get("ordering"),
                "resolution": str(share),
                "candidate_id": diagnostic.selected_candidate_id,
                "makespan_ns": diagnostic.selected_makespan_ns,
            }
            if diagnostic.selected_makespan_ns is None:
                record.update(
                    status="infeasible",
                    failures=[
                        {
                            "candidate_id": item.candidate_id,
                            "status": item.status,
                            "kind": item.failure_kind,
                            "detail": item.failure_detail,
                        }
                        for item in diagnostic.candidate_evaluations
                    ],
                )
            else:
                if parent is None:
                    parent = plan_program(
                        problem,
                        execution_budget=collection["execution_budget_bytes"],
                        spill_budget=collection["spill_budget_bytes"],
                        transfer_bandwidths=reference.transfer_bandwidths,
                        search_options=options,
                        artifact_store=output / "artifacts" / entry["name"],
                        keep_resolutions=True,
                        verbose=False,
                    )
                    if (
                        parent.result.schedule.digest
                        != reference.result.schedule.digest
                        or parent.simulation.makespan_ns
                        != reference.simulation.makespan_ns
                    ):
                        raise ValueError(
                            f"saved search did not reproduce: {entry['name']}"
                        )
                kept = next(
                    item
                    for item in parent.result.resolutions
                    if item.selection_id == diagnostic.selection_id
                )
                selected, logical_result = select_resolution(parent, kept)
                candidate = next(
                    item
                    for item in diagnostic.candidate_evaluations
                    if item.candidate_id == kept.candidate_id
                )
                if selected.result.schedule.digest != candidate.schedule_digest:
                    raise ValueError(
                        "retained schedule differs from original diagnostic"
                    )
                write(case / "plan.json", selected.to_dict())
                semantics = json.loads((source / entry["semantics"]).read_text())
                if semantics:
                    write(case / "semantics.json", semantics)
                trace_options = replace(options, trace_directory=str(case / "traces"))
                trace = record_plan_trace(
                    logical_result,
                    options=trace_options,
                    placement=placement_facts(
                        selected.effective_facts,
                        scratch_reserve_bytes=problem.dynamic_scratch_reserve_bytes,
                    ),
                )
                context = {
                    "program_digest": problem.program.digest,
                    "workload": capture.get("workload"),
                    "metadata": capture.get("metadata", {}),
                    "geometry": entry.get("geometry"),
                    "microbatches": entry.get("microbatches"),
                    "ordering": entry.get("ordering"),
                    "physical_budget_bytes": collection.get("physical_budget_bytes"),
                    "execution_budget_bytes": collection["execution_budget_bytes"],
                    "spill_budget_bytes": collection["spill_budget_bytes"],
                    "transfer_bandwidths": selected.transfer_bandwidths.to_dict(),
                }
                write(trace / "collection_context.json", context)
                if semantics:
                    write(trace / "semantic_labels.json", semantics)
                trajectory = Trajectory.load(trace)
                trajectory.trace.verify()
                record.update(
                    status="ready",
                    search_makespan_ns=diagnostic.selected_makespan_ns,
                    makespan_ns=selected.simulation.makespan_ns,
                    plan="plan.json",
                    semantics="semantics.json" if semantics else None,
                    trace=str(trace.relative_to(case)),
                    schedule_digest=selected.result.schedule.digest,
                    answer_reduction=trajectory.trace.answer_reduction,
                    reductions=len(trajectory.trace.reductions),
                    cuts_per_reduction=[
                        len(r.cuts) for r in trajectory.trace.reductions
                    ],
                )
                metadata = {**context, **record}
                if context["physical_budget_bytes"] is not None:
                    metadata["physical_budget_gib"] = (
                        context["physical_budget_bytes"] / 2**30
                    )
                write(case / "metadata.json", metadata)
            write(saved, record)
            manifest["cases"].append(record)
            write(manifest_path, manifest)
            print(f"{record['status'].upper()} {relative}", flush=True)
    manifest["complete"] = True
    write(manifest_path, manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser(
        "prepare", help="Keep and trace every resolution winner"
    )
    command.add_argument(
        "planning", type=Path, help="Collection directory with plans.json"
    )
    command.add_argument(
        "--dump", type=Path, help="Capture directory, if moved since planning"
    )
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("workers must be positive")
    prepare(args.planning, args.output, dump=args.dump, workers=args.workers)


if __name__ == "__main__":
    main()
