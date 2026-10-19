"""Dump on the accelerator; plan, trace and export in separate CPU processes.

dump OUT -- <ordinary quickstart preset/factory flags>
plan DUMP --output OUT [--execution-gib N --spill-gib N]
trace PLAN --output OUT [--programs all --all-candidates | --candidates-json FILE]
export TRACE_DIRECTORY --output OUT

The directory arguments to plan and trace contain dump.json and plans.json.
An export contains one npz and one annotated JSON record per selected policy.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from shadowspill.diagnostics.pressurefit_trace import PressureFitTrace, read_manifest
from shadowspill.planner import GenericPlanningOptions, SearchOptions, plan_program
from shadowspill.planner.annotated_plan import AnnotatedProgramPlan
from shadowspill.planner.program_inputs import (
    ShadowSpillPlanningProblem,
    TransferBandwidths,
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def dump(args):
    # Reuse the quickstart's model import, initialization and factory protocol.
    # No model name or geometry is baked into this collector.
    from benchmarking.quickstart.cli import resolve_request
    from benchmarking.quickstart.options import parse_arguments
    from benchmarking.quickstart.runner import Tour, open_runtime, plan_budgets
    from benchmarking.quickstart.storage import Ledger, RunPaths
    from shadowspill.pytorch import build_step_programs
    from shadowspill.pytorch.failures import device_exhausted
    from shadowspill.pytorch.state.registry import registry_for
    from shadowspill.pytorch.step_search.sweep import _refuse_state_left_behind
    from shadowspill.search import default_orderings
    from shadowspill.step import StepDataOrdering

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    previous_argv = sys.argv
    sys.argv = [sys.argv[0], *args.quickstart]
    try:
        parser, arguments = parse_arguments()
    finally:
        sys.argv = previous_argv
    request, setup = resolve_request(parser, arguments)
    if request.run_budgets or arguments.distributed:
        raise ValueError("dump needs search-only, single-device quickstart flags")
    ledger = Ledger()
    paths = RunPaths(
        output, output / "artifact_store", arguments.build_store, output / "plan_store"
    )
    records = []
    failures = []
    runtime = open_runtime(
        request, ledger, host_headroom_gib=arguments.host_headroom_gib
    )
    try:
        budgets = plan_budgets(runtime, request)
        experiment = setup(device=request.device)
        context = experiment.get("context", contextlib.nullcontext)
        with context():
            tour = Tour(arguments, request, paths, budgets, runtime, ledger, experiment)
            try:
                for geometry, microbatches in tour.candidates.items():
                    orderings = (
                        default_orderings(len(microbatches))
                        if arguments.orderings == "factors"
                        else (StepDataOrdering.depth_first(len(microbatches)),)
                    )
                    started = time.perf_counter()
                    held = tuple(registry_for(runtime).values())
                    earlier_failure = runtime.last_failure
                    try:
                        steps = build_step_programs(
                            tour.model,
                            objective=tour.objective,
                            optimizer=tour.optimizer,
                            hyperparams=tuple(tour.hyperparams),
                            example_inputs=microbatches,
                            runtime=runtime,
                            execution="execution",
                            spill="spill",
                            execution_device=request.device,
                            optimizer_ordering="stage_interleaved",
                            orderings=orderings,
                            artifact_store=paths.store,
                            build_store=paths.build_store,
                            build_store_mode="reuse",
                            export_bypass_key=arguments.export_bypass_key,
                            profiling_options=tour.profiling,
                            graph_pair_variants=tour.variants,
                            **tour.plan_options,
                        )
                    except Exception as error:
                        if not device_exhausted(error):
                            raise
                        _refuse_state_left_behind(
                            runtime, held, geometry, len(microbatches), error
                        )
                        failure = runtime.last_failure
                        record = {
                            "geometry": geometry,
                            "microbatches": len(microbatches),
                            "orderings": [ordering.label for ordering in orderings],
                            "status": "profiling_infeasible",
                            "error": f"{type(error).__name__}: {error}",
                            "physical_capacity_bytes": request.physical_capacity,
                            "build_seconds": time.perf_counter() - started,
                            "allocator": (
                                failure.as_dict()
                                if failure is not None
                                and failure is not earlier_failure
                                else None
                            ),
                        }
                        failures.append(record)
                        write(output / f"infeasible_{len(failures):04d}.json", record)
                        print(
                            "SKIP", geometry, "profiling capacity exceeded", flush=True
                        )
                        continue
                    for ordering, step in zip(orderings, steps, strict=True):
                        name = f"program_{len(records):04d}"
                        write(output / f"{name}.problem.json", step.problem.to_dict())
                        write(output / f"{name}.step_program.json", step.to_dict())
                        write(
                            output / f"{name}.semantics.json",
                            json.loads(step.semantic_labels_json),
                        )
                        records.append(
                            {
                                "name": name,
                                "geometry": geometry,
                                "microbatches": len(microbatches),
                                "ordering": ordering.label,
                                "program_digest": step.problem.program.digest,
                                "problem": f"{name}.problem.json",
                                "semantics": f"{name}.semantics.json",
                                "build_seconds": time.perf_counter() - started,
                            }
                        )
                        print(
                            "DUMP",
                            geometry,
                            ordering.label,
                            step.problem.program.digest,
                            flush=True,
                        )
            finally:
                tour.close()
        write(
            output / "dump.json",
            {
                "schema": "pressurefit_collection/v1",
                "programs": records,
                "infeasible_geometries": failures,
                "execution_budget_bytes": min(budgets.search),
                "spill_budget_bytes": request.spill_budget,
                "physical_budget_bytes": min(request.search_budgets),
                "physical_capacity_bytes": request.physical_capacity,
                "budget_mapping": [
                    {"physical_bytes": physical, "execution_bytes": execution}
                    for physical, execution in budgets.planned.items()
                ],
                "workload": request.label,
                "metadata": tour.metadata,
                "quickstart": args.quickstart,
            },
        )
    finally:
        runtime.close()


def execution_budget(dump, *, execution_gib=None, physical_gib=None):
    """Resolve a physical cap with the capture's measured runtime carve-out."""
    if execution_gib is not None and physical_gib is not None:
        raise ValueError("execution and physical budget overrides are alternatives")
    if execution_gib is not None:
        execution = int(execution_gib * 2**30)
        physical = None
    elif physical_gib is not None:
        physical = int(physical_gib * 2**30)
        fixed = dump["physical_budget_bytes"] - dump["execution_budget_bytes"]
        execution = physical - fixed
    else:
        execution = dump["execution_budget_bytes"]
        physical = dump.get("physical_budget_bytes")
    if execution <= 0:
        raise ValueError("budget leaves no execution capacity after runtime overhead")
    return execution, physical


def transfer_bandwidths(value):
    """Parse decimal GB/s without changing the captured transfer latencies."""
    try:
        rates = tuple(int(float(part.strip()) * 1e9) for part in value.split(","))
        if len(rates) == 2:
            rates = (rates[0], rates[0], rates[1], rates[1])
        if len(rates) != 4:
            raise ValueError("expected two or four rates")
        return TransferBandwidths(
            *rates,
            provenance=f"tools.pressurefit_trace plan --transfer-bandwidths {value}",
        )
    except (ValueError, OverflowError) as error:
        raise argparse.ArgumentTypeError(
            "use positive finite GB/s rates: FETCH,EVICT or "
            "FETCH_SOLO,FETCH_CONCURRENT,EVICT_SOLO,EVICT_CONCURRENT"
        ) from error


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
    records = []
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


def trace(args):
    planning = args.input.resolve()
    plans = json.loads((planning / "plans.json").read_text())
    source = Path(plans["source"])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.all_candidates and args.candidates_json is not None:
        raise ValueError("--all-candidates and --candidates-json are alternatives")
    entries = (
        plans["programs"]
        if args.programs == "all"
        else [plans["programs"][plans["winner"]]]
    )
    collection = {"source": str(planning), "directories": [], "programs": []}
    for entry in entries:
        problem = ShadowSpillPlanningProblem.from_json(
            (source / entry["problem"]).read_text()
        )
        reference = AnnotatedProgramPlan.from_json(
            (planning / f"{entry['name']}.plan.json").read_text()
        )
        pairs = (
            None
            if args.candidates_json is None
            else tuple(
                tuple(item) for item in json.loads(args.candidates_json.read_text())
            )
        )
        if args.all_candidates:
            pairs = tuple(
                (resolution.selection_id, candidate.candidate_id)
                for resolution in reference.result.diagnostics.resolved_programs
                for candidate in resolution.candidate_evaluations
            )
        destination = output / entry["name"]
        options = replace(
            SearchOptions.from_dict(plans["search_options"], "search_options"),
            trace_directory=str(destination),
            trace_candidates=pairs,
            workers=1,
        )
        started = time.perf_counter()
        selected = plan_program(
            problem,
            execution_budget=plans["execution_budget_bytes"],
            spill_budget=plans["spill_budget_bytes"],
            transfer_bandwidths=reference.transfer_bandwidths,
            search_options=options,
            artifact_store=planning / "artifact_store",
            verbose=True,
        )
        assert selected.result.schedule.digest == reference.result.schedule.digest
        assert (
            selected.result.simulation.makespan_ns
            == reference.result.simulation.makespan_ns
        )
        directories = [path.parent for path in destination.glob("*/manifest.json")]
        metadata = json.loads((source / entry["semantics"]).read_text())
        for directory in directories:
            if metadata:
                assert metadata["program_digest"] == problem.program.digest
                write(directory / "semantic_labels.json", metadata)
            capture = json.loads((source / "dump.json").read_text())
            write(
                directory / "collection_context.json",
                {
                    "program_digest": problem.program.digest,
                    "workload": capture.get("workload"),
                    "metadata": capture.get("metadata", {}),
                    "geometry": entry.get("geometry"),
                    "microbatches": entry.get("microbatches"),
                    "ordering": entry.get("ordering"),
                    "physical_budget_bytes": plans.get("physical_budget_bytes"),
                    "execution_budget_bytes": plans["execution_budget_bytes"],
                    "spill_budget_bytes": plans["spill_budget_bytes"],
                    "transfer_bandwidths": reference.transfer_bandwidths.to_dict(),
                },
            )
        record = {
            "name": entry["name"],
            "geometry": entry.get("geometry"),
            "ordering": entry.get("ordering"),
            "candidates": len(pairs) if pairs else 1,
            "wall_seconds": time.perf_counter() - started,
        }
        collection["directories"].extend(str(p) for p in directories)
        collection["programs"].append(record)
        temporary = output / "collection.json.tmp"
        temporary.write_text(json.dumps(collection, indent=2) + "\n")
        temporary.replace(output / "collection.json")
        print("TRACE", record, flush=True)


def export(args):
    directory = args.input.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    collection = directory / "collection.json"
    if collection.is_file():
        records = []
        for path in json.loads(collection.read_text())["directories"]:
            trace_directory = Path(path)
            destination = output / trace_directory.parent.name
            destination.mkdir()
            records.append(export_directory(trace_directory, destination))
        write(
            output / "collection.json", {"source": str(directory), "programs": records}
        )
    else:
        export_directory(directory, output)


def export_directory(directory, output):
    manifest = read_manifest(directory)
    summary = {
        "source": str(directory),
        "candidates": [],
        "reductions": 0,
        "cuts": 0,
        "bytes": 0,
    }
    semantics = directory / "semantic_labels.json"
    if semantics.is_file():
        write(output / "semantic_labels.json", json.loads(semantics.read_text()))
        summary["semantic_labels"] = "semantic_labels.json"
    # Shared resolution metadata is labelled once. Candidate JSON references it;
    # do not expand the same large alias table for every policy in the export.
    for index, context in enumerate(manifest.get("contexts", ())):
        write(output / f"context_{index:04d}.json", context)
    for index, candidate in enumerate(manifest["candidates"]):
        trace = PressureFitTrace.read(directory / candidate["file"])
        started = time.perf_counter()
        trace.verify()
        arrays = {
            name: getattr(trace, name)
            for name in (
                "header",
                "alias_sizes",
                "seed",
                "initial",
                "step_reductions",
                "steps",
                "outcome",
            )
        }
        for reduction, value in enumerate(trace.reductions):
            for name in ("changes", "cuts", "pressure_ops", "outcome"):
                arrays[f"reduction_{reduction:04d}_{name}"] = getattr(value, name)
            if value.reason is not None:
                arrays[f"reduction_{reduction:04d}_reason"] = value.reason
        np.savez_compressed(output / f"candidate_{index:04d}.npz", **arrays)
        candidate.update(
            reductions=len(trace.reductions),
            cuts=sum(len(r.cuts) for r in trace.reductions),
            answer_step=trace.answer_step,
            answer_reduction=trace.answer_reduction,
            reduction_reasons=[
                trace.reduction_reason(rid) for rid in range(len(trace.reductions))
            ],
            bytes=(directory / candidate["file"]).stat().st_size,
            verification_seconds=time.perf_counter() - started,
        )
        metadata = dict(candidate)
        if "context_index" in metadata:
            metadata.pop("aliases")
            metadata.pop("task_ids")
            metadata["context_file"] = f"context_{metadata['context_index']:04d}.json"
        # Boundary labels are derived from tasks and sidecar metadata by the reader.
        metadata.pop("boundaries")
        write(output / f"candidate_{index:04d}.json", metadata)
        summary["candidates"].append(
            {
                key: candidate[key]
                for key in (
                    "candidate_id",
                    "context_index",
                    "reductions",
                    "cuts",
                    "bytes",
                    "verification_seconds",
                )
                if key in candidate
            }
            | {
                "metadata": f"candidate_{index:04d}.json",
                "arrays": f"candidate_{index:04d}.npz",
            }
        )
        for key in ("reductions", "cuts", "bytes"):
            summary[key] += candidate[key]
        print(
            "EXPORT",
            {
                key: candidate[key]
                for key in (
                    "candidate_id",
                    "reductions",
                    "cuts",
                    "bytes",
                    "verification_seconds",
                )
            },
            flush=True,
        )
    write(output / "manifest.json", summary)
    return summary


def add_commands(modes):
    dumping = modes.add_parser(
        "dump", help="Capture profiled quickstart programs on the accelerator"
    )
    dumping.add_argument("output", type=Path)
    dumping.add_argument("quickstart", nargs=argparse.REMAINDER)
    dumping.set_defaults(run=dump)
    planning = modes.add_parser(
        "plan", help="Plan every captured program without an accelerator"
    )
    planning.add_argument("input", type=Path)
    planning.add_argument("--output", type=Path, required=True)
    budgets = planning.add_mutually_exclusive_group()
    budgets.add_argument(
        "--execution-gib", type=float, help="Usable planner slab, excluding overhead"
    )
    budgets.add_argument(
        "--physical-gib",
        type=float,
        help="Physical device cap; subtract the captured runtime/headroom carve-out",
    )
    planning.add_argument("--spill-gib", type=float)
    planning.add_argument(
        "--transfer-bandwidths",
        type=transfer_bandwidths,
        help="Override decimal GB/s: FETCH,EVICT or "
        "FETCH_SOLO,FETCH_CONCURRENT,EVICT_SOLO,EVICT_CONCURRENT; "
        "retain captured task profiles and transfer latencies",
    )
    planning.add_argument("--workers", type=int, default=0)
    planning.set_defaults(run=plan)
    tracing = modes.add_parser(
        "trace", help="Record the winner, explicit pairs, or all candidates"
    )
    tracing.add_argument("input", type=Path)
    tracing.add_argument("--output", type=Path, required=True)
    selection = tracing.add_mutually_exclusive_group()
    selection.add_argument(
        "--candidates-json",
        type=Path,
        help="JSON list of [selection_id, candidate_id] pairs",
    )
    tracing.add_argument("--programs", choices=("winner", "all"), default="winner")
    selection.add_argument(
        "--all-candidates",
        action="store_true",
        help="Record every resolution x policy in each selected program",
    )
    tracing.set_defaults(run=trace)
    exporting = modes.add_parser(
        "export", help="Verify and export traces to annotated JSON / NumPy"
    )
    exporting.add_argument("input", type=Path)
    exporting.add_argument("--output", type=Path, required=True)
    exporting.set_defaults(run=export)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    add_commands(modes)
    args = parser.parse_args()
    if args.mode == "dump" and args.quickstart[:1] == ["--"]:
        args.quickstart = args.quickstart[1:]
    args.run(args)


if __name__ == "__main__":
    main()
