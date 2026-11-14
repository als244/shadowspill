"""Record deterministic planner decisions and preserve source semantics."""

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from shadowspill.planner import SearchOptions, plan_program
from shadowspill.planner.annotated_plan import AnnotatedProgramPlan
from shadowspill.planner.program_inputs import ShadowSpillPlanningProblem

from .io import write


def trace(args):
    planning = args.input.resolve()
    plans = json.loads((planning / "plans.json").read_text())
    source = Path(plans["source"])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.all_candidates and args.candidates_json is not None:
        raise ValueError("--all-candidates and --candidates-json are alternatives")
    entries = (
        [entry for entry in plans["programs"] if "error" not in entry]
        if args.programs == "all"
        else [plans["programs"][plans["winner"]]]
    )
    collection: dict[str, Any] = {
        "source": str(planning),
        "directories": [],
        "programs": [],
    }
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
