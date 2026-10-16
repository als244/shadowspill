"""Record selected policies after the ordinary search, outside the plan key."""

from __future__ import annotations

import json
import tempfile
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from shadowspill.diagnostics.occupancy import ROLE_PRIORITY, ProgramFacts
from shadowspill.planner.admission import AdmissionFacts
from shadowspill.planner.diagnostics import INCUMBENT_CANDIDATE_ID
from shadowspill.planner.result import ProgramPlanResult
from shadowspill.planner.search import SearchOptions
from shadowspill.planner.search.toolkit.resolution import recompute_share

from .candidates import CProblemResult, decode_candidate_diagnostic
from .options import PressureFitOptions
from .search import (
    SelectionProblem,
    _materialize,
    _selection_id,
    build_problems,
    run_problems,
)


def _pressure_accounting(
    compiled: SelectionProblem, replay: CProblemResult
) -> dict[str, Any]:
    """Save the fixed resident reservations, checked against the C slice size."""
    program = compiled.indexed_template.program
    layout = compiled.indexed_placement or compiled.indexed_admission
    alignment = max(1, int(layout.value.minimum_alignment)) if layout else 1
    mutations = Counter(
        int(program.mutation_aliases[i]) for i in range(program.mutation_count)
    )
    cursor = 0
    reservations = []
    for alias in replay.resident_aliases:
        before = cursor
        generations = 1 + mutations[alias]
        for _ in range(generations):
            cursor = ((cursor + alignment - 1) // alignment) * alignment
            cursor += int(program.alias_size_bytes[alias])
        reservations.append([alias, generations, cursor - before])
    if cursor != replay.resident_slice_bytes:
        raise RuntimeError("resident reservations do not match the C slice size")
    return {
        "resident_aliases": replay.resident_aliases,
        "resident_slice_bytes": replay.resident_slice_bytes,
        "resident_reservations": reservations,
        "resident_alignment_bytes": alignment,
        "final_device_aliases": [
            int(program.final_aliases[i])
            for i in range(program.final_count)
            if program.final_locations[i] == 0
        ],
    }


def record_plan_trace(
    result: ProgramPlanResult,
    *,
    options: SearchOptions,
    admission: AdmissionFacts | None = None,
    placement: AdmissionFacts | None = None,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Record an isolated deterministic rerun, preserving the original answer."""
    from . import ordered_resolutions

    if options.trace_directory is None:
        raise ValueError("recording requires trace_directory")
    if not options.generic.deterministic:
        raise ValueError("automatic PressureFit tracing requires deterministic=True")
    if len(result.simulation_config.devices) != 1:
        raise ValueError("PressureFit tracing currently requires exactly one device")
    algorithm = options.resolved_algorithm.options
    if not isinstance(algorithm, PressureFitOptions):
        raise TypeError("PressureFit tracing requires PressureFitOptions")
    diagnostics = result.diagnostics
    pairs = options.trace_candidates
    if pairs is None:
        if diagnostics.selected_candidate_id == INCUMBENT_CANDIDATE_ID:
            raise ValueError(
                "the returned incumbent has no cut history in this search; "
                "set trace_candidates to record current candidates explicitly"
            )
        pairs = (
            (diagnostics.selected_selection_id, diagnostics.selected_candidate_id),
        )
    resolutions = {
        _selection_id(resolution): (index, resolution)
        for index, resolution in enumerate(
            ordered_resolutions(result.program, algorithm.resolution_options)
        )
    }
    candidates = {
        (parent.selection_id, candidate.candidate_id): candidate
        for parent in diagnostics.resolved_programs
        for candidate in parent.candidate_evaluations
    }
    for pair in pairs:
        if pair not in candidates or pair[0] not in resolutions:
            raise ValueError(f"trace candidate was not evaluated: {pair!r}")

    parent = Path(options.trace_directory).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="pressurefit-", dir=parent))
    manifest: dict[str, Any] = {
        "schema": "shadowspill.pressurefit_trace/v1",
        "status": "recording",
        "program_digest": result.program.digest,
        "program": "program.json",
        "search_options": options.to_dict(),
        "simulation_config": {
            "devices": [asdict(item) for item in result.simulation_config.devices],
            "spill_capacity_bytes": result.simulation_config.spill_capacity_bytes,
        },
        "answer": {
            "selection_id": diagnostics.selected_selection_id,
            "candidate_id": diagnostics.selected_candidate_id,
            "makespan_ns": result.simulation.makespan_ns,
            "schedule_digest": result.schedule.digest,
        },
        "contexts": [],
        "candidates": [],
    }

    def save_manifest() -> None:
        temporary = directory / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(directory / "manifest.json")

    program = result.program.to_dict()
    (directory / "program.json").write_text(json.dumps(program) + "\n")
    save_manifest()
    compiled_resolutions: dict[str, tuple[SelectionProblem, int]] = {}
    try:
        for ordinal, pair in enumerate(pairs):
            reference = candidates[pair]
            original_index, resolution = resolutions[pair[0]]
            cached = compiled_resolutions.get(pair[0])
            if cached is None:
                compiled = build_problems(
                    result.program,
                    result.initial_residency,
                    result.final_residency,
                    result.simulation_config,
                    admission,
                    placement=placement,
                    resolutions=(resolution,),
                    progress=None,
                )[0]
                context_index = len(manifest["contexts"])
                facts = ProgramFacts.build(
                    program,
                    [
                        {"group_id": item.group_id, "option_id": item.option_id}
                        for item in resolution
                    ],
                )
                objects: dict[str, list[dict[str, Any]]] = {}
                for fact in facts.objects.values():
                    objects.setdefault(fact.alias_group_id, []).append(asdict(fact))
                aliases = []
                for alias_id in compiled.indexed_template.alias_ids:
                    members = sorted(
                        objects.get(alias_id, []),
                        key=lambda member: (
                            ROLE_PRIORITY.get(member["role"], len(ROLE_PRIORITY)),
                            -member["size_bytes"],
                            member["object_id"],
                        ),
                    )
                    aliases.append(
                        {
                            "alias_id": alias_id,
                            "label": (
                                f"{members[0]['category']}: {members[0]['object_id']}"
                                if members
                                else alias_id
                            ),
                            "role": members[0]["role"] if members else "unattributed",
                            "category": members[0]["category"]
                            if members
                            else "unattributed",
                            "objects": members,
                        }
                    )
                manifest["contexts"].append(
                    {
                        "selection_id": pair[0],
                        "aliases": aliases,
                        "task_ids": compiled.indexed_template.task_ids,
                    }
                )
                compiled_resolutions[pair[0]] = compiled, context_index
            else:
                compiled, context_index = cached
            destination = directory / f"candidate_{ordinal:04d}"
            destination.mkdir()
            selection = replace(
                algorithm,
                residency_strategies=(reference.residency_strategy,),
                fetch_rules=(reference.fetch_rule,),
            )
            replay = run_problems(
                (compiled,),
                options.generic,
                selection,
                workers=1,
                trace_directory=str(destination),
            )[0]
            if replay is None or len(replay.candidates) != 1:
                raise RuntimeError("traced candidate did not produce its diagnostic")
            context = manifest["contexts"][context_index]
            if "pressure_accounting" not in context:
                context["pressure_accounting"] = _pressure_accounting(compiled, replay)
            actual = decode_candidate_diagnostic(
                replay.candidates[0],
                selection_id=pair[0],
                simulation=compiled.indexed_template,
            )
            if (
                actual.status != reference.status
                or actual.makespan_ns != reference.makespan_ns
                or actual.schedule_digest != reference.schedule_digest
            ):
                raise RuntimeError(
                    f"traced rerun differs from the original candidate: {pair!r}"
                )
            plan_makespan = None
            intervals = {}
            if replay.selected_schedule is not None:
                schedule, simulation, _ = _materialize(compiled, replay)
                plan_makespan = simulation.makespan_ns
                intervals = {item.task_id: item for item in simulation.task_intervals}
                if pair == (
                    diagnostics.selected_selection_id,
                    diagnostics.selected_candidate_id,
                ) and (
                    schedule.digest != result.schedule.digest
                    or simulation.makespan_ns != result.simulation.makespan_ns
                ):
                    raise RuntimeError(
                        "traced winner differs at the requested capacity"
                    )
            tasks_by_id = {task.task_id: task for task in result.program.tasks}
            profiles = {
                profile.profile_id: profile for profile in result.program.profiles
            }
            tasks = [
                {
                    "task_id": task_id,
                    "phase": tasks_by_id[task_id].phase,
                    "profile_id": tasks_by_id[task_id].profile_id,
                    "profiled_duration_ns": profiles[
                        tasks_by_id[task_id].profile_id
                    ].runtime_ns,
                    "plan_ready_ns": intervals[task_id].ready_ns
                    if task_id in intervals
                    else None,
                    "plan_start_ns": intervals[task_id].start_ns
                    if task_id in intervals
                    else None,
                    "plan_end_ns": intervals[task_id].end_ns
                    if task_id in intervals
                    else None,
                    "plan_duration_ns": (
                        intervals[task_id].end_ns - intervals[task_id].start_ns
                        if task_id in intervals
                        else None
                    ),
                    "plan_stall_ns": intervals[task_id].stall_ns
                    if task_id in intervals
                    else None,
                }
                for task_id in compiled.indexed_template.task_ids
            ]
            files = tuple(destination.glob("*.pftrace"))
            if len(files) != 1:
                raise RuntimeError("expected one trace file for the selected candidate")
            manifest["candidates"].append(
                {
                    "selection_id": pair[0],
                    "candidate_id": pair[1],
                    "original_problem_index": original_index,
                    "file_problem_index": 0,
                    "recompute_share": (
                        sum(item.option_id == "recompute" for item in resolution)
                        / len(resolution)
                        if resolution
                        else 0.0
                    ),
                    "resolution_recompute_fraction": str(
                        recompute_share(result.program, resolution)
                    ),
                    "file": str(files[0].relative_to(directory)),
                    "context_index": context_index,
                    "tasks": tasks,
                    "plan_makespan_ns": plan_makespan,
                    "diagnostic": actual.to_dict(),
                    "verified_against_original": True,
                }
            )
            if progress is not None:
                progress(
                    f"PressureFit trace candidate {ordinal + 1}/{len(pairs)}: "
                    f"{reference.candidate_id}"
                )
        manifest["status"] = "complete"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save_manifest()
    if progress is not None:
        progress(f"PressureFit trace: {directory}")
    return directory
