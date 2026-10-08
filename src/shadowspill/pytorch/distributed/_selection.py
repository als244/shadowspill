"""Common task choices with independent, physically admitted local schedules."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from functools import partial
from typing import Any

from shadowspill.errors import (
    AdmissionError,
    PlanInfeasibleError,
    PlanSearchExhaustedError,
)
from shadowspill.ir import MemoryLocation, ShadowSpillProgram, TaskAlternativeChoice
from shadowspill.planner import SearchOptions
from shadowspill.planner.admission import AdmissionFacts
from shadowspill.planner.admission.layout import (
    FixedLayoutAdmission,
    build_fixed_layout_admission,
)
from shadowspill.planner.admission.refinement import FixedLayoutSelection
from shadowspill.planner.annotated_plan import AnnotatedProgramPlan
from shadowspill.planner.diagnostics import TaskAlternativeChoiceDiagnostic
from shadowspill.planner.diagnostics.resolved_programs import ResolvedProgramDiagnostics
from shadowspill.planner.result import ProgramPlanResult, ResolutionPlan
from shadowspill.planner.search.toolkit.resolution import (
    DEFAULT_RESOLUTION_OPTIONS,
    resolutions,
)
from shadowspill.store import ArtifactStore, atomic_text

from . import current
from ._control import Control
from ._plan_exchange import pack, record_plan

type ChoiceKey = tuple[tuple[str, str], ...]
type LocalPlan = FixedLayoutSelection | AnnotatedProgramPlan


def fixed_program(
    program: ShadowSpillProgram, selection: Sequence[TaskAlternativeChoice]
) -> ShadowSpillProgram:
    return replace(
        program,
        tasks=program.selected_tasks(tuple(selection)),
        task_alternative_groups=(),
    )


def fixed_admission(
    admission: AdmissionFacts, program: ShadowSpillProgram
) -> AdmissionFacts:
    by_id = {item.task_id: item for item in admission.tasks}
    return replace(
        admission, tasks=tuple(by_id[task.task_id] for task in program.tasks)
    )


def _key(selection: Sequence[TaskAlternativeChoice]) -> ChoiceKey:
    return tuple((item.group_id, item.option_id) for item in selection)


def _candidate_keys(control, program, options, incumbent):
    """Agree on the search request and collect each rank's offered resolutions."""
    control.agree("selection/options", options.to_dict())
    control.agree(
        "selection/alternatives",
        [
            [group.group_id, [option.option_id for option in group.options]]
            for group in program.task_alternative_groups
        ],
    )
    shares = getattr(
        options.resolved_algorithm.options,
        "resolution_options",
        DEFAULT_RESOLUTION_OPTIONS,
    )
    proposed = [_key(value) for value in resolutions(program, shares)]
    if incumbent is not None:
        proposed.append(_key(incumbent.selections))
    offered = control.exchange("selection/candidates", proposed)
    return sorted(
        {tuple(tuple(pair) for pair in choice) for rank in offered for choice in rank}
    )


def _candidate_program(
    program: ShadowSpillProgram, key: ChoiceKey
) -> ShadowSpillProgram:
    return fixed_program(
        program, tuple(TaskAlternativeChoice(group, option) for group, option in key)
    )


def _attempt_candidate[R: (FixedLayoutSelection, AnnotatedProgramPlan)](
    key: ChoiceKey,
    *,
    program: ShadowSpillProgram,
    attempt: Callable[[ShadowSpillProgram, ProgramPlanResult | None], R],
    incumbent: ProgramPlanResult | None,
) -> R | dict[str, str]:
    fixed = _candidate_program(program, key)
    carried = (
        replace(incumbent, program=fixed, selections=(), resolutions=())
        if incumbent is not None and _key(incumbent.selections) == key
        else None
    )
    try:
        return attempt(fixed, carried)
    except (PlanInfeasibleError, PlanSearchExhaustedError, AdmissionError) as error:
        return {"error": str(error)}


def _shared_candidates[R: (FixedLayoutSelection, AnnotatedProgramPlan)](
    control: Control,
    candidates: Sequence[ChoiceKey],
    run: Callable[[ChoiceKey], R | dict[str, str]],
    receive: Callable[[ChoiceKey, dict[str, Any]], R],
) -> tuple[dict[ChoiceKey, R | dict[str, str]], dict[ChoiceKey, int]]:
    """Search each candidate on one owner, then admit it on every participant."""
    owners = {
        key: control.members[index % len(control.members)]
        for index, key in enumerate(candidates)
    }
    held: dict[ChoiceKey, R] = {}

    def search_owned():
        results = []
        for key in candidates:
            if owners[key] != control.rank:
                continue
            control.check_failure()
            record = run(key)
            if isinstance(record, dict):
                results.append({"choices": key, **record})
            else:
                held[key] = record
                results.append({"choices": key, "plan": pack(record)})
        return results

    payloads = control.run("selection/shared_search", search_owned)
    peers = control.exchange("selection/shared_results", payloads)

    def admit_received():
        restored = {}
        for rank, records in zip(control.members, peers, strict=True):
            for row in records:
                key = tuple(tuple(pair) for pair in row["choices"])
                if key not in owners or owners[key] != rank or key in restored:
                    raise ValueError("invalid shared candidate ownership")
                if "error" in row:
                    restored[key] = {"error": row["error"]}
                elif key in held:
                    restored[key] = held[key]
                else:
                    restored[key] = receive(key, row["plan"])
        if set(restored) != set(candidates):
            raise ValueError("shared candidate results are incomplete")
        return restored

    return control.run("selection/shared_admission", admit_received), owners


def _candidate_status(record: LocalPlan | dict[str, str]) -> dict[str, Any]:
    if isinstance(record, dict):
        return dict(record)
    simulation = (
        record.admission.simulation
        if isinstance(record, FixedLayoutSelection)
        else record.simulation
    )
    return {
        "ns": simulation.makespan_ns,
        "tasks": [[task.task_id, task.phase] for task in record.result.program.tasks],
    }


def choose[R: (FixedLayoutSelection, AnnotatedProgramPlan)](
    program: ShadowSpillProgram,
    attempt: Callable[[ShadowSpillProgram, ProgramPlanResult | None], R],
    *,
    search_options: SearchOptions | None = None,
    incumbent: ProgramPlanResult | None = None,
    receive: Callable[[dict[str, Any], ShadowSpillProgram], R] | None = None,
) -> tuple[R, ChoiceKey, dict[ChoiceKey, R], dict[str, Any]]:
    """Choose the lowest worst-rank time among locally admitted candidates.

    ``attempt`` uses the ordinary planner/admission path. A ``receive`` callback
    enables one-owner search; otherwise every rank searches each candidate.
    The same task-order agreement and winner selection follow either path.
    """
    bound = current()
    assert bound is not None
    control = bound.control
    candidates = _candidate_keys(
        control, program, search_options or SearchOptions(), incumbent
    )
    run = partial(
        _attempt_candidate, program=program, attempt=attempt, incumbent=incumbent
    )
    shared, owners = None, {}
    if receive is not None:
        shared, owners = _shared_candidates(
            control,
            candidates,
            run,
            lambda key, payload: receive(payload, _candidate_program(program, key)),
        )
    successful: dict[ChoiceKey, R] = {}
    scores: list[tuple[int, ChoiceKey]] = []
    evidence: list[dict[str, Any]] = []
    for index, key in enumerate(candidates):
        record = (
            control.run(f"selection/{index}/plan", partial(run, key))
            if shared is None
            else shared[key]
        )
        peers = control.exchange(
            f"selection/{index}/outcome", _candidate_status(record)
        )
        evidence.append(
            {
                "choices": key,
                "ranks": peers,
                **({"searched_by_rank": owners[key]} if owners else {}),
            }
        )
        if any("error" in peer for peer in peers):
            continue
        if any(peer["tasks"] != peers[0]["tasks"] for peer in peers):
            raise ValueError(
                "distributed planning produced different ordered task sequences"
            )
        assert not isinstance(record, dict)
        successful[key] = record
        scores.append((max(int(peer["ns"]) for peer in peers), key))
    if not scores:
        raise AdmissionError(
            "no common task sequence is feasible on every participating rank"
        )
    score, winner = min(scores)
    control.agree("selection/winner", winner)
    decisions = {
        "version": 1,
        "members": control.members,
        "rank": control.rank,
        "local_program": program.digest,
        "winner": winner,
        "worst_rank_ns": score,
        "evaluated": evidence,
    }
    return successful[winner], winner, successful, decisions


def restore_result(
    local: LocalPlan,
    program: ShadowSpillProgram,
    key: ChoiceKey,
    successful: Mapping[ChoiceKey, LocalPlan],
    *,
    keep_resolutions: bool,
) -> ProgramPlanResult:
    """Restore open-alternative metadata without altering the admitted schedule."""
    results = {}
    diagnostics: list[ResolvedProgramDiagnostics] = []
    for candidate, record in successful.items():
        result = record.result
        identity = hashlib.sha256(json.dumps(candidate).encode()).hexdigest()
        choices = tuple(
            TaskAlternativeChoiceDiagnostic(group, option)
            for group, option in candidate
        )
        local_diagnostics = tuple(
            replace(
                problem,
                selection_id=identity,
                choices=choices,
                candidate_evaluations=tuple(
                    replace(item, selection_id=identity)
                    for item in problem.candidate_evaluations
                ),
            )
            for problem in result.diagnostics.resolved_programs
        )
        # A fixed program has exactly one resolution.
        if len(local_diagnostics) != 1:
            raise ValueError(
                "fixed distributed candidate did not produce one resolution"
            )
        diagnostics.extend(local_diagnostics)
        results[candidate] = (result, identity)
    result, identity = results[key]
    selection = tuple(TaskAlternativeChoice(group, option) for group, option in key)
    restored = replace(
        result,
        program=program,
        selections=selection,
        diagnostics=replace(
            result.diagnostics,
            selected_selection_id=identity,
            resolved_programs=tuple(diagnostics),
        ),
        resolutions=tuple(
            ResolutionPlan(
                entry_id,
                tuple(
                    TaskAlternativeChoice(group, option) for group, option in candidate
                ),
                value.diagnostics.selected_candidate_id,
                value.schedule,
                value.simulation,
                value.resident_slice,
            )
            for candidate, (value, entry_id) in results.items()
        )
        if keep_resolutions
        else (),
    )
    assert (
        restored.program.selected_tasks(restored.selections)
        == local.result.program.tasks
    )
    return restored


def certify_restored(
    result: ProgramPlanResult,
    original_facts: AdmissionFacts,
    effective_facts: AdmissionFacts,
    *,
    scratch_reserve_bytes: int,
) -> tuple[AdmissionFacts, FixedLayoutAdmission]:
    """Certify the selected tasks under their original program identity.

    Candidate search removes unselected alternatives. Restoring those alternatives
    changes the program and admission-facts identities, even though execution is
    identical. Rebuild the certificate through normal admission before sealing.
    """
    facts = replace(effective_facts, tasks=original_facts.tasks)
    dynamic_aliases = frozenset(
        item.alias_group_id
        for item in result.schedule.final_residency
        if item.location is MemoryLocation.DEVICE
    )
    bound = current()
    assert bound is not None
    admission = bound.control.run(
        "selection/restored_admission",
        lambda: build_fixed_layout_admission(
            result,
            facts,
            dynamic_alias_group_ids=dynamic_aliases,
            scratch_reserve_bytes=scratch_reserve_bytes,
        ),
    )
    return facts, admission


def record_decision(
    store: ArtifactStore, decisions: Mapping[str, Any], *, plans=()
) -> None:
    if not store.plan_policy.write_enabled:
        return
    if plans:
        decisions = {
            **decisions,
            "shared_plans": [record_plan(store, plan) for plan in plans],
        }
    encoded = json.dumps(decisions, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    path = store.planning / "distributed" / digest / "selection.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_text(path, encoded)
    store.record(
        category="planning",
        kind="distributed_selection",
        digest=digest,
        path=path,
        access="write",
        schema="distributed_selection/v1",
        dependencies=(decisions["local_program"],),
    )
