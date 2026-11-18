"""Admit a common logical schedule without sharing target-specific GPU code."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from shadowspill.artifacts.plan import Plan
from shadowspill.errors import PlanInfeasibleError, PlanSearchExhaustedError
from shadowspill.ir.validation import ValidationError
from shadowspill.planning.admission.program import caller_output_aliases
from shadowspill.planning.collectives import collective_sequence, first_difference
from shadowspill.planning.costs import simulation_inputs
from shadowspill.planning.memory_search.admission.layout import (
    FixedLayoutInfeasibleError,
    build_fixed_layout_admission,
)
from shadowspill.planning.memory_search.algorithm import SearchOptions
from shadowspill.planning.memory_search.diagnostics import (
    INCUMBENT_CANDIDATE_ID,
    IncumbentDiagnostic,
    PlanningDiagnostics,
    ResolvedProgramDiagnostics,
    TaskAlternativeChoiceDiagnostic,
)
from shadowspill.planning.memory_search.result import ProgramPlanResult
from shadowspill.planning.modes import bundle_gradient_modes
from shadowspill.simulator import SimulationInfeasibleError, simulate


def _difference(plans: Mapping[int, Plan]) -> str | None:
    for accumulated in (False, True):
        paths = [
            plan._accumulation_plan if accumulated else plan
            for _, plan in sorted(plans.items())
        ]
        if all(path is None for path in paths):
            continue
        if any(path is None for path in paths):
            raise ValueError("workers disagree on gradient accumulation paths")
        difference = first_difference(
            "accumulation" if accumulated else "main",
            [collective_sequence(path) for path in paths if path is not None],
        )
        if difference is not None:
            return difference
    return None


def coordinate_plans(plans: Mapping[int, Plan]) -> dict[int, Plan]:
    """Keep agreeing Plans or select one schedule admitted on every worker.

    Independent searches supply candidate schedules, not permission to issue
    different collectives. When their issue orders differ, try each distinct
    answer on every worker's original program, measured costs and capacity.
    Retention/recomputation and transfer boundaries travel together. Kernels,
    profiles, allocation evidence and physical layouts never cross targets.

    This bounded search can decline; it does not prove that no common schedule
    exists. Opaque provider collective contracts remain the caller's obligation.
    """
    if not plans:
        raise ValueError("provide at least one worker Plan")
    difference = _difference(plans)
    if difference is None:
        return dict(plans)
    best: dict[int, Plan] | None = None
    best_duration: int | None = None
    seen = set()
    failures = []
    for rank, source in sorted(plans.items()):
        identity = tuple(
            (path.schedule.digest, path.selections)
            for path in (source, source._accumulation_plan)
            if path is not None
        )
        if identity in seen:
            continue
        seen.add(identity)
        try:
            candidate = {
                worker: _replay_modes(target, source, rank)
                for worker, target in sorted(plans.items())
            }
            mismatch = _difference(candidate)
            if mismatch is not None:
                raise PlanInfeasibleError(mismatch, kind="prepared_collective_order")
        except (
            PlanInfeasibleError,
            SimulationInfeasibleError,
            FixedLayoutInfeasibleError,
            ValidationError,
        ) as error:
            failures.append(f"Worker {rank}'s schedule: {error}")
            continue
        duration = max(
            path.timeline.makespan_ns
            for plan in candidate.values()
            for path in (plan, plan._accumulation_plan)
            if path is not None
        )
        if best_duration is None or duration < best_duration:
            best, best_duration = candidate, duration
    if best is None:
        raise PlanSearchExhaustedError(
            "no searched schedule admitted on every cooperating worker; " + difference,
            diagnostics=tuple(failures),
        )
    return best


def _replay_modes(target: Plan, source: Plan, rank: int) -> Plan:
    fresh = _replay(target, source, rank)
    if target._accumulation_plan is None:
        return fresh
    assert source._accumulation_plan is not None
    accumulated = _replay(target._accumulation_plan, source._accumulation_plan, rank)
    return bundle_gradient_modes(fresh, accumulated)


def _replay(target: Plan, source: Plan, rank: int) -> Plan:
    if target is source:
        return target
    ours, theirs = target._costed_program, source._costed_program
    # Names alone are not a topology contract. Require the same task dataflow,
    # alternatives, alias ownership and invocation boundaries. Shapes and local
    # implementations may differ; replay and admission use only the target's.
    compatible = (
        target.program.initial_residency == source.program.initial_residency
        and target.program.final_residency == source.program.final_residency
        and ours.task_alternative_groups == theirs.task_alternative_groups
        and tuple(replace(t, profile_id="local") for t in ours.tasks)
        == tuple(replace(t, profile_id="local") for t in theirs.tasks)
        and tuple(replace(a, size_bytes=0, transfer=None) for a in ours.alias_groups)
        == tuple(replace(a, size_bytes=0, transfer=None) for a in theirs.alias_groups)
        and tuple(replace(o, size_bytes=0, offset_bytes=0) for o in ours.objects)
        == tuple(replace(o, size_bytes=0, offset_bytes=0) for o in theirs.objects)
    )
    if not compatible:
        raise PlanInfeasibleError(
            "common scheduling requires matching task and storage topology",
            kind="coordinated_topology",
        )
    if target.memory_layout is None or target._admission_facts is None:
        raise ValueError("coordinated execution requires physical admission evidence")
    _, config = simulation_inputs(
        target.profiled_program,
        target.memory_budgets,
        target.communication_info,
        target.pool_memory,
    )
    source.schedule.validate(ours, source.selections)
    simulation = simulate(
        ours, source.schedule, selections=source.selections, config=config
    )
    selected = ProgramPlanResult(
        ours,
        SearchOptions(),
        target.program.initial_residency,
        target.program.final_residency,
        config,
        source.schedule,
        source.selections,
        simulation,
        source.diagnostics,
    )
    certified = build_fixed_layout_admission(
        selected,
        target._admission_facts,
        dynamic_alias_group_ids=caller_output_aliases(ours),
    )
    duration = certified.simulation.makespan_ns
    selection = source.diagnostics.selected_selection_id
    diagnostics = PlanningDiagnostics(
        INCUMBENT_CANDIDATE_ID,
        selection,
        duration,
        (
            ResolvedProgramDiagnostics(
                selection,
                tuple(
                    TaskAlternativeChoiceDiagnostic(s.group_id, s.option_id)
                    for s in source.selections
                ),
                INCUMBENT_CANDIDATE_ID,
                duration,
                (),
                incumbent=IncumbentDiagnostic(
                    "valid",
                    duration,
                    certified.layout.required_bytes,
                    True,
                    schedule_digest=source.schedule.digest,
                    found_by=(
                        f"worker {rank}: {source.diagnostics.selected_candidate_id}"
                    ),
                    found_at_capacity_bytes=source.effective_execution_bytes,
                ),
            ),
        ),
        search="coordinated_schedule",
    )
    return replace(
        target,
        schedule=source.schedule,
        selections=source.selections,
        timeline=certified.simulation,
        memory_layout=certified.layout,
        diagnostics=diagnostics,
        _accumulation_plan=None,
    )
