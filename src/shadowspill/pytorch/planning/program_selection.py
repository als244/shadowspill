"""PressureFit and physical admission for a saved pre-PressureFit Program."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace

from shadowspill.planner import PressureFitOptions, PressureFitResult
from shadowspill.pytorch.cache import PlanningCache

from ..program import (
    AnnotatedProgramPlan,
    MemoryBudgets,
    PressureFitProgram,
    TransferBandwidths,
)
from .admission import resolve_fixed_layout_selection
from .repositories import open_artifact_repositories


def select_program(
    program: PressureFitProgram,
    *,
    execution_budget_bytes: int | None,
    spill_budget_bytes: int | None,
    transfer_bandwidths: TransferBandwidths | None,
    options: PressureFitOptions | None,
    planning_cache: PlanningCache,
    verbose: bool,
) -> AnnotatedProgramPlan:
    """Select and physically admit one reusable Program."""

    started = time.perf_counter_ns()
    config, facts = program.pressurefit_inputs(
        execution_budget_bytes=execution_budget_bytes,
        spill_budget_bytes=spill_budget_bytes,
        transfer_bandwidths=transfer_bandwidths,
    )
    selected_options = options or program.options
    repositories = open_artifact_repositories(planning_cache)
    progress = _progress_printer() if verbose else None
    selection = resolve_fixed_layout_selection(
        config,
        facts,
        lambda candidate_config: repositories.resolve_pressurefit(
            program.program,
            initial_residency=program.initial_residency,
            final_residency=program.final_residency,
            config=candidate_config,
            options=selected_options,
            progress=progress,
        ),
        scratch_reserve_bytes=program.dynamic_scratch_reserve_bytes,
        progress=progress,
    )
    physical_result = _with_physical_prediction(
        selection.result,
        selection.admission.simulation,
        facts=selection.facts,
    )
    selected_transfer = transfer_bandwidths or program.transfer_bandwidths
    return AnnotatedProgramPlan(
        program=program,
        memory_budgets=MemoryBudgets(
            execution_bytes=(
                program.source_execution_budget_bytes
                if execution_budget_bytes is None
                else execution_budget_bytes
            ),
            spill_bytes=(
                program.simulation_config.spill_capacity_bytes
                if spill_budget_bytes is None
                else spill_budget_bytes
            ),
        ),
        transfer_bandwidths=selected_transfer,
        result=physical_result,
        effective_facts=selection.facts,
        fixed_layout=selection.admission.layout,
        simulation_admission=selection.admission.simulator_input,
        simulation=selection.admission.simulation,
        attempts=selection.attempts,
        pressurefit_cache_hit=selection.cache_hit,
        wall_time_ns=time.perf_counter_ns() - started,
    )


def _with_physical_prediction(
    selected: PressureFitResult,
    simulation: object,
    *,
    facts: object,
) -> PressureFitResult:
    """Replace logical timing with dependency-certified physical simulation."""

    from shadowspill.planner import AdmissionFacts
    from shadowspill.simulator import SimulationResult

    if not isinstance(simulation, SimulationResult):
        raise TypeError("physical simulation has an invalid type")
    if not isinstance(facts, AdmissionFacts):
        raise TypeError("effective facts has an invalid type")
    return replace(
        selected,
        simulation=simulation,
        diagnostics=selected.diagnostics.with_selected_makespan(
            simulation.makespan_ns
        ),
        admission_facts=facts,
    )


def _progress_printer() -> Callable[[str], None]:
    def progress(message: str) -> None:
        print(f"PressureFit: {message}", flush=True)

    return progress


__all__ = ["select_program"]
