"""CPU scheduling and simulation, independent of stores and execution pools."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from shadowspill.artifacts.plan import Plan
from shadowspill.artifacts.profiled_program import ProfiledProgram
from shadowspill.communication import CommunicationInfo

from .budgets import MemoryBudgets
from .costs import simulation_inputs
from .training.bind import bind_training

if TYPE_CHECKING:
    from shadowspill.planner.search import SearchOptions
    from shadowspill.step.ordering import StepDataOrdering


def plan_program(
    program: ProfiledProgram,
    *,
    memory_budgets: MemoryBudgets,
    communication_info: CommunicationInfo,
    search_options: SearchOptions | None = None,
    ordering: StepDataOrdering | None = None,
) -> Plan:
    """Plan a fully supplied candidate without compilation, measurement or CUDA.

    The existing search raises diagnostic PlanInfeasibleError or
    PlanSearchExhaustedError. No cache directory or allocator is created.
    """
    from shadowspill.planner.search import SearchOptions

    if not isinstance(program, ProfiledProgram):
        raise TypeError(
            "plan_program requires a ProfiledProgram; call apply_profiles first"
        )
    if search_options is not None and not isinstance(search_options, SearchOptions):
        raise TypeError("search_options must be SearchOptions")
    options = search_options or SearchOptions()
    program = replace(program, program=bind_training(program.program, ordering))
    costed, config = simulation_inputs(program, memory_budgets, communication_info)
    result = options.resolved_algorithm(
        costed,
        initial_residency=program.program.initial_residency,
        final_residency=program.program.final_residency,
        config=config,
        generic=options.generic,
        workers=options.workers,
    )
    return Plan(
        program,
        memory_budgets,
        communication_info,
        result.schedule,
        result.selections,
        result.simulation,
        result.diagnostics,
        costed,
    )
