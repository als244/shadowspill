"""CPU scheduling and simulation, independent of stores and execution pools."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Literal

from shadowspill.artifacts.plan import Plan
from shadowspill.artifacts.profiled_program import ProfiledProgram
from shadowspill.communication import CommunicationInfo
from shadowspill.communication.memory import communication_overhead
from shadowspill.errors import PlanInfeasibleError

from .admission.program import caller_output_aliases, program_admission
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
    optimizer_ordering: Literal["stage_interleaved", "tail"] = "stage_interleaved",
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
    fresh = _plan_bound(
        replace(
            program,
            program=bind_training(
                program.program, ordering, optimizer_ordering=optimizer_ordering
            ),
        ),
        memory_budgets=memory_budgets,
        communication_info=communication_info,
        options=options,
    )
    if program.program.bindings.get("kind") != "forward_backward":
        return fresh
    accumulated = _plan_bound(
        replace(
            program,
            program=bind_training(
                program.program,
                ordering,
                accumulate=True,
                optimizer_ordering=optimizer_ordering,
            ),
        ),
        memory_budgets=memory_budgets,
        communication_info=communication_info,
        options=options,
    )
    from .modes import bundle_gradient_modes

    return bundle_gradient_modes(fresh, accumulated)


def _plan_bound(
    program: ProfiledProgram,
    *,
    memory_budgets: MemoryBudgets,
    communication_info: CommunicationInfo,
    options: SearchOptions,
) -> Plan:
    costed, config = simulation_inputs(program, memory_budgets, communication_info)
    facts = program_admission(
        program,
        costed,
        memory_budgets.execution_bytes
        - program.fixed_device_bytes
        - communication_overhead(program.program, communication_info)[0],
    )
    result = options.resolved_algorithm(
        costed,
        initial_residency=program.program.initial_residency,
        final_residency=program.program.final_residency,
        config=config,
        generic=options.generic,
        workers=options.workers,
        # Executor uses the certified fixed layout, not the dynamic allocator's
        # first-fit placement. Requiring both can reject a layout that fits:
        # the dynamic replay fragments ranges before fixed placement is tried.
        # Search still places every candidate; certification below proves the
        # selected offsets and their completion/reuse dependencies.
        admission=None,
        placement=facts,
    )
    layout = None
    timeline = result.simulation
    if facts is not None:
        from shadowspill.planner.admission.layout import (
            FixedLayoutInfeasibleError,
            build_fixed_layout_admission,
        )

        try:
            certified = build_fixed_layout_admission(
                result, facts, dynamic_alias_group_ids=caller_output_aliases(costed)
            )
        except FixedLayoutInfeasibleError as error:
            raise PlanInfeasibleError(
                str(error),
                kind="physical_layout",
                required_bytes=error.required_bytes,
                capacity_bytes=error.capacity_bytes,
            ) from error
        layout, timeline = certified.layout, certified.simulation
    return Plan(
        program,
        memory_budgets,
        communication_info,
        result.schedule,
        result.selections,
        timeline,
        result.diagnostics,
        costed,
        layout,
        facts,
    )
