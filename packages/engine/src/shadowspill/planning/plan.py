"""CPU scheduling and simulation, independent of stores and execution pools."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Literal

from shadowspill.artifacts.budgets import MemoryBudgets
from shadowspill.artifacts.plan import Plan
from shadowspill.artifacts.pool_memory import PoolMemoryProfile
from shadowspill.artifacts.profiled_program import ProfiledProgram
from shadowspill.communication import CommunicationInfo
from shadowspill.communication.memory import communication_overhead
from shadowspill.errors import PlanInfeasibleError

from .admission.program import caller_output_aliases, program_admission
from .costs import simulation_inputs
from .training.bind import bind_training

if TYPE_CHECKING:
    from shadowspill.planning.memory_search.algorithm import SearchOptions
    from shadowspill.step.ordering import StepDataOrdering


def plan_program(
    program: ProfiledProgram,
    *,
    memory_budgets: MemoryBudgets,
    communication_info: CommunicationInfo,
    pool_memory: PoolMemoryProfile | None = None,
    search_options: SearchOptions | None = None,
    ordering: StepDataOrdering | None = None,
    optimizer_ordering: Literal["stage_interleaved", "tail"] = "stage_interleaved",
) -> Plan:
    """Choose a Plan for a profiled Program within memory budgets.

    The search chooses each stage's save or recompute form, where each object
    resides and when it moves, and then places every allocation in a fixed
    device layout when the memory evidence allows. It runs on the CPU: no
    compilation, measurement, GPU or pools.

    Parameters
    ----------
    program : ProfiledProgram
        The Program and its profiles, from `apply_profiles`.
    memory_budgets : MemoryBudgets
        The worker's budgets. The device budget covers the fixed overhead as
        well as the execution pool.
    communication_info : CommunicationInfo
        Transfer and collective costs.
    pool_memory : PoolMemoryProfile, optional
        Measured spill pool memory for each of the Program's logical ranks,
        at the spill budget's capacity. It caps the device budget at the
        memory observed available, and adds the pool's process memory to the
        fixed overhead.
    search_options : SearchOptions, optional
        Search settings. By default ``SearchOptions()``.
    ordering : StepDataOrdering, optional
        Traversal of a training Program's examples. By default depth-first.
    optimizer_ordering : str, default "stage_interleaved"
        When each optimizer update runs: ``"stage_interleaved"``, once the
        gradients it needs are written and the old weights it changes are
        read; or ``"tail"``, after the whole backward.

    Returns
    -------
    Plan
        The chosen Plan.

    Raises
    ------
    PlanInfeasibleError
        If no schedule or layout fits the budgets, the fixed overhead alone
        exceeds the device budget, or ``ordering`` breaks the Program's
        ordering constraints.
    PlanSearchExhaustedError
        If the bounded search stops without a Plan or a proof that none
        exists.
    TypeError
        If ``program`` is not a `ProfiledProgram`, or ``search_options`` or
        ``ordering`` has another type.
    ValueError
        If ``ordering`` is given for a Program without training examples, or
        ``pool_memory`` differs from the budgets, ranks, device or spill
        backend.
    """
    from shadowspill.planning.memory_search.algorithm import SearchOptions

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
        pool_memory=pool_memory,
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
        pool_memory=pool_memory,
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
    pool_memory: PoolMemoryProfile | None = None,
) -> Plan:
    costed, config = simulation_inputs(
        program, memory_budgets, communication_info, pool_memory
    )
    facts = program_admission(
        program,
        costed,
        (
            memory_budgets.execution_bytes
            if pool_memory is None
            else min(memory_budgets.execution_bytes, pool_memory.physical_limit_bytes)
        )
        - program.fixed_device_bytes
        - (0 if pool_memory is None else pool_memory.process_reservation_bytes)
        - communication_overhead(program.program, communication_info)[0],
        communication_info,
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
        from shadowspill.planning.memory_search.admission.layout import (
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
    if options.trace_directory is not None:
        options.resolved_algorithm.record_trace(
            result, options=options, placement=facts
        )
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
        pool_memory=pool_memory,
    )
