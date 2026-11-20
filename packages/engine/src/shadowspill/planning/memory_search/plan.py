"""Search and admit a costed memory-planning problem.

`plan_program` is what a caller reaches for. It fixes the machine from a
budget, keys the answer in the plan store, hands the question to a
search, holds that search to any plan it was given, and admits the winner
physically.

Which search runs is the caller's choice and this module's ignorance:
`search_options.algorithm` is any
:class:`~shadowspill.planning.memory_search.SearchAlgorithm`, carrying its options
into the plan key and read by nothing here. PressureFit is the search that ships.

This module is framework-neutral: it needs a program, a budget and a
machine, and nothing that belongs to a frontend.
"""

from __future__ import annotations

import os

from shadowspill.ir import ResidencySpec, ShadowSpillProgram
from shadowspill.planning.memory_search.admission import AdmissionFacts
from shadowspill.planning.memory_search.algorithm import SearchOptions
from shadowspill.planning.memory_search.program import (
    AnnotatedProgramPlan,
    ShadowSpillPlanningProblem,
    TransferBandwidths,
)
from shadowspill.simulator import SimulationConfig
from shadowspill.store import ArtifactStore, StoreMode


def validate_schedule_feasibility(
    program: ShadowSpillProgram,
    *,
    initial_residency: tuple[ResidencySpec, ...],
    final_residency: tuple[ResidencySpec, ...] = (),
    config: SimulationConfig,
    admission: AdmissionFacts | None = None,
    search_options: SearchOptions | None = None,
) -> None:
    """Refuse a machine no schedule can fit, before a search is paid for.

    It asks the search that will run, through its
    `SearchAlgorithm.preflight`. Passing promises no plan.

    Parameters
    ----------
    program : ShadowSpillProgram
        The program, its task alternatives still open.
    initial_residency : tuple of ResidencySpec
        Where each alias group is before the first task.
    final_residency : tuple of ResidencySpec, default ()
        Where each alias group must be after the last task.
    config : SimulationConfig
        The machine to check against.
    admission : AdmissionFacts, optional
        The execution pool's physical facts, when the caller has them.
    search_options : SearchOptions, optional
        Which search to ask, and what it is told. None asks the search that
        ships, with its defaults.

    Raises
    ------
    PlanInfeasibleError
        If the search can tell that no schedule fits.
    """

    chosen = search_options if search_options is not None else SearchOptions()
    chosen.resolved_algorithm.preflight(
        program,
        initial_residency=initial_residency,
        final_residency=final_residency,
        config=config,
        admission=admission,
        generic=chosen.generic,
    )


__all__ = ["plan_program", "validate_schedule_feasibility"]


def plan_program(
    problem: ShadowSpillPlanningProblem,
    *,
    execution_budget: int | None = None,
    spill_budget: int | None = None,
    transfer_bandwidths: TransferBandwidths | None = None,
    search_options: SearchOptions | None = None,
    incumbent: AnnotatedProgramPlan | None = None,
    artifact_store: str | os.PathLike[str] | None = None,
    plan_store: str | os.PathLike[str] | None = None,
    verbose: bool = True,
    plan_store_mode: StoreMode = "contribute",
    export_bypass_key: str | None = None,
    keep_resolutions: bool = False,
) -> AnnotatedProgramPlan:
    """Plan one problem: search it, then certify the answer's physical layout.

    The answer is keyed in the plan store. When the store holds it, the
    plan is read back with the simulation and layout certificate recorded
    beside it, and nothing is searched, simulated or placed; when it holds
    a refusal, the refusal is raised again. Otherwise the search runs, its
    answer is held to ``incumbent``, its fixed layout is certified, and the
    result is stored as the mode allows. No capture, compilation or
    profiling is needed, so one problem serves a frontier of budgets and
    bandwidths.

    Parameters
    ----------
    problem : ShadowSpillPlanningProblem
        The program, its residency boundaries and the machine it was built
        for.
    execution_budget : int, optional
        Device bytes to plan for, at most the problem's
        ``maximum_execution_budget_bytes``. None uses its
        ``source_execution_budget_bytes``.
    spill_budget : int, optional
        Spill bytes to plan for, with no ceiling. None uses the problem's
        spill capacity.
    transfer_bandwidths : TransferBandwidths, optional
        Copy rates to price against. None uses the problem's own.
    search_options : SearchOptions, optional
        Which search runs and what it is told. None runs the search that
        ships, with its defaults.
    incumbent : AnnotatedProgramPlan, optional
        A plan for the same problem found under another budget, to beat. It
        reaches the search as a bound, and is the answer when it is strictly
        faster here and its layout still fits, so a sweep over ascending
        budgets never plans worse with more memory. A stored answer or
        refusal is searched again when the incumbent is faster. The plan key
        does not include it.
    artifact_store : str or path-like, optional
        Root of the artifact store. None uses ``~/.cache/shadowspill``.
    plan_store : str or path-like, optional
        Root of a planning tree of its own, so runs that share an artifact
        store each keep their own plans. None uses the artifact store's.
    verbose : bool, default True
        Print one line per search milestone.
    plan_store_mode : str, default "contribute"
        What the call does with the planning tree: ``"contribute"`` reads a
        hit and writes a miss, ``"reuse"`` reads a hit and writes nothing,
        ``"require"`` refuses a miss, and ``"refresh"`` ignores hits and
        replaces them.
    export_bypass_key : str, optional
        The caller's label for the code the build was made from. The store
        validates and keeps it; the plan key does not include it.
    keep_resolutions : bool, default False
        Also store every resolved program's best plan beside the answer,
        certified like it, when the search runs. Each is as large as the
        answer. A resolution bounded away before it placed a plan has none,
        unless the search options are ``deterministic``.

    Returns
    -------
    AnnotatedProgramPlan
        The certified plan, with its layout, simulation and timing.

    Raises
    ------
    TypeError
        If ``search_options`` is neither a `SearchOptions` nor None.
    ValueError
        If a budget, mode or key is invalid, or a stored record is malformed.
    FixedLayoutInfeasibleError
        If the answer's fixed layout does not fit the pool.
    PlanInfeasibleError
        If the budget leaves no capacity for objects, or no schedule fits.
    PlanSearchExhaustedError
        If a schedule may exist, but the search did not reach one.
    LookupError
        If ``plan_store_mode`` is ``"require"`` and the store lacks the answer.
    """

    from shadowspill.planning.memory_search.selection import select_program

    if search_options is not None and not isinstance(search_options, SearchOptions):
        raise TypeError("search_options must be SearchOptions or None")
    cache = ArtifactStore.resolve(
        artifact_store,
        plan_store=plan_store,
        plan_store_mode=plan_store_mode,
        export_bypass_key=export_bypass_key,
    )
    cache.initialize()
    return select_program(
        problem,
        execution_budget_bytes=execution_budget,
        spill_budget_bytes=spill_budget,
        transfer_bandwidths=transfer_bandwidths,
        search_options=search_options,
        incumbent=None if incumbent is None else incumbent.result,
        artifact_store=cache,
        verbose=verbose,
        keep_resolutions=keep_resolutions,
    )
