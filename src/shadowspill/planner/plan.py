"""The one entry to planning: a question in, an admitted plan out.

`plan_program` is what a caller reaches for. It fixes the machine from a
budget, keys the answer in the planning store, hands the question to a
search, holds that search to any plan it was given, and admits the winner
physically.

Which search runs is the caller's choice and this module's ignorance:
`search_options.algorithm` is any
:class:`~shadowspill.planner.SearchAlgorithm`, carrying its own options into
the plan key and read by nothing here. PressureFit is the search that ships.

This module is framework-neutral: it needs a program, a budget and a
machine, and nothing that belongs to a frontend.
"""

from __future__ import annotations

import os

from shadowspill.ir import ResidencySpec, ShadowSpillProgram
from shadowspill.simulator import SimulationConfig
from shadowspill.store import ArtifactStore, StoreMode

from .admission import AdmissionFacts
from .program import (
    AnnotatedProgramPlan,
    ShadowSpillPlanningProblem,
    TransferBandwidths,
)
from .search import SearchOptions


def validate_schedule_feasibility(
    program: ShadowSpillProgram,
    *,
    initial_residency: tuple[ResidencySpec, ...],
    final_residency: tuple[ResidencySpec, ...] = (),
    config: SimulationConfig,
    admission: AdmissionFacts | None = None,
    search_options: SearchOptions | None = None,
) -> None:
    """Reject an irreducible capacity failure before a search is paid for.

    The check is the search's own -- only it knows what it could reach --
    so this asks the one that will run. What passes here is what that
    search can reach; it is not a promise that a plan exists.
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
    implementation_revision: str | None = None,
) -> AnnotatedProgramPlan:
    """Plan one problem: search, simulate, and physically admit the winner.

    ``problem`` is normally ``build_step_program(...).recurrent`` or the
    value reconstructed by :meth:`ShadowSpillPlanningProblem.from_value`. It
    carries the program, where it must start and end, and what the machine
    is; it carries no policy, so this call is model- and runtime-independent
    and may be repeated for a budget/bandwidth frontier without capture,
    compilation, or profiling.

    ``execution_budget``, ``spill_budget`` and ``transfer_bandwidths``
    override what the problem was captured under, which is how one problem
    serves a whole frontier.

    ``search_options`` is the whole of what this call is told about
    searching: ``generic``, which any search understands, and
    ``algorithm``, which is the search itself carrying its own options.
    Both reach the plan key; ``search_options.workers`` does not, because
    it says how much machine to spend rather than what to decide. Leaving
    ``algorithm`` unset runs the search that ships; a caller with a search of
    their own subclasses :class:`~shadowspill.planner.SearchAlgorithm` and
    passes an instance.

    ``incumbent`` is the plan to beat: a plan for this same program found
    under another budget. It reaches the search as a bound, and the answer
    is held to it here -- a plan that fits in less memory fits in more, so a
    sweep that plans budgets ascending hands each one the best plan below it
    and never plans worse with more, whichever search is running.

    ``plan_store`` keeps this call's request, selection and plan manifest
    apart from the artifact store, so one store can serve many runs that
    each own their plans; ``None`` keeps them in the store.
    """

    from .selection import select_program

    if search_options is not None and not isinstance(search_options, SearchOptions):
        raise TypeError("search_options must be SearchOptions or None")
    cache = ArtifactStore.resolve(
        artifact_store,
        plan_store=plan_store,
        plan_store_mode=plan_store_mode,
        implementation_revision=implementation_revision,
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
    )
