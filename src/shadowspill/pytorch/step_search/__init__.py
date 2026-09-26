"""A step planned at every geometry, budget and walk.

Which geometries there are and how each is walked is in ``geometries``,
what the search answered in ``report``, the rule one point is answered by in
``planner``, and the walk itself in ``sweep``; ``plan_step_search`` below
wires the three together.
"""

from collections.abc import Callable, Sequence
from os import PathLike
from types import MappingProxyType
from typing import Any, Literal

import torch
from torch import nn

from shadowspill.planner import (
    SearchOptions,
    StepDataOrdering,
)
from shadowspill.planner.diagnostics import (
    GraphPairOutcome,
)
from shadowspill.planner.program_inputs import (
    TransferBandwidths,
)
from shadowspill.pytorch.runtime import Runtime
from shadowspill.search.geometries import default_orderings, search_geometries
from shadowspill.search.planner import _Planner
from shadowspill.search.report import (
    StepSearchGeometryBuild,
    StepSearchPoint,
    StepSearchReport,
)
from shadowspill.store import StoreMode

from .sweep import _Build, _Sweep

__all__ = [
    "GraphPairOutcome",
    "StepSearchGeometryBuild",
    "StepSearchPoint",
    "StepSearchReport",
    "default_orderings",
    "plan_step_search",
    "search_geometries",
]


def plan_step_search(
    model: nn.Module,
    *,
    objective: Any,
    optimizer: Any,
    hyperparams: Sequence[str] = (),
    example_microbatches: Callable[[int, int], Sequence[Sequence[Any]]],
    total_sequences_per_step: int,
    sequence_length: int,
    budgets: Sequence[tuple[int, int]],
    runtime: Runtime,
    execution: str,
    spill: str,
    transfer_bandwidths: TransferBandwidths | None = None,
    min_tokens_per_microbatch: int | None = None,
    max_tokens_per_microbatch: int | None = None,
    optimizer_ordering: Literal["stage_interleaved", "tail"] = "stage_interleaved",
    orderings: Callable[[int], Sequence[StepDataOrdering]] | None = None,
    search_options: SearchOptions | None = None,
    incumbents: bool = True,
    artifact_store: str | PathLike[str] | None = None,
    build_store: str | PathLike[str] | None = None,
    plan_store: str | PathLike[str] | None = None,
    build_store_mode: StoreMode = "contribute",
    plan_store_mode: StoreMode = "contribute",
    verbose: bool = False,
    progress: Callable[[str], None] | None = None,
    export_bypass_key: str | None = None,
    master_dtype: torch.dtype | None = None,
    grad_dtype: torch.dtype | None = None,
) -> StepSearchReport:
    """Plan every admitted geometry under every budget; execute nothing.

    ``example_microbatches(sequences, accumulation)`` supplies the example
    inputs for one geometry — structure is what matters, values are not.
    ``transfer_bandwidths`` overrides the calibration each step program
    embeds from the runtime; leave it unset to plan against the measured
    routes. Either way the report records the calibration each geometry's
    program embeds, and the override when there was one, so two searches
    can be compared or one pinned to another's. ``search_options`` reaches
    every point unchanged, so a value set here is the value searched under.
    Failures are outcomes, not errors: a geometry-budget point that proves
    infeasible or exhausts its search budget is reported with that status
    while the search continues. A geometry whose build exhausts the
    device -- profiling runs real kernels, so the largest microbatch can --
    reports every one of its budgets ``infeasible`` with the exhaustion as
    the point's error, and the search moves to the next geometry; that
    geometry contributes no build to the report, because it produced no
    program. ``progress`` is called with a short line at every geometry and
    point boundary; ``verbose`` additionally forwards each planning call's
    own phase reporting.

    ``orderings`` maps a geometry's accumulation count to the
    :class:`StepDataOrdering` values to try for it; each ordering is lowered
    into its own program, sharing the geometry's capture and profiles, and
    planned under every budget. The default, :func:`default_orderings`, is
    every ``depth x breadth`` factor pair with the flags at their defaults.

    ``search_options`` names the resolutions every point is searched
    over, with the meaning it has for :func:`plan_step`; ``None`` is the
    library's default of every quarter. Options that are not valid are
    rejected before any geometry is built. ``master_dtype`` and
    ``grad_dtype`` have their :func:`plan_step` meanings too: every geometry
    is built with the masters and the gradients the step it plans will keep.

    ``incumbents`` hands each point the best plan found at a smaller budget
    of the same program, as the plan to beat: budgets are planned ascending,
    a plan that fits in less memory fits in more, and the search answers
    with it unless it does strictly better, so no program plans worse with
    more memory. A point that answered with a handed-in plan records the
    budget it came from as ``incumbent_budget_bytes``. ``False`` searches
    every point alone, which is how the two are compared.

    ``plan_store`` keeps every point's plan records apart from the
    artifact store, so a search can reuse another run's captures, profiles
    and lowering and still plan every point itself; ``None`` keeps them in
    the store, where a matching plan would be read back instead of planned.

    On a warm store a point is answered from the summary kept beside its
    plan -- makespan, ``PlanSummary`` and graph-pair outcomes -- without
    reading the plan. Whole plans are read only for each budget's winner,
    which is what ``winner_plans`` hands the run phase, and for a plan to
    beat the moment a later point has to beat it. A point whose plan in hand
    claims to beat the stored answer is searched, as the store itself would
    search it, and ``plan_store_mode`` ``require`` refuses a point the store
    cannot answer either way.
    """

    def announce(message: str) -> None:
        if progress is not None:
            progress(message)

    if not budgets:
        raise ValueError("at least one (execution, spill) budget is required")
    geometries, skipped = search_geometries(
        total_sequences_per_step,
        sequence_length=sequence_length,
        min_tokens_per_microbatch=min_tokens_per_microbatch,
        max_tokens_per_microbatch=max_tokens_per_microbatch,
    )
    orderings_for = default_orderings if orderings is None else orderings
    per_geometry = [
        tuple(orderings_for(accumulation)) for _sequences, accumulation in geometries
    ]
    sweep = _Sweep(
        ask=_Planner(
            transfer_bandwidths,
            search_options,
            artifact_store,
            plan_store,
            plan_store_mode,
            verbose,
        ),
        budgets=tuple(budgets),
        incumbents=incumbents,
        announce=announce,
        point_total=sum(len(item) for item in per_geometry) * len(budgets),
    )
    sweep.run(
        geometries,
        per_geometry,
        _Build(
            model=model,
            objective=objective,
            optimizer=optimizer,
            hyperparams=hyperparams,
            example_microbatches=example_microbatches,
            runtime=runtime,
            execution=execution,
            spill=spill,
            optimizer_ordering=optimizer_ordering,
            verbose=verbose,
            artifact_store=artifact_store,
            build_store=build_store,
            build_store_mode=build_store_mode,
            export_bypass_key=export_bypass_key,
            master_dtype=master_dtype,
            grad_dtype=grad_dtype,
        ),
    )
    return StepSearchReport(
        total_sequences_per_step=total_sequences_per_step,
        sequence_length=sequence_length,
        budgets=tuple(budgets),
        geometries=tuple(sweep.builds),
        points=tuple(sweep.points),
        skipped=skipped,
        search_options=search_options,
        transfer_bandwidths=transfer_bandwidths,
        winner_plans=MappingProxyType(sweep.winner_plans()),
    )
