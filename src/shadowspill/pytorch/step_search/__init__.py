"""A step planned at every geometry, budget and walk.

Which geometries there are and how each is walked is in ``geometries``,
what the search answered in ``report``, the rule one point is answered by in
``planner``, and the walk itself in ``sweep``; ``plan_step_search`` below
wires the three together.
"""

from collections.abc import Callable, Mapping, Sequence
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
from shadowspill.pytorch.capture.retention import MEMORY_BOUND_FLOPS_PER_BYTE
from shadowspill.pytorch.distributed import Distributed
from shadowspill.pytorch.distributed import current as distributed_preparation
from shadowspill.pytorch.distributed._preparation import prepared
from shadowspill.pytorch.partition import PartitionSpec
from shadowspill.pytorch.runtime import Runtime
from shadowspill.search.geometries import default_orderings
from shadowspill.search.planner import _Planner
from shadowspill.search.report import (
    StepSearchGeometryBuild,
    StepSearchPoint,
    StepSearchReport,
)
from shadowspill.store import StoreMode
from shadowspill.task.profiling import ProfilingOptions

from .sweep import _Build, _Sweep

__all__ = [
    "GraphPairOutcome",
    "StepSearchGeometryBuild",
    "StepSearchPoint",
    "StepSearchReport",
    "default_orderings",
    "plan_step_search",
]


@prepared
def plan_step_search(
    model: nn.Module,
    *,
    objective: Any,
    optimizer: Any,
    hyperparams: Sequence[str] = (),
    candidates: Mapping[str, Sequence[Sequence[Any]]],
    budgets: Sequence[tuple[int, int]],
    runtime: Runtime,
    distributed: Distributed | None = None,
    shard_optimizer: bool = True,
    execution: str,
    spill: str,
    transfer_bandwidths: TransferBandwidths | None = None,
    metadata: Mapping[str, Any] | None = None,
    partition: PartitionSpec = "auto",
    execution_device: int | str | torch.device | None = None,
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
    parameter_metrics: Callable[[torch.Tensor, torch.Tensor], Any] | None = None,
    round_accumulation_once: bool = False,
    memory_bound_flops_per_byte: float = MEMORY_BOUND_FLOPS_PER_BYTE,
    keep_resolutions: bool = False,
    profiling_options: ProfilingOptions | None = None,
) -> StepSearchReport:
    """Search named representative updates across budgets and task orderings.

    ``candidates`` maps a caller-chosen name to the positional input sequences
    for its microbatches. Each candidate must implement the same full update;
    neither shapes nor the number of microbatches imply its normalization.
    The caller's objective and scalar input values define that math.

    ``metadata`` is optional JSON-compatible report annotation. It never
    changes capture or planning. Text recipes can record token geometry here;
    the planner itself makes no assumption about tokens, sequences or losses.

    Each candidate is captured/profiled once, then lowered under the requested
    orderings and searched at every budget. Capture device exhaustion and
    infeasible searches are retained as outcomes. Other capture errors surface
    immediately. ``incumbents`` carries an earlier budget's best plan forward.
    The report retains winner plans for admission through ``plan_step``.

    Other arguments have their ``plan_step`` meanings. ``progress`` receives
    candidate/point boundaries and ``verbose`` enables full phase reporting.
    """

    def announce(message: str) -> None:
        if progress is not None:
            progress(message)

    if not budgets:
        raise ValueError("at least one (execution, spill) budget is required")
    if not candidates:
        raise ValueError("at least one named microbatch candidate is required")
    if any(
        not isinstance(name, str) or not name or not inputs
        for name, inputs in candidates.items()
    ):
        raise ValueError("each candidate needs a nonempty name and microbatch inputs")
    geometries = tuple((name, len(inputs)) for name, inputs in candidates.items())
    orderings_for = default_orderings if orderings is None else orderings
    per_geometry = [
        tuple(orderings_for(accumulation)) for _sequences, accumulation in geometries
    ]
    planner = _Planner
    prepared = distributed_preparation()
    if prepared is not None:
        from shadowspill.pytorch.distributed._search import DistributedPlanner

        planner = DistributedPlanner
        prepared.control.agree(
            "sweep/candidates",
            [
                [name, count, [item.label for item in walks]]
                for (name, count), walks in zip(geometries, per_geometry, strict=True)
            ],
        )
        prepared.control.agree("sweep/budget_count", len(budgets))
    sweep = _Sweep(
        ask=planner(
            transfer_bandwidths,
            search_options,
            artifact_store,
            plan_store,
            plan_store_mode,
            verbose,
            keep_resolutions,
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
            profiling_options=profiling_options or ProfilingOptions(),
            objective=objective,
            optimizer=optimizer,
            hyperparams=hyperparams,
            candidates=candidates,
            partition=partition,
            execution_device=execution_device,
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
            parameter_metrics=parameter_metrics,
            round_accumulation_once=round_accumulation_once,
            memory_bound_flops_per_byte=memory_bound_flops_per_byte,
        ),
    )
    return StepSearchReport(
        metadata=dict(metadata or {}),
        budgets=tuple(budgets),
        geometries=tuple(sweep.builds),
        points=tuple(sweep.points),
        search_options=search_options,
        transfer_bandwidths=transfer_bandwidths,
        winner_plans=MappingProxyType(sweep.winner_plans()),
    )
