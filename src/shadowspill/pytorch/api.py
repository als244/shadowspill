"""Small public orchestration functions for forward and training planning."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Literal

import torch
import torch.nn as nn

from shadowspill.pytorch.cache import PlanningCache
from shadowspill.pytorch.callables import PlannedForward, PlannedTrainStep
from shadowspill.pytorch.partition import PartitionSpec
from shadowspill.pytorch.runtime_adapter import Runtime


def plan_forward(
    model: nn.Module,
    *,
    example_inputs: Sequence[Any],
    runtime: Runtime,
    execution: str,
    spill: str,
    execution_budget: int | None = None,
    spill_budget: int | None = None,
    execution_device: int | str | torch.device | None = None,
    partition: PartitionSpec = "auto",
    verbose: bool = True,
    planning_cachedir: str | os.PathLike[str] | None = None,
    profiling_metadata: object = None,
    save_plan: bool = True,
    force_fresh: bool = False,
    overwrite_plan: bool = False,
    implementation_revision: str | None = None,
) -> PlannedForward:
    """Plan one fixed-shape forward program around ordinary PyTorch tasks.

    The runtime and pool roles are explicit. The original model remains
    runtime-owned until the returned callable is closed. ``profiling_metadata``
    is a JSON-compatible, key-only description of value-sensitive profiling
    behavior. It is not passed to the model or returned callable.

    ``planning_cachedir`` selects the shared artifact store. ``force_fresh``
    disables cache reads; ``save_plan`` controls writes; and
    ``overwrite_plan`` replaces an existing identity only during a saved fresh
    run. ``implementation_revision`` invalidates compiler/profile artifacts
    when a lower-level custom implementation changes without changing its
    exported graph.

    ``partition`` accepts ``"auto"``, ``"whole"``, or a
    :class:`PartitionPolicy`. Partitioning only creates ordered stage
    occurrences; it does not choose training graph-pair alternatives.
    """

    from .planning.forward import build_forward

    memory = runtime._resolve_plan(
        execution=execution,
        spill=spill,
        execution_budget=execution_budget,
        spill_budget=spill_budget,
        execution_device=execution_device,
    )
    cache = PlanningCache.resolve(
        planning_cachedir,
        save_plan=save_plan,
        force_fresh=force_fresh,
        overwrite_plan=overwrite_plan,
        implementation_revision=implementation_revision,
    )
    try:
        with cache.activate_pytorch():
            return build_forward(
                model,
                example_inputs=example_inputs,
                memory=memory,
                partition=partition,
                verbose=verbose,
                planning_cache=cache,
                profiling_metadata=profiling_metadata,
            )
    except BaseException as error:
        try:
            runtime._abort_plan()
        except BaseException as cleanup_error:
            error.add_note(f"Runtime planning cleanup also failed: {cleanup_error}")
        raise


def plan_step(
    model: nn.Module,
    *,
    objective: Any,
    opt: Any,
    example_inputs: Sequence[Sequence[Any]],
    runtime: Runtime,
    execution: str,
    spill: str,
    execution_budget: int | None = None,
    spill_budget: int | None = None,
    execution_device: int | str | torch.device | None = None,
    partition: PartitionSpec = "auto",
    optimizer_ordering: Literal["stage_interleaved", "tail"] = "stage_interleaved",
    verbose: bool = True,
    planning_cachedir: str | os.PathLike[str] | None = None,
    profiling_metadata: Sequence[object] | None = None,
    save_plan: bool = True,
    force_fresh: bool = False,
    overwrite_plan: bool = False,
    implementation_revision: str | None = None,
) -> PlannedTrainStep:
    """Plan a fixed accumulated forward/objective/backward/update program.

    ``verbose=True`` reports each planning phase and unique structural ABI as
    it starts. Set it to ``False`` for silent embedding; diagnostics are still
    retained in :attr:`PlannedTrainStep.plan_report` either way.

    ``profiling_metadata`` has one JSON-compatible entry per example
    microbatch. It only distinguishes value-sensitive task measurements and
    their downstream plans; it is never passed to the objective or runtime.
    Cache policy arguments have the same meaning as :func:`plan_forward`.
    ``partition`` uses the same stage-only policy contract as forward
    planning. A later graph-pair phase independently shares differentiation
    portfolios across structurally equivalent stage occurrences.
    """

    from .planning.training import build_training

    memory = runtime._resolve_plan(
        execution=execution,
        spill=spill,
        execution_budget=execution_budget,
        spill_budget=spill_budget,
        execution_device=execution_device,
    )
    cache = PlanningCache.resolve(
        planning_cachedir,
        save_plan=save_plan,
        force_fresh=force_fresh,
        overwrite_plan=overwrite_plan,
        implementation_revision=implementation_revision,
    )
    try:
        with cache.activate_pytorch():
            return build_training(
                model,
                objective=objective,
                opt=opt,
                example_inputs=example_inputs,
                memory=memory,
                partition=partition,
                optimizer_ordering=optimizer_ordering,
                verbose=verbose,
                planning_cache=cache,
                profiling_metadata=profiling_metadata,
            )
    except BaseException as error:
        try:
            runtime._abort_plan()
        except BaseException as cleanup_error:
            error.add_note(f"Runtime planning cleanup also failed: {cleanup_error}")
        raise


__all__ = [
    "plan_forward",
    "plan_step",
]
