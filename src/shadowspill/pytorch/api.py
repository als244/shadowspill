"""Small public orchestration functions for forward and training planning."""

from __future__ import annotations

import os
import traceback
from collections.abc import Sequence
from typing import Any, Literal, NoReturn

import torch
import torch.nn as nn

from shadowspill.planner import PressureFitOptions
from shadowspill.planner.program import (
    AnnotatedProgramPlan,
    PressureFitProgram,
    StepProgram,
    TransferBandwidths,
)
from shadowspill.pytorch.cache import PlanningCache
from shadowspill.pytorch.callables import PlannedForward, PlannedTrainStep
from shadowspill.pytorch.partition import PartitionSpec
from shadowspill.pytorch.runtime_adapter import Runtime
from shadowspill.pytorch.sharing import SharedOutput
from shadowspill.pytorch.state.model import require_model_state_for_plan
from shadowspill.pytorch.state.storage import restore_persistent_object_ids


def _cleanup_failed_plan(
    runtime: Runtime,
    *,
    planning_started: bool,
    error: BaseException,
) -> None:
    """Best-effort rollback while retaining every cleanup failure as problem."""

    operations: list[tuple[str, Any]] = []
    if planning_started:
        operations.append(("abort runtime plan", runtime._abort_plan))
    operations.append(
        (
            "restore persistent object identities",
            lambda: restore_persistent_object_ids(runtime),
        )
    )
    for description, operation in operations:
        try:
            operation()
        except BaseException as cleanup_error:
            error.add_note(f"Failed to {description}: {cleanup_error}")


def _surface_failed_plan(
    runtime: Runtime,
    *,
    planning_started: bool,
    operation: str,
    error: BaseException,
) -> NoReturn:
    """Prepare allocator teardown, roll back, and preserve the first error."""

    runtime._prepare_failure_cleanup(
        error,
        operation=operation,
        synchronize_unlatched=False,
    )
    _cleanup_failed_plan(
        runtime,
        planning_started=planning_started,
        error=error,
    )
    _clear_failure_frame_locals(error)
    raise error


def _clear_failure_frame_locals(error: BaseException) -> None:
    """Release task-local tensors without discarding traceback locations."""

    pending = [error]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in visited:
            continue
        visited.add(identity)
        if current.__traceback__ is not None:
            traceback.clear_frames(current.__traceback__)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)


def plan_forward(
    model: nn.Module,
    *,
    example_inputs: Sequence[Any],
    runtime: Runtime,
    execution: str,
    spill: str,
    execution_budget: int | None = None,
    spill_budget: int | None = None,
    dynamic_scratch_reserve_bytes: int | None = None,
    execution_device: int | str | torch.device | None = None,
    partition: PartitionSpec = "auto",
    verbose: bool = True,
    planning_cachedir: str | os.PathLike[str] | None = None,
    profiling_metadata: object = None,
    allocation_probe_seeds: int = 1,
    allocation_probe_repetitions: int = 2,
    shared_outputs: Sequence[SharedOutput] = (),
    save_plan: bool = True,
    force_fresh: bool = False,
    overwrite_plan: bool = False,
    implementation_revision: str | None = None,
) -> PlannedForward:
    """Plan one fixed-shape forward program around ordinary PyTorch tasks.

    ``model`` must be the value returned by :func:`import_model_state` for
    this ``runtime`` and ``spill`` pool. Planning consumes existing runtime
    bindings; it never imports or releases model storage.

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

    ``dynamic_scratch_reserve_bytes`` optionally raises the physical reserve
    for bounded allocation-path insertions above the automatically profiled
    requirement. It never reduces the measured reserve.

    ``allocation_probe_seeds`` controls independent randomized activation
    probes per structural contract. ``allocation_probe_repetitions`` repeats each
    seed identically to expose first-use allocation paths. The defaults are
    one seed and two repetitions.

    An ``example_inputs`` leaf may be wrapped with :func:`shared_input` to bind
    an existing runtime-owned object without copying it. ``shared_outputs``
    names public tensor leaves that remain runtime-owned
    and identifies the pool or pools in which each leaf must be retained.
    Undeclared leaves keep ordinary caller-owned output behavior.
    """

    from .planning.forward import build_forward

    require_model_state_for_plan(
        model,
        runtime=runtime,
        pool=spill,
    )
    planning_started = False
    try:
        memory = runtime._resolve_plan(
            execution=execution,
            spill=spill,
            execution_budget=execution_budget,
            spill_budget=spill_budget,
            dynamic_scratch_reserve_bytes=dynamic_scratch_reserve_bytes,
            execution_device=execution_device,
        )
        planning_started = True
        cache = PlanningCache.resolve(
            planning_cachedir,
            save_plan=save_plan,
            force_fresh=force_fresh,
            overwrite_plan=overwrite_plan,
            implementation_revision=implementation_revision,
        )
        with cache.activate_pytorch():
            return build_forward(
                model,
                example_inputs=example_inputs,
                memory=memory,
                partition=partition,
                verbose=verbose,
                planning_cache=cache,
                profiling_metadata=profiling_metadata,
                allocation_probe_seeds=allocation_probe_seeds,
                allocation_probe_repetitions=allocation_probe_repetitions,
                shared_outputs=shared_outputs,
            )
    except BaseException as error:
        _surface_failed_plan(
            runtime,
            planning_started=planning_started,
            operation="plan forward",
            error=error,
        )


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
    dynamic_scratch_reserve_bytes: int | None = None,
    execution_device: int | str | torch.device | None = None,
    partition: PartitionSpec = "auto",
    optimizer_ordering: Literal["stage_interleaved", "tail"] = "stage_interleaved",
    verbose: bool = True,
    planning_cachedir: str | os.PathLike[str] | None = None,
    profiling_metadata: Sequence[object] | None = None,
    allocation_probe_seeds: int = 1,
    allocation_probe_repetitions: int = 2,
    save_plan: bool = True,
    force_fresh: bool = False,
    overwrite_plan: bool = False,
    implementation_revision: str | None = None,
) -> PlannedTrainStep:
    """Plan a fixed accumulated forward/objective/backward/update program.

    ``model`` must be the value returned by :func:`import_model_state` for
    this ``runtime`` and ``spill`` pool. Planning consumes existing runtime
    bindings; it never imports or releases model storage.

    ``verbose=True`` reports each planning phase and unique structural contract as
    it starts. Set it to ``False`` for silent embedding; diagnostics are still
    retained in :attr:`PlannedTrainStep.plan_report` either way.

    ``profiling_metadata`` has one JSON-compatible entry per example
    microbatch. It only distinguishes value-sensitive task measurements and
    their downstream plans; it is never passed to the objective or runtime.
    Cache policy arguments have the same meaning as :func:`plan_forward`.
    ``partition`` uses the same stage-only policy contract as forward
    planning. A later graph-pair phase independently shares differentiation
    graph pairs across structurally equivalent stage occurrences.

    ``dynamic_scratch_reserve_bytes`` has the same minimum-reserve semantics
    as :func:`plan_forward`.

    Allocation-path probe settings have the same semantics and defaults as
    :func:`plan_forward`.
    """

    from .planning.training import build_training

    require_model_state_for_plan(
        model,
        runtime=runtime,
        pool=spill,
    )
    planning_started = False
    try:
        memory = runtime._resolve_plan(
            execution=execution,
            spill=spill,
            execution_budget=execution_budget,
            spill_budget=spill_budget,
            dynamic_scratch_reserve_bytes=dynamic_scratch_reserve_bytes,
            execution_device=execution_device,
        )
        planning_started = True
        cache = PlanningCache.resolve(
            planning_cachedir,
            save_plan=save_plan,
            force_fresh=force_fresh,
            overwrite_plan=overwrite_plan,
            implementation_revision=implementation_revision,
        )
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
                allocation_probe_seeds=allocation_probe_seeds,
                allocation_probe_repetitions=allocation_probe_repetitions,
            )
    except BaseException as error:
        _surface_failed_plan(
            runtime,
            planning_started=planning_started,
            operation="plan training step",
            error=error,
        )


def make_step_program(
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
    dynamic_scratch_reserve_bytes: int | None = None,
    execution_device: int | str | torch.device | None = None,
    partition: PartitionSpec = "auto",
    optimizer_ordering: Literal["stage_interleaved", "tail"] = "stage_interleaved",
    verbose: bool = True,
    planning_cachedir: str | os.PathLike[str] | None = None,
    profiling_metadata: Sequence[object] | None = None,
    allocation_probe_seeds: int = 1,
    allocation_probe_repetitions: int = 2,
    save_plan: bool = True,
    force_fresh: bool = False,
    overwrite_plan: bool = False,
    implementation_revision: str | None = None,
) -> StepProgram:
    """Capture, profile, and lower a reusable step without running PressureFit.

    The returned :class:`StepProgram` is a fully self-contained JSON boundary.
    It can be passed to :func:`pressurefit_program` repeatedly with different
    budgets and transfer bandwidths. Temporary compilation/materialization
    state is released before this function returns; no runtime callable remains
    active.
    """

    from .planning.training import make_training_program

    require_model_state_for_plan(model, runtime=runtime, pool=spill)
    planning_started = False
    try:
        memory = runtime._resolve_plan(
            execution=execution,
            spill=spill,
            execution_budget=execution_budget,
            spill_budget=spill_budget,
            dynamic_scratch_reserve_bytes=dynamic_scratch_reserve_bytes,
            execution_device=execution_device,
        )
        planning_started = True
        cache = PlanningCache.resolve(
            planning_cachedir,
            save_plan=save_plan,
            force_fresh=force_fresh,
            overwrite_plan=overwrite_plan,
            implementation_revision=implementation_revision,
        )
        with cache.activate_pytorch():
            result = make_training_program(
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
                allocation_probe_seeds=allocation_probe_seeds,
                allocation_probe_repetitions=allocation_probe_repetitions,
            )
        try:
            runtime._abort_plan()
        finally:
            planning_started = False
        restore_persistent_object_ids(runtime)
        return result
    except BaseException as error:
        _surface_failed_plan(
            runtime,
            planning_started=planning_started,
            operation="make training step Program",
            error=error,
        )


def pressurefit_program(
    program: PressureFitProgram,
    *,
    execution_budget: int | None = None,
    spill_budget: int | None = None,
    transfer_bandwidths: TransferBandwidths | None = None,
    options: PressureFitOptions | None = None,
    planning_cachedir: str | os.PathLike[str] | None = None,
    verbose: bool = True,
    save_plan: bool = True,
    force_fresh: bool = False,
    overwrite_plan: bool = False,
    implementation_revision: str | None = None,
) -> AnnotatedProgramPlan:
    """Run recomputation, PressureFit, simulation, and physical admission.

    ``program`` is normally ``make_step_program(...).recurrent`` or the value
    reconstructed by :meth:`PressureFitProgram.from_value`. This operation is
    model- and runtime-independent and may be repeated for a budget/bandwidth
    frontier without capture, compilation, or profiling.
    """

    from .planning.program_selection import select_program

    if options is not None and not isinstance(options, PressureFitOptions):
        raise TypeError("options must be PressureFitOptions or None")
    cache = PlanningCache.resolve(
        planning_cachedir,
        save_plan=save_plan,
        force_fresh=force_fresh,
        overwrite_plan=overwrite_plan,
        implementation_revision=implementation_revision,
    )
    cache.initialize()
    return select_program(
        program,
        execution_budget_bytes=execution_budget,
        spill_budget_bytes=spill_budget,
        transfer_bandwidths=transfer_bandwidths,
        options=options,
        planning_cache=cache,
        verbose=verbose,
    )


__all__ = [
    "make_step_program",
    "plan_forward",
    "plan_step",
    "pressurefit_program",
]
