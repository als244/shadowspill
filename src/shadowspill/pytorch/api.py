"""Small public orchestration functions for forward and training planning."""

from __future__ import annotations

import os
import traceback
from collections.abc import Callable, Sequence
from typing import Any, Literal, NoReturn

import torch
import torch.nn as nn

from shadowspill.planner import SearchOptions
from shadowspill.planner.annotated_plan import AnnotatedProgramPlan
from shadowspill.planner.program_inputs import TransferBandwidths
from shadowspill.pytorch.callables import PlannedForward, PlannedTrainStep
from shadowspill.pytorch.partition import PartitionSpec
from shadowspill.pytorch.runtime import Runtime
from shadowspill.pytorch.sharing import SharedOutput
from shadowspill.pytorch.state.model import (
    adopt_model_state_for_plan,
    require_model_state_for_plan,
)
from shadowspill.pytorch.state.storage import restore_persistent_object_ids
from shadowspill.pytorch.store import FrameworkArtifacts
from shadowspill.runtime.plan import (
    abort_plan,
    begin_plan,
)
from shadowspill.runtime.teardown import prepare_failure_cleanup
from shadowspill.step import StepDataOrdering, StepProgram
from shadowspill.store import ArtifactStore, StoreMode


def _cleanup_failed_plan(
    runtime: Runtime,
    *,
    planning_started: bool,
    error: BaseException,
) -> None:
    """Best-effort rollback while retaining every cleanup failure as problem."""

    operations: list[tuple[str, Any]] = []
    if planning_started:
        operations.append(("abort runtime plan", lambda: abort_plan(runtime)))
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

    prepare_failure_cleanup(
        runtime,
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
    search_options: SearchOptions | None = None,
    execution_device: int | str | torch.device | None = None,
    partition: PartitionSpec = "auto",
    verbose: bool = True,
    artifact_store: str | os.PathLike[str] | None = None,
    build_store: str | os.PathLike[str] | None = None,
    plan_store: str | os.PathLike[str] | None = None,
    profiling_metadata: object = None,
    allocation_probe_seeds: int = 1,
    allocation_probe_repetitions: int = 2,
    shared_outputs: Sequence[SharedOutput] = (),
    build_store_mode: StoreMode = "contribute",
    plan_store_mode: StoreMode = "contribute",
    export_bypass_key: str | None = None,
    transfer_bandwidths: TransferBandwidths | None = None,
) -> PlannedForward:
    """Plan one fixed-shape forward program around ordinary PyTorch tasks.

    ``model`` may be one whose state the caller imported into ``spill``, in
    which case planning adopts it and it outlives the plan, or one whose
    state has not been imported, in which case planning imports it in place
    and owns it: closing the callable releases that state and empties the
    parameters that viewed it, so read what you need before the close.

    The runtime and pool roles are explicit. The original model remains
    runtime-owned until the returned callable is closed. ``profiling_metadata``
    is a JSON-compatible, key-only description of value-sensitive profiling
    behavior. It is not passed to the model or returned callable.

    A store has two trees, rooted and permitted apart. ``artifact_store`` roots
    both; ``build_store`` and ``plan_store`` override either, so one build
    store can serve many runs that each keep their own plans. Each tree's mode
    says what this run does with it: ``contribute`` reads what is there and
    writes back what is not, ``reuse`` reads and persists nothing, ``require``
    refuses a miss, and ``refresh`` ignores what is there and writes over it.
    ``export_bypass_key`` is the caller's name for the code the build is made
    from -- model, objective and optimizer -- and every compiled, profiled and
    archived artifact is filed under it, so a change to a lower-level
    implementation that leaves the exported graph unchanged is told apart by a
    new key.

    ``partition`` accepts ``"auto"``, ``"whole"``, or a
    :class:`PartitionPolicy`. Partitioning only creates ordered stage
    occurrences; it does not choose training graph-pair alternatives.

    ``dynamic_scratch_reserve_bytes`` optionally raises the physical reserve
    for bounded allocation-path insertions above the automatically profiled
    requirement. It never reduces the measured reserve.

    ``transfer_bandwidths`` prices every copy at the given rates instead of
    the calibration the runtime measured, as
    :func:`shadowspill.planner.plan_program` takes them. A calibration moves
    from run to run on one machine, and the plan is keyed by what it was
    priced against, so a plan that has to be the one an earlier search chose
    is planned against the lanes that search planned against.

    What any search is told -- which objects are too small to be worth
    cutting, and whether the search must reproduce exactly at any worker
    count -- is `search_options.generic`, a `GenericPlanningOptions`.

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

    planning_started = False
    try:
        memory = begin_plan(
            runtime,
            execution=execution,
            spill=spill,
            execution_budget=execution_budget,
            spill_budget=spill_budget,
            dynamic_scratch_reserve_bytes=dynamic_scratch_reserve_bytes,
            execution_device=execution_device,
        )
        planning_started = True
        # After the handle exists, so state imported here can name the plan
        # that will release it.
        adopt_model_state_for_plan(
            model,
            runtime=runtime,
            pool=spill,
            owning_plan=memory.plan_handle,
        )
        cache = ArtifactStore.resolve(
            artifact_store,
            build_store=build_store,
            plan_store=plan_store,
            build_store_mode=build_store_mode,
            plan_store_mode=plan_store_mode,
            export_bypass_key=export_bypass_key,
        )
        with FrameworkArtifacts(cache).activate():
            return build_forward(
                model,
                example_inputs=example_inputs,
                memory=memory,
                partition=partition,
                verbose=verbose,
                artifact_store=cache,
                profiling_metadata=profiling_metadata,
                allocation_probe_seeds=allocation_probe_seeds,
                allocation_probe_repetitions=allocation_probe_repetitions,
                shared_outputs=shared_outputs,
                search_options=search_options,
                transfer_bandwidths=transfer_bandwidths,
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
    optimizer: Any,
    optimizer_state_init: Callable[[str, torch.Tensor, torch.nn.Parameter], None]
    | None = None,
    hyperparams: Sequence[str] = (),
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
    depth: int | None = None,
    breadth: int | None = None,
    reverse_breadth: bool = True,
    pair_loss: bool = True,
    search_options: SearchOptions | None = None,
    incumbent: AnnotatedProgramPlan | None = None,
    verbose: bool = True,
    artifact_store: str | os.PathLike[str] | None = None,
    build_store: str | os.PathLike[str] | None = None,
    plan_store: str | os.PathLike[str] | None = None,
    profiling_metadata: Sequence[object] | None = None,
    allocation_probe_seeds: int = 1,
    allocation_probe_repetitions: int = 2,
    build_store_mode: StoreMode = "contribute",
    plan_store_mode: StoreMode = "contribute",
    export_bypass_key: str | None = None,
    transfer_bandwidths: TransferBandwidths | None = None,
) -> PlannedTrainStep:
    """Plan a fixed accumulated forward/objective/backward/update program.

    ``model`` may be one whose state the caller imported into ``spill``, in
    which case planning adopts it and it outlives the plan, or one whose
    state has not been imported, in which case planning imports it in place
    and owns it: closing the callable releases that state and empties the
    parameters that viewed it, so read what you need before the close.

    ``verbose=True`` reports each planning phase and unique structural contract as
    it starts. Set it to ``False`` for silent embedding; diagnostics are still
    retained in :attr:`PlannedTrainStep.plan_report` either way.

    ``optimizer_state_init`` fills one declared optimizer-state entry, given
    the entry's name, the tensor to fill, and the parameter the entry belongs
    to. The optimizer declares what state exists by being run on meta
    parameters, which costs nothing; ShadowSpill builds that in ordinary
    memory; this supplies the values, because a default would be an assumption
    that fails silently; and the import that adopts the optimizer's state for
    the plan is what moves it into the spill pool. It is not needed when ``optimizer``
    returns an optimizer whose state the caller has already imported:
    planning adopts the state of the optimizer it is handed, and that object
    is the reference. State imported for some other optimizer is invisible to
    planning, which neither knows nor cares about it.

    A value that varies between steps -- a scheduled learning rate, say --
    is passed to the optimizer as a **tensor** rather than a float, and
    written in place between steps. A tensor enters the captured update's
    identity by geometry alone, so one capture serves every value it takes,
    while a float enters by value and would capture again for each one. See
    :doc:`the optimizer </architecture/optimizer>`.

    ``profiling_metadata`` has one JSON-compatible entry per example
    microbatch. It only distinguishes value-sensitive task measurements and
    their downstream plans; it is never passed to the objective or runtime.
    Cache policy arguments, ``plan_store`` included, have the same meaning
    as :func:`plan_forward`.
    ``partition`` uses the same stage-only policy contract as forward
    planning. A later graph-pair phase independently shares differentiation
    graph pairs across structurally equivalent stage occurrences.

    ``depth`` and ``breadth`` say how the step walks its microbatches:
    ``depth`` passes of ``breadth`` microbatches each, every microbatch of a
    pass running one stage before any runs the next, so each stage's
    parameters are fetched once per pass rather than once per microbatch.
    Their product must be ``len(example_inputs)``; give one and the other
    follows, give neither and the step runs depth-first, one microbatch after
    another, as before. ``pair_loss`` runs each microbatch's last stage
    forward and backward together so the loss's saved state is consumed as
    it is produced, and ``reverse_breadth`` walks a pass's microbatches in
    reverse during backward; both are on by default and vacuous at
    ``breadth=1``.

    ``incumbent`` is the plan to beat for the recurrent program: a plan a
    search already found for this step, which the replan here measures at
    this budget and answers with unless it does strictly better, so a step
    run after a sweep executes the plan the sweep chose even when the
    calibration or the budget of the replan differs from the sweep's.
    ``search_options`` names the resolutions the search plans: the shares
    of flexible groups to recompute, one resolved program each, as exact
    fractions such as ``("0", "1/2", "1")``. ``None`` plans the library's
    default of every quarter. The options are part of the plan's identity in
    the store and are recorded on the report; naming the default is the same
    as naming nothing.

    ``dynamic_scratch_reserve_bytes`` and ``transfer_bandwidths`` have the
    same semantics and defaults as :func:`plan_forward`. A step that runs
    what :func:`plan_step_search` chose is planned against the report's
    ``planned_lanes``, so it asks the store the search's question and
    executes the plan the search chose.

    Allocation-path probe settings have the same semantics and defaults as
    :func:`plan_forward`.
    """

    from .planning.training import build_training

    data_ordering = StepDataOrdering.resolve(
        microbatches=len(example_inputs),
        depth=depth,
        breadth=breadth,
        reverse_breadth=reverse_breadth,
        pair_loss=pair_loss,
    )
    planning_started = False
    try:
        memory = begin_plan(
            runtime,
            execution=execution,
            spill=spill,
            execution_budget=execution_budget,
            spill_budget=spill_budget,
            dynamic_scratch_reserve_bytes=dynamic_scratch_reserve_bytes,
            execution_device=execution_device,
        )
        planning_started = True
        # After the handle exists, so state imported here can name the plan
        # that will release it.
        adopt_model_state_for_plan(
            model,
            runtime=runtime,
            pool=spill,
            owning_plan=memory.plan_handle,
        )
        cache = ArtifactStore.resolve(
            artifact_store,
            build_store=build_store,
            plan_store=plan_store,
            build_store_mode=build_store_mode,
            plan_store_mode=plan_store_mode,
            export_bypass_key=export_bypass_key,
        )
        with FrameworkArtifacts(cache).activate():
            return build_training(
                model,
                objective=objective,
                build_optimizer=optimizer,
                optimizer_state_init=optimizer_state_init,
                hyperparams=hyperparams,
                example_inputs=example_inputs,
                memory=memory,
                partition=partition,
                optimizer_ordering=optimizer_ordering,
                data_ordering=data_ordering,
                verbose=verbose,
                artifact_store=cache,
                profiling_metadata=profiling_metadata,
                allocation_probe_seeds=allocation_probe_seeds,
                allocation_probe_repetitions=allocation_probe_repetitions,
                search_options=search_options,
                incumbent=incumbent,
                transfer_bandwidths=transfer_bandwidths,
            )
    except BaseException as error:
        _surface_failed_plan(
            runtime,
            planning_started=planning_started,
            operation="plan training step",
            error=error,
        )


def build_step_programs(
    model: nn.Module,
    *,
    objective: Any,
    optimizer: Any,
    optimizer_state_init: Callable[[str, torch.Tensor, torch.nn.Parameter], None]
    | None = None,
    hyperparams: Sequence[str] = (),
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
    orderings: Sequence[StepDataOrdering] | None = None,
    verbose: bool = True,
    artifact_store: str | os.PathLike[str] | None = None,
    build_store: str | os.PathLike[str] | None = None,
    profiling_metadata: Sequence[object] | None = None,
    allocation_probe_seeds: int = 1,
    allocation_probe_repetitions: int = 2,
    build_store_mode: StoreMode = "contribute",
    export_bypass_key: str | None = None,
) -> tuple[StepProgram, ...]:
    """Capture, profile, and lower a reusable step without searching.

    Returns one :class:`StepProgram` per ordering in ``orderings``, in that
    order, from one capture, one materialization and one profiling; the
    orderings differ only in the walk the lowering emits. ``None`` builds the
    depth-first ordering alone. Every ordering must cover
    ``len(example_inputs)`` microbatches.

    Each program is a fully self-contained JSON boundary that can be passed
    to :func:`plan_program` repeatedly with different budgets and transfer
    bandwidths. Temporary compilation and materialization state is released
    before this function returns; no runtime callable remains active.

    Takes no plan-store arguments, because it writes no plans. What it
    produces is build work -- exports, graph pairs, profiles, lowered
    programs -- so ``build_store`` and ``build_store_mode`` are the only store
    controls that mean anything here. With an ``export_bypass_key``, each
    ordering's program is first looked up in the build store's step archive
    under the identity the request has before any capture, and only the
    orderings not found there are built; without one, every build captures.
    The other arguments mean what they mean for :func:`plan_step`; the
    ordering is recorded in each program.
    """

    from .planning.training import make_training_programs

    microbatches = len(example_inputs)
    resolved = (
        (StepDataOrdering.depth_first(microbatches),)
        if orderings is None
        else tuple(orderings)
    )
    if not resolved:
        raise ValueError("at least one ordering is required")
    for ordering in resolved:
        if ordering.microbatches != microbatches:
            raise ValueError(
                f"ordering {ordering.label} covers {ordering.microbatches}"
                f" microbatches, but {microbatches} example inputs were given"
            )
    require_model_state_for_plan(model, runtime=runtime, pool=spill)
    planning_started = False
    try:
        memory = begin_plan(
            runtime,
            execution=execution,
            spill=spill,
            execution_budget=execution_budget,
            spill_budget=spill_budget,
            dynamic_scratch_reserve_bytes=dynamic_scratch_reserve_bytes,
            execution_device=execution_device,
        )
        planning_started = True
        cache = ArtifactStore.resolve(
            artifact_store,
            build_store=build_store,
            build_store_mode=build_store_mode,
            export_bypass_key=export_bypass_key,
        )
        with FrameworkArtifacts(cache).activate():
            result = make_training_programs(
                model,
                objective=objective,
                build_optimizer=optimizer,
                optimizer_state_init=optimizer_state_init,
                hyperparams=hyperparams,
                example_inputs=example_inputs,
                memory=memory,
                partition=partition,
                optimizer_ordering=optimizer_ordering,
                data_orderings=resolved,
                verbose=verbose,
                artifact_store=cache,
                profiling_metadata=profiling_metadata,
                allocation_probe_seeds=allocation_probe_seeds,
                allocation_probe_repetitions=allocation_probe_repetitions,
            )
        try:
            abort_plan(runtime)
        finally:
            planning_started = False
        restore_persistent_object_ids(runtime)
        return result
    except BaseException as error:
        _surface_failed_plan(
            runtime,
            planning_started=planning_started,
            operation="build training step programs",
            error=error,
        )


__all__ = [
    "build_step_programs",
    "plan_forward",
    "plan_step",
]
