"""Materialization: the model's state registered with the runtime and the optimizer
captured once, and the rollback that returns ownership to the CPU when a plan is
abandoned or fails."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, NoReturn

import torch
import torch.nn as nn

from shadowspill.errors import (
    PlanningError,
)
from shadowspill.pipeline.common import PlanningTimer
from shadowspill.pytorch.materialization.training import (
    TrainingMaterializedState,
)
from shadowspill.pytorch.optimizer import (
    capture_optimizer,
    training_parameter_stage_owners,
    training_parameters_with_gradients,
)
from shadowspill.pytorch.state.optimizer import (
    adopt_optimizer_state_for_plan,
    declare_varying_hyperparams,
    install_declared_optimizer_state,
    release_optimizer_state_from_plan,
)
from shadowspill.runtime import Runtime
from shadowspill.runtime.plan import (
    PlanMemory,
    RuntimeBridge,
)
from shadowspill.runtime.teardown import prepare_failure_cleanup

from ..artifacts import (
    TrainingCaptureArtifacts,
    TrainingMaterializationArtifacts,
)
from ..stores import PlanningStores


def materialize_training_state(
    model: nn.Module,
    captured: TrainingCaptureArtifacts,
    *,
    build_optimizer: Callable[[Any], torch.optim.Optimizer],
    hyperparams: Sequence[str],
    memory: PlanMemory,
    stores: PlanningStores,
    timer: PlanningTimer,
) -> TrainingMaterializationArtifacts:
    """Materialize registered state and invoke/capture the optimizer exactly once."""

    runtime = memory.runtime
    bridge = RuntimeBridge(
        runtime,
        captured.layout.program,
        memory.plan_handle,
        execution_pool_id=memory.execution.pool_id,
        spill_pool_id=memory.spill.pool_id,
    )
    state: TrainingMaterializedState | None = None
    optimizer: torch.optim.Optimizer | None = None
    try:
        with timer.measure("model_materialization"):
            state = TrainingMaterializedState(
                model,
                captured.layout,
                captured.captures,
                captured.cpu_inputs,
                bridge,
                runtime=runtime,
                device_ordinal=captured.device_ordinal,
            )
        with timer.measure("optimizer_capture"):
            optimizer = build_optimizer(model.parameters())
            if not isinstance(optimizer, torch.optim.Optimizer):
                raise PlanningError("optimizer must return a torch.optim.Optimizer")
            # The values a step is allowed to set are held in tensors before
            # anything is captured, so the capture takes them as inputs rather
            # than folding them in. Only the named ones are touched.
            declare_varying_hyperparams(model, optimizer, hyperparams)
            state.restore_model_cpu_for_optimizer_capture()
            # The optimizer declares what it keeps on meta, which allocates
            # nothing; each entry is then created in the spill pool as this
            # plan's, and filled there by the caller, so the state never sits
            # in ordinary host memory beside the pool. Capture finds the state
            # already present and does not create any of its own.
            # A parameter the objective never reaches receives no gradient
            # however it is flagged, and eager training skips it. The plan
            # has to skip it too, or it keeps state nothing steps and
            # reserves a gradient nothing writes.
            receives_gradient = training_parameters_with_gradients(
                captured.partitioned,
                dict(model.named_parameters()),
            )
            with timer.measure("optimizer_state_install"):
                installed_entries = install_declared_optimizer_state(
                    model,
                    optimizer,
                    runtime=runtime,
                    pool=memory.spill.name,
                    owning_plan=memory.plan_handle,
                    receives_gradient=receives_gradient,
                )
            optimizer_capture = capture_optimizer(
                dict(model.named_parameters()),
                optimizer,
                parameter_stage_owners=training_parameter_stage_owners(
                    captured.partitioned,
                    dict(model.named_parameters()),
                ),
                receives_gradient=receives_gradient,
                store=stores.optimizer_captures,
                timer=timer,
            )
            if optimizer_capture.initialized_state_dict is not None:
                optimizer.load_state_dict(optimizer_capture.initialized_state_dict)
        with timer.measure("optimizer_state_import"):
            adopt_optimizer_state_for_plan(
                optimizer,
                runtime=runtime,
                pool=memory.spill.name,
                owning_plan=memory.plan_handle,
            )

        with timer.measure("model_placeholder_restoration"):
            state.restore_device_placeholders_after_optimizer_capture()
        if optimizer_capture.recurrent is None:
            raise PlanningError(
                "the optimizer state/update cannot be bounded: "
                f"{optimizer_capture.opaque_reason}"
            )
        return TrainingMaterializationArtifacts(
            state, optimizer, optimizer_capture, installed_entries
        )
    except BaseException as error:
        if state is not None:

            def rollback_partial_materialization() -> None:
                _restore_training_ownership(
                    model,
                    state,
                    optimizer,
                    runtime=runtime,
                )

            rollback_training_failure(
                runtime,
                error,
                rollback_partial_materialization,
                operation="materialize training state",
            )
        raise


def rollback_training_materialization(
    model: nn.Module,
    materialized: TrainingMaterializationArtifacts,
) -> None:
    """Restore CPU ownership when a caller abandons an intermediate plan."""

    _restore_training_ownership(
        model,
        materialized.state,
        materialized.optimizer,
        runtime=materialized.state.runtime,
    )


def _restore_training_ownership(
    model: nn.Module,
    state: TrainingMaterializedState,
    optimizer: torch.optim.Optimizer | None,
    *,
    runtime: Runtime,
) -> None:
    """Attempt optimizer and model restoration even if either cleanup fails."""

    for parameter in model.parameters():
        parameter.grad = None
    release_error: BaseException | None = None
    if optimizer is not None:
        try:
            if release_optimizer_state_from_plan(optimizer, runtime=runtime):
                # Only state this plan created is ours to end.
                optimizer.state.clear()
        except BaseException as error:
            release_error = error
    try:
        state.restore_cpu_and_unregister()
    except BaseException as error:
        if release_error is not None:
            release_error.add_note(f"Model-state rollback also failed: {error}")
        else:
            raise
    if release_error is not None:
        raise release_error


def rollback_training_failure(
    runtime: Runtime,
    error: BaseException,
    rollback: Callable[[], None],
    *,
    operation: str,
) -> NoReturn:
    """Recover a planning OOM before releasing materialized frontend state."""

    prepare_failure_cleanup(
        runtime,
        error,
        operation=operation,
        synchronize_unlatched=False,
    )
    try:
        rollback()
    except BaseException as cleanup_error:
        error.add_note(
            f"Failed to roll back materialized training state: {cleanup_error}"
        )
    raise error
