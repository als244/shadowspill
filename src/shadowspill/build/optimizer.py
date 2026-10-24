"""Capture optimizer geometry and defaults without allocating model-sized state."""

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from shadowspill.errors import CaptureError
from shadowspill.parallelism.ownership import ParameterOwnership
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.optimizer import OptimizerCapture, capture_optimizer
from shadowspill.pytorch.optimizer.artifacts import OptimizerTensorRole
from shadowspill.pytorch.optimizer.bindings import tensor_bindings
from shadowspill.pytorch.optimizer.capture import declare_optimizer_state
from shadowspill.pytorch.optimizer.discovery import validate_optimizer_inputs
from shadowspill.pytorch.optimizer.sandbox import (
    copy_optimizer_to_meta,
    optimizer_parameters,
)
from shadowspill.pytorch.optimizer.staging import (
    training_parameter_stage_owners,
    training_parameters_with_gradients,
)
from shadowspill.pytorch.optimizer.starts import (
    ConstantStart,
    HeldStart,
    NoStart,
    ParameterStart,
    StateStart,
    ValueStart,
)

from .optimizer_parameters import local_optimizer_parameters
from .training_capture import TrainingCapture


@dataclass(frozen=True)
class BuiltOptimizer:
    captured: OptimizerCapture
    initialization: dict[str, Any]


def capture_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    training: TrainingCapture,
    *,
    gradient_dtype: torch.dtype | None,
    master_dtype: torch.dtype | None = None,
    parameter_ownership: tuple[ParameterOwnership, ...] = (),
) -> BuiltOptimizer:
    named = dict(model.named_parameters())
    trained = training_parameters_with_gradients(training.examples, named)
    validate_optimizer_inputs(named, optimizer)
    copied, names = copy_optimizer_to_meta(
        optimizer, {id(value): name for name, value in named.items()}
    )
    copied, names, compute_copies = local_optimizer_parameters(
        copied,
        names,
        named,
        trained,
        master_dtype,
        {item.name: item for item in parameter_ownership},
    )
    parameters = {names[id(value)]: value for value in optimizer_parameters(copied)}
    declared = declare_optimizer_state(parameters, copied, receives_gradient=trained)
    initialization = {
        name: {"kind": "parameter", "name": name} for name in compute_copies
    }
    for entry in declared:
        parameter = parameters[entry.parameter_name]
        initialization[f"optimizer.{entry.parameter_name}.{entry.entry_name}"] = (
            _initial_value(entry.start, entry.parameter_name)
        )
        if entry.entry_name not in copied.state[parameter]:
            # Payloads remain geometry-only. Scalar controls are bounded CPU values.
            if not entry.shape:
                value = (
                    entry.start.value
                    if isinstance(entry.start, ConstantStart)
                    else entry.start.value.item()
                    if isinstance(entry.start, ValueStart)
                    else 0
                )
                tensor = torch.tensor(value, dtype=entry.dtype, device="cpu")
            else:
                tensor = torch.empty(entry.shape, dtype=entry.dtype, device="meta")
            copied.state[parameter][entry.entry_name] = tensor
    for binding in tensor_bindings(copied, names, require_gradients=False):
        if binding.role is OptimizerTensorRole.HYPERPARAMETER:
            initialization[binding.name] = (
                {"kind": "external"}
                if binding.tensor.is_meta
                else _initial_value(ValueStart(binding.tensor), binding.name)
            )
    result = capture_optimizer(
        parameters,
        copied,
        receives_gradient=trained,
        parameter_stage_owners=training_parameter_stage_owners(
            training.examples, named
        ),
        gradient_dtype=gradient_dtype,
        compute_copies=compute_copies,
    )
    if result.update_is_opaque or any(
        not isinstance(task.artifact, GraphArtifact) for task in result.update_tasks
    ):
        raise CaptureError(
            "portable build_step requires a traceable optimizer update: "
            f"{result.opaque_reason or result.optimizer_type}"
        )
    return BuiltOptimizer(result, initialization)


def _initial_value(start: StateStart, parameter_name: str) -> dict[str, Any]:
    if isinstance(start, ConstantStart):
        return {"kind": "constant", "value": start.value}
    if isinstance(start, ParameterStart):
        return {"kind": "parameter", "name": parameter_name}
    if isinstance(start, ValueStart):
        if start.value.numel() > 1024:
            raise CaptureError("optimizer host initialization value is not bounded")
        return {"kind": "value", "value": start.value.tolist()}
    if isinstance(start, HeldStart):
        return {"kind": "external"}
    assert isinstance(start, NoStart)
    raise CaptureError(f"optimizer state needs a first-use initializer: {start.reason}")
